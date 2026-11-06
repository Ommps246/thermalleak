"""
Radiometric normalisation between two frames.

Two thermal captures of the same object, minutes apart, differ by more than the
defect you are looking for. Ambient temperature drifts, the camera's internal
calibration shifts, the device is under a different workload. All of these
produce a roughly affine relation between the frames:

    T_inspection  ~  a * T_baseline + b

Subtracting raw frames therefore measures drift, not damage. The fix is to
estimate (a, b) and remove it -- but the estimate must not be corrupted by the
leak itself, which is exactly the signal being preserved.

That is why this uses **iteratively reweighted robust regression** rather than
ordinary least squares. A leak occupying a few percent of the frame will drag
an OLS fit enough to partially erase itself; a robust fit treats it as an
outlier and leaves it standing proud of the background. This is the difference
between a detector that finds weak leaks and one that only finds obvious ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Normalisation:
    """Estimated affine relation between two frames, plus the residual map."""

    gain: float
    offset: float
    difference: np.ndarray       # inspection - (gain * baseline + offset), kelvin
    inlier_mask: np.ndarray      # pixels the robust fit treated as background
    noise_sigma: float           # robust estimate of per-pixel noise, kelvin
    iterations: int


def robust_affine_fit(baseline: np.ndarray, inspection: np.ndarray,
                      valid: np.ndarray | None = None,
                      iterations: int = 8, tukey_c: float = 4.685
                      ) -> tuple[float, float, np.ndarray, int]:
    """
    Fit inspection ~ gain * baseline + offset with Tukey biweight reweighting.

    Tukey's biweight fully rejects gross outliers (weight goes to exactly zero
    beyond c*sigma) rather than merely downweighting them, which is the correct
    behaviour here: a leak is not a slightly unusual background pixel, it is a
    different population entirely.
    """
    if valid is None:
        valid = np.ones_like(baseline, dtype=bool)

    x = baseline[valid].astype(float).ravel()
    y = inspection[valid].astype(float).ravel()

    if x.size < 32:
        return 1.0, float(np.median(y - x) if y.size else 0.0), valid.copy(), 0

    weights = np.ones_like(x)
    gain, offset = 1.0, float(np.median(y - x))
    used = 0

    for step in range(1, iterations + 1):
        sw = weights.sum()
        if sw <= 1e-9:
            break
        mx = float((weights * x).sum() / sw)
        my = float((weights * y).sum() / sw)
        cov = float((weights * (x - mx) * (y - my)).sum())
        var = float((weights * (x - mx) ** 2).sum())

        new_gain = cov / var if var > 1e-12 else 1.0
        # Guard against a degenerate fit on a low-contrast scene.
        new_gain = float(np.clip(new_gain, 0.5, 2.0))
        new_offset = my - new_gain * mx

        residual = y - (new_gain * x + new_offset)

        # Centre the residual on its median before weighting.
        #
        # This is not cosmetic. A large anomaly biases the offset estimate on the
        # first (unweighted) pass, which shifts the whole residual distribution
        # away from zero. Weighting the raw residual then measures every pixel's
        # distance from zero rather than from the bulk of the data, so the good
        # pixels -- all sitting together at the shifted location -- are scored as
        # far outliers and given zero weight, while the routine reports no
        # inliers at all. Robust reweighting must be about the current centre of
        # the distribution, not about the origin.
        centre = float(np.median(residual))
        deviation = residual - centre
        sigma = 1.4826 * float(np.median(np.abs(deviation)))

        # Scale floor tied to the data, not an absolute constant: when the two
        # frames agree almost exactly outside the anomaly the MAD collapses
        # toward zero and the weights become arbitrarily brittle.
        sigma = max(sigma, 1e-6 * (float(np.max(y)) - float(np.min(y))), 1e-9)

        u = deviation / (tukey_c * sigma)
        new_weights = np.where(np.abs(u) < 1.0, (1.0 - u ** 2) ** 2, 0.0)

        # Never accept an update that leaves nothing to fit on the next pass;
        # keep the last estimate that was supported by real data.
        if new_weights.sum() <= 1e-9:
            break

        gain, offset, weights, used = new_gain, new_offset, new_weights, step

    inlier = np.zeros_like(baseline, dtype=bool)
    inlier[valid] = weights > 1e-3
    return float(gain), float(offset), inlier, used


def normalise(baseline: np.ndarray, inspection: np.ndarray,
              valid: np.ndarray | None = None) -> Normalisation:
    """
    Remove drift between two frames and return the residual difference map.

    The returned `noise_sigma` is a robust (median-absolute-deviation) estimate
    of the per-pixel noise in the *difference*, measured on background pixels
    only. Every threshold downstream is expressed in units of this sigma, so the
    detector adapts to the camera it is given instead of relying on a magic
    constant tuned to one dataset.
    """
    gain, offset, inlier, iters = robust_affine_fit(baseline, inspection, valid)

    difference = inspection - (gain * baseline + offset)

    # Force the difference to exactly zero where the frames do not overlap.
    #
    # `register` already fills unreachable pixels from the baseline so nothing
    # wild enters here, but that is not sufficient on its own: the affine drift
    # correction is applied afterwards, so those pixels come out at
    # (1 - gain) * baseline - offset rather than at zero. Measured on real
    # frames that residual reached 0.649 K -- smaller than the 6.28 K step the
    # old frame-median fill produced, but still comparable to the ~0.9 K peaks
    # the detector fires on. Zeroing it here is the only place the gain and
    # offset are known, and it makes "no data" mean "no difference" exactly.
    background = difference[inlier] if inlier.any() else difference.ravel()
    med = float(np.median(background))
    sigma = 1.4826 * float(np.median(np.abs(background - med)))
    sigma = max(sigma, 1e-4)

    difference = difference - med
    if valid is not None:
        difference = np.where(valid, difference, 0.0)

    return Normalisation(gain=gain, offset=offset, difference=difference,
                         inlier_mask=inlier, noise_sigma=sigma, iterations=iters)
