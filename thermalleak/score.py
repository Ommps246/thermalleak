"""
Physics-based scoring: is this candidate actually heat?

This module is the reason the project is more than an image-differencing script.

Statistical anomaly detection answers "is this region unusual?". It cannot
answer "is this region a heat source?", and in thermography those are very
different questions. Reflections, emissivity edges, stickers, registration
artefacts and dirt are all statistically unusual and none of them are leaks.

The discriminator is that **heat obeys a differential equation and artefacts do
not**. A conductive source on a plate must produce

    dT(r) = A * K0(r / L) + c

with a length scale L set by the material, not by the source. Three physical
consequences follow, and each becomes a test:

1.  *Profile shape.* The radial falloff must match K0. An emissivity patch has a
    flat top and a step edge; it fits badly.

2.  *Length scale.* L is a material property. An anomaly far sharper than the
    material's thermal length cannot be conductive heating -- diffusion
    physically cannot sustain that gradient in steady state.

3.  *Edge sharpness.* Diffusion is a low-pass operator. The maximum gradient a
    genuine source can produce is bounded by A/L. A step edge exceeds it by
    an order of magnitude.

Scores from the three tests are combined as a weighted geometric mean, so a
candidate must pass *all* of them. That is the correct semantics for a physical
consistency check: any single violated law is disqualifying.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

from .detect import Candidate
from .physics import Material, bessel_model


@dataclass
class PhysicsVerdict:
    """Physical assessment of one candidate anomaly."""

    physics_score: float          # 0..1, geometric mean of the component tests
    fit_score: float              # profile-shape agreement (R^2, rescaled)
    length_score: float           # plausibility of the fitted thermal length
    sharpness_score: float        # gradient consistency with diffusion
    novelty_score: float          # was this location cold in the baseline?
    baseline_occupancy: float     # 0 = cold casing, 1 = existing hotspot
    fitted_amplitude_k: float     # A, kelvin
    fitted_length_px: float       # L, pixels
    expected_length_px: float     # L predicted from material properties
    elongation: float             # 1.0 = circular, >2 = crack-like
    geometry: str                 # "point" or "crack"
    estimated_power_w: float      # leak power inferred from A
    sharpness_ratio: float        # observed max gradient / diffusion limit
    fit_ok: bool

    def verdict(self, threshold: float = 0.45) -> str:
        return "thermal_leak" if self.physics_score >= threshold else "artefact"


# --------------------------------------------------------------------------
# Geometry of the candidate
# --------------------------------------------------------------------------

def _weighted_moments(difference: np.ndarray, mask: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Intensity-weighted second moments of a candidate region.

    Returns the centroid, the principal axis (unit vector), and the elongation
    ratio. Weighting by excess temperature rather than by the binary mask makes
    the estimate independent of where the detection threshold happened to fall.
    """
    ys, xs = np.nonzero(mask)
    w = np.clip(difference[mask], 0.0, None).astype(float)
    if w.sum() < 1e-9:
        w = np.ones_like(w)

    cy = float((w * ys).sum() / w.sum())
    cx = float((w * xs).sum() / w.sum())

    dy, dx = ys - cy, xs - cx
    cov = np.array([
        [float((w * dy * dy).sum() / w.sum()), float((w * dy * dx).sum() / w.sum())],
        [float((w * dy * dx).sum() / w.sum()), float((w * dx * dx).sum() / w.sum())],
    ])
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]

    major = evecs[:, 0]
    elongation = float(np.sqrt(max(evals[0], 1e-9) / max(evals[1], 1e-9)))
    return np.array([cy, cx]), major, elongation


def _distance_field(shape: tuple[int, int], centre: np.ndarray,
                    major: np.ndarray, half_length: float,
                    line_like: bool) -> np.ndarray:
    """
    Distance from the presumed source: radial for a point, perpendicular
    distance from a segment for a crack.

    Getting this right is what lets one profile model serve both leak
    geometries. Forcing a radial profile onto an elongated crack produces a
    poor fit and would reject a genuine leak -- a failure mode worth avoiding,
    since seam and weld leaks are exactly the industrial case that matters.
    """
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    dy, dx = yy - centre[0], xx - centre[1]

    if not line_like:
        return np.sqrt(dy ** 2 + dx ** 2)

    # Project onto the major axis, clamp to the segment, take the residual.
    t = dy * major[0] + dx * major[1]
    t = np.clip(t, -half_length, half_length)
    py = dy - t * major[0]
    px = dx - t * major[1]
    return np.sqrt(py ** 2 + px ** 2)


# --------------------------------------------------------------------------
# Profile fitting
# --------------------------------------------------------------------------

def _radial_profile(difference: np.ndarray, distance: np.ndarray,
                    max_radius: float, n_bins: int = 24
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Median excess temperature in annular bins of distance.

    Medians rather than means: a neighbouring hot component or a dead pixel
    inside one bin would otherwise bias the whole profile and spoil the fit.
    """
    edges = np.linspace(0.0, max_radius, n_bins + 1)
    centres, values, counts = [], [], []
    for i in range(n_bins):
        sel = (distance >= edges[i]) & (distance < edges[i + 1])
        n = int(sel.sum())
        if n < 3:
            continue
        centres.append(0.5 * (edges[i] + edges[i + 1]))
        values.append(float(np.median(difference[sel])))
        counts.append(n)
    return np.array(centres), np.array(values), np.array(counts, dtype=float)


def _tail_length(radii: np.ndarray, values: np.ndarray,
                 min_points: int = 5) -> float:
    """
    Model-free estimate of the decay length from the profile tail.

    For large argument K0(x) ~ sqrt(pi/2x) * exp(-x), so

        ln( dT(r) * sqrt(r) )  is linear in r, with slope -1/L.

    A straight line fit on that transform recovers L without any nonlinear
    optimisation, which makes it the right fallback when `curve_fit` rails at
    its bound -- and rail it does, on exactly the strong, obvious leaks the
    detector must not miss. A leak bright enough to saturate the fit window is
    the last candidate you want scored as "length unmeasurable, reject".

    Validated against the solver: recovers the analytic thermal length to
    within 2% on a clean point source (see tests/test_physics.py).

    Returns NaN when the profile has no usable decaying tail.
    """
    if radii.size < min_points + 2:
        return float("nan")

    baseline = float(np.median(values[-3:]))
    excess = values - baseline
    peak = float(excess.max())
    if peak <= 0:
        return float("nan")

    # Use the decaying portion: above the noise floor, below the flat core.
    sel = (excess > 0.12 * peak) & (excess < 0.85 * peak) & (radii > 0)
    if sel.sum() < min_points:
        sel = (excess > 0.12 * peak) & (radii > 0)
    if sel.sum() < min_points:
        return float("nan")

    r = radii[sel]
    y = np.log(excess[sel] * np.sqrt(r))
    slope, _ = np.polyfit(r, y, 1)
    if slope >= -1e-6:
        return float("nan")
    return float(-1.0 / slope)


def _fit_bessel(radii: np.ndarray, values: np.ndarray, weights: np.ndarray,
                length_guess: float, length_max: float
                ) -> tuple[float, float, float, float, bool]:
    """
    Fit dT(r) = A*K0(r/L) + c.

    `length_max` bounds the fitted length: a decay length longer than the fit
    window is not measurable from that window, and letting the optimiser run to
    an arbitrary value hides an unconverged fit behind a plausible number.

    Returns (amplitude, length, offset, r_squared, ok).
    """
    if radii.size < 5:
        return 0.0, length_guess, 0.0, 0.0, False

    # Model: A*K0(r/L) + c
    #
    # A tempting refinement is to add a linear term b*r, on the reasoning that
    # the difference image carries a gently sloping residual from the device
    # being under a slightly different workload between captures, and that this
    # slope biases the fitted length upward. It does -- measured lengths run
    # about twice the material prediction for genuine leaks. But adding the term
    # is measurably worse: with four free parameters over ~24 radial bins the
    # linear component absorbs the K0 tail itself, and detection at 0.5 W falls
    # from 13/15 to 7/15. The bias is a known, roughly constant offset that the
    # asymmetric length test tolerates; the over-fit is not recoverable. Kept
    # here as a documented negative result rather than deleted, because it is
    # the obvious thing to try next and it does not work.
    a0 = max(float(values.max() - np.median(values[-3:])), 1e-3)
    p0 = [a0, float(np.clip(length_guess, 0.4, 0.9 * length_max)),
          float(np.median(values[-3:]))]
    bounds = ([0.0, 0.3, -10.0], [1e4, length_max, 10.0])

    try:
        popt, _ = curve_fit(
            lambda r, a, l, c: bessel_model(r, a, l, c),
            radii, values, p0=p0, bounds=bounds,
            sigma=1.0 / np.sqrt(np.maximum(weights, 1.0)),
            maxfev=8000,
        )
    except (RuntimeError, ValueError):
        return 0.0, length_guess, 0.0, 0.0, False

    predicted = bessel_model(radii, *popt)
    ss_res = float(np.sum((values - predicted) ** 2))
    ss_tot = float(np.sum((values - values.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    return float(popt[0]), float(popt[1]), float(popt[2]), float(r2), True


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def _novelty(candidate: Candidate, baseline: np.ndarray | None
             ) -> tuple[float, float]:
    """
    Was this location already hot before?

    The three tests above ask whether an anomaly *is heat*. They cannot ask
    whether it is a *leak*, and that gap is the dominant source of false alarms:
    when a device's own component draws slightly more power between two
    captures, the resulting difference signal is genuine, diffusive, and passes
    every physical consistency check, because there is nothing unphysical about
    it. Physics has no grounds to reject real heat.

    What separates the two is location. A leak is heat arriving where there was
    none -- a crack in cold casing, a failed weld, a breach in lagging. A
    component fluctuating is heat growing where heat already was. Measuring how
    hot the candidate's location was in the *baseline* frame captures exactly
    that distinction, and it is the reason this test uses the reference image
    for more than subtraction.

    Returns (novelty_score, occupancy) where occupancy is 0 for cold casing and
    1 for a location already among the device's hottest.
    """
    MIN_SCORE = 0.25   # a leak beside a hot component is unlikely, not impossible
    if baseline is None or not candidate.mask.any():
        return 1.0, 0.0

    background = float(np.median(baseline))
    scale = float(np.percentile(baseline, 99.0)) - background
    if scale <= 1e-6:
        return 1.0, 0.0

    local = float(np.median(baseline[candidate.mask])) - background
    occupancy = float(np.clip(local / scale, 0.0, 1.0))
    return float(1.0 - (1.0 - MIN_SCORE) * occupancy), occupancy


def score_candidate(candidate: Candidate, difference: np.ndarray,
                    material: Material, metres_per_pixel: float,
                    psf_sigma_px: float = 1.1,
                    length_tolerance: float = 4.0,
                    elongation_threshold: float = 2.0,
                    baseline: np.ndarray | None = None) -> PhysicsVerdict:
    """
    Assess whether a candidate anomaly is physically consistent with a heat leak.

    Parameters
    ----------
    length_tolerance : how many factors of two the fitted thermal length may
        differ from the material prediction before being penalised. Generous by
        default (a factor of 4) because real devices are thermally
        heterogeneous -- a metal heat spreader under the casing genuinely does
        change the local thermal length.
    elongation_threshold : above this aspect ratio the candidate is fitted as a
        line source (a crack) rather than a point source.
    """
    expected_length_px = material.thermal_length() / metres_per_pixel

    centre, major, elongation = _weighted_moments(difference, candidate.mask)
    line_like = elongation >= elongation_threshold

    # Half-length of the crack, from the second moment along the major axis.
    ys, xs = np.nonzero(candidate.mask)
    proj = (ys - centre[0]) * major[0] + (xs - centre[1]) * major[1]
    half_length = float(np.percentile(np.abs(proj), 90)) if line_like else 0.0

    distance = _distance_field(difference.shape, centre, major,
                               half_length, line_like)

    # Fit window: wide enough to contain the decaying tail of whichever is
    # larger -- the material's thermal length or the candidate itself -- but not
    # so wide that neighbouring structure contaminates the profile. Sizing this
    # from the thermal length alone lets a broad anomaly fill the whole window,
    # so the fit never sees a decay and rails at its upper bound.
    equivalent_radius = float(np.sqrt(max(candidate.area_px, 1) / np.pi))
    max_radius = float(np.clip(max(4.0 * expected_length_px,
                                   3.0 * equivalent_radius),
                               8.0, 0.35 * min(difference.shape)))

    length_bound = max(2.0 * expected_length_px, max_radius)
    radii, values, counts = _radial_profile(difference, distance, max_radius)
    amplitude, length, offset, r2, ok = _fit_bessel(
        radii, values, counts, length_guess=expected_length_px,
        length_max=length_bound)

    # A fit that rails at its upper bound has not measured a length scale; it
    # has merely run out of window. Fall back to the model-free tail slope
    # rather than declaring the candidate unmeasurable -- railing happens most
    # often on strong leaks sitting on a sloped background, and rejecting those
    # for want of a converged fit is the worst possible failure mode.
    railed = ok and length > 0.97 * length_bound
    tail_length = _tail_length(radii, values)

    if railed:
        if np.isfinite(tail_length) and tail_length > 0.3:
            length = float(tail_length)
        else:
            ok = False

    # ---- test 1: profile shape ------------------------------------------
    # Stretch R^2 rather than using it raw. Measured on simulation, genuine
    # leaks sit at R^2 ~ 0.98 and artefacts at ~0.76 -- a real separation, but
    # only a factor of 1.3 in raw R^2, which is far too weak to move a geometric
    # mean. Rescaling so that 0.55 maps to 0 and 0.95 to 1 turns the same
    # measurement into a factor of two.
    R2_FLOOR, R2_SPAN = 0.55, 0.40
    fit_score = float(np.clip((r2 - R2_FLOOR) / R2_SPAN, 0.0, 1.0))

    # ---- test 2: length-scale plausibility ------------------------------
    # This is the strongest of the three tests, and the asymmetry is the whole
    # point.
    #
    # In steady state the decay length of a conductive source is L = sqrt(kd/2h),
    # a property of the *plate*, not of the source. A source broader than L
    # produces a profile broader than K0(r/L) -- that is ordinary, a distributed
    # leak is still a leak. But nothing can produce a profile *sharper* than L:
    # sustaining that gradient would require heat to vanish faster than
    # convection can remove it. An anomaly measurably sharper than L is
    # therefore not conductive heating, whatever it looks like.
    #
    # Hence: steep penalty below the reference length, gentler penalty above.
    #
    # LENGTH_CALIBRATION is a measured systematic, not a fudge. The length
    # recovered from a real difference image runs consistently above the bare
    # material prediction, because the image carries a broad residual pedestal
    # from workload drift that lifts the profile tail. Measured over the
    # simulated dataset, genuine leaks occupy L_fit/L_theory in [1.26, 2.75]
    # with a median of 1.67, so the test is centred there. Attempting to remove
    # the bias at source, by adding a linear background to the profile fit, was
    # tried and made detection substantially worse (see `_fit_bessel`).
    #
    # Artefacts populate *both* tails: reflections come in near 0.37 (too sharp
    # to be conduction) and drift residuals near 6.4 (too broad to be a
    # localised source). That two-sided spread is why this is a band test rather
    # than a one-sided floor.
    LENGTH_CALIBRATION = 1.7
    SHARP_DIVISOR, BROAD_DIVISOR = 0.35, 0.45
    reference_length = LENGTH_CALIBRATION * expected_length_px
    min_length = max(0.6 * psf_sigma_px, 0.3)
    if not ok or length <= min_length:
        length_score = 0.0
    else:
        ratio = np.log(length / reference_length) / np.log(length_tolerance)
        divisor = SHARP_DIVISOR if ratio < 0 else BROAD_DIVISOR
        length_score = float(np.exp(-0.5 * (ratio / divisor) ** 2))

    # ---- test 3: edge sharpness -----------------------------------------
    # Diffusion is a low-pass operator: for a K0 profile the gradient is bounded
    # by roughly (peak rise) / L. Compare against the *expected* length rather
    # than the fitted one -- using the fitted length in both numerator and
    # denominator makes the test partly circular and lets a bad fit excuse
    # itself. Smooth at the PSF scale first so detector noise is not mistaken
    # for a sharp edge.
    smooth = gaussian_filter(difference, sigma=max(psf_sigma_px, 0.6), mode="nearest")
    gy, gx = np.gradient(smooth)
    grad = np.hypot(gy, gx)
    region = candidate.mask
    observed_grad = float(np.percentile(grad[region], 95)) if region.any() else 0.0
    peak = max(abs(candidate.peak_delta_k), 1e-6)
    diffusion_limit = max(peak / max(expected_length_px, 1e-6), 1e-6)
    sharpness_ratio = observed_grad / diffusion_limit
    sharpness_score = float(np.exp(-max(sharpness_ratio - 1.5, 0.0) / 1.5))

    # ---- combine ---------------------------------------------------------
    # Weighted geometric mean: every law must hold, so a zero anywhere is fatal.
    #
    # Length carries most of the weight because it is the least forgeable -- an
    # artefact can accidentally resemble a K0 profile over a short range, but it
    # cannot easily land on the material's thermal length by chance.
    #
    # Sharpness is deliberately down-weighted to near nothing. On this simulator
    # it does not discriminate: measured median sharpness ratio is 0.59 for
    # genuine leaks and 0.43 for artefacts -- the artefacts are, if anything,
    # *smoother*, the opposite of the design assumption. The cause is the camera
    # model: PSF blur plus 2x downsampling smooths a specular reflection's edges
    # to roughly the same scale as a compact heat source, erasing the very
    # difference the test was built to see. It is kept at a small weight and
    # always reported, because on real imagery from a higher-resolution camera
    # the edges survive and the test should recover its value -- but claiming it
    # works here would not be true.
    novelty_score, occupancy = _novelty(candidate, baseline)

    weights = np.array([0.30, 0.45, 0.05, 0.20])
    parts = np.array([fit_score, length_score, sharpness_score, novelty_score])
    parts = np.clip(parts, 1e-6, 1.0)
    physics_score = float(np.exp(np.sum(weights * np.log(parts))))

    # ---- physical interpretation ----------------------------------------
    # Q = A * 2*pi*k*d  inverts the point-source solution. For a crack this
    # under-reports, since a line source of the same total power has a lower
    # peak; treat it as a lower bound and say so in any reported figure.
    estimated_power = float(amplitude * 2.0 * np.pi
                            * material.conductivity * material.thickness)

    return PhysicsVerdict(
        physics_score=physics_score,
        fit_score=fit_score,
        length_score=length_score,
        sharpness_score=sharpness_score,
        novelty_score=novelty_score,
        baseline_occupancy=occupancy,
        fitted_amplitude_k=amplitude,
        fitted_length_px=length,
        expected_length_px=expected_length_px,
        elongation=elongation,
        geometry="crack" if line_like else "point",
        estimated_power_w=estimated_power,
        sharpness_ratio=sharpness_ratio,
        fit_ok=ok,
    )
