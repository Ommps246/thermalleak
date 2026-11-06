"""
Candidate extraction from a normalised difference map.

This stage is deliberately *permissive*. Its job is recall, not precision: find
every region that is statistically warmer than the background, and let the
physics stage in `score.py` decide which of them could actually be heat. A
detector that thresholds hard here throws away weak leaks it can never recover,
whereas false candidates are cheap -- they get rejected two steps later.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class Candidate:
    """One connected region of anomalous temperature."""

    label: int
    centroid: tuple[float, float]        # (row, col) in pixels
    area_px: int
    bbox: tuple[int, int, int, int]      # (row0, col0, row1, col1)
    peak_delta_k: float                  # brightest excess temperature, kelvin
    mean_delta_k: float
    z_peak: float                        # peak / noise_sigma
    mask: np.ndarray = field(repr=False)

    def integrated_delta(self) -> float:
        """Sum of excess temperature over the region -- a crude energy proxy."""
        return float(self.mean_delta_k * self.area_px)


def matched_filter(difference: np.ndarray, thermal_length_px: float,
                   valid: np.ndarray | None = None) -> np.ndarray:
    """
    Smooth the difference map at the scale a real leak would occupy.

    A leak is not a single hot pixel; it is a diffuse blob of characteristic
    width set by the material's thermal length. Smoothing at that scale is the
    matched filter for the expected signal, and it improves the signal-to-noise
    ratio of a weak leak by roughly the square root of the number of pixels it
    covers. Using the *physics* to choose the filter width -- rather than a
    hand-tuned constant -- is the first place physics buys detection sensitivity.

    Filtering is done as a **normalised convolution**: the masked data and the
    mask itself are each smoothed with zero padding, then divided. The kernel
    therefore averages only pixels that carry real measurements, and its weights
    are renormalised wherever part of it hangs over nothing.

    That is not a refinement, it is the fix for a measured defect. An ordinary
    `mode="nearest"` filter has to invent values wherever the kernel runs past
    the data -- at pixels the registration warp could not reach, and at the
    frame boundary itself -- and whatever it invents becomes signal. Trimming
    the border was found not to help: each trim creates a fresh boundary for the
    filter to invent against, so on real data the edge-adjacent share of false
    alarms *rose* from 73% to 96% as the crop shrank. Nothing is invented here,
    so no new border is created, and trimming is unnecessary.
    """
    sigma = max(0.35 * thermal_length_px, 0.8)
    weight = (np.ones_like(difference, dtype=float) if valid is None
              else valid.astype(float))

    numerator = gaussian_filter(difference * weight, sigma=sigma,
                                mode="constant", cval=0.0)
    denominator = gaussian_filter(weight, sigma=sigma, mode="constant", cval=0.0)

    # Guard the divide. Where too little of the kernel overlaps real data the
    # estimate is not merely noisy, it is unsupported -- return zero rather than
    # amplifying one surviving pixel by a large reciprocal.
    MIN_SUPPORT = 0.05
    return np.where(denominator > MIN_SUPPORT,
                    numerator / np.maximum(denominator, 1e-12), 0.0)


def extract_candidates(difference: np.ndarray, noise_sigma: float,
                       thermal_length_px: float,
                       valid: np.ndarray | None = None,
                       z_threshold: float = 3.5,
                       min_area_factor: float = 0.35,
                       polarity: str = "hot",
                       max_candidates: int = 12) -> list[Candidate]:
    """
    Threshold and label anomalous regions.

    Parameters
    ----------
    z_threshold : detection threshold in units of the robust noise sigma,
        applied to the matched-filtered map.
    min_area_factor : minimum blob area as a fraction of the area a source of
        one thermal length would occupy. Rejects single-pixel noise spikes
        without rejecting genuinely small leaks.
    polarity : "hot" for heat escaping, "cold" for cryogenic or coolant leaks,
        "both" when the sign is unknown.
    """
    if valid is None:
        valid = np.ones_like(difference, dtype=bool)

    filtered = matched_filter(difference, thermal_length_px, valid=valid)

    # The matched filter reduces noise by roughly the filter's effective width;
    # rescale sigma so the z threshold keeps its stated meaning.
    sigma_f = max(1.4826 * float(np.median(np.abs(
        filtered[valid] - np.median(filtered[valid])))), 1e-5)

    if polarity == "hot":
        signed = filtered
    elif polarity == "cold":
        signed = -filtered
    else:
        signed = np.abs(filtered)

    z_map = signed / sigma_f
    mask = (z_map > z_threshold) & valid

    if not mask.any():
        return []

    # Close small gaps so a leak split by noise is not counted twice.
    kernel = np.ones((3, 3), np.uint8)
    mask_u8 = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, 8)

    min_area = max(4.0, min_area_factor * np.pi * thermal_length_px ** 2)

    candidates: list[Candidate] = []
    for lab in range(1, n_labels):
        if int(stats[lab, cv2.CC_STAT_AREA]) < min_area:
            continue
        component = labels == lab

        # Locate the peak on the filtered map, not the mask centre of mass: the
        # mask boundary moves with the threshold, the peak does not.
        sub = np.where(component, signed, -np.inf)
        py, px = np.unravel_index(np.argmax(sub), sub.shape)
        peak_value = float(signed[py, px])

        # Refine the mask to the half-maximum region.
        #
        # Thresholding at a fixed multiple of *detector* noise is the wrong
        # extent criterion here: scene nuisances (drift, reflections, emissivity
        # edges) sit tens of sigma above detector noise, so a z-threshold mask
        # bleeds outward until it swallows unrelated structure, and every
        # downstream shape and profile measurement is then made on the wrong
        # region. Half-maximum is scale-free, tracks the source rather than the
        # noise floor, and is the standard extent definition in source
        # extraction for exactly this reason.
        region = component & (signed >= 0.5 * peak_value)
        region = cv2.morphologyEx(region.astype(np.uint8), cv2.MORPH_CLOSE,
                                  kernel).astype(bool)
        # Keep only the part still connected to the peak.
        n_sub, sub_labels = cv2.connectedComponents(region.astype(np.uint8), 8)
        if n_sub > 1:
            region = sub_labels == sub_labels[py, px]
        if region.sum() < 4:
            region = component

        area = int(region.sum())
        vals = difference[region]
        peak = float(np.max(np.abs(vals)) * np.sign(np.mean(vals)))

        ys, xs = np.nonzero(region)
        bbox = (int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1)

        candidates.append(Candidate(
            label=lab,
            centroid=(float(py), float(px)),
            area_px=area,
            bbox=bbox,
            peak_delta_k=peak,
            mean_delta_k=float(np.mean(vals)),
            z_peak=float(z_map[py, px]),
            mask=region,
        ))

    candidates.sort(key=lambda c: c.z_peak, reverse=True)
    return candidates[:max_candidates]
