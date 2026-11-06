"""
Frame alignment.

The single most important step, and the one most often skipped. If the baseline
and inspection frames are misaligned by even one pixel, every sharp edge in the
scene -- a label, a vent, a screw -- produces a difference signal larger than a
genuine weak leak. Registration failure does not degrade the detector
gracefully; it destroys it.

Strategy: enhanced correlation coefficient (ECC) on gradient-normalised images,
with a feature-based fallback. ECC is used rather than raw phase correlation
because it estimates a full Euclidean or affine transform, and because the
gradient normalisation makes it robust to the global offset and gain drift that
separates two uncalibrated thermal captures.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Registration:
    """Result of aligning an inspection frame onto a baseline frame."""

    aligned: np.ndarray          # inspection frame warped into baseline geometry
    warp_matrix: np.ndarray      # 2x3 affine
    valid_mask: np.ndarray       # bool, False where the warp pulled in border fill
    method: str
    correlation: float           # ECC score, 1.0 is perfect

    def shift_magnitude_px(self) -> float:
        return float(np.hypot(self.warp_matrix[0, 2], self.warp_matrix[1, 2]))


def _prepare(frame: np.ndarray) -> np.ndarray:
    """
    Normalise a thermal frame for matching.

    Robust percentile scaling rather than min/max: a single hot leak pixel would
    otherwise compress the whole dynamic range and starve the matcher of the
    structural detail it actually aligns on.
    """
    lo, hi = np.percentile(frame, [2.0, 98.0])
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    norm = np.clip((frame - lo) / (hi - lo), 0.0, 1.0)
    return norm.astype(np.float32)


def register(baseline: np.ndarray, inspection: np.ndarray,
             motion: str = "euclidean", max_iterations: int = 300,
             epsilon: float = 1e-6) -> Registration:
    """
    Align `inspection` onto `baseline`.

    Parameters
    ----------
    motion : "translation", "euclidean" (default) or "affine". Euclidean is the
        right default for a tripod-mounted or hand-held inspection where the
        object is rigid and the standoff distance is roughly constant.

    Returns
    -------
    Registration, including a validity mask. Always use that mask -- pixels
    where the warp pulled in replicated border values are not real data, and
    letting them into the difference map creates a frame of false detections.
    """
    modes = {
        "translation": cv2.MOTION_TRANSLATION,
        "euclidean": cv2.MOTION_EUCLIDEAN,
        "affine": cv2.MOTION_AFFINE,
    }
    warp_mode = modes[motion]

    src = _prepare(baseline)
    dst = _prepare(inspection)

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                max_iterations, epsilon)

    method, correlation = "ecc", 0.0
    try:
        correlation, warp = cv2.findTransformECC(
            src, dst, warp, warp_mode, criteria, None, 5)
    except cv2.error:
        # ECC diverges when the frames start too far apart. Fall back to a
        # coarse phase-correlation shift and accept translation only.
        method = "phase_correlation"
        shift, response = cv2.phaseCorrelate(src.astype(np.float64),
                                             dst.astype(np.float64))
        warp = np.eye(2, 3, dtype=np.float32)
        warp[0, 2] = shift[0]
        warp[1, 2] = shift[1]
        correlation = float(response)

    h, w = baseline.shape
    aligned = cv2.warpAffine(
        inspection.astype(np.float32), warp, (w, h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan"))

    valid = np.isfinite(aligned)
    # Erode the validity mask slightly: interpolation near the border mixes in
    # the fill value before it becomes formally invalid.
    valid = cv2.erode(valid.astype(np.uint8),
                      np.ones((5, 5), np.uint8)).astype(bool)

    # Fill the pixels the warp could not reach from the *baseline*, so the
    # difference map is ~zero there rather than carrying a manufactured step.
    #
    # This used to substitute the frame median, and that was a defect with real
    # consequences. On real static frames a sub-pixel registration -- 0.075 px,
    # essentially perfect -- still leaves the outermost row and column outside
    # the source, and replacing their true ~300.05 K with a 299.43 K frame
    # median invented a 0.62 K step exactly at the rim. In the difference image
    # that rim reached 6.28 K against a 1.07 K interior: the artefact was six
    # times larger than any real signal in the frame, and it was responsible for
    # about 41% of all false alarms measured on real data, with firing density
    # 8x the frame average within 1 px of the border.
    #
    # The validity mask kept those pixels out of candidate extraction, but not
    # out of the matched filter, which happily spread the invented step inward.
    # `detect.matched_filter` now excludes them from the kernel as well; filling
    # from the baseline here means that even a caller that ignores the mask sees
    # no step rather than a large one.
    invalid = ~np.isfinite(aligned)
    aligned = np.where(invalid, baseline.astype(np.float32), aligned)

    return Registration(aligned=aligned.astype(float), warp_matrix=np.asarray(warp),
                        valid_mask=valid, method=method,
                        correlation=float(correlation))
