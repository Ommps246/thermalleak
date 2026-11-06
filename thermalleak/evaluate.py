"""
Evaluation: ROC, detection limits, localisation, and the physics ablation.

The ablation is the experiment that decides whether this project has a point.
Turning the physics stage off leaves a competent conventional change detector:
register, normalise, difference, threshold. If the physics-informed version does
not beat that, the physics is decoration and it is far better to discover this
in week two than in peer review.

A detection counts as correct only if it is *localised* on the true leak. A run
that flags a reflection on the other side of the frame while a leak is present
is a false alarm that happens to coincide with a positive label, and scoring it
as a hit is the most common way an anomaly-detection evaluation flatters itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pipeline import detect_leak
from .simulate import ScenePair


@dataclass
class PairOutcome:
    """Scored result for one pair."""

    has_leak: bool
    leak_power_watts: float
    score: float
    localised: bool
    distance_px: float
    physics_score: float
    estimated_power_w: float


def evaluate_pair(pair: ScenePair, use_physics: bool = True,
                  localisation_tolerance_px: float = 8.0,
                  **kwargs) -> PairOutcome:
    """Run the pipeline on one pair and score it against ground truth."""
    result = detect_leak(pair.baseline, pair.inspection,
                         material=pair.material,
                         metres_per_pixel=pair.metres_per_pixel,
                         use_physics=use_physics, **kwargs)

    best = result.best()
    distance = float("inf")
    localised = False

    if best is not None and pair.leak_centre_px is not None:
        distance = float(np.hypot(best.centroid[0] - pair.leak_centre_px[0],
                                  best.centroid[1] - pair.leak_centre_px[1]))
        localised = distance <= localisation_tolerance_px

    return PairOutcome(
        has_leak=pair.has_leak,
        leak_power_watts=pair.leak_power_watts,
        score=float(result.score),
        localised=localised,
        distance_px=distance,
        physics_score=float(best.verdict.physics_score)
        if (best is not None and best.verdict is not None) else float("nan"),
        estimated_power_w=float(best.verdict.estimated_power_w)
        if (best is not None and best.verdict is not None) else float("nan"),
    )


def roc_curve(outcomes: list[PairOutcome], require_localisation: bool = True
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    ROC over the detection score, plus the area under it.

    Parameters
    ----------
    require_localisation : if True (the default and the honest choice), a
        positive pair only counts as a true positive when the top-ranked
        candidate actually sits on the leak.

    Returns
    -------
    (false_positive_rate, true_positive_rate, thresholds, auc)
    """
    scores = np.array([o.score for o in outcomes], dtype=float)
    if require_localisation:
        labels = np.array([o.has_leak and o.localised for o in outcomes])
    else:
        labels = np.array([o.has_leak for o in outcomes])

    order = np.argsort(-scores)
    scores, labels = scores[order], labels[order]

    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([]), float("nan")

    tp = np.cumsum(labels)
    fp = np.cumsum(~labels)
    tpr = np.concatenate([[0.0], tp / n_pos, [1.0]])
    fpr = np.concatenate([[0.0], fp / n_neg, [1.0]])

    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") \
        else float(np.trapz(tpr, fpr))
    return fpr, tpr, scores, auc


def detection_rate_by_power(outcomes: list[PairOutcome], threshold: float,
                            bins: np.ndarray | None = None
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detection rate as a function of leak power -- the sensitivity curve.

    This is the figure an engineer actually wants: not "the AUC was 0.94" but
    "leaks above N watts are caught reliably, below that they are not".

    Returns (bin_centres_watts, detection_rate, counts).
    """
    positives = [o for o in outcomes if o.has_leak]
    if not positives:
        return np.array([]), np.array([]), np.array([])

    powers = np.array([o.leak_power_watts for o in positives])
    hits = np.array([o.score >= threshold and o.localised for o in positives])

    if bins is None:
        bins = np.geomspace(max(powers.min(), 1e-3), powers.max(), 8)

    centres, rates, counts = [], [], []
    for i in range(len(bins) - 1):
        sel = (powers >= bins[i]) & (powers < bins[i + 1])
        if sel.sum() == 0:
            continue
        centres.append(float(np.sqrt(bins[i] * bins[i + 1])))
        rates.append(float(hits[sel].mean()))
        counts.append(int(sel.sum()))
    return np.array(centres), np.array(rates), np.array(counts)


def threshold_at_fpr(outcomes: list[PairOutcome], target_fpr: float = 0.05
                     ) -> float:
    """
    Score threshold giving at most `target_fpr` false alarms on no-leak pairs.

    Operating points must be chosen on the negatives, never on the positives.
    Picking the threshold that maximises accuracy over the whole set quietly
    tunes on the test labels.
    """
    negatives = np.array([o.score for o in outcomes if not o.has_leak])
    if negatives.size == 0:
        return 0.0
    return float(np.quantile(negatives, 1.0 - target_fpr))


def summarise(outcomes: list[PairOutcome], label: str,
              target_fpr: float = 0.05) -> dict:
    """Headline numbers for one configuration."""
    fpr, tpr, _, auc = roc_curve(outcomes)
    thr = threshold_at_fpr(outcomes, target_fpr)

    positives = [o for o in outcomes if o.has_leak]
    negatives = [o for o in outcomes if not o.has_leak]

    detected = [o for o in positives if o.score >= thr and o.localised]
    false_alarms = [o for o in negatives if o.score >= thr]

    distances = [o.distance_px for o in detected if np.isfinite(o.distance_px)]

    return dict(
        label=label,
        n_pairs=len(outcomes),
        auc=auc,
        threshold=thr,
        recall_at_fpr=len(detected) / max(len(positives), 1),
        false_alarm_rate=len(false_alarms) / max(len(negatives), 1),
        median_localisation_px=float(np.median(distances)) if distances else float("nan"),
    )
