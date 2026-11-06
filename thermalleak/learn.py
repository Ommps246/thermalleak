"""
A learned arm for the ablation: can a classifier beat the hand-weighted physics?

This module is deliberately **additive**. It imports the physics/CV pipeline and
never modifies it, because that pipeline is the baseline being measured against
and a baseline you edit is not a baseline. Every feature here is something
`score.py` already computes on its way to a verdict; nothing new is measured
from the image.

The question
------------
`score.py` combines four physical tests as a weighted geometric mean with
weights chosen by hand -- profile 0.30, length 0.45, sharpness 0.05, novelty
0.20 -- and the 0.45 on the length test is the single most consequential
constant in the project. It was set by reasoning about which test is hardest to
forge, not by fitting. A classifier given the same four numbers will weight them
however the data says to. If it lands somewhere near 0.45 on length, the hand
weighting is vindicated; if it does not, that is worth knowing before the weight
is defended in a paper.

Two models are trained on exactly that four-number input (`SCORE_FEATURES`) so
their importances are directly comparable to the hand weights, and again on a
wider set (`ALL_FEATURES`) that adds the raw geometry the scores were derived
from, to see whether the compression into four scores is throwing anything away.

`physics_score` itself is never a feature. Handing the model the hand-weighted
answer and asking whether it can reproduce it would be circular.

One result from this module needs a caveat carried with it wherever it is
quoted. The novelty test scores a permutation importance around 0.02 against its
hand-assigned 0.20, which reads as a hand weight badly overset -- but novelty
exists to catch a component drawing slightly more power between captures, and
that nuisance is largely absent from the current simulator. `SOURCE_JITTER` in
`simulate.py` was reduced from +/-10% to +/-2.5% as physically unrealistic for a
device held at matched load; at the old setting it moved the SoC by ~0.26 W
between frames, larger than most of the leaks being hunted, and it dominated the
false positives. The README's "0.19 -> 0.78" figure for adding novelty comes
from that configuration and is not reproducible here. Both numbers measure how
much a nuisance matters, taken either side of a change in how much of it the
simulator contains; neither settles what novelty is worth on real hardware.

Methodology, and why it is fussy
--------------------------------
**Grouping.** Candidates are not independent. One frame pair yields up to a
dozen of them, sharing a scene, a noise realisation, a registration error and a
drift. Splitting at the candidate level puts siblings on both sides of the split
and reports a number that has partly memorised the test set. Every split here is
by *simulator seed*, so an entire pair lands wholly in train, validation or test.

**Thresholds.** Chosen on validation, applied unchanged to test and to real
data. A threshold picked on the data it is scored against is not a threshold.

**Generalisation.** Three tests of decreasing kindness, because a model that
only works on its training distribution has not learned physics, it has learned
this simulator:

  1. held-out seeds, same material  -- the easy case
  2. a material never trained on    -- `expected_length_px` changes by 4x
  3. real OTCBVS frames             -- nothing about them was simulated

The gap between (1) and (3) is the result worth reporting.

scikit-learn is required for this module and this module only; the detection
pipeline does not import it. It is deliberately not in `requirements.txt`, which
pins the stack the published numbers reproduce on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .detect import Candidate
from .physics import MATERIALS, Material
from .pipeline import detect_leak
from .score import PhysicsVerdict
from .simulate import ScenePair, generate_pair

# The four test scores, in the order `score.py` weights them. Kept as its own
# list so a model trained on these alone yields importances that sit directly
# beside the hand weights (0.30, 0.45, 0.05, 0.20).
SCORE_FEATURES: tuple[str, ...] = (
    "fit_score", "length_score", "sharpness_score", "novelty_score",
)

# The raw quantities the scores were computed from, plus the candidate geometry.
# `log_length_ratio` is included because it is what the length test actually
# thresholds -- `length_score` is a Gaussian in exactly this quantity, so giving
# the model the pre-squashed version tests whether the squashing loses anything.
GEOMETRY_FEATURES: tuple[str, ...] = (
    "fitted_length_px", "expected_length_px", "log_length_ratio",
    "elongation", "peak_delta_k", "z_peak", "area_px", "sharpness_ratio",
)

ALL_FEATURES: tuple[str, ...] = SCORE_FEATURES + GEOMETRY_FEATURES

# Hand weights from score.py, for side-by-side comparison with what is learned.
HAND_WEIGHTS: dict[str, float] = {
    "fit_score": 0.30, "length_score": 0.45,
    "sharpness_score": 0.05, "novelty_score": 0.20,
}

LOCALISATION_TOLERANCE_PX = 8.0   # same criterion evaluate.py uses for a "hit"


@dataclass
class CandidateSet:
    """Per-candidate features, labels and the group each candidate came from."""

    features: np.ndarray             # (n_candidates, n_features)
    labels: np.ndarray               # (n_candidates,) 1 = on the true leak
    groups: np.ndarray               # (n_candidates,) simulator seed / frame id
    physics_score: np.ndarray        # (n_candidates,) the hand-weighted verdict
    feature_names: tuple[str, ...]
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def subset(self, names: tuple[str, ...]) -> np.ndarray:
        """Columns for a named feature subset, in the order given."""
        index = {n: i for i, n in enumerate(self.feature_names)}
        return self.features[:, [index[n] for n in names]]

    def select_groups(self, groups) -> "CandidateSet":
        keep = np.isin(self.groups, np.asarray(list(groups)))
        return CandidateSet(
            features=self.features[keep], labels=self.labels[keep],
            groups=self.groups[keep], physics_score=self.physics_score[keep],
            feature_names=self.feature_names, meta=dict(self.meta))

    def summary(self) -> str:
        n_pos = int(self.labels.sum())
        return (f"{len(self)} candidates from {len(np.unique(self.groups))} "
                f"frame pairs, {n_pos} on a true leak "
                f"({100.0 * n_pos / max(len(self), 1):.1f}%)")


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------

def candidate_features(candidate: Candidate, verdict: PhysicsVerdict) -> dict:
    """
    Everything `score.py` already worked out about one candidate.

    Nothing is measured from the image here. If a feature is in this dict it is
    because the physics pipeline computed it on the way to a verdict, which
    keeps the comparison honest: the classifier and the geometric mean are
    looking at identical information.
    """
    expected = max(float(verdict.expected_length_px), 1e-6)
    fitted = max(float(verdict.fitted_length_px), 1e-6)
    return {
        "fit_score": float(verdict.fit_score),
        "length_score": float(verdict.length_score),
        "sharpness_score": float(verdict.sharpness_score),
        "novelty_score": float(verdict.novelty_score),
        "fitted_length_px": fitted,
        "expected_length_px": expected,
        "log_length_ratio": float(np.log(fitted / expected)),
        "elongation": float(verdict.elongation),
        "peak_delta_k": float(candidate.peak_delta_k),
        "z_peak": float(candidate.z_peak),
        "area_px": float(candidate.area_px),
        "sharpness_ratio": float(verdict.sharpness_ratio),
    }


def _rows_from_result(result, truth_centre, group: int) -> tuple[list, list, list, list]:
    feats, labels, groups, physics = [], [], [], []
    for finding in result.findings:
        if finding.verdict is None:
            continue
        row = candidate_features(finding.candidate, finding.verdict)
        on_leak = False
        if truth_centre is not None:
            distance = float(np.hypot(finding.centroid[0] - truth_centre[0],
                                      finding.centroid[1] - truth_centre[1]))
            on_leak = distance <= LOCALISATION_TOLERANCE_PX
        feats.append([row[n] for n in ALL_FEATURES])
        labels.append(int(on_leak))
        groups.append(group)
        physics.append(float(finding.verdict.physics_score))
    return feats, labels, groups, physics


def pair_to_rows(pair: ScenePair, group: int) -> tuple[list, list, list, list]:
    """Run the unmodified pipeline on one labelled pair and harvest candidates."""
    result = detect_leak(pair.baseline, pair.inspection,
                         material=pair.material,
                         metres_per_pixel=pair.metres_per_pixel)
    return _rows_from_result(result, pair.leak_centre_px, group)


# --------------------------------------------------------------------------
# Dataset construction
# --------------------------------------------------------------------------

def build_simulated(seeds, material: Material | str = "abs_casing",
                    leak_power_range: tuple[float, float] = (0.01, 0.5),
                    positive_fraction: float = 0.5,
                    progress: bool = False,
                    **pair_kwargs) -> CandidateSet:
    """
    Generate one labelled pair per seed and harvest its candidates.

    One seed produces one pair, so `group == seed == pair`. That identity is the
    point: it makes a group-wise split impossible to get wrong later, which a
    single dataset-level seed (as `generate_dataset` uses) would not.

    Power is drawn log-uniformly over `leak_power_range`, matching
    `simulate.generate_dataset`, so weak leaks -- the regime that decides
    everything -- are properly represented rather than swamped by obvious ones.

    `pair_kwargs` pass straight to `generate_pair` -- `extent_m`, for instance,
    changes the field of view and so the pixel scale, which is how the
    camera-distance transfer test is built. Omitted, nothing changes.
    """
    if isinstance(material, str):
        material = MATERIALS[material]

    feats, labels, groups, physics = [], [], [], []
    for i, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed))
        if rng.random() < positive_fraction:
            lo, hi = np.log(leak_power_range[0]), np.log(leak_power_range[1])
            power = float(np.exp(rng.uniform(lo, hi)))
        else:
            power = 0.0
        pair = generate_pair(rng, material=material, leak_power_watts=power,
                             **pair_kwargs)
        f, l, g, p = pair_to_rows(pair, int(seed))
        feats += f; labels += l; groups += g; physics += p
        if progress and (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(seeds)} pairs", flush=True)

    return CandidateSet(
        features=np.asarray(feats, dtype=float).reshape(-1, len(ALL_FEATURES)),
        labels=np.asarray(labels, dtype=int),
        groups=np.asarray(groups, dtype=int),
        physics_score=np.asarray(physics, dtype=float),
        feature_names=ALL_FEATURES,
        meta=dict(source="simulation", material=material.name,
                  n_pairs=len(list(seeds))))


def build_from_frame_pairs(pairs, material: Material | str = "abs_casing",
                           metres_per_pixel: float = 0.0025,
                           truth_centres=None) -> CandidateSet:
    """
    Harvest candidates from arbitrary frame pairs -- real ones, in practice.

    `truth_centres` is None for genuinely static pairs, where every candidate is
    by construction a false alarm, and a (row, col) per pair when a leak was
    injected at a known place.
    """
    if isinstance(material, str):
        material = MATERIALS[material]

    feats, labels, groups, physics = [], [], [], []
    for i, (baseline, inspection) in enumerate(pairs):
        result = detect_leak(baseline, inspection, material=material,
                             metres_per_pixel=metres_per_pixel)
        centre = None if truth_centres is None else truth_centres[i]
        f, l, g, p = _rows_from_result(result, centre, i)
        feats += f; labels += l; groups += g; physics += p

    return CandidateSet(
        features=np.asarray(feats, dtype=float).reshape(-1, len(ALL_FEATURES)),
        labels=np.asarray(labels, dtype=int),
        groups=np.asarray(groups, dtype=int),
        physics_score=np.asarray(physics, dtype=float),
        feature_names=ALL_FEATURES,
        meta=dict(source="frames", material=material.name,
                  n_pairs=len(list(pairs))))


def split_seeds(seeds, fractions=(0.6, 0.2, 0.2), seed: int = 0):
    """
    Partition *seeds* -- never candidates -- into train / validation / test.

    Shuffled first so the split does not coincide with any ordering the seed
    numbers might carry.
    """
    seeds = np.asarray(list(seeds))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(seeds)
    n = len(shuffled)
    a = int(round(fractions[0] * n))
    b = a + int(round(fractions[1] * n))
    return shuffled[:a], shuffled[a:b], shuffled[b:]


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def make_models(random_state: int = 0) -> dict:
    """
    Gradient boosting and logistic regression.

    Deliberately small: a few hundred candidates per class over at most twelve
    features does not support a large model, and an over-parameterised one would
    make the simulation-to-real gap look like a modelling failure rather than
    the distribution shift it is.

    `class_weight="balanced"` on the logistic regression because candidates run
    roughly six negatives to one positive; gradient boosting is left unweighted
    and compared on ranking metrics, which are insensitive to the base rate.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.9, random_state=random_state),
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced",
                                      random_state=random_state)),
        ]),
    }


# The direction each physics score is allowed to push a candidate's leak
# probability. Every test score is built so that 1.0 means "passes this physical
# law", so a candidate that passes a test *better* must never become *less*
# likely to be a leak. That is not a fitted preference; it is what the scores
# mean. Geometry features carry no such guarantee -- a fitted length can be
# wrong in either direction -- so they are left free (0).
PHYSICS_DIRECTION: dict[str, int] = {
    "fit_score": +1, "length_score": +1,
    "sharpness_score": +1, "novelty_score": +1,
}


def make_constrained_models(features: tuple[str, ...],
                            random_state: int = 0) -> dict:
    """
    The physics-informed learner, and the control that makes it testable.

    Both are HistGradientBoostingClassifier with identical hyperparameters,
    sized to match `make_models`. The only difference is `monotonic_cst`: the
    constrained model must respect PHYSICS_DIRECTION, the control need not.

    The control is not optional. Monotonic constraints are unavailable on the
    GradientBoostingClassifier used by `make_models`, so the constrained model
    necessarily changes algorithm as well. Without an unconstrained model on the
    *same* algorithm, any difference could be the algorithm swap rather than the
    physics, and the experiment would not be able to tell them apart.

    `early_stopping=False` so a run is deterministic regardless of dataset size;
    sklearn otherwise switches it on above 10,000 samples.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    common = dict(max_iter=200, max_depth=3, learning_rate=0.05,
                  early_stopping=False, random_state=random_state)
    constraint = [PHYSICS_DIRECTION.get(f, 0) for f in features]
    return {
        "hist_gb (unconstrained)": HistGradientBoostingClassifier(**common),
        "hist_gb (physics-monotone)": HistGradientBoostingClassifier(
            monotonic_cst=constraint, **common),
    }


def threshold_at_fpr(scores: np.ndarray, labels: np.ndarray,
                     target_fpr: float) -> float:
    """
    Lowest threshold whose false-positive rate on *these* candidates is at or
    below `target_fpr`.

    Always call this on validation data. Choosing an operating point on the same
    candidates it is then scored against reports the best threshold in
    hindsight, which no deployment ever gets.
    """
    negatives = np.sort(scores[labels == 0])
    n = negatives.size
    if n == 0:
        return 0.5

    # `allowed` is how many negatives may sit at or above the threshold. Taking
    # the (n - allowed)-th order statistic gives exactly that count when scores
    # are distinct -- one past the naive quantile, which admits one negative too
    # many.
    allowed = int(np.floor(target_fpr * n))
    if allowed <= 0:
        return float(np.nextafter(negatives[-1], np.inf))

    threshold = float(negatives[n - allowed])

    # Ties break the order-statistic argument: many candidates score exactly
    # 0.0 when a fit fails, so a threshold landing on a repeated value can admit
    # far more than `allowed`. Step up to the next distinct score, which
    # undershoots the target rather than overshooting it. Undershooting is the
    # safe direction -- it reports a stricter operating point than requested,
    # never a looser one dressed up as the requested rate.
    if int((negatives >= threshold).sum()) > allowed:
        higher = negatives[negatives > threshold]
        threshold = (float(np.nextafter(negatives[-1], np.inf))
                     if higher.size == 0 else float(higher[0]))
    return threshold


def evaluate_scores(scores: np.ndarray, labels: np.ndarray,
                    threshold: float) -> dict:
    """AUC plus the confusion at a threshold fixed elsewhere."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    both = len(np.unique(labels)) > 1
    predicted = scores >= threshold
    pos, neg = labels == 1, labels == 0
    return dict(
        n=int(len(labels)), n_pos=int(pos.sum()), n_neg=int(neg.sum()),
        auc=float(roc_auc_score(labels, scores)) if both else float("nan"),
        average_precision=(float(average_precision_score(labels, scores))
                           if both else float("nan")),
        tpr=float(predicted[pos].mean()) if pos.any() else float("nan"),
        fpr=float(predicted[neg].mean()) if neg.any() else float("nan"),
        threshold=float(threshold),
    )


def fit_and_evaluate(train: CandidateSet, val: CandidateSet,
                     tests: dict[str, CandidateSet],
                     features: tuple[str, ...] = SCORE_FEATURES,
                     target_fpr: float = 0.05,
                     random_state: int = 0,
                     models: dict | None = None) -> dict:
    """
    Train on `train`, pick every threshold on `val`, report on each of `tests`.

    The hand-weighted `physics_score` is carried through the identical path --
    same candidates, same threshold rule, same validation set -- so the
    comparison is like for like rather than a learned model at its best
    threshold against a fixed 0.45 cut.

    `models` defaults to `make_models(random_state)`, so every existing caller
    gets exactly what it got before.
    """
    Xtr, ytr = train.subset(features), train.labels
    Xva, yva = val.subset(features), val.labels

    out: dict = {"features": features, "results": {}, "models": {}}

    if models is None:
        models = make_models(random_state)
    for name, model in models.items():
        model.fit(Xtr, ytr)
        val_scores = model.predict_proba(Xva)[:, 1]
        threshold = threshold_at_fpr(val_scores, yva, target_fpr)
        entry = {"validation": evaluate_scores(val_scores, yva, threshold)}
        for tname, tset in tests.items():
            s = model.predict_proba(tset.subset(features))[:, 1]
            entry[tname] = evaluate_scores(s, tset.labels, threshold)
        out["results"][name] = entry
        out["models"][name] = model

    # The incumbent, through the same machinery.
    threshold = threshold_at_fpr(val.physics_score, yva, target_fpr)
    entry = {"validation": evaluate_scores(val.physics_score, yva, threshold)}
    for tname, tset in tests.items():
        entry[tname] = evaluate_scores(tset.physics_score, tset.labels, threshold)
    out["results"]["physics_score (hand-weighted)"] = entry

    return out


# --------------------------------------------------------------------------
# Feature importance
# --------------------------------------------------------------------------

def feature_importances(model, features: tuple[str, ...],
                        val: CandidateSet, random_state: int = 0) -> dict:
    """
    Impurity/coefficient importance plus permutation importance.

    Permutation importance is measured on *validation*, and it is the one to
    trust. Impurity importance is biased toward features with many distinct
    values, which here means the continuous geometry columns would outrank the
    four bounded scores for reasons that have nothing to do with usefulness.
    """
    from sklearn.inspection import permutation_importance

    X, y = val.subset(features), val.labels

    native = None
    if hasattr(model, "feature_importances_"):
        native = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "named_steps") and "lr" in getattr(model, "named_steps", {}):
        # Coefficients on standardised inputs are comparable across features.
        native = np.abs(model.named_steps["lr"].coef_.ravel())

    perm = permutation_importance(model, X, y, n_repeats=20,
                                  random_state=random_state, scoring="roc_auc")

    def normalise(v):
        v = np.clip(np.asarray(v, dtype=float), 0.0, None)
        total = v.sum()
        return v / total if total > 0 else v

    return {
        "features": list(features),
        "native": None if native is None else normalise(native).tolist(),
        "permutation": normalise(perm.importances_mean).tolist(),
        "permutation_raw": perm.importances_mean.tolist(),
        "permutation_std": perm.importances_std.tolist(),
    }
