"""
Tests for the learned arm.

Separate file from `test_physics.py` on purpose: this module needs scikit-learn,
which the detection pipeline does not, and a missing optional dependency should
skip these rather than break the suite that guards the physics.

The tests that matter here are the methodological ones. A leaked split or a
threshold chosen on the test set produces a number that looks better and means
nothing, and neither failure is visible by reading the output -- so they get
asserted rather than trusted.
"""

from __future__ import annotations

import numpy as np
import pytest

from thermalleak.learn import (
    ALL_FEATURES, SCORE_FEATURES, CandidateSet, build_simulated,
    candidate_features, split_seeds, threshold_at_fpr,
)

sklearn = pytest.importorskip("sklearn")


def _toy(n=240, seed=0) -> CandidateSet:
    """A separable toy set with 20 candidates in each of 12 groups."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(12), n // 12)
    labels = (rng.random(n) < 0.3).astype(int)
    feats = rng.normal(size=(n, len(ALL_FEATURES)))
    feats[:, 1] += 2.0 * labels          # length_score carries the signal
    return CandidateSet(features=feats, labels=labels, groups=groups,
                        physics_score=rng.random(n),
                        feature_names=ALL_FEATURES)


def test_split_is_by_group_and_leaks_nothing():
    """
    No group may appear on both sides of a split.

    This is the failure the whole module is arranged to avoid: candidates from
    one frame pair share a scene, a drift and a noise realisation, so a split
    that separates them reports partly-memorised performance.
    """
    train, val, test = split_seeds(np.arange(200), seed=3)
    assert len(train) + len(val) + len(test) == 200
    for a, b in ((train, val), (train, test), (val, test)):
        assert not set(a.tolist()) & set(b.tolist())


def test_select_groups_keeps_candidates_together():
    data = _toy()
    subset = data.select_groups([0, 1, 2])
    assert set(np.unique(subset.groups).tolist()) == {0, 1, 2}
    assert len(subset) == sum(int((data.groups == g).sum()) for g in (0, 1, 2))


def test_threshold_hits_the_requested_false_positive_rate():
    rng = np.random.default_rng(0)
    labels = np.r_[np.zeros(800, int), np.ones(200, int)]
    scores = np.r_[rng.normal(0, 1, 800), rng.normal(2.5, 1, 200)]
    thr = threshold_at_fpr(scores, labels, 0.05)
    achieved = float((scores[labels == 0] >= thr).mean())
    assert achieved <= 0.05 + 1e-9
    assert achieved > 0.02, "threshold is far stricter than asked -- check the quantile"


def test_physics_score_is_not_among_the_features():
    """
    Handing the model the hand-weighted answer would make the comparison
    circular: it could recover `physics_score` by copying it.
    """
    assert "physics_score" not in ALL_FEATURES
    assert set(SCORE_FEATURES) <= set(ALL_FEATURES)


def test_candidate_features_are_finite_and_complete():
    from thermalleak.detect import extract_candidates
    from thermalleak.normalize import normalise
    from thermalleak.register import register
    from thermalleak.score import score_candidate
    from thermalleak.simulate import generate_pair

    pair = generate_pair(np.random.default_rng(4), leak_power_watts=0.3)
    reg = register(pair.baseline, pair.inspection)
    norm = normalise(pair.baseline, reg.aligned, valid=reg.valid_mask)
    length = pair.material.thermal_length() / pair.metres_per_pixel
    cands = extract_candidates(norm.difference, norm.noise_sigma, length,
                               valid=reg.valid_mask)
    assert cands, "expected at least one candidate on a 0.3 W leak"

    verdict = score_candidate(cands[0], norm.difference, pair.material,
                              pair.metres_per_pixel, baseline=pair.baseline)
    row = candidate_features(cands[0], verdict)
    assert set(row) == set(ALL_FEATURES)
    assert all(np.isfinite(v) for v in row.values()), row


def test_end_to_end_on_a_small_simulated_set():
    """
    The whole path: simulate, harvest, split by seed, train, threshold on
    validation. Small and slow-ish, but it is the only test that would catch a
    shape or ordering mistake between the pipeline and the feature matrix.
    """
    from thermalleak.learn import fit_and_evaluate

    seeds = np.arange(9000, 9040)
    data = build_simulated(seeds)
    assert len(data) > 20
    assert data.features.shape[1] == len(ALL_FEATURES)
    assert np.isfinite(data.features).all()

    train_s, val_s, test_s = split_seeds(seeds, seed=1)
    train, val, test = (data.select_groups(s) for s in (train_s, val_s, test_s))
    if train.labels.sum() == 0 or val.labels.sum() == 0:
        pytest.skip("too few positives in this small split to fit anything")

    out = fit_and_evaluate(train, val, {"test": test}, features=SCORE_FEATURES)
    for name, res in out["results"].items():
        assert 0.0 <= res["test"]["fpr"] <= 1.0, name
        assert np.isfinite(res["test"]["auc"]) or test.labels.sum() == 0


def test_monotone_model_never_uses_a_physics_score_backwards():
    """
    The physics-monotone learner must be non-decreasing in every test score.

    This is the whole content of the constraint, so it is checked directly
    rather than trusted to sklearn: sweep each score from 0 to 1 with the other
    inputs held fixed, and the predicted leak probability may never fall.
    """
    from thermalleak.learn import PHYSICS_DIRECTION, make_constrained_models

    data = _toy()
    X = data.subset(SCORE_FEATURES)
    model = make_constrained_models(SCORE_FEATURES)["hist_gb (physics-monotone)"]
    model.fit(X, data.labels)

    base = np.median(X, axis=0)
    for j, name in enumerate(SCORE_FEATURES):
        assert PHYSICS_DIRECTION[name] == +1
        grid = np.tile(base, (50, 1))
        grid[:, j] = np.linspace(X[:, j].min(), X[:, j].max(), 50)
        p = model.predict_proba(grid)[:, 1]
        assert np.all(np.diff(p) >= -1e-12), f"{name} used backwards"


def test_geometry_features_are_left_unconstrained():
    """Only the physics scores carry a guaranteed direction; geometry must stay free."""
    from thermalleak.learn import make_constrained_models

    model = make_constrained_models(ALL_FEATURES)["hist_gb (physics-monotone)"]
    cst = list(model.monotonic_cst)
    for name, c in zip(ALL_FEATURES, cst):
        assert c == (1 if name in SCORE_FEATURES else 0), name


def test_fit_and_evaluate_defaults_are_unchanged():
    """
    Adding the `models=` argument must not change what existing callers get.
    run_learning.py relies on the default being exactly make_models().
    """
    from thermalleak.learn import fit_and_evaluate, make_models

    data = _toy(n=480, seed=1)
    tr, va = data.select_groups(range(0, 6)), data.select_groups(range(6, 12))
    a = fit_and_evaluate(tr, va, {}, features=SCORE_FEATURES, random_state=3)
    b = fit_and_evaluate(tr, va, {}, features=SCORE_FEATURES, random_state=3,
                         models=make_models(3))
    assert set(a["results"]) == set(b["results"])
    for name in a["results"]:
        assert a["results"][name]["validation"]["auc"] == b["results"][name]["validation"]["auc"]
