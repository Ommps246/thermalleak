#!/usr/bin/env python3
"""
Does constraining the learner to respect physics buy transfer?

    python3 scripts/run_constrained_learning.py --outdir results

The question
------------
`run_learning.py` found that gradient boosting on the four physics scores beats
the hand-weighted geometric mean on the material it was trained on, then loses
to it on a material it never saw. The learner rediscovers the thermal length
from data and overfits the material.

One explanation is that an unconstrained learner is free to use a physics score
*backwards* wherever the training material happens to reward it -- to decide,
say, that a slightly *better* length match is slightly *less* leak-like in some
corner of feature space. Nothing about the physics permits that, and a new
material is exactly where such accidents stop paying.

If that is the cause, forbidding it should help. Each test score is built so
that 1.0 means "passes this physical law", so a monotonic constraint requiring
leak probability to be non-decreasing in every score encodes the physics
directly and nothing else. That is the physics-informed learner tested here.

Design
------
Same data as `run_learning.py` (seed base 100000, 600 abs_casing pairs, 200
held-out steel_pipe pairs), same seed-wise 60/20/20 splits, same five repeats,
thresholds chosen on validation and applied unchanged. Five scorers per split:

  gradient_boosting           the original learner (sklearn GBC)
  logistic_regression         linear, so monotone by form but sign is free
  hist_gb (unconstrained)     CONTROL: same algorithm as the next row, no constraint
  hist_gb (physics-monotone)  the physics-informed learner
  physics_score               the hand-weighted geometric mean

The control is the row that makes this an experiment. Monotonic constraints
need HistGradientBoosting, a different algorithm from the original, so without
an unconstrained run of the *same* algorithm any change could be the swap.

Prediction, written before running
----------------------------------
If backwards use of physics scores causes the transfer loss, then on the four
score features the monotone model should keep in-distribution AUC close to the
unconstrained models AND close most of the transfer gap to the physics score.

If the monotone and unconstrained HistGB transfer equally badly, the constraint
is not what matters and the hypothesis is wrong -- report that.

If the unconstrained HistGB already transfers well, the original loss was
specific to sklearn's GradientBoostingClassifier, and the story is about the
algorithm, not the physics -- report that too.

What was found -- added after the first run, kept apart from the prediction
--------------------------------------------------------------------------
The prediction failed: on the four scores the constraint moves transfer AUC by
+0.014, 95% CI [-0.009, +0.037], indistinguishable from zero.

An unpredicted pattern appeared on the 12-feature set: logistic regression lost
nothing in transfer while every tree model collapsed.
The likely mechanism, from inspecting the fitted models: `expected_length_px`
is constant within one training material, so its coefficient is exactly zero --
no learner trained on one material can learn how the thermal length matters;
the hand-weighted score has it as an input instead. And four geometry features
shift 2-7x between materials, putting transfer candidates outside the training
range, where trees saturate and lose rank order while a linear model
extrapolates and keeps it. This is post-hoc and rests on one transfer material;
it needs a second before it is a claim.

Second prediction, written before the camera-scale run
-------------------------------------------------------
If the mechanism above is right -- trees fail because pixel-scale geometry lands
outside their training range, not because the material changed -- then imaging
the SAME material at a different camera distance must reproduce it: on the 12
features, tree models lose substantially (gap > 0.05) while logistic regression
loses little (gap < 0.02), and the physics score, being in physical units,
holds. If logistic regression also collapses under a pure scale change, the
mechanism is wrong or incomplete, and the steel-pipe result was luck.

Outcome of the second prediction: partly falsified
--------------------------------------------------
Run with `--transfer-fields 0.16 0.64` (results/constrained_learning.txt).
On the 12 features, worst-case drop from in-distribution AUC -- the "transfer
gap" column the script prints, all values means over the five splits:

    physics score        0.015    (steel 0.951, x2.0 0.910, x0.5 0.905)
    logistic regression  0.057    (steel 0.968, x2.0 0.932, x0.5 0.912)
    gradient boosting    0.145
    trees (HistGB)       0.163-0.167

The tree half held: trees lost 0.10-0.12 zooming out and 0.052-0.065 zooming
in, over the 0.05 threshold both ways (gradient boosting only just, zooming in).
The linear half failed: logistic regression lost 0.036 and 0.056, not < 0.02.
So trees do degrade out of range as predicted, but linear models are not immune
-- the steel-pipe "zero loss" was partly specific to that shift, and the
explanation was half right. Recorded rather than rescued.

What survived every shift is the hand-weighted physics score: worst drop 0.015
across a new material and a 2x camera-distance change in both directions. That
is the project's central thesis, now tested three ways with intervals.

Separate, post-hoc observation, not the prediction: logistic regression on the
12 features is never significantly WORSE than the physics score in absolute
terms on any of the three shifts (steel +0.018 [+0.000, +0.036], x2.0 +0.024
[+0.013, +0.035], x0.5 +0.007 [-0.004, +0.019]) while being +0.048 better
in-distribution. Its own drop is larger only because it starts higher. That
makes it a candidate hybrid -- physics supplies the features, a linear model
tunes their combination -- but it was noticed, not predicted, and needs real
data before it is a claim.

Rung: full simulation. These are candidate-level AUCs and must not be set
beside the pair-level ablation AUC in summary.txt as though comparable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermalleak.learn import (  # noqa: E402
    ALL_FEATURES, PHYSICS_DIRECTION, SCORE_FEATURES, build_simulated,
    fit_and_evaluate, make_constrained_models, make_models, split_seeds,
)
from thermalleak.physics import MATERIALS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-seeds", type=int, default=600)
    ap.add_argument("--seed-base", type=int, default=100000)
    ap.add_argument("--material", default="abs_casing")
    ap.add_argument("--transfer-materials", nargs="+", default=["steel_pipe"])
    ap.add_argument("--n-transfer", type=int, default=1000,
                    help="transfer pairs. 200 (run_learning.py) yields only ~30 positive "
                         "candidates on steel_pipe, too few to resolve the AUC gaps at issue")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--transfer-fields", nargs="*", type=float, default=[],
                    help="extra transfer sets: the TRAINING material imaged with a "
                         "different field of view in metres (default scene 0.32), i.e. "
                         "the camera moved closer or further. Tests scale, not material")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    emit("PHYSICS-CONSTRAINED LEARNER  --  rung: full simulation (candidate-level AUC)")
    emit(f"monotone directions: {PHYSICS_DIRECTION}; geometry features unconstrained")
    emit()

    seeds = np.arange(args.seed_base, args.seed_base + args.n_seeds)
    emit(f"simulating {args.n_seeds} pairs of {args.material} ...")
    sim = build_simulated(seeds, material=args.material, progress=True)
    emit(f"  {sim.summary()}")

    transfers = {}
    for i, mat in enumerate(args.transfer_materials):
        # Same offset run_learning.py uses for its transfer set, so the
        # steel_pipe set is identical; further materials step on from there.
        t_seeds = np.arange(args.seed_base + 500000 + i * 100000,
                            args.seed_base + 500000 + i * 100000 + args.n_transfer)
        emit(f"simulating {args.n_transfer} pairs of {mat} (never trained on) ...")
        ts = build_simulated(t_seeds, material=mat)
        emit(f"  {ts.summary()}")
        n_pos, n_neg = int(ts.labels.sum()), int((ts.labels == 0).sum())
        if n_pos < 20 or n_neg < 20:
            # Not a failure of the method: a material whose thermal length is
            # comparable to the plate spreads heat across the whole frame, so
            # there is nothing local to detect at device scale. Reporting an
            # AUC on a handful of candidates would be noise dressed as a result.
            if len(ts) < 10:
                why = (f"thermal length {MATERIALS[mat].thermal_length()*1000:.0f} mm is "
                       f"comparable to the 320 mm plate, so leaks do not localise at "
                       f"device scale")
            else:
                why = "too few for a stable AUC -- raise --n-transfer"
            emit(f"  SKIPPED: {n_pos} positives / {n_neg} negatives; {why}.")
            continue
        transfers[f"transfer ({mat})"] = ts

    for j, field_m in enumerate(args.transfer_fields):
        t_seeds = np.arange(args.seed_base + 900000 + j * 100000,
                            args.seed_base + 900000 + j * 100000 + args.n_transfer)
        zoom = 0.32 / field_m
        emit(f"simulating {args.n_transfer} pairs of {args.material} with a {field_m:.2f} m "
             f"field ({zoom:.1f}x the default magnification) ...")
        ts = build_simulated(t_seeds, material=args.material, extent_m=field_m)
        emit(f"  {ts.summary()}")
        if int(ts.labels.sum()) < 20 or int((ts.labels == 0).sum()) < 20:
            emit("  SKIPPED: too few candidates for a stable AUC.")
            continue
        transfers[f"camera scale x{zoom:.1f}"] = ts
    emit()

    record: dict = {"rung": "full simulation", "level": "candidate",
                    "directions": PHYSICS_DIRECTION, "feature_sets": {}}

    for label, features in (("four physics scores", SCORE_FEATURES),
                            ("all features (scores + geometry)", ALL_FEATURES)):
        emit("=" * 92)
        emit(f"FEATURE SET: {label} ({len(features)} features)")
        emit("=" * 92)

        runs = []
        transfer_scores: dict = {}     # transfer set -> model -> [scores per split]
        for k in range(args.n_splits):
            tr_s, va_s, te_s = split_seeds(seeds, seed=k)
            tr, va, te = (sim.select_groups(s) for s in (tr_s, va_s, te_s))
            models = {**make_models(k), **make_constrained_models(features, k)}
            out = fit_and_evaluate(tr, va, {"test (held-out seeds)": te, **transfers},
                                   features=features, target_fpr=args.target_fpr,
                                   random_state=k, models=models)
            runs.append(out["results"])
            for tname, tset in transfers.items():
                X = tset.subset(features)
                for mname, m in out["models"].items():
                    transfer_scores.setdefault(tname, {}).setdefault(mname, []).append(
                        m.predict_proba(X)[:, 1])

        names = list(runs[0])
        splits = ["test (held-out seeds)", *transfers]

        def stat(name, split, metric="auc"):
            v = np.array([r[name][split][metric] for r in runs], dtype=float)
            return float(np.nanmean(v)), float(np.nanstd(v))

        # Paired in-distribution comparison. Each split has its own test set, so
        # the difference is taken within a split and its spread over splits is
        # the uncertainty -- the same pairing logic as the bootstrap below.
        paired_in = {}
        for a in ("logistic_regression", "gradient_boosting", "hist_gb (physics-monotone)"):
            d = np.array([r[a]["test (held-out seeds)"]["auc"]
                          - r["physics_score (hand-weighted)"]["test (held-out seeds)"]["auc"]
                          for r in runs])
            paired_in[a] = (float(d.mean()), float(d.std()), int((d > 0).sum()))

        hdr = f"{'scorer':<30}" + "".join(f"{s[:22]:>24}" for s in splits)
        if transfers:
            hdr += f"{'transfer gap':>14}"
        emit(hdr)
        emit("-" * len(hdr))

        fs_rec = {}
        for name in names:
            cells, rec = [], {}
            for s in splits:
                m, sd = stat(name, s)
                cells.append(f"{m:.3f} +- {sd:.3f}")
                rec[s] = dict(auc_mean=m, auc_std=sd,
                              tpr_mean=stat(name, s, "tpr")[0],
                              fpr_mean=stat(name, s, "fpr")[0])
            row = f"{name:<30}" + "".join(f"{c:>24}" for c in cells)
            if transfers:
                # In-distribution minus the worst transfer: how much is lost
                # moving to a material the model never saw. Smaller is better.
                worst = min(rec[s]["auc_mean"] for s in transfers)
                gap = rec["test (held-out seeds)"]["auc_mean"] - worst
                row += f"{gap:>+14.3f}"
                rec["transfer_gap"] = gap
            emit(row)
            fs_rec[name] = rec
        emit()
        emit("in-distribution, paired against physics within each split (mean +- sd over splits):")
        for a, (m, sd, wins) in paired_in.items():
            emit(f"  {a + ' - physics':<44}{m:>+8.3f} +- {sd:.3f}   better in {wins}/{len(runs)} splits")
        emit()
        fs_rec["paired_in_distribution_vs_physics"] = {
            a: dict(diff_mean=m, diff_sd=sd, splits_better=w) for a, (m, sd, w) in paired_in.items()}

        # ---- grouped bootstrap on the transfer set -------------------------
        # Split-to-split spread says nothing about how noisy the *fixed*
        # transfer set is. Average each model's scores over the five splits,
        # then resample transfer PAIRS (not candidates -- siblings share a
        # scene) and recompute the AUC of every scorer on each resample. The
        # paired differences are the comparisons the paper actually makes.
        from sklearn.metrics import roc_auc_score
        rng = np.random.default_rng(0)
        for tname, tset in transfers.items():
            scores = {m: np.mean(v, axis=0) for m, v in transfer_scores[tname].items()}
            scores["physics_score (hand-weighted)"] = tset.physics_score
            y, g = tset.labels, tset.groups
            uniq = np.unique(g)
            idx_by_g = {u: np.flatnonzero(g == u) for u in uniq}
            boots = {m: [] for m in scores}
            for _ in range(args.n_boot):
                pick = np.concatenate([idx_by_g[u] for u in rng.choice(uniq, size=uniq.size)])
                if len(np.unique(y[pick])) < 2:
                    continue
                for m, sc in scores.items():
                    boots[m].append(roc_auc_score(y[pick], sc[pick]))
            boots = {m: np.asarray(v) for m, v in boots.items()}
            emit(f"{tname}: {int(y.sum())} positive / {int((y == 0).sum())} negative candidates "
                 f"from {uniq.size} pairs; {args.n_boot} grouped bootstrap resamples")
            emit(f"  {'ensemble AUC (5 splits averaged)':<44}{'AUC':>8}{'95% CI':>20}")
            for m in scores:
                lo, hi = np.percentile(boots[m], [2.5, 97.5])
                emit(f"  {m:<44}{roc_auc_score(y, scores[m]):>8.3f}{f'[{lo:.3f}, {hi:.3f}]':>20}")
            emit(f"  {'paired difference':<44}{'diff':>8}{'95% CI':>20}{'P(diff>0)':>11}")
            pairs_ = [("hist_gb (physics-monotone)", "hist_gb (unconstrained)"),
                      ("hist_gb (physics-monotone)", "physics_score (hand-weighted)"),
                      ("gradient_boosting", "physics_score (hand-weighted)"),
                      ("hist_gb (unconstrained)", "gradient_boosting"),
                      ("logistic_regression", "physics_score (hand-weighted)"),
                      ("logistic_regression", "gradient_boosting")]
            diffs = {}
            for a, b in pairs_:
                d = boots[a] - boots[b]
                lo, hi = np.percentile(d, [2.5, 97.5])
                point = roc_auc_score(y, scores[a]) - roc_auc_score(y, scores[b])
                emit(f"  {a + ' - ' + b.split(' (')[0]:<44}{point:>+8.3f}"
                     f"{f'[{lo:+.3f}, {hi:+.3f}]':>20}{(d > 0).mean():>11.2f}")
                diffs[f"{a} - {b}"] = dict(diff=point, ci95=[float(lo), float(hi)],
                                            p_positive=float((d > 0).mean()))
            fs_rec.setdefault("bootstrap", {})[tname] = dict(
                ensemble_auc={m: float(roc_auc_score(y, scores[m])) for m in scores},
                ci95={m: [float(x) for x in np.percentile(boots[m], [2.5, 97.5])] for m in scores},
                paired_differences=diffs, n_pos=int(y.sum()), n_neg=int((y == 0).sum()),
                n_pairs=int(uniq.size))
            emit()
        record["feature_sets"][label] = fs_rec

    # ---- verdict, against the prediction written in the docstring ----------
    fs = record["feature_sets"]["four physics scores"]
    if transfers:
        t0 = next(iter(transfers))
        mono, ctrl, gbc, phys = (fs["hist_gb (physics-monotone)"], fs["hist_gb (unconstrained)"],
                                 fs["gradient_boosting"], fs["physics_score (hand-weighted)"])
        emit("VERDICT (four physics scores, primary transfer material)")
        emit(f"  in-distribution:  monotone {mono['test (held-out seeds)']['auc_mean']:.3f}"
             f"   control {ctrl['test (held-out seeds)']['auc_mean']:.3f}"
             f"   original GB {gbc['test (held-out seeds)']['auc_mean']:.3f}"
             f"   physics {phys['test (held-out seeds)']['auc_mean']:.3f}")
        emit(f"  transfer:         monotone {mono[t0]['auc_mean']:.3f}"
             f"   control {ctrl[t0]['auc_mean']:.3f}"
             f"   original GB {gbc[t0]['auc_mean']:.3f}"
             f"   physics {phys[t0]['auc_mean']:.3f}")
        # The bootstrap above, not the split-to-split spread, is the yardstick:
        # the transfer set is fixed, so its own sampling noise is what decides
        # whether a difference is real.
        b = fs.get("bootstrap", {}).get(t0, {}).get("paired_differences", {})
        for key, text in (("hist_gb (physics-monotone) - hist_gb (unconstrained)",
                           "constraint effect (monotone - control)"),
                          ("hist_gb (physics-monotone) - physics_score (hand-weighted)",
                           "monotone learner vs hand-weighted physics"),
                          ("gradient_boosting - physics_score (hand-weighted)",
                           "original GB vs hand-weighted physics")):
            if key in b:
                d = b[key]
                real = "real" if (d["ci95"][0] > 0 or d["ci95"][1] < 0) else "NOT distinguishable from zero"
                emit(f"  {text:<44}{d['diff']:+.3f}  95% CI [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]  -> {real}")
        record["verdict"] = b

    if args.outdir:
        d = Path(args.outdir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "constrained_learning.txt").write_text("\n".join(lines) + "\n")
        (d / "constrained_learning.json").write_text(json.dumps(record, indent=2))
        print(f"wrote {d/'constrained_learning.txt'} and {d/'constrained_learning.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
