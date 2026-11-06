#!/usr/bin/env python3
"""
Model 1 of the learned arm: physics features into a classifier.

    python3 scripts/run_learning.py --n-seeds 600 --outdir /tmp/learn
    python3 scripts/run_learning.py --n-seeds 600 --real-frames frames/ --outdir /tmp/learn

Trains gradient boosting and logistic regression on the per-candidate features
`score.py` already computes, and compares them with the hand-weighted geometric
mean those same features feed. Nothing in the detection pipeline is touched;
this script only reads it.

What it answers
---------------
1. Does a learned combination beat the hand weighting on held-out simulation?
2. **Does the length test empirically earn its 0.45?** Importances from a model
   trained on the four test scores alone sit directly beside the hand weights.
3. How much of that survives a material the model never saw, and real frames
   nothing about which was simulated?

Splitting is by simulator seed throughout, so all candidates from one frame pair
land on the same side. Thresholds are chosen on validation and applied unchanged
to every test set, including the real one.

`--real-frames` wants a directory of .npy frames in kelvin, as written by
`scripts/convert_radiometric.py`. Without it the real-data section is skipped
and the script says so rather than quietly reporting two thirds of an answer.

Requires scikit-learn, which the detection pipeline does not.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermalleak.learn import (  # noqa: E402
    ALL_FEATURES, HAND_WEIGHTS, SCORE_FEATURES, build_from_frame_pairs,
    build_simulated, feature_importances, fit_and_evaluate, split_seeds,
)


def _row(name: str, r: dict) -> str:
    return (f"{name:<34}{r['auc']:>8.3f}{r['average_precision']:>8.3f}"
            f"{r['tpr']:>9.3f}{r['fpr']:>8.3f}{r['n_pos']:>7d}{r['n_neg']:>7d}")


HEADER = (f"{'model':<34}{'AUC':>8}{'AP':>8}{'TPR':>9}{'FPR':>8}"
          f"{'n_pos':>7}{'n_neg':>7}")


def load_real_pairs(directory: str, n_static: int, n_injected: int,
                    mpp: float, material: str, seed: int = 0):
    """
    Two real-data sets from a directory of frames.

    *static* pairs contain no leak at all, so every candidate is a false alarm
    and the only measurable quantity is the false positive rate. *injected*
    pairs put a known leak into the later frame of a genuinely different pair,
    which is the only way to get a true positive rate on real backgrounds --
    and the leak's position is exact because we put it there.
    """
    from thermalleak.inject import InjectionSpec, inject_leak

    files = sorted(glob.glob(str(Path(directory) / "*.npy")))
    if len(files) < 4:
        raise SystemExit(f"need at least 4 .npy frames in {directory}, found {len(files)}")
    frames = [np.load(f).astype(float) for f in files]
    shapes = {f.shape for f in frames}
    if len(shapes) > 1:
        raise SystemExit(f"frames have differing shapes: {sorted(shapes)}")

    rng = np.random.default_rng(seed)
    static = []
    while len(static) < n_static:
        i, j = rng.choice(len(frames), 2, replace=False)
        static.append((frames[i], frames[j]))

    injected, centres = [], []
    powers = np.geomspace(0.02, 0.4, 8)
    for k in range(n_injected):
        i, j = rng.choice(len(frames), 2, replace=False)
        pair = inject_leak(frames[j], InjectionSpec(
            power_watts=float(powers[k % len(powers)]), kind="point"),
            material=material, metres_per_pixel=mpp, seed=int(rng.integers(1 << 30)),
            baseline_field=frames[i])
        injected.append((pair.baseline, pair.inspection))
        centres.append(pair.leak_centre_px)

    return static, injected, centres


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-seeds", type=int, default=600,
                    help="simulated frame pairs, one per seed")
    ap.add_argument("--seed-base", type=int, default=100000)
    ap.add_argument("--material", default="abs_casing")
    ap.add_argument("--transfer-material", default="steel_pipe",
                    help="material held out entirely, for the transfer test")
    ap.add_argument("--n-transfer", type=int, default=200)
    ap.add_argument("--target-fpr", type=float, default=0.05,
                    help="candidate-level FPR the threshold is set to on validation")
    ap.add_argument("--n-splits", type=int, default=5,
                    help="independent train/val/test partitions OF THE SAME pairs. "
                         "The test set carries ~60 positives, so a single split "
                         "resolves AUC to about a hundredth; reporting mean and "
                         "spread over several stops a split-lucky number being "
                         "read as a real difference")
    ap.add_argument("--real-frames", default=None,
                    help="directory of real .npy frames in kelvin")
    ap.add_argument("--real-mpp", type=float, default=0.0025)
    ap.add_argument("--n-real-static", type=int, default=200)
    ap.add_argument("--n-real-injected", type=int, default=120)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    # ---- data -----------------------------------------------------------
    seeds = np.arange(args.seed_base, args.seed_base + args.n_seeds)
    emit(f"simulating {args.n_seeds} pairs of {args.material} ...")
    sim = build_simulated(seeds, material=args.material, progress=True)
    emit(f"  all: {sim.summary()}")
    emit(f"  {args.n_splits} independent splits by seed (60/20/20)")
    emit()

    transfer_seeds = np.arange(args.seed_base + 500000,
                               args.seed_base + 500000 + args.n_transfer)
    emit(f"simulating {args.n_transfer} pairs of {args.transfer_material} "
         f"(never trained on) ...")
    transfer = build_simulated(transfer_seeds, material=args.transfer_material)
    emit(f"  transfer: {transfer.summary()}")
    emit()

    fixed_tests = {f"transfer ({args.transfer_material})": transfer}

    real_static = real_injected = None
    if args.real_frames:
        emit(f"loading real frames from {args.real_frames} ...")
        static, injected, centres = load_real_pairs(
            args.real_frames, args.n_real_static, args.n_real_injected,
            args.real_mpp, args.material)
        real_static = build_from_frame_pairs(static, material=args.material,
                                             metres_per_pixel=args.real_mpp)
        real_injected = build_from_frame_pairs(injected, material=args.material,
                                               metres_per_pixel=args.real_mpp,
                                               truth_centres=centres)
        emit(f"  real static  : {real_static.summary()}")
        emit(f"  real injected: {real_injected.summary()}")
        fixed_tests["real static (all false)"] = real_static
        fixed_tests["real injected"] = real_injected
        emit()
    else:
        emit("[no --real-frames given: the real-data comparison is SKIPPED, and "
             "the simulation numbers below say nothing about it]")
        emit()

    # ---- the two feature sets -------------------------------------------
    split_names = ["validation", "test (held-out seeds)"]
    all_names = split_names + list(fixed_tests)

    for label, features in (("four physics test scores only", SCORE_FEATURES),
                            ("all features (scores + geometry)", ALL_FEATURES)):
        emit("=" * 88)
        emit(f"FEATURE SET: {label}   ({len(features)} features)")
        emit("=" * 88)

        runs = []
        importances = []
        for k in range(args.n_splits):
            tr_s, va_s, te_s = split_seeds(seeds, seed=k)
            tr, va, te = (sim.select_groups(s) for s in (tr_s, va_s, te_s))
            tests = {"test (held-out seeds)": te, **fixed_tests}
            out = fit_and_evaluate(tr, va, tests, features=features,
                                   target_fpr=args.target_fpr, random_state=k)
            runs.append(out["results"])
            importances.append({n: feature_importances(m, features, va,
                                                       random_state=k)
                                for n, m in out["models"].items()})

        model_names = list(runs[0])
        for split in all_names:
            emit(f"-- {split} " + "-" * max(4, 86 - len(split)))
            emit(f"{'model':<34}{'AUC':>16}{'TPR':>16}{'FPR':>16}")
            for name in model_names:
                cells = []
                for metric in ("auc", "tpr", "fpr"):
                    v = np.array([r[name][split][metric] for r in runs], dtype=float)
                    cells.append("nan" if np.isnan(v).all()
                                 else f"{np.nanmean(v):.3f}+-{np.nanstd(v):.3f}")
                emit(f"{name:<34}" + "".join(f"{c:>16}" for c in cells))
            emit()

        emit("FEATURE IMPORTANCE (permutation on validation, AUC drop, "
             f"mean over {args.n_splits} splits)")
        emit(f"{'feature':<24}{'grad boosting':>17}{'logistic':>17}"
             + (f"{'hand weight':>14}" if features == SCORE_FEATURES else ""))
        gb = np.array([[i["gradient_boosting"]["permutation"][j]
                        for j in range(len(features))] for i in importances])
        lr = np.array([[i["logistic_regression"]["permutation"][j]
                        for j in range(len(features))] for i in importances])
        order = np.argsort(gb.mean(axis=0))[::-1]
        for i in order:
            name = features[i]
            row = (f"{name:<24}"
                   f"{f'{gb[:, i].mean():.3f}+-{gb[:, i].std():.3f}':>17}"
                   f"{f'{lr[:, i].mean():.3f}+-{lr[:, i].std():.3f}':>17}")
            if features == SCORE_FEATURES:
                row += f"{HAND_WEIGHTS.get(name, float('nan')):>14.2f}"
            emit(row)
        emit()
        if features == SCORE_FEATURES:
            emit("Permutation importance is an AUC drop normalised to sum to 1; the")
            emit("hand weights are exponents in a geometric mean. They are different")
            emit("quantities and the columns are comparable in ORDER and rough")
            emit("magnitude, not as like-for-like numbers.")
            emit()
            emit("READ THE NOVELTY ROW WITH CARE. It scores far below its hand")
            emit("weight of 0.20, but the nuisance it was built to catch -- a")
            emit("component drawing slightly more power between captures -- is")
            emit("largely absent from the current simulator. SOURCE_JITTER in")
            emit("simulate.py was cut from +/-10% to +/-2.5% as unrealistic for a")
            emit("device held at matched load, and at +/-10% that jitter moved the")
            emit("SoC by ~0.26 W between frames, larger than most leaks being")
            emit("hunted. The README's '0.19 -> 0.78' figure for adding the novelty")
            emit("test dates from that configuration. So this low importance is")
            emit("evidence about THIS simulator, not about whether novelty earns")
            emit("its weight on real hardware, where component drift is set by the")
            emit("device. Only the capture protocol settles that.")
            emit()

    if args.outdir:
        d = Path(args.outdir); d.mkdir(parents=True, exist_ok=True)
        (d / "learning_summary.txt").write_text("\n".join(lines) + "\n")
        print(f"wrote {d / 'learning_summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
