#!/usr/bin/env python3
"""
Run the full evaluation and the physics ablation.

    python scripts/run_evaluation.py --n-pairs 400 --outdir results

Produces:
    results/evaluation.png      ROC, sensitivity curve, score separation
    results/example_*.png       worked inspection reports
    results/summary.txt         the numbers, in text
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermalleak.evaluate import (  # noqa: E402
    detection_rate_by_power, evaluate_pair, roc_curve, summarise,
    threshold_at_fpr,
)
from thermalleak.pipeline import detect_leak  # noqa: E402
from thermalleak.simulate import generate_dataset, generate_pair  # noqa: E402
from thermalleak.viz import plot_detection, plot_evaluation  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-pairs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--target-fpr", type=float, default=0.05)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"generating {args.n_pairs} labelled pairs ...", flush=True)
    pairs = generate_dataset(args.n_pairs, seed=args.seed)
    n_pos = sum(p.has_leak for p in pairs)
    print(f"  {n_pos} with a leak, {len(pairs) - n_pos} without", flush=True)

    print("evaluating physics-informed ...", flush=True)
    out_physics = [evaluate_pair(p, use_physics=True) for p in pairs]
    print("evaluating differencing-only baseline ...", flush=True)
    out_baseline = [evaluate_pair(p, use_physics=False) for p in pairs]

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("=" * 72)
    emit("PHYSICS ABLATION")
    emit("=" * 72)
    emit(f"{'configuration':<24}{'AUC':>8}{'recall':>9}{'FPR':>8}{'loc px':>9}")
    for out, label in ((out_baseline, "differencing only"),
                       (out_physics, "physics-informed")):
        s = summarise(out, label, args.target_fpr)
        emit(f"{s['label']:<24}{s['auc']:>8.3f}{s['recall_at_fpr']:>9.3f}"
             f"{s['false_alarm_rate']:>8.3f}{s['median_localisation_px']:>9.1f}")
    emit()
    emit(f"recall is measured at a {args.target_fpr:.0%} false-alarm rate, with the")
    emit("threshold chosen on no-leak pairs only, and requires the top-ranked")
    emit("candidate to be localised on the true leak.")
    emit()

    emit("=" * 72)
    emit("SENSITIVITY BY LEAK POWER")
    emit("=" * 72)
    emit(f"{'leak power (W)':<18}{'diff-only':>12}{'physics':>12}{'n':>6}")
    thr_b = threshold_at_fpr(out_baseline, args.target_fpr)
    thr_p = threshold_at_fpr(out_physics, args.target_fpr)
    cb, rb, nb = detection_rate_by_power(out_baseline, thr_b)
    cp, rp, _ = detection_rate_by_power(out_physics, thr_p)
    for i in range(len(cb)):
        rp_i = rp[i] if i < len(rp) else float("nan")
        emit(f"{cb[i]:<18.3f}{rb[i]:>12.2f}{rp_i:>12.2f}{nb[i]:>6d}")
    emit()

    detectable = [c for c, r in zip(cp, rp) if r >= 0.5]
    if detectable:
        emit(f"physics-informed 50% detection limit: ~{min(detectable):.3f} W")
    detectable_b = [c for c, r in zip(cb, rb) if r >= 0.5]
    if detectable_b:
        emit(f"differencing-only 50% detection limit: ~{min(detectable_b):.3f} W")
    emit()

    # --- candidate-level artefact rejection ------------------------------
    emit("=" * 72)
    emit("ARTEFACT REJECTION (candidate level)")
    emit("=" * 72)
    true_scores, false_scores = [], []
    for pair in pairs[: min(120, len(pairs))]:
        result = detect_leak(pair.baseline, pair.inspection,
                             material=pair.material,
                             metres_per_pixel=pair.metres_per_pixel)
        for f in result.findings:
            if f.verdict is None:
                continue
            on_leak = False
            if pair.leak_centre_px is not None:
                d = np.hypot(f.centroid[0] - pair.leak_centre_px[0],
                             f.centroid[1] - pair.leak_centre_px[1])
                on_leak = d <= 8.0
            (true_scores if on_leak else false_scores).append(f.verdict.physics_score)

    if true_scores and false_scores:
        t = np.percentile(true_scores, [10, 50, 90])
        f_ = np.percentile(false_scores, [10, 50, 90])
        emit(f"physics score on {len(true_scores)} true leaks:  "
             f"p10 {t[0]:.2f}  median {t[1]:.2f}  p90 {t[2]:.2f}")
        emit(f"physics score on {len(false_scores)} artefacts:   "
             f"p10 {f_[0]:.2f}  median {f_[1]:.2f}  p90 {f_[2]:.2f}")
        kept = np.mean(np.array(true_scores) >= 0.45)
        killed = np.mean(np.array(false_scores) < 0.45)
        emit(f"at a 0.45 cut: {kept:.0%} of true leaks kept, "
             f"{killed:.0%} of artefacts rejected")
    emit()

    # --- figures ----------------------------------------------------------
    fig = plot_evaluation(out_physics, out_baseline,
                          str(outdir / "evaluation.png"), args.target_fpr)
    emit(f"wrote {fig}")

    for power, name in ((0.30, "strong"), (0.06, "weak"), (0.0, "none")):
        rng = np.random.default_rng(7)
        pair = generate_pair(rng, leak_power_watts=power)
        result = detect_leak(pair.baseline, pair.inspection,
                             material=pair.material,
                             metres_per_pixel=pair.metres_per_pixel)
        path = outdir / f"example_{name}.png"
        plot_detection(result, pair.baseline, pair.inspection, str(path),
                       truth_centre=pair.leak_centre_px)
        emit(f"wrote {path}")

    (outdir / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {outdir / 'summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
