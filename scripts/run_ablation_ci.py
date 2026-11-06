#!/usr/bin/env python3
"""
Error bars on the headline ablation.

    python3 scripts/run_ablation_ci.py --outdir results

`run_evaluation.py` reports the ablation on one dataset (500 pairs, seed 11):
AUC 0.705 -> 0.965, 50% detection limit 0.214 W -> 0.023 W. Those are point
estimates. A reviewer will ask two separate questions, and this script answers
both:

1. **How sure are we, on this dataset?** Pair-level bootstrap. Whole pairs are
   resampled with replacement -- never candidates, which share a scene -- and
   both arms are scored on the *same* resample, so the AUC difference is a
   paired comparison and its interval is tighter and more honest than two
   independent intervals would suggest. Recall is re-thresholded at 5% FPR on
   each resample's own negatives, exactly as `evaluate.threshold_at_fpr` does.

2. **Was seed 11 a lucky dataset?** The whole ablation repeated on independent
   datasets (different seeds -> different scenes, leaks, nuisances). The spread
   across seeds is the between-dataset variability a single seed cannot show.

The detection limit is reported two ways. The bin convention (lowest bin centre
reaching 50%) is what summary.txt quotes, but it can only land on one of seven
bin centres, so its bootstrap distribution is lumpy. The interpolated estimate
(log-power interpolation between bins, as `run_detector_sweep.py` uses) is the
one whose interval means something.

The seed-11 row reproduces summary.txt digit for digit; if it does not, the
dataset generation has drifted and nothing else here should be trusted.

Rung: full simulation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermalleak import evaluate as ev                   # noqa: E402
from thermalleak.simulate import generate_dataset        # noqa: E402


def auc_of(outcomes) -> float:
    return ev.roc_curve(outcomes)[3]


def recall_at(outcomes, target_fpr=0.05) -> float:
    thr = ev.threshold_at_fpr(outcomes, target_fpr)
    pos = [o for o in outcomes if o.has_leak]
    return float(np.mean([o.score >= thr and o.localised for o in pos])) if pos else float("nan")


def limits(outcomes, bins, target_fpr=0.05) -> tuple[float, float]:
    """(bin-convention limit, interpolated limit) in watts; nan if undefined."""
    thr = ev.threshold_at_fpr(outcomes, target_fpr)
    c, r, n = ev.detection_rate_by_power(outcomes, thr, bins=bins)
    ok = n > 0
    c, r = c[ok], r[ok]
    hit = [cc for cc, rr in zip(c, r) if rr >= 0.5]
    bin_lim = float(min(hit)) if hit else float("nan")
    interp = float("nan")
    if len(c) >= 2 and r.max() >= 0.5 and r.min() < 0.5:
        for i in range(1, len(c)):
            if r[i] >= 0.5 >= r[i - 1]:
                span = r[i] - r[i - 1]
                frac = 0.0 if span <= 0 else (0.5 - r[i - 1]) / span
                lo, hi = np.log10(c[i - 1]), np.log10(c[i])
                interp = float(10 ** (lo + frac * (hi - lo)))
                break
    return bin_lim, interp


def run_seed(n_pairs: int, seed: int):
    pairs = generate_dataset(n_pairs, seed=seed)          # exactly as run_evaluation.py
    phys = [ev.evaluate_pair(p, use_physics=True) for p in pairs]
    diff = [ev.evaluate_pair(p, use_physics=False) for p in pairs]
    return phys, diff


def ci(x, level=95):
    x = np.asarray([v for v in x if np.isfinite(v)])
    if x.size == 0:
        return float("nan"), float("nan")
    a = (100 - level) / 2
    return float(np.percentile(x, a)), float(np.percentile(x, 100 - a))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-pairs", type=int, default=500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13, 14, 15])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    lines: list[str] = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say("ERROR BARS ON THE HEADLINE ABLATION  --  rung: full simulation")
    say(f"{args.n_pairs} pairs per dataset; seeds {args.seeds}; first seed is bootstrapped "
        f"({args.n_boot} pair-level resamples, both arms on the same resample)")
    say()

    per_seed = []
    boot_rec = None
    for k, seed in enumerate(args.seeds):
        t0 = time.time()
        phys, diff = run_seed(args.n_pairs, seed)
        # Fixed bins from this dataset's powers, so bootstrap resamples are
        # binned identically to the point estimate rather than re-binned.
        powers = np.array([o.leak_power_watts for o in phys if o.has_leak])
        bins = np.geomspace(max(powers.min(), 1e-3), powers.max(), 8)

        row = dict(seed=seed,
                   auc_phys=auc_of(phys), auc_diff=auc_of(diff),
                   recall_phys=recall_at(phys), recall_diff=recall_at(diff))
        row["lim_phys_bin"], row["lim_phys_interp"] = limits(phys, bins)
        row["lim_diff_bin"], row["lim_diff_interp"] = limits(diff, bins)
        per_seed.append(row)
        say(f"seed {seed}: AUC diff-only {row['auc_diff']:.3f}  physics {row['auc_phys']:.3f}   "
            f"recall {row['recall_diff']:.3f} -> {row['recall_phys']:.3f}   "
            f"50% limit {row['lim_diff_bin']:.3f} W -> {row['lim_phys_bin']:.3f} W   "
            f"({time.time() - t0:.0f}s)")

        if k == 0:
            rng = np.random.default_rng(0)
            n = len(phys)
            B = {key: [] for key in ("auc_p", "auc_d", "auc_gain", "rec_p", "rec_d",
                                     "lim_p", "lim_d", "lim_ratio")}
            for _ in range(args.n_boot):
                idx = rng.integers(0, n, size=n)
                bp = [phys[i] for i in idx]
                bd = [diff[i] for i in idx]
                ap_, ad_ = auc_of(bp), auc_of(bd)
                B["auc_p"].append(ap_); B["auc_d"].append(ad_); B["auc_gain"].append(ap_ - ad_)
                B["rec_p"].append(recall_at(bp)); B["rec_d"].append(recall_at(bd))
                lp, ld = limits(bp, bins)[1], limits(bd, bins)[1]
                B["lim_p"].append(lp); B["lim_d"].append(ld)
                B["lim_ratio"].append(ld / lp if np.isfinite(lp) and np.isfinite(ld) and lp > 0
                                      else float("nan"))
            boot_rec = dict(seed=seed, point=row,
                            ci95={key: ci(v) for key, v in B.items()},
                            p_gain_positive=float(np.mean(np.array(B["auc_gain"]) > 0)),
                            n_boot=args.n_boot)

    say()
    say("=" * 84)
    say(f"BOOTSTRAP, seed {boot_rec['seed']} (the summary.txt dataset)")
    say("=" * 84)
    p, c = boot_rec["point"], boot_rec["ci95"]
    say(f"{'':34s}{'point':>10s}{'95% CI':>24s}")
    say(f"{'AUC, differencing only':34s}{p['auc_diff']:>10.3f}{f'[{c['auc_d'][0]:.3f}, {c['auc_d'][1]:.3f}]':>24s}")
    say(f"{'AUC, physics-informed':34s}{p['auc_phys']:>10.3f}{f'[{c['auc_p'][0]:.3f}, {c['auc_p'][1]:.3f}]':>24s}")
    say(f"{'AUC gain (paired)':34s}{p['auc_phys'] - p['auc_diff']:>+10.3f}"
        f"{f'[{c['auc_gain'][0]:+.3f}, {c['auc_gain'][1]:+.3f}]':>24s}")
    say(f"{'recall at 5% FPR, differencing':34s}{p['recall_diff']:>10.3f}{f'[{c['rec_d'][0]:.3f}, {c['rec_d'][1]:.3f}]':>24s}")
    say(f"{'recall at 5% FPR, physics':34s}{p['recall_phys']:>10.3f}{f'[{c['rec_p'][0]:.3f}, {c['rec_p'][1]:.3f}]':>24s}")
    say(f"{'50% limit, physics (interp, mW)':34s}{p['lim_phys_interp']*1000:>10.1f}"
        f"{f'[{c['lim_p'][0]*1000:.1f}, {c['lim_p'][1]*1000:.1f}]':>24s}")
    say(f"{'50% limit, differencing (interp, mW)':34s}{p['lim_diff_interp']*1000:>10.1f}"
        f"{f'[{c['lim_d'][0]*1000:.1f}, {c['lim_d'][1]*1000:.1f}]':>24s}")
    ratio = p["lim_diff_interp"] / p["lim_phys_interp"]
    say(f"{'sensitivity gain (limit ratio)':34s}{ratio:>9.1f}x{f'[{c['lim_ratio'][0]:.1f}x, {c['lim_ratio'][1]:.1f}x]':>24s}")
    say(f"AUC gain > 0 in {boot_rec['p_gain_positive']*100:.1f}% of resamples.")
    say("The bin-convention limits (summary.txt: 0.023 / 0.214 W) can only fall on a bin")
    say("centre, so their interval is not meaningful; the interpolated limit is the one to quote")
    say("with an interval.")

    say()
    say("=" * 84)
    say(f"ACROSS {len(per_seed)} INDEPENDENT DATASETS (mean +- sd, min-max)")
    say("=" * 84)

    def agg(key, scale=1.0, fmt=".3f"):
        v = np.array([r[key] for r in per_seed], dtype=float) * scale
        n_all = v.size
        v = v[np.isfinite(v)]
        if v.size == 0:
            return "undefined on every dataset (curve never crosses 50% inside the bins)"
        tail = "" if v.size == n_all else f"   [defined on {v.size}/{n_all}]"
        return f"{v.mean():{fmt}} +- {v.std(ddof=1) if v.size > 1 else 0:{fmt}}   ({v.min():{fmt}} - {v.max():{fmt}})" + tail

    say(f"{'AUC, differencing only':34s}{agg('auc_diff')}")
    say(f"{'AUC, physics-informed':34s}{agg('auc_phys')}")
    gains = np.array([r["auc_phys"] - r["auc_diff"] for r in per_seed])
    say(f"{'AUC gain':34s}{gains.mean():.3f} +- {gains.std(ddof=1):.3f}   "
        f"({gains.min():.3f} - {gains.max():.3f}); positive in {int((gains > 0).sum())}/{len(gains)}")
    say(f"{'recall, differencing':34s}{agg('recall_diff')}")
    say(f"{'recall, physics':34s}{agg('recall_phys')}")
    say(f"{'50% limit, differencing (mW)':34s}{agg('lim_diff_interp', 1000, '.1f')}")
    say(f"{'50% limit, physics (mW)':34s}{agg('lim_phys_interp', 1000, '.1f')}")
    rat = np.array([r["lim_diff_interp"] / r["lim_phys_interp"] for r in per_seed])
    rat = rat[np.isfinite(rat)]
    if rat.size > 1:
        say(f"{'sensitivity gain':34s}{rat.mean():.1f}x +- {rat.std(ddof=1):.1f}x   "
            f"({rat.min():.1f}x - {rat.max():.1f}x)   [defined on {rat.size}/{len(per_seed)}]")

    if args.outdir:
        d = Path(args.outdir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "ablation_ci.txt").write_text("\n".join(lines) + "\n")
        (d / "ablation_ci.json").write_text(json.dumps(
            dict(rung="full simulation", n_pairs=args.n_pairs, seeds=args.seeds,
                 per_seed=per_seed, bootstrap=boot_rec), indent=2,
            default=lambda x: None))
        print(f"wrote {d/'ablation_ci.txt'} and {d/'ablation_ci.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
