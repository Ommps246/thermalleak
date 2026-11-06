#!/usr/bin/env python3
"""
Two checks a reviewer will ask for, both on the full-simulation rung.

    python3 scripts/run_robustness_heldout.py robustness --outdir results
    python3 scripts/run_robustness_heldout.py heldout    --outdir results

robustness
    The thermal length L = sqrt(k d / 2h) is the one number the physics tests
    take from the user. In practice k and h are known only roughly. Here the
    data are generated with the true abs_casing values and the detector is
    *told* a wrong k or h (scaled by 0.5 ... 1.5, one at a time, plus the two
    corners that move L furthest). The dataset is identical in every row; only
    the material handed to the detector changes. LENGTH_CALIBRATION is left at
    1.7 -- nothing is re-tuned for the wrong material.

heldout
    The test weights (0.30/0.45/0.05/0.20) and the 0.45 cut were chosen by hand
    on simulated abs_casing. Two questions: (a) do they work unchanged on a
    material and a dataset they were never set on; (b) does re-tuning the
    weights on one dataset help or overfit when scored once on another? The
    four sub-scores of every candidate are cached, so any weight vector can be
    re-scored without re-running the pipeline. Weights are searched on a
    simplex grid (step 0.05) maximising pair-level AUC on the TUNING set only;
    the held-out sets are scored once.

    tune: abs_casing seed 101      held-out: abs_casing seed 201 (fresh scenes),
                                              steel_pipe seed 301 (new material)
    and the mirror: tune on steel_pipe seed 301, score abs_casing seed 201.

Rung: full simulation. Nothing here is real-camera evidence.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermalleak import evaluate as ev                     # noqa: E402
from thermalleak.physics import MATERIALS                  # noqa: E402
from thermalleak.pipeline import detect_leak               # noqa: E402
from thermalleak.simulate import generate_dataset          # noqa: E402

DEFAULT_W = np.array([0.30, 0.45, 0.05, 0.20])
TOL_PX = 8.0


# ---------------------------------------------------------------- metrics
def auc_of(o):
    return ev.roc_curve(o)[3]


def recall_at(o, fpr=0.05):
    thr = ev.threshold_at_fpr(o, fpr)
    pos = [x for x in o if x.has_leak]
    return float(np.mean([x.score >= thr and x.localised for x in pos])) if pos else float("nan")


def limit_interp(o, bins, fpr=0.05):
    thr = ev.threshold_at_fpr(o, fpr)
    c, r, n = ev.detection_rate_by_power(o, thr, bins=bins)
    ok = n > 0
    c, r = c[ok], r[ok]
    if len(c) >= 2 and r.max() >= 0.5 and r.min() < 0.5:
        for i in range(1, len(c)):
            if r[i] >= 0.5 >= r[i - 1]:
                span = r[i] - r[i - 1]
                frac = 0.0 if span <= 0 else (0.5 - r[i - 1]) / span
                lo, hi = np.log10(c[i - 1]), np.log10(c[i])
                return float(10 ** (lo + frac * (hi - lo)))
    return float("nan")


def boot(o_list, fn, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(o_list)
    v = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v.append(fn([o_list[i] for i in idx]))
    v = np.asarray([x for x in v if np.isfinite(x)])
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v.size else (float("nan"),) * 2


# ---------------------------------------------------------------- robustness
def run_robustness(args, say):
    base = MATERIALS["abs_casing"]
    configs = [("true values", 1.0, 1.0)]
    for f in (0.5, 0.8, 0.9, 1.1, 1.2, 1.5):
        configs.append((f"k x{f}", f, 1.0))
    for f in (0.5, 0.8, 0.9, 1.1, 1.2, 1.5):
        configs.append((f"h x{f}", 1.0, f))
    configs.append(("k x1.5, h x0.5 (L biggest)", 1.5, 0.5))
    configs.append(("k x0.5, h x1.5 (L smallest)", 0.5, 1.5))

    out = {}
    for seed in args.seeds:
        pairs = generate_dataset(args.n_pairs, seed=seed)
        bins = None
        say(f"\nseed {seed}: {args.n_pairs} pairs, abs_casing true L = "
            f"{base.thermal_length() * 1000:.1f} mm")
        say(f"{'material given to the detector':32s} {'L err':>7s} {'AUC':>7s} {'recall':>7s} "
            f"{'50% (mW)':>9s}")
        rows = []
        for name, fk, fh in configs:
            t0 = time.time()
            m = dataclasses.replace(base, conductivity=base.conductivity * fk,
                                    convection=base.convection * fh)
            outc = [ev.evaluate_pair(dataclasses.replace(p, material=m), use_physics=True)
                    for p in pairs]
            lerr = np.sqrt(fk / fh)
            auc, rec, lim = auc_of(outc), recall_at(outc), limit_interp(outc, bins)
            rows.append(dict(name=name, k_scale=fk, h_scale=fh, L_ratio=float(lerr),
                             auc=auc, recall=rec, limit_mw=lim * 1000 if np.isfinite(lim) else None))
            say(f"{name:32s} {lerr:7.2f} {auc:7.3f} {rec:7.3f} "
                f"{(lim * 1000 if np.isfinite(lim) else float('nan')):9.1f}   ({time.time() - t0:.0f}s)")
        out[seed] = rows
    return out


# ---------------------------------------------------------------- held-out
def cache_dataset(material_name, n_pairs, seed, power_scale=1.0):
    """Run the pipeline once per pair; keep per-candidate sub-scores.

    power_scale multiplies the leak-power range (default 0.01-0.5 W). A leak of
    the same wattage is far fainter on a thicker, better-conducting plate
    (dT scales as 1/(k d)), so a fair material-transfer test scales the range by
    the ratio of k*d to the abs_casing value rather than leaving most leaks
    below anything a camera could see.
    """
    mat = MATERIALS[material_name]
    rng_range = (0.01 * power_scale, 0.5 * power_scale)
    pairs = generate_dataset(n_pairs, seed=seed, material=mat, leak_power_range=rng_range)
    diff = [ev.evaluate_pair(p, use_physics=False) for p in pairs]
    cached = []
    for p in pairs:
        res = detect_leak(p.baseline, p.inspection, material=p.material,
                          metres_per_pixel=p.metres_per_pixel, use_physics=True)
        cands = []
        for f in res.findings:
            v = f.verdict
            d = (float(np.hypot(f.centroid[0] - p.leak_centre_px[0], f.centroid[1] - p.leak_centre_px[1]))
                 if (p.leak_centre_px is not None) else float("inf"))
            sig = f.candidate.z_peak / (f.candidate.z_peak + 3.5)
            cands.append((sig, v.fit_score, v.length_score, v.sharpness_score, v.novelty_score, d))
        cached.append((p.has_leak, p.leak_power_watts, np.array(cands, dtype=float).reshape(-1, 6)))
    return cached, diff


def outcomes_for(cached, w):
    logw = np.asarray(w, dtype=float)
    out = []
    for has, pw, c in cached:
        if len(c) == 0:
            out.append(ev.PairOutcome(has, pw, 0.0, False, float("inf"), float("nan"), float("nan")))
            continue
        parts = np.clip(c[:, 1:5], 1e-6, 1.0)
        phys = np.exp(np.log(parts) @ logw)
        comb = c[:, 0] * phys
        i = int(np.argmax(comb))
        out.append(ev.PairOutcome(has, pw, float(comb[i]), bool(has and c[i, 5] <= TOL_PX),
                                  float(c[i, 5]), float(phys[i]), float("nan")))
    return out


def candidate_cut_stats(cached, w, cut=0.45):
    """Share of true-leak candidates kept and of artefact candidates rejected at `cut`."""
    kl = tl = ra = ta = 0
    for has, pw, c in cached:
        if len(c) == 0:
            continue
        parts = np.clip(c[:, 1:5], 1e-6, 1.0)
        phys = np.exp(np.log(parts) @ np.asarray(w))
        is_leak = has & (c[:, 5] <= TOL_PX)
        kl += int(np.sum(is_leak & (phys >= cut))); tl += int(np.sum(is_leak))
        ra += int(np.sum(~is_leak & (phys < cut))); ta += int(np.sum(~is_leak))
    return (kl / tl if tl else float("nan")), (ra / ta if ta else float("nan")), tl, ta


def simplex(step=0.05):
    n = int(round(1 / step))
    for a in range(n + 1):
        for b in range(n + 1 - a):
            for c in range(n + 1 - a - b):
                yield np.array([a, b, c, n - a - b - c]) * step


def run_heldout(args, say):
    ref = MATERIALS["abs_casing"]
    ratio = lambda m: MATERIALS[m].conductivity * MATERIALS[m].thickness / (ref.conductivity * ref.thickness)
    sets = {
        "abs_casing seed 101 (tuning)": ("abs_casing", 101, 1.0),
        "abs_casing seed 201": ("abs_casing", 201, 1.0),
        "steel_pipe seed 301": ("steel_pipe", 301, ratio("steel_pipe")),
    }
    say(f"steel_pipe leak-power range scaled by k*d ratio = {ratio('steel_pipe'):.1f}x "
        f"({0.01 * ratio('steel_pipe'):.2f}-{0.5 * ratio('steel_pipe'):.1f} W)")
    cache, diffcache = {}, {}
    for name, (mat, seed, sc) in sets.items():
        t0 = time.time()
        cache[name], diffcache[name] = cache_dataset(mat, args.n_pairs, seed, sc)
        say(f"cached {name}: {len(cache[name])} pairs ({time.time() - t0:.0f}s)")

    grid = list(simplex(0.05))

    def tune(tune_name):
        best, best_auc = None, -1
        for w in grid:
            a = auc_of(outcomes_for(cache[tune_name], w))
            if a > best_auc:
                best, best_auc = w, a
        return best, best_auc

    results = {}
    for tune_name, test_names in [
        ("abs_casing seed 101 (tuning)", ["abs_casing seed 201", "steel_pipe seed 301"]),
        ("steel_pipe seed 301", ["abs_casing seed 201"]),
    ]:
        w_best, a_best = tune(tune_name)
        say(f"\nTUNE on {tune_name}: best weights (profile, length, sharpness, novelty) = "
            f"{np.round(w_best, 2).tolist()}  tuning AUC {a_best:.3f}  "
            f"(shipped weights on the same set: {auc_of(outcomes_for(cache[tune_name], DEFAULT_W)):.3f})")
        results[tune_name] = dict(weights=w_best.tolist(), tuning_auc=a_best, tests={})
        say(f"{'scored once on':32s} {'shipped AUC [95% CI]':>26s} {'re-tuned AUC [95% CI]':>26s} "
            f"{'kept/rejected @0.45 (shipped)':>32s}")
        for tn in test_names:
            o_def = outcomes_for(cache[tn], DEFAULT_W)
            o_tun = outcomes_for(cache[tn], w_best)
            a_d, a_t = auc_of(o_def), auc_of(o_tun)
            ci_d, ci_t = boot(o_def, auc_of), boot(o_tun, auc_of)
            kept, rej, nl, na = candidate_cut_stats(cache[tn], DEFAULT_W)
            say(f"{tn:32s} {a_d:7.3f} [{ci_d[0]:.3f}, {ci_d[1]:.3f}]   {a_t:7.3f} [{ci_t[0]:.3f}, {ci_t[1]:.3f}]"
                f"   {kept * 100:4.0f}% of {nl} leaks / {rej * 100:4.0f}% of {na} artefacts")
            results[tune_name]["tests"][tn] = dict(
                shipped_auc=a_d, shipped_ci=ci_d, tuned_auc=a_t, tuned_ci=ci_t,
                recall_shipped=recall_at(o_def), recall_tuned=recall_at(o_tun),
                cut_kept=kept, cut_rejected=rej, n_leak_cands=nl, n_art_cands=na)
    # differencing-free reference: shipped weights on every set, for the record
    say("\nshipped weights vs differencing only, on each set (AUC / recall at 5% FPR):")
    for n, c in cache.items():
        o = outcomes_for(c, DEFAULT_W)
        d = diffcache[n]
        say(f"  {n:32s} physics {auc_of(o):.3f} / {recall_at(o):.3f}    differencing {auc_of(d):.3f} / {recall_at(d):.3f}")
        results.setdefault("reference", {})[n] = dict(physics_auc=auc_of(o), physics_recall=recall_at(o),
                                                       diff_auc=auc_of(d), diff_recall=recall_at(d))
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["robustness", "heldout"])
    ap.add_argument("--n-pairs", type=int, default=500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[11])
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say(f"{args.mode.upper()}  --  rung: full simulation")
    res = run_robustness(args, say) if args.mode == "robustness" else run_heldout(args, say)
    if args.outdir:
        d = Path(args.outdir)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{args.mode}.txt").write_text("\n".join(lines) + "\n")
        (d / f"{args.mode}.json").write_text(json.dumps(res, indent=1, default=float))
        print(f"\nwrote {d}/{args.mode}.txt and .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
