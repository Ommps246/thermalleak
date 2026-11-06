"""
Does a better detector find fainter leaks?

The question behind the project title. "Quantum thermography" means, concretely,
quantum detectors: photon-counting devices whose noise floor is Poisson shot
noise rather than the thermal fluctuation of a bolometer membrane. They are
unambiguously better *sensors*. This script asks whether that reaches the
*output* -- whether the leak detection limit tracks NETD, or whether something
else is in the way.

Two outcomes, pointing opposite ways:

  photon-limited    the limit falls with NETD; a quantum detector buys real
                    sensitivity and the amount is quantified here.
  nuisance-limited  the limit is flat against NETD; drift, misregistration and
                    workload mismatch set the floor, and no detector upgrade
                    helps until inspection procedure improves.

If the answer is nuisance-limited, a second stage removes the nuisances one at
a time with an ideal detector, to find which one actually is the floor --
rather than inferring it, which is the mistake this project has made before.

Every limit here is on the **full-simulation** rung of the realism ladder in
CLAUDE.md. Do not quote it beside injection-rung figures as though comparable.

    python3 scripts/run_detector_sweep.py --n-pairs 500 --outdir results

About 12 minutes. Writes results/detector_sweep.json and detector_sweep.txt.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermalleak import camera as cam                       # noqa: E402
from thermalleak import evaluate as ev                      # noqa: E402
from thermalleak.detector import DETECTORS                  # noqa: E402
from thermalleak.physics import MATERIALS                   # noqa: E402
from thermalleak.simulate import generate_dataset           # noqa: E402

RUNG = "full simulation"


def fifty_percent_limit(powers, rates, counts) -> float:
    """
    Leak power at which half of leaks are detected and correctly localised,
    interpolated in log-power -- the run_evaluation.py convention.

    Returns nan in both degenerate cases, and the second matters: when every
    bin is already above 0.5 the limit lies *below* the sampled range. An
    earlier version returned the lowest bin there, which reads like a measured
    limit but is only the floor of the sampling grid. That bug made the nuisance
    ladder look far less dramatic than it is.
    """
    ok = counts > 0
    powers, rates = np.asarray(powers)[ok], np.asarray(rates)[ok]
    if len(powers) < 2 or rates.max() < 0.5 or rates.min() >= 0.5:
        return float("nan")
    for i in range(1, len(powers)):
        if rates[i] >= 0.5 >= rates[i - 1]:
            span = rates[i] - rates[i - 1]
            if span <= 0:
                return float(powers[i])
            frac = (0.5 - rates[i - 1]) / span
            lo, hi = np.log10(powers[i - 1]), np.log10(powers[i])
            return float(10 ** (lo + frac * (hi - lo)))
    return float("nan")


def bin_limit(powers, rates, counts) -> float:
    """
    The convention run_evaluation.py uses: the lowest bin centre whose detection
    rate reaches 0.5. Coarse -- it can only land on a bin centre -- but it is the
    convention summary.txt and the realism ladder quote, so it is the column
    that must agree with them. The interpolated estimate is finer and is what
    the transfer exponent is computed from.
    """
    ok = [c for c, r, n in zip(powers, rates, counts) if n > 0 and r >= 0.5]
    return float(min(ok)) if ok else float("nan")


def photon_bound(netd_k: float, material, metres_per_pixel: float, z: float = 5.0) -> float:
    """
    Order-of-magnitude lower bound on detectable power from detector noise alone.

    Matched filtering at the thermal length averages ~pi (L/p)^2 pixels; with
    dT = P/(2 pi k d) K0(r/L) and k d = 2 h L^2 this gives

        P_min ~ 4 sqrt(pi) * z * NETD * h * L * p

    Linear in NETD, with no fitted constants. It ignores the K0 core and the
    finite source, so treat the prefactor as order-unity. Its use is as a
    reference: the gap between it and the measured limit is how far above the
    photon floor the pipeline is operating.
    """
    return (4.0 * np.sqrt(np.pi) * z * netd_k
            * material.convection * material.thermal_length() * metres_per_pixel)


def measure(cfg, material, n_pairs, seed, power_range=(0.01, 0.5), **pair_kw):
    pairs = generate_dataset(n_pairs, seed=seed, cfg=cfg, material=material,
                             leak_power_range=power_range, **pair_kw)
    outs = [ev.evaluate_pair(p, use_physics=True) for p in pairs]
    _, _, _, auc = ev.roc_curve(outs)
    thr = ev.threshold_at_fpr(outs, 0.05)
    p_, r_, c_ = ev.detection_rate_by_power(outs, thr)
    recall = float(np.mean([o.score >= thr and o.localised for o in outs if o.has_leak]))
    return dict(auc=float(auc), recall=recall, limit=fifty_percent_limit(p_, r_, c_),
                limit_bin=bin_limit(p_, r_, c_), mpp=float(pairs[0].metres_per_pixel))


def fmt_limit(w: float, floor_w: float) -> str:
    return f"{w * 1000:7.2f} mW" if np.isfinite(w) else f"< {floor_w * 1000:4.1f} mW"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-pairs", type=int, default=500)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--material", default="abs_casing")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--detectors", nargs="*", default=list(DETECTORS))
    ap.add_argument("--skip-isolation", action="store_true")
    args = ap.parse_args()

    material = MATERIALS[args.material]
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say(f"DETECTOR SWEEP  --  rung: {RUNG}")
    say(f"material {args.material}: thermal length {material.thermal_length()*1000:.1f} mm, "
        f"h = {material.convection} W/m^2K; {args.n_pairs} pairs per row, seed {args.seed}")
    say("Identical scenes, leaks, drift and misalignment in every row; only the noise draw changes.")
    say("The camera is built exactly as run_evaluation.py builds it (seed=None, so the same")
    say("fixed-pattern map), which makes the microbolometer row a reproduction of summary.txt.")
    say()
    hdr = (f"{'detector':16s} {'kind':8s} {'NETD':>9s} {'AUC':>7s} {'recall':>7s} "
           f"{'50% (bin)':>10s} {'50% (interp)':>13s} {'photon bound':>13s}")
    say(hdr)
    say("-" * len(hdr))

    rows = []
    for key in args.detectors:
        det = DETECTORS[key]
        netd = det.netd_kelvin(300.0)
        t0 = time.time()
        m = measure(cam.CameraConfig(detector=det), material, args.n_pairs, args.seed)
        bound = photon_bound(netd, material, m["mpp"])
        say(f"{key:16s} {det.kind:8s} {netd*1000:6.2f} mK {m['auc']:7.3f} {m['recall']:7.3f} "
            f"{m['limit_bin']:8.3f} W {fmt_limit(m['limit'], 0.01):>13s} "
            f"{bound*1000:10.3f} mW   ({time.time()-t0:.0f}s)")
        rows.append(dict(detector=key, name=det.name, kind=det.kind, netd_mk=netd * 1000,
                         auc=m["auc"], recall_at_5pct_fpr=m["recall"],
                         limit_bin_w=m["limit_bin"], limit_w=m["limit"],
                         photon_bound_w=bound))

    valid = [r for r in rows if np.isfinite(r["limit_w"])]
    verdict = "inconclusive"
    if len(valid) >= 2:
        netd_span = max(r["netd_mk"] for r in valid) / min(r["netd_mk"] for r in valid)
        lim_span = max(r["limit_w"] for r in valid) / min(r["limit_w"] for r in valid)
        expo = np.log(lim_span) / np.log(netd_span) if netd_span > 1 else 0.0
        verdict = ("photon-limited" if expo > 0.5 else
                   "nuisance-limited" if expo < 0.2 else "partly photon-limited")
        say()
        say(f"NETD spans {netd_span:.0f}x across these detectors; the detection limit spans {lim_span:.2f}x.")
        say(f"transfer exponent {expo:.2f}  ->  {verdict.upper()}")

    ladder = []
    if not args.skip_isolation and verdict != "photon-limited":
        weak = (5e-4, 0.5)
        say()
        say("ISOLATING THE FLOOR -- ideal (BLIP) detector, nuisances removed one at a time")
        say(f"(leak power sampled down to {weak[0]*1000:.1f} mW so a low floor stays measurable,")
        say(" which makes this a different dataset from the table above; limits interpolated)")
        say(f"{'configuration':32s} {'50% (interp)':>12s} {'recall':>7s}")
        say("-" * 52)
        stages = [
            ("all nuisances present", {}, {}),
            ("fixed-pattern drift removed", dict(fpn_drift=0.0), {}),
            ("+ misalignment removed", dict(fpn_drift=0.0), dict(add_misalignment=False)),
            ("+ reflections removed", dict(fpn_drift=0.0),
             dict(add_misalignment=False, add_reflection=False)),
            ("+ workload drift removed", dict(fpn_drift=0.0),
             dict(add_misalignment=False, add_reflection=False, workload_drift=False)),
        ]
        for label, cfg_kw, pair_kw in stages:
            m = measure(cam.CameraConfig(detector=DETECTORS["blip"], **cfg_kw),
                        material, args.n_pairs, args.seed, power_range=weak, **pair_kw)
            say(f"{label:32s} {fmt_limit(m['limit'], weak[0]):>12s} {m['recall']:7.3f}")
            ladder.append(dict(configuration=label, limit_w=m["limit"], recall=m["recall"],
                               below_sampled_range=not np.isfinite(m["limit"])))

    (out / "detector_sweep.txt").write_text("\n".join(lines) + "\n")
    def clean(x):
        # nan -> null, so the file is strict JSON; a null limit means
        # "below the sampled range", which below_sampled_range spells out.
        if isinstance(x, float) and not np.isfinite(x):
            return None
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, list):
            return [clean(v) for v in x]
        return x

    with open(out / "detector_sweep.json", "w") as fh:
        json.dump(clean(dict(rung=RUNG, material=args.material, n_pairs=args.n_pairs,
                             seed=args.seed, verdict=verdict, rows=rows,
                             isolation_ladder=ladder)), fh, indent=2)
    print(f"\nwrote {out/'detector_sweep.txt'} and {out/'detector_sweep.json'}")


if __name__ == "__main__":
    main()
