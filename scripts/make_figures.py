#!/usr/bin/env python3
"""
Paper figures from the recorded results -- no simulation is re-run here.

    python3 scripts/make_figures.py            # reads results/*.json, writes results/fig_*.png

Each figure is drawn only from a JSON file that a script in scripts/ wrote, so
a figure can never disagree with the text file next to it. A missing JSON just
skips that figure.

  fig_ablation_ci.png        headline ablation across 5 datasets, with the
                             seed-11 bootstrap interval   (run_ablation_ci.py)
  fig_detector_limit.png     the quantum figure: 50% detection limit vs
                             detector NETD, against the sensor-noise-only
                             bound                         (run_detector_sweep.py)
  fig_nuisance_ladder.png    what actually sets the limit (run_detector_sweep.py)
  fig_transfer_stability.png the ML figure: in-distribution AUC vs worst AUC
                             under shift, per scorer      (run_constrained_learning.py)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

# Okabe-Ito, colour-blind safe. Physics is always the same blue on every figure.
PHYS = "#0072B2"
DIFF = "#999999"
LEARN = "#D55E00"
MONO = "#E69F00"
LIN = "#009E73"
INK = "#222222"

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#555555", "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "legend.frameon": False, "savefig.dpi": 220, "savefig.bbox": "tight",
})


def load(name):
    p = RES / name
    if not p.exists():
        print(f"skip: {p.name} not found")
        return None
    return json.loads(p.read_text())


def save(fig, name):
    out = RES / name
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


# --------------------------------------------------------------------------- ablation
def fig_ablation(d):
    rows = d["per_seed"]
    seeds = [r["seed"] for r in rows]
    boot = d["bootstrap"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.6), gridspec_kw={"wspace": 0.35})

    x = np.arange(len(rows))
    for i, r in enumerate(rows):
        a1.plot([i, i], [r["auc_diff"], r["auc_phys"]], color="#cccccc", lw=1.5, zorder=1)
    a1.scatter(x, [r["auc_diff"] for r in rows], color=DIFF, s=36, zorder=2, label="differencing only")
    a1.scatter(x, [r["auc_phys"] for r in rows], color=PHYS, s=36, zorder=2, label="physics-informed")
    c = boot["ci95"]
    a1.errorbar([0], [boot["point"]["auc_phys"]], yerr=[[boot["point"]["auc_phys"] - c["auc_p"][0]],
                [c["auc_p"][1] - boot["point"]["auc_phys"]]], color=PHYS, capsize=4, lw=1.2, zorder=3)
    a1.errorbar([0], [boot["point"]["auc_diff"]], yerr=[[boot["point"]["auc_diff"] - c["auc_d"][0]],
                [c["auc_d"][1] - boot["point"]["auc_diff"]]], color=DIFF, capsize=4, lw=1.2, zorder=3)
    a1.set_xticks(x, [f"{s}" for s in seeds])
    a1.set_xlabel("dataset seed (500 frame pairs each)")
    a1.set_ylabel("AUC (leak found and localised)")
    a1.set_ylim(0.4, 1.0)
    a1.set_title("Detection AUC on 5 independent datasets", loc="left")
    a1.errorbar([], [], yerr=[], color="#555555", capsize=4, lw=1.2,
                label="95% bootstrap CI (seed 11 = summary.txt)")
    a1.legend(loc="lower right", fontsize=8.5)

    ld = np.array([r["lim_diff_interp"] for r in rows], dtype=float) * 1000
    lp = np.array([r["lim_phys_interp"] for r in rows], dtype=float) * 1000
    for i in range(len(rows)):
        if np.isfinite(ld[i]) and np.isfinite(lp[i]):
            a2.plot([i, i], [ld[i], lp[i]], color="#cccccc", lw=1.5, zorder=1)
    a2.scatter(x, ld, color=DIFF, s=36, zorder=2)
    a2.scatter(x, lp, color=PHYS, s=36, zorder=2)
    # Where physics already detects >=50% in the weakest bin the limit is only
    # an upper bound: draw it at that bin centre as a downward triangle.
    for i, r in enumerate(rows):
        if not np.isfinite(lp[i]) and r.get("lim_phys_bin"):
            ub = r["lim_phys_bin"] * 1000
            a2.plot([i, i], [ld[i], ub], color="#cccccc", lw=1.5, zorder=1)
            a2.scatter([i], [ub], marker="v", color=PHYS, s=44, zorder=2)
            a2.annotate(f"< {ub:.0f}", (i, ub), textcoords="offset points", xytext=(0, -14),
                        ha="center", fontsize=7.5, color=PHYS)
    a2.set_yscale("log")
    a2.set_xticks(x, [f"{s}" for s in seeds])
    a2.set_xlabel("dataset seed")
    a2.set_ylabel("50% detection limit (mW, at 5% FPR)")
    a2.set_title("Weakest leak found half the time", loc="left")
    a2.set_ylim(8, 260)
    ratio = ld / lp
    ratio = ratio[np.isfinite(ratio)]
    bounds = [ld[i] / (r["lim_phys_bin"] * 1000) for i, r in enumerate(rows)
              if not np.isfinite(lp[i]) and r.get("lim_phys_bin")]
    extra = "".join(f"\n(> {b:.0f}x where only a bound)" for b in bounds[:1])
    box = dict(fc="white", ec="none", alpha=0.9, pad=2)
    if ratio.size:
        a2.text(0.98, 0.5, f"physics finds leaks\n{ratio.min():.0f}-{ratio.max():.0f}x weaker{extra}",
                transform=a2.transAxes, ha="right", va="center", fontsize=9, color=PHYS, bbox=box)
    save(fig, "fig_ablation_ci.png")


# --------------------------------------------------------------------------- quantum
LABELS = {"microbolometer": "uncooled\nmicrobolometer", "qwip": "QWIP", "hgcdte": "HgCdTe",
          "sota_lwir": "large-well\nLWIR", "blip": "BLIP\n(ideal photon\ncounting)"}


def fig_detector(d):
    rows = sorted(d["rows"], key=lambda r: -r["netd_mk"])
    netd = np.array([r["netd_mk"] for r in rows])
    lim = np.array([r["limit_w"] for r in rows]) * 1000
    bound = np.array([r["photon_bound_w"] for r in rows]) * 1000
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(netd, bound, "--", color="#777777", lw=1.3, marker="o", ms=4, mfc="white",
            label="if sensor noise were the only limit")
    ax.plot(netd, lim, "-", color=PHYS, lw=2, marker="o", ms=7,
            label="measured, full simulation (physics pipeline)")
    # Three detectors sit within a factor 2 in NETD; stagger their labels.
    offsets = {"microbolometer": (-6, 10, "right"), "hgcdte": (0, -16, "center"),
               "qwip": (6, 10, "left")}
    for n, l, r in zip(netd, lim, rows):
        dx, dy, ha = offsets.get(r["detector"], (0, 9, "center"))
        ax.annotate(LABELS.get(r["detector"], r["detector"]).replace("\n", " ")
                    if r["detector"] in offsets else LABELS.get(r["detector"], r["detector"]),
                    (n, l), textcoords="offset points", xytext=(dx, dy), ha=ha,
                    va="top" if dy < 0 else "bottom", fontsize=8, color=INK)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("detector NETD (mK)  -- better detector to the right")
    ax.set_ylabel("50% detection limit (mW)")
    ax.set_ylim(bound.min() / 3, lim.max() * 6)
    gap = lim[0] / bound[0]
    ax.set_title("A 140x better detector does not move the detection limit", loc="left")
    ax.text(0.02, 0.04, f"photon-noise model: Planck 8-14 um, Poisson shot + read noise.\n"
            f"Uncooled camera already sits {gap:.0f}x above its own noise bound;\n"
            f"the gap is set by scene nuisances, not by photons.",
            transform=ax.transAxes, fontsize=8, color="#555555", va="bottom")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.62), fontsize=8.5)
    save(fig, "fig_detector_limit.png")


def fig_ladder(d):
    lad = d["isolation_ladder"]
    names = [r["configuration"] for r in lad]
    floor_mw = 0.5
    vals = [r["limit_w"] * 1000 if r["limit_w"] else None for r in lad]
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    y = np.arange(len(lad))[::-1]
    for yi, v, r in zip(y, vals, lad):
        if v is None:
            ax.barh(yi, floor_mw, color=PHYS, alpha=0.35, height=0.6)
            ax.text(floor_mw * 1.15, yi, f"< {floor_mw:g} mW  (below the weakest leak simulated;"
                    f" recall {r['recall']*100:.0f}%)", va="center", fontsize=8.5, color=PHYS)
        else:
            ax.barh(yi, v, color=PHYS if yi == y[0] else "#7fb3d5", height=0.6)
            ax.text(v * 1.08, yi, f"{v:.1f} mW", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, names)
    ax.set_xscale("log")
    ax.set_xlim(0.2, 80)
    ax.set_xlabel("50% detection limit (mW), ideal (BLIP) detector, nuisances removed one at a time")
    ax.set_title("What sets the limit: workload drift, not detector noise", loc="left")
    ax.text(0.99, -0.32, "one dataset per row; steps of a few mW between the upper rows are within sampling noise",
            transform=ax.transAxes, ha="right", fontsize=7.5, color="#777777")
    save(fig, "fig_nuisance_ladder.png")


# --------------------------------------------------------------------------- ML
SHORT = {"gradient_boosting": "gradient boosting", "logistic_regression": "logistic regression",
         "hist_gb (unconstrained)": "hist-GB, unconstrained",
         "hist_gb (physics-monotone)": "hist-GB, physics-monotone",
         "physics_score (hand-weighted)": "physics score (no training)"}
COL = {"gradient_boosting": LEARN, "logistic_regression": LIN, "hist_gb (unconstrained)": LEARN,
       "hist_gb (physics-monotone)": MONO, "physics_score (hand-weighted)": PHYS}


def fig_transfer(d):
    sets = d["feature_sets"]
    fig, axes = plt.subplots(1, len(sets), figsize=(10, 3.6), sharex=True,
                             gridspec_kw={"wspace": 0.85})
    axes = np.atleast_1d(axes)
    for ax, (fs, m) in zip(axes, sets.items()):
        models = [k for k in m if k in SHORT]
        shifts = [k for k in m[models[0]] if k not in ("test (held-out seeds)", "transfer_gap")]
        recs = []
        for k in models:
            test = m[k]["test (held-out seeds)"]["auc_mean"]
            worst = min(m[k][s]["auc_mean"] for s in shifts)
            recs.append((k, test, worst))
        recs.sort(key=lambda t: t[1] - t[2])
        y = np.arange(len(recs))[::-1]
        for yi, (k, t, w) in zip(y, recs):
            ax.plot([w, t], [yi, yi], color="#cccccc", lw=3, solid_capstyle="round", zorder=1)
            ax.scatter([t], [yi], color=COL[k], s=40, zorder=2, edgecolor="white")
            ax.scatter([w], [yi], color="white", edgecolor=COL[k], s=40, lw=1.6, zorder=2)
            ax.text(w - 0.007, yi, f"-{t - w:.3f}", va="center", ha="right", fontsize=8.5,
                    color=PHYS if "physics_score" in k else "#555555")
        ax.set_yticks(y, [SHORT[k] for k, *_ in recs])
        n_feat = "4 features" if "four" in fs else "12 features"
        ax.set_title(f"{fs} ({n_feat})" if "features" not in fs else fs, loc="left", fontsize=10)
        ax.set_xlabel("candidate AUC")
    axes[0].set_xlim(0.76, 1.0)
    axes[0].scatter([], [], color=INK, s=40, label="held-out seeds (same material)")
    axes[0].scatter([], [], color="white", edgecolor=INK, s=40, label="worst of: new material, 2x, 0.5x zoom")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.1, -0.2), ncol=2, fontsize=8.5)
    fig.suptitle("Learned scorers win in-distribution and lose most under shift; "
                 "the physics score barely moves", x=0.02, ha="left", y=1.03, fontsize=11)
    save(fig, "fig_transfer_stability.png")


def main() -> int:
    made = 0
    for fname, fn in (("ablation_ci.json", fig_ablation), ("detector_sweep.json", fig_detector),
                      ("detector_sweep.json", fig_ladder), ("constrained_learning.json", fig_transfer)):
        d = load(fname)
        if d is not None:
            fn(d)
            made += 1
    return 0 if made else 1


if __name__ == "__main__":
    sys.exit(main())
