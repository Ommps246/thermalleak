"""
Figures: inspection reports and evaluation plots.

Two audiences. `plot_detection` is the operator's view -- what was found and
where. `plot_evaluation` is the researcher's view -- does the method work, and
where does it stop working. Keeping them separate stops either from becoming a
compromise that serves neither.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from .evaluate import PairOutcome, detection_rate_by_power, roc_curve
from .pipeline import DetectionResult

# "Ironbow"-adjacent ramp: dark cold, bright hot, monotonic in luminance so it
# stays readable in greyscale and to colour-blind viewers.
THERMAL_CMAP = "inferno"
DIFF_CMAP = "RdBu_r"


def plot_detection(result: DetectionResult, baseline: np.ndarray,
                   inspection: np.ndarray, path: str,
                   truth_centre: tuple[float, float] | None = None,
                   title: str | None = None) -> str:
    """
    Four-panel inspection report: baseline, inspection, difference, verdict.

    The difference panel is deliberately shown on a symmetric diverging scale
    clipped to a robust percentile. A linear scale spanning the full range makes
    every difference image look empty, which is how a real anomaly gets missed
    by a human reviewer glancing at the report.
    """
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))

    vmin = float(min(baseline.min(), inspection.min()))
    vmax = float(max(np.percentile(baseline, 99.5), np.percentile(inspection, 99.5)))

    for ax, img, name in ((axes[0], baseline, "Baseline"),
                          (axes[1], inspection, "Inspection")):
        im = ax.imshow(img - 273.15, cmap=THERMAL_CMAP,
                       vmin=vmin - 273.15, vmax=vmax - 273.15)
        ax.set_title(name, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, label="°C")

    diff = result.difference
    lim = float(np.percentile(np.abs(diff), 99.5)) or 1e-3
    im = axes[2].imshow(diff, cmap=DIFF_CMAP, vmin=-lim, vmax=lim)
    axes[2].set_title("Normalised difference", fontsize=11)
    axes[2].set_xticks([]); axes[2].set_yticks([])
    plt.colorbar(im, ax=axes[2], fraction=0.046, label="K")

    axes[3].imshow(inspection - 273.15, cmap="gray")
    axes[3].set_title("Verdict", fontsize=11)
    axes[3].set_xticks([]); axes[3].set_yticks([])

    # Label with rank and the score that produced that rank.
    #
    # These circles used to carry the physics score alone, which quietly
    # misrepresented the panel: `pipeline` orders findings by
    # `combined_score = significance * physics_score`, not by physics alone, so
    # two candidates could print an identical 0.80 while sitting first and
    # second -- and nothing on the figure explained the ordering. Showing the
    # rank and the combined score means the numbers on the image are the numbers
    # that ranked them.
    #
    # Colour still encodes the *physics* verdict (green accepted, red rejected),
    # because that is the physical judgement and it is independent of how bright
    # the candidate happened to be. So a green #3 is a candidate that is
    # physically plausible but was outranked, which is exactly the distinction
    # worth seeing. The per-test breakdown stays in the caption.
    for rank, finding in enumerate(result.findings, start=1):
        row, col = finding.centroid
        phys = finding.verdict.physics_score if finding.verdict else 1.0
        accepted = phys >= 0.45
        colour = "#2ecc71" if accepted else "#e74c3c"
        radius = max(6.0, 1.5 * result.thermal_length_px)
        axes[3].add_patch(Circle((col, row), radius, fill=False,
                                 color=colour, lw=2.0))

        # Flip the label inside the frame near the right edge and clamp it
        # vertically. Candidates cluster at frame borders on real imagery, so a
        # label that always sits to the right gets cut off exactly where the
        # interesting cases are.
        height, width = inspection.shape
        if col > 0.72 * width:
            x, align = col - radius - 2, "right"
        else:
            x, align = col + radius + 2, "left"
        y = float(np.clip(row, 0.03 * height, 0.97 * height))

        axes[3].text(x, y, f"#{rank}  {finding.combined_score:.2f}",
                     color=colour, fontsize=9, va="center", ha=align,
                     weight="bold",
                     path_effects=[pe.withStroke(linewidth=2.0,
                                                 foreground="black", alpha=0.55)])

    if truth_centre is not None:
        axes[3].plot(truth_centre[1], truth_centre[0], "x", color="#3498db",
                     markersize=13, markeredgewidth=2.5, label="true leak")
        axes[3].legend(loc="lower right", fontsize=8, framealpha=0.85)

    # Wrap the caption explicitly rather than relying on Matplotlib's `wrap`.
    # With `wrap=True` and `bbox_inches="tight"` a long summary silently
    # expands the figure canvas, leaving most of the saved PNG as blank space
    # below the panels.
    import textwrap
    caption = title or result.summary()
    wrapped = "\n".join(textwrap.wrap(caption, width=150)[:3])
    n_lines = wrapped.count("\n") + 1

    fig.tight_layout(rect=(0, 0, 1, 1 - 0.045 * n_lines))
    fig.text(0.5, 0.985, wrapped, ha="center", va="top", fontsize=9.5)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_evaluation(outcomes_physics: list[PairOutcome],
                    outcomes_baseline: list[PairOutcome],
                    path: str, target_fpr: float = 0.05) -> str:
    """
    Three-panel ablation figure: ROC, sensitivity curve, score separation.

    The middle panel is the one that matters to a practitioner and the one most
    papers omit -- accuracy averaged over a dataset hides the fact that
    performance is near-perfect on obvious leaks and near-chance on the leaks
    you actually needed help finding.
    """
    from .evaluate import threshold_at_fpr

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # --- ROC --------------------------------------------------------------
    for out, label, colour in ((outcomes_baseline, "Differencing only", "#95a5a6"),
                               (outcomes_physics, "Physics-informed", "#c0392b")):
        fpr, tpr, _, auc = roc_curve(out)
        axes[0].plot(fpr, tpr, color=colour, lw=2.2,
                     label=f"{label} (AUC {auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate (localised)")
    axes[0].set_title("Detection ROC", fontsize=11)
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].grid(alpha=0.25)

    # --- sensitivity ------------------------------------------------------
    for out, label, colour in ((outcomes_baseline, "Differencing only", "#95a5a6"),
                               (outcomes_physics, "Physics-informed", "#c0392b")):
        thr = threshold_at_fpr(out, target_fpr)
        centres, rates, counts = detection_rate_by_power(out, thr)
        if centres.size:
            axes[1].semilogx(centres, rates, "o-", color=colour, lw=2.0,
                             markersize=5, label=label)
    axes[1].axhline(0.5, color="k", ls=":", lw=1, alpha=0.5)
    axes[1].set_xlabel("Leak power (W)")
    axes[1].set_ylabel(f"Detection rate at {target_fpr:.0%} FPR")
    axes[1].set_title("Sensitivity vs leak power", fontsize=11)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc="lower right", fontsize=9)
    axes[1].grid(alpha=0.25, which="both")

    # --- score separation -------------------------------------------------
    pos = [o.score for o in outcomes_physics if o.has_leak and o.localised]
    neg = [o.score for o in outcomes_physics if not o.has_leak]
    bins = np.linspace(0, 1, 26)
    axes[2].hist(neg, bins=bins, alpha=0.65, color="#95a5a6", label="No leak")
    axes[2].hist(pos, bins=bins, alpha=0.65, color="#c0392b", label="Leak")
    axes[2].set_xlabel("Detection score")
    axes[2].set_ylabel("Pairs")
    axes[2].set_title("Score separation (physics-informed)", fontsize=11)
    axes[2].legend(fontsize=9)
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_profile(verdict, path: str) -> str:
    """Radial profile with the fitted K0 model -- the physics test, made visible."""
    from .physics import bessel_model

    r = np.linspace(0.5, 5 * verdict.fitted_length_px, 200)
    model = bessel_model(r, verdict.fitted_amplitude_k, verdict.fitted_length_px, 0.0)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(r, model, color="#c0392b", lw=2,
            label=f"K0 fit, L = {verdict.fitted_length_px:.1f} px")
    ax.axvline(verdict.expected_length_px, color="#2980b9", ls="--", lw=1.5,
               label=f"material L = {verdict.expected_length_px:.1f} px")
    ax.set_xlabel("Radius (px)")
    ax.set_ylabel("Excess temperature (K)")
    ax.set_title(f"Physics score {verdict.physics_score:.2f} "
                 f"({verdict.verdict()})", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
