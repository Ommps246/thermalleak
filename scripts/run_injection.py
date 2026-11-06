#!/usr/bin/env python3
"""
Test the detector on a REAL image by injecting a simulated leak into it.

    python scripts/run_injection.py my_thermal_photo.jpg --power 0.15 --report inj.png

    # detection limit against this background
    python scripts/run_injection.py my_thermal_photo.jpg --sweep

Any image works -- a thermal photo off the web, a frame from a dataset, or a
capture from your own camera. The background is real; the leak is simulated and
its position and power are therefore known exactly, which is what makes the
result scorable.

Two frames from a sequence -- prefer this
-----------------------------------------
With a single image the baseline and the inspection frame share an *identical*
background, so nothing drifted between them and the detection limit comes out
optimistically low. Given a thermal sequence from a fixed camera, use two
different frames instead:

    python scripts/run_injection.py --baseline seq_t000.npy \
        --background seq_t300.npy --sweep

Now the pair carries the sequence's genuine inter-frame differences -- real
sensor drift, real scene change, real registration error, real reflection
movement, none of it modelled -- while ground truth stays exact, because the
leak was placed. Add --compare to sweep both ways and print the gap.

Read the honest caveats in `thermalleak/inject.py` before quoting numbers from
this. In short: injection validates the detector against real *clutter*, not
against real *leaks*. The injected signal obeys the same physics the detector
looks for, so it cannot tell you whether real leaks look like the model. Only
real paired data can.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermalleak import MATERIALS, detect_leak  # noqa: E402
from thermalleak.inject import (  # noqa: E402
    InjectionSpec, inject_leak, load_as_field, sweep_power,
)
from thermalleak.viz import plot_detection  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?",
                    help="any image file to use as the real background; the "
                         "baseline is the same frame, so see --baseline")
    ap.add_argument("--baseline",
                    help="reference frame, a DIFFERENT real frame from the same "
                         "sequence (use with --background)")
    ap.add_argument("--background",
                    help="frame the leak is injected into (use with --baseline)")
    ap.add_argument("--compare", action="store_true",
                    help="with --baseline/--background and --sweep: also sweep "
                         "the shared-background way and report the gap")
    ap.add_argument("--no-misalign", action="store_true",
                    help="skip the synthetic rigid shift. Worth setting with two "
                         "real frames, which already carry real misregistration")
    ap.add_argument("--power", type=float, default=0.15,
                    help="injected leak power in watts")
    ap.add_argument("--kind", default="point", choices=["point", "crack"])
    ap.add_argument("--material", default="abs_casing", choices=sorted(MATERIALS))
    ap.add_argument("--mpp", type=float, default=0.0025,
                    help="assumed metres per pixel")
    ap.add_argument("--temp-range", type=float, nargs=2, default=(293.15, 318.15),
                    help="kelvin range that image brightness is mapped onto")
    ap.add_argument("--max-size", type=int, default=512,
                    help="downscale the image so its longest side is at most this")
    ap.add_argument("--at", type=float, nargs=2, default=None,
                    metavar=("ROW", "COL"),
                    help="inject here instead of the automatic quiet location")
    ap.add_argument("--reflection", action="store_true",
                    help="also add a synthetic specular reflection confounder")
    ap.add_argument("--sweep", action="store_true",
                    help="sweep leak power to find the detection limit")
    ap.add_argument("--sweep-range", type=float, nargs=2, default=(0.001, 1.0),
                    metavar=("MIN", "MAX"),
                    help="watts spanned by --sweep (default 0.001 1.0; the "
                         "bottom must sit below the limit or the sweep floors)")
    ap.add_argument("--sweep-points", type=int, default=13,
                    help="number of powers in the sweep")
    ap.add_argument("--repeats", type=int, default=5,
                    help="independent draws per power. The limit is the lowest "
                         "power detected and localised in at least half of "
                         "them, which is what makes it comparable to the 50%% "
                         "limit in evaluate.py")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", help="write a PNG inspection report here")
    args = ap.parse_args()

    if bool(args.baseline) != bool(args.background):
        raise SystemExit("--baseline and --background must be given together")
    if not args.baseline and not args.image:
        raise SystemExit("give an image, or --baseline with --background")
    if args.baseline and args.image:
        raise SystemExit("give either a single image or --baseline/--background, "
                         "not both")

    paired = bool(args.baseline)
    tr, ms = tuple(args.temp_range), args.max_size

    if paired:
        baseline_field = load_as_field(args.baseline, tr, ms)
        field = load_as_field(args.background, tr, ms)
        if baseline_field.shape != field.shape:
            raise SystemExit(f"baseline {baseline_field.shape} and background "
                             f"{field.shape} differ in shape")
        drift = float(np.median(np.abs(field - baseline_field)))
        print(f"loaded a PAIR from one sequence: {field.shape[1]}x{field.shape[0]} px")
        print(f"  baseline   {args.baseline}")
        print(f"  background {args.background}")
        print(f"  median inter-frame difference {drift * 1000:.0f} mK -- this is "
              f"real drift the single-image mode does not have")
    else:
        baseline_field = None
        field = load_as_field(args.image, tr, ms)
        print(f"loaded {args.image}: {field.shape[1]}x{field.shape[0]} px, "
              f"{field.min() - 273.15:.1f} to {field.max() - 273.15:.1f} C")
        print("[note] baseline and inspection share an identical background, so "
              "any detection limit from this run is optimistic and not "
              "comparable to the simulated one. See --baseline/--background.")
    print(f"assuming {args.material}, {args.mpp * 1000:.2f} mm/px "
          f"(thermal length {MATERIALS[args.material].thermal_length() / args.mpp:.1f} px)")
    print()

    misalign = not args.no_misalign

    if args.sweep:
        powers = np.geomspace(args.sweep_range[0], args.sweep_range[1],
                              args.sweep_points)

        def run_sweep(base, title):
            rows = sweep_power(field, powers, material=args.material,
                               metres_per_pixel=args.mpp, seed=args.seed,
                               kind=args.kind, baseline_field=base,
                               add_misalignment=misalign, repeats=args.repeats)
            print(title)
            print(f"{'power W':>9} {'peak K':>8} {'hits':>8} {'rate':>7} "
                  f"{'phys':>6} {'score':>7}")
            for r in rows:
                print(f"{r['power_watts']:>9.4f} {r['peak_rise_k']:>8.2f} "
                      f"{r['n_hits']:>4}/{r['n_repeats']:<3} "
                      f"{r['detect_rate']:>7.2f} "
                      f"{r['physics_score']:>6.2f} {r['score']:>7.3f}")
            # 50% limit: lowest power at or above half detection that *stays*
            # there for the next few points, so one lucky bin does not set it.
            #
            # Confirmed over a window rather than over every stronger power,
            # which sounds stricter but is not: a sweep taken far past the
            # useful range eventually injects absurd leaks -- hundreds of
            # kelvin, swamping the frame -- where localisation fails and the
            # rate falls back to zero. Requiring *all* higher powers to hold
            # then reports "never reached" for a detector that was at 100%
            # eight bins earlier.
            CONFIRM = 3
            limit = None
            for i, r in enumerate(rows):
                window = rows[i:i + CONFIRM + 1]
                if r["detect_rate"] >= 0.5 and all(
                        q["detect_rate"] >= 0.5 for q in window):
                    limit = r["power_watts"]
                    break
            print()
            return limit

        label = ("two frames from the sequence (real inter-frame drift)"
                 if paired else "shared background (one frame used twice)")
        limit = run_sweep(baseline_field, f"--- {label} ---")
        if limit is not None:
            print(f"50% detection limit, {label}: ~{limit:.4f} W")
            if limit <= powers[0] * 1.001:
                print("[warn] the limit sits at the bottom of the sweep, so it "
                      "is a floor, not a measurement -- lower --sweep-range")
        else:
            print(f"no injected leak reached 50% detection, {label} -- this "
                  f"background is harder than the simulator, which is exactly "
                  f"the sort of thing worth reporting.")

        if args.compare and paired:
            print()
            shared = run_sweep(None, "--- same sequence, shared background ---")
            print(f"50% detection limit, shared background: "
                  f"{'~%.4f W' % shared if shared else 'not reached'}")
            print()
            if limit and shared:
                print(f"GAP: the shared-background shortcut reports "
                      f"{limit / shared:.1f}x better sensitivity "
                      f"({shared:.4f} W vs {limit:.4f} W) on identical data. "
                      f"That factor is the cost of the shortcut, and it is why "
                      f"the two numbers are not comparable.")
        return 0

    spec = InjectionSpec(power_watts=args.power, kind=args.kind,
                         centre_px=tuple(args.at) if args.at else None)
    pair = inject_leak(field, spec, material=args.material,
                       metres_per_pixel=args.mpp, seed=args.seed,
                       add_reflection=args.reflection,
                       add_misalignment=misalign,
                       baseline_field=baseline_field)

    print(f"injected {args.power} W {args.kind} leak at "
          f"(row {pair.leak_centre_px[0]:.0f}, col {pair.leak_centre_px[1]:.0f}), "
          f"peak rise {pair.truth['peak_rise_k']:.2f} K")
    print()

    result = detect_leak(pair.baseline, pair.inspection,
                         material=pair.material,
                         metres_per_pixel=pair.metres_per_pixel)
    print(result.summary())
    print()

    if result.findings:
        print(f"{'#':<3} {'row':>6} {'col':>6} {'peak K':>8} {'z':>8} "
              f"{'phys':>6} {'L fit':>7} {'verdict':>13}")
        for i, f in enumerate(result.findings, 1):
            v = f.verdict
            print(f"{i:<3} {f.centroid[0]:>6.0f} {f.centroid[1]:>6.0f} "
                  f"{f.candidate.peak_delta_k:>8.2f} {f.candidate.z_peak:>8.1f} "
                  f"{(v.physics_score if v else float('nan')):>6.2f} "
                  f"{(v.fitted_length_px if v else float('nan')):>7.1f} "
                  f"{(v.verdict() if v else 'n/a'):>13}")

    best = result.best()
    if best is not None and pair.leak_centre_px is not None:
        d = np.hypot(best.centroid[0] - pair.leak_centre_px[0],
                     best.centroid[1] - pair.leak_centre_px[1])
        print()
        print(f"localisation error: {d:.1f} px "
              f"({'HIT' if d <= 8 else 'MISS -- top candidate is not the leak'})")

    if args.report:
        path = plot_detection(result, pair.baseline, pair.inspection, args.report,
                              truth_centre=pair.leak_centre_px)
        print(f"\nreport written to {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
