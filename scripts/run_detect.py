#!/usr/bin/env python3
"""
Detect a thermal leak by comparing two images.

    python scripts/run_detect.py baseline.npy inspection.npy --report out.png

Input formats
-------------
.npy / .npz    A 2D array of temperature in kelvin (or celsius, with --celsius).
               This is the format to prefer: it preserves radiometric values.
.csv           Comma-separated temperature values, one row per image row.
.png/.jpg/...  Read as greyscale and mapped linearly onto --temp-range. Use this
               only for a quick look. A colour-mapped screenshot from a camera
               app has already thrown away the radiometric data, and no amount
               of processing puts it back -- see the note in the README.

Averaging repeated captures
---------------------------
The capture protocol takes 10 frames per condition, because averaging cuts
temporal noise by roughly sqrt(N) and the interesting result lives at the weak
end where that matters. Pass --average and give a directory or a quoted glob
instead of a single file:

    python scripts/run_detect.py baseline_frames/ powered_frames/ --average
    python scripts/run_detect.py 'base_*.npy' 'insp_*.npy' --average --mpp 0.0021

Quote the globs, or the shell expands them into many positional arguments before
this script sees them. Averaging happens in kelvin, after any --celsius
conversion, so the result is a temperature field and every downstream threshold
keeps its physical meaning.

If you have no data yet, run without file arguments to use a simulated pair:

    python scripts/run_detect.py --simulate --leak-power 0.15
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermalleak import MATERIALS, detect_leak, generate_pair  # noqa: E402
from thermalleak.viz import plot_detection  # noqa: E402


def load_frame(path: str, celsius: bool, temp_range: tuple[float, float]) -> np.ndarray:
    suffix = Path(path).suffix.lower()

    if suffix == ".npy":
        frame = np.load(path).astype(float)
    elif suffix == ".npz":
        data = np.load(path)
        frame = data[list(data.keys())[0]].astype(float)
    elif suffix == ".csv":
        frame = np.loadtxt(path, delimiter=",", dtype=float)
    else:
        import cv2
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise SystemExit(f"could not read image: {path}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(float)
        scale = 65535.0 if img.max() > 255 else 255.0
        lo, hi = temp_range
        frame = lo + (hi - lo) * (img / scale)
        print(f"[warn] {path} read as a plain image and mapped onto "
              f"{lo}-{hi} K. Absolute temperatures and the physics tests "
              f"are only as good as that assumption.", file=sys.stderr)
        return frame

    if frame.ndim != 2:
        raise SystemExit(f"expected a 2D array, got shape {frame.shape}")
    return frame + 273.15 if celsius else frame


SUFFIXES = (".npy", ".npz", ".csv", ".tif", ".tiff", ".png", ".jpg", ".jpeg")


def resolve_frames(spec: str) -> list[str]:
    """
    Expand one side of the comparison into a sorted list of frame files.

    Accepts a directory, a glob pattern, or a single file. Sorted so that
    `frame_2` and `frame_10` land in capture order only if they were named with
    zero padding -- which the protocol's numbering does, and which is worth
    doing, because an averaged stack does not care about order but a human
    reading the file list does.
    """
    path = Path(spec)
    if path.is_dir():
        files = sorted(str(p) for p in path.iterdir()
                       if p.suffix.lower() in SUFFIXES)
        if not files:
            raise SystemExit(f"no frame files ({', '.join(SUFFIXES)}) in {spec}")
        return files
    matches = sorted(glob.glob(spec))
    if matches:
        return matches
    if path.exists():
        return [spec]
    raise SystemExit(f"no such file, directory or pattern: {spec}")


def load_side(spec: str, celsius: bool, temp_range: tuple[float, float],
              average: bool, label: str) -> np.ndarray:
    """
    Load one side of the comparison, averaging a stack if asked.

    Averaging is done after conversion to kelvin so the mean is a temperature,
    not a mean of mixed units. The measured scatter is reported next to the
    sqrt(N) expectation: if a stack does not fall roughly that way, something
    moved between frames, and averaging it is hiding a problem rather than
    reducing noise.
    """
    files = resolve_frames(spec)

    if not average:
        if len(files) > 1:
            raise SystemExit(
                f"{label}: '{spec}' matches {len(files)} files. Pass --average "
                f"to average them, or name a single frame.")
        return load_frame(files[0], celsius, temp_range)

    frames = [load_frame(f, celsius, temp_range) for f in files]
    shapes = {f.shape for f in frames}
    if len(shapes) > 1:
        raise SystemExit(f"{label}: frames have differing shapes {sorted(shapes)}")

    stack = np.stack(frames, axis=0)
    mean = stack.mean(axis=0)

    if len(frames) > 1:
        # Scatter across the stack, median over pixels so a hot component or a
        # dead pixel does not set the number.
        scatter = float(np.median(stack.std(axis=0, ddof=1)))
        print(f"[{label}] averaged {len(frames)} frames from '{spec}': "
              f"per-pixel scatter {scatter * 1000:.0f} mK, "
              f"so the mean carries about {scatter / np.sqrt(len(frames)) * 1000:.0f} mK "
              f"({np.sqrt(len(frames)):.1f}x reduction)")
    else:
        print(f"[{label}] --average given but '{spec}' matched a single frame; "
              f"nothing to average")
    return mean


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline", nargs="?", help="intact reference frame")
    ap.add_argument("inspection", nargs="?", help="frame to inspect")
    ap.add_argument("--simulate", action="store_true",
                    help="use a simulated pair instead of files")
    ap.add_argument("--leak-power", type=float, default=0.15,
                    help="simulated leak power in watts (0 for no leak)")
    ap.add_argument("--material", default="abs_casing", choices=sorted(MATERIALS),
                    help="plate material preset")
    ap.add_argument("--mpp", type=float, default=0.0025,
                    help="metres per pixel at the object")
    ap.add_argument("--average", action="store_true",
                    help="treat each positional as a directory or quoted glob of "
                         "repeated captures and average them before detecting")
    ap.add_argument("--celsius", action="store_true",
                    help="input arrays are in degrees celsius")
    ap.add_argument("--temp-range", type=float, nargs=2, default=(283.15, 333.15),
                    help="kelvin range for plain-image input")
    ap.add_argument("--polarity", default="hot", choices=["hot", "cold", "both"],
                    help="'cold' for cryogenic or coolant leaks")
    ap.add_argument("--no-physics", action="store_true",
                    help="ablate the physics stage (differencing only)")
    ap.add_argument("--unknown-order", action="store_true",
                    help="do not assume the first frame is the intact one")
    ap.add_argument("--report", help="write a PNG inspection report here")
    args = ap.parse_args()

    truth_centre = None

    if args.simulate or not (args.baseline and args.inspection):
        rng = np.random.default_rng(0)
        pair = generate_pair(rng, material=MATERIALS[args.material],
                             leak_power_watts=args.leak_power)
        baseline, inspection = pair.baseline, pair.inspection
        mpp = pair.metres_per_pixel
        truth_centre = pair.leak_centre_px
        print(f"[simulated] leak power {args.leak_power} W, "
              f"kind '{pair.leak_kind}', {mpp * 1000:.2f} mm/px")
    else:
        baseline = load_side(args.baseline, args.celsius, tuple(args.temp_range),
                             args.average, "baseline")
        inspection = load_side(args.inspection, args.celsius, tuple(args.temp_range),
                               args.average, "inspection")
        if baseline.shape != inspection.shape:
            raise SystemExit(f"baseline {baseline.shape} and inspection "
                             f"{inspection.shape} differ in shape")
        mpp = args.mpp

    result = detect_leak(
        baseline, inspection,
        material=args.material, metres_per_pixel=mpp,
        known_baseline=None if args.unknown_order else "a",
        use_physics=not args.no_physics,
        polarity=args.polarity,
    )

    print()
    print(result.summary())
    print()

    if result.findings:
        print(f"{'#':<3} {'row':>6} {'col':>6} {'peak K':>8} {'z':>8} "
              f"{'phys':>6} {'L fit':>7} {'geom':>6} {'verdict':>12}")
        for i, f in enumerate(result.findings, 1):
            v = f.verdict
            print(f"{i:<3} {f.centroid[0]:>6.0f} {f.centroid[1]:>6.0f} "
                  f"{f.candidate.peak_delta_k:>8.2f} {f.candidate.z_peak:>8.1f} "
                  f"{(v.physics_score if v else float('nan')):>6.2f} "
                  f"{(v.fitted_length_px if v else float('nan')):>7.1f} "
                  f"{(v.geometry if v else '-'):>6} "
                  f"{(v.verdict() if v else 'n/a'):>12}")

    if args.report:
        path = plot_detection(result, baseline, inspection, args.report,
                              truth_centre=truth_centre)
        print(f"\nreport written to {path}")

    return 0 if result.leak_detected else 1


if __name__ == "__main__":
    raise SystemExit(main())
