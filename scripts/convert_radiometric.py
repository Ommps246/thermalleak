#!/usr/bin/env python3
"""
Convert a directory of radiometric camera exports into the .npy frames the
pipeline expects.

    python3 scripts/convert_radiometric.py raw_captures/ --outdir frames/ --units celsius
    python3 scripts/convert_radiometric.py 'export_*.csv' --outdir frames/ --units celsius
    python3 scripts/convert_radiometric.py raw/ --outdir frames/ --flir-jpeg

Output is one `.npy` per input, a 2D float array of temperature in **kelvin**,
which is what every threshold in `thermalleak` is expressed in.

The go/no-go
------------
This script is where the protocol's camera check actually happens. It refuses
8-bit and colour input by default, because a false-colour render has already
thrown the temperature away and no processing puts it back. Run it on one frame
before collecting a session:

    python3 scripts/convert_radiometric.py one_frame.jpg --dry-run

If that refuses, your export path is discarding the radiometric data. Fix the
export before capturing a dataset, not after.

Input formats
-------------
.csv              Comma-separated temperatures, one row per image row. This is
                  the most reliable export path -- it is already physical units,
                  so only --units is needed. FLIR Tools writes celsius.
.tif/.tiff/.png   16-bit single-channel. Contents depend on the camera: some
                  store temperature directly scaled (centikelvin is common),
                  some store raw sensor counts. Use --units raw with --scale and
                  --offset, which apply as  kelvin = raw * scale + offset.
.npy/.npz         Passed through, with --units applied. Useful for restamping a
                  stack that is already extracted but in celsius.
radiometric JPEG  With --flir-jpeg, the embedded raw thermal image is pulled out
                  with `exiftool` and inverted through the camera's Planck
                  constants. See the caveat on that option below.
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

IMAGE_SUFFIXES = (".tif", ".tiff", ".png")
ARRAY_SUFFIXES = (".npy", ".npz")
JPEG_SUFFIXES = (".jpg", ".jpeg")
ALL_SUFFIXES = IMAGE_SUFFIXES + ARRAY_SUFFIXES + JPEG_SUFFIXES + (".csv",)


class NotRadiometric(Exception):
    """Raised when an input has already lost its temperature information."""


# --------------------------------------------------------------------------
# Readers. Each returns an array in whatever units the file holds; the caller
# applies --units.
# --------------------------------------------------------------------------

def _read_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", dtype=float)


def _read_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        return data[list(data.keys())[0]].astype(float)
    return np.load(path).astype(float)


def _read_image(path: Path, force: bool) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise NotRadiometric(f"could not read {path.name}")

    if img.ndim == 3:
        raise NotRadiometric(
            f"{path.name} has {img.shape[2]} colour channels. A colour thermal "
            f"image is a palette render -- the temperature is gone. Export "
            f"radiometric data instead (16-bit single channel, or CSV).")

    if img.dtype == np.uint8 and not force:
        raise NotRadiometric(
            f"{path.name} is 8-bit. That is 256 levels across the whole scene: "
            f"even if it began as temperature it no longer resolves the "
            f"sub-kelvin differences this pipeline works in. Export 16-bit "
            f"radiometric data, or pass --force to proceed knowing the result "
            f"is not physically meaningful.")

    return img.astype(float)


def _flir_metadata(path: Path) -> dict:
    """Pull the Planck constants out of a FLIR radiometric JPEG."""
    out = subprocess.run(
        ["exiftool", "-json", "-PlanckR1", "-PlanckR2", "-PlanckB", "-PlanckF",
         "-PlanckO", "-RawThermalImageType", str(path)],
        capture_output=True, text=True, check=True)
    return json.loads(out.stdout)[0]


def _read_flir_jpeg(path: Path) -> np.ndarray:
    """
    Extract and linearise the raw thermal image embedded in a FLIR JPEG.

    The sensor stores a raw signal S; temperature follows from the camera's
    calibrated Planck fit,

        T = B / ln( R1 / (R2 * (S + O)) + F )

    with R1, R2, B, F, O written into the EXIF by the camera.

    **Caveat, and it is not a small one.** This inverts the radiance curve
    assuming the pixel's radiance is the object's own. It applies no emissivity
    or reflected-background correction, so a low-emissivity surface will read
    cold by exactly the amount the camera app would have corrected for. That is
    the same physics `camera.apparent_temperature` models in the forward
    direction. For a matte casing at emissivity ~0.95 the error is a few tenths
    of a kelvin; for bare metal it is large. Put a matte black tape patch in
    frame, as the protocol says, and check this output against the camera's own
    reading of that patch before trusting a session.
    """
    import cv2

    if shutil.which("exiftool") is None:
        raise NotRadiometric(
            "--flir-jpeg needs `exiftool` on PATH (brew install exiftool). It is "
            "what reaches the raw thermal image embedded inside the JPEG; the "
            "visible picture is only a preview.")

    meta = _flir_metadata(path)
    missing = [k for k in ("PlanckR1", "PlanckR2", "PlanckB", "PlanckF", "PlanckO")
               if meta.get(k) is None]
    if missing:
        raise NotRadiometric(
            f"{path.name} carries no FLIR Planck constants ({', '.join(missing)} "
            f"absent). This is a plain JPEG, not a radiometric one -- the "
            f"default share path on FLIR One and Seek strips the raw data. Use "
            f"the app's radiometric or CSV export.")

    raw = subprocess.run(["exiftool", "-b", "-RawThermalImage", str(path)],
                         capture_output=True, check=True).stdout
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 2:
        raise NotRadiometric(f"could not decode the raw thermal image in {path.name}")

    signal = img.astype(float)
    R1, R2 = float(meta["PlanckR1"]), float(meta["PlanckR2"])
    B, F, O = float(meta["PlanckB"]), float(meta["PlanckF"]), float(meta["PlanckO"])

    denom = R2 * (signal + O)
    with np.errstate(divide="ignore", invalid="ignore"):
        arg = R1 / np.where(np.abs(denom) < 1e-12, np.nan, denom) + F
        kelvin = B / np.log(np.where(arg > 1.0 + 1e-12, arg, np.nan))

    if not np.isfinite(kelvin).any():
        raise NotRadiometric(f"Planck inversion produced no finite values for "
                             f"{path.name}; the constants may not match the data")

    # A few dead pixels inverting out of range is normal; a mostly-NaN frame is
    # not, and silently filling it would hide a broken conversion.
    bad = float(np.mean(~np.isfinite(kelvin)))
    if bad > 0.02:
        raise NotRadiometric(f"{path.name}: {bad:.0%} of pixels failed the Planck "
                             f"inversion -- refusing rather than interpolating")
    if bad > 0:
        kelvin = np.where(np.isfinite(kelvin), kelvin,
                          float(np.nanmedian(kelvin)))
    return kelvin


# --------------------------------------------------------------------------

def read_any(path: Path, flir_jpeg: bool, force: bool) -> tuple[np.ndarray, bool]:
    """Returns (array, already_kelvin)."""
    suffix = path.suffix.lower()
    if suffix in JPEG_SUFFIXES:
        if not flir_jpeg:
            raise NotRadiometric(
                f"{path.name} is a JPEG. Pass --flir-jpeg to pull the embedded "
                f"raw thermal image out with exiftool; without it, a JPEG is a "
                f"palette render and the temperature is gone.")
        return _read_flir_jpeg(path), True
    if suffix == ".csv":
        return _read_csv(path), False
    if suffix in ARRAY_SUFFIXES:
        return _read_array(path), False
    if suffix in IMAGE_SUFFIXES:
        return _read_image(path, force), False
    raise NotRadiometric(f"unsupported file type: {path.name}")


def to_kelvin(values: np.ndarray, units: str, scale: float, offset: float
              ) -> np.ndarray:
    if units == "kelvin":
        return values
    if units == "celsius":
        return values + 273.15
    return values * scale + offset


def sanity(kelvin: np.ndarray) -> str | None:
    """A plausibility note, not a hard rule -- cryogenic work is legitimate."""
    lo, hi = float(np.nanmin(kelvin)), float(np.nanmax(kelvin))
    if not np.isfinite([lo, hi]).all():
        return "contains non-finite values"
    if hi < 173.15 or lo > 773.15:
        return (f"spans {lo:.1f}-{hi:.1f} K ({lo - 273.15:.0f} to "
                f"{hi - 273.15:.0f} C) -- check --units, --scale and --offset")
    if hi - lo < 0.05:
        return f"spans only {hi - lo:.3f} K -- is this a blank or saturated frame?"
    return None


def resolve(spec: str) -> list[Path]:
    path = Path(spec)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in ALL_SUFFIXES)
        if not files:
            raise SystemExit(f"no convertible files in {spec}")
        return files
    matches = sorted(Path(m) for m in glob.glob(spec))
    if matches:
        return matches
    if path.exists():
        return [path]
    raise SystemExit(f"no such file, directory or pattern: {spec}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="directory, quoted glob, or single file")
    ap.add_argument("--outdir", default="frames",
                    help="where to write the .npy frames (default: frames/)")
    ap.add_argument("--units", default="celsius",
                    choices=["celsius", "kelvin", "raw"],
                    help="units held in the input (default: celsius)")
    ap.add_argument("--scale", type=float, default=0.01,
                    help="with --units raw: kelvin = raw * scale + offset "
                         "(default 0.01, i.e. centikelvin)")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="with --units raw: additive term, kelvin")
    ap.add_argument("--flir-jpeg", action="store_true",
                    help="extract the raw thermal image from FLIR radiometric "
                         "JPEGs via exiftool and invert the Planck curve")
    ap.add_argument("--force", action="store_true",
                    help="convert 8-bit input anyway; the output is not "
                         "physically meaningful and this says so on every frame")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would happen, write nothing -- this is the "
                         "protocol's go/no-go check")
    args = ap.parse_args()

    files = resolve(args.source)
    outdir = Path(args.outdir)
    if not args.dry_run:
        outdir.mkdir(parents=True, exist_ok=True)

    print(f"{len(files)} file(s) from '{args.source}'"
          f"{' [dry run]' if args.dry_run else f' -> {outdir}/'}")
    if args.units == "raw":
        print(f"raw scaling: kelvin = raw * {args.scale} + {args.offset}")
    print()

    written = failed = 0
    for path in files:
        try:
            values, already_kelvin = read_any(path, args.flir_jpeg, args.force)
        except NotRadiometric as exc:
            print(f"  REFUSED  {path.name}: {exc}")
            failed += 1
            continue
        except (subprocess.CalledProcessError, OSError, ValueError) as exc:
            print(f"  FAILED   {path.name}: {exc}")
            failed += 1
            continue

        if values.ndim != 2:
            print(f"  FAILED   {path.name}: expected a 2D frame, got shape {values.shape}")
            failed += 1
            continue

        kelvin = values if already_kelvin else to_kelvin(
            values, args.units, args.scale, args.offset)

        note = sanity(kelvin)
        target = outdir / (path.stem + ".npy")
        detail = (f"{kelvin.shape[1]}x{kelvin.shape[0]}  "
                  f"{np.nanmin(kelvin) - 273.15:6.1f} to "
                  f"{np.nanmax(kelvin) - 273.15:6.1f} C")

        if not args.dry_run:
            np.save(target, kelvin)
        written += 1
        flag = f"  [!] {note}" if note else ""
        print(f"  {'would write' if args.dry_run else 'wrote'}  "
              f"{target.name:<28} {detail}{flag}")

    print()
    if args.dry_run:
        verdict = ("GO -- radiometric data present" if written and not failed
                   else "NO-GO -- see refusals above" if failed
                   else "NO-GO -- nothing readable")
        print(f"{verdict}  ({written} convertible, {failed} refused)")
        return 0 if written and not failed else 1

    print(f"{written} written, {failed} refused or failed")
    if written:
        print(f"\nnext:  python3 scripts/run_detect.py {outdir}/ <powered_dir>/ "
              f"--average --mpp <measured>")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
