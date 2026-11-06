"""
Injection studies: put a simulated leak into a real thermal image.

Why this exists
---------------
Pure simulation is the weakest form of evidence, and it is the first thing a
reviewer attacks. But real *paired* data -- the same object photographed intact
and then leaking, from the same viewpoint -- barely exists, and producing it
means damaging equipment on purpose.

Injection sidesteps the problem. Take one real thermal image as the baseline.
Solve the heat equation for a leak of known power. Add it. Render both frames
through the camera model. You now have a labelled pair whose *background* is
real -- real scene texture, real emissivity structure, real sensor noise, real
clutter -- with ground truth you control exactly, because you put the leak
there.

This is a standard and respectable evaluation technique in detection work. It
does not replace real paired data, but it is far stronger than simulation alone,
because the thing a detector actually struggles against is the background, and
here the background is not modelled at all.

What it cannot tell you
-----------------------
Injection validates the detector against realistic *clutter*. It cannot validate
the *leak model*: the injected signal obeys the same K0 physics the detector
tests for, so a perfect score here partly reflects that agreement rather than
proving real leaks look this way. Say so plainly in any write-up. Only real
paired data closes that loop.

**Detection limits from single-image injection are optimistic, and are not
comparable to the limits in `evaluate.py`.** The reason is structural: with one
image the baseline and the inspection frame share an *identical* background,
because one was made from the other. The scene did not drift, the device did not
change workload, no component grew hotter, the emissivity map is the same map.
All that separates the two frames is sensor noise, a small misalignment, and the
injected leak.

The size of that effect is measured, on a real static-camera thermal sequence
(OTCBVS Dataset 03, sequence 3a; 128x128 crop; 349 frames in which nothing moves
inside it; fixed gain verified). Sweeping the same frames both ways --
`baseline_field` set to a genuinely different frame, and left as None so the
background is shared -- gives:

    shared background            50% limit ~0.0100 W
    two frames from the sequence 50% limit ~0.0178 W
                                 -------------------
                                 shortcut flatters by 1.8x

12 base pairs x 3 draws per power, 13 powers from 1 mW to 1 W, identical frames
and identical injected leaks in both arms; the only difference is which frame the
detector was handed as the reference. Reproduce with:

    python3 scripts/run_injection.py --baseline t000.npy --background t300.npy \
        --sweep --compare --repeats 8

What those numbers are not. The frames are **8-bit**, so this validates the
detector against real texture and real inter-frame nuisances but not against
real radiometry -- the watt figures rest on a declared 20-45 C span across the
8-bit range, and they scale with that assumption. And neither arm reaches the
~0.023 W of the full simulated evaluation, because injection places its leak in
a deliberately quiet location and assumes uniform emissivity, so the emissivity
structure that makes the simulated task hard cancels in the difference. An
earlier version of this note quoted 0.0032 W / 0.0100 W and a 3.2x gap, measured
on a simulator no-leak pair standing in for a sequence; the real-sequence
figures above supersede it.

The practical consequence: **prefer `baseline_field`.** Given any sequence from
a fixed camera, use t=0 as the baseline and a later frame as the injection
background. The pair then carries real sensor drift, real scene change, real
registration error and real reflection movement -- none of it modelled -- while
ground truth stays exact because the leak was placed. If you must quote a
single-image number, quote it beside the two-frame one, or not at all.

A note on false-colour images
-----------------------------
Most thermal images you will find online are 8-bit false-colour renders. The
temperature is gone -- a colour map is not invertible without knowing the
palette and the scale bar, and even then only approximately. `load_as_field`
therefore maps image brightness onto a *declared* temperature range and says so.
For an injection study that is defensible, because injection and detection use
the same declared scale and the experiment stays self-consistent. It is not
defensible for measuring anything absolute, and this module will warn you.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from . import camera as cam
from . import physics as phys
from .simulate import ScenePair


@dataclass
class InjectionSpec:
    """Where and how hard to inject."""

    power_watts: float = 0.15
    centre_px: tuple[float, float] | None = None   # None -> automatic
    kind: str = "point"                            # "point" or "crack"
    source_sigma_px: float = 2.0
    crack_length_px: float = 18.0
    crack_angle_deg: float = 30.0


def load_as_field(path: str, temp_range: tuple[float, float] = (293.15, 318.15),
                  max_size: int = 512, verbose: bool = True) -> np.ndarray:
    """
    Read a frame and interpret it as a temperature field in kelvin.

    `.npy`, `.npz` and `.csv` are taken as **already being kelvin** and are used
    as-is -- that is what `scripts/convert_radiometric.py` writes, and remapping
    a real radiometric frame onto a declared brightness range would throw away
    the very measurement that made it worth capturing.

    For image files, greyscale conversion is deliberately naive: a false-colour
    thermal render has already lost the measurement, and pretending otherwise by
    inverting a guessed palette would add error while looking more rigorous.
    Brightness is mapped linearly onto `temp_range` and the caller is told.

    Large inputs are downscaled, because the solver cost grows with pixel count
    and nothing in the physics benefits from a 4000-pixel-wide input.
    """
    import cv2

    suffix = Path(path).suffix.lower()
    if suffix in (".npy", ".npz", ".csv"):
        if suffix == ".csv":
            field = np.loadtxt(path, delimiter=",", dtype=float)
        elif suffix == ".npz":
            data = np.load(path)
            field = data[list(data.keys())[0]].astype(float)
        else:
            field = np.load(path).astype(float)
        if field.ndim != 2:
            raise ValueError(f"{path}: expected a 2D frame, got shape {field.shape}")

        h, w = field.shape
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            field = cv2.resize(field, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA).astype(float)
        if verbose:
            print(f"[ok] {path} read as radiometric data, "
                  f"{field.min() - 273.15:.1f} to {field.max() - 273.15:.1f} C. "
                  f"Absolute temperatures are meaningful.")
        return field

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")

    radiometric = img.ndim == 2 and img.dtype in (np.uint16, np.float32, np.float64)

    if img.ndim == 3:
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
    img = img.astype(float)

    h, w = img.shape
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)

    lo, hi = float(img.min()), float(img.max())
    if hi - lo < 1e-9:
        raise ValueError(f"{path} has no contrast -- is it a blank image?")

    t_lo, t_hi = temp_range
    field = t_lo + (t_hi - t_lo) * (img - lo) / (hi - lo)

    if verbose and not radiometric:
        print(f"[note] {path} is not radiometric data. Brightness has been "
              f"mapped onto {t_lo - 273.15:.0f}-{t_hi - 273.15:.0f} C. "
              f"Detection results are self-consistent for an injection study, "
              f"but no absolute temperature or leak power from this run is "
              f"physically meaningful.")

    return field


def _pick_quiet_location(field: np.ndarray, margin: int = 24) -> tuple[float, float]:
    """
    Choose an injection site in a cool, unstructured part of the scene.

    Dropping a leak on top of the brightest existing hotspot would make the
    novelty test fire and produce a misleadingly hard case; dropping it at the
    frame edge truncates the radial profile the physics tests need. Pick the
    coolest interior region instead, which is also where a casing crack or a
    lagging breach would realistically show up.
    """
    ny, nx = field.shape
    if min(ny, nx) <= 2 * margin + 4:
        return (ny / 2.0, nx / 2.0)

    smooth = gaussian_filter(field, sigma=6.0, mode="nearest")
    interior = smooth[margin:ny - margin, margin:nx - margin]
    idx = int(np.argmin(interior))
    row, col = np.unravel_index(idx, interior.shape)
    return (float(row + margin), float(col + margin))


def inject_leak(field: np.ndarray,
                spec: InjectionSpec | None = None,
                material: phys.Material | str = "abs_casing",
                metres_per_pixel: float = 0.0025,
                cfg: cam.CameraConfig | None = None,
                seed: int = 0,
                add_misalignment: bool = True,
                add_reflection: bool = False,
                baseline_field: np.ndarray | None = None) -> ScenePair:
    """
    Build a labelled baseline / inspection pair from real image data.

    The inspection frame is `field` plus the steady-state temperature rise from
    a leak of `spec.power_watts`, rendered through the camera model with fresh
    noise -- and, by default, a small rigid misalignment, because two real
    captures are never perfectly aligned and a pipeline that is never asked to
    register is never tested.

    Parameters
    ----------
    field : the background the leak is injected into.
    baseline_field : the reference frame. **This is the argument that decides
        what the experiment measures.**

        Left as None, the baseline is `field` itself, and the two frames share
        an identical background by construction. Nothing drifted, no component
        changed workload, the emissivity map is the same map; only sensor noise,
        a small shift and the leak separate them. That is a much easier problem
        than the one the detector faces in the field, and it is why detection
        limits from this mode run optimistically low.

        Given a *different* real frame from the same sequence -- t=0 as the
        baseline, t=300s as the injection background -- the pair inherits the
        genuine inter-frame differences of that sequence: real sensor drift,
        real scene change, real registration error, real reflection movement.
        None of it is modelled, all of it is measured, and the ground truth is
        still exact because the leak was placed. That is most of the way to a
        real paired experiment, and it is the mode to prefer whenever a sequence
        is available.

    Returns a `ScenePair`, so everything in `evaluate.py` works on it unchanged.
    """
    spec = spec or InjectionSpec()
    if isinstance(material, str):
        material = phys.MATERIALS[material]
    cfg = cfg or cam.CameraConfig(downsample=1, seed=seed)

    rng = np.random.default_rng(seed)
    shape = field.shape

    paired_background = baseline_field is not None
    if baseline_field is None:
        baseline_field = field
    elif baseline_field.shape != shape:
        raise ValueError(
            f"baseline_field {baseline_field.shape} and field {shape} must have "
            f"the same shape -- they should be two frames of one sequence")

    centre = spec.centre_px or _pick_quiet_location(field)

    # Solve for the leak's contribution alone, on a uniform plate. Superposing
    # it on the real field is valid because steady-state conduction is linear:
    # the temperature rise from an added source does not depend on what other
    # sources are already present.
    conductivity = np.full(shape, material.conductivity, dtype=float)

    if spec.kind == "crack":
        angle = np.deg2rad(spec.crack_angle_deg)
        half = spec.crack_length_px / 2.0
        start = (centre[0] - half * np.sin(angle), centre[1] - half * np.cos(angle))
        end = (centre[0] + half * np.sin(angle), centre[1] + half * np.cos(angle))
        power_map = phys.crack_source(shape, start, end, spec.power_watts,
                                      width_px=2.0, dx=metres_per_pixel)
    else:
        power_map = phys.gaussian_source(shape, centre, spec.power_watts,
                                         sigma_px=spec.source_sigma_px,
                                         dx=metres_per_pixel)

    zero = np.zeros(shape, dtype=float)
    ambient = float(np.median(field))
    leak_rise = (phys.solve_steady_state(conductivity, power_map, material,
                                         metres_per_pixel, ambient)
                 - phys.solve_steady_state(conductivity, zero, material,
                                           metres_per_pixel, ambient))

    inspection_field = field + leak_rise

    # Emissivity is unknown for a real image; assume the material's nominal
    # value uniformly. This is a real limitation -- the emissivity structure
    # baked into the source image is already present in *both* frames and so
    # cancels in the difference, which makes injection studies slightly kinder
    # than reality on exactly the nuisance that matters most.
    emissivity = np.full(shape, material.emissivity, dtype=float)
    emis_insp = emissivity

    if add_misalignment:
        shift = (rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0))
        rotation = rng.uniform(-0.4, 0.4)
        inspection_field = cam.apply_misalignment(inspection_field, shift, rotation)
        centre = _shift_point(centre, shift, rotation, shape)

    reflection = None
    if add_reflection:
        reflection = cam.specular_reflection(
            shape,
            centre_px=(rng.uniform(0.2, 0.8) * shape[0],
                       rng.uniform(0.2, 0.8) * shape[1]),
            amplitude_k=rng.uniform(0.8, 2.5),
            radius_px=rng.uniform(0.02, 0.05) * max(shape),
            sharpness=8.0,
        )

    baseline = cam.capture(baseline_field, emissivity, cfg, rng)
    inspection = cam.capture(inspection_field, emis_insp, cfg, rng,
                             reflection=reflection,
                             gain=1.0 + rng.uniform(-0.003, 0.003),
                             offset=rng.uniform(-0.5, 0.5))

    f = int(cfg.downsample)
    scale = metres_per_pixel * f

    return ScenePair(
        baseline=baseline,
        inspection=inspection,
        has_leak=spec.power_watts > 0,
        leak_power_watts=float(spec.power_watts),
        leak_centre_px=(centre[0] / f, centre[1] / f) if spec.power_watts > 0 else None,
        leak_kind=spec.kind if spec.power_watts > 0 else "none",
        material=material,
        metres_per_pixel=scale,
        truth=dict(injected=True, peak_rise_k=float(leak_rise.max()),
                   ambient=ambient, paired_background=paired_background),
    )


def _shift_point(point: tuple[float, float], shift: tuple[float, float],
                 rotation_deg: float, shape: tuple[int, int]) -> tuple[float, float]:
    ny, nx = shape
    cy, cx = ny / 2.0, nx / 2.0
    theta = np.deg2rad(rotation_deg)
    y, x = point[0] - cy, point[1] - cx
    xr = x * np.cos(theta) + y * np.sin(theta)
    yr = -x * np.sin(theta) + y * np.cos(theta)
    return (yr + cy + shift[0], xr + cx + shift[1])


def sweep_power(field: np.ndarray, powers, material="abs_casing",
                metres_per_pixel: float = 0.0025, seed: int = 0,
                baseline_field: np.ndarray | None = None,
                add_misalignment: bool = True, repeats: int = 1, **kwargs):
    """
    Inject leaks of increasing power into the same real background.

    This is the measurement worth reporting from an injection study: the
    detection limit *against real clutter*, which is the number that says how
    the method would behave in the field.

    Pass `baseline_field` to sweep against a genuinely different reference frame
    from the same sequence. Running the sweep both ways on one sequence and
    quoting the two limits together is the honest way to report it -- the gap is
    the price of the shared-background shortcut, measured rather than asserted.

    `repeats` runs each power with independent noise and misalignment draws and
    returns a detection *rate*. With repeats=1 the reported limit is the lowest
    power that happened to succeed once, which near the threshold is a coin
    flip; the 50% limit quoted in `evaluate.py` is a rate, and comparing a
    single-shot number against it is not a comparison.
    """
    from .pipeline import detect_leak

    rows = []
    for i, power in enumerate(powers):
        hits, scores, physics, peak, dists = 0, [], [], 0.0, []
        for j in range(max(1, int(repeats))):
            spec = InjectionSpec(power_watts=float(power), **kwargs)
            pair = inject_leak(field, spec, material=material,
                               metres_per_pixel=metres_per_pixel,
                               seed=seed + i * 1000 + j,
                               add_misalignment=add_misalignment,
                               baseline_field=baseline_field)
            result = detect_leak(pair.baseline, pair.inspection,
                                 material=pair.material,
                                 metres_per_pixel=pair.metres_per_pixel)
            best = result.best()
            distance = float("inf")
            if best is not None and pair.leak_centre_px is not None:
                distance = float(np.hypot(best.centroid[0] - pair.leak_centre_px[0],
                                          best.centroid[1] - pair.leak_centre_px[1]))
            hits += bool(result.leak_detected and distance <= 8.0)
            scores.append(float(result.score))
            physics.append(best.verdict.physics_score
                           if best and best.verdict else float("nan"))
            dists.append(distance)
            peak = pair.truth["peak_rise_k"]

        n = max(1, int(repeats))
        rows.append(dict(
            power_watts=float(power),
            n_repeats=n,
            n_hits=int(hits),
            detect_rate=hits / n,
            # Kept so single-shot callers and older code read the same keys.
            detected=hits > 0,
            localised=hits > 0,
            distance_px=float(np.min(dists)),
            peak_rise_k=peak,
            # nanmean over an all-NaN list warns; an all-NaN list here just
            # means no candidate was extracted at any repeat, which is a
            # legitimate outcome at the bottom of a sweep, not a problem.
            physics_score=(float(np.nanmean(physics))
                           if np.any(np.isfinite(physics)) else float("nan")),
            score=float(np.mean(scores)),
        ))
    return rows
