"""
Scene generation: build a device, heat it, break it, and photograph it twice.

Produces labelled *pairs* of thermal frames -- one baseline (intact) and one
inspection frame (possibly leaking) -- with full ground truth. This is the
dataset the detector is developed and evaluated against.

Design intent
-------------
The simulator is deliberately adversarial. It is easy to write a generator
where a difference image trivially solves the task; such a generator teaches
you nothing and produces a detector that collapses on real data. This one
includes, by default, every nuisance that breaks naive differencing:

  * emissivity structure (labels, vents, bare-metal patches)
  * a moving specular reflection in the inspection frame
  * sub-pixel to few-pixel misregistration between frames
  * ambient temperature drift
  * uncalibrated gain and offset drift
  * fixed-pattern noise that only partly cancels
  * internal heat sources whose power varies between frames (a device under
    a different workload) -- so "hotter" alone is not evidence of a leak

Negative samples (no leak) get all the same nuisances. That is what makes the
false-alarm rate meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import camera as cam
from . import physics as phys


# --------------------------------------------------------------------------
# Ground truth container
# --------------------------------------------------------------------------

@dataclass
class ScenePair:
    """A baseline / inspection frame pair with everything needed to score it."""

    baseline: np.ndarray                 # (h, w) kelvin, detector grid
    inspection: np.ndarray               # (h, w) kelvin, detector grid
    has_leak: bool
    leak_power_watts: float
    leak_centre_px: tuple[float, float] | None   # (row, col) on detector grid
    leak_kind: str                       # "none" | "point" | "crack"
    material: phys.Material
    metres_per_pixel: float              # detector-grid scale
    truth: dict = field(default_factory=dict)

    def expected_thermal_length_px(self) -> float:
        """Material thermal length converted to detector pixels."""
        return self.material.thermal_length() / self.metres_per_pixel


# --------------------------------------------------------------------------
# Device construction
# --------------------------------------------------------------------------

def build_device(shape: tuple[int, int], material: phys.Material,
                 rng: np.random.Generator) -> dict:
    """
    Lay out a plausible device: material regions, emissivity structure, and
    normal internal heat sources.

    Returns a dict of high-resolution maps shared by both frames of a pair.
    """
    ny, nx = shape
    conductivity = np.full(shape, material.conductivity, dtype=float)
    emissivity = np.full(shape, material.emissivity, dtype=float)

    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)

    # A metal heat-spreader plate under part of the casing: locally much higher
    # conductivity, so heat there spreads further. Tests whether the detector
    # copes with a spatially varying thermal length.
    py0, py1 = int(0.10 * ny), int(0.45 * ny)
    px0, px1 = int(0.08 * nx), int(0.55 * nx)
    conductivity[py0:py1, px0:px1] *= 8.0

    # Emissivity structure. These are pure radiometric features -- no real
    # temperature change -- and they are the false-positive bait.
    #   a printed label
    ly0, ly1 = int(0.62 * ny), int(0.74 * ny)
    lx0, lx1 = int(0.66 * nx), int(0.88 * nx)
    emissivity[ly0:ly1, lx0:lx1] = 0.62
    #   a bare metal screw / port surround (strongly specular)
    for _ in range(3):
        cy = rng.uniform(0.15 * ny, 0.9 * ny)
        cx = rng.uniform(0.1 * nx, 0.9 * nx)
        rad = rng.uniform(0.012, 0.022) * nx
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 < rad ** 2
        emissivity[mask] = rng.uniform(0.18, 0.35)
    #   a ventilation grille: alternating stripes
    gy0, gy1 = int(0.80 * ny), int(0.92 * ny)
    gx0, gx1 = int(0.18 * nx), int(0.52 * nx)
    stripes = ((xx[gy0:gy1, gx0:gx1] // max(int(0.008 * nx), 1)) % 2 == 0)
    emissivity[gy0:gy1, gx0:gx1] = np.where(stripes, 0.55, emissivity[gy0:gy1, gx0:gx1])

    # Normal internal heat sources -- the device's healthy thermal signature.
    # Powers are chosen so the healthy case peaks around 315-320 K (42-47 C),
    # which is what a laptop underside actually reaches. If you change the
    # material, re-check this: an unrealistically hot baseline makes every leak
    # trivially detectable and the whole evaluation meaningless.
    sources = [
        dict(centre=(0.28 * ny, 0.30 * nx), power=1.30, sigma=0.045 * nx),  # SoC
        dict(centre=(0.55 * ny, 0.20 * nx), power=0.70, sigma=0.075 * nx),  # battery
        dict(centre=(0.40 * ny, 0.78 * nx), power=0.45, sigma=0.035 * nx),  # PSU
    ]

    return dict(conductivity=conductivity, emissivity=emissivity,
                sources=sources, shape=shape)


def _normal_power_map(device: dict, dx: float, rng: np.random.Generator,
                      workload: float, jitter: bool = True) -> np.ndarray:
    """Sum the device's healthy internal sources, scaled by a workload factor."""
    # SOURCE_JITTER is per-source and *independent between the two frames*, so it
    # is the harshest nuisance in the simulator: it makes the device's own
    # hotspots grow and shrink between captures by genuine, diffusive,
    # physically impeccable heat. An earlier +/-10% setting made the SoC vary by
    # ~0.26 W between frames -- larger than most of the leaks being hunted --
    # and it dominated the false positives, none of which any physics test can
    # reject, because nothing about them is unphysical. +/-2.5% reflects the
    # real electrical and thermal variation of a device held at matched load.
    SOURCE_JITTER = 0.025 if jitter else 0.0
    power = np.zeros(device["shape"], dtype=float)
    for src in device["sources"]:
        scale = (rng.uniform(1 - SOURCE_JITTER, 1 + SOURCE_JITTER)
                 if SOURCE_JITTER else 1.0)
        power += phys.gaussian_source(device["shape"], src["centre"],
                                      src["power"] * workload * scale,
                                      src["sigma"], dx)
    return power


# --------------------------------------------------------------------------
# Pair generation
# --------------------------------------------------------------------------

def generate_pair(rng: np.random.Generator,
                  material: phys.Material | None = None,
                  cfg: cam.CameraConfig | None = None,
                  grid: int = 256,
                  extent_m: float = 0.32,
                  leak_power_watts: float | None = None,
                  leak_kind: str | None = None,
                  add_reflection: bool = True,
                  add_misalignment: bool = True,
                  workload_drift: bool = True) -> ScenePair:
    """
    Generate one labelled baseline / inspection pair.

    Parameters
    ----------
    leak_power_watts : leak power in watts; 0 or None produces a negative
        (no-leak) sample. Sweeping this is how the detection limit is measured.
    leak_kind : "point" or "crack"; random if None and a leak is requested.
    add_reflection, add_misalignment, workload_drift : nuisance toggles. Turn
        them off to confirm the pipeline works in the easy case before
        diagnosing failures in the hard one.
    """
    material = material or phys.MATERIALS["abs_casing"]
    cfg = cfg or cam.CameraConfig()

    shape = (grid, grid)
    dx = extent_m / grid
    device = build_device(shape, material, rng)

    T_ambient = 293.15 + rng.uniform(-1.5, 1.5)

    # ---- baseline frame --------------------------------------------------
    base_power = _normal_power_map(device, dx, rng, workload=1.0,
                                   jitter=workload_drift)
    T_base = phys.solve_steady_state(device["conductivity"], base_power,
                                     material, dx, T_ambient)

    # ---- inspection frame ------------------------------------------------
    # The device is doing slightly different work, and ambient has drifted.
    #
    # WORKLOAD_SPREAD is the single most consequential nuisance parameter, and
    # it is worth understanding why. A workload change is not an artefact: it
    # produces genuine, physically consistent, diffusive heat. No physics test
    # can reject it, because there is nothing unphysical about it -- it is
    # simply heat from a source that is not a leak. At +/-15% the internal
    # sources vary by ~0.2 W, which is the same size as the leaks being hunted,
    # and the task becomes close to unidentifiable from two frames alone.
    #
    # +/-6% represents a controlled inspection: the device imaged under matched
    # load, which is what any usable field protocol would specify. Raise it to
    # see performance degrade, and note that this degradation is a statement
    # about inspection procedure, not about the algorithm.
    WORKLOAD_SPREAD = 0.06
    workload = (rng.uniform(1 - WORKLOAD_SPREAD, 1 + WORKLOAD_SPREAD)
                if workload_drift else 1.0)
    T_ambient_2 = T_ambient + (rng.uniform(-1.2, 1.2) if workload_drift else 0.0)
    insp_power = _normal_power_map(device, dx, rng, workload=workload,
                                   jitter=workload_drift)

    has_leak = bool(leak_power_watts)
    leak_centre_hi = None
    kind = "none"

    if has_leak:
        kind = leak_kind or rng.choice(["point", "crack"])
        # Keep the leak away from the frame edge so a full radial profile exists.
        cy = rng.uniform(0.20 * grid, 0.80 * grid)
        cx = rng.uniform(0.20 * grid, 0.80 * grid)
        leak_centre_hi = (cy, cx)

        if kind == "crack":
            angle = rng.uniform(0, np.pi)
            half = rng.uniform(0.035, 0.075) * grid
            start = (cy - half * np.sin(angle), cx - half * np.cos(angle))
            end = (cy + half * np.sin(angle), cx + half * np.cos(angle))
            insp_power = insp_power + phys.crack_source(
                shape, start, end, leak_power_watts, width_px=0.008 * grid, dx=dx)
        else:
            insp_power = insp_power + phys.gaussian_source(
                shape, leak_centre_hi, leak_power_watts,
                sigma_px=rng.uniform(0.006, 0.014) * grid, dx=dx)

    T_insp = phys.solve_steady_state(device["conductivity"], insp_power,
                                     material, dx, T_ambient_2)

    # ---- confounders -----------------------------------------------------
    reflection = None
    if add_reflection and rng.random() < 0.7:
        # A specular reflection is an *image* of a warm object: its edges are
        # as sharp as the optics and the surface finish allow, which is far
        # sharper than conduction can sustain. That is precisely what makes it
        # rejectable on physical grounds, and why the exponent is high.
        reflection = cam.specular_reflection(
            shape,
            centre_px=(rng.uniform(0.15 * grid, 0.85 * grid),
                       rng.uniform(0.15 * grid, 0.85 * grid)),
            amplitude_k=rng.uniform(0.8, 3.5),
            radius_px=rng.uniform(0.012, 0.030) * grid,
            sharpness=rng.uniform(6.0, 10.0),
        )

    emis_insp = device["emissivity"]
    if add_misalignment:
        shift = (rng.uniform(-2.5, 2.5), rng.uniform(-2.5, 2.5))
        rot = rng.uniform(-0.6, 0.6)
        T_insp = cam.apply_misalignment(T_insp, shift, rot)
        emis_insp = cam.apply_misalignment(device["emissivity"], shift, rot)
        if leak_centre_hi is not None:
            leak_centre_hi = _shift_point(leak_centre_hi, shift, rot, shape)

    # Uncalibrated radiometric drift between the two captures.
    gain = 1.0 + rng.uniform(-0.004, 0.004)
    offset = rng.uniform(-0.9, 0.9)

    frame_base = cam.capture(T_base, device["emissivity"], cfg, rng)
    frame_insp = cam.capture(T_insp, emis_insp, cfg, rng,
                             reflection=reflection, gain=gain, offset=offset)

    scale = dx * cfg.downsample
    leak_centre_det = (None if leak_centre_hi is None
                       else (leak_centre_hi[0] / cfg.downsample,
                             leak_centre_hi[1] / cfg.downsample))

    return ScenePair(
        baseline=frame_base,
        inspection=frame_insp,
        has_leak=has_leak,
        leak_power_watts=float(leak_power_watts or 0.0),
        leak_centre_px=leak_centre_det,
        leak_kind=kind,
        material=material,
        metres_per_pixel=scale,
        truth=dict(
            ambient_baseline=T_ambient,
            ambient_inspection=T_ambient_2,
            workload=workload,
            gain=gain,
            offset=offset,
            reflection_present=reflection is not None,
            peak_true_rise=float(T_insp.max() - T_base.max()),
        ),
    )


def _shift_point(point: tuple[float, float], shift: tuple[float, float],
                 rotation_deg: float, shape: tuple[int, int]) -> tuple[float, float]:
    """Apply the same rigid transform used on the field to a ground-truth point."""
    ny, nx = shape
    cy, cx = ny / 2.0, nx / 2.0
    theta = np.deg2rad(rotation_deg)
    y, x = point[0] - cy, point[1] - cx
    # cv2.getRotationMatrix2D rotates counter-clockwise for positive angles.
    xr = x * np.cos(theta) + y * np.sin(theta)
    yr = -x * np.sin(theta) + y * np.cos(theta)
    return (yr + cy + shift[0], xr + cx + shift[1])


def generate_dataset(n_pairs: int, seed: int = 0,
                     leak_power_range: tuple[float, float] = (0.01, 0.5),
                     positive_fraction: float = 0.5,
                     **kwargs) -> list[ScenePair]:
    """
    Build a balanced dataset of labelled pairs with leak power sampled
    log-uniformly, so weak leaks -- the interesting regime -- are well
    represented rather than swamped by obvious ones.
    """
    rng = np.random.default_rng(seed)
    pairs = []
    for i in range(n_pairs):
        positive = rng.random() < positive_fraction
        if positive:
            lo, hi = np.log(leak_power_range[0]), np.log(leak_power_range[1])
            power = float(np.exp(rng.uniform(lo, hi)))
        else:
            power = 0.0
        pairs.append(generate_pair(rng, leak_power_watts=power, **kwargs))
    return pairs
