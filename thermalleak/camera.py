"""
Forward camera model: true temperature field -> what the camera actually records.

This is the chain the detector has to invert, and every stage of it is a place
where a naive difference-image method goes wrong:

    true temperature T(x, y)
      -> apparent temperature       (emissivity < 1, reflected background)
      -> reflection artefacts       (specular hotspots that are not heat)
      -> optical blur               (point spread function)
      -> sampling                   (detector pitch)
      -> noise                      (NETD: shot + readout, plus fixed pattern)
      -> gain / offset drift        (uncalibrated between frames)
      -> quantisation               (16-bit radiometric output)

The emissivity stage is the one most applied-thermography papers get wrong.
A camera does not measure temperature; it measures radiance. For a surface of
emissivity eps in front of a background at T_refl:

    sigma * T_app^4 = eps * sigma * T_true^4 + (1 - eps) * sigma * T_refl^4

so a low-emissivity region reads *colder* than it is, and a shiny region mostly
shows you a reflection of the room. A sticker on a laptop lid is a step change
in apparent temperature with no step change in real temperature -- physically
impossible for conduction, and therefore exactly what the physics score in
`score.py` is built to reject.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter, zoom


@dataclass
class CameraConfig:
    """Sensor and optics parameters for a synthetic thermal camera."""

    psf_sigma_px: float = 1.1       # optical blur, in high-res pixels
    downsample: int = 2             # high-res grid -> detector pitch
    netd_kelvin: float = 0.05       # noise equivalent temperature difference
    fpn_kelvin: float = 0.08        # fixed-pattern (per-pixel) non-uniformity
    fpn_drift: float = 0.15         # fraction of FPN that changes between frames
    quantisation_kelvin: float = 0.01   # 16-bit radiometric step
    reflected_temperature: float = 295.15   # T_refl for the emissivity model
    seed: int | None = None

    # Optional physical detector model. When None -- the default, and what
    # every recorded result in this repository was measured with -- temporal
    # noise is drawn as Gaussian at `netd_kelvin`, exactly as before.
    #
    # When set to a `detector.Detector`, `netd_kelvin` is ignored and the noise
    # is derived instead: Planck photon flux through the optics, converted to
    # photoelectrons by the detector's quantum efficiency, drawn from a Poisson
    # distribution. Opt-in deliberately, so the ablation in results/ stays
    # reproducible bit for bit.
    detector: "object | None" = None

    _fpn_cache: dict = field(default_factory=dict, repr=False)


def apparent_temperature(true_temperature: np.ndarray, emissivity: np.ndarray,
                         reflected_temperature: float) -> np.ndarray:
    """
    Convert true surface temperature to the apparent temperature a camera reads.

    Radiance balance in the fourth power of temperature -- the nonlinearity is
    mild over a 30 K span but not negligible, and getting it right is what
    lets the pipeline report answers in kelvin rather than in arbitrary units.
    """
    t4 = np.clip(true_temperature, 1.0, None) ** 4
    r4 = float(reflected_temperature) ** 4
    apparent4 = emissivity * t4 + (1.0 - emissivity) * r4
    return np.clip(apparent4, 1.0, None) ** 0.25


def _fixed_pattern_noise(cfg: CameraConfig, shape: tuple[int, int]) -> np.ndarray:
    """
    Per-pixel offset that is *static* for a given camera.

    This matters: because FPN is identical in both frames it largely cancels in
    a difference image. Only the drifting fraction survives. Modelling that
    correctly stops the simulator from being unrealistically kind or
    unrealistically harsh to the detector.
    """
    key = ("fpn", shape)
    if key not in cfg._fpn_cache:
        rng = np.random.default_rng(0 if cfg.seed is None else cfg.seed)
        cfg._fpn_cache[key] = rng.normal(0.0, cfg.fpn_kelvin, size=shape)
    return cfg._fpn_cache[key]


def capture(true_temperature: np.ndarray, emissivity: np.ndarray,
            cfg: CameraConfig, rng: np.random.Generator,
            reflection: np.ndarray | None = None,
            gain: float = 1.0, offset: float = 0.0) -> np.ndarray:
    """
    Render one thermal frame.

    Parameters
    ----------
    true_temperature : (ny, nx) kelvin, on the high-resolution physics grid.
    emissivity       : (ny, nx) surface emissivity map, same grid.
    cfg              : camera parameters.
    rng              : random generator (pass the same one through a sequence
                       of frames so noise is independent but reproducible).
    reflection       : optional additive apparent-temperature field in kelvin,
                       applied *after* the emissivity stage. Represents specular
                       reflections of warm objects -- not real surface heat.
    gain, offset     : uncalibrated radiometric drift between frames.

    Returns
    -------
    (ny // downsample, nx // downsample) frame in kelvin.
    """
    frame = apparent_temperature(true_temperature, emissivity,
                                 cfg.reflected_temperature)

    if reflection is not None:
        frame = frame + reflection

    # Optics: point spread function.
    if cfg.psf_sigma_px > 0:
        frame = gaussian_filter(frame, sigma=cfg.psf_sigma_px, mode="nearest")

    # Detector sampling. Area-average rather than point-sample, which is what a
    # real detector element does over its active area.
    f = int(cfg.downsample)
    if f > 1:
        ny, nx = frame.shape
        frame = frame[: ny - ny % f, : nx - nx % f]
        frame = frame.reshape(frame.shape[0] // f, f, frame.shape[1] // f, f).mean(axis=(1, 3))

    # Radiometric drift: gain and offset the pipeline must estimate away.
    frame = gain * frame + offset

    # Noise: temporal (independent per frame) plus fixed pattern (mostly static).
    # With a detector model attached, temporal noise comes from the photon
    # statistics of the scene -- signal dependent, because a warmer region
    # delivers more photons. This sits after downsampling deliberately: the
    # noise belongs to the detector element, not to the physics grid.
    if cfg.detector is not None:
        frame = frame + cfg.detector.sample_noise(frame, rng)
    else:
        frame = frame + rng.normal(0.0, cfg.netd_kelvin, size=frame.shape)
    fpn = _fixed_pattern_noise(cfg, frame.shape)
    drift = rng.normal(0.0, cfg.fpn_kelvin * cfg.fpn_drift, size=frame.shape)
    frame = frame + fpn + drift

    # Quantisation.
    if cfg.quantisation_kelvin > 0:
        frame = np.round(frame / cfg.quantisation_kelvin) * cfg.quantisation_kelvin

    return frame


def apply_misalignment(field: np.ndarray, shift_px: tuple[float, float],
                       rotation_deg: float) -> np.ndarray:
    """
    Rigidly move the scene, simulating a camera or object that shifted between
    the baseline and inspection frames.

    Misregistration is the dominant source of false alarms in real change
    detection: a one-pixel shift across a sharp emissivity edge creates a
    difference signal far larger than a genuine small leak. The pipeline must
    register before differencing, and the simulator has to make it earn that.
    """
    import cv2

    ny, nx = field.shape
    centre = (nx / 2.0, ny / 2.0)
    M = cv2.getRotationMatrix2D(centre, rotation_deg, 1.0)
    M[0, 2] += shift_px[1]
    M[1, 2] += shift_px[0]
    return cv2.warpAffine(field.astype(np.float32), M, (nx, ny),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE).astype(float)


def specular_reflection(shape: tuple[int, int], centre_px: tuple[float, float],
                        amplitude_k: float, radius_px: float,
                        sharpness: float = 6.0) -> np.ndarray:
    """
    A moving specular reflection: a warm object reflected off the surface.

    This is the detector's hardest confounder, and it is deliberately built to
    *look* like a hotspot while being physically impossible as one. The
    super-Gaussian exponent gives it a much flatter top and much sharper
    shoulder than diffusion permits -- an apparent thermal length far below
    what the material could sustain.
    """
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    cy, cx = centre_px
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return amplitude_k * np.exp(-((r / max(radius_px, 1e-6)) ** sharpness))
