"""
Photon-limited detector model: where the noise floor actually comes from.

Why this module exists
----------------------
Everywhere else in this project, sensor noise was a single number --
`CameraConfig.netd_kelvin = 0.05`. That is a stand-in, and it hides the one
genuinely quantum process in the whole measurement chain.

A thermal camera does not measure temperature. It counts photoelectrons. The
scene emits photons at a rate set by Planck's law; a fraction eta of those that
land on a detector element are converted to electrons; and the number arriving
in an integration time is *Poisson distributed*, because photon arrival is a
quantum process and not a smooth flow. The noise floor of a good infrared
detector is the square root of that count.

So NETD is not a property you choose. It is a consequence of

    NETD  =  sigma_N / (dN/dT)  =  1 / ( sqrt(N) * dlnM/dT )

where N is the collected photoelectron count and M is the in-band photon
exitance of the scene. Written that way the scaling laws fall out immediately:

    NETD  proportional to  1/sqrt(eta)            quantum efficiency
                           1/sqrt(t_int)          integration time
                           1/sqrt(A_detector)     pixel area
                           F/#                    optics speed

This is the axis on which "quantum detectors" actually mean something. It is
also the reason the comparison is not a simple ladder:

    thermal detectors (microbolometers) do not count photons at all. They
    measure the temperature rise of an absorbing membrane, so their noise is
    thermal fluctuation and Johnson noise, and *eta does not appear*. You
    cannot improve an uncooled bolometer by improving its quantum efficiency,
    because it does not have one in this sense. That is the real dividing line
    between a thermal and a quantum detector, and it is why the presets below
    treat the bolometer as a fixed floor rather than a derived one.

The background-flooded result
-----------------------------
In the 8-14 um band at room temperature the scene delivers so many photons
that a real detector fills its charge well in tens to hundreds of
microseconds, long before a video frame ends. Integration time is therefore
set by *well capacity*, not chosen -- and a low-efficiency detector simply
integrates longer to reach the same count. Sensitivity in this regime is
bought with charge handling, not with quantum efficiency. Published arrays
confirm it: sub-5 mK QWIP arrays and 2 mK designs with pixel-level ADCs both
get there by enlarging the well. This model produced that behaviour before it
was checked against the literature, which is the best evidence it is right.

What this module does NOT claim
-------------------------------
Nothing here is sub-shot-noise. Quantum imaging with undetected photons does
beat the background limit at 6-10 um (Kornienko et al., arXiv:2608.20015,
2026), but it measures transmittance by never detecting the infrared photon,
so it cannot measure emitted radiance and cannot do passive thermography. The
`blip` preset is the *classical* photon limit and is the strongest honest
claim available. Say so plainly in any write-up rather than letting "quantum"
imply more than the physics delivers.

Preset sources
--------------
Uncooled microbolometer NETD 50-500 mK: Yu et al., Micromachines 11, 800
(2020). Cooled LWIR arrays reaching sub-5 mK: Infrared Physics & Technology
(2024). Survey context: Kohli, IJISRT 10(8) (2025). `tests/test_detector.py`
pins the derived NETD against those ranges, which is what makes the model
falsifiable rather than merely plausible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Physical constants (SI, CODATA).
PLANCK = 6.62607015e-34          # J s
LIGHT = 2.99792458e8             # m / s
BOLTZMANN = 1.380649e-23         # J / K


def photon_exitance(temperature: float | np.ndarray,
                    band_m: tuple[float, float] = (8e-6, 14e-6),
                    n_samples: int = 512) -> np.ndarray:
    """
    In-band photon exitance of a blackbody, in photons per second per m^2.

    Planck's law in photon units rather than energy units,

        M_q(lambda, T) = (2 pi c / lambda^4) / (exp(hc / lambda k T) - 1)

    integrated over the detector band. Photon units are the right currency: a
    detector counts photoelectrons, so shot noise is the square root of a
    *photon* count, not of an energy. Integrated on a log-spaced grid because
    the integrand spans several decades across a typical band.
    """
    t = np.asarray(temperature, dtype=float)
    lam = np.logspace(np.log10(band_m[0]), np.log10(band_m[1]), n_samples)

    lam_b = lam.reshape((1,) * t.ndim + (-1,))
    t_b = t[..., None]

    exponent = PLANCK * LIGHT / (lam_b * BOLTZMANN * np.clip(t_b, 1.0, None))
    denom = np.expm1(np.clip(exponent, 0.0, 700.0))
    spectral = (2.0 * np.pi * LIGHT / lam_b ** 4) / denom

    return np.trapezoid(spectral, lam, axis=-1)


def band_contrast(temperature: float, band_m: tuple[float, float],
                  delta: float = 0.01) -> float:
    """
    d(ln M_q)/dT by central difference -- the thermal contrast of the band.

    Converts a photon-count fluctuation into a temperature uncertainty. A
    property of the band and the scene temperature, not of the detector.
    """
    hi = float(photon_exitance(temperature + delta, band_m))
    lo = float(photon_exitance(temperature - delta, band_m))
    mid = float(photon_exitance(temperature, band_m))
    return (hi - lo) / (2.0 * delta * mid)


@dataclass(frozen=True)
class Detector:
    """
    A detector class, described by the parameters that set its noise floor.

    `kind` is load-bearing:
      "photon"  -- counts photoelectrons. NETD is derived, and the per-frame
                   noise is drawn from a Poisson distribution.
      "thermal" -- a bolometer. Does not count photons; `fixed_netd_k` is used
                   directly and quantum efficiency is meaningless.
    """

    name: str
    kind: str = "photon"
    quantum_efficiency: float = 0.70
    f_number: float = 2.0
    pixel_pitch_m: float = 25e-6
    fill_factor: float = 0.9
    transmission: float = 0.9
    read_noise_e: float = 300.0
    well_capacity_e: float = 8e6
    well_fill_fraction: float = 0.7
    frame_rate_hz: float = 30.0
    band_m: tuple[float, float] = (8e-6, 14e-6)
    fixed_netd_k: float = 0.05
    note: str = ""

    def photon_rate(self, scene_temperature_k: float) -> float:
        """
        Photoelectrons per second per element, for a scene filling the field:

            Phi = eta * tau * A_det * M_q(T) / (4 F^2 + 1)

        the denominator being the projected-solid-angle factor of the f-cone.
        """
        area = (self.pixel_pitch_m ** 2) * self.fill_factor
        return (self.quantum_efficiency * self.transmission * area
                * float(photon_exitance(scene_temperature_k, self.band_m))
                / (4.0 * self.f_number ** 2 + 1.0))

    def integration_time_s(self, scene_temperature_k: float) -> float:
        """
        Integration time, set by the hardware rather than chosen:

            t_int = min( fill * N_well / Phi ,  1 / frame_rate )

        An earlier version of this module picked microsecond integration times
        by hand and derived a HgCdTe NETD of 71 mK, about three times worse
        than any real cooled detector. The physics was right; the mistake was
        treating a constrained quantity as a free parameter.
        """
        rate = self.photon_rate(scene_temperature_k)
        if rate <= 0.0:
            return 1.0 / self.frame_rate_hz
        to_fill = self.well_fill_fraction * self.well_capacity_e / rate
        return min(to_fill, 1.0 / self.frame_rate_hz)

    def collected_electrons(self, scene_temperature_k: float) -> float:
        """Mean photoelectrons per element per frame."""
        return (self.photon_rate(scene_temperature_k)
                * self.integration_time_s(scene_temperature_k))

    def netd_kelvin(self, scene_temperature_k: float = 300.0) -> float:
        """NETD, derived for photon detectors and quoted for thermal ones."""
        if self.kind == "thermal":
            return float(self.fixed_netd_k)
        n = self.collected_electrons(scene_temperature_k)
        if n <= 0.0:
            return float("inf")
        noise_e = np.sqrt(n + self.read_noise_e ** 2)
        return float(noise_e / (n * band_contrast(scene_temperature_k, self.band_m)))

    def is_photon_limited(self, scene_temperature_k: float = 300.0) -> bool:
        """True when shot noise dominates readout noise -- the BLIP condition."""
        if self.kind != "photon":
            return False
        return self.collected_electrons(scene_temperature_k) > self.read_noise_e ** 2

    def sample_noise(self, frame_k: np.ndarray, rng: np.random.Generator,
                     scene_temperature_k: float | None = None) -> np.ndarray:
        """
        Draw one frame of detector noise, in kelvin.

        For a photon detector the draw is Poisson on the electron count, then
        converted back to temperature. Unlike Gaussian noise at a fixed sigma,
        the variance tracks local scene brightness: a warm region is noisier in
        absolute terms and quieter in relative terms. Above ~1e6 electrons the
        Gaussian approximation is used, where the two are indistinguishable and
        `rng.poisson` becomes slow.
        """
        if self.kind == "thermal":
            return rng.normal(0.0, self.fixed_netd_k, size=frame_k.shape)

        t_ref = (scene_temperature_k if scene_temperature_k is not None
                 else float(np.median(frame_k)))
        contrast = band_contrast(t_ref, self.band_m)

        # First-order expansion of the exitance about the reference; accurate
        # to well under a percent over the <10 K span of a real scene.
        n_ref = self.collected_electrons(t_ref)
        n_pix = np.clip(n_ref * (1.0 + contrast * (frame_k - t_ref)),
                        1.0, n_ref * 4.0)

        if n_ref > 1e6:
            electrons = rng.normal(n_pix, np.sqrt(n_pix))
        else:
            electrons = rng.poisson(n_pix).astype(float)

        if self.read_noise_e > 0:
            electrons = electrons + rng.normal(0.0, self.read_noise_e,
                                               size=frame_k.shape)

        return (electrons - n_pix) / (n_ref * contrast)


# ---------------------------------------------------------------------------
# Detector classes spanning the real hardware range and the physical floor.
# ---------------------------------------------------------------------------

DETECTORS: dict[str, Detector] = {
    "microbolometer": Detector(
        name="Uncooled VOx microbolometer",
        kind="thermal",
        fixed_netd_k=0.050,
        pixel_pitch_m=17e-6,
        f_number=1.0,
        note="Thermal detector: no photon counting, so no quantum efficiency. "
             "50 mK is the good end of the 50-500 mK range for uncooled "
             "arrays (Yu et al. 2020). The baseline used throughout.",
    ),
    "qwip": Detector(
        name="QWIP (cooled, LWIR)",
        kind="photon",
        quantum_efficiency=0.15,
        f_number=2.0,
        well_capacity_e=1.1e7,
        read_noise_e=500.0,
        note="Quantum well photodetector: low quantum efficiency from "
             "intersubband absorption, but in a background-flooded band that "
             "costs integration time, not sensitivity.",
    ),
    "hgcdte": Detector(
        name="HgCdTe (cooled, LWIR)",
        kind="photon",
        quantum_efficiency=0.70,
        f_number=2.0,
        well_capacity_e=8e6,
        read_noise_e=300.0,
        note="Mercury cadmium telluride, the workhorse cooled quantum detector.",
    ),
    "sota_lwir": Detector(
        name="Digital-pixel LWIR array (state of the art)",
        kind="photon",
        quantum_efficiency=0.70,
        f_number=2.0,
        well_capacity_e=7e8,
        read_noise_e=300.0,
        note="Identical to `hgcdte` except for charge handling -- which is the "
             "point. Published arrays reach sub-5 mK and 2 mK by enlarging "
             "well capacity, not efficiency. The best detector you can buy, "
             "as opposed to `blip`, the best physics allows.",
    ),
    "blip": Detector(
        name="BLIP limit (ideal photon detector)",
        kind="photon",
        quantum_efficiency=1.0,
        f_number=1.0,
        transmission=1.0,
        read_noise_e=0.0,
        well_capacity_e=np.inf,
        note="Background-limited floor: perfect efficiency, no read noise, no "
             "charge ceiling, integration over the full frame. Beating it needs "
             "quantum correlations, which cannot do passive thermography.",
    ),
}
