"""
Tests for the photon-limited detector model.

`test_derived_netd_matches_published_ranges` is the important one, in the same
way `test_solver_matches_analytic_thermal_length` is in test_physics.py. The
model derives NETD from first principles; if that derivation drifts, every
detection limit measured downstream is quoted against a sensor that does not
exist, while still producing plausible-looking numbers.

The asserted ranges are detector *classes* from the literature, deliberately
loose -- wide enough that ordinary tuning will not break them, tight enough that
the error actually made while writing this module (hand-picked integration
times putting HgCdTe at 71 mK) fails.
"""

from __future__ import annotations

import numpy as np
import pytest

from thermalleak.detector import DETECTORS, Detector, band_contrast, photon_exitance


def test_photon_exitance_scales_near_t_cubed():
    """Total photon exitance goes as T^3; the exponent must come out of the integral."""
    lo = float(photon_exitance(280.0))
    hi = float(photon_exitance(320.0))
    exponent = np.log(hi / lo) / np.log(320.0 / 280.0)
    assert 3.0 < exponent < 6.0


def test_band_contrast_is_positive_and_small():
    """d(ln M)/dT near ambient is a fraction of a percent per kelvin."""
    assert 0.005 < band_contrast(300.0, (8e-6, 14e-6)) < 0.05


def test_derived_netd_matches_published_ranges():
    """
    Uncooled bolometers 50-500 mK (Yu et al. 2020); cooled LWIR photon
    detectors 10-40 mK typical; state-of-the-art arrays sub-5 mK; the
    background-limited floor well below that.
    """
    assert 0.030 <= DETECTORS["microbolometer"].netd_kelvin() <= 0.060
    assert 0.010 <= DETECTORS["hgcdte"].netd_kelvin() <= 0.040
    assert 0.010 <= DETECTORS["qwip"].netd_kelvin() <= 0.040
    assert 0.0015 <= DETECTORS["sota_lwir"].netd_kelvin() <= 0.005
    assert DETECTORS["blip"].netd_kelvin() < 0.002


def test_integration_is_well_limited_for_real_detectors():
    """
    Real LWIR detectors fill their wells long before the frame ends -- the fact
    that makes the band background-flooded. If this fails, the paper's argument
    about quantum efficiency needs revisiting, not just the test.
    """
    for key in ("hgcdte", "qwip", "sota_lwir"):
        det = DETECTORS[key]
        assert det.integration_time_s(300.0) < 1.0 / det.frame_rate_hz


def test_low_quantum_efficiency_is_compensated_by_longer_integration():
    """4.7x lower efficiency must not mean 4.7x worse: it integrates longer."""
    qwip, hgcdte = DETECTORS["qwip"], DETECTORS["hgcdte"]
    assert qwip.quantum_efficiency < 0.25 * hgcdte.quantum_efficiency
    assert qwip.netd_kelvin() < 2.0 * hgcdte.netd_kelvin()


def test_sota_beats_hgcdte_only_through_well_capacity():
    """
    sota_lwir differs from hgcdte in exactly one parameter. If someone later
    'improves' it by raising efficiency instead, this fails and the code stops
    supporting the paper's argument about background-flooded LWIR.
    """
    sota, base = DETECTORS["sota_lwir"], DETECTORS["hgcdte"]
    assert sota.quantum_efficiency == base.quantum_efficiency
    assert sota.f_number == base.f_number
    assert sota.well_capacity_e > 10 * base.well_capacity_e
    assert sota.netd_kelvin() < 0.5 * base.netd_kelvin()


def test_netd_improves_as_root_eta_when_time_limited():
    """When the frame, not the well, caps integration, NETD ~ 1/sqrt(eta) returns."""
    kw = dict(f_number=6.0, well_capacity_e=np.inf, read_noise_e=0.0, frame_rate_hz=1000.0)
    base = Detector(name="starved", quantum_efficiency=0.10, **kw)
    better = Detector(name="starved-4x", quantum_efficiency=0.40, **kw)
    assert base.netd_kelvin() / better.netd_kelvin() == pytest.approx(2.0, rel=0.05)


def test_poisson_noise_has_the_derived_sigma():
    """Sampled noise and advertised NETD come from different code paths and must agree."""
    det = DETECTORS["hgcdte"]
    noise = det.sample_noise(np.full((256, 256), 300.0), np.random.default_rng(0), 300.0)
    assert noise.std() == pytest.approx(det.netd_kelvin(300.0), rel=0.08)
    assert abs(noise.mean()) < 0.1 * det.netd_kelvin(300.0)


def test_noise_tracks_scene_brightness():
    """Shot noise is signal dependent -- the one thing a fixed-NETD model cannot express."""
    det = DETECTORS["hgcdte"]
    rng = np.random.default_rng(1)
    cold = det.sample_noise(np.full((400, 400), 290.0), rng, 300.0)
    hot = det.sample_noise(np.full((400, 400), 330.0), rng, 300.0)
    assert hot.std() > cold.std()


def test_thermal_detector_ignores_quantum_efficiency():
    """A bolometer does not count photons, so eta must not touch its noise."""
    a = DETECTORS["microbolometer"]
    b = Detector(name="bolo-2", kind="thermal", fixed_netd_k=a.fixed_netd_k,
                 quantum_efficiency=0.99)
    assert b.netd_kelvin() == a.netd_kelvin()
    assert not b.is_photon_limited()


def test_default_camera_path_is_unchanged():
    """
    With no detector attached, capture() must draw exactly the noise it always
    did. This is what keeps results/summary.txt reproducible digit for digit.
    """
    from thermalleak.camera import CameraConfig, capture

    field = 300.0 + np.random.default_rng(3).random((64, 64))
    emis = np.full((64, 64), 0.94)
    a = capture(field, emis, CameraConfig(seed=1), np.random.default_rng(9))
    b = capture(field, emis, CameraConfig(seed=1, detector=None), np.random.default_rng(9))
    assert np.array_equal(a, b)
