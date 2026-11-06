"""
Tests. The first one is the important one.

`test_solver_matches_analytic_thermal_length` checks the numerical solver
against the closed-form point-source solution. If that fails, every physics test
downstream is measuring the wrong length scale and the whole detector is
silently miscalibrated -- while still producing plausible-looking output, which
is the dangerous kind of wrong. Run it after any change to the discretisation.
"""

from __future__ import annotations

import numpy as np
import pytest

from thermalleak import detect_leak, generate_pair
from thermalleak.normalize import robust_affine_fit
from thermalleak.physics import (
    MATERIALS, gaussian_source, point_source_profile, solve_steady_state,
)


def test_solver_matches_analytic_thermal_length():
    """
    Numerical steady state must reproduce L = sqrt(k*d / 2h).

    Measured by fitting ln(dT * sqrt(r)) against r, which is linear with slope
    -1/L because K0(x) ~ sqrt(pi/2x) * exp(-x) for large x.
    """
    material = MATERIALS["abs_casing"]
    n, extent = 512, 0.64          # domain >> L so the insulated edges do not bite
    dx = extent / n

    conductivity = np.full((n, n), material.conductivity)
    power = gaussian_source((n, n), (n / 2, n / 2), 1.0, sigma_px=1.0, dx=dx)
    temperature = solve_steady_state(conductivity, power, material, dx, 293.15)
    rise = temperature - 293.15

    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(yy - n / 2, xx - n / 2) * dx
    sel = (r > 0.010) & (r < 0.045)
    slope, _ = np.polyfit(r[sel], np.log(rise[sel] * np.sqrt(r[sel])), 1)

    measured = -1.0 / slope
    assert measured == pytest.approx(material.thermal_length(), rel=0.05)


def test_analytic_profile_decays():
    material = MATERIALS["abs_casing"]
    L = material.thermal_length()
    r = np.array([0.2 * L, L, 3 * L])
    dT = point_source_profile(r, power=1.0, material=material)
    assert dT[0] > dT[1] > dT[2] > 0


def test_thermal_length_ordering():
    """Aluminium spreads heat much further than a polymer casing."""
    assert (MATERIALS["aluminium"].thermal_length()
            > MATERIALS["steel_pipe"].thermal_length()
            > MATERIALS["abs_casing"].thermal_length())


def test_robust_fit_ignores_outliers():
    """
    The normalisation fit must not be dragged by the leak it is meant to expose.

    A plain least-squares fit absorbs part of a bright anomaly into the gain,
    partially erasing the very signal being looked for.
    """
    rng = np.random.default_rng(0)
    base = rng.normal(300.0, 2.0, size=(64, 64))
    insp = 1.01 * base + 0.5
    insp[20:28, 20:28] += 25.0          # a large, bright anomaly

    gain, offset, inlier, _ = robust_affine_fit(base, insp)
    assert gain == pytest.approx(1.01, abs=0.01)
    assert offset == pytest.approx(0.5, abs=0.3)
    assert not inlier[24, 24]           # the anomaly is treated as an outlier


def test_detects_and_localises_a_clear_leak():
    rng = np.random.default_rng(4)
    pair = generate_pair(rng, leak_power_watts=0.4, leak_kind="point")
    result = detect_leak(pair.baseline, pair.inspection,
                         material=pair.material,
                         metres_per_pixel=pair.metres_per_pixel)

    assert result.leak_detected
    best = result.best()
    distance = np.hypot(best.centroid[0] - pair.leak_centre_px[0],
                        best.centroid[1] - pair.leak_centre_px[1])
    assert distance < 8.0
    assert best.verdict.physics_score > 0.5


def test_clean_pair_is_not_flagged():
    """No leak, no nuisances: the detector must stay quiet."""
    rng = np.random.default_rng(5)
    pair = generate_pair(rng, leak_power_watts=0.0, add_reflection=False,
                         add_misalignment=False, workload_drift=False)
    result = detect_leak(pair.baseline, pair.inspection,
                         material=pair.material,
                         metres_per_pixel=pair.metres_per_pixel)
    assert not result.leak_detected


def test_frames_must_match_in_shape():
    with pytest.raises(ValueError):
        detect_leak(np.zeros((32, 32)), np.zeros((16, 16)))


def test_unknown_ordering_identifies_the_leaking_frame():
    """
    With the arguments swapped and no baseline declared, the leaking frame must
    still usually be named.

    Asserted as an aggregate rate rather than per-case. Unknown ordering is
    genuinely ambiguous when an artefact in one direction scores close to the
    leak in the other, so a single hand-picked seed would either be cherry-
    picked or flaky. The measured rate is 39/40; the threshold is set below that
    so ordinary variation does not break the build, but a real regression will.
    """
    correct = detected = 0
    for seed in range(20):
        rng = np.random.default_rng(900 + seed)
        pair = generate_pair(rng, leak_power_watts=0.25, leak_kind="point")
        # Pass the leaky frame first, so the answer should be "a".
        result = detect_leak(pair.inspection, pair.baseline,
                             material=pair.material,
                             metres_per_pixel=pair.metres_per_pixel,
                             known_baseline=None)
        if result.leak_detected:
            detected += 1
            correct += result.leaking_frame == "a"

    assert detected >= 18, f"only {detected}/20 leaks detected at 0.25 W"
    assert correct / detected >= 0.85, (
        f"correct frame named in only {correct}/{detected} cases")


def test_declared_baseline_is_unambiguous():
    """When the caller says which frame is the reference, there is no ambiguity."""
    rng = np.random.default_rng(6)
    pair = generate_pair(rng, leak_power_watts=0.4, leak_kind="point")
    result = detect_leak(pair.inspection, pair.baseline,
                         material=pair.material,
                         metres_per_pixel=pair.metres_per_pixel,
                         known_baseline="b")
    assert result.leak_detected
    assert result.leaking_frame == "a"


def test_injection_roundtrip_finds_the_injected_leak():
    """
    A leak injected into a real image at a known place must be found there.

    Uses a synthetic stand-in for a downloaded photo, but exercises the real
    code path: 8-bit input, brightness mapped to a declared temperature range,
    leak solved and superposed, both frames rendered through the camera.
    """
    from thermalleak.inject import InjectionSpec, inject_leak

    rng = np.random.default_rng(3)
    background = 293.15 + 12.0 * rng.random((160, 160))
    background = np.clip(background, 290.0, None)

    spec = InjectionSpec(power_watts=0.25, centre_px=(80.0, 80.0), kind="point")
    pair = inject_leak(background, spec, material="abs_casing",
                       metres_per_pixel=0.0025, seed=1)

    assert pair.has_leak
    assert pair.truth["peak_rise_k"] > 1.0

    result = detect_leak(pair.baseline, pair.inspection,
                         material=pair.material,
                         metres_per_pixel=pair.metres_per_pixel)
    assert result.leak_detected
    best = result.best()
    distance = np.hypot(best.centroid[0] - pair.leak_centre_px[0],
                        best.centroid[1] - pair.leak_centre_px[1])
    assert distance < 8.0


def test_injecting_zero_power_changes_nothing_detectable():
    """A 0 W injection must not produce a detection -- the null control."""
    from thermalleak.inject import InjectionSpec, inject_leak

    rng = np.random.default_rng(4)
    background = 293.15 + 8.0 * rng.random((160, 160))

    pair = inject_leak(background, InjectionSpec(power_watts=0.0),
                       material="abs_casing", metres_per_pixel=0.0025,
                       seed=2, add_misalignment=False)
    result = detect_leak(pair.baseline, pair.inspection,
                         material=pair.material,
                         metres_per_pixel=pair.metres_per_pixel)
    assert not result.leak_detected


def test_injection_accepts_two_different_background_frames():
    """
    The baseline and the injection background may be different frames.

    This is the mode that matters: with one frame used twice the two sides share
    an identical background and the task is structurally easier than reality.
    Passing a genuinely different reference must change the baseline that comes
    out, and must not disturb ground truth.
    """
    from thermalleak.inject import InjectionSpec, inject_leak

    rng = np.random.default_rng(11)
    first = 293.15 + 10.0 * rng.random((128, 128))
    second = first + rng.normal(0.0, 0.4, first.shape)      # the scene drifted

    spec = InjectionSpec(power_watts=0.3, centre_px=(64.0, 64.0), kind="point")
    shared = inject_leak(second, spec, metres_per_pixel=0.0025, seed=5,
                         add_misalignment=False)
    paired = inject_leak(second, spec, metres_per_pixel=0.0025, seed=5,
                         add_misalignment=False, baseline_field=first)

    assert shared.truth["paired_background"] is False
    assert paired.truth["paired_background"] is True

    # Same leak, same inspection frame; only the reference differs.
    assert paired.leak_centre_px == shared.leak_centre_px
    assert np.allclose(paired.inspection, shared.inspection)
    assert not np.allclose(paired.baseline, shared.baseline)

    # And the harder pair is still solvable, since the leak is real.
    result = detect_leak(paired.baseline, paired.inspection,
                         material=paired.material,
                         metres_per_pixel=paired.metres_per_pixel)
    assert result.leak_detected


def test_injection_rejects_mismatched_background_shapes():
    from thermalleak.inject import inject_leak

    with pytest.raises(ValueError):
        inject_leak(np.full((64, 64), 300.0),
                    baseline_field=np.full((32, 32), 300.0))


def test_radiometric_arrays_load_without_brightness_remapping(tmp_path):
    """
    `.npy` input is already kelvin and must be used as-is.

    Remapping a real radiometric frame onto a declared brightness range would
    discard the measurement that made it worth capturing -- and this is the
    format `scripts/convert_radiometric.py` writes, so the injection path has to
    read it faithfully.
    """
    from thermalleak.inject import load_as_field

    field = 293.15 + 12.0 * np.random.default_rng(2).random((64, 64))
    path = tmp_path / "frame.npy"
    np.save(path, field)

    loaded = load_as_field(str(path), temp_range=(200.0, 400.0), verbose=False)
    assert np.allclose(loaded, field)
