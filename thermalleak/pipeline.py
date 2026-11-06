"""
End-to-end leak detection: two thermal frames in, a verdict out.

    register  ->  normalise  ->  difference  ->  extract  ->  physics-score

The `use_physics` flag exists specifically to support the ablation in
`evaluate.py`. Being able to switch the physics off and measure what breaks is
the experiment that justifies the whole approach -- and running it early is how
you find out in week two, rather than in peer review, whether the physics term
is actually doing any work.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .detect import Candidate, extract_candidates
from .normalize import Normalisation, normalise
from .physics import MATERIALS, Material
from .register import Registration, register
from .score import PhysicsVerdict, score_candidate


@dataclass
class Finding:
    """One candidate with its physical assessment."""

    candidate: Candidate
    verdict: PhysicsVerdict | None
    combined_score: float

    @property
    def centroid(self) -> tuple[float, float]:
        return self.candidate.centroid


@dataclass
class DetectionResult:
    """Everything the pipeline concluded about one pair of frames."""

    leak_detected: bool
    # Which *input argument* holds the leak: "a" (frame_a), "b" (frame_b), or
    # "neither". Naming these by argument position rather than by role avoids
    # the ambiguity that arises when the caller does not know which frame is the
    # intact one -- which is precisely the case `known_baseline=None` handles.
    leaking_frame: str
    score: float                         # scalar detection statistic
    findings: list[Finding]
    difference: np.ndarray = field(repr=False)
    normalisation: Normalisation = field(repr=False)
    registration: Registration = field(repr=False)
    thermal_length_px: float = 0.0
    used_physics: bool = True

    def best(self) -> Finding | None:
        return self.findings[0] if self.findings else None

    def summary(self) -> str:
        """One-paragraph plain-language report."""
        if not self.leak_detected:
            return (f"No thermal leak detected (score {self.score:.2f}). "
                    f"Frames aligned by {self.registration.method} "
                    f"(shift {self.registration.shift_magnitude_px():.2f} px); "
                    f"residual noise {self.normalisation.noise_sigma * 1000:.0f} mK.")
        f = self.best()
        assert f is not None
        row, col = f.centroid
        name = {"a": "first", "b": "second"}.get(self.leaking_frame, self.leaking_frame)
        parts = [
            f"Thermal leak detected in the {name} frame "
            f"at pixel (row {row:.0f}, col {col:.0f}).",
            f"Peak excess temperature {f.candidate.peak_delta_k:.2f} K "
            f"({f.candidate.z_peak:.1f} sigma above background).",
        ]
        if f.verdict is not None:
            v = f.verdict
            parts.append(
                f"Physical consistency {v.physics_score:.2f} "
                f"(profile {v.fit_score:.2f}, length {v.length_score:.2f}, "
                f"sharpness {v.sharpness_score:.2f}, "
                f"novelty {v.novelty_score:.2f}); geometry {v.geometry}; "
                f"fitted thermal length {v.fitted_length_px:.1f} px versus "
                f"{v.expected_length_px:.1f} px expected."
            )
            parts.append(f"Estimated leak power {v.estimated_power_w:.2f} W "
                         f"(lower bound for extended sources).")
        return " ".join(parts)


def _analyse_one_direction(baseline: np.ndarray, inspection: np.ndarray,
                           material: Material, metres_per_pixel: float,
                           use_physics: bool, z_threshold: float,
                           psf_sigma_px: float, polarity: str,
                           physics_threshold: float,
                           motion: str) -> DetectionResult:
    reg = register(baseline, inspection, motion=motion)
    norm = normalise(baseline, reg.aligned, valid=reg.valid_mask)

    thermal_length_px = material.thermal_length() / metres_per_pixel

    candidates = extract_candidates(
        norm.difference, norm.noise_sigma, thermal_length_px,
        valid=reg.valid_mask, z_threshold=z_threshold, polarity=polarity)

    findings: list[Finding] = []
    for cand in candidates:
        # Statistical significance, compressed onto (0, 1).
        #
        # Raw z is a poor ranking statistic here. Detector noise is ~50 mK while
        # scene nuisances -- drift, reflections, emissivity edges -- are whole
        # kelvins, so *everything* interesting is tens or hundreds of sigma above
        # the noise floor and z spans three orders of magnitude. Past the point
        # where a feature is clearly not noise, more sigma does not make it more
        # likely to be a leak. Saturating z keeps it as a floor on significance
        # while letting physical consistency decide the ranking.
        significance = cand.z_peak / (cand.z_peak + z_threshold)

        if use_physics:
            verdict = score_candidate(cand, norm.difference, material,
                                      metres_per_pixel, psf_sigma_px=psf_sigma_px,
                                      baseline=baseline)
            combined = significance * verdict.physics_score
        else:
            verdict = None
            combined = significance
        findings.append(Finding(candidate=cand, verdict=verdict,
                                combined_score=float(combined)))

    findings.sort(key=lambda f: f.combined_score, reverse=True)
    score = findings[0].combined_score if findings else 0.0

    if use_physics:
        best = findings[0] if findings else None
        detected = bool(
            best is not None
            and best.candidate.z_peak >= z_threshold
            and best.verdict is not None
            and best.verdict.physics_score >= physics_threshold
        )
    else:
        detected = bool(findings) and findings[0].candidate.z_peak >= z_threshold

    return DetectionResult(
        leak_detected=detected,
        leaking_frame="b" if detected else "neither",
        score=float(score),
        findings=findings,
        difference=norm.difference,
        normalisation=norm,
        registration=reg,
        thermal_length_px=thermal_length_px,
        used_physics=use_physics,
    )


def detect_leak(frame_a: np.ndarray, frame_b: np.ndarray,
                material: Material | str = "abs_casing",
                metres_per_pixel: float = 0.0025,
                known_baseline: str | None = "a",
                use_physics: bool = True,
                z_threshold: float = 3.5,
                physics_threshold: float = 0.45,
                psf_sigma_px: float = 1.1,
                polarity: str = "hot",
                motion: str = "euclidean") -> DetectionResult:
    """
    Compare two thermal frames and decide whether one contains a leak.

    Parameters
    ----------
    frame_a, frame_b : (h, w) arrays of temperature in kelvin. If your camera
        exports 16-bit counts, convert to kelvin first -- every threshold in
        this pipeline is expressed in physical units, which is what makes them
        transferable between cameras.
    material : a `Material` or a key of `physics.MATERIALS`. Sets the expected
        thermal length, which drives the matched filter and the physics tests.
    metres_per_pixel : ground sample distance at the object. Measure it from a
        ruler in frame; guessing this is the most common way to make the
        physics tests silently wrong.
    known_baseline : "a" or "b" if you know which frame is the intact
        reference; None to let the pipeline decide by running both directions.
    use_physics : set False to ablate the physics stage.
    polarity : "hot", "cold" (cryogenic or coolant leaks), or "both".

    Returns
    -------
    DetectionResult
    """
    if isinstance(material, str):
        material = MATERIALS[material]

    frame_a = np.asarray(frame_a, dtype=float)
    frame_b = np.asarray(frame_b, dtype=float)
    if frame_a.shape != frame_b.shape:
        raise ValueError(f"frames must have the same shape, got "
                         f"{frame_a.shape} and {frame_b.shape}")

    kwargs = dict(material=material, metres_per_pixel=metres_per_pixel,
                  use_physics=use_physics, z_threshold=z_threshold,
                  psf_sigma_px=psf_sigma_px, polarity=polarity,
                  physics_threshold=physics_threshold, motion=motion)

    # `_analyse_one_direction(reference, test)` looks for anomalies present in
    # `test` and absent from `reference`, so the leaking frame is always the
    # second argument of whichever direction wins.
    if known_baseline == "a":
        return _analyse_one_direction(frame_a, frame_b, **kwargs)

    if known_baseline == "b":
        result = _analyse_one_direction(frame_b, frame_a, **kwargs)
        result.leaking_frame = "a" if result.leak_detected else "neither"
        return result

    # Unknown ordering: analyse both directions and keep the stronger.
    #
    # This mode is best-effort. Measured over simulated pairs with leaks in the
    # 0.08-0.6 W range it names the correct frame in 39 of 40 cases. Failures
    # occur when an artefact in one direction scores close to the leak in the
    # other: both composite scores sit near the top of their range, so the
    # margin between them is thin and noise can flip it.
    #
    # A signed single-pass probe was tried instead, ranking candidates by the
    # polarity of the strongest anomaly. It is worse: scoring a negative-going
    # anomaly against a K0 model whose amplitude is constrained positive drives
    # the true leak's physics score to zero, so the probe ranks artefacts above
    # it and reads the sign off the wrong feature.
    #
    # In practice this rarely matters. The caller almost always knows which
    # frame is the intact reference, because they took it first -- pass
    # `known_baseline` and the ambiguity disappears entirely.
    forward = _analyse_one_direction(frame_a, frame_b, **kwargs)   # leak in b
    reverse = _analyse_one_direction(frame_b, frame_a, **kwargs)   # leak in a

    if reverse.score > forward.score:
        reverse.leaking_frame = "a" if reverse.leak_detected else "neither"
        return reverse
    forward.leaking_frame = "b" if forward.leak_detected else "neither"
    return forward
