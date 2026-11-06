"""
thermalleak -- physics-informed detection of thermal anomalies from paired
thermal images.

Given a baseline frame of an intact object and an inspection frame of the same
object, locate regions where heat is escaping that should not be, and decide
whether each candidate is physically consistent with a conductive heat source
rather than a reflection, an emissivity artefact or a registration error.

Quick start
-----------
    import numpy as np
    from thermalleak import detect_leak, generate_pair

    rng = np.random.default_rng(0)
    pair = generate_pair(rng, leak_power_watts=0.8)

    result = detect_leak(pair.baseline, pair.inspection,
                         material="abs_casing",
                         metres_per_pixel=pair.metres_per_pixel)
    print(result.summary())
"""

from .camera import CameraConfig, capture
from .detect import Candidate, extract_candidates
from .inject import InjectionSpec, inject_leak, load_as_field, sweep_power
from .normalize import normalise
from .physics import MATERIALS, Material, solve_steady_state
from .pipeline import DetectionResult, Finding, detect_leak
from .register import register
from .score import PhysicsVerdict, score_candidate
from .simulate import ScenePair, generate_dataset, generate_pair

__version__ = "0.1.0"

__all__ = [
    "CameraConfig", "capture",
    "Candidate", "extract_candidates",
    "InjectionSpec", "inject_leak", "load_as_field", "sweep_power",
    "normalise",
    "MATERIALS", "Material", "solve_steady_state",
    "DetectionResult", "Finding", "detect_leak",
    "register",
    "PhysicsVerdict", "score_candidate",
    "ScenePair", "generate_dataset", "generate_pair",
]
