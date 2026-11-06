#!/usr/bin/env python3
"""
Show the noise floor of each detector class, derived from first principles.

    python3 scripts/show_detectors.py

Instant -- meant to be run live. Nothing here is a quoted specification: each
NETD is computed from Planck's law (in-band photon flux at 300 K), the optics,
the quantum efficiency and the charge well, exactly as the simulator uses it.
The microbolometer is the exception and says so: it does not count photons,
so the photon derivation does not apply to it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermalleak.detector import DETECTORS, band_contrast, photon_exitance  # noqa: E402


def main() -> int:
    t = 300.0
    print(f"Scene at {t:.0f} K, 8-14 um band")
    print(f"  in-band photon exitance  {float(photon_exitance(t)):.3e} photons/s/m^2")
    print(f"  thermal contrast         {band_contrast(t, (8e-6, 14e-6))*100:.2f} % per kelvin")
    print()
    hdr = (f"{'detector':<16}{'type':<9}{'q.eff':>6}{'well (e-)':>11}"
           f"{'integrates':>12}{'electrons':>12}{'NETD':>11}")
    print(hdr)
    print("-" * len(hdr))
    for key, d in DETECTORS.items():
        if d.kind == "thermal":
            print(f"{key:<16}{'thermal':<9}{'--':>6}{'--':>11}{'--':>12}{'--':>12}"
                  f"{d.netd_kelvin(t)*1000:>8.2f} mK   (bolometer: quoted, not derived)")
            continue
        well = "unlimited" if d.well_capacity_e == float("inf") else f"{d.well_capacity_e:.1e}"
        ti = d.integration_time_s(t)
        ti_s = f"{ti*1e3:.1f} ms" if ti >= 1e-3 else f"{ti*1e6:.0f} us"
        print(f"{key:<16}{'photon':<9}{d.quantum_efficiency:>6.2f}{well:>11}{ti_s:>12}"
              f"{d.collected_electrons(t):>12.2e}{d.netd_kelvin(t)*1000:>8.2f} mK")
    print()
    q, h = DETECTORS["qwip"], DETECTORS["hgcdte"]
    print(f"Note the QWIP: {h.quantum_efficiency/q.quantum_efficiency:.1f}x lower quantum efficiency "
          f"than HgCdTe, yet NETD {q.netd_kelvin(t)*1000:.1f} vs {h.netd_kelvin(t)*1000:.1f} mK.")
    print("In LWIR the scene floods the detector with photons, so every real detector fills its")
    print("well in microseconds. Sensitivity is set by charge handling, not quantum efficiency --")
    print("the same route published 2-5 mK arrays take.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
