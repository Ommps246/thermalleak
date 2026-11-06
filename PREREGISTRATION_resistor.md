# Pre-registered prediction for the resistor experiment (DRAFT, to be adopted by the authors)

Written 5 November 2026, before any real thermal frame of a known-power source
has been captured. It only counts as pre-registered once it is committed to the
public repository (or deposited on Zenodo) with a timestamp BEFORE the data are
collected. Edit the numbers below to what you actually believe, then commit.

Setup: PROTOCOL_real_data.md (resistor taped to the casing, powers 0.02, 0.05,
0.1, 0.2, 0.5 W, baseline = resistor unpowered). Material constants and pixel
scale are fixed from the device before looking at any detection result.

Simulation reference (full-simulation rung, ABS casing): 50% detection limit
17.4 mW (95% CI 14.4-25.3 mW) at a 5% false-alarm rate; recall 0.777.

Predictions, each stated so it can be shown wrong:

P1. The 50% detection limit on the real camera is between 17 mW and 150 mW.
    Reason: the simulator is optimistic about clutter (real-image false-alarm
    rate 39.8% against 5% in simulation), and Table 7 shows material errors of
    up to 50% in k or h move the limit to 14-45 mW.
P2. At 0.5 W the resistor is detected and localised in at least 90% of repeats.
P3. At 0.02 W it is detected in fewer than 50% of repeats.
P4. Differencing alone has a limit at least 3 times higher than the physics
    pipeline on the same frames (simulation: 10.8 times).
P5. Estimated power from the fit is within a factor of 2 of the electrical
    power, and is biased LOW (heat leaves the resistor body by convection and
    radiation, so delivered power is below V*I).

What would falsify the central claim: P4 failing (physics no better than
differencing on real frames), or a detection limit above 150 mW.
