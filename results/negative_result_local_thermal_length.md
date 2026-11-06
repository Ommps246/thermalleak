# Negative result: a spatially varying thermal length makes detection worse

Attempt at known limitation #2 (homogeneous-material assumption). Implemented,
measured, and reverted. Recorded here rather than deleted, because it is the
obvious thing to try next and because *why* it fails is a stronger result than
the fix would have been.

## What was tried

Limitation #2 says the expected thermal length comes from one material value,
while the simulator's heat spreader raises local conductivity 8x and therefore
the true local thermal length by sqrt(8) = 2.79x. Genuine leaks over the
spreader are broad because the *plate* is broad, and the length test penalises
them for it.

The baseline frame can supply the missing map. In steady state, away from any
source, with `theta = T - T_ambient`:

    div(k*d grad(theta)) = 2*h*theta   =>   laplacian(theta) = theta / L^2

so `L^2 = theta / laplacian(theta)`. The device's own internal heat — SoC,
battery, PSU, already warming the plate in the reference image — is a free,
always-present probe of the local thermal length. No leak needed, no training
data, no new calibration: the same law the detector applies to candidates, run
backwards to measure the plate it applies them on.

Two properties made it look right:

* **Smoothing does not bias it.** The Laplacian commutes with Gaussian
  convolution, so if `laplacian(theta) = theta/L^2` holds, it holds for
  `G*theta` too. The estimator can be smoothed as hard as noise demands without
  dragging the recovered length toward the smoothing scale — the failure mode of
  any "measure the local blur width" heuristic.
* **It degrades to current behaviour.** Returned as a ratio normalised to unit
  median, so a uniform device gives 1.0 everywhere and `score.py` is unchanged.

## The estimator worked

Validated against the known conductivity map, 8 no-leak pairs:

| quantity | measured | truth |
|---|---|---|
| spreader/casing length contrast | **1.87x** | 2.79x |
| absolute median length | 7.71 px | 4.54 px (material) |

Weighting by the curvature rather than by `theta` mattered: weighting by `theta`
gave only 1.53x contrast, because `theta` is largest exactly where the source
term `P/(k*d)` contaminates the relation. Excluding the hottest 3% of the frame
(the source cores) was necessary for the same reason.

Contrast is compressed (1.87x vs 2.79x) partly by the aggregation window and
partly for a real physical reason: the spreader is only a few thermal lengths
across, so heat crossing its edge means the true local length near the boundary
genuinely is intermediate.

## The detector got worse anyway

500 pairs, seed 11, all three rows measured in the same environment:

| configuration | ROC AUC | recall @5% FPR | leaks kept | artefacts rejected |
|---|---|---|---|---|
| unmodified (reference) | **0.970** | **0.782** | **98%** | **79%** |
| local length, two-sided | 0.959 | 0.739 | 94% | 73% |
| local length, one-sided (ratio >= 1) | 0.963 | 0.769 | 94% | 75% |

`differencing only` stayed at 0.706 / 0.210 in every run — the map was fed only
to the physics scorer, never to candidate extraction, so the ablation's baseline
arm is bit-identical throughout and the comparison is clean.

## Why it fails

Per-candidate, 220 pairs, mean length-test score before -> after:

| group | n | ratio | length score | physics score |
|---|---|---|---|---|
| true leak, over spreader | 25 | 1.76 | 0.737 -> 0.826 (**+0.088**) | 0.730 -> 0.787 |
| true leak, elsewhere | 73 | 0.99 | 0.814 -> 0.781 (-0.032) | 0.877 -> 0.855 |
| artefact, over spreader | 134 | 1.61 | 0.167 -> 0.222 (+0.054) | 0.181 -> 0.192 |
| artefact, elsewhere | 447 | 0.99 | 0.264 -> 0.274 (+0.011) | 0.267 -> 0.296 |

The intended mechanism works exactly as designed: leaks over the spreader gain
+0.088 of length score, the largest single effect in the table. It is simply
outvoted. **Over the spreader there are 134 artefacts to 25 true leaks.**
Widening the acceptance band in a region admits five times more junk than
signal, so a locally *correct* band is a globally *worse* one.

The second, smaller cost is estimator variance: 26% of candidates drew a ratio
below 0.95 in nominally uniform casing, narrowing their band and costing genuine
leaks elsewhere -0.032. The one-sided variant (`ratio >= 1`) removes exactly
that term and recovers about half the loss — 0.959 to 0.963 — which confirms the
diagnosis and still does not reach 0.970, because the dominant term is the
artefact influx, and that one is inherent to the approach.

## What this means

The homogeneous-material assumption is not merely an unfixed limitation. It is
**load-bearing**. The length test carries 0.45 weight because it is a *tight*
band; any change that makes the band locally more truthful also makes it locally
looser, and looseness is precisely what artefacts exploit. Being right about the
physics and being useful as a discriminator are, here, in tension.

This does not say the limitation is unfixable. It says the fix cannot be "widen
the band where the plate is broader" alone. A version that also *narrows*
somewhere — using the local length to predict the whole expected profile rather
than just to rescale one tolerance — would keep the discriminating power the
band currently gets for free. That is the direction worth trying, and it is a
larger change than this one.

## Recovering the implementation

`estimate_length_ratio_map(baseline)` — Gaussian derivatives at `sigma=3.0` for
`theta` and `laplacian(theta)`; mask to `laplacian > 0`, `theta > 0.3 K`, and
below the 97th percentile of `theta`; weight by `laplacian`; aggregate numerator
and denominator separately under a `sigma=5.0` Gaussian; take
`sqrt(num/den)`; normalise to unit median; clip to `[0.5, 3.0]`. Wired in by
computing it once per direction in `pipeline._analyse_one_direction` from the
reference frame and passing it to `score_candidate`, which multiplied
`expected_length_px` by the median of the map over the candidate's own mask.
`LENGTH_CALIBRATION = 1.7` needed no change: unit-median normalisation means the
typical candidate still sees the reference the factor was measured against.

---

# Attempt 2: local L on the sharp side only — also worse

The first attempt failed because widening the acceptance band over the spreader
admitted five times more artefacts than leaks. That diagnosis suggests a variant
that does not widen anything, and it is not a retune — it comes from noticing
that the two sides of the length test are not the same kind of constraint.

**"Cannot be sharper than L" is a physical law.** Sustaining a gradient sharper
than the local thermal length would require heat to vanish faster than
convection removes it, and L belongs to the plate, so where the plate changes
the floor moves with it.

**"Should not be enormous" is not a law.** It is a localisation heuristic — a
leak is a compact object, and a candidate spanning half the frame is drift
residual whatever the plate underneath is doing. Nothing in that argument
depends on conductivity, so the broad limit stays global.

So: local L on the sharp floor, global on the broad limit. Implemented as two
independent one-sided constraints combined by `min()`, rather than one peaked
band, since with a local ratio above 1 the sharp floor rises past the broad
limit and a single band cannot express it. With `length_ratio = 1.0` the two
references coincide and the form reduces algebraically to the original — verified
bit-exact over 230 candidates.

The direction is the point: over the spreader the ratio is >1, so the floor
*rises* and the test becomes **more** restrictive exactly where artefacts
cluster — the opposite of attempt 1.

## Result

| configuration | ROC AUC | recall @5% FPR | leaks kept | artefacts rejected |
|---|---|---|---|---|
| unmodified (local baseline) | **0.970** | **0.782** | **98%** | **79%** |
| attempt 1, two-sided | 0.959 | 0.739 | 94% | 73% |
| attempt 1, one-sided | 0.963 | 0.769 | 94% | 75% |
| attempt 2, sharp side only | 0.955 | 0.731 | 94% | 78% |

The worst of the three. `differencing only` held at 0.706 / 0.210 throughout.

## Why: the fitted length does not track the local length

The reasoning was sound and the premise was false. Three quantities that should
have been the same number are not:

| quantity | over the spreader |
|---|---|
| true local thermal length | 2.79x material |
| what `lengthmap` recovers | 1.76x material |
| **what the fitter actually returns for a genuine leak** | **1.47x** the global reference |

Each stage compresses. Raising the floor by the map's 1.76x while genuine leaks
only reach 1.47x puts the floor above the population it was meant to protect:
**64% of true leaks over the spreader fall below the raised floor** and are
penalised as too sharp. Their mean length score drops 0.737 -> 0.595, the
largest single movement in the data, and it is a loss on precisely the
candidates the change was designed to help.

Per-candidate, mean length score before -> after:

| group | n | change |
|---|---|---|
| true leak, over spreader | 25 | 0.737 -> 0.595 (**-0.142**) |
| true leak, elsewhere | 73 | 0.814 -> 0.814 (+0.000) |
| artefact, over spreader | 134 | 0.167 -> 0.128 (-0.040) |
| artefact, elsewhere | 447 | 0.264 -> 0.273 (+0.009) |

The hypothesis was not wrong about *direction* — the floor does push artefacts
over the spreader down, by -0.040. It is wrong about *magnitude*, and it loses
because artefacts there are already bimodal (p25 0.22, p75 2.96): the sharp ones
were scoring near zero and cannot be rejected any harder, so raising the floor
collects mostly leaks.

The compression is the same effect `LENGTH_CALIBRATION = 1.7` exists to absorb.
The fit window is sized on the global thermal length and the profile sits on a
workload pedestal, so the fitted length is pulled toward the global scale
regardless of what the plate is doing underneath.

## Standing conclusion

Two attempts, from opposite directions — loosening the band where the material
permits, and tightening the floor where the material demands — both lose against
an unmodified global length. Limitation #2 stands as a **structural property**,
not an unfinished chore: the length test earns its 0.45 weight by being tight
and by being *measured on the same scale it is compared against*, and local
truthfulness costs both.

Per the stopping rule agreed before attempt 2, there is no third attempt. One
observation is recorded without being acted on, for whoever picks this up: the
floor is over-corrected by roughly the ratio 1.47/1.76, and a fix that scaled it
by the fitter's *measured* response rather than the map's would be a constant
tuned in image units — exactly what non-negotiable #8 forbids. Making the fitted
length actually track local L, by sizing the fit window from the local map, is
the only version of this idea that would not be a fudge, and it is a larger
change than either attempt here.
