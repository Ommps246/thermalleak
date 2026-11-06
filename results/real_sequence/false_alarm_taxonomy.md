# What the detector mistakes for a leak on real data

400 pairs of genuinely static frames from OTCBVS Dataset 03 sequence 3a — no
scene change, no injected leak. **141 fired (35.2%).** This is what they were.

> **Numbers below describe the pipeline as it stood when the analysis was run.**
> A border defect identified here has since been fixed; the current rate is
> **39.8%**, and fixing it changed neither the rate nor the spatial pattern.
> See "Correction" below — the causal claim in this document was tested and
> proved wrong, and is left standing with the refutation beside it.

Figures: `02_false_alarm_taxonomy.png` (where they land, density, trimming),
`false_alarms/fa_01..12.png` (the twelve highest-scoring, full four-panel
reports), `03_physics_on_vs_off.png` (one case physics rejects and differencing
does not).

## The short version

Almost none of it is the scene in the sense of "something happened". Two
groupings account for 98% of firings. The border grouping was originally
attributed to a defect in `register.py`; that defect was real and has since been
fixed, and **fixing it changed nothing** — see the correction below before
relying on the causal story.

| class | n | share | what it is |
|---|---|---|---|
| **A — frame-border rim** | 58 | 41% | registration warp corrupts the outermost row/column |
| **B — structural edge** | 80 | 57% | a high-contrast thermal boundary breathing between frames |
| **C — diffuse drift** | 3 | 2% | genuine broad thermal change with no edge involved |

Class boundaries are a heuristic (A: centroid within 4 px of the border; B:
dipole ratio > 0.6 or elongation > 2.5; C: the rest). The quantitative claims
below do not depend on that threshold.

## Class A: the frame-border rim — a real bug, not a scene effect

Firing density against distance from the frame border, normalised by the area
available at each distance:

| band | firings | density vs average |
|---|---|---|
| 0–4 px | 58 | **3.40×** |
| 4–8 px | 29 | 1.82× |
| 8–16 px | 29 | 1.01× |
| 16–32 px | 25 | 0.57× |
| 32–64 px | **0** | **0.00×** |

Within **1 px** of the border the density is **8.0× average**. Beyond 32 px from
any edge, across 400 pairs, the detector never fired once.

The mechanism, traced end to end on one pair:

1. ECC finds a sub-pixel shift — here **0.075 px**, essentially perfect
   registration.
2. `cv2.warpAffine` with `borderValue=NaN` still leaves **255 NaN pixels**, on
   the last row and last column, because interpolation there samples outside the
   source.
3. `register.py` then does `np.nan_to_num(aligned, nan=np.nanmedian(aligned))`.
   The true values at those pixels were ~300.05 K; the frame median is 299.43 K.
   That substitution **manufactures a 0.62 K step** exactly at the rim.
4. In the difference image the rim reaches **6.28 K against a 1.07 K interior** —
   the artefact is six times larger than any real signal in the frame.

`valid_mask` is eroded by a 5×5 kernel and does exclude that 1-px ring, so the
rim itself cannot become a candidate. What it does not exclude is the *matched
filter's* response to it, and the half-maximum region grown from a peak sitting
just inside the mask.

Trimming pixels from each side confirms the diagnosis and its limit:

| trimmed | physics FAR | differencing FAR |
|---|---|---|
| 0 px | 35.2% | 49.5% |
| 2 px | 27.5% | 41.8% |
| 4 px | 25.5% | 33.2% |
| 8 px | 18.8% | 24.2% |
| 12 px | 12.5% | 21.0% |

Trimming helps a lot — and never finishes, because **a new border is created by
the trim**. The edge-adjacent share of the survivors *rises* from 73% to 96% as
the crop shrinks. This is a property of the processing chain, not of this
dataset, and it will appear on any real capture.

### Correction: the defect was real, and it was not the cause

The paragraph above originally predicted that fixing the NaN substitution would
"remove a large fraction of real-data false alarms". **It was fixed, and it did
not.** The prediction was an inference from spatial coincidence — false alarms
cluster at the border, the border carries a manufactured step, therefore one
causes the other — and the intervention falsified it. Recorded here rather than
quietly rewritten, because the reasoning error is more instructive than the
result.

What the fix did (`register.py` fills unreachable pixels from the baseline,
`normalize.py` forces the difference to exactly zero there, `detect.py` uses
normalised convolution so the kernel averages only real pixels):

| | before | after |
|---|---|---|
| difference at the rim | 6.28 K | **0.00 K** (exact) |
| real-data FAR, physics | 35.2% | **39.8%** |
| real-data FAR, differencing | 49.5% | 56.0% |
| firing density within 1 px of border | 8.0× | **7.7×** |

The manufactured step is gone. The false alarms are not — the rate rose, and the
border enrichment is essentially unchanged. The defect was genuine and worth
removing on correctness grounds, but it was never what the detector was firing
on: `valid_mask` had always kept the rim itself out of candidate extraction, and
the matched filter's spread of it was evidently not what pushed interior pixels
over threshold.

**Where the border effect actually lives.** Measured after the fix, at the pixel
level the border is not special: the supra-threshold pixel fraction there is
1.29× the interior, and the mean raw difference is *smaller* (0.081 K against
0.145 K). Yet the *top-ranked candidate* lands within 4 px of the border at 3.5×
average density, and at 6.4× within the sunlit region considered on its own — so
it is not a scene-activity confound either. A 1.3× effect at the pixel level
producing a 6× effect in the ranking points at the candidate-formation and
scoring stages, where a truncated radial profile is fitted against a model that
assumes a complete one. That is in `score.py`, which was held bit-identical
throughout this work and was not investigated.

The FAR rise is the expected direction, incidentally: `mode="nearest"` smoothed
real structure near edges by replicating it, and the faithful filter no longer
does, so genuine near-edge thermal changes now survive to be detected. They are
real changes. They are still not leaks.

## Class B: structural edges, and one particular kerb

**63 of the 80 class-B firings (79%) land on the same feature** — the curved
kerb in the lower-left, the boundary between sunlit paving and shaded stone. It
is visible as the dense red cluster in `02_false_alarm_taxonomy.png`.

Median elongation 2.2, median **dipole ratio 0.93** — the negative lobe is
almost exactly as large as the positive one. That is the signature of a
*boundary that moved or breathed*, not of a source that appeared: a source adds
heat, an edge shifting redistributes it, so the difference integrates to nearly
zero across the feature.

Physically this is real and unremarkable. The kerb is a high-contrast thermal
boundary; over 20–90 s the sun moves, the two materials cool at different rates,
and the boundary's position in temperature-space shifts by ~1 K. The detector
has no test that asks "does this integrate to zero?", so a dipole is scored on
its positive lobe alone.

The remaining 17 class-B firings sit at a median of 11 px from the border —
close enough that the rim's filtered response plausibly contributes.

## Class C: what a real false alarm looks like

Only **3 of 141** are diffuse thermal change with no edge and no border
involvement — median elongation 1.4, dipole 0.55, and the lowest physics scores
of the three groups (0.56 against 0.75 and 0.77).

**This is the encouraging number.** The nuisance the simulator models most
carefully — broad, genuine, diffusive thermal drift that no physics test can
reject — turns out to be responsible for about **2% of real false alarms, or
0.75% of all pairs.** The simulator is not wrong about that nuisance; it is
simply not the one that dominates real data.

## What this changes

**The 35.2% is not a statement about physics.** It is roughly 41% border bug,
57% moving edges, 2% the drift the simulator was built around. Removing the
border artefact alone would take it to somewhere near 20% without any change to
`score.py`.

**The physics stage earns its place, visibly.** `03_physics_on_vs_off.png` shows
a case in the second class: differencing scores it 0.769 and calls it a leak;
the length test scores **0.00** because the fitted decay length is **1.3 px
against 4.5 px expected** — far too sharp for conduction — and the physics score
collapses to 0.04. That is the 50% → 35% reduction in a single frame.

**Two tests the false alarms suggest, neither of which exists yet.** A *dipole
test* — reject candidates whose local neighbourhood integrates to near zero —
would target class B directly, and 79% of that class is one kerb. And a *border
exclusion* proportional to the matched-filter width would target class A. Both
are cheap; neither is implemented; both should be measured against the ablation
before being believed, since the last two attempts at improving `score.py`
looked equally obvious and both lost.
