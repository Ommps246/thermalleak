# Capture protocol: real paired thermal data

This closes limitation #1, the only one that blocks publication. Everything in
the README is simulation; a reviewer asks about real data on the first page, and
no amount of simulation refinement answers them.

The obstacle is that a *leak* normally means damaging something. It does not
have to.

## The trick

**Tape a resistor to the surface and drive it at known power.**

A resistor dissipates `P = V*I` watts. Tape it to the casing, photograph the
device with it attached but unpowered (baseline), then powered (inspection).
The only difference between the two frames is heat -- same geometry, same
emissivity, same reflections. That is a cleaner control than a real crack would
give you, because a crack changes the surface as well as the heat.

Sweeping the voltage sweeps leak power across the same 0.01-0.5 W range the
simulator explores, so you get a **detection limit measured against a real
camera and a real device**, directly comparable to the simulated 0.023 W.

That one number is the paper's strongest sentence.

## Equipment (~500 rupees plus a camera)

- Thermal camera with **radiometric export** -- see the go/no-go below
- Metal-film resistor, 10 ohm 1 W (optionally also 4.7 and 22 ohm)
- Variable DC supply, or a USB power bank plus a USB power meter reading V and I
- Kapton or electrical tape
- A ruler that fits in frame
- Matte black electrical tape (emissivity ~0.95) as a reference patch
- Anything rigid to hold the camera still: tripod, clamp, stack of books
- Optional, for a crack-shaped source: a few cm of nichrome wire

## Go/no-go: the camera check

**Do this before anything else.** Export one frame and confirm you can get
16-bit radiometric data -- per-pixel temperature, not a colour-mapped JPEG.

FLIR One and Seek both store radiometric data inside the JPEG, but the default
share/export path discards it. Look for "radiometric JPEG" or CSV export, or run
`exiftool` on a file and check for embedded raw thermal data. `thermalleak`
works in kelvin throughout; a false-colour render is unusable and no processing
recovers it.

If you cannot export radiometric data, stop and find another camera. Do not
collect a dataset you will have to throw away.

## Setup

1. Mount the camera rigidly, framing the device. **Do not move it between
   captures.** Registration handles a few pixels; it should not have to.
2. Tape the resistor to the surface, wires led out of frame. Leave it attached
   for every capture in the session, powered or not.
3. Put the ruler in frame for one shot, then remove it. Measure
   `metres_per_pixel` from that shot.
4. Stick a patch of matte black tape in frame. Known emissivity ~0.95, so it
   anchors the radiometry.
5. Record ambient temperature.

## Capture sequence

    Step 1  Equilibrium   device on, resistor unpowered, wait 20-30 min
    Step 2  NULL PAIR     two captures 5 min apart, both unpowered
    Step 3  Baseline      10 frames, unpowered
    Step 4  Power sweep   for each of ~0.02, 0.05, 0.1, 0.2, 0.5 W:
                            set V, record V and I, wait 5 min for steady state,
                            capture 10 frames
    Step 5  Repeat        whole sequence 3+ times, for error bars

Run the null pair through the same command as everything else --
`run_detect.py frames/null_a frames/null_b --average` -- and it should report no
detection. Whatever it *does* report is your false-alarm floor.

**The null pair in step 2 is not optional.** Two captures of an unchanged scene
give the false-alarm floor of the real system -- drift, reflections, breathing
on the lens, everything the simulator only approximates. Without it you have no
denominator and the detection limit means nothing.

Averaging 10 frames per condition cuts temporal noise by ~3x, which matters at
the weak end where the interesting result lives.

## Analysis

**Step 0, before the session.** Run the go/no-go on a single exported frame.
This is the check that decides whether the dataset is worth collecting:

    python3 scripts/convert_radiometric.py one_frame.jpg --dry-run

It refuses 8-bit and colour input, which is exactly the failure the default
share path on FLIR One and Seek produces. `GO` means the radiometric data
survived the export.

**Step 1, convert the session.** Whole directories at a time:

    python3 scripts/convert_radiometric.py raw/baseline/ --outdir frames/baseline --units celsius
    python3 scripts/convert_radiometric.py raw/p050/    --outdir frames/p050    --units celsius

Use `--units raw --scale 0.01` for 16-bit exports holding centikelvin, and
`--flir-jpeg` to pull the raw thermal image out of a FLIR radiometric JPEG via
`exiftool`. Output is always kelvin, which is what every threshold here is in.

**Step 2, detect on the averaged stacks.** `--average` takes a directory or a
quoted glob per side and averages the ten frames before detecting, which is the
point of capturing ten:

    python3 scripts/run_detect.py frames/baseline frames/p050 --average \
        --mpp <measured> --material abs_casing --report out.png

It prints the measured per-pixel scatter and the reduction achieved. If ten
frames do not give roughly a 3x reduction, something moved during the stack and
averaging is hiding it rather than helping.

Set `metres_per_pixel` from the ruler shot. Estimate material properties for
your subject and add a `Material` entry -- the values in `MATERIALS` are
effective in-plane values for a skin plus backing structure, not handbook
numbers for bare plastic.

Report the real detection limit beside the simulated one, and expect it to be
worse. That is the honest and interesting comparison.

## Caveats to state in the paper

- **Delivered power is less than electrical power.** The resistor dissipates
  `V*I`, but some convects and radiates from the resistor body rather than
  conducting into the plate. Quote electrical power as an upper bound on leak
  power. It can be bounded better by integrating convective loss over the
  measured steady-state field: `P_delivered ~ 2h * integral(dT dA)`.
- **A taped resistor is not a crack.** It is a surface-mounted point source. It
  tests detection and localisation honestly; it does not prove the detector
  finds real cracks. Say so.
- **One device is one device.** Material, format and finish all matter. Three
  subjects -- laptop underside, power bank, drone pack -- is much stronger than
  thirty runs on one.

## Safety

Nothing here is dangerous and it should stay that way. A 1 W resistor at 1 W is
hot enough to burn: tape it, do not hold it. Do not drive a lithium cell outside
its normal charge and discharge, and do not deliberately induce thermal runaway
-- that produces fire and hydrogen fluoride, and belongs in a blast chamber with
trained staff. Warm is data. Runaway is a fire.
