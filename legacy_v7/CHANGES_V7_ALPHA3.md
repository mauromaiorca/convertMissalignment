> **Superseded by alpha4.** Alpha3 correctly mapped IMOD MRC storage axes to
> Warp volume axes, but incorrectly applied the detector quarter turn to the
> reconstruction-volume X/Y extents.

# v7 alpha3: explicit IMOD–Warp volume-axis contract

Version: `7.0.0-alpha3-imod-warp-axis-contract`

## Defect corrected

Alpha2 treated the three values in the source reconstruction MRC header as if
they were already Warp logical XYZ. That is not the correct storage-to-Warp
mapping for an IMOD tilt reconstruction.

The source geometry used by this pipeline is now declared as:

```text
IMOD reconstruction MRC storage: (X, Y_thickness, Z_detector_vertical)
Warp logical tomogram:           (X, Y_detector_vertical, Z_thickness)
```

The base dimension mapping is therefore:

```text
Warp (X,Y,Z) = IMOD-MRC (X,Z,Y)
```

If `quarter-turn-affine` materialises an odd `np.rot90` operation, the detector
plane is then rotated about Warp Z:

```text
Warp current (X,Y,Z) = (base Y, base X, base Z)
```

The thickness axis is not exchanged with a detector-plane axis.

For the real `testABC` geometry:

```text
IMOD-MRC target:       2046 x 494 x 2880
Warp base:             2046 x 2880 x 494
Warp after odd rot90:  2880 x 2046 x 494
```

## Software changes

The converter now writes the current Warp volume shape to each
`*.conversion.json` together with the source frame, target frame, axis
permutation and quarter-turn index. `VolumeDimensionsAngstrom` is generated
from that current Warp shape.

The PRE and pre/full WarpTools reconstruction executors no longer infer axis
order from similar dimensions. They require the conversion frame contract and
reject legacy conversion manifests.

`run_warp_conversion.py` detects a legacy `_converted.marker` and reconverts
rather than silently reusing it. Newly generated `phase2a` jobs always invoke
the converter so that this validation is executed; current conversions remain
idempotent.

## Existing projects

Existing alpha1/alpha2 Warp projects must be reconverted. The source stacks,
TOML files and IMOD geometry are not modified. The conversion output and PRE
reconstruction are regenerated from the authoritative staging manifest.
