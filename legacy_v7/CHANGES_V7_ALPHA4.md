# v7 alpha4: detector-frame quarter-turn contract

Version: `7.0.0-alpha4-detector-frame-quarter-turn`

## Defect corrected

Alpha3 correctly mapped the source IMOD reconstruction MRC dimensions into the
Warp volume frame:

```text
Warp volume (X,Y,Z) = IMOD-MRC (X,Z,Y)
```

However, alpha3 then incorrectly applied the exact projection-image quarter
turn to the 3-D volume dimensions. For `raw_xf_affine_fixed`, this exchanged the
requested Warp volume X/Y extents. The projection geometry and reconstruction
were otherwise coherent, but the output bounding box had the wrong X/Y size.
The translation condition was unaffected because it has no detector quarter
turn.

## Correct contract

The `np.rot90` operation is a 2-D detector-frame basis change. It acts on:

```text
projection-image width and height
tilt-axis angle
detector shifts
2-D affine coordinates and movement grids
```

It does not rotate the Warp reconstruction-volume frame. Thus both conditions
use the same volume extent mapping:

```text
translation volume XYZ:          (IMOD X, IMOD Z, IMOD Y)
quarter-turn-affine volume XYZ:  (IMOD X, IMOD Z, IMOD Y)
```

For an odd quarter turn, only the detector image shape changes:

```text
detector XY before: (X, Y)
detector XY after:  (Y, X)
volume XYZ:         unchanged
```

## Migration

Contract-v1 translation conversions are safe to reuse because their quarter
turn index is even. Contract-v1 affine conversions are marked stale and are
reconverted automatically. Contract v2 records:

```text
projection_quarter_turn_scope = detector_frame_only
reconstruction_shape_warp_xyz
volume_shape_invariant_under_detector_quarter_turn = true
```

The existing IMOD discovery, TOML configuration and quantitative source stack
are unchanged.
