> **Superseded by alpha3.** Alpha2 used the XML order directly but did not
> account for the IMOD reconstruction MRC storage-to-Warp axis mapping. Do not
> use alpha2 for new conversion or reconstruction.

# v7 alpha2: reconstruction volume-frame correction

Version: `7.0.0-alpha2-volume-frame-fix`

## Defect

The alpha1 diagnostic WarpTools executors derived `--tomo_dimensions` from the
projection-stack width and height. This is invalid for `quarter-turn-affine`:
the exact `rot90` operation swaps projection X/Y, while the requested
reconstruction volume remains defined by the target Warp volume frame.
Consequently, a successful reconstruction could have X and Y dimensions
interchanged.

## Correction

Both diagnostic executors now derive the reconstruction voxel shape from the
Warp XML `VolumeDimensionsAngstrom` in its declared X,Y,Z order and the input
pixel size:

```text
volume_shape_xyz = round(VolumeDimensionsAngstrom_xyz / input_angpix)
```

Projection-stack dimensions are recorded separately and are not used to
permute the reconstruction volume. The same correction applies to:

```text
scripts/pipeline/pre_conversion_reconstruction.py
scripts/pipeline/warptools_reconstruction.py
```

The converter remains unchanged in this respect: the lossless projection-stack
quarter turn swaps image width/height, whereas `volume_dimensions_physical`
continues to describe the configured target reconstruction volume.

## Validation

Added tests verify that a quarter-turned projection stack does not swap the
requested reconstruction-volume X/Y dimensions, and that non-integral physical
volume extents are rejected rather than rounded silently.
