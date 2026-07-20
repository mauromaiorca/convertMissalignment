# missalign_script_v7

Version: `7.0.0-alpha3-imod-warp-axis-contract`

## Purpose

v7 keeps the v6 workflow but changes the `raw_xf_affine_fixed` conversion and
adds a mandatory geometry check before MissAlignment.

The default conversion mode for `raw_xf_affine_fixed` is now:

```text
quarter-turn-affine
```

A common exact quarter turn (`0/90/180/270°`) is selected for the whole tilt
series. The stack is permuted with `rot90` without interpolation. Only the
remaining small affine is encoded in Warp `GridMovementX/Y`.

For the current `lam8_ts_004` geometry, the expected choice is approximately:

```text
original affine rotation:  -84.5°
exact stack quarter turn:   -90.0°  (np.rot90 k=1)
residual affine rotation:    +5.5°
```

## New workflow

```text
init/prepare
→ phase2a_convert_and_pre_reconstruct.sbatch
→ inspect PRE map
→ record acceptance
→ phase2.sbatch (MissAlignment)
→ post-MissAlignment reconstructions
```

The PRE job performs only:

```text
canonical staging manifest
→ Warp conversion
→ WarpTools PRE reconstruction
```

It does not start MissAlignment.

The source reconstruction geometry and the Warp volume geometry use explicit
frames. An IMOD reconstruction MRC stores the thickness in MRC Y, whereas Warp
uses logical Z for thickness. The base mapping is therefore:

```text
Warp (X,Y,Z) = IMOD-MRC (X,Z,Y)
```

The optional exact detector-plane `rot90` swaps the projection-image width and
height and transforms the detector-coordinate geometry. It does not rotate the
3-D reconstruction-volume frame. Therefore `VolumeDimensionsAngstrom` and
`--tomo_dimensions` remain in the base Warp order `(IMOD X, IMOD Z, IMOD Y)`
for both translation and quarter-turn-affine conditions.

## PRE reconstruction output

```text
<RUN_DIR>/diagnostics/warp_reconstruction/pre_conversion/latest_success/
```

The map is under:

```text
output_pre_conversion/reconstruction/*.mrc
```

The conversion manifest records the quarter-turn and residual affine under:

```text
<RUN_DIR>/warp/warp_raw_xf_affine_fixed/*.conversion.json
```

## Manual acceptance

After inspecting the PRE map:

```bash
MISS_PY=/path/to/missalignment-environment/bin/python
REPO=/path/to/missalign_script_v7

"$MISS_PY" \
  "$REPO/scripts/pipeline/accept_pre_conversion.py" \
  --project-settings /absolute/path/to/project_settings.toml \
  --note "PRE map visually correct"
```

For `raw_xf_affine_fixed`, `phase2.sbatch` refuses to run without this acceptance
record. Emergency override:

```bash
ALLOW_WITHOUT_PRE_CONVERSION_ACCEPTANCE=1 sbatch phase2.sbatch
```

The override should not be used for normal production.

## Scientific status

Locally verified for alpha3:

- exact matrix factorization `A = A_residual @ Q`;
- coordinate equivalence on centres, corners and edges;
- lossless pixel permutation for the quarter turn;
- explicit IMOD-MRC `(X,Y_thickness,Z)` to Warp `(X,Z,Y)` mapping;
- odd-quarter-turn exchange of Warp X/Y with Warp Z unchanged;
- generated Slurm syntax and rejection of legacy conversion manifests.

Observed on the cluster before alpha3:

- WarpTools completed PRE reconstructions for the translation and quarter-turn
  conditions;
- the initial correction selected the wrong axis pair, which motivated the
  explicit storage-frame contract implemented here.

Not yet cluster-verified for alpha3:

- the regenerated translation and quarter-turn PRE maps;
- MissAlignment training/refinement on the corrected quarter-turned project;
- the final IMOD/Warp coordinate round trip after the v7 full run.

Keep v6 results unchanged until the v7 PRE map has been inspected and accepted.
