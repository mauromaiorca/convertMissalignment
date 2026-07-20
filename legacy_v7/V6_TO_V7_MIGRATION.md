# v6 to v7 migration

Do not overwrite `missalign_script_v6`. Install v7 beside it:

```text
/path/to/missalign_script_v6
/path/to/missalign_script_v7
```

Do not reuse a v6 run directory. Create a new v7 output/project directory so
all manifests and coordinate frames remain unambiguous.

## Minimal cluster sequence

1. Extract the v7 source beside v6.
2. Create or initialize a new v7 project settings file.
3. Confirm:

```toml
[conversion.condition_modes]
raw_xf_affine_fixed = "quarter-turn-affine"
```

4. Run `prepare` to generate the run directory and jobs.
5. Submit only:

```bash
sbatch <RUN_DIR>/jobs/phase2a_convert_and_pre_reconstruct.sbatch
```

6. Inspect:

```text
<RUN_DIR>/diagnostics/warp_reconstruction/pre_conversion/latest_success/
    output_pre_conversion/reconstruction/*.mrc
```

7. Accept the PRE map with `scripts/pipeline/accept_pre_conversion.py`.
8. Submit `<RUN_DIR>/jobs/phase2.sbatch`.

## Compatibility

`full-affine` remains accepted as an explicit legacy mode. It is no longer the
default for `raw_xf_affine_fixed` in v7.
