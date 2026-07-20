# WarpTools Phase-1 job generation patch

Version: `6.0.2-operational-warptools-diagnostic`

Phase 1 now generates:

```text
jobs/phase3_warptools_pre_vs_full_reconstruct.sbatch
```

The job calls `scripts/pipeline/warptools_reconstruction.py` and:

- uses the immutable `warp/pre_missalign` and `warp/missalign_full` snapshots;
- verifies that both snapshots reference the same quantitative tilt stack;
- creates a fresh attempt directory for every run;
- splits the Warp stack into individual tilt images;
- materializes the Warp 2.0.39 `average/` layout;
- patches copies of the XML metadata, never the source snapshots;
- preserves valid cumulative dose, or records a diagnostic epsilon-dose fallback;
- reconstructs pre/full with identical WarpTools options;
- writes preparation/result manifests and a `latest_success` symlink;
- checks executor and authoritative TOML hashes at runtime.

The job is diagnostic. Synthetic epsilon dose is not experimental dose metadata,
and the resulting volumes must not be used for FSC or quantitative dose/CTF
validation unless real metadata are supplied and independently validated.

For an existing prepared project after replacing the repository:

```bash
MISS_PY=/path/to/missalignment-environment/bin/python
REPO=/path/to/working_scripts_v6

"$MISS_PY" "$REPO/prepare_imod_to_warp.py" regenerate-jobs \
  "$REPO/testABCDE/project_settings.toml"
```

Then submit at the input Warp pixel size:

```bash
cd "$REPO/testABCDE/64x_Vero_02_raw_xf_affine_fixed_standard"
sbatch jobs/phase3_warptools_pre_vs_full_reconstruct.sbatch
```

Or submit a binned diagnostic reconstruction:

```bash
sbatch --export=ALL,OUTPUT_ANGPIX=10.9040002823 \
  jobs/phase3_warptools_pre_vs_full_reconstruct.sbatch
```
