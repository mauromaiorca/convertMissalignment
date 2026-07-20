# User guide

## Existing workflow compatibility

Version 0.1.12 is designed to replace the previous editable installation without changing the historical executable name.

The following command remains supported:

```bash
convertMissalignment setup \
  --data-dir /path/to/etomo_project \
  --basename tomo2 \
  --out-dir /path/to/output/tomo2_translation \
  --condition translation
```

The compatibility layer converts `translation` to `raw_xf_translation`. The generated project uses the canonical name in `project_settings.toml` and in provenance records.

After setup, reconstruct the imported dataset. `setup` generates the job but never submits
it, so run:

```bash
convertMissalignment reconstruct /path/to/output/tomo2_translation
```

It finds the generated batch, submits it with `sbatch`, and prints the log and output
locations. Useful options:

```bash
convertMissalignment reconstruct PROJECT --dataset 5.45Apx   # when several datasets exist
convertMissalignment reconstruct PROJECT --print             # show the command, submit nothing
convertMissalignment reconstruct PROJECT --local             # run it here (inside an interactive allocation)
```

The equivalent manual submission remains available:

```bash
sbatch /path/to/output/tomo2_translation/batches/warp_data/<dataset-id>/reconstruct.sbatch
```

Then prepare the MissAlignment snapshots:

```bash
convertMissalignment input \
  --directory /path/to/output/tomo2_translation
```

The source-checkout compatibility form also remains available:

```bash
python missalignment_script_prepare.py \
  --directory /path/to/output/tomo2_translation
```

## Finding the installation

```bash
command -v convertMissalignment
convertMissalignment where
python -m pip show convertMissAlignment
```

For an editable installation, `pip show` reports an editable project location and `convertMissalignment where` reports the source root.

## Version checks

```bash
convertMissalignment --version
convertMissalignment version
```

Both commands report package version `0.1.12`.

The bundled pipeline revision is shown by:

```bash
convertMissalignment where
```

Package version and pipeline revision are intentionally separate. The package version identifies this installable distribution. The pipeline revision identifies the underlying scientific workflow snapshot.

## Dataset selection

The input command selects the dataset automatically when the choice is unambiguous:

```bash
convertMissalignment input --directory PROJECT
```

List available datasets:

```bash
convertMissalignment input --directory PROJECT --list-datasets
```

Select a dataset explicitly by identifier or path:

```bash
convertMissalignment input --directory PROJECT --dataset 16.356Apx
convertMissalignment input --directory PROJECT \
  --dataset PROJECT/warp_data/16.356Apx
```

## Lower-resolution preprocessing

```bash
convertMissalignment preprocess --project PROJECT --bin 3
sbatch PROJECT/batches/warp_data/<processed-dataset-id>/preprocess.sbatch
sbatch PROJECT/batches/warp_data/<processed-dataset-id>/reconstruct.sbatch
```

After successful preprocessing and reconstruction, the processed dataset can become the default MissAlignment input.

## Diagnostics

```bash
convertMissalignment doctor
```

The diagnostic command checks:

- Python version and package metadata
- required and optional Python modules
- Slurm commands
- WarpTools and WarpWorker
- IMOD commands
- the MissAlignment executable
- packaged configuration and helper files

Missing cluster commands are expected on a workstation. Missing `numpy` or `mrcfile` means that the base installation is incomplete.

## Source directory independence

The command resolves package resources relative to the installed modules. It does not require the checkout to be named `missalign_script_v8` and does not require a specific GPFS or macOS path.

An editable installation points directly to the checkout. After moving or renaming that checkout, reinstall it:

```bash
python -m pip install -e .
```

A wheel installation is independent of the original source checkout because its files are copied into the Python environment.
