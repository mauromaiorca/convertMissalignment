# convertMissAlignment

`convertMissAlignment` is a command-line pipeline for importing IMOD/eTomo tilt-series projects into Warp, preparing MissAlignment inputs, generating Slurm jobs, validating reconstructions, and exporting results back to IMOD.

The Python distribution version is **0.1.10**. The bundled processing pipeline is based on **MissAlignment pipeline 8.0.0-alpha6-optional-smoke-run**.

## Compatibility

The historical command and Python entry point are retained:

```text
command:       convertMissalignment
entry point:   convertMissalignment.cli:main
distribution:  convertMissAlignment
version:       0.1.10
```

Existing commands such as the following remain valid:

```bash
convertMissalignment setup \
  --data-dir /path/to/etomo_project \
  --basename TS1 \
  --out-dir /path/to/output/TS1_translation \
  --condition translation
```

The legacy condition name `translation` is normalised to the canonical v8 condition `raw_xf_translation` before the project configuration is written.

The historical script name `missalignment_script_prepare.py` is also retained as a compatibility wrapper. The equivalent installed command is:

```bash
convertMissalignment input --directory /path/to/project
```

## Installation

The repository can be installed from any directory. Two example checkout locations are:

```text
Cluster: /gpfs/cssb/user/maiorcam/software/convertMissalignment
Local:   /Users/mauro/progetti_software/missaling/convertMissalignment
```

These paths are examples only. No repository location is hard-coded into the package.

From the repository root:

```bash
python -m pip install -e .
```

For a regular, non-editable installation:

```bash
python -m pip install .
```

See [INSTALL.md](INSTALL.md) for cluster, local, wheel, upgrade, and removal instructions. See [USER_GUIDE.md](USER_GUIDE.md) for workflow details, [VERSIONING.md](VERSIONING.md) for release metadata, and [GITHUB.md](GITHUB.md) for repository publication.

## Verify the installation

```bash
convertMissalignment --version
convertMissalignment where
convertMissalignment doctor
```

Expected package version:

```text
0.1.10
```

`convertMissalignment where` reports the executable, Python interpreter, environment, installed package directory, and source root. This is the recommended way to identify an editable checkout.

## Main workflow

### 1. Create a project

```bash
convertMissalignment setup \
  --data-dir /path/to/etomo_project \
  --basename TS1 \
  --out-dir /path/to/output/TS1 \
  --condition translation
```

Canonical condition names are also accepted directly:

```text
raw_identity
raw_xf
raw_xf_translation
raw_xf_affine_fixed
ali_identity
```

### 2. Reconstruct the imported Warp dataset

Setup generates the reconstruction job but never submits it. Run it straight away with:

```bash
convertMissalignment reconstruct /path/to/output/TS1
```

The command locates the generated batch, submits it with `sbatch`, and prints where the
logs and the reconstruction will appear. Add `--dataset 5.45Apx` when the project holds
several datasets, `--print` to only show the command, or `--local` to run it directly
inside an interactive allocation.

The manual equivalent still works:

```bash
sbatch /path/to/output/TS1/batches/warp_data/5.45Apx/reconstruct.sbatch
```

### 3. Prepare MissAlignment inputs

```bash
convertMissalignment input --directory /path/to/output/TS1
```

The compatibility script remains available when running from a source checkout:

```bash
python missalignment_script_prepare.py --directory /path/to/output/TS1
```

### 4. Submit smoke or full refinement

```bash
sbatch /path/to/output/TS1/batches/missalignment/5.45Apx/run_smoke.sbatch
sbatch /path/to/output/TS1/batches/missalignment/5.45Apx/run_full.sbatch
```

The smoke run is recommended as a bounded safety check but is not required by the full-run launcher.

### 5. Compare and export

```bash
sbatch /path/to/output/TS1/batches/missalignment/5.45Apx/compare_reconstructions.sbatch
sbatch /path/to/output/TS1/batches/export/5.45Apx/export_imod_and_reconstruct.sbatch
```

## Project layout

```text
PROJECT/
├── project_settings.toml
├── imported_data/imod/
├── warp_data/<pixel_size>/
├── batches/{import,warp_data,missalignment,export}/
├── logs/
├── missalignment/runs/<pixel_size>/
├── provenance/
└── .internal/
```

`project_settings.toml` is the authoritative resolved configuration. Do not edit `.internal/` manually. Each `warp_data/<pixel_size>/` directory is a complete Warp tilt-series dataset.

## Runtime configuration

The source checkout location and the scientific Python environment are separate concepts.

The package location is determined automatically from the installed Python modules. The MissAlignment scientific environment is resolved in this order:

1. `--missalign-env /absolute/path/to/environment`
2. the `MISSALIGN_ENV` environment variable
3. `missalign_environment` in the selected cluster profile, when non-empty
4. the active Python environment (`sys.prefix`)

For example:

```bash
export MISSALIGN_ENV=/path/to/missalignment-environment
convertMissalignment setup ...
```

The bundled `maxwell` profile contains shared cluster settings but no personal user environment path.

## Commands

```text
convertMissalignment setup       create and prepare a project
convertMissalignment reconstruct reconstruct the imported Warp dataset
convertMissalignment inventory   show what a project produced and where the data are
convertMissalignment prepare     run lower-level preparation operations
convertMissalignment input       prepare MissAlignment input snapshots
convertMissalignment preprocess  create a lower-resolution Warp dataset
convertMissalignment status      show project and dataset state
convertMissalignment export      export results to IMOD
convertMissalignment refine      run local refinement utilities
convertMissalignment imod-recon  run the IMOD reconstruction entry point
convertMissalignment version     print the installed package version
convertMissalignment where       show installation and source locations
convertMissalignment doctor      check Python and external dependencies
```

The shorter aliases `missalign` and `convertmissalignment` are provided, but `convertMissalignment` remains the compatibility command.

## External dependencies

The base package installs `numpy` and `mrcfile`. Individual workflows may also require:

```text
warpylib
PyTorch
torch-projectors
MissAlignment
WarpTools and WarpWorker
IMOD
Slurm
```

GPU and cluster dependencies should normally be installed or provided by the target HPC environment rather than forced by the base package.

## Tests

Install test dependencies and run the portable suite:

```bash
python -m pip install -e '.[test]'
pytest -q \
  tests/test_python_compiles.py \
  tests/test_alignment_models.py \
  tests/test_project_config.py \
  tests/test_v8_layout.py \
  tests/test_v8_optional_smoke.py \
  tests/test_v8_preprocess.py \
  tests/test_cli_compatibility.py
```

Some integration tests require IMOD, WarpTools, MissAlignment, Slurm, or real microscopy data.

## Repository publication

The repository includes cluster profiles, test fixtures, and integration code. Review these files before making the repository public. No open-source licence is included in this release; without a licence, third parties do not automatically receive permission to copy, modify, or redistribute the code.
