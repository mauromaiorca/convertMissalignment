# convertMissAlignment

`convertMissAlignment` is a command-line pipeline for importing IMOD/eTomo tilt-series projects into Warp, preparing MissAlignment inputs, generating Slurm jobs, validating reconstructions, and exporting results back to IMOD.

The Python distribution version is **0.1.15**. The bundled processing pipeline is based on **MissAlignment pipeline 8.0.0-alpha6-optional-smoke-run**.

## Compatibility

The historical command and Python entry point are retained:

```text
command:       convertMissalignment
entry point:   convertMissalignment.cli:main
distribution:  convertMissAlignment
version:       0.1.15
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
0.1.15
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

## IMOD tomogram positioning (tilt.com)

`setup` parses four `tilt.com` reconstruction-positioning parameters (tilt.com is
authoritative; tilt.log is only a recorded fallback) and records them under
`[geometry.imod_positioning]` in `project_settings.toml` and in every conversion manifest:

| tilt.com | resolved field | units | Warp representation |
| --- | --- | --- | --- |
| `OFFSET` | `tilt_angle_offset_deg` | degrees | `LevelAngleY` (effective angle = raw + OFFSET, applied once; raw `.tlt` kept) |
| `XAXISTILT` | `x_axis_tilt_deg` | degrees | `LevelAngleX` (sign confirmed on the cluster) |
| `SHIFT sx sz` | `shift_x/z_unbinned_px` | unbinned pixels | object-space translation (`apply_tomogram_shift_3d`), Å = px × **unbinned** IMOD pixel size |
| `THICKNESS` | `thickness_unbinned_px` | unbinned pixels | target-volume geometry |

These three are **different** quantities and are kept separate: the detector
`tilt_axis_angle_deg`, the tilt-`tilt_angle_offset_deg`, and the `x_axis_tilt_deg`.
Precedence per field: explicit project/CLI override > `tilt.com` > documented zero default.
Override any field in `[geometry.imod_positioning]`. A non-zero `SHIFT` without a resolvable
unbinned pixel size fails loudly (no guessed scale). Inspect the resolved values with
`convertMissalignment inventory` / the `imod_positioning` block in `*.conversion.json`.
The positioning hash is part of the conversion cache: changing any value forces reconversion.

The Warp geometric correctness (the `LevelAngleX` sign and the XML round-trip) is validated
**on the cluster** by `scripts/pipeline/validate_warp_positioning.py`, which uses the installed
warpylib. Until it reports success the Warp positioning is not considered validated.

## Reconstruction block tiling and the seam artefact

`ts_reconstruct` runs with explicit `--subvolume_size` (default 64) and `--subvolume_padding`
(default 6; this project **requires >= 6**, lower values are rejected, never silently replaced).
`subvolume_padding` is a single **isotropic XYZ** padding factor: the padded reconstruction side
is `subvolume_size * padding * 2` (768 px for 64/6), in X, Y **and** Z. It provides padded
reconstruction *context*, **not** true final-volume overlap — blocks are strided by
`subvolume_size`, the central crop is copied directly into the output, so the actual output
overlap is **0**. The block-boundary artefact can look stronger in XZ than XY because of
tomographic anisotropy, but the padding acts in all three axes.

Padding 6 costs `(6/3)^3 = 8×` the cubic allocation of Warp's default 3; a preflight reports
the padded side and ratios. On CUDA OOM reduce `--perdevice 1` first, never silently drop the
padding. Reconstructions carry a parameter-specific identity (`reconstruction_s64_p6`) and a
contract hash (tiling + angpix + normalisation + WarpTools version + numeric locale); changing
any of these invalidates the cache, so a padding-3 volume is never reused for a padding-6 request.
Every WarpTools subprocess runs with `LC_ALL=C`/`LANG=C` so no comma decimals corrupt the XML
(validated after save). Measure the seam quantitatively with `scripts/pipeline/seam_diagnostic.py`
(XY/XZ/YZ boundary-to-control ratios). Padding 6 is not claimed to fix the artefact unless its
seam ratio is lower than the padding-3 reference.

## Revised IMOD alignment export (source-aware round trip)

`convertMissalignment export revise <settings>` publishes the refined alignment back to
IMOD as a **source-aware round trip** — original IMOD → Warp → MissAlignment → revised
IMOD — not a generic Warp→IMOD converter and never a new unrelated IMOD project. Both
result backends (`constrained_json`, `warp_xml`) converge into one typed
`ImodAlignmentRevision` before any file is written, and the composition is always

```text
H_final = DeltaH @ H_original          (in the validated IMOD (n-1)/2 centre convention)
```

where `H_original` is the imported raw→aligned `.xf`, `DeltaH` is the MissAlignment
correction in the aligned frame, and `H_final` is the revised raw→aligned transform.

There is **one** physical export directory and **one** compatibility symlink:

```text
exported_data/imod/<condition_id>/                 # the single physical export
    configuration/  <series>.xf .residual.xf .tlt .xtilt tilt.com newst.com
    data/           <series>.mrc -> imported raw stack (relative symlink, never copied)
    reconstruct_with_imod.sh
    manifest.json
    alignment_change_report.json / .tsv / _summary.txt
    scipion_compatibility.json
missalignment/runs/<condition_id>/export/imod  ->  exported_data/imod/<condition_id>
```

`<condition_id>` is the canonical dataset id (e.g. `17.6Apx`), never the measured pixel
size (`17.596357Apx`), which is stored in the manifest. `<series>.xf` is the **complete**
revised transform; `<series>.residual.xf` is the diagnostic `DeltaH` only and is never
referenced as the complete alignment. `OFFSET/XAXISTILT/SHIFT/THICKNESS` are preserved
(effective angle = `.tlt` + `OFFSET`, applied once; `XAXISTILT` is only revised after its
Warp sign is cluster-validated). A per-tilt refined mapping that is not representable as an
affine within tolerance is rejected under `non_affine_policy = "fail"` rather than exported
as a misleading `.xf`. `reconstruct_with_imod.sh` (self-locating, `set -euo pipefail`) checks
the IMOD executables, the raw-stack link, source hashes, `.xf`/`.tlt` row counts and that the
output is not under `imported_data`, then runs `newstack` + `tilt`; it never overwrites the
imported aligned stack or reconstruction. Configure it under `[export.imod_revision]`; the
Scipion audit (`scipion_compatibility.json`) is an optional compatibility check, never the
production writer, and the export works when Scipion is absent.

## Project layout

```text
PROJECT/
├── project_settings.toml
├── imported_data/imod/
├── warp_data/<pixel_size>/
├── exported_data/imod/<condition_id>/
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

### Scientific environment via modules (no MISSALIGN_ENV needed)

When the cluster provides warpylib + MissAlignment as **environment modules** rather than a
conda env you own, you do **not** need to install your own copy or set `MISSALIGN_ENV`. Load
the site modules and run:

```bash
module load cssb/rarely   # site-specific: exposes the missalign module
module load missalign     # ships the WarpTools/MissAlignment python (with warpylib) + miss-alignment
module load warp/2.0.39   # WarpTools/WarpWorker binaries
convertMissalignment setup ...
```

The generated jobs load these same modules, and every Warp step auto-selects the Python that
actually has `warpylib`: if the launcher's own interpreter lacks it, the pipeline falls back to
the module-provided `python` (the one `module load missalign` puts on `PATH`). `MISSALIGN_ENV`
/ `--missalign-env` still take precedence when you want to pin an explicit environment. The
`missalign` module and an explicit `MISSALIGN_ENV` pointing at the same env prefix are
equivalent — the module *is* that env; you are not expected to build a private duplicate.

## Commands

```text
convertMissalignment setup       create and prepare a project
convertMissalignment reconstruct reconstruct the imported Warp dataset
convertMissalignment input       prepare MissAlignment input snapshots
convertMissalignment inventory   what a project produced, and what to run next
convertMissalignment export      export the refined alignment back to IMOD
convertMissalignment preprocess  create a lower-resolution Warp dataset
convertMissalignment prepare     run lower-level preparation operations
convertMissalignment refine      run local refinement utilities
convertMissalignment imod-recon  run the IMOD reconstruction entry point
convertMissalignment version     print the installed package version
convertMissalignment info        installation locations and environment check
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
