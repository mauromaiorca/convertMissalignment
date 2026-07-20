# MissAlignment 8.0.0-alpha1

Version 8 changes the project layout and user workflow while retaining the validated version 7 conversion and quarter-turn geometry.

The project is organised by scientific meaning: imported IMOD data, complete Warp datasets indexed by pixel size, central batches and logs, per-dataset MissAlignment runs, exports, provenance, and hidden technical workspaces under `.internal`.

`warp_preprocess.py --project PROJECT --bin N` plans a derived Warp dataset, generates its pixel-size-specific batches, and records the detector resampling and coordinate transform. MissAlignment consumes the complete Warp dataset, not its reconstructed MRC volume.

Public batch names no longer use phase numbers. Setup creates import, reconstruction, MissAlignment and export batches for the native pixel size. Derived datasets receive their own batches when planned.

## 8.0.0-alpha2: synchronous Warp import

Project setup now completes the native IMOD-to-Warp import synchronously. A successful
`setup_missalign_project.py` run therefore leaves `warp_data/<pixel_size>/.warp_project`
validated and ready for the Warp reconstruction batch.

The generated `batches/import/import_imod_to_warp.sbatch` file is retained as an explicit
re-import and recovery path. It is no longer required in the normal workflow. Setup uses
the configured Warp module and the configured scientific Python interpreter, invokes the
same `scripts/run_warp_conversion.py` entry point as the recovery batch, and records the
import result in the prepare manifest and project status.

The conversion remains idempotent. Existing valid Warp datasets are reused unless
`--force-warp-import` is supplied. Setup does not submit Slurm jobs and does not run a
reconstruction, MissAlignment or CUDA workload.


## 8.0.0-alpha3: synchronous MissAlignment input preparation

MissAlignment input snapshot preparation is now a direct CPU command rather than a required
Slurm step. Run `prepare_missalignment_input.py` after the imported Warp reconstruction has
been accepted. The command validates the canonical imported Warp dataset and acceptance record,
then creates isolated `before`, `smoke` and `full` snapshots atomically enough for downstream
MissAlignment jobs. Existing valid snapshots are reused; `--force` is required to replace them.

The previous `prepare_input.sbatch` is retained as a CPU-only recovery wrapper. It no longer
requests a GPU, loads WarpTools or probes CUDA/MissAlignment. Smoke and full jobs now direct the
user to the synchronous command when snapshots are missing.

## 8.0.0-alpha4: guided project and dataset selection

`prepare_missalignment_input.py` now uses `--directory PROJECT` as its public interface.
The compatibility-only `--project-settings` alias remains accepted but is hidden from help.
The command finds `project_settings.toml`, discovers public Warp datasets, accepts either a
dataset ID or a `warp_data/<id>` path, and selects the recorded default when `--dataset` is
omitted.

The native imported dataset is selected initially. Successful preprocessing records the
derived dataset as the new default. Early v8 projects with exactly one complete processed
dataset also select it automatically. Multiple processed datasets without a recorded choice
are treated as ambiguous and are listed rather than selected by filename or modification time.

`--list-datasets` reports origin, completion state, reconstruction state, validation and source
dataset. Alpha4 still required a separate acceptance command; alpha5 supersedes that gate with
automatic technical validation after a successful reconstruction.

## 8.0.0-alpha5: automatic reconstruction validation

A successful Warp dataset reconstruction now records technical validation automatically.
The record verifies that WarpTools completed, the result manifest is complete, and the
reconstruction exists and is non-empty. This validation is sufficient for synchronous
MissAlignment input preparation and removes the required `accept_pre_conversion.py` step.

Technical validation does not claim visual inspection. The existing acceptance command is
retained as an optional provenance operation that upgrades the record to `visual` review.
Projects created with alpha4 are handled without rerunning the reconstruction: the next
`prepare_missalignment_input.py --directory PROJECT` call backfills technical validation from
the existing `latest_success` result.

The imported IMOD reconstruction remains an optional comparison control. It is not made a
dependency because its success does not validate the converted Warp geometry and would add an
unnecessary Slurm job to the standard path.

## 8.0.0-alpha6: optional smoke run

The MissAlignment smoke run is now a recommended safety check rather than a mandatory
prerequisite for the full run. Generated `run_full.sbatch` jobs require prepared full input
snapshots but no longer require `smoke_verdict.json`. When no smoke result is present, the
batch prints an explicit warning and proceeds with the full run.

`prepare_missalignment_input.py` now prints both available next actions: the optional
`run_smoke.sbatch` safety check and the direct `run_full.sbatch` command. Final run provenance
records whether smoke testing was performed and stores an empty smoke result when it was
skipped. Existing generated batches must be regenerated after installing alpha6.
