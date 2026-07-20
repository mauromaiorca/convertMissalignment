# Changelog

## 0.1.12

- `inventory` prints each artefact path on a single line, and reports the attempt that
  produced it (`from: attempt_<jobid> (output_...)`), so a published reconstruction can be
  traced to the job that made it without following symlinks by hand.
- `inventory` now shows the MissAlignment input manifest and the before/smoke/full snapshot
  directory, and the full-run manifest, instead of leaving those stages without a location.
- Folds `status` into `inventory`, which ends with a PROJECT RECORD section listing the
  manifests and the event count. `status` still runs and redirects.

## 0.1.11

- Rewrites `inventory` as a numbered, top-to-bottom report: each stage says what it is,
  whether it is done, where its data are, and the command to run it. Lines stay inside a
  standard terminal width and paths are relative to the project.
- Merges `where` and `doctor` into `info` (installation locations plus environment check).
  `where` and `doctor` keep working but are no longer advertised.
- `status` now defaults to the current directory.
- `export` with no arguments explains what export does, lists the projects it can act on,
  and prints the exact command to run.
- Drops `prepare-input` from the public commands; `input` is the single name. The old
  spelling is still accepted.

## 0.1.10

- Adds `convertMissalignment inventory`, which maps a project: which steps have run, where each
  artefact lives (resolving symlinks into `.internal`), and the exact command for each missing step.
- Adds the `missalign-inventory` console entry point.

## 0.1.9

- Adds `convertMissalignment reconstruct`, which runs the Warp reconstruction of the imported
  dataset straight after `setup`: it locates the generated `batches/warp_data/<dataset>/reconstruct.sbatch`,
  submits it with `sbatch`, and reports the log and output locations.
- Supports `--dataset` (when a project holds several datasets), `--print` (show the command
  without submitting) and `--local` (run the batch directly inside an interactive allocation).
- Adds the `missalign-reconstruct` console entry point.

## 0.1.8

- Keeps the historical `convertMissalignment` executable and Python entry point `convertMissalignment.cli:main`.
- Uses `0.1.8` as the single Python distribution version.
- Adds `convertMissalignment --version`, `version`, `where`, and `doctor`.
- Accepts the historical setup condition `translation` and stores it as the canonical v8 condition `raw_xf_translation`.
- Adds the compatibility filename `missalignment_script_prepare.py`.
- Adds the shorter `missalign` command without removing the historical command.
- Packages configuration files, helper modules, shell scripts, and cluster integration resources.
- Removes personal source and environment paths from runtime defaults and public examples.
- Resolves the scientific Python environment from `--missalign-env`, `MISSALIGN_ENV`, the cluster profile, or the active environment.
- Replaces the Italian guide with English installation and user documentation.

## 0.1.0

- Previous local editable installation.
