# Changelog

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
