# MissAlignment constrained integration (translation / rigid / similarity)

A **maintained patch deliverable** that adds the constrained alignment modes
(`alignment: translation | rigid | similarity`) to the real MissAlignment
dispatcher and differentiable forward pass. Stock MissAlignment understands only
`global`/`anchoring`/`[N,N]` movement grids; this integration teaches it to
optimize the **constrained model parameters** through the real projector.

This package is delivered as code + scripts because the upstream MissAlignment
source is **not present in this repo** (and is not installable on the development
Mac). The line-level `.patch` is regenerated against the pinned upstream commit on
the cluster, where the source IS present, by `install.sh`. The *content* of the
integration — the dispatch + the constrained→detector forward — is real and
reviewable in `constrained_integration.py`.

## Contents
- `PINNED_VERSION` — the supported upstream MissAlignment version/commit.
- `constrained_integration.py` — the reference module the fork wires in: maps
  `alignment: translation|rigid|similarity` to `optimize_constrained_2d` with the
  Option-B detector-field materialization and the **real** `reconstruct_and_score`
  hook (the production projector), plus warm starts and telemetry.
- `patches/` — the patch target. `0001-constrained-dispatch.patch.template` documents
  the exact upstream insertion points; `install.sh` materializes the real diff.
- `install.sh` — locate/clone the pinned version, apply the patch, `pip install
  --no-deps -e`, verify import path + supported modes.
- `uninstall.sh` — roll back to the pinned upstream (revert the patch).
- `probe.sh` — run the capability probe and assert the modes are supported.

## Install (on the cluster)
```bash
export MISSALIGN_SRC=/path/to/miss-alignment        # existing checkout, or leave unset to clone
export MISSALIGN_REPO=<git url>                       # used only if MISSALIGN_SRC unset
bash cluster_integration/missalignment_patch/install.sh
bash cluster_integration/missalignment_patch/probe.sh # asserts translation/rigid/similarity supported
```

## Safety
- Never modifies an installed site-package silently: `install.sh` uses an editable
  install of a checked-out, patched source tree and prints the resolved import path.
- `uninstall.sh` reverts the patch (git restore) and reinstalls the pinned upstream.
- The integration keeps `apply_ctf: false` semantics (external IMOD CTF) and the
  canonical result contract (`constrained_alignment.json/.pt`, `run_manifest.json`,
  `stage_history.json`).

## Status
**CLUSTER EXECUTION NOT YET RUN.** The integration is authored and reviewable
locally; building it against the real upstream, importing it, and running CUDA
forward/backward are cluster-only steps (see `MISSALIGNMENT_PATCH_DESIGN.md` and
`IMAGE_BASED_CLUSTER_VALIDATION.md`).
