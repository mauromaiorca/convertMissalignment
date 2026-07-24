# Changelog

## 0.1.15

- Converts the align.com `RotationAngle` into Warp's tilt-axis convention instead of copying it
  verbatim. The two are measured on OPPOSITE sides of the image vertical: with `alpha` the
  tilt-axis azimuth from detector `+X`, IMOD's `RotationAngle = 90 - alpha` while Warp's
  `TiltAxisAngle = 90 + alpha`. A verbatim copy therefore REFLECTS the axis rather than rotating
  it, leaving the reconstruction rotated in-plane by `2*alpha` (~11.8 deg for tomo2, matching the
  observed residual). The conversion is the supplement, `warp = 180 - rotation`
  (`imod_rotation_angle_to_warp_axis_angle`): tomo2 84.1 -> **95.9**. Self-consistency check: feeding
  the tiltalign-refined 84.505 returns 95.495, exactly the value the `.xf` yields independently
  (`alpha = 5.4947`); three sources agree on `alpha ~ 5.5` -- the `.xf` polar rotation, the `.mdoc`
  header `TiltAxisAngle = -174.09`, and the stack geometry (`newst.com` emits a 512x720 aligned
  stack from a 720x512 binned raw stack, only possible with the axis along the long detector
  direction). The axis stays FIXED for all views (per-view `.xf` extraction remains reverted);
  everything else is unchanged: tilt-angle sign -1, view order identity, OFFSET +11.5 (baked,
  `LevelAngleY=0`), `LevelAngleX=-1.82`, SHIFT, volume orientation, `offsets_xy_A`.
  `WARP_AXIS_ANGLE_CONVENTION_VERSION` -> 5 (invalidates Warp XML / conversion / reconstruction /
  export caches).

- Reverts the per-view `.xf` tilt-axis extraction entirely. The raw path once again writes the
  FIXED align.com axis (`axis_input_angle` / `tilt_axis_angle_deg`, ~+84° for tomo2) to every
  view -- the behaviour prior to per-view extraction (commit 022bc22). The per-view experiment
  kept reversing the tilt-axis direction (a `/` slope rendered as `\`) across three rejected
  branches: `+180` (~+84.5°), the raw IMOD-layout polar `atan2(A21,A11)` (~-95.5°), and the `A.T`
  layout `atan2(A12,A11)` (~+95.5°). The extraction helpers (`warp_tilt_axis_angle_from_xf`,
  `warp_axis_angle_from_xf_layout`, `warp_axis_layout_matrix`) are removed; `imod_xf_rotation_angle_deg`
  stays as a diagnostic only. Everything else is UNCHANGED: tilt-angle sign -1, view order
  identity, OFFSET +11.5° (baked into Angles, `LevelAngleY=0`), `LevelAngleX=-1.82°`, SHIFT
  mapping, volume orientation, `offsets_xy_A` (`-inv(A)@d·angpix`), the aligned-frame per-view
  transform (a separate mechanism, untouched), and the revised-IMOD export. The manifest provenance
  records `source="fixed_aligncom_axis"` and `per_view_xf_axis_extraction_reverted=true`.
  `WARP_AXIS_ANGLE_CONVENTION_VERSION` -> 4 (invalidates Warp XML / conversion / reconstruction /
  export caches; any per-view axis marker is stale). The two entries below describe the reverted
  per-view work and are retained only as history.

- Corrects the reversed in-plane tilt-axis DIRECTION. The per-view Warp `TiltAxisAngle` is now
  extracted from Warp's OFFICIAL `.xf` layout -- the rotation built from `VecX=(A11,A21)`,
  `VecY=(A12,A22)` (i.e. `A.T`) fed to `EulerFromMatrix(...).Z`, which for the near-conformal
  tomo2 matrices equals `degrees(atan2(A12, A11))` (~**+95.5**; first real row +95.478°). The two
  earlier branches were both on the wrong side of 90° and reversed the axis (a `/` slope rendered
  as `\`): the `+180` branch (~+84.5°) and the raw IMOD-layout polar branch `atan2(A21,A11)`
  (~-95.5°). No `+180` adjustment and no branch normalisation toward the align.com 84.1° estimate;
  the exact Euler branch is stored (`axis_extraction_layout` records the `A.T`/VecX/VecY
  arrangement). This is the SAME `A.T` convention `offsets_xy_A` already uses, so axis and offset
  finally derive from one canonical matrix -- `offsets_xy_A` (`-inv(A)@d·angpix`) are unchanged.
  The polar-of-`A.T` extraction is scale-unbiased (a positive `.xf` scale error cannot flip `/`
  into `\`); `.xf` isotropic scale stays a separate concern. Preserved: tilt-angle sign -1, view
  order identity, OFFSET +11.5° (baked into Angles, `LevelAngleY=0`), `LevelAngleX=-1.82°`, SHIFT
  mapping, volume orientation, `tomo2.tlt`/`tomo2.xf` sources; no global X/Y flip, no stack
  reversal. `WARP_AXIS_ANGLE_CONVENTION_VERSION` -> 3 (invalidates Warp XML / conversion /
  reconstruction / export caches; a +84.5 or -95.5 axis marker is stale). The installed Warp
  `Matrix3`/`EulerFromMatrix` remains the cluster-side authority; the local `atan2` form is
  confirmed to agree exactly for all rows.

- Removes a double reversal of the directed tilt axis. `TiltAxisAngles` are now the source
  `.xf` polar rotation DIRECTLY (~-95.5° for tomo2), matching Warp's official
  `TiltSeries.ImportAlignments` (`EulerFromMatrix` assigned straight); the previous `+180°`
  adjustment (added because the tilt angles are negated) double-reversed the axis, since the
  `.xf` branch is already the reversed direction relative to the +84.5° branch. No `+180°`, no
  branch normalisation to 84.1 (kept only as align.com estimate/provenance/fallback);
  `axis_direction_adjustment_deg = 0`. OFFSET is now baked into `Angles = sign*(tlt+OFFSET)`
  with `LevelAngleY = 0` (applied once, sharing the tilt rotation order with `LevelAngleX`
  rather than a separate global rotation); tilt angles come from `tomo2.tlt`. Effective Warp
  tilt stays `-(tlt+OFFSET)` (−55.88°…+66.28°). The `-inv(A)@d` offsets, tilt-angle sign (−1),
  SHIFT (Warp `[0,0,+17.82]`, det +1 frame), `LevelAngleX = -1.82°` and view order are
  unchanged. Records the per-axis half-pixel rounding correction (4092/8=511.5→512 → Y pixel
  17.583 Å vs X 17.6 Å). Convention version bumped to 2 (invalidates conversion/reconstruction/
  export caches; a fixed-84.1 or `+180` marker is stale).

- Fixes two angular errors in the IMOD->Warp conversion. (1) The tilt-angle sign was inverted
  to -1 without reversing the tilt-axis DIRECTION; since `rotation(axis,theta) == rotation(-axis,
  -theta)`, the axis now gets a +180 deg reversal for sign -1. (2) `TiltAxisAngles` were a fixed
  align.com value (`[84.1]*41`, introduced by the raw-frame `axis_angles = [axis_input_angle]*n`),
  discarding the per-view `.xf` rotation; they are now extracted per view from each source `.xf`
  via polar decomposition (`imod_affine.warp_tilt_axis_angle_from_xf`), branch-selected near the
  align.com estimate (kept only as reference/fallback/provenance). For tomo2 this changes the
  fixed 84.1 to per-view 84.277-84.700 (mean 84.505). OFFSET stays applied exactly once
  (`Angles=sign*tlt`, `LevelAngleY=sign*OFFSET`, effective `= sign*(tlt+OFFSET)`), guarded by a
  conversion-time assertion; view order stays identity; translation-only refinement keeps the
  source `.xf` rotation. The per-view axis provenance (source axis, sign, 180 deg adjustment,
  final axis) + a hash of all final Warp axis angles + `WARP_AXIS_ANGLE_CONVENTION_VERSION` are
  recorded in the conversion manifest and threaded into the conversion/reconstruction/export
  cache identities (a fixed-84.1 marker is stale). The identity IMOD->Warp->IMOD round trip
  preserves `.tlt`/`.xf` from the preserved source affine (< 1e-6 / < 1e-4). The tilt-angle sign,
  SHIFT, XAXISTILT, `.xf` translation import and volume-frame permutation are unchanged.

- Corrects the IMOD reconstruction SHIFT into the Warp volume frame. The tilt-angle sign was
  changed to -1 without applying the corresponding signed 3-D frame transform to SHIFT. SHIFT
  is now built in native IMOD-MRC axis order `[sx_A, sz_A, 0]` (SHIFT Z is thickness -> MRC Y)
  and transformed ONCE with the signed `IMOD_MRC_TO_WARP` orientation (`[[1,0,0],[0,0,1],[0,-1,0]]`,
  det +1, hand-preserving for sign -1): `SHIFT Z -8.1 @ 2.2 A -> Warp [0,0,+17.82] A` (was
  `[0,0,-17.82]`). The shape permutation `(0,2,1)` is unchanged (it carries no signs); the signed
  matrix is added for coordinates/vectors. SHIFT stays the single global `apply_tomogram_shift_3d`
  representation (never added to the `.xf`-derived `TiltAxisOffsetX/Y`). The orientation matrix +
  determinant + handedness + both shift vectors are recorded in the conversion and export
  manifests; `POSITIONING_CONTRACT_VERSION` bumped to 2 (invalidates conversion/reconstruction
  caches). Adds the exact inverse `warp_shift_to_imod_reconstruction` (SHIFT Z from MRC component
  1). `angle_sign = +1` uses the corresponding det -1 policy, not the -1 matrix. The `.xf` import,
  view order and tilt-angle sign are unchanged.

- Corrects the IMOD->Warp tilt-angle convention. One canonical
  `IMOD_TO_WARP_TILT_ANGLE_SIGN = -1` (allowed -1/+1) is applied EXACTLY ONCE to both the
  per-view angles (`warp = sign * imod_raw`) and OFFSET (`LevelAngleY = sign * OFFSET`), so
  the effective angles are the sign-transformed IMOD effective angles. Direct-stack view
  order stays identity (Warp row i == source section i; no reversal, no per-view resorting).
  The sign flows through the resolved config, warp staging manifest, local + cluster
  conversion, the conversion/validation manifests (with `tilt_view_order` + `tilt_angle_convention`),
  the conversion/reconstruction/export cache identities (a sign +1 artefact is stale vs -1),
  and the revised-IMOD export, which applies the exact inverse (`imod = sign * warp`, its own
  inverse) read from the conversion manifest. `.xf` matrices are unaffected;
  `BASE_AXIS_PERMUTATION`, SHIFT, XAXISTILT/LevelAngleX and `AreAnglesInverted` are unchanged.
  Adds `scripts/pipeline/diagnose_tilt_angle_sign.py` (IMOD `clip rotx` vs Warp sign +1/-1
  reconstruction comparison; cluster-run generation, local NCC comparison).


- Preserves the full IMOD tilt.com tomogram-positioning geometry (THICKNESS, OFFSET,
  XAXISTILT, SHIFT) end to end. New canonical `geometry.imod_positioning` module (parser
  with tilt.com authority, `ImodPositioning`, documented IMOD->Warp conversion functions,
  numpy projection oracle, positioning hash) plus the guarded converter application
  (LevelAngleY/LevelAngleX/apply_tomogram_shift_3d) and `[geometry.imod_positioning]` config.
- Propagates the positioning contract through the whole pipeline: it is parsed at `setup`,
  carried on the `Geometry` dataclass, written to `[geometry.imod_positioning]` in the
  resolved `project_settings.toml`, threaded through the warp staging manifest, the local
  and cluster conversion (`run_warp_conversion.py`, `etomo_to_warp.process_tilt_series`) and
  the legacy `02_convert_using_params.py`, and recorded in the conversion/validation
  manifests. Its hash is part of the local cache identity and the cluster conversion marker,
  so any OFFSET/XAXISTILT/SHIFT/pixel change forces reconversion and stale markers are
  detected.
- Adds `convertMissalignment export revise` — a source-aware revised-IMOD export (original
  IMOD -> Warp -> MissAlignment -> revised IMOD). Both result backends converge into one
  typed `ImodAlignmentRevision`; `H_final = DeltaH @ H_original` is composed in the validated
  IMOD `(n-1)/2` centre convention. Publishes ONE physical `exported_data/imod/<condition_id>`
  (configuration/, data/ relative raw-stack symlink, `reconstruct_with_imod.sh`, manifest,
  per-tilt alignment-change report JSON/TSV/summary, `scipion_compatibility.json`) with a
  single compatibility symlink at `missalignment/runs/<condition_id>/export/imod`. Complete
  vs residual `.xf` are distinct; positioning is preserved unless refined; a non-affine
  refined mapping is rejected (`non_affine_policy = "fail"`); the reconstruction script and
  writer never touch `imported_data`. Configured under `[export.imod_revision]`; the optional
  Scipion audit is a compatibility check only.
- Adds canonical WarpTools reconstruction tiling (`--subvolume_size` 64, `--subvolume_padding`
  6, padding >= 6 enforced), isotropic-XYZ padded context (not overlap), LC_ALL=C locale for
  every WarpTools subprocess, reconstruction contract hash, resource preflight, pixel-size
  consistency checks and a quantitative seam diagnostic. Wired into both ts_reconstruct paths.
- Adds `scripts/pipeline/validate_warp_positioning.py` (cluster-side warpylib validation of the
  LevelAngleX sign and XML round trip). Warp geometry is not treated as validated until it passes.

## 0.1.14

- Corrects misleading wording in `inventory`. For conditions such as `raw_xf_affine_fixed`
  the affine from the IMOD `.xf` is applied by the import itself, so the pre-MissAlignment
  volume is already aligned. The stage is now called "RECONSTRUCTION AT IMPORT
  (pre-MissAlignment)" and states the alignment the import applied, read from the
  conversion manifest, making clear that "before" means before MissAlignment refines that
  alignment rather than an unaligned raw volume.

## 0.1.13

- `inventory` ends with a TOMOGRAMS section that states plainly which volume is the one
  before MissAlignment and which is the one after, and says that a completed full run
  refines the alignment without reconstructing a volume.
- The before volume now quotes the purpose the engine recorded beside it
  (`engine: Warp dataset geometry validation before MissAlignment`), so the stage a file
  belongs to no longer has to be guessed from the series name.

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
