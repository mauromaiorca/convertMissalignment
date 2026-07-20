#!/usr/bin/env python3
"""
Create MissAlignment config files and optionally run MissAlignment for Warp datasets
created by 02_convert_using_params.py.

Important for eTomo -> etomo_to_warp.py workflows
-------------------------------------------------
The converted Warp folders already contain tiltstack/*.st files. Therefore, do
NOT pass --prepare-stacks unless you are starting from native Warp XML files with
movie paths. Passing --prepare-stacks to eTomo-converted XMLs can fail with:
"Tilt series has no movie paths".

On some HPC systems, PyTorch/TorchInductor may try to JIT-compile kernels and
fail if the conda compiler wrapper is absent. By default this script disables
TorchDynamo/TorchInductor for stability:
  TORCH_COMPILE_DISABLE=1
  TORCHDYNAMO_DISABLE=1

Example
-------
python 03_run_missalignment.py \
  --params ./missalign_params_lam8/etomo_missalign_params.json \
  --warp-parent ./missalign_result_smoke \
  --conditions raw_xf \
  --mode smoke \
  --training-devices 0 \
  --reconstruction-devices 0 \
  --dataloaders-per-trainer 1 \
  --cuda-visible-devices 0 \
  --conda-sh /path/to/conda/etc/profile.d/conda.sh \
  --conda-env /path/to/missalignment-environment \
  --clean \
  --run
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ALL_CONDITIONS = ["raw_identity", "raw_xf", "raw_xf_translation", "raw_xf_affine_fixed", "ali_identity"]
DEFAULT_CONDITIONS = ["raw_xf_affine_fixed", "ali_identity"]


def load_params(path: Path) -> dict[str, Any]:
    with path.open("r") as handle:
        return json.load(handle)


def config_text(training_directory: Path, mode: str, patch_size: int | None = None) -> str:
    training_directory_str = str(training_directory.resolve())

    if mode == "smoke":
        iter_settings = "    - { downsample: 1, alignment: global }"
        max_epochs = 2
        warmup_steps = 20
        milestones = "[1]"
        steps_per_epoch = 50
        batch_size = 8
        patch = patch_size or 64
        align_batch = 8
        patch_overlap = 0.25
        traj = 4.0
        jitter = 1.0
        outlier = 6.0
        fracture = 6.0
    elif mode == "standard":
        iter_settings = "    - { downsample: 2, alignment: anchoring }\n    - { downsample: 1, alignment: global }\n    - { downsample: 1, alignment: global }"
        max_epochs = 10
        warmup_steps = 200
        milestones = "[5]"
        steps_per_epoch = 500
        batch_size = 16
        patch = patch_size or 96
        align_batch = 16
        patch_overlap = 0.10
        traj = 10.0
        jitter = 2.0
        outlier = 20.0
        fracture = 20.0
    elif mode == "translation":
        # Image-based per-tilt translation == MissAlignment's native `global` alignment.
        iter_settings = "    - { downsample: 2, alignment: global }\n    - { downsample: 1, alignment: global }"
        max_epochs = 10
        warmup_steps = 200
        milestones = "[5]"
        steps_per_epoch = 500
        batch_size = 16
        patch = patch_size or 96
        align_batch = 16
        patch_overlap = 0.10
        traj = 10.0
        jitter = 2.0
        outlier = 20.0
        fracture = 20.0
    elif mode in ("rigid", "similarity"):
        # Image-based constrained refinement. `alignment: rigid`/`similarity` is a
        # constrained alignment type consumed by the differentiable forward pass
        # (constrained -> detector movement field; see IMAGE_BASED_CONSTRAINED_INTEGRATION.md).
        # REQUIRES the MissAlignment fork with the constrained integration; stock
        # MissAlignment understands only global/anchoring/[N,N]. Warm-started from
        # global (and rigid for similarity).
        if mode == "rigid":
            iter_settings = ("    - { downsample: 2, alignment: global }\n"
                             "    - { downsample: 2, alignment: rigid }\n"
                             "    - { downsample: 1, alignment: rigid }")
        else:
            iter_settings = ("    - { downsample: 2, alignment: global }\n"
                             "    - { downsample: 2, alignment: rigid }\n"
                             "    - { downsample: 1, alignment: rigid }\n"
                             "    - { downsample: 1, alignment: similarity }")
        max_epochs = 10
        warmup_steps = 200
        milestones = "[5]"
        steps_per_epoch = 500
        batch_size = 16
        patch = patch_size or 96
        align_batch = 16
        patch_overlap = 0.10
        traj = 10.0
        jitter = 2.0
        outlier = 20.0
        fracture = 20.0
    elif mode == "affine2d":
        # Experimental: a final 2x2 movement-grid refinement can alter rotation, scale and shear.
        # Export is accepted only when warp_to_imod_affine.py reports an affine residual below tolerance.
        iter_settings = "    - { downsample: 2, alignment: anchoring }\n    - { downsample: 1, alignment: global }\n    - { downsample: 1, alignment: [2, 2] }"
        max_epochs = 10
        warmup_steps = 200
        milestones = "[5]"
        steps_per_epoch = 1000
        batch_size = 16
        patch = patch_size or 96
        align_batch = 16
        patch_overlap = 0.10
        traj = 10.0
        jitter = 2.0
        outlier = 20.0
        fracture = 20.0
    else:  # pragma: no cover
        raise ValueError(mode)

    return f"""general:
  training_directory: {training_directory_str}
  apply_ctf: False
  iteration_settings:
{iter_settings}
  seed: 45132

model_training:
  model_architecture: 'default'
  model_checkpoint: null
  loss_margin: 0.5
  learning_rate: 1.0e-3
  weight_decay: 1.0e-4
  max_epochs_per_iteration: {max_epochs}
  warmup_steps: {warmup_steps}
  multistep_lr_scheduler:
    milestones: {milestones}
    gamma: 0.5

data_loading:
  batch_size: {batch_size}
  patch_size: {patch}
  steps_per_epoch: {steps_per_epoch}

shift_generation:
  trajectory_probability: .5
  trajectory_max_shift: {traj}
  jitter_probability: .5
  jitter_max_std: {jitter}
  outlier_probability: .5
  outlier_max_shift: {outlier}
  fracture_probability: .5
  fracture_max_shift: {fracture}

tilt_series_alignment:
  patch_size: {patch}
  patch_overlap: {patch_overlap}
  batch_size: {align_batch}
"""


def shell_quote(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def missalignment_command(
    config_path: Path,
    training_devices: str,
    reconstruction_devices: str,
    dataloaders_per_trainer: int,
    prepare_stacks: float | None,
    start_at_iteration: int | None,
    executable: str,
) -> list[str]:
    cmd = [
        executable,
        "--config-file",
        str(config_path),
        "--training-devices",
        training_devices,
        "--reconstruction-devices",
        reconstruction_devices,
        "--dataloaders-per-trainer",
        str(dataloaders_per_trainer),
    ]
    if start_at_iteration is not None:
        cmd.extend(["--start-at-iteration", str(start_at_iteration)])
    if prepare_stacks is not None:
        cmd.extend(["--prepare-stacks", str(prepare_stacks)])
    return cmd


def check_warp_dir(warp_dir: Path) -> None:
    if not warp_dir.exists():
        raise SystemExit(f"ERROR: Warp directory does not exist: {warp_dir}")
    xmls = list(warp_dir.glob("*.xml"))
    stacks = list((warp_dir / "tiltstack").glob("*/*.st")) if (warp_dir / "tiltstack").exists() else []
    if not xmls:
        raise SystemExit(f"ERROR: no XML files found in {warp_dir}")
    if not stacks:
        raise SystemExit(f"ERROR: no tiltstack/*/*.st files found in {warp_dir}")


def clean_warp_dir(warp_dir: Path) -> None:
    for child in warp_dir.iterdir():
        if child.is_dir() and child.name.startswith("iter"):
            shutil.rmtree(child)
    for name in ["models"]:
        p = warp_dir / name
        if p.exists():
            shutil.rmtree(p)
    for pattern in ["model.ckpt", "*.log", "*_alignment_loss.json"]:
        for p in warp_dir.glob(pattern):
            if p.is_file() or p.is_symlink():
                p.unlink()


def build_run_env(disable_torch_compile: bool, cuda_visible_devices: str | None) -> dict[str, str]:
    env = os.environ.copy()
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    if disable_torch_compile:
        env["TORCH_COMPILE_DISABLE"] = "1"
        env["TORCHDYNAMO_DISABLE"] = "1"
    return env


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create configs and run MissAlignment on converted Warp datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--params", required=True, type=Path, help="JSON from 01_extract_etomo_params.py.")
    parser.add_argument("--warp-parent", required=True, type=Path, help="Parent containing warp_raw_xf, warp_ali_identity, etc.")
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS, choices=ALL_CONDITIONS)
    parser.add_argument("--mode", choices=["smoke", "standard", "translation", "rigid", "similarity", "affine2d"], default="standard")
    parser.add_argument("--patch-size", type=int, default=None, help="Override patch_size in generated configs.")
    parser.add_argument("--affine-fit-rms-tolerance-px", type=float, default=0.10)
    parser.add_argument("--affine-fit-max-tolerance-px", type=float, default=0.25)
    parser.add_argument("--training-devices", default="0", help="Passed to miss-alignment --training-devices.")
    parser.add_argument("--reconstruction-devices", default="0", help="Passed to miss-alignment --reconstruction-devices.")
    parser.add_argument("--dataloaders-per-trainer", type=int, default=1)
    parser.add_argument(
        "--prepare-stacks",
        type=float,
        default=None,
        help="Only use for native Warp XMLs with movie paths. Leave unset for etomo_to_warp.py converted tiltstacks.",
    )
    parser.add_argument("--start-at-iteration", type=int, default=0)
    parser.add_argument("--executable", default="miss-alignment", help="MissAlignment executable name/path.")
    parser.add_argument("--cuda-visible-devices", default=None, help="Optional CUDA_VISIBLE_DEVICES value, e.g. 0.")
    parser.add_argument(
        "--disable-torch-compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable TorchDynamo/TorchInductor. Recommended on HPC unless compiler wrappers are installed.",
    )
    parser.add_argument("--conda-sh", type=Path, default=None, help="Optional conda.sh to source in generated run script.")
    parser.add_argument("--conda-env", default=None, help="Optional conda env name or full path to activate in generated run script.")
    parser.add_argument("--clean", action="store_true", help="Remove previous iter*, models, logs, and loss files before running.")
    parser.add_argument("--run", action="store_true", help="Actually run MissAlignment. Without this, only writes configs and run script.")
    args = parser.parse_args()

    _params = load_params(args.params.resolve())  # retained for provenance/checking; current script does not need values by default
    warp_parent = args.warp_parent.resolve()
    if not warp_parent.exists():
        raise SystemExit(f"ERROR: --warp-parent does not exist: {warp_parent}")

    if args.prepare_stacks is not None:
        print(
            "WARNING: --prepare-stacks was requested. This is usually wrong for eTomo-converted "
            "tiltstacks from etomo_to_warp.py and may fail with 'Tilt series has no movie paths'."
        )

    commands: list[tuple[str, list[str], Path]] = []
    for condition in args.conditions:
        warp_dir = warp_parent / f"warp_{condition}"
        check_warp_dir(warp_dir)
        config_path = warp_dir / "config.yaml"
        config_path.write_text(config_text(warp_dir, args.mode, patch_size=args.patch_size))
        cmd = missalignment_command(
            config_path=config_path,
            training_devices=args.training_devices,
            reconstruction_devices=args.reconstruction_devices,
            dataloaders_per_trainer=args.dataloaders_per_trainer,
            prepare_stacks=args.prepare_stacks,
            start_at_iteration=args.start_at_iteration,
            executable=args.executable,
        )
        commands.append((condition, cmd, warp_dir))

    run_script = warp_parent / "run_missalignment.sh"
    with run_script.open("w") as handle:
        handle.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        if args.conda_sh is not None:
            handle.write(f"source {args.conda_sh}\n")
        if args.conda_env is not None:
            handle.write(f"conda activate {args.conda_env}\n")
        if args.cuda_visible_devices is not None:
            handle.write(f"export CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}\n")
        if args.disable_torch_compile:
            handle.write("export TORCH_COMPILE_DISABLE=1\n")
            handle.write("export TORCHDYNAMO_DISABLE=1\n")
        handle.write("\n")
        for condition, cmd, warp_dir in commands:
            log_path = warp_dir / "missalignment.log"
            handle.write(f"echo '=== Running {condition} ==='\n")
            if args.clean:
                # Quote the directory prefix so paths with spaces are safe; the
                # trailing globs stay outside the quotes so the shell expands them.
                wd = shlex.quote(str(warp_dir))
                handle.write(f"rm -rf {wd}/iter* {wd}/models {wd}/model.ckpt {wd}/*.log {wd}/*_alignment_loss.json\n")
            handle.write(shell_quote(cmd) + f" 2>&1 | tee {str(log_path)!r}\n\n")
    run_script.chmod(0o755)

    export_helper = Path(__file__).resolve().parent / "export_condition_results.py"
    export_script = warp_parent / "export_results.sh"
    with export_script.open("w") as handle:
        handle.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        for condition, _cmd, warp_dir in commands:
            output_dir = warp_parent / "exports" / condition
            export_cmd = [
                sys.executable, str(export_helper),
                "--params", str(args.params.resolve()),
                "--warp-dir", str(warp_dir),
                "--condition", condition,
                "--out-dir", str(output_dir),
                "--rms-tolerance-px", str(args.affine_fit_rms_tolerance_px),
                "--max-tolerance-px", str(args.affine_fit_max_tolerance_px),
            ]
            handle.write(shell_quote(export_cmd) + "\n")
    export_script.chmod(0o755)

    print(f"Wrote MissAlignment configs and run script: {run_script}")
    print(f"Wrote validated export script: {export_script}")
    for condition, cmd, warp_dir in commands:
        print(f"\n[{condition}] config: {warp_dir / 'config.yaml'}")
        print(shell_quote(cmd))

    if args.run:
        if shutil.which(args.executable) is None and not Path(args.executable).exists():
            raise SystemExit(f"ERROR: executable not found: {args.executable}")
        env = build_run_env(args.disable_torch_compile, args.cuda_visible_devices)
        for condition, cmd, warp_dir in commands:
            if args.clean:
                clean_warp_dir(warp_dir)
            log_path = warp_dir / "missalignment.log"
            print(f"\n=== Running {condition}; log: {log_path} ===")
            with log_path.open("w") as log:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
                assert proc.stdout is not None
                for line in proc.stdout:
                    print(line, end="")
                    log.write(line)
                ret = proc.wait()
            if ret != 0:
                raise SystemExit(f"ERROR: MissAlignment failed for {condition} with exit code {ret}.")
        print("\nMissAlignment runs complete.")
    else:
        print("\nMissAlignment was not run. To run it now:")
        print(f"  bash {run_script}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
