from __future__ import annotations

import shlex
from pathlib import Path

from .config import ClusterConfig
from .stages import StageSpec


def _sbatch_header(stage: StageSpec, cluster: ClusterConfig, run_dir: Path) -> str:
    gpu = stage.resources.partition_kind == "gpu"
    partition = cluster.gpu_partition if gpu else cluster.cpu_partition
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={stage.stage_id}",
    ]
    if partition:
        lines.append(f"#SBATCH --partition={partition}")
    if gpu:
        gres = cluster.gres or ""
        if not (cluster.profile == "maxwell" and cluster.gpu_partition == "vds" and gres == "gpu:1"):
            if gres:
                lines.append(f"#SBATCH --gres={gres}")
        if cluster.gpu_constraint:
            lines.append(f"#SBATCH --constraint={cluster.gpu_constraint}")
    lines.extend([
        f"#SBATCH --time={stage.resources.time or (cluster.time_gpu if gpu else cluster.time_cpu)}",
        f"#SBATCH --cpus-per-task={stage.resources.cpus or cluster.cpus}",
        f"#SBATCH --output={run_dir}/logs/stages/{stage.stage_id}_%j.out",
        f"#SBATCH --error={run_dir}/logs/stages/{stage.stage_id}_%j.err",
    ])
    if cluster.memory:
        lines.append(f"#SBATCH --mem={cluster.memory}")
    if cluster.account:
        lines.append(f"#SBATCH --account={cluster.account}")
    if cluster.qos:
        lines.append(f"#SBATCH --qos={cluster.qos}")
    return "\n".join(lines) + "\n"


def _activation(cluster: ClusterConfig) -> str:
    lines = ["", "# environment/module activation"]
    if cluster.module_init_script:
        lines.append(f'[ -f {shlex.quote(cluster.module_init_script)} ] && source {shlex.quote(cluster.module_init_script)} || true')
    if cluster.environment:
        env = cluster.environment
        lines.append(f'if [ -f {shlex.quote(env + "/bin/activate")} ]; then source {shlex.quote(env + "/bin/activate")}; else export PATH={shlex.quote(env + "/bin")}:"$PATH"; fi')
    if cluster.imod_module:
        lines.append(f"module load {shlex.quote(cluster.imod_module)} 2>/dev/null || true")
    return "\n".join(lines) + "\n"


def _body(stage: StageSpec, settings_path: Path, toml_hash: str, run_dir: Path, cluster: ClusterConfig) -> str:
    quoted_settings = shlex.quote(str(settings_path))
    repo_root = Path(__file__).resolve().parents[2]
    stage_result = repo_root / "scripts" / "v6" / "stage_result.py"
    executor = repo_root / "scripts" / "v6" / "execute_stage.py"
    return f'''set -Eeuo pipefail
RUN_DIR={shlex.quote(str(run_dir))}
STAGE_ID={shlex.quote(stage.stage_id)}
SETTINGS={quoted_settings}
EXPECTED_TOML_HASH={shlex.quote(toml_hash)}
mkdir -p "$RUN_DIR/logs/stages" "$RUN_DIR/logs/environment" "$RUN_DIR/logs/resources" "$RUN_DIR/manifests"
on_error() {{
  rc="$1"; line="$2"; cmd="$3"
  python3 {shlex.quote(str(stage_result))} --run-dir "$RUN_DIR" --stage-id "$STAGE_ID" \
    --status failed --exit-code "$rc" --failed-command "$cmd" \
    --log-path "$RUN_DIR/logs/stages/${{STAGE_ID}}_${{SLURM_JOB_ID:-local}}.out" || true
  echo "[v6:$STAGE_ID] FAILED rc=$rc line=$line command=$cmd" >&2
  exit "$rc"
}}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR
{_activation(cluster)}
echo "[v6:$STAGE_ID] environment activated"
date --iso-8601=seconds 2>/dev/null || date
hostname -f 2>/dev/null || hostname
env | grep -Ev 'TOKEN|KEY|SECRET|PASSWORD|AUTH|COOKIE' | sort > "$RUN_DIR/logs/environment/${{STAGE_ID}}_${{SLURM_JOB_ID:-local}}.env" || true
which WarpTools || true
which miss-alignment || true
which newstack || true
nvidia-smi 2>/dev/null || true
echo "[v6:$STAGE_ID] START {stage.scientific_purpose}"
python3 {shlex.quote(str(executor))} --stage "$STAGE_ID" --settings "$SETTINGS" \
  --run-dir "$RUN_DIR" --expected-toml-hash "$EXPECTED_TOML_HASH"
echo "stage_id={stage.stage_id}"
echo "input_snapshot={stage.input_snapshot}"
echo "output_snapshot={stage.output_snapshot}"
echo "validation={stage.validation_function}"
echo "[v6:$STAGE_ID] DONE"
'''


JOB_NAMES = {
    "10_warp_ingest": "10_warp_ingest.sbatch",
    "20_initial_alignment_and_qc": "20_initial_alignment_and_qc.sbatch",
    "30_missalignment": "30_missalignment.sbatch",
    "40_warp_postprocess": "40_warp_postprocess.sbatch",
    "50_particle_export": "50_particle_export.sbatch",
}


def generate_stage_jobs(
    *,
    jobs_dir: Path,
    run_dir: Path,
    settings_path: Path,
    toml_hash: str,
    cluster: ClusterConfig,
    stages: list[StageSpec],
) -> dict[str, str]:
    jobs_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for stage in stages:
        name = JOB_NAMES[stage.stage_id]
        path = jobs_dir / name
        path.write_text(_sbatch_header(stage, cluster, run_dir) +
                        _body(stage, settings_path, toml_hash, run_dir, cluster))
        path.chmod(0o755)
        written[stage.stage_id] = str(path)
    return written
