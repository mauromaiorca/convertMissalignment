#!/usr/bin/env python3
"""Generate v8 semantic Slurm jobs and helpers (spec §25-§28, §33).

Pure text generation — nothing here runs Slurm/CUDA/MissAlignment. Every job
carries: ``set -Eeuo pipefail`` + an ERR trap that writes a failure summary, a
diagnostic preamble (env/devices/versions), and a background resource monitor.
The generated jobs are validated STATICALLY by tests (no execution).
"""
from __future__ import annotations

import json
import shlex
from types import SimpleNamespace
from pathlib import Path
import hashlib

from .runlayout import RunLayout

PYTHON = "python"  # resolved on the cluster via the activated environment


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_existing(path: Path) -> str:
    return _sha256_file(path) if Path(path).is_file() else "unavailable"




def _preamble(job_name: str, layout: RunLayout, *, record_failure: bool = False) -> str:
    rd = layout.run_dir
    helper = layout.helpers_dir / "record_missalignment_result.py"
    failure_hook = ""
    if record_failure:
        failure_hook = f'''\n  if [[ -x {shlex.quote(str(helper))} ]]; then\n    {PYTHON} {shlex.quote(str(helper))} --project-root "$RUN_DIR" --status failed \\
      --result-manifest {shlex.quote(str(layout.manifest("result_manifest.json")))} \\
      --run-manifest {shlex.quote(str(layout.manifest("missalignment_run_manifest.json")))} \\
      --smoke-verdict {shlex.quote(str(layout.results_dir / "smoke_verdict.json"))} \\
      --failed-line "$line" --failed-command "$cmd" || true\n  fi\n'''
    return f'''set -Eeuo pipefail
RUN_DIR={shlex.quote(str(rd))}
JOB_NAME={shlex.quote(job_name)}
JOB_ID="${{SLURM_JOB_ID:-local-$$}}"
ENV_JSON="$RUN_DIR/logs/environment/${{JOB_NAME}}_${{SLURM_JOB_ID:-local}}.json"
mkdir -p "$RUN_DIR/logs/environment" "$RUN_DIR/logs/resources" "$RUN_DIR/.internal/diagnostics/postmortem"

on_error() {{
  rc="$1"; line="$2"; cmd="$3"
  pm="$RUN_DIR/.internal/diagnostics/postmortem"
  {{
    echo "{{"
    echo "  \\"job\\": \\"$JOB_NAME\\", \\"job_id\\": \\"$JOB_ID\\","
    echo "  \\"return_code\\": $rc, \\"failed_line\\": $line,"
    echo "  \\"failed_command\\": \\"$(echo "$cmd" | sed 's/\\\\/\\\\\\\\/g; s/\"/\\\\\"/g')\\","
    echo "  \\"hostname\\": \\"$(hostname -f 2>/dev/null || hostname)\\","
    echo "  \\"timestamp\\": \\"$(date --iso-8601=seconds 2>/dev/null || date)\\""
    echo "}}"
  }} > "$pm/${{JOB_NAME}}_failure.json" || true
{failure_hook}  echo "[error] $JOB_NAME failed rc=$rc at line $line: $cmd" >&2
  exit "$rc"
}}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR
'''


def _diagnostics(job_name: str, layout: RunLayout) -> str:
    helper = layout.helpers_dir / "env_report.py"
    return f'''
JOB_PYTHON="${{PIPELINE_PYTHON:-{PYTHON}}}"
echo "===== ENVIRONMENT: $JOB_NAME ($JOB_ID) ====="
echo "date=$(date --iso-8601=seconds 2>/dev/null || date)"
echo "hostname=$(hostname -f 2>/dev/null || hostname)"
echo "partition=${{SLURM_JOB_PARTITION:-}}"
echo "cpus=${{SLURM_CPUS_PER_TASK:-}}"
echo "PATH=$PATH"
( module list ) 2>&1 || true
"$JOB_PYTHON" --version 2>&1 || true
for program in submfg newstack tilt ctfphaseflip miss-alignment WarpTools WarpWorker; do
  resolved="$(command -v "$program" 2>/dev/null || true)"
  [[ -n "$resolved" ]] && echo "$program=$resolved"
done
nvidia-smi -L 2>/dev/null || true
df -h "$RUN_DIR" || true
"$JOB_PYTHON" {shlex.quote(str(helper))} "$JOB_NAME" > "$ENV_JSON" 2>/dev/null || true
echo "===== END ENVIRONMENT ====="
'''


def _monitor(job_name: str, layout: RunLayout) -> str:
    checkpoints = layout.missalignment_run_dir / "checkpoints"
    return f'''
MON_TSV="$RUN_DIR/logs/resources/${{JOB_NAME}}_${{JOB_ID}}.tsv"
echo -e "ts\\tgpu_util\\tgpu_mem_mb\\tgpu_temp\\tcpu_load\\trss_kb\\tproject_kb\\tlatest_ckpt" > "$MON_TSV"
__monitor() {{
  while true; do
    ts="$(date +%s)"
    gpu="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo 'NA,NA,NA')"
    load="$(cut -d' ' -f1 /proc/loadavg 2>/dev/null || echo NA)"
    rss="$(ps -o rss= -p $$ 2>/dev/null | tr -d ' ' || echo NA)"
    odir="$(du -sk "$RUN_DIR" 2>/dev/null | cut -f1 || echo NA)"
    ckpt="$(ls -t {shlex.quote(str(checkpoints))}/* 2>/dev/null | head -1 || echo none)"
    echo -e "$ts\\t${{gpu//,/\\t}}\\t$load\\t$rss\\t$odir\\t$ckpt" >> "$MON_TSV"
    sleep "${{MONITOR_INTERVAL:-30}}"
  done
}}
__monitor & MON_PID=$!
cleanup_monitor() {{ kill "$MON_PID" 2>/dev/null || true; }}
trap cleanup_monitor EXIT
'''


def _sbatch_header(job_name: str, *, gpu: bool, profile: str, time: str, cpus: int,
                   log_dir: Path, cluster=None) -> str:
    c = cluster
    part = (getattr(c, "partition", None) or "vds") if gpu else getattr(c, "cpu_partition", None)
    lines = [f"#SBATCH --job-name={job_name}"]
    if part:
        lines.append(f"#SBATCH --partition={part}")
    if gpu:
        gres = getattr(c, "gres", None) if c else None
        if profile == "maxwell" and part == "vds" and gres == "gpu:1":
            gres = None
        if gres:
            lines.append(f"#SBATCH --gres={gres}")
        constraint = getattr(c, "constraint", None) or "V100"
        if constraint:
            lines.append(f"#SBATCH --constraint={constraint}")
    lines.append(f"#SBATCH --time={time}")
    lines.append(f"#SBATCH --cpus-per-task={cpus}")
    for attr, flag in (("memory", "--mem"), ("account", "--account"),
                       ("qos", "--qos"), ("nodelist", "--nodelist")):
        value = getattr(c, attr, None) if c else None
        if value:
            lines.append(f"#SBATCH {flag}={value}")
    lines.append(f"#SBATCH --output={log_dir}/{job_name}_%j.out")
    lines.append(f"#SBATCH --error={log_dir}/{job_name}_%j.err")
    lines.append(f"# cluster profile: {profile}")
    return "#!/usr/bin/env bash\n" + "\n".join(lines) + "\n"

def _imod_cluster(cluster, reconstruction_config: dict | None):
    cluster_options = ((reconstruction_config or {}).get("cluster") or {})
    source = getattr(cluster, "__dict__", {}) if cluster else {}
    data = dict(source)
    data["cpu_partition"] = cluster_options.get("partition") or source.get("cpu_partition")
    data["memory"] = cluster_options.get("memory") or source.get("memory")
    data["account"] = cluster_options.get("account") or source.get("account")
    data["qos"] = cluster_options.get("qos") or source.get("qos")
    data["nodelist"] = cluster_options.get("nodelist") or source.get("nodelist")
    data["cpus"] = int(cluster_options.get("cpus_per_task") or source.get("cpus") or 16)
    return SimpleNamespace(**data), cluster_options


def _warptools_cluster(cluster, reconstruction_config: dict | None):
    """GPU resources for the diagnostic WarpTools pre/full reconstruction."""
    rec = reconstruction_config or {}
    wt = dict(rec.get("warptools", {}) or {})
    cluster_options = dict(rec.get("warptools_cluster", {}) or wt.get("cluster", {}) or {})
    source = getattr(cluster, "__dict__", {}) if cluster else {}
    data = dict(source)
    data["partition"] = cluster_options.get("partition") or source.get("partition") or "vds"
    data["constraint"] = cluster_options.get("constraint") or source.get("constraint") or "V100"
    data["gres"] = cluster_options.get("gres") if "gres" in cluster_options else source.get("gres")
    data["memory"] = cluster_options.get("memory") or "128G"
    data["account"] = cluster_options.get("account") or source.get("account")
    data["qos"] = cluster_options.get("qos") or source.get("qos")
    data["nodelist"] = cluster_options.get("nodelist") or source.get("nodelist")
    data["cpus"] = int(cluster_options.get("cpus_per_task") or 16)
    data["time"] = str(cluster_options.get("time") or "24:00:00")
    return SimpleNamespace(**data), cluster_options


def _warptools_activation(cluster) -> str:
    """Strict WarpTools + scientific Python activation for the GPU job."""
    module_init = (
        getattr(cluster, "module_init_script", None)
        or "/usr/share/Modules/init/bash"
    )
    warp_module = getattr(cluster, "warp_module", None) or "warp/2.0.39"
    environment = getattr(cluster, "environment", None) or ""
    python_exe = str(Path(environment) / "bin" / "python") if environment else "python"
    return f'''
# ---- WarpTools reconstruction environment activation ----
export PATH="/usr/local/bin:/usr/bin:/bin:${{PATH:-}}"
if [[ -r {shlex.quote(module_init)} ]]; then
  source {shlex.quote(module_init)}
elif [[ -r /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
else
  echo "ERROR: Environment Modules initialisation not found" >&2
  exit 2
fi
module purge
module load {shlex.quote(warp_module)}
export LC_ALL=C
export LANG=C
export DOTNET_CLI_TELEMETRY_OPTOUT=1
PIPELINE_PYTHON={shlex.quote(python_exe)}
if [[ ! -x "$PIPELINE_PYTHON" ]]; then
  echo "ERROR: reconstruction Python is not executable: $PIPELINE_PYTHON" >&2
  exit 2
fi
command -v WarpTools >/dev/null 2>&1 || {{ echo "ERROR: WarpTools not found" >&2; exit 2; }}
command -v WarpWorker >/dev/null 2>&1 || {{ echo "ERROR: WarpWorker not found" >&2; exit 2; }}
"$PIPELINE_PYTHON" - <<'PY_WT_PREFLIGHT'
import sys
import mrcfile
import numpy
import warpylib
print("WarpTools reconstruction Python:", sys.executable)
print("NumPy:", numpy.__version__)
print("mrcfile:", mrcfile.__version__)
print("warpylib:", getattr(warpylib, "__version__", "unknown"))
PY_WT_PREFLIGHT
'''


def _missalignment_activation(cluster) -> str:
    """GPU/Warp/MissAlignment activation from [cluster]."""
    if cluster is None:
        return "\n# (no [cluster] env config; relying on a pre-activated shell)\n"
    out = ["", "# ---- MissAlignment environment activation (from [cluster]) ----"]
    if getattr(cluster, "module_init_script", None):
        out.append(f'[ -f {shlex.quote(cluster.module_init_script)} ] && source {shlex.quote(cluster.module_init_script)} || true')
    if getattr(cluster, "warp_module", None):
        out.append(f'module load {shlex.quote(cluster.warp_module)} 2>/dev/null || true')
    if getattr(cluster, "environment", None):
        env = cluster.environment
        out.append(f'if [ -f {shlex.quote(env + "/bin/activate")} ]; then source {shlex.quote(env + "/bin/activate")}; '
                   f'else export PATH={shlex.quote(env + "/bin")}:"$PATH"; fi')
    if getattr(cluster, "omp_num_threads", None):
        out.append(f'export OMP_NUM_THREADS={cluster.omp_num_threads}')
    if getattr(cluster, "cuda_visible_devices", None):
        out.append(f'export CUDA_VISIBLE_DEVICES={cluster.cuda_visible_devices}')
    out.append("which miss-alignment || true; which WarpTools || true; which WarpWorker || true")
    out.append("")
    return "\n".join(out) + "\n"


def _imod_activation(cluster, reconstruction_config: dict | None) -> str:
    """Activate IMOD while keeping the configured scientific Python explicit."""
    rec = reconstruction_config or {}
    imod = (rec.get("imod") or {})
    module_init = getattr(cluster, "module_init_script", None) or imod.get("module_init_script") or "/usr/share/Modules/init/bash"
    imod_module = imod.get("imod_module") or getattr(cluster, "imod_module", None) or "imod/5.1.11"
    env = str(getattr(cluster, "environment", None) or "")
    python_exe = str(Path(env) / "bin" / "python") if env else "python"
    imod_bin = imod.get("imod_bin_dir") or getattr(cluster, "imod_bin_dir", None) or ""
    return f'''
# ---- IMOD reconstruction environment activation ----
export PATH="/usr/local/bin:/usr/bin:/bin:${{PATH:-}}"
if [[ -r {shlex.quote(module_init)} ]]; then
  source {shlex.quote(module_init)}
elif [[ -r /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
else
  echo "ERROR: Environment Modules initialisation not found" >&2
  exit 2
fi
module purge
module load {shlex.quote(imod_module)}
PIPELINE_PYTHON={shlex.quote(python_exe)}
if [[ {shlex.quote(env)} != "" ]]; then
  export PATH={shlex.quote(env + "/bin")}:"$PATH"
fi
if [[ "$PIPELINE_PYTHON" != */* ]]; then
  PIPELINE_PYTHON="$(command -v "$PIPELINE_PYTHON" 2>/dev/null || true)"
fi
if [[ -z "$PIPELINE_PYTHON" || ! -x "$PIPELINE_PYTHON" ]]; then
  echo "ERROR: configured reconstruction Python is not executable: {python_exe}" >&2
  exit 2
fi
IMOD_BIN={shlex.quote(imod_bin)}
if [[ -n "$IMOD_BIN" ]]; then export PATH="$IMOD_BIN:$PATH"; fi
for program in submfg newstack tilt; do
  command -v "$program" >/dev/null 2>&1 || {{
    echo "ERROR: missing IMOD executable: $program" >&2
    exit 2
  }}
done
"$PIPELINE_PYTHON" - <<'PY_IMOD_PREFLIGHT'
import sys
if sys.version_info < (3, 11):
    raise SystemExit(
        f"ERROR: IMOD reconstruction requires Python >=3.11; got {{sys.version.split()[0]}} "
        f"at {{sys.executable}}"
    )
import tomllib
import mrcfile
import numpy
import warpylib
print("IMOD reconstruction Python:", sys.executable)
print("Python:", sys.version.split()[0])
print("NumPy:", numpy.__version__)
print("mrcfile:", mrcfile.__version__)
print("warpylib:", getattr(warpylib, "__version__", "unknown"))
PY_IMOD_PREFLIGHT
'''




def _smoke_verdict_helper() -> str:
    return r'''#!/usr/bin/env python3
import argparse, json, pathlib, re
ap = argparse.ArgumentParser()
ap.add_argument("--training-dir", required=True)
ap.add_argument("--log", required=True)
ap.add_argument("--mode", required=True)
ap.add_argument("--output", required=True)
a = ap.parse_args()
training = pathlib.Path(a.training_dir)
log = pathlib.Path(a.log)
checks = {}
xmls = [p for p in training.glob("*.xml") if p.is_file() and p.stat().st_size > 0]
iters = [p for p in training.glob("iter*") if p.is_dir()]
ckpts = [p for p in training.rglob("*.ckpt") if p.is_file() and p.stat().st_size > 0]
checks["root_xml_count"] = len(xmls)
checks["iteration_dirs"] = len(iters)
checks["checkpoints"] = len(ckpts)
checks["log_nonempty"] = log.is_file() and log.stat().st_size > 0
text = log.read_text(errors="ignore") if log.is_file() else ""
checks["no_nan_inf_in_log"] = not bool(re.search(r"(?<![A-Za-z])(?:nan|inf)(?![A-Za-z])", text, re.I))
checks["same_project_has_tiltstack"] = bool(list((training / "tiltstack").glob("*/*.st")))
ok = (len(xmls) == 1 and len(iters) >= 1 and len(ckpts) >= 1 and
      checks["log_nonempty"] and checks["no_nan_inf_in_log"] and
      checks["same_project_has_tiltstack"])
verdict = {
    "smoke": "ok" if ok else "failed",
    "mode": a.mode,
    "training_directory": str(training),
    "log": str(log),
    "checks": checks,
    "xml": str(xmls[0]) if len(xmls) == 1 else None,
    "checkpoints": [str(p) for p in ckpts],
    "iteration_directories": [str(p) for p in iters],
}
out = pathlib.Path(a.output)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(verdict, indent=2) + "\n")
print("smoke verdict:", verdict["smoke"], checks)
raise SystemExit(0 if ok else 1)
'''

def _env_report_helper() -> str:
    return '''#!/usr/bin/env python3
"""Tiny env reporter invoked from the Slurm preamble (no repo imports required)."""
import json, os, sys, platform
name = sys.argv[1] if len(sys.argv) > 1 else "job"
rep = {"job": name, "sys_executable": sys.executable, "python_version": sys.version,
       "sys_path": sys.path, "platform": platform.platform(),
       "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
try:
    import torch
    rep["torch_version"] = torch.__version__
    rep["cuda_available"] = bool(torch.cuda.is_available())
    rep["cuda_build"] = getattr(torch.version, "cuda", None)
    if torch.cuda.is_available():
        rep["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
except Exception as e:
    rep["torch_error"] = str(e)
for mod in ("warpylib", "torch_projectors", "miss_alignment"):
    try:
        import importlib.util
        spec = importlib.util.find_spec(mod)
        rep[mod + "_path"] = spec.origin if spec else None
    except Exception as e:
        rep[mod + "_error"] = str(e)
print(json.dumps(rep, indent=2, default=str))
'''




def _result_helper() -> str:
    return r'''#!/usr/bin/env python3
import argparse, json, os, pathlib, socket, tempfile
from datetime import datetime, timezone

def atomic_json(path, data):
    path = pathlib.Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    with os.fdopen(fd, "w") as handle:
        json.dump(data, handle, indent=2); handle.write("\n")
    os.replace(tmp, path)

def load(path):
    path = pathlib.Path(path)
    return json.loads(path.read_text()) if path.is_file() else {}

ap = argparse.ArgumentParser()
ap.add_argument("--project-root", required=True)
ap.add_argument("--status", choices=("completed", "failed"), required=True)
ap.add_argument("--result-manifest", required=True)
ap.add_argument("--run-manifest", required=True)
ap.add_argument("--smoke-verdict", required=True)
ap.add_argument("--command-log", default="")
ap.add_argument("--failed-line", default="")
ap.add_argument("--failed-command", default="")
a = ap.parse_args()
result_path = pathlib.Path(a.result_manifest)
run_path = pathlib.Path(a.run_manifest)
result = load(result_path)
smoke_path = pathlib.Path(a.smoke_verdict)
smoke_data = load(smoke_path)
record = {
    "status": a.status,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    "hostname": socket.getfqdn() or socket.gethostname(),
    "smoke_performed": smoke_path.is_file() and smoke_path.stat().st_size > 0,
    "smoke_verdict": str(smoke_path),
    "smoke_result": smoke_data,
    "result_manifest": str(result_path),
    "completed_at": datetime.now(timezone.utc).isoformat(),
}
if a.status == "failed":
    record.update({"failed_line": a.failed_line, "failed_command": a.failed_command})
    atomic_json(run_path, record)
    raise SystemExit(0)
full_dir = pathlib.Path(result.get("training_directory", ""))
xmls = sorted(p for p in full_dir.glob("*.xml") if p.is_file() and p.stat().st_size > 0)
if len(xmls) != 1:
    record.update({"status": "failed", "error": f"expected one final XML in {full_dir}, found {len(xmls)}"})
    atomic_json(run_path, record)
    raise SystemExit(record["error"])
result.update({
    "final_xml": str(xmls[0]),
    "missalignment_completed_at": record["completed_at"],
    "missalignment_slurm_job_id": record["slurm_job_id"],
    "missalignment_hostname": record["hostname"],
})
atomic_json(result_path, result)
record["final_xml"] = str(xmls[0])
atomic_json(run_path, record)
print("recorded final XML:", xmls[0])
'''


def generate_jobs(layout: RunLayout, *, profile: str = "maxwell",
                  ma_command: str, run_script: str, settings_path: str,
                  working_recon: bool = False, halfmaps: bool = False,
                  smoke_command: str = "", cluster=None,
                  warp_staging_manifest: str = "",
                  reconstruction_config: dict | None = None,
                  include_import: bool = True,
                  preprocess_command: str = "") -> dict:
    # Generate all v8 batches for one Warp pixel-size dataset.
    layout.create()
    written: dict[str, str] = {}
    helpers = layout.helpers_dir
    helpers.mkdir(parents=True, exist_ok=True)

    def helper(name: str, text: str) -> Path:
        path = helpers / name
        path.write_text(text)
        path.chmod(0o755)
        written[f"helper:{name}"] = str(path)
        return path

    helper("env_report.py", _env_report_helper())
    smoke_helper = helper("smoke_verdict.py", _smoke_verdict_helper())
    result_helper = helper("record_missalignment_result.py", _result_helper())
    probe_src = Path(__file__).resolve().parents[2] / "tools" / "cluster_capability_probe.py"
    probe = helpers / "cluster_capability_probe.py"
    if probe_src.is_file():
        probe.write_text(probe_src.read_text()); probe.chmod(0o755)
        written["helper:cluster_capability_probe.py"] = str(probe)
    clone_src = Path(__file__).resolve().parents[1] / "clone_warp_projects.py"
    clone = helpers / "clone_warp_projects.py"
    prepare_input_script = Path(__file__).resolve().parents[2] / "prepare_missalignment_input.py"
    if clone_src.is_file():
        clone.write_text(clone_src.read_text()); clone.chmod(0o755)
        written["helper:clone_warp_projects.py"] = str(clone)

    def write_batch(category: str, name: str, body: str) -> Path:
        path = layout.batch_path(category, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
        path.chmod(0o755)
        key = f"{category}/{name}" if category == "import" else f"{category}/{layout.dataset_id}/{name}"
        written[key] = str(path)
        return path

    repo_root = Path(__file__).resolve().parents[2]
    conversion_script = repo_root / "scripts" / "run_warp_conversion.py"
    pre_recon_script = Path(__file__).resolve().parent / "pre_conversion_reconstruction.py"
    warp_recon_script = Path(__file__).resolve().parent / "warptools_reconstruction.py"
    imod_recon_script = Path(__file__).resolve().parent / "imod_reconstruction.py"
    imported_imod_recon_script = Path(__file__).resolve().parent / "imported_imod_reconstruction.py"
    staging = warp_staging_manifest or str(layout.manifest("warp_staging_manifest.json"))
    smoke_cmd = smoke_command or ma_command
    gpu_activation = _missalignment_activation(cluster)
    rec = reconstruction_config or {}
    wt_cfg = dict(rec.get("warptools", {}) or {})
    wt_cluster, wt_cluster_cfg = _warptools_cluster(cluster, rec)
    wt_time = str(wt_cluster_cfg.get("time") or getattr(wt_cluster, "time", "24:00:00"))
    wt_cpus = int(wt_cluster_cfg.get("cpus_per_task") or getattr(wt_cluster, "cpus", 16) or 16)
    wt_activation = _warptools_activation(cluster)
    output_angpix = wt_cfg.get("output_angpix_A", 0.0)
    device_list = str(wt_cfg.get("device_list", "0"))
    perdevice = int(wt_cfg.get("perdevice", 1))

    if include_import:
        name = "import_imod_to_warp"
        write_batch(
            "import", name + ".sbatch",
            _sbatch_header(name, gpu=True, profile=profile, time=wt_time, cpus=wt_cpus,
                           log_dir=layout.log_dir("import"), cluster=wt_cluster)
            + _preamble(name, layout) + wt_activation + _diagnostics(name, layout)
            + _monitor(name, layout) + f'''
echo "[import] converting IMOD geometry into Warp dataset {layout.dataset_id}"
STAGING={shlex.quote(staging)}
[[ -s "$STAGING" ]] || {{ echo "ERROR: missing staging manifest: $STAGING" >&2; exit 2; }}
FORCE_ARG=""
[[ "${{FORCE_WARP_CONVERSION:-0}}" == "1" ]] && FORCE_ARG="--force"
"$PIPELINE_PYTHON" {shlex.quote(str(conversion_script))} \\
  --staging-manifest "$STAGING" --training-dir {shlex.quote(str(layout.training_dir))} $FORCE_ARG
test -s {shlex.quote(str(layout.training_dir / "_converted.marker"))}
echo "[import] complete: {layout.warp_dataset_dir}"
''')

    if include_import and rec.get("enabled", True):
        imported_cluster, imported_cfg = _imod_cluster(cluster, rec)
        imported_activation = _imod_activation(cluster, rec)
        name = "reconstruct_imported_imod"
        write_batch(
            "import", name + ".sbatch",
            _sbatch_header(name, gpu=False, profile=profile,
                           time=str(imported_cfg.get("time") or "08:00:00"),
                           cpus=int(imported_cfg.get("cpus_per_task") or getattr(imported_cluster, "cpus", 16) or 16),
                           log_dir=layout.log_dir("import"), cluster=imported_cluster)
            + _preamble(name, layout) + imported_activation + _diagnostics(name, layout)
            + f'''
"$PIPELINE_PYTHON" {shlex.quote(str(imported_imod_recon_script))} \
  --project-settings {shlex.quote(settings_path)}
''')

    if preprocess_command:
        name = "preprocess"
        preprocess_cluster, preprocess_cfg = _imod_cluster(cluster, rec)
        preprocess_activation = _imod_activation(cluster, rec)
        write_batch(
            "warp_data", name + ".sbatch",
            _sbatch_header(name + "_" + layout.dataset_id, gpu=False, profile=profile,
                           time=str(preprocess_cfg.get("time") or "08:00:00"),
                           cpus=int(preprocess_cfg.get("cpus_per_task") or getattr(preprocess_cluster, "cpus", 16) or 16),
                           log_dir=layout.log_dir("warp_data"), cluster=preprocess_cluster)
            + _preamble(name, layout) + preprocess_activation + _diagnostics(name, layout)
            + f'''\necho "[preprocess] creating {layout.dataset_id}"\n{preprocess_command}\necho "[preprocess] complete"\n''')

    name = "reconstruct_warp_dataset"
    write_batch(
        "warp_data", "reconstruct.sbatch",
        _sbatch_header(name + "_" + layout.dataset_id, gpu=True, profile=profile,
                       time=wt_time, cpus=wt_cpus, log_dir=layout.log_dir("warp_data"),
                       cluster=wt_cluster)
        + _preamble(name, layout) + wt_activation + _diagnostics(name, layout)
        + _monitor(name, layout) + f'''
echo "[reconstruct] Warp dataset {layout.dataset_id}"
OUTPUT_ANGPIX="${{OUTPUT_ANGPIX:-{output_angpix}}}"
"$PIPELINE_PYTHON" {shlex.quote(str(pre_recon_script))} run \\
  --project-settings {shlex.quote(settings_path)} --dataset {shlex.quote(layout.dataset_id)} \\
  --output-angpix "$OUTPUT_ANGPIX" --device-list {shlex.quote(device_list)} --perdevice {perdevice}
echo "[reconstruct] complete and technically validated: {layout.warp_reconstructions_dir}"
echo "[reconstruct] next: {PYTHON} {shlex.quote(str(prepare_input_script))} --directory {shlex.quote(str(layout.run_dir))}"
''')

    name = "prepare_missalignment_input"
    write_batch(
        "missalignment", "prepare_input.sbatch",
        _sbatch_header(name + "_" + layout.dataset_id, gpu=False, profile=profile,
                       time="00:30:00", cpus=2,
                       log_dir=layout.log_dir("missalignment"), cluster=cluster)
        + _preamble(name, layout) + _diagnostics(name, layout) + f'''
# Recovery-only wrapper. Normal use is the synchronous command printed after
# reconstruction validation; this batch does not require CUDA or WarpTools.
{PYTHON} {shlex.quote(str(prepare_input_script))} \
  --directory {shlex.quote(str(layout.run_dir))} \
  --dataset {shlex.quote(layout.dataset_id)}
''')

    name = "run_missalignment_smoke"
    write_batch(
        "missalignment", "run_smoke.sbatch",
        _sbatch_header(name + "_" + layout.dataset_id, gpu=True, profile=profile,
                       time="04:00:00", cpus=int(getattr(cluster, "cpus", 16) or 16),
                       log_dir=layout.log_dir("missalignment"), cluster=cluster)
        + _preamble(name, layout) + gpu_activation + _diagnostics(name, layout)
        + _monitor(name, layout) + f'''
[[ -d {shlex.quote(str(layout.smoke_warp_dir))} ]] || {{ echo "ERROR: run prepare_missalignment_input.py first" >&2; exit 2; }}
SMOKE_LOG={shlex.quote(str(layout.log_dir("missalignment")))}/smoke_${{JOB_ID}}.log
{smoke_cmd} 2>&1 | tee "$SMOKE_LOG"
{PYTHON} {shlex.quote(str(smoke_helper))} \\
  --training-dir {shlex.quote(str(layout.smoke_warp_dir))} --log "$SMOKE_LOG" \\
  --mode {shlex.quote(layout.refinement_mode)} --output {shlex.quote(str(layout.results_dir / "smoke_verdict.json"))}
''')

    name = "run_missalignment_full"
    write_batch(
        "missalignment", "run_full.sbatch",
        _sbatch_header(name + "_" + layout.dataset_id, gpu=True, profile=profile,
                       time=getattr(cluster, "time", None) or "7-00:00:00",
                       cpus=int(getattr(cluster, "cpus", 16) or 16),
                       log_dir=layout.log_dir("missalignment"), cluster=cluster)
        + _preamble(name, layout, record_failure=True) + gpu_activation
        + _diagnostics(name, layout) + _monitor(name, layout) + f'''
VERDICT={shlex.quote(str(layout.results_dir / "smoke_verdict.json"))}
[[ -d {shlex.quote(str(layout.full_warp_dir))} ]] || {{ echo "ERROR: run prepare_missalignment_input.py first" >&2; exit 2; }}
if [[ -s "$VERDICT" ]]; then
  echo "[full] smoke verdict found: $VERDICT"
else
  echo "[full] WARNING: no smoke verdict found; proceeding directly with the full run"
  echo "[full] smoke testing is recommended but optional"
fi
FULL_LOG={shlex.quote(str(layout.log_dir("missalignment")))}/full_${{JOB_ID}}.log
{ma_command} 2>&1 | tee "$FULL_LOG"
{PYTHON} {shlex.quote(str(result_helper))} --project-root {shlex.quote(str(layout.run_dir))} \\
  --status completed --result-manifest {shlex.quote(str(layout.manifest("result_manifest.json")))} \\
  --run-manifest {shlex.quote(str(layout.manifest("missalignment_run_manifest.json")))} \\
  --smoke-verdict "$VERDICT" --command-log "$FULL_LOG"
''')

    if wt_cfg.get("enabled", True):
        name = "compare_warp_reconstructions"
        write_batch(
            "missalignment", "compare_reconstructions.sbatch",
            _sbatch_header(name + "_" + layout.dataset_id, gpu=True, profile=profile,
                           time=wt_time, cpus=wt_cpus, log_dir=layout.log_dir("missalignment"),
                           cluster=wt_cluster)
            + _preamble(name, layout) + wt_activation + _diagnostics(name, layout)
            + _monitor(name, layout) + f'''
OUTPUT_ANGPIX="${{OUTPUT_ANGPIX:-{output_angpix}}}"
"$PIPELINE_PYTHON" {shlex.quote(str(warp_recon_script))} run \\
  --project-settings {shlex.quote(settings_path)} --dataset {shlex.quote(layout.dataset_id)} \\
  --output-angpix "$OUTPUT_ANGPIX" --device-list {shlex.quote(device_list)} --perdevice {perdevice} \\
  --expected-executor-sha {_sha256_existing(warp_recon_script)} \\
  --expected-settings-sha {_sha256_existing(Path(settings_path))}
''')

    if rec.get("enabled", True):
        cpu_cluster, cpu_cfg = _imod_cluster(cluster, rec)
        cpu_time = str(cpu_cfg.get("time") or "08:00:00")
        cpu_cpus = int(cpu_cfg.get("cpus_per_task") or getattr(cpu_cluster, "cpus", 16) or 16)
        activation = _imod_activation(cluster, rec)
        executor_sha = _sha256_existing(imod_recon_script)
        settings_sha = _sha256_existing(Path(settings_path))
        for snapshot, filename in (("pre_missalign", "reconstruct_before.sbatch"),
                                   ("smoke", "reconstruct_smoke.sbatch")):
            name = "imod_" + snapshot
            write_batch(
                "missalignment", filename,
                _sbatch_header(name + "_" + layout.dataset_id, gpu=False, profile=profile,
                               time=cpu_time, cpus=cpu_cpus,
                               log_dir=layout.log_dir("missalignment"), cluster=cpu_cluster)
                + _preamble(name, layout) + activation + _diagnostics(name, layout) + f'''
"$PIPELINE_PYTHON" {shlex.quote(str(imod_recon_script))} run \\
  --project-settings {shlex.quote(settings_path)} --dataset {shlex.quote(layout.dataset_id)} \\
  --snapshot {snapshot} --expected-executor-sha {executor_sha} \\
  --expected-settings-sha {settings_sha}
''')
        name = "export_imod_and_reconstruct"
        export_wr_script = repo_root / "export_warp_to_imod.py"
        revised_recon_script = layout.exported_imod_dir / "reconstruct_with_imod.sh"
        write_batch(
            "export", "export_imod_and_reconstruct.sbatch",
            _sbatch_header(name + "_" + layout.dataset_id, gpu=False, profile=profile,
                           time=cpu_time, cpus=cpu_cpus, log_dir=layout.log_dir("export"),
                           cluster=cpu_cluster)
            + _preamble(name, layout) + activation + _diagnostics(name, layout) + f'''
# 1. canonical revised-IMOD export -> exported_data/imod/{layout.dataset_id}
#    (final .xf = DeltaH @ H_original, residual .xf, revised .tlt/.xtilt/tilt.com/newst.com,
#     manifest, change reports, reconstruct_with_imod.sh). Needs a completed finalize.
"$PIPELINE_PYTHON" {shlex.quote(str(export_wr_script))} revise \\
  {shlex.quote(settings_path)} --dataset {shlex.quote(layout.dataset_id)}

# 2. run the generated reconstruction script (revised newstack + tilt on the ORIGINAL raw
#    stack via the data/ symlink; never writes under imported_data/imod).
RECON_SCRIPT={shlex.quote(str(revised_recon_script))}
if [[ -x "$RECON_SCRIPT" ]]; then
  "$RECON_SCRIPT"
else
  echo "ERROR: revised reconstruction script missing: $RECON_SCRIPT" >&2
  exit 4
fi
''')

    return written
