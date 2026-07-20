#!/usr/bin/env bash
# Validate the constrained integration on the real cluster.
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="${1:-$(pwd)/_probe_run}"; mkdir -p "$RUN_DIR"
python "$HERE/../../tools/cluster_capability_probe.py" --run-dir "$RUN_DIR" \
  --require-cuda --require-missalignment --require-modes translation rigid similarity
python - <<'PY'
import miss_alignment.constrained_integration as c
assert set(("translation","rigid","similarity")) <= set(c.SUPPORTED_ALIGNMENTS), c.SUPPORTED_ALIGNMENTS
print("constrained integration import OK; modes:", c.SUPPORTED_ALIGNMENTS)
PY
echo "[probe] OK"
