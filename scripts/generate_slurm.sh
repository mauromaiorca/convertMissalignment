#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 12 ]; then
    echo "Usage:"
    echo "  $0 HELPERS_DIR OUT_DIR MISSALIGN_ENV CONDITIONS GPU_PARTITION GPU_CONSTRAINT SMOKE_TIME SMOKE_CPUS SMOKE_DATALOADERS STANDARD_TIME STANDARD_CPUS STANDARD_DATALOADERS"
    exit 1
fi

HELPERS_DIR="$1"
OUT_DIR="$2"
MISSALIGN_ENV="$3"
CONDITIONS="$4"
GPU_PARTITION="$5"
GPU_CONSTRAINT="$6"
SMOKE_TIME="$7"
SMOKE_CPUS="$8"
SMOKE_DATALOADERS="$9"
STANDARD_TIME="${10}"
STANDARD_CPUS="${11}"
STANDARD_DATALOADERS="${12}"

LOGS_DIR="${OUT_DIR}/logs"
MAXWELL_SBATCH_DIR="${OUT_DIR}/maxwell_sbatch"
SMOKE_RESULT_DIR="${OUT_DIR}/missalign_result_smoke"
STANDARD_RESULT_DIR="${OUT_DIR}/missalign_result_standard"

SMOKE_SCRIPT="${MAXWELL_SBATCH_DIR}/missalign_smoke.sbatch"
STANDARD_SCRIPT="${MAXWELL_SBATCH_DIR}/missalign_standard.sbatch"

echo
echo "### Generate Maxwell sbatch files ###"
echo "HELPERS_DIR:          ${HELPERS_DIR}"
echo "OUT_DIR:              ${OUT_DIR}"
echo "MISSALIGN_ENV:        ${MISSALIGN_ENV}"
echo "CONDITIONS:           ${CONDITIONS}"
echo "GPU_PARTITION:        ${GPU_PARTITION}"
echo "GPU_CONSTRAINT:       ${GPU_CONSTRAINT}"
echo "MAXWELL_SBATCH_DIR:   ${MAXWELL_SBATCH_DIR}"
echo "SMOKE_RESULT_DIR:     ${SMOKE_RESULT_DIR}"
echo "STANDARD_RESULT_DIR:  ${STANDARD_RESULT_DIR}"
echo

test -d "${HELPERS_DIR}"
test -d "${SMOKE_RESULT_DIR}"
test -d "${STANDARD_RESULT_DIR}"
test -f "${SMOKE_RESULT_DIR}/run_missalignment.sh"
test -f "${STANDARD_RESULT_DIR}/run_missalignment.sh"

mkdir -p "${LOGS_DIR}"
mkdir -p "${MAXWELL_SBATCH_DIR}"

cat > "${SMOKE_SCRIPT}" <<EOF_SMOKE
#!/usr/bin/env bash
#SBATCH --partition=${GPU_PARTITION}
#SBATCH --constraint=${GPU_CONSTRAINT}
#SBATCH --time=${SMOKE_TIME}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${SMOKE_CPUS}
#SBATCH --job-name=missalign_smoke
#SBATCH --output=${LOGS_DIR}/missalign_smoke_%j.log

set -euo pipefail

MISSALIGN_ENV="${MISSALIGN_ENV}"
RESULT_DIR="${SMOKE_RESULT_DIR}"

export PATH="\${MISSALIGN_ENV}/bin:\${PATH}"
export CUDA_VISIBLE_DEVICES=0
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

cd "\${RESULT_DIR}"

echo
echo "### MissAlignment smoke ###"
echo "Host: \$(hostname)"
echo "Date: \$(date)"
echo "Job ID: \${SLURM_JOB_ID}"
echo "RESULT_DIR: \${RESULT_DIR}"
echo "CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-unset}"
echo

nvidia-smi || true

echo
echo "### Python check ###"
which python
python -V
which miss-alignment

python -c 'import torch; print("torch:", torch.__version__); print("cuda build:", torch.version.cuda); print("cuda available:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'

echo
echo "### Run MissAlignment smoke ###"

bash "\${RESULT_DIR}/run_missalignment.sh"

echo
echo "### Smoke finished ###"
date
EOF_SMOKE

cat > "${STANDARD_SCRIPT}" <<EOF_STANDARD
#!/usr/bin/env bash
#SBATCH --partition=${GPU_PARTITION}
#SBATCH --constraint=${GPU_CONSTRAINT}
#SBATCH --time=${STANDARD_TIME}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${STANDARD_CPUS}
#SBATCH --job-name=missalign_standard
#SBATCH --output=${LOGS_DIR}/missalign_standard_%j.log

set -euo pipefail

MISSALIGN_ENV="${MISSALIGN_ENV}"
RESULT_DIR="${STANDARD_RESULT_DIR}"

export PATH="\${MISSALIGN_ENV}/bin:\${PATH}"
export CUDA_VISIBLE_DEVICES=0
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

cd "\${RESULT_DIR}"

echo
echo "### MissAlignment standard ###"
echo "Host: \$(hostname)"
echo "Date: \$(date)"
echo "Job ID: \${SLURM_JOB_ID}"
echo "RESULT_DIR: \${RESULT_DIR}"
echo "CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-unset}"
echo

nvidia-smi || true

echo
echo "### Python check ###"
which python
python -V
which miss-alignment

python -c 'import torch; print("torch:", torch.__version__); print("cuda build:", torch.version.cuda); print("cuda available:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'

echo
echo "### Run MissAlignment standard ###"

bash "\${RESULT_DIR}/run_missalignment.sh"

echo
echo "### Standard finished ###"
date
EOF_STANDARD

chmod +x "${SMOKE_SCRIPT}"
chmod +x "${STANDARD_SCRIPT}"

echo "Created:"
echo "  ${SMOKE_SCRIPT}"
echo "  ${STANDARD_SCRIPT}"
echo
echo "Run smoke with:"
echo "  sbatch ${SMOKE_SCRIPT}"
echo
echo "Run standard with:"
echo "  sbatch ${STANDARD_SCRIPT}"
