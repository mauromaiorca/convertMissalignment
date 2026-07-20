#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 7 ]; then
    echo "Usage:"
    echo "  $0 HELPERS_DIR DATA_DIR OUT_DIR MISSALIGN_ENV CONDITIONS SMOKE_DATALOADERS STANDARD_DATALOADERS"
    exit 1
fi

HELPERS_DIR="$1"
DATA_DIR="$2"
OUT_DIR="$3"
MISSALIGN_ENV="$4"
CONDITIONS="$5"
SMOKE_DATALOADERS="$6"
STANDARD_DATALOADERS="$7"

PYTHON="${MISSALIGN_ENV}/bin/python"
CONDA_SH="${MISSALIGN_CONDA_SH:-}"
if [[ -z "${CONDA_SH}" ]]; then
    if [[ -n "${CONDA_EXE:-}" ]]; then
        CONDA_ROOT="$(cd "$(dirname "${CONDA_EXE}")/.." && pwd)"
        CANDIDATE_CONDA_SH="${CONDA_ROOT}/etc/profile.d/conda.sh"
        [[ -f "${CANDIDATE_CONDA_SH}" ]] && CONDA_SH="${CANDIDATE_CONDA_SH}"
    elif command -v conda >/dev/null 2>&1; then
        CONDA_BIN="$(command -v conda)"
        CONDA_ROOT="$(cd "$(dirname "${CONDA_BIN}")/.." && pwd)"
        CANDIDATE_CONDA_SH="${CONDA_ROOT}/etc/profile.d/conda.sh"
        [[ -f "${CANDIDATE_CONDA_SH}" ]] && CONDA_SH="${CANDIDATE_CONDA_SH}"
    fi
fi

ETOMO_PARAM_DIR="${OUT_DIR}/etomo_input_params"
MISSALIGN_INPUT_TEMPLATE="${OUT_DIR}/missalign_input_template"
SCRIPTS_USED_DIR="${OUT_DIR}/scripts_used"
LOGS_DIR="${OUT_DIR}/logs"
MAXWELL_SBATCH_DIR="${OUT_DIR}/maxwell_sbatch"

SMOKE_RESULT_DIR="${OUT_DIR}/missalign_result_smoke"
STANDARD_RESULT_DIR="${OUT_DIR}/missalign_result_standard"

echo
echo "### MissAlignment input setup ###"
echo "HELPERS_DIR:              ${HELPERS_DIR}"
echo "DATA_DIR:                 ${DATA_DIR}"
echo "OUT_DIR:                  ${OUT_DIR}"
echo "MISSALIGN_ENV:            ${MISSALIGN_ENV}"
echo "PYTHON:                   ${PYTHON}"
echo "CONDITIONS:               ${CONDITIONS}"
echo "ETOMO_PARAM_DIR:          ${ETOMO_PARAM_DIR}"
echo "MISSALIGN_INPUT_TEMPLATE: ${MISSALIGN_INPUT_TEMPLATE}"
echo "SMOKE_RESULT_DIR:         ${SMOKE_RESULT_DIR}"
echo "STANDARD_RESULT_DIR:      ${STANDARD_RESULT_DIR}"
echo

echo "### Checks ###"

test -d "${HELPERS_DIR}"
test -d "${DATA_DIR}"
test -x "${PYTHON}"

test -f "${HELPERS_DIR}/01_extract_etomo_params.py"
test -f "${HELPERS_DIR}/02_convert_using_params.py"
test -f "${HELPERS_DIR}/03_run_missalignment.py"
test -f "${HELPERS_DIR}/etomo_to_warp.py"
test -f "${HELPERS_DIR}/generate_aligned_stack.py"
test -f "${HELPERS_DIR}/imod_affine.py"
test -f "${HELPERS_DIR}/warp_to_imod_affine.py"
test -f "${HELPERS_DIR}/export_condition_results.py"

echo "Checks OK."
echo

echo "### Create output folders ###"

mkdir -p "${ETOMO_PARAM_DIR}"
mkdir -p "${MISSALIGN_INPUT_TEMPLATE}"
mkdir -p "${SCRIPTS_USED_DIR}"
mkdir -p "${LOGS_DIR}"
mkdir -p "${MAXWELL_SBATCH_DIR}"

# Record the source path without creating a directory symlink.
printf '%s\n' "${DATA_DIR}" > "${OUT_DIR}/ORIGINAL_ETOMO_DATA_PATH.txt"

cp -p "${HELPERS_DIR}/01_extract_etomo_params.py" "${SCRIPTS_USED_DIR}/"
cp -p "${HELPERS_DIR}/02_convert_using_params.py" "${SCRIPTS_USED_DIR}/"
cp -p "${HELPERS_DIR}/03_run_missalignment.py" "${SCRIPTS_USED_DIR}/"
cp -p "${HELPERS_DIR}/etomo_to_warp.py" "${SCRIPTS_USED_DIR}/"
cp -p "${HELPERS_DIR}/generate_aligned_stack.py" "${SCRIPTS_USED_DIR}/"
cp -p "${HELPERS_DIR}/imod_affine.py" "${SCRIPTS_USED_DIR}/"
cp -p "${HELPERS_DIR}/warp_to_imod_affine.py" "${SCRIPTS_USED_DIR}/"
cp -p "${HELPERS_DIR}/export_condition_results.py" "${SCRIPTS_USED_DIR}/"

echo
echo "### Environment check ###"

"${PYTHON}" -c '
packages = ["torch", "torch_projectors", "warpylib", "miss_alignment", "mrcfile", "numpy"]
failed = []
for pkg in packages:
    try:
        __import__(pkg)
        print(f"{pkg:18s} OK")
    except Exception as exc:
        failed.append(pkg)
        print(f"{pkg:18s} FAILED: {type(exc).__name__}: {exc}")
if failed:
    raise SystemExit("Required Python imports failed: " + ", ".join(failed))
'

echo
echo "### Step 01: extract eTomo parameters ###"

"${PYTHON}" "${HELPERS_DIR}/01_extract_etomo_params.py" \
  --etomo-dir "${DATA_DIR}" \
  --out-dir "${ETOMO_PARAM_DIR}" \
  --overwrite

echo
echo "### Parameter report ###"

cat "${ETOMO_PARAM_DIR}/etomo_missalign_params.txt"

echo
echo "### Step 02: prepare MissAlignment input template ###"

convert_args=(
  --params "${ETOMO_PARAM_DIR}/etomo_missalign_params.json"
  --converter "${HELPERS_DIR}/etomo_to_warp.py"
  --out-dir "${MISSALIGN_INPUT_TEMPLATE}"
  --conditions ${CONDITIONS}
  --movement-grid-shape ${MISSALIGN_MOVEMENT_GRID_SHAPE:-5 5}
  --module-mode "${MISSALIGN_MODULE_MODE:-auto}"
  --imod-module "${MISSALIGN_IMOD_MODULE:-imod}"
  --run
  --overwrite
)
if [[ -n "${MISSALIGN_MODULE_INIT_SCRIPT:-}" ]]; then
  convert_args+=(--module-init-script "${MISSALIGN_MODULE_INIT_SCRIPT}")
fi
if [[ -n "${MISSALIGN_ALIGNED_STACK_OUTPUT:-}" ]]; then
  convert_args+=(--aligned-stack-output "${MISSALIGN_ALIGNED_STACK_OUTPUT}")
fi
if [[ "${MISSALIGN_GENERATE_ALIGNED_STACK:-1}" == "0" ]]; then
  convert_args+=(--no-generate-aligned-stack)
else
  convert_args+=(--generate-aligned-stack)
fi
"${PYTHON}" "${HELPERS_DIR}/02_convert_using_params.py" "${convert_args[@]}"

echo
echo "### MissAlignment input template ###"

find "${MISSALIGN_INPUT_TEMPLATE}" -maxdepth 5 -type f | sort

prepare_result_dir() {
    mode="$1"
    dataloaders="$2"
    result_dir="$3"

    echo
    echo "### Prepare ${mode} MissAlignment result directory ###"
    echo "RESULT_DIR: ${result_dir}"

    rm -rf "${result_dir}"
    mkdir -p "${result_dir}"

    for cond in ${CONDITIONS}; do
        src="${MISSALIGN_INPUT_TEMPLATE}/warp_${cond}"
        dst="${result_dir}/warp_${cond}"

        if [ ! -d "${src}" ]; then
            echo "ERROR: missing input condition: ${src}"
            exit 1
        fi

        mkdir -p "${dst}"

        cp -a "${src}/"*.xml "${dst}/"
        cp -a "${src}/"*.conversion.json "${dst}/" 2>/dev/null || true
        cp -a "${src}/config.yaml" "${dst}/" 2>/dev/null || true

        if [ ! -d "${src}/tiltstack" ]; then
            echo "ERROR: missing tiltstack: ${src}/tiltstack"
            exit 1
        fi

        # Never symlink directories. Recreate the directory tree locally, copy
        # small metadata files, and symlink only individual heavy data files.
        mkdir -p "${dst}/tiltstack"
        while IFS= read -r -d '' item; do
            rel="${item#${src}/tiltstack/}"
            target="${dst}/tiltstack/${rel}"
            if [ -d "${item}" ]; then
                mkdir -p "${target}"
                continue
            fi
            mkdir -p "$(dirname "${target}")"
            case "${item,,}" in
                *.mrc|*.st|*.ali|*.rec|*.map)
                    ln -sfn "$(readlink -f "${item}")" "${target}"
                    ;;
                *)
                    cp -a "${item}" "${target}"
                    ;;
            esac
        done < <(find "${src}/tiltstack" -mindepth 1 -print0)
    done

    echo
    echo "### Generate MissAlignment ${mode} command file ###"

    conda_args=()
    if [ -f "${CONDA_SH}" ]; then
        conda_args+=(--conda-sh "${CONDA_SH}" --conda-env "${MISSALIGN_ENV}")
    fi

    "${PYTHON}" "${HELPERS_DIR}/03_run_missalignment.py" \
      --params "${ETOMO_PARAM_DIR}/etomo_missalign_params.json" \
      --warp-parent "${result_dir}" \
      --conditions ${CONDITIONS} \
      --mode "${mode}" \
      --training-devices 0 \
      --reconstruction-devices 0 \
      --dataloaders-per-trainer "${dataloaders}" \
      --cuda-visible-devices 0 \
      --affine-fit-rms-tolerance-px "${MISSALIGN_AFFINE_FIT_RMS_TOLERANCE_PX:-0.10}" \
      --affine-fit-max-tolerance-px "${MISSALIGN_AFFINE_FIT_MAX_TOLERANCE_PX:-0.25}" \
      "${conda_args[@]}"

    if [ ! -f "${result_dir}/run_missalignment.sh" ]; then
        echo "ERROR: run_missalignment.sh was not created in ${result_dir}"
        exit 1
    fi
}

prepare_result_dir "smoke" "${SMOKE_DATALOADERS}" "${SMOKE_RESULT_DIR}"
prepare_result_dir "standard" "${STANDARD_DATALOADERS}" "${STANDARD_RESULT_DIR}"

echo
echo "### Input setup DONE ###"
echo "eTomo parameters:              ${ETOMO_PARAM_DIR}"
echo "MissAlignment input template:  ${MISSALIGN_INPUT_TEMPLATE}"
echo "Smoke result directory:        ${SMOKE_RESULT_DIR}"
echo "Standard result directory:     ${STANDARD_RESULT_DIR}"
