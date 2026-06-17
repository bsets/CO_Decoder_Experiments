#!/usr/bin/env bash
# Run HGS large-graph maximum clique decoder for one selected dataset.
# Runs 3-feature model followed by 10-feature model, with checkpoint/resume,
# progress Excel logging, and optional automatic AWS shutdown.
#
# Example:
#   ./run_hgs_large_graph_dataset_3_then_10_checkpoint_progress_then_shutdown.sh --graph_dataset com-amazon
#
# Run from GitHub repo clone on AWS:
#   cd ~/AWS_poc/repo/CO_Decoder_Experiments/hgs_aws/scripts
#   ./run_hgs_large_graph_dataset_3_then_10_checkpoint_progress_then_shutdown.sh --graph_dataset com-amazon

set -u
set -o pipefail

usage() {
  cat <<'EOF'
Usage:
  run_hgs_large_graph_dataset_3_then_10_checkpoint_progress_then_shutdown.sh --graph_dataset DATASET [options]

Required:
  --graph_dataset DATASET          Example: com-amazon, com-dblp, web-google, as-skitter, ca-dblp-2012

Optional, defaults match the Com-DBLP run:
  --candidate_limit N              Default: 1000
  --graph_timeout_sec N            Default: 10800
  --checkpoint_every N             Default: 10
  --decoder R|P|both               Default: both
  --walkers_ratio FLOAT            Default: 0.02
  --hourly_rate_usd FLOAT          Default: 0.714
  --instance_type NAME             Default: c7i.4xlarge
  --code_dir PATH                  Default: auto-detect repo hgs_aws/code, else ~/AWS_poc/code
  --aws_root PATH                  Default: ~/AWS_poc
  --shutdown_after_minutes N       Default: 10
  --no_shutdown                    Do not schedule shutdown at the end
  --help                           Show this help
EOF
}

GRAPH_DATASET=""
CANDIDATE_LIMIT=1000
GRAPH_TIMEOUT_SEC=10800
CHECKPOINT_EVERY=10
DECODER="both"
WALKERS_RATIO=0.02
HOURLY_RATE_USD=0.714
INSTANCE_TYPE="c7i.4xlarge"
AWS_ROOT="${HOME}/AWS_poc"
CODE_DIR=""
SHUTDOWN_AFTER_MINUTES=10
DO_SHUTDOWN=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --graph_dataset)
      GRAPH_DATASET="$2"; shift 2 ;;
    --candidate_limit)
      CANDIDATE_LIMIT="$2"; shift 2 ;;
    --graph_timeout_sec)
      GRAPH_TIMEOUT_SEC="$2"; shift 2 ;;
    --checkpoint_every)
      CHECKPOINT_EVERY="$2"; shift 2 ;;
    --decoder)
      DECODER="$2"; shift 2 ;;
    --walkers_ratio|--num_passes_as_percentage_of_total_nodes)
      WALKERS_RATIO="$2"; shift 2 ;;
    --hourly_rate_usd)
      HOURLY_RATE_USD="$2"; shift 2 ;;
    --instance_type)
      INSTANCE_TYPE="$2"; shift 2 ;;
    --code_dir)
      CODE_DIR="$2"; shift 2 ;;
    --aws_root)
      AWS_ROOT="$2"; shift 2 ;;
    --shutdown_after_minutes)
      SHUTDOWN_AFTER_MINUTES="$2"; shift 2 ;;
    --no_shutdown)
      DO_SHUTDOWN=0; shift ;;
    --help|-h)
      usage; exit 0 ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ -z "${GRAPH_DATASET}" ]]; then
  echo "ERROR: --graph_dataset is required. Example: --graph_dataset com-amazon" >&2
  usage
  exit 2
fi

# Filesystem-safe dataset label for logs/run labels.
DATASET_SLUG="$(echo "${GRAPH_DATASET}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+|_+$//g')"
if [[ -z "${DATASET_SLUG}" ]]; then
  DATASET_SLUG="dataset"
fi

# Auto-detect the code directory.
# Preferred: run from repo hgs_aws/scripts, with sibling ../code.
if [[ -z "${CODE_DIR}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ -f "${SCRIPT_DIR}/../code/run_hgs_large_graph_decoder.py" ]]; then
    CODE_DIR="$(cd "${SCRIPT_DIR}/../code" && pwd)"
  elif [[ -f "${AWS_ROOT}/code/run_hgs_large_graph_decoder.py" ]]; then
    CODE_DIR="${AWS_ROOT}/code"
  else
    echo "ERROR: Could not find run_hgs_large_graph_decoder.py." >&2
    echo "Tried: ${SCRIPT_DIR}/../code and ${AWS_ROOT}/code" >&2
    exit 2
  fi
fi

SCRIPT="${CODE_DIR}/run_hgs_large_graph_decoder.py"
PREPOSS_ROOT="${AWS_ROOT}/data/Preposs"
DATA_ROOT="${AWS_ROOT}/data/online_graphs"
RESULTS_ROOT="${AWS_ROOT}/results"
LOG_DIR="${AWS_ROOT}/logs"
CHECKPOINT_DIR="${RESULTS_ROOT}/checkpoints"
PROGRESS_XLSX="${RESULTS_ROOT}/hgs_decoder_progress_by_interval.xlsx"
SUMMARY_XLSX="${RESULTS_ROOT}/hgs_large_graph_decoder_summary.xlsx"

MODEL_3="${AWS_ROOT}/trained_models/trained_hgs_model_BHOSLIB_and_DIMACS_epochs_12_penalty_coefficient_0.06_lr_0.0004_3features.pth"
MODEL_10="${AWS_ROOT}/trained_models/trained_hgs_model_BHOSLIB_and_DIMACS_epochs_12_penalty_coefficient_0.06_lr_0.0004_10features.pth"

source "${HOME}/venvs/hgs-aws/bin/activate"

mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}" "${RESULTS_ROOT}"

# Cancel any previous scheduled shutdown so an old timer does not stop this run early.
sudo shutdown -c 2>/dev/null || true

RUN_TS="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${LOG_DIR}/hgs_${DATASET_SLUG}_3_then_10_checkpoint_progress_${RUN_TS}.log"

# Basic safety checks.
if [[ ! -f "${SCRIPT}" ]]; then
  echo "ERROR: Missing decoder script: ${SCRIPT}" | tee -a "${MASTER_LOG}"
  exit 2
fi
if [[ ! -f "${MODEL_3}" ]]; then
  echo "ERROR: Missing 3-feature model: ${MODEL_3}" | tee -a "${MASTER_LOG}"
  exit 2
fi
if [[ ! -f "${MODEL_10}" ]]; then
  echo "ERROR: Missing 10-feature model: ${MODEL_10}" | tee -a "${MASTER_LOG}"
  exit 2
fi

if ! python3 "${SCRIPT}" --help | grep -q -- "--graph_dataset"; then
  echo "ERROR: Decoder script does not expose --graph_dataset." | tee -a "${MASTER_LOG}"
  exit 2
fi
if ! python3 "${SCRIPT}" --help | grep -q -- "--checkpoint_every"; then
  echo "ERROR: Decoder script does not expose --checkpoint_every." | tee -a "${MASTER_LOG}"
  exit 2
fi

echo "============================================================" | tee -a "${MASTER_LOG}"
echo "Starting HGS large-graph decoder dataset run" | tee -a "${MASTER_LOG}"
echo "Dataset: ${GRAPH_DATASET}" | tee -a "${MASTER_LOG}"
echo "Code dir: ${CODE_DIR}" | tee -a "${MASTER_LOG}"
echo "Script: ${SCRIPT}" | tee -a "${MASTER_LOG}"
echo "AWS root: ${AWS_ROOT}" | tee -a "${MASTER_LOG}"
echo "Candidate limit: ${CANDIDATE_LIMIT}" | tee -a "${MASTER_LOG}"
echo "Graph timeout seconds: ${GRAPH_TIMEOUT_SEC}" | tee -a "${MASTER_LOG}"
echo "Checkpoint every: ${CHECKPOINT_EVERY}" | tee -a "${MASTER_LOG}"
echo "Decoder: ${DECODER}" | tee -a "${MASTER_LOG}"
echo "Walkers ratio: ${WALKERS_RATIO}" | tee -a "${MASTER_LOG}"
echo "Instance type: ${INSTANCE_TYPE}" | tee -a "${MASTER_LOG}"
echo "Hourly rate USD: ${HOURLY_RATE_USD}" | tee -a "${MASTER_LOG}"
echo "Started at: $(date)" | tee -a "${MASTER_LOG}"
echo "============================================================" | tee -a "${MASTER_LOG}"

cd "${CODE_DIR}"

run_case() {
  local FEATURES="$1"
  local MODEL_PATH="$2"
  local RUN_LABEL="aws_hgs_${DATASET_SLUG}_${FEATURES}features_cl${CANDIDATE_LIMIT}_t${GRAPH_TIMEOUT_SEC}"
  local CASE_LOG="${LOG_DIR}/${RUN_LABEL}_checkpoint_progress.log"

  echo "============================================================" | tee -a "${MASTER_LOG}" "${CASE_LOG}"
  echo "Starting case: ${RUN_LABEL}" | tee -a "${MASTER_LOG}" "${CASE_LOG}"
  echo "Dataset: ${GRAPH_DATASET}" | tee -a "${MASTER_LOG}" "${CASE_LOG}"
  echo "Features: ${FEATURES}" | tee -a "${MASTER_LOG}" "${CASE_LOG}"
  echo "Model path: ${MODEL_PATH}" | tee -a "${MASTER_LOG}" "${CASE_LOG}"
  echo "Started at: $(date)" | tee -a "${MASTER_LOG}" "${CASE_LOG}"
  echo "============================================================" | tee -a "${MASTER_LOG}" "${CASE_LOG}"

  python3 "${SCRIPT}" \
    --graph_dataset "${GRAPH_DATASET}" \
    --number_of_features "${FEATURES}" \
    --model_path "${MODEL_PATH}" \
    --trained_on bhoslib_and_dimacs \
    --preprocess_if_missing \
    --preposs_root "${PREPOSS_ROOT}" \
    --data_root "${DATA_ROOT}" \
    --results_root "${RESULTS_ROOT}" \
    --decoder "${DECODER}" \
    --num_passes_as_percentage_of_total_nodes "${WALKERS_RATIO}" \
    --candidate_limit "${CANDIDATE_LIMIT}" \
    --graph_timeout_sec "${GRAPH_TIMEOUT_SEC}" \
    --checkpoint_every "${CHECKPOINT_EVERY}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --progress_xlsx "${PROGRESS_XLSX}" \
    --run_label "${RUN_LABEL}" \
    --hourly_rate_usd "${HOURLY_RATE_USD}" \
    --instance_type "${INSTANCE_TYPE}" \
    --out_xlsx "${SUMMARY_XLSX}" \
    2>&1 | tee -a "${MASTER_LOG}" "${CASE_LOG}"

  local STATUS=${PIPESTATUS[0]}

  echo "============================================================" | tee -a "${MASTER_LOG}" "${CASE_LOG}"
  echo "Finished case: ${RUN_LABEL}" | tee -a "${MASTER_LOG}" "${CASE_LOG}"
  echo "Ended at: $(date)" | tee -a "${MASTER_LOG}" "${CASE_LOG}"
  echo "Exit status: ${STATUS}" | tee -a "${MASTER_LOG}" "${CASE_LOG}"
  echo "============================================================" | tee -a "${MASTER_LOG}" "${CASE_LOG}"

  return "${STATUS}"
}

STATUS_3=0
STATUS_10=0

run_case 3 "${MODEL_3}"
STATUS_3=$?

# Always attempt the 10-feature run after the 3-feature run, even if the first one fails.
run_case 10 "${MODEL_10}"
STATUS_10=$?

FINAL_STATUS=0
if [[ "${STATUS_3}" -ne 0 ]] || [[ "${STATUS_10}" -ne 0 ]]; then
  FINAL_STATUS=1
fi

echo "All requested cases attempted." | tee -a "${MASTER_LOG}"
echo "Dataset: ${GRAPH_DATASET}" | tee -a "${MASTER_LOG}"
echo "3-feature exit status: ${STATUS_3}" | tee -a "${MASTER_LOG}"
echo "10-feature exit status: ${STATUS_10}" | tee -a "${MASTER_LOG}"
echo "Final status: ${FINAL_STATUS}" | tee -a "${MASTER_LOG}"
echo "Progress Excel: ${PROGRESS_XLSX}" | tee -a "${MASTER_LOG}"
echo "Summary Excel: ${SUMMARY_XLSX}" | tee -a "${MASTER_LOG}"
echo "Master log: ${MASTER_LOG}" | tee -a "${MASTER_LOG}"

if [[ "${DO_SHUTDOWN}" -eq 1 ]]; then
  echo "Scheduling instance shutdown in ${SHUTDOWN_AFTER_MINUTES} minutes..." | tee -a "${MASTER_LOG}"
  sudo shutdown -h +"${SHUTDOWN_AFTER_MINUTES}"
else
  echo "No shutdown scheduled because --no_shutdown was passed." | tee -a "${MASTER_LOG}"
fi

exit "${FINAL_STATUS}"
