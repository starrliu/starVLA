#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   FRAMEWORK=qwenoft NUM_PROCESSES=8 bash .../run_cogact_train.sh
#   FRAMEWORK=qwenpi_v3 NUM_PROCESSES=8 bash .../run_cogact_train.sh
# On a machine without a CUDA toolkit/working DeepSpeed installation, set
# USE_DEEPSPEED=0. This is primarily useful for single-GPU smoke tests.

FRAMEWORK=${FRAMEWORK:-qwenoft}
NUM_PROCESSES=${NUM_PROCESSES:-8}
USE_DEEPSPEED=${USE_DEEPSPEED:-1}
RUN_ROOT_DIR=${RUN_ROOT_DIR:-results/Checkpoints}
MAX_STEPS=${MAX_STEPS:-100000}
BATCH_SIZE=${BATCH_SIZE:-8}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-512}
NUM_WORKERS=${NUM_WORKERS:-8}
DATA_ROOT=${DATA_ROOT:-/data/cogact_dataset}
FREEZE_MODULES=${FREEZE_MODULES:-}
DIT_HIDDEN_DIM=${DIT_HIDDEN_DIM:-}
REPEATED_DIFFUSION_STEPS=${REPEATED_DIFFUSION_STEPS:-}

batch_denominator=$((BATCH_SIZE * NUM_PROCESSES))
if (( GLOBAL_BATCH_SIZE % batch_denominator != 0 )); then
  echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be divisible by BATCH_SIZE*NUM_PROCESSES=${batch_denominator}." >&2
  exit 2
fi
GRAD_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / batch_denominator))
export STARVLA_GRADIENT_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS}"

case "${FRAMEWORK}" in
  qwenoft)
    CONFIG=${CONFIG:-examples/realRobots/CogACT/train_files/starvla_qwenoft_cogact.yaml}
    RUN_ID=${RUN_ID:-starvla_qwenoft_cogact}
    ;;
  qwenpi_v3)
    CONFIG=${CONFIG:-examples/realRobots/CogACT/train_files/starvla_qwenpi_v3_cogact.yaml}
    RUN_ID=${RUN_ID:-starvla_qwenpi_v3_cogact}
    ;;
  *)
    echo "FRAMEWORK must be 'qwenoft' or 'qwenpi_v3', got: ${FRAMEWORK}" >&2
    exit 2
    ;;
esac

mkdir -p "${RUN_ROOT_DIR}/${RUN_ID}"
cp "$0" "${RUN_ROOT_DIR}/${RUN_ID}/"

train_args=(
  starVLA/training/train_starvla.py
  --config_yaml "${CONFIG}"
  --datasets.vla_data.data_root_dir "${DATA_ROOT}"
  --datasets.vla_data.per_device_batch_size "${BATCH_SIZE}"
  --datasets.vla_data.num_workers "${NUM_WORKERS}"
  --trainer.freeze_modules "${FREEZE_MODULES}"
  --trainer.max_train_steps "${MAX_STEPS}"
  --trainer.gradient_accumulation_steps "${GRAD_ACCUMULATION_STEPS}"
  --run_root_dir "${RUN_ROOT_DIR}"
  --run_id "${RUN_ID}"
)

if [[ -n "${DIT_HIDDEN_DIM}" ]]; then
  train_args+=(--framework.action_model.diffusion_model_cfg.action_dit_hidden_dim "${DIT_HIDDEN_DIM}")
fi
if [[ -n "${REPEATED_DIFFUSION_STEPS}" ]]; then
  train_args+=(--framework.action_model.repeated_diffusion_steps "${REPEATED_DIFFUSION_STEPS}")
fi

if [[ "${USE_DEEPSPEED}" == "1" ]]; then
  accelerate launch \
    --config_file examples/realRobots/CogACT/train_files/accelerate_zero2.yaml \
    --num_processes "${NUM_PROCESSES}" \
    "${train_args[@]}"
else
  if [[ "${NUM_PROCESSES}" != "1" ]]; then
    echo "USE_DEEPSPEED=0 currently supports NUM_PROCESSES=1 only." >&2
    exit 2
  fi
  STARVLA_USE_DEEPSPEED=0 python "${train_args[@]}"
fi
