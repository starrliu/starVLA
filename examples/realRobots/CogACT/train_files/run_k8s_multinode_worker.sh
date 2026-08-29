#!/usr/bin/env bash
set -euo pipefail

: "${CODE_REV:?CODE_REV is required}"
: "${NODE_RANK:?NODE_RANK is required}"
: "${NUM_MACHINES:?NUM_MACHINES is required}"
: "${NUM_PROCESSES:?NUM_PROCESSES (global world size) is required}"
: "${MASTER_ADDR:?MASTER_ADDR is required}"
: "${MASTER_PORT:?MASTER_PORT is required}"
: "${FRAMEWORK:?FRAMEWORK is required}"
: "${RUN_ID:?RUN_ID is required}"

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends git libgl1 libglib2.0-0

job_tag="starvla-${RUN_ID}-node${NODE_RANK}"
source_dir="/data/yuming/code_copies/${job_tag}"
if [[ ! -d "${source_dir}/.git" ]]; then
  git clone --depth 20 --branch cogact_baseline https://github.com/starrliu/starVLA.git "${source_dir}"
fi
cd "${source_dir}"
git fetch --depth 20 origin cogact_baseline
git checkout "${CODE_REV}"
test "$(git rev-parse HEAD)" = "${CODE_REV}"

if [[ ! -x .venv/bin/accelerate ]]; then
  python -m venv --system-site-packages .venv
  . .venv/bin/activate
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install --no-cache-dir -r examples/realRobots/CogACT/requirements-cluster.txt
else
  . .venv/bin/activate
fi
python -m pip install --no-deps -e .

python - <<'PY'
import torch
import transformers
import deepspeed
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("deepspeed", deepspeed.__version__)
print("local_gpus", torch.cuda.device_count())
assert torch.cuda.device_count() == 8
PY

export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800000
export OPENCV_FFMPEG_READ_ATTEMPTS=10000000

FRAMEWORK="${FRAMEWORK}" \
NUM_PROCESSES="${NUM_PROCESSES}" \
NUM_MACHINES="${NUM_MACHINES}" \
MACHINE_RANK="${NODE_RANK}" \
MASTER_ADDR="${MASTER_ADDR}" \
MASTER_PORT="${MASTER_PORT}" \
USE_DEEPSPEED=1 \
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-512}" \
BATCH_SIZE="${BATCH_SIZE:-16}" \
NUM_WORKERS="${NUM_WORKERS:-4}" \
DATA_ROOT="${DATA_ROOT:-/data/liluo/msra_process}" \
MAX_STEPS="${MAX_STEPS:-20}" \
SAVE_FINAL_MODEL="${SAVE_FINAL_MODEL:-false}" \
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-1}" \
RUN_ROOT_DIR="${RUN_ROOT_DIR:-/data/yuming/starvla/checkpoints}" \
RUN_ID="${RUN_ID}" \
  bash examples/realRobots/CogACT/train_files/run_cogact_train.sh
