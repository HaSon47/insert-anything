#!/usr/bin/env bash
# Removal fine-tuning launcher (removal_training_plan.md, sections 4 and 4.2). Run from the repo root,
# either as a Slurm job or inside an existing GPU allocation:
#     sbatch scripts/train_removal.sh
#     bash   scripts/train_removal.sh
# Re-running the same command continues the run from <save_path>/<RUN_NAME>/resume/last.ckpt.
# Environment overrides:
#     RUN_NAME=removal_v1  XFL_CONFIG=experiments/config/removal.yaml  MAX_STEPS=<n>
#     AUTO_RESTART=1       retry after a crash (up to MAX_RESTARTS=20 times, 60 s apart)
# Log: <save_path>/<RUN_NAME>/train.log (appended by every launch).

#SBATCH --job-name=rm_train
#SBATCH --output=removal_train_%j.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "[$(date)] Node: $(hostname)  SLURM_JOB_ID=${SLURM_JOB_ID:-}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv || true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen_edit

# TMPDIR on NFS makes DataLoader workers spam "Device or resource busy: .nfs..." on exit.
export TMPDIR="/dev/shm/${USER}_tmp"
mkdir -p "$TMPDIR"
# Peak memory at batch 8 is ~74 GB of 80; expandable segments reduce fragmentation.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_NAME="${RUN_NAME:-removal_v1}"
CONFIG="${XFL_CONFIG:-experiments/config/removal.yaml}"
cmd=(python -u -m src.train.train_removal --config "$CONFIG" --run_name "$RUN_NAME" --resume auto)
if [ -n "${MAX_STEPS:-}" ]; then
  cmd+=(--max_steps "$MAX_STEPS")
fi

if [ "${AUTO_RESTART:-0}" != 1 ]; then
  "${cmd[@]}"
  exit $?
fi

for ((i = 0; i <= ${MAX_RESTARTS:-20}; i++)); do
  "${cmd[@]}"
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "[$(date)] training finished"
    exit 0
  fi
  if [ $rc -eq 2 ]; then  # config / argument error: retrying cannot help
    echo "[$(date)] exit 2 (config or argument error), not retrying"
    exit 2
  fi
  echo "[$(date)] exit $rc, restart $((i + 1)) in 60 s"
  sleep 60
done
echo "[$(date)] giving up after ${MAX_RESTARTS:-20} restarts"
exit 1
