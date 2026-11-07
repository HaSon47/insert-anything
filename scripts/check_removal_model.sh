#!/usr/bin/env bash
# Milestone M2 GPU check of the removal model (real FLUX weights), see removal_training_plan.md.
# Submit from the repo root:   sbatch scripts/check_removal_model.sh
# Optional:                    PROBE_BATCH=4 sbatch scripts/check_removal_model.sh
# Report: runs_removal/_checks/m2/m2_report.json (+ parity_*.jpg); log: removal_m2_<jobid>.log

#SBATCH --job-name=rm_m2
#SBATCH --output=removal_m2_%j.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "[$(date)] Node: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true
df -h /dev/shm | tail -1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen_edit

# TMPDIR on NFS makes DataLoader workers spam "Device or resource busy: .nfs..." on exit.
export TMPDIR="/dev/shm/${USER}_tmp"
mkdir -p "$TMPDIR"

python -m src.models.removal_model --gpu_check --probe_batch "${PROBE_BATCH:-8}" --out runs_removal/_checks/m2
echo "[$(date)] done"
