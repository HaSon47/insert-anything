#!/usr/bin/env bash
# Evaluate a removal LoRA on raw output tiles (removal_training_plan.md, section 6). From the repo root:
#     LORA=runs_removal/removal_v1/ckpt/5000 NAME=v1_5000 sbatch scripts/eval_removal.sh
#     LORA=runs_removal/removal_v1/ckpt/5000 NAME=v1_5000 bash   scripts/eval_removal.sh   # inside an allocation
# Baseline (InsertAnything LoRA, no removal training):
#     LORA=/home/user03/aiotlab/hachi/Image_Editing/ckpt/Insert-Anything/20250321_steps5000_pytorch_lora_weights.safetensors NAME=baseline_ia ...
# Environment: DATASET=objectstitch_test|objectstitch_val|omnipaint_val (default objectstitch_test),
#              STEPS=50 GUIDANCE=30 SEED=42, LIMIT / START for a slice. Re-running skips finished samples.
# Results: outputs/removal_eval/<NAME>/<DATASET>/{metrics.csv,metrics_summary.json,grid_image/,...}

#SBATCH --job-name=rm_eval
#SBATCH --output=removal_eval_%j.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${LORA:?set LORA to a ckpt/<step> dir, a .safetensors file, or none}"
: "${NAME:?set NAME, the output folder name under outputs/removal_eval/}"
DATASET="${DATASET:-objectstitch_test}"
echo "[$(date)] Node: $(hostname)  SLURM_JOB_ID=${SLURM_JOB_ID:-}  LORA=$LORA  DATASET=$DATASET"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen_edit
export TMPDIR="/dev/shm/${USER}_tmp"
mkdir -p "$TMPDIR"

args=(--lora "$LORA" --dataset "$DATASET" --work_dir "outputs/removal_eval/$NAME/$DATASET"
      --steps "${STEPS:-50}" --guidance_scale "${GUIDANCE:-30}" --seed "${SEED:-42}"
      --cond outputs/removal_eval/white_ref_cond.pt)
if [ -n "${LIMIT:-}" ]; then args+=(--limit "$LIMIT"); fi
if [ -n "${START:-}" ]; then args+=(--start "$START"); fi
python -u eval_removal.py "${args[@]}"
echo "[$(date)] done"
