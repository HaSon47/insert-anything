#!/bin/bash
# Insert-Anything over processed_dataset.
#
# Same shape as scripts/infer_fsc_final.sh, but the dataset is flat (no splits):
# one python invocation for the whole sweep, so FLUX.1-Fill-dev, the LoRA and
# FLUX.1-Redux-dev are loaded once and reused.
#
#   STEPS=8 ./scripts/infer_processed_dataset.sh          # 8-step sweep
#   ./scripts/infer_processed_dataset.sh                  # pipeline default (50)
#   WORK_DIR=/somewhere/else ./scripts/infer_processed_dataset.sh
#   ./scripts/infer_processed_dataset.sh --limit 4 --dry_run
#
# On another machine, override the paths that are not the same there:
#   DATA_ROOT=... FLUX=... REDUX=... LORA=... WORK_DIR=... STEPS=8 \
#       ./scripts/infer_processed_dataset.sh
set -e

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

DATA_ROOT=${DATA_ROOT:-/home/user03/aiotlab/hachi/Image_Editing/data/processed_dataset}
FLUX=${FLUX:-/home/user03/aiotlab/hachi/Image_Editing/ckpt/black-forest-labs/FLUX.1-Fill-dev}
REDUX=${REDUX:-/home/user03/aiotlab/hachi/Image_Editing/ckpt/black-forest-labs/FLUX.1-Redux-dev}
LORA=${LORA:-/home/user03/aiotlab/hachi/Image_Editing/ckpt/Insert-Anything/20250321_steps5000_pytorch_lora_weights.safetensors}

# empty STEPS => the pipeline's own default (50)
STEPS=${STEPS:-}

if [ -n "${STEPS}" ]; then
    DEFAULT_WORK_DIR=/home/user03/aiotlab/hachi/Image_Editing/output/processed_dataset/InsertAnything_steps${STEPS}
    STEP_ARG="--steps ${STEPS}"
else
    DEFAULT_WORK_DIR=/home/user03/aiotlab/hachi/Image_Editing/output/processed_dataset/InsertAnything
    STEP_ARG=""
fi
WORK_DIR=${WORK_DIR:-${DEFAULT_WORK_DIR}}

echo "--------------run processed_dataset (single model load) --------------------"
echo "data_root : ${DATA_ROOT}"
echo "steps     : ${STEPS:-pipeline default (50)}"
echo "work_dir  : ${WORK_DIR}"

python "${REPO}/inference_processed_dataset.py" \
    --data_root "${DATA_ROOT}" \
    --work_dir "${WORK_DIR}" \
    --flux "${FLUX}" \
    --redux "${REDUX}" \
    --lora "${LORA}" \
    --loc_index 0 \
    --exemplar_index 0 \
    --ref_mask_source mask \
    --seed 666 \
    --size 768 \
    ${STEP_ARG} \
    "$@"
