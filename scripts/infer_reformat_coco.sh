#!/bin/bash
# Insert-Anything over Reformat_COCO (multi_animal_val reformatted by
# scripts/reformat_coco_multi_animal.py).
#
# Same shape as scripts/infer_fsc_final.sh, but the dataset is flat (no splits):
# one python invocation for the whole sweep, so FLUX.1-Fill-dev, the LoRA and
# FLUX.1-Redux-dev are loaded once and reused.
#
#   STEPS=8 ./scripts/infer_reformat_coco.sh          # 8-step sweep
#   ./scripts/infer_reformat_coco.sh                  # pipeline default (50)
#   WORK_DIR=/somewhere/else ./scripts/infer_reformat_coco.sh
#   ./scripts/infer_reformat_coco.sh --limit 4 --dry_run
#
# On another machine, override the paths that are not the same there:
#   DATA_ROOT=... FLUX=... REDUX=... LORA=... WORK_DIR=... STEPS=8 \
#       ./scripts/infer_reformat_coco.sh
set -e

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

DATA_ROOT=${DATA_ROOT:-/home/user03/aiotlab/hachi/Image_Editing/data/Reformat_COCO}
FLUX=${FLUX:-/home/user03/aiotlab/hachi/Image_Editing/ckpt/black-forest-labs/FLUX.1-Fill-dev}
REDUX=${REDUX:-/home/user03/aiotlab/hachi/Image_Editing/ckpt/black-forest-labs/FLUX.1-Redux-dev}
LORA=${LORA:-/home/user03/aiotlab/hachi/Image_Editing/ckpt/Insert-Anything/20250321_steps5000_pytorch_lora_weights.safetensors}

# empty STEPS => the pipeline's own default (50)
STEPS=${STEPS:-}

if [ -n "${STEPS}" ]; then
    DEFAULT_WORK_DIR=/home/user03/aiotlab/hachi/Image_Editing/output/Reformat_COCO/InsertAnything_steps${STEPS}
    STEP_ARG="--steps ${STEPS}"
else
    DEFAULT_WORK_DIR=/home/user03/aiotlab/hachi/Image_Editing/output/Reformat_COCO/InsertAnything
    STEP_ARG=""
fi
WORK_DIR=${WORK_DIR:-${DEFAULT_WORK_DIR}}

echo "--------------run Reformat_COCO (single model load) --------------------"
echo "data_root : ${DATA_ROOT}"
echo "steps     : ${STEPS:-pipeline default (50)}"
echo "work_dir  : ${WORK_DIR}"

python "${REPO}/inference_reformat_coco.py" \
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
