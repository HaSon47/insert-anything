#!/bin/bash
# Insert-Anything over Location_with_anno_test, inserting into the first
# CE-predicted box (pred_ce130.jsonl, selected_boxes[0]) instead of
# phase1_output_k3 pred_box[0].  Otherwise identical to scripts/infer_location.sh.
#
# One python invocation for the whole sweep, so FLUX.1-Fill-dev, the LoRA and
# FLUX.1-Redux-dev are loaded once and reused.
#
#   STEPS=8 ./scripts/infer_location_celoc.sh              # 8-step sweep
#   ./scripts/infer_location_celoc.sh                      # pipeline default (50)
#   TURN=2 ./scripts/infer_location_celoc.sh               # another turn
#   WORK_DIR=/somewhere/else ./scripts/infer_location_celoc.sh
#   ./scripts/infer_location_celoc.sh --limit 4 --dry_run
#
# On another machine, override the paths that are not the same there:
#   TEST_DIR=... PRED_PATH=... OBJ_ROOT=... FLUX=... REDUX=... LORA=... WORK_DIR=... \
#       ./scripts/infer_location_celoc.sh
set -e

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

TEST_DIR=${TEST_DIR:-/home/user03/aiotlab/hachi/Image_Editing/data/Location_with_anno_test}
PRED_PATH=${PRED_PATH:-${TEST_DIR}/pred_ce130.jsonl}
OBJ_ROOT=${OBJ_ROOT:-/home/user03/aiotlab/hachi/Image_Editing/data/OBJ_INS_FSC}
FLUX=${FLUX:-/home/user03/aiotlab/hachi/Image_Editing/ckpt/black-forest-labs/FLUX.1-Fill-dev}
REDUX=${REDUX:-/home/user03/aiotlab/hachi/Image_Editing/ckpt/black-forest-labs/FLUX.1-Redux-dev}
LORA=${LORA:-/home/user03/aiotlab/hachi/Image_Editing/ckpt/Insert-Anything/20250321_steps5000_pytorch_lora_weights.safetensors}

TURN=${TURN:-1}
# empty STEPS => the pipeline's own default (50)
STEPS=${STEPS:-}

if [ -n "${STEPS}" ]; then
    DEFAULT_WORK_DIR=/home/user03/aiotlab/hachi/Image_Editing/output/Location_with_anno_test/InsertAnything_celoc_steps${STEPS}/Turn_${TURN}
    STEP_ARG="--steps ${STEPS}"
else
    DEFAULT_WORK_DIR=/home/user03/aiotlab/hachi/Image_Editing/output/Location_with_anno_test/InsertAnything_celoc/Turn_${TURN}
    STEP_ARG=""
fi
WORK_DIR=${WORK_DIR:-${DEFAULT_WORK_DIR}}

echo "--------------run Location @ CE pred boxes (single model load) --------------------"
echo "test_dir : ${TEST_DIR}"
echo "pred_path: ${PRED_PATH}"
echo "turn     : ${TURN}"
echo "steps    : ${STEPS:-pipeline default (50)}"
echo "work_dir : ${WORK_DIR}"

python "${REPO}/inference_location_celoc.py" \
    --test_dir "${TEST_DIR}" \
    --pred_path "${PRED_PATH}" \
    --pred_box_index 0 \
    --obj_root "${OBJ_ROOT}" \
    --work_dir "${WORK_DIR}" \
    --flux "${FLUX}" \
    --redux "${REDUX}" \
    --lora "${LORA}" \
    --turn "${TURN}" \
    --ref_mask_source mask \
    --seed 666 \
    --size 768 \
    ${STEP_ARG} \
    "$@"
