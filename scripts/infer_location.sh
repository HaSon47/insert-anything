#!/bin/bash
# Insert-Anything over Location_with_anno_test, mirroring
# UniCombine_Ins_Rm/scripts/infer_location.sh (turn 1).
#
# One python invocation for the whole sweep, so FLUX.1-Fill-dev, the LoRA and
# FLUX.1-Redux-dev are loaded once and reused.
#
#   STEPS=8 ./scripts/infer_location.sh              # 8-step sweep
#   ./scripts/infer_location.sh                      # pipeline default (50)
#   TURN=2 ./scripts/infer_location.sh               # another turn
#   WORK_DIR=/somewhere/else ./scripts/infer_location.sh
#   ./scripts/infer_location.sh --limit 4 --dry_run
#
# On another machine, override the paths that are not the same there:
#   TEST_DIR=... OBJ_ROOT=... FLUX=... REDUX=... LORA=... WORK_DIR=... \
#       ./scripts/infer_location.sh
set -e

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

TEST_DIR=${TEST_DIR:-/home/user03/aiotlab/hachi/Image_Editing/data/Location_with_anno_test}
OBJ_ROOT=${OBJ_ROOT:-/home/user03/aiotlab/hachi/Image_Editing/data/OBJ_INS_FSC}
FLUX=${FLUX:-/home/user03/aiotlab/hachi/Image_Editing/ckpt/black-forest-labs/FLUX.1-Fill-dev}
REDUX=${REDUX:-/home/user03/aiotlab/hachi/Image_Editing/ckpt/black-forest-labs/FLUX.1-Redux-dev}
LORA=${LORA:-/home/user03/aiotlab/hachi/Image_Editing/ckpt/Insert-Anything/20250321_steps5000_pytorch_lora_weights.safetensors}

TURN=${TURN:-1}
# empty STEPS => the pipeline's own default (50)
STEPS=${STEPS:-}

if [ -n "${STEPS}" ]; then
    DEFAULT_WORK_DIR=/home/user03/aiotlab/hachi/Image_Editing/output/Location_with_anno_test/InsertAnything_steps${STEPS}/Turn_${TURN}
    STEP_ARG="--steps ${STEPS}"
else
    DEFAULT_WORK_DIR=/home/user03/aiotlab/hachi/Image_Editing/output/Location_with_anno_test/InsertAnything/Turn_${TURN}
    STEP_ARG=""
fi
WORK_DIR=${WORK_DIR:-${DEFAULT_WORK_DIR}}

echo "--------------run Location (single model load) --------------------"
echo "test_dir : ${TEST_DIR}"
echo "turn     : ${TURN}"
echo "steps    : ${STEPS:-pipeline default (50)}"
echo "work_dir : ${WORK_DIR}"

python "${REPO}/inference_location.py" \
    --test_dir "${TEST_DIR}" \
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
