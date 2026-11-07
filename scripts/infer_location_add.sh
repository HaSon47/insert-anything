#!/bin/bash
# FLUX.1-Fill-dev zero-shot add over Location_with_anno_test, on the same cases
# as scripts/infer_location.sh (Insert-Anything).
#
#   ./scripts/infer_location_add.sh                  # turn 1, 50 steps, cfg 30
#   STEPS=8 GUIDANCE=30 TURN=2 ./scripts/infer_location_add.sh
#   ./scripts/infer_location_add.sh --limit 4 --dry_run
set -e

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

TEST_DIR=${TEST_DIR:-/home/user03/aiotlab/hachi/Image_Editing/data/Location_with_anno_test}
OBJ_ROOT=${OBJ_ROOT:-/home/user03/aiotlab/hachi/Image_Editing/data/OBJ_INS_FSC}
FLUX=${FLUX:-/home/user03/aiotlab/hachi/Image_Editing/ckpt/black-forest-labs/FLUX.1-Fill-dev}

TURN=${TURN:-1}
STEPS=${STEPS:-50}
GUIDANCE=${GUIDANCE:-30}

DEFAULT_WORK_DIR=/home/user03/aiotlab/hachi/Image_Editing/output/Location_with_anno_test/FLUX1Fill_Add_steps${STEPS}_cfg${GUIDANCE}/Turn_${TURN}
WORK_DIR=${WORK_DIR:-${DEFAULT_WORK_DIR}}

echo "--------- run Location add, FLUX.1-Fill-dev zero-shot (single model load) ---------"
echo "test_dir    : ${TEST_DIR}"
echo "turn        : ${TURN}"
echo "steps       : ${STEPS}"
echo "guidance    : ${GUIDANCE}"
echo "work_dir    : ${WORK_DIR}"

python "${REPO}/inference_location_FLUX1_add.py" \
    --test_dir "${TEST_DIR}" \
    --obj_root "${OBJ_ROOT}" \
    --work_dir "${WORK_DIR}" \
    --flux "${FLUX}" \
    --turn "${TURN}" \
    --seed 666 \
    --size 768 \
    --steps "${STEPS}" \
    --guidance_scale "${GUIDANCE}" \
    "$@"
