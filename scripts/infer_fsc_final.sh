#!/bin/bash
# Insert-Anything over FSC_final, mirroring
# UniCombine_Ins_Rm/scripts/infer_fsc_final.sh (turn 1, all four splits).
#
# One python invocation for every split, so FLUX.1-Fill-dev, the LoRA and
# FLUX.1-Redux-dev are loaded once and reused across the whole sweep.
#
#   STEPS=8 ./scripts/infer_fsc_final.sh              # 8-step sweep, all splits
#   SPLITS="FSC_easy_overlap" ./scripts/infer_fsc_final.sh
#   WORK_DIR=/somewhere/else ./scripts/infer_fsc_final.sh
set -e

# empty STEPS => the pipeline's own default (50)
STEPS=${STEPS:-}
SPLITS=${SPLITS:-FSC_difficult_no_overlap FSC_difficult_overlap FSC_easy_no_overlap FSC_easy_overlap}

if [ -n "${STEPS}" ]; then
    DEFAULT_WORK_DIR=/home/user03/aiotlab/hachi/Image_Editing/output/FSC_final/InsertAnything_steps${STEPS}/Turn_1
    STEP_ARG="--steps ${STEPS}"
else
    DEFAULT_WORK_DIR=/home/user03/aiotlab/hachi/Image_Editing/output/FSC_final/InsertAnything_WhileRef/Turn_1
    STEP_ARG=""
fi
WORK_DIR=${WORK_DIR:-${DEFAULT_WORK_DIR}}

echo "--------------run FSC_final (single model load) --------------------"
echo "splits   : ${SPLITS}"
echo "steps    : ${STEPS:-pipeline default (50)}"
echo "work_dir : ${WORK_DIR}"

python inference_fsc_final.py \
    --test_root /home/user03/aiotlab/hachi/Image_Editing/data/FSC_final \
    --splits ${SPLITS} \
    --obj_root /home/user03/aiotlab/hachi/Image_Editing/data/OBJ_INS_FSC \
    --work_dir "${WORK_DIR}" \
    --turn 1 \
    --ref_mask_source mask \
    --seed 666 \
    --size 768 \
    ${STEP_ARG} \
    "$@"
