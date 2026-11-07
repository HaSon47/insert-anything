#!/bin/bash
# Zero-shot object removal over FSC_final with the plain FLUX.1-Fill-dev
# backbone (no Insert-Anything LoRA, no Redux, no diptych), turn 1, all four
# splits: remove inpainted_bboxes[1] from inpainted_turn_1.png.
#
# One python invocation for every split, so FLUX.1-Fill-dev is loaded once and
# reused across the whole sweep.
#
#   ./scripts/infer_fsc_final_removal.sh                     # bbox mask, 50 steps
#   STEPS=28 GUIDANCE=30 ./scripts/infer_fsc_final_removal.sh
#   SPLITS="FSC_easy_overlap" ./scripts/infer_fsc_final_removal.sh --limit 5
#   WORK_DIR=/somewhere/else ./scripts/infer_fsc_final_removal.sh
set -e

STEPS=${STEPS:-50}
GUIDANCE=${GUIDANCE:-30}
SPLITS=${SPLITS:-FSC_difficult_no_overlap FSC_difficult_overlap FSC_easy_no_overlap FSC_easy_overlap}

DEFAULT_WORK_DIR=/home/user03/aiotlab/hachi/Image_Editing/output/FSC_final/FLUX1Fill_Removal_steps${STEPS}_cfg${GUIDANCE}/Turn_1
WORK_DIR=${WORK_DIR:-${DEFAULT_WORK_DIR}}

echo "--------- run FSC_final removal, FLUX.1-Fill-dev zero-shot (single model load) ---------"
echo "splits      : ${SPLITS}"
echo "steps       : ${STEPS}"
echo "guidance    : ${GUIDANCE}"
echo "work_dir    : ${WORK_DIR}"

python inference_fsc_final_FLUX1_removal.py \
    --test_root /home/user03/aiotlab/hachi/Image_Editing/data/FSC_final \
    --splits ${SPLITS} \
    --obj_root /home/user03/aiotlab/hachi/Image_Editing/data/OBJ_INS_FSC \
    --work_dir "${WORK_DIR}" \
    --turn 1 \
    --seed 666 \
    --size 768 \
    --steps "${STEPS}" \
    --guidance_scale "${GUIDANCE}" \
    "$@"
