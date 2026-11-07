#!/bin/bash
# Insert-Anything over FSC_final with N objects inserted SEQUENTIALLY, one per
# forward pass, mirroring scripts/infer_fsc_final_multi.sh (N objects in one pass).
#
# Pass k inserts loc bbox k into the output of pass k-1.  Every pass uses the
# multi run's crop (around the union of all N boxes), the same reference
# object, seed and settings, so the work dirs are directly comparable.
#
# MODE picks what is carried between passes (each mode gets its own work dir):
#   full_image : paste each output panel back into the full image, re-crop it
#   panel      : feed the 768x768 output panel on as-is, crop_back only at the end
#
# One python invocation for every split, so FLUX.1-Fill-dev, the LoRA and
# FLUX.1-Redux-dev are loaded once and reused across the whole sweep.
#
#   ./scripts/infer_fsc_final_seq.sh                  # full_image, n=3, turn 3, all splits
#   MODE=panel ./scripts/infer_fsc_final_seq.sh       # panel mode
#   N=2 ./scripts/infer_fsc_final_seq.sh              # n=2, turn 2
#   STEPS=8 ./scripts/infer_fsc_final_seq.sh          # 8-step sweep
#   SPLITS="FSC_easy_overlap" ./scripts/infer_fsc_final_seq.sh
#   WORK_DIR=/somewhere/else ./scripts/infer_fsc_final_seq.sh
set -e

MODE=${MODE:-full_image}
N=${N:-3}
TURN=${TURN:-${N}}
# empty STEPS => the pipeline's own default (50)
STEPS=${STEPS:-}
SPLITS=${SPLITS:-FSC_easy_overlap }

if [ -n "${STEPS}" ]; then
    DEFAULT_WORK_DIR=/home/user03/aiotlab/hachi/Image_Editing/output/FSC_final/InsertAnything_seq${N}_${MODE}_steps${STEPS}/Turn_${TURN}
    STEP_ARG="--steps ${STEPS}"
else
    DEFAULT_WORK_DIR=/home/user03/aiotlab/hachi/Image_Editing/output/FSC_final/InsertAnything_seq${N}_${MODE}/Turn_${TURN}
    STEP_ARG=""
fi
WORK_DIR=${WORK_DIR:-${DEFAULT_WORK_DIR}}

echo "--------- run FSC_final sequential multi-object (single model load) -----------"
echo "mode     : ${MODE}"
echo "n        : ${N} objects, one per inference pass"
echo "turn     : ${TURN}"
echo "splits   : ${SPLITS}"
echo "steps    : ${STEPS:-pipeline default (50)}"
echo "work_dir : ${WORK_DIR}"

python inference_fsc_final_seq.py \
    --mode "${MODE}" \
    --test_root /home/user03/aiotlab/hachi/Image_Editing/data/FSC_final \
    --splits ${SPLITS} \
    --obj_root /home/user03/aiotlab/hachi/Image_Editing/data/OBJ_INS_FSC \
    --work_dir "${WORK_DIR}" \
    --n "${N}" \
    --turn "${TURN}" \
    --ref_mask_source mask \
    --seed 666 \
    --size 768 \
    ${STEP_ARG} \
    "$@"
