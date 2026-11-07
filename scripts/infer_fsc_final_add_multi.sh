set -e

N=${N:-3}
TURN=${TURN:-${N}}
STEPS=${STEPS:-50}
GUIDANCE=${GUIDANCE:-30}
SPLITS=${SPLITS:-FSC_difficult_no_overlap FSC_difficult_overlap FSC_easy_no_overlap FSC_easy_overlap}

DEFAULT_WORK_DIR=/home/user03/aiotlab/hachi/Image_Editing/output/FSC_final/FLUX1Fill_Add_multi${N}_steps${STEPS}_cfg${GUIDANCE}/Turn_${TURN}
WORK_DIR=${WORK_DIR:-${DEFAULT_WORK_DIR}}

echo "--------- run FSC_final multi add, FLUX.1-Fill-dev zero-shot (single model load) ---------"
echo "splits      : ${SPLITS}"
echo "n           : ${N}"
echo "turn        : ${TURN}"
echo "steps       : ${STEPS}"
echo "guidance    : ${GUIDANCE}"
echo "work_dir    : ${WORK_DIR}"

python inference_fsc_final_FLUX1_add_multi.py \
    --test_root /home/user03/aiotlab/hachi/Image_Editing/data/FSC_final \
    --splits ${SPLITS} \
    --obj_root /home/user03/aiotlab/hachi/Image_Editing/data/OBJ_INS_FSC \
    --work_dir "${WORK_DIR}" \
    --n "${N}" \
    --turn "${TURN}" \
    --seed 666 \
    --size 768 \
    --steps "${STEPS}" \
    --guidance_scale "${GUIDANCE}" \
    "$@"
