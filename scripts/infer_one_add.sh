#!/bin/bash
# Zero-shot FLUX.1-Fill-dev addition on one case (the single-case version of
# scripts/infer_reformat_coco_add.sh: same crop, prompt and paste-back).
#
# A case from DATA_ROOT (Image/<case>.jpg + Anno/<case>.json):
#   CASE=000000016010_24 ./scripts/infer_one_add.sh                # loc_bbox 0, 50 steps, cfg 30
#   CASE=000000016010_24 LOC_INDEX=2 STEPS=28 ./scripts/infer_one_add.sh
#
# Any image, box (x1 y1 x2 y2) and prompt:
#   IMAGE=path/to/img.jpg LOC_BBOX="215 263 268 333" PROMPT="Add exactly one zebra." \
#       ./scripts/infer_one_add.sh
#   IMAGE=path/to/img.jpg LOC_BBOX="215 263 268 333" CLASS_NAME=zebra ./scripts/infer_one_add.sh
#
# LOC_BBOX / PROMPT / CLASS_NAME also override the case's own values when CASE is set:
#   CASE=000000016010_24 PROMPT="Add a zebra drinking water." ./scripts/infer_one_add.sh
#   CASE=000000016010_24 LOC_BBOX="455 292 545 353" ./scripts/infer_one_add.sh
set -e

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

DATA_ROOT=${DATA_ROOT:-/home/user03/aiotlab/hachi/Image_Editing/data/Reformat_COCO}
FLUX=${FLUX:-/home/user03/aiotlab/hachi/Image_Editing/ckpt/black-forest-labs/FLUX.1-Fill-dev}

CASE=${CASE:-}
LOC_INDEX=${LOC_INDEX:-0}
IMAGE=${IMAGE:-"/home/user03/aiotlab/hachi/Image_Editing/data/FSC_final/FSC_difficult_no_overlap/577_b2.png"}
LOC_BBOX=${LOC_BBOX:-"187 142 226 187"}          # "x1 y1 x2 y2"
PROMPT=${PROMPT:-"A Cat."}              # empty => "Add exactly one <class_name>."
CLASS_NAME=${CLASS_NAME:-}
STEPS=${STEPS:-50}
GUIDANCE=${GUIDANCE:-30}
SEED=${SEED:-666}

DEFAULT_OUT_DIR=/home/user03/aiotlab/hachi/Image_Editing/output/one_FLUX1_add/steps${STEPS}_cfg${GUIDANCE}
OUT_DIR=${OUT_DIR:-${DEFAULT_OUT_DIR}}

# an array so a PROMPT with spaces stays one argument
CASE_ARGS=()
if [ -n "${CASE}" ]; then
    CASE_ARGS+=(--case "${CASE}" --loc_index "${LOC_INDEX}")
fi
if [ -n "${IMAGE}" ]; then
    CASE_ARGS+=(--image "${IMAGE}")
fi
if [ -n "${LOC_BBOX}" ]; then
    read -r -a BBOX <<< "${LOC_BBOX}"
    if [ "${#BBOX[@]}" -ne 4 ]; then
        echo "[error] LOC_BBOX needs 4 numbers \"x1 y1 x2 y2\", got \"${LOC_BBOX}\"" >&2
        exit 1
    fi
    CASE_ARGS+=(--loc_bbox "${BBOX[@]}")
fi
if [ -n "${PROMPT}" ]; then
    CASE_ARGS+=(--prompt "${PROMPT}")
fi
if [ -n "${CLASS_NAME}" ]; then
    CASE_ARGS+=(--class_name "${CLASS_NAME}")
fi

echo "------ run one case add, FLUX.1-Fill-dev zero-shot ------"
echo "data_root : ${DATA_ROOT}"
echo "case      : ${CASE:-${IMAGE:-from args}}"
echo "loc_bbox  : ${LOC_BBOX:-loc_bbox[${LOC_INDEX}] of the case}"
echo "prompt    : ${PROMPT:-Add exactly one <class_name>.}"
echo "steps     : ${STEPS}"
echo "guidance  : ${GUIDANCE}"
echo "out_dir   : ${OUT_DIR}"

python "${REPO}/inference_one_FLUX1_add.py" \
    --data_root "${DATA_ROOT}" \
    --out_dir "${OUT_DIR}" \
    --flux "${FLUX}" \
    "${CASE_ARGS[@]}" \
    --seed "${SEED}" \
    --size 768 \
    --steps "${STEPS}" \
    --guidance_scale "${GUIDANCE}" \
    "$@"
