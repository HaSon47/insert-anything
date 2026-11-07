#!/bin/bash
# Zero-shot FLUX.1-Fill-dev addition over processed_dataset (the baseline of
# scripts/infer_processed_dataset.sh: same cases, no LoRA, no Redux).
#
#   ./scripts/infer_processed_dataset_add.sh                   # 50 steps, cfg 30
#   STEPS=28 GUIDANCE=30 ./scripts/infer_processed_dataset_add.sh
#   ./scripts/infer_processed_dataset_add.sh --limit 4 --dry_run
set -e

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

DATA_ROOT=${DATA_ROOT:-/home/user03/aiotlab/hachi/Image_Editing/data/processed_dataset}
FLUX=${FLUX:-/home/user03/aiotlab/hachi/Image_Editing/ckpt/black-forest-labs/FLUX.1-Fill-dev}

STEPS=${STEPS:-50}
GUIDANCE=${GUIDANCE:-30}

DEFAULT_WORK_DIR=/home/user03/aiotlab/hachi/Image_Editing/output/processed_dataset/FLUX1Fill_Add_steps${STEPS}_cfg${GUIDANCE}
WORK_DIR=${WORK_DIR:-${DEFAULT_WORK_DIR}}

echo "------ run processed_dataset add, FLUX.1-Fill-dev zero-shot (single model load) ------"
echo "data_root : ${DATA_ROOT}"
echo "steps     : ${STEPS}"
echo "guidance  : ${GUIDANCE}"
echo "work_dir  : ${WORK_DIR}"

python "${REPO}/inference_processed_dataset_FLUX1_add.py" \
    --data_root "${DATA_ROOT}" \
    --work_dir "${WORK_DIR}" \
    --flux "${FLUX}" \
    --loc_index 0 \
    --exemplar_index 0 \
    --seed 666 \
    --size 768 \
    --steps "${STEPS}" \
    --guidance_scale "${GUIDANCE}" \
    "$@"
