"""Reformat multi_animal_val (COCO-derived) into the Insert Anything test layout.

For every <img>.json in <src>/anno_with_loc_bbox and every key (category_id) of
its "loc_bbox", one case "<img>_<key>" is written:

    <dst>/Image/<img>_<key>.jpg              copy of the source image
    <dst>/Anno/<img>_<key>.json              category_id, class_name, loc_bbox,
                                             exam_bbox, segmentation
    <dst>/Exam_mask/<img>_<key>/mask_NNN.png binary mask (0/255) of exam_bbox[NNN]

Exemplars are the non-crowd COCO annotations of that category; exam_bbox is
converted from COCO xywh to xyxy to match loc_bbox, and segmentation[i] is the
polygon list of exam_bbox[i].
"""
import argparse
import glob
import json
import os
import shutil

import cv2
import numpy as np


def polygons_to_mask(polygons, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polygons:
        pts = np.round(np.array(poly, dtype=np.float64).reshape(-1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="/home/user03/aiotlab/hachi/Image_Editing/data/multi_animal_val")
    p.add_argument("--dst", default="/home/user03/aiotlab/hachi/Image_Editing/data/Reformat_COCO")
    args = p.parse_args()

    image_out = os.path.join(args.dst, "Image")
    anno_out = os.path.join(args.dst, "Anno")
    mask_out = os.path.join(args.dst, "Exam_mask")
    for d in (image_out, anno_out, mask_out):
        os.makedirs(d, exist_ok=True)

    n_cases, n_masks, skipped = 0, 0, []
    for anno_path in sorted(glob.glob(os.path.join(args.src, "anno_with_loc_bbox", "*.json"))):
        with open(anno_path) as f:
            data = json.load(f)
        info = data["image"]
        stem = os.path.splitext(info["file_name"])[0]
        h, w = info["height"], info["width"]
        src_image = os.path.join(args.src, "image", info["file_name"])
        cat_names = {c["id"]: c["name"] for c in data["categories"]}

        for key, loc_bboxes in data["loc_bbox"].items():
            cat_id = int(key)
            exemplars = [a for a in data["annotations"]
                         if a["category_id"] == cat_id and not a["iscrowd"]]
            case = f"{stem}_{key}"
            if not exemplars:
                skipped.append(case)
                continue

            shutil.copy2(src_image, os.path.join(image_out, case + ".jpg"))

            exam_bbox = [[x, y, x + bw, y + bh] for x, y, bw, bh in (a["bbox"] for a in exemplars)]
            anno = {
                "category_id": cat_id,
                "class_name": cat_names[cat_id],
                "loc_bbox": loc_bboxes,
                "exam_bbox": exam_bbox,
                "segmentation": [a["segmentation"] for a in exemplars],
            }
            with open(os.path.join(anno_out, case + ".json"), "w") as f:
                json.dump(anno, f, indent=2)

            case_mask_dir = os.path.join(mask_out, case)
            os.makedirs(case_mask_dir, exist_ok=True)
            for i, a in enumerate(exemplars):
                mask = polygons_to_mask(a["segmentation"], h, w)
                cv2.imwrite(os.path.join(case_mask_dir, f"mask_{i:03d}.png"), mask)
                n_masks += 1
            n_cases += 1

    print(f"wrote {n_cases} cases, {n_masks} masks to {args.dst}")
    if skipped:
        print(f"skipped {len(skipped)} cases with no non-crowd exemplar: {skipped}")


if __name__ == "__main__":
    main()
