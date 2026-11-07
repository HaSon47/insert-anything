"""Run zero-shot FLUX.1-Fill-dev addition over `processed_dataset`, at predicted locations.

Same as inference_processed_dataset_FLUX1_add.py, except that the box to insert
into is not the annotated anno["loc_bbox"][0] but the first box of
"selected_boxes" in the CoCount prediction file:

    <data_root>/pred_cocount.jsonl   one row per case, {"id": <case>,
                                     "selected_boxes": [[x0, y0, x1, y1], ...], ...}

Cases are listed as in inference_processed_dataset.py and then restricted to the
ones with a usable predicted box.  Predicted boxes are reordered to
x0 <= x1, y0 <= y1 (a few rows carry flipped corners) and clipped to the image;
cases with no prediction, an empty "selected_boxes" or a box that is degenerate
after clipping are skipped and counted.  The annotated box is kept in meta as
gt_loc_bbox.  Outputs are laid out as in inference_processed_dataset_FLUX1_add.py.
"""

import argparse
import json
import os
import traceback

import numpy as np
from PIL import Image
from tqdm import tqdm

from inference_fsc_final import OUT_DIRS, REQUIRED, _outline, build_target, make_grid
from inference_processed_dataset import get_inputs
from inference_processed_dataset_FLUX1_add import load_case, make_prompt
from utils.utils import crop_back


# ------------------------------------------------------- dataset / inputs --

def load_pred_boxes(pred_path, box_index):
    """{case: (box, (w, h)) or reason} from the CoCount prediction file."""
    preds = {}
    with open(pred_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            boxes = row.get("selected_boxes") or []
            if len(boxes) <= box_index:
                preds[row["id"]] = "too_few_pred_boxes"
                continue

            w, h = row["wh"]
            x0, y0, x1, y1 = (float(v) for v in boxes[box_index])
            x0, x1 = sorted((x0, x1))
            y0, y1 = sorted((y0, y1))
            x0, x1 = min(max(x0, 0.0), w), min(max(x1, 0.0), w)
            y0, y1 = min(max(y0, 0.0), h), min(max(y1, 0.0), h)
            if x1 - x0 < 1 or y1 - y0 < 1:
                preds[row["id"]] = "degenerate_pred_box"
                continue
            preds[row["id"]] = [x0, y0, x1, y1]
    return preds


# ------------------------------------------------------------------ main --

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", default="/home/user03/aiotlab/hachi/Image_Editing/data/processed_dataset",
                   help="root holding Anno/, Image/ and masks/")
    p.add_argument("--anno_dir", default=None, help="default: <data_root>/Anno")
    p.add_argument("--image_dir", default=None, help="default: <data_root>/Image")
    p.add_argument("--mask_dir", default=None, help="default: <data_root>/masks")
    p.add_argument("--pred_path", default=None, help="default: <data_root>/pred_cocount.jsonl")
    p.add_argument("--pred_box_index", type=int, default=0,
                   help="which entry of selected_boxes to insert into (default 0, the first)")
    p.add_argument("--work_dir",
                   default="/home/user03/aiotlab/hachi/Image_Editing/output/processed_dataset/FLUX1Fill_add_celoc")
    p.add_argument("--exemplar_index", type=int, default=0,
                   help="unused by FLUX; only filters the case list to match "
                        "inference_processed_dataset.py (default 0)")
    p.add_argument("--seed", type=int, default=666)
    p.add_argument("--size", type=int, default=768,
                   help="tile resolution, a multiple of 16; inference.py uses 768")
    p.add_argument("--steps", "--num_inference_steps", dest="steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=30.0,
                   help="FLUX.1-Fill-dev distilled guidance (the pipeline default)")
    p.add_argument("--cell", type=int, default=340, help="grid thumbnail size in px")
    p.add_argument("--limit", type=int, default=None, help="first N cases")
    p.add_argument("--start", type=int, default=0, help="skip the first N cases")
    p.add_argument("--overwrite", action="store_true",
                   help="redo cases whose outputs already exist")
    p.add_argument("--dry_run", action="store_true",
                   help="list the work without loading FLUX")
    p.add_argument("--offload", choices=("none", "model", "sequential"), default="none",
                   help="none: all on GPU (~34GB) | "
                        "model: per-module offload + vae slicing | "
                        "sequential: per-submodule offload (~5GB, much slower)")
    p.add_argument("--flux", default="/home/user03/aiotlab/hachi/Image_Editing/ckpt/black-forest-labs/FLUX.1-Fill-dev")

    args = p.parse_args()
    if args.size % 16:
        p.error("--size must be a multiple of 16 for FLUX")
    args.anno_dir = args.anno_dir or os.path.join(args.data_root, "Anno")
    args.image_dir = args.image_dir or os.path.join(args.data_root, "Image")
    args.mask_dir = args.mask_dir or os.path.join(args.data_root, "masks")
    args.pred_path = args.pred_path or os.path.join(args.data_root, "pred_cocount.jsonl")
    return args


def collect(args):
    """The case list, with predicted boxes, after listing, slicing and resume filtering."""
    for d in (args.anno_dir, args.image_dir, args.mask_dir):
        if not os.path.isdir(d):
            raise SystemExit(f"[error] no such directory: {d}")
    if not os.path.isfile(args.pred_path):
        raise SystemExit(f"[error] no such file: {args.pred_path}")

    # loc_index 0 only drops cases without any annotated loc_bbox
    items, skipped = get_inputs(args.anno_dir, args.image_dir, args.mask_dir,
                                0, args.exemplar_index)
    preds = load_pred_boxes(args.pred_path, args.pred_box_index)

    skipped.update(missing_pred=0, too_few_pred_boxes=0, degenerate_pred_box=0)
    kept_pred = []
    for it in items:
        pred = preds.get(it["case"], "missing_pred")
        if isinstance(pred, str):
            skipped[pred] += 1
            continue
        it["gt_loc_bbox"] = it["loc_bbox"]
        it["loc_bbox"] = pred
        kept_pred.append(it)
    items = kept_pred

    total = len(items)
    items = items[args.start:]
    if args.limit is not None:
        items = items[:args.limit]

    if not args.overwrite:
        kept = [it for it in items
                if not all(os.path.exists(os.path.join(args.work_dir, d, f"{it['case']}.png"))
                           for d in REQUIRED)]
    else:
        kept = items

    print(f"processed_dataset (celoc): usable={total} selected={len(items)} todo={len(kept)} "
          f"skipped={skipped}")
    return kept


def main():
    args = parse_args()
    size = (args.size, args.size)

    items = collect(args)
    todo = len(items)
    print(f"\ntotal cases to run: {todo}")
    if args.dry_run or todo == 0:
        return

    import torch
    from diffusers import FluxFillPipeline

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # The released backbone only: no LoRA, no Redux.
    print("loading FLUX.1-Fill-dev ...")
    pipe = FluxFillPipeline.from_pretrained(args.flux, torch_dtype=dtype)
    pipe.set_progress_bar_config(disable=True)

    # pipe.enable_vae_slicing() is gone from FluxFillPipeline in diffusers 0.41
    if args.offload == "model":
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_slicing()
    elif args.offload == "sequential":
        pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_slicing()
    else:
        pipe.to(device)

    done = failed = 0

    for d in OUT_DIRS:
        os.makedirs(os.path.join(args.work_dir, d), exist_ok=True)

    # one bar over every case, in place of the per-call denoising bars
    bar = tqdm(total=todo, unit="case", dynamic_ncols=True, smoothing=0.05)

    for item in items:
        case = item["case"]
        bar.set_description(case[:40])
        try:
            tar_image, tar_mask = load_case(item)
            ctx = build_target(tar_image, tar_mask, size)
            prompt = make_prompt(item["class_name"])

            in_tile = Image.fromarray(ctx["tile"])
            in_tile_mask = Image.fromarray((ctx["mask"] > 0).astype(np.uint8) * 255)

            generator = torch.Generator(device).manual_seed(args.seed)
            out_tile = pipe(
                prompt=prompt,
                image=in_tile,
                mask_image=in_tile_mask,
                height=args.size,
                width=args.size,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                max_sequence_length=512,
                generator=generator,
            ).images[0]

            # paste the whole output tile back into the full source image
            out_full = Image.fromarray(crop_back(
                np.array(out_tile), ctx["old_tar_image"].copy(),
                np.array([ctx["H1"], ctx["W1"], ctx["H2"], ctx["W2"]]),
                np.array(ctx["box"])))

            in_tile.save(os.path.join(args.work_dir, "input_model_image", f"{case}.png"))
            out_tile.save(os.path.join(args.work_dir, "output_model_image", f"{case}.png"))
            out_full.save(os.path.join(args.work_dir, "output_full_image", f"{case}.png"))
            in_tile_mask.save(os.path.join(args.work_dir, "input_model_mask", f"{case}.png"))
            Image.fromarray(tar_mask * 255).save(os.path.join(args.work_dir, "mask", f"{case}.png"))

            make_grid(
                [
                    ("source + pred loc bbox", _outline(tar_image, tar_mask)),
                    ("model input", _outline(in_tile, ctx["mask"])),
                    ("model mask (dilated)", ctx["mask"]),
                    ("model output (raw)", _outline(out_tile, ctx["mask"])),
                    ("final (crop_back)", out_full),
                ],
                os.path.join(args.work_dir, "grid_image", f"{case}.png"),
                cols=5, cell=args.cell,
                title=f"{case}   add [{item['class_name']}] @ pred[{args.pred_box_index}]  "
                      f"seed {args.seed}  steps {args.steps}  cfg {args.guidance_scale:g}",
            )

            with open(os.path.join(args.work_dir, "meta", f"{case}.json"), "w") as f:
                json.dump(dict(
                    task="addition", model=args.flux,
                    case=case, seed=args.seed,
                    steps=args.steps, guidance_scale=args.guidance_scale,
                    prompt=prompt,
                    class_name=item["class_name"],
                    source_image=item["img_path"],
                    loc_source=args.pred_path,
                    pred_box_index=args.pred_box_index,
                    loc_bbox=item["loc_bbox"],
                    gt_loc_bbox=item["gt_loc_bbox"],
                    source_size=[int(tar_image.shape[1]), int(tar_image.shape[0])],
                    tar_box_yyxx_crop=[int(v) for v in ctx["box"]],
                    H1=int(ctx["H1"]), W1=int(ctx["W1"]),
                    H2=int(ctx["H2"]), W2=int(ctx["W2"]),
                    tile_size=args.size,
                ), f, indent=2)

            done += 1
        except Exception as e:  # keep the sweep going
            failed += 1
            # tqdm.write keeps the bar intact instead of interleaving with it
            bar.write(f"[fail] {case}: {type(e).__name__}: {e}")
            bar.write(traceback.format_exc().rstrip())

        bar.set_postfix(ok=done, fail=failed)
        bar.update(1)

    bar.close()
    print(f"\ndone={done} failed={failed} -> {args.work_dir}")


if __name__ == "__main__":
    main()
