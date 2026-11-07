"""Run the FLUX.1-Fill-dev backbone (zero-shot, no LoRA, no Redux) over the
Location_with_anno_test benchmark.

Cases are listed by inference_location.get_inputs, so the case set and the
<case>.png output names are identical to the Insert-Anything Location run:

    source_image : <test_dir>/Image/<file>.png
    mask_image   : filled rectangle of phase1_output_k3 pred_box[0]  (loc bbox)
    prompt       : "Add exactly one <class>."  (class from Anno/<file>.json)

The target tile / paste-back follow inference_fsc_final_FLUX1_add.py, and the
same output folders are written (grid_image/, input_model_image/,
output_model_image/, output_full_image/, mask/, input_model_mask/, meta/).
"""

import argparse
import json
import os
import traceback

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from inference_fsc_final import (
    OUT_DIRS,
    REQUIRED,
    _outline,
    build_target,
    create_mask_from_bbox,
    make_grid,
)
from inference_location import get_inputs
from utils.utils import crop_back


def _read_rgb(path):
    img = cv2.imread(path)
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_case(item):
    """Source image and the loc_bbox mask."""
    tar_image = _read_rgb(item["img_path"])
    if tar_image is None:
        raise RuntimeError(f"cannot read {item['img_path']}")
    h, w = tar_image.shape[:2]

    tar_mask = (np.array(create_mask_from_bbox((w, h), item["loc_bbox"])) > 128).astype(np.uint8)
    if tar_mask.sum() < 10:
        raise RuntimeError("empty mask")
    return tar_image, tar_mask


# ------------------------------------------------------------------ main --

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test_dir",
                   default="/home/user03/aiotlab/hachi/Image_Editing/data/Location_with_anno_test",
                   help="folder holding Anno/, Image/ and phase1_output_k3/")
    p.add_argument("--obj_root", default="/home/user03/aiotlab/hachi/Image_Editing/data/OBJ_INS_FSC",
                   help="root with test/ and val/ case folders")
    p.add_argument("--work_dir",
                   default="/home/user03/aiotlab/hachi/Image_Editing/output/Location_with_anno_test/"
                           "FLUX1Fill_Add/Turn_1")
    p.add_argument("--turn", type=int, default=1)
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
    return args


def collect(args):
    """The case list after listing, slicing and resume filtering."""
    for d in ("Anno", "Image", "phase1_output_k3"):
        if not os.path.isdir(os.path.join(args.test_dir, d)):
            raise SystemExit(f"[error] no such directory: {os.path.join(args.test_dir, d)}")

    items, skipped = get_inputs(args.test_dir, args.turn, args.obj_root)
    # the OBJ_INS_FSC case folder holds the un-removed ground_truth.jpg
    for it in items:
        it["gt_path"] = os.path.join(os.path.dirname(it["exam_mask_path"]), "ground_truth.jpg")

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

    print(f"Location turn {args.turn}: usable={total} selected={len(items)} todo={len(kept)} "
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
        prompt = f"Add exactly one {item['class_name']}."
        bar.set_description(case)
        try:
            tar_image, tar_mask = load_case(item)
            ctx = build_target(tar_image, tar_mask, size)

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

            gt = _read_rgb(item["gt_path"])
            make_grid(
                [
                    ("source + loc bbox", _outline(tar_image, tar_mask)),
                    ("model input", _outline(in_tile, ctx["mask"])),
                    ("model mask (dilated)", ctx["mask"]),
                    ("model output (raw)", _outline(out_tile, ctx["mask"])),
                    ("final (crop_back)", _outline(out_full, tar_mask)),
                    ("GT", gt if gt is not None else np.zeros_like(tar_image)),
                ],
                os.path.join(args.work_dir, "grid_image", f"{case}.png"),
                cols=6, cell=args.cell,
                title=f"Location / {case}   add [{item['class_name']}]   "
                      f"seed {args.seed}  steps {args.steps}  cfg {args.guidance_scale:g}",
            )

            with open(os.path.join(args.work_dir, "meta", f"{case}.json"), "w") as f:
                json.dump(dict(
                    task="addtion", model=args.flux,
                    case=case, file=item["file"], turn=args.turn, seed=args.seed,
                    steps=args.steps, guidance_scale=args.guidance_scale,
                    prompt=prompt,
                    class_name=item["class_name"],
                    source_image=item["img_path"],
                    addtion_gt_image=item["gt_path"],
                    loc_bbox=item["loc_bbox"], loc_is_valid=item["loc_is_valid"],
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
