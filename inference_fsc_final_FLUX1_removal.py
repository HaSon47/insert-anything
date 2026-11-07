"""Zero-shot object REMOVAL over FSC_final with the plain FLUX.1-Fill-dev backbone.

Removal counterpart of inference_fsc_final.py, meant as the baseline before any
removal-specific fine-tuning.  The model is FLUX.1-Fill-dev exactly as released:
no Insert-Anything LoRA, no FLUX.1-Redux-dev, no reference image and no diptych.
The single source tile is inpainted under a background-oriented text prompt.

Inputs, for --turn T (default 1):

    source_image : OBJ_INS_FSC/{test,val}/<case>/inpainted_turn_<T>.png
    mask_image   : filled rectangle of annotation["inpainted_bboxes"][T]
                   (== [1] for T=1).  inpainted_turn_<T>.png only has boxes
                   [0..T-1] removed, so this object is still in it.
    ground truth : <case>/inpainted_turn_<T+1>.png, the same image with the object
                   removed (recorded in meta/, shown in the grid)

The crop / dilation / pad / resize / crop_back stages are imported from
inference_fsc_final.py.  Per case:

    grid_image/          labelled visualisation of the whole pass
    input_model_image/   the tile handed to FLUX.1-Fill                (768x768)
    output_model_image/  the model's raw output tile                   (768x768)
    output_full_image/   final result pasted back with crop_back
    mask/                removal mask on the full source image (before dilation)
    input_model_mask/    dilated removal mask on the tile, as given to the model
    meta/<case>.json     prompt, settings, crop geometry
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
    SPLITS,
    _outline,
    build_target,
    create_mask_from_bbox,
    make_grid,
)
from utils.utils import crop_back

DEFAULT_PROMPT = (
    "Fill the masked region with a natural, context-consistent background, "
    "preserving the surrounding textures, lighting, and scene structure "
    "without generating the removed object."
)

# DEFAULT_PROMPT = (
#     "A natural, seamless continuation of the surrounding background, with matching textures and lighting."
# )


# ------------------------------------------------------- dataset / inputs --

def get_inputs(test_dir, turn, root_path):
    """One entry per usable case in `test_dir`, resolved against OBJ_INS_FSC."""
    inputs, skipped = [], {"missing_case": 0, "missing_file": 0, "too_few_bboxes": 0}

    for img in sorted(os.listdir(test_dir)):
        img_folder = img.split(".")[0]
        img_folder_path = os.path.join(root_path, "test", img_folder)
        if not os.path.exists(img_folder_path):
            img_folder_path = os.path.join(root_path, "val", img_folder)
        if not os.path.isdir(img_folder_path):
            skipped["missing_case"] += 1
            continue

        # the source still holds every object from bboxes[turn] on
        img_path = os.path.join(img_folder_path, f"inpainted_turn_{turn}.png")

        anno_path = os.path.join(img_folder_path, "annotation.json")
        if not (os.path.exists(img_path) and os.path.exists(anno_path)):
            skipped["missing_file"] += 1
            continue
        with open(anno_path, "r") as f:
            anno = json.load(f)

        # need an object left to remove
        if len(anno["inpainted_bboxes"]) <= turn:
            skipped["too_few_bboxes"] += 1
            continue

        inputs.append(dict(
            case=img_folder,
            class_name=anno["class_based_caption"],
            img_path=img_path,
            loc_bbox=anno["inpainted_bboxes"][turn],
            gt_path=os.path.join(img_folder_path, f"inpainted_turn_{turn + 1}.png"),
        ))

    return inputs, skipped


def _read_rgb(path):
    img = cv2.imread(path)
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_case(item):
    """Source image and the loc_bbox removal mask."""
    tar_image = _read_rgb(item["img_path"])
    if tar_image is None:
        raise RuntimeError(f"cannot read {item['img_path']}")
    h, w = tar_image.shape[:2]

    # PIL clips the -1 coordinates some FSC boxes carry
    tar_mask = (np.array(create_mask_from_bbox((w, h), item["loc_bbox"])) > 128).astype(np.uint8)
    if tar_mask.sum() < 10:
        raise RuntimeError("empty removal mask")
    return tar_image, tar_mask


# ------------------------------------------------------------------ main --

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test_root", default="/mnt/disk1/aiotlab/hachi/data/FSC_final",
                   help="folder holding the FSC_* split listings")
    p.add_argument("--splits", nargs="+", default=None,
                   help=f"subset of {SPLITS} (default: every split dir found)")
    p.add_argument("--obj_root", default="/mnt/disk1/aiotlab/hachi/data/OBJ_INS_FSC",
                   help="root with test/ and val/ case folders")
    p.add_argument("--work_dir",
                   default="/mnt/disk1/aiotlab/hachi/Output/FSC_final/FLUX1Fill_Removal/Turn_1")
    p.add_argument("--turn", type=int, default=1,
                   help="source inpainted_turn_<turn>.png, remove inpainted_bboxes[turn]")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--seed", type=int, default=666)
    p.add_argument("--size", type=int, default=768,
                   help="tile resolution, a multiple of 16; inference.py uses 768")
    p.add_argument("--steps", "--num_inference_steps", dest="steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=30.0,
                   help="FLUX.1-Fill-dev distilled guidance (the pipeline default)")
    p.add_argument("--cell", type=int, default=340, help="grid thumbnail size in px")
    p.add_argument("--limit", type=int, default=None, help="first N cases per split")
    p.add_argument("--start", type=int, default=0, help="skip the first N cases per split")
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
    """[(split, [items...])] after listing, slicing and resume filtering."""
    splits = args.splits
    if splits is None:
        splits = [d for d in SPLITS if os.path.isdir(os.path.join(args.test_root, d))]
        splits += sorted(d for d in os.listdir(args.test_root)
                         if d not in SPLITS and os.path.isdir(os.path.join(args.test_root, d)))
    plan = []
    for split in splits:
        test_dir = os.path.join(args.test_root, split)
        if not os.path.isdir(test_dir):
            print(f"[warn] no such split dir: {test_dir}")
            continue
        items, skipped = get_inputs(test_dir, args.turn, args.obj_root)
        total = len(items)
        items = items[args.start:]
        if args.limit is not None:
            items = items[:args.limit]

        out_root = os.path.join(args.work_dir, split)
        if not args.overwrite:
            kept = [it for it in items
                    if not all(os.path.exists(os.path.join(out_root, d, f"{it['case']}.png"))
                               for d in REQUIRED)]
        else:
            kept = items
        print(f"{split}: usable={total} selected={len(items)} todo={len(kept)} skipped={skipped}")
        plan.append((split, kept))
    return plan


def main():
    args = parse_args()
    size = (args.size, args.size)

    plan = collect(args)
    todo = sum(len(v) for _, v in plan)
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

    # one bar over every case of every split, in place of the per-call denoising bars
    bar = tqdm(total=todo, unit="case", dynamic_ncols=True, smoothing=0.05)

    for split, items in plan:
        out_root = os.path.join(args.work_dir, split)
        for d in OUT_DIRS:
            os.makedirs(os.path.join(out_root, d), exist_ok=True)

        for item in items:
            case = item["case"]
            bar.set_description(f"{split}/{case}")
            try:
                tar_image, tar_mask = load_case(item)
                ctx = build_target(tar_image, tar_mask, size)

                in_tile = Image.fromarray(ctx["tile"])
                in_tile_mask = Image.fromarray((ctx["mask"] > 0).astype(np.uint8) * 255)

                generator = torch.Generator(device).manual_seed(args.seed)
                out_tile = pipe(
                    prompt=args.prompt,
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

                in_tile.save(os.path.join(out_root, "input_model_image", f"{case}.png"))
                out_tile.save(os.path.join(out_root, "output_model_image", f"{case}.png"))
                out_full.save(os.path.join(out_root, "output_full_image", f"{case}.png"))
                in_tile_mask.save(os.path.join(out_root, "input_model_mask", f"{case}.png"))
                Image.fromarray(tar_mask * 255).save(os.path.join(out_root, "mask", f"{case}.png"))

                gt = _read_rgb(item["gt_path"])
                make_grid(
                    [
                        ("source + removal box", _outline(tar_image, tar_mask)),
                        ("model input", _outline(in_tile, ctx["mask"])),
                        ("model mask (dilated)", ctx["mask"]),
                        ("model output (raw)", out_tile),
                        ("final (crop_back)", out_full),
                        (f"GT inpainted_turn_{args.turn + 1}",
                         gt if gt is not None else np.zeros_like(tar_image)),
                    ],
                    os.path.join(out_root, "grid_image", f"{case}.png"),
                    cols=6, cell=args.cell,
                    title=f"{split} / {case}   remove [{item['class_name']}]   "
                          f"seed {args.seed}  steps {args.steps}  cfg {args.guidance_scale:g}",
                )

                with open(os.path.join(out_root, "meta", f"{case}.json"), "w") as f:
                    json.dump(dict(
                        task="removal", model=args.flux,
                        case=case, split=split, turn=args.turn, seed=args.seed,
                        steps=args.steps, guidance_scale=args.guidance_scale,
                        prompt=args.prompt,
                        class_name=item["class_name"],
                        source_image=item["img_path"],
                        removal_gt_image=item["gt_path"],
                        loc_bbox=item["loc_bbox"],
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
                bar.write(f"[fail] {split}/{case}: {type(e).__name__}: {e}")
                bar.write(traceback.format_exc().rstrip())

            bar.set_postfix(ok=done, fail=failed)
            bar.update(1)

    bar.close()
    print(f"\ndone={done} failed={failed} -> {args.work_dir}")


if __name__ == "__main__":
    main()
