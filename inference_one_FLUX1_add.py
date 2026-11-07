"""Run zero-shot FLUX.1-Fill-dev addition on one case.

The single-case version of inference_reformat_coco_FLUX1_add.py: same loc_bbox
mask, same 768x768 target crop, same prompt "Add exactly one <class_name>.",
same paste-back.  The case is given either by name from a dataset laid out as
Reformat_COCO / processed_dataset (Image/<case>.jpg + Anno/<case>.json):

    python inference_one_FLUX1_add.py --case 000000016010_24
    python inference_one_FLUX1_add.py --case 000000016010_24 --loc_index 2

or directly by an image, a box and a class name (or a full prompt):

    python inference_one_FLUX1_add.py --image some.jpg \\
        --loc_bbox 215 263 268 333 --class_name zebra

Written to <out_dir>/<name>/:

    grid.png          labelled visualisation of the whole pass
    input_model.png   the crop handed to the model                 (768x768)
    input_mask.png    its (dilated) mask                           (768x768)
    output_model.png  the model's raw output                       (768x768)
    output_full.png   final result pasted back into the full source image
    mask.png          the loc_bbox mask at source resolution
    meta.json         prompt, boxes and crop geometry
"""

import argparse
import json
import os

import numpy as np
from PIL import Image

from inference_fsc_final import _outline, build_target, make_grid
from inference_processed_dataset import find_image
from inference_processed_dataset_FLUX1_add import load_case, make_prompt
from utils.utils import crop_back


def resolve_case(args):
    """The item dict load_case expects, plus a name for the output folder."""
    if args.case:
        img_path = find_image(os.path.join(args.data_root, "Image"), args.case)
        if img_path is None:
            raise SystemExit(f"[error] no image for case {args.case} under {args.data_root}/Image")
        with open(os.path.join(args.data_root, "Anno", f"{args.case}.json"), "r") as f:
            anno = json.load(f)
        loc_bboxes = anno.get("loc_bbox") or []
        if len(loc_bboxes) <= args.loc_index:
            raise SystemExit(f"[error] {args.case} has {len(loc_bboxes)} loc_bbox, "
                             f"--loc_index {args.loc_index} is out of range")
        loc_bbox = args.loc_bbox or loc_bboxes[args.loc_index]
        class_name = args.class_name or anno.get("class_name", "")
        name = f"{args.case}_loc{args.loc_index}" if not args.loc_bbox else args.case
    else:
        if not (args.image and args.loc_bbox):
            raise SystemExit("[error] give --case, or --image with --loc_bbox")
        img_path = args.image
        loc_bbox = args.loc_bbox
        class_name = args.class_name or ""
        name = os.path.splitext(os.path.basename(args.image))[0]

    if not (args.prompt or class_name):
        raise SystemExit("[error] no class_name to build the prompt from; give --class_name or --prompt")

    return name, dict(img_path=img_path, class_name=class_name,
                      loc_bbox=[float(v) for v in loc_bbox])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", default=None, help="case name, e.g. 000000016010_24")
    p.add_argument("--data_root", default="/home/user03/aiotlab/hachi/Image_Editing/data/Reformat_COCO",
                   help="root holding Anno/ and Image/ (with --case)")
    p.add_argument("--loc_index", type=int, default=0,
                   help="which loc_bbox of the case to insert into (default 0)")
    p.add_argument("--image", default=None, help="source image (without --case)")
    p.add_argument("--loc_bbox", type=float, nargs=4, default=None, metavar=("X1", "Y1", "X2", "Y2"),
                   help="box to insert into; overrides the case's loc_bbox")
    p.add_argument("--class_name", default=None, help="overrides the case's class_name")
    p.add_argument("--prompt", default=None,
                   help='full prompt; default "Add exactly one <class_name>."')
    p.add_argument("--out_dir",
                   default="/home/user03/aiotlab/hachi/Image_Editing/output/one_FLUX1_add")
    p.add_argument("--seed", type=int, default=666)
    p.add_argument("--size", type=int, default=768,
                   help="tile resolution, a multiple of 16; inference.py uses 768")
    p.add_argument("--steps", "--num_inference_steps", dest="steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=30.0,
                   help="FLUX.1-Fill-dev distilled guidance (the pipeline default)")
    p.add_argument("--cell", type=int, default=340, help="grid thumbnail size in px")
    p.add_argument("--offload", choices=("none", "model", "sequential"), default="none",
                   help="none: all on GPU (~34GB) | "
                        "model: per-module offload + vae slicing | "
                        "sequential: per-submodule offload (~5GB, much slower)")
    p.add_argument("--flux", default="/home/user03/aiotlab/hachi/Image_Editing/ckpt/black-forest-labs/FLUX.1-Fill-dev")

    args = p.parse_args()
    if args.size % 16:
        p.error("--size must be a multiple of 16 for FLUX")
    return args


def main():
    args = parse_args()
    size = (args.size, args.size)

    name, item = resolve_case(args)
    prompt = args.prompt or make_prompt(item["class_name"])
    out_dir = os.path.join(args.out_dir, name)

    # fail on a bad image / box before spending time loading FLUX
    tar_image, tar_mask = load_case(item)
    ctx = build_target(tar_image, tar_mask, size)

    print(f"case     : {name}")
    print(f"image    : {item['img_path']}")
    print(f"loc_bbox : {item['loc_bbox']}")
    print(f"prompt   : {prompt}")
    print(f"out_dir  : {out_dir}")

    import torch
    from diffusers import FluxFillPipeline

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # The released backbone only: no LoRA, no Redux.
    print("loading FLUX.1-Fill-dev ...")
    pipe = FluxFillPipeline.from_pretrained(args.flux, torch_dtype=dtype)

    # pipe.enable_vae_slicing() is gone from FluxFillPipeline in diffusers 0.41
    if args.offload == "model":
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_slicing()
    elif args.offload == "sequential":
        pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_slicing()
    else:
        pipe.to(device)

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

    os.makedirs(out_dir, exist_ok=True)
    in_tile.save(os.path.join(out_dir, "input_model.png"))
    in_tile_mask.save(os.path.join(out_dir, "input_mask.png"))
    out_tile.save(os.path.join(out_dir, "output_model.png"))
    out_full.save(os.path.join(out_dir, "output_full.png"))
    Image.fromarray(tar_mask * 255).save(os.path.join(out_dir, "mask.png"))

    make_grid(
        [
            ("source + loc bbox", _outline(tar_image, tar_mask)),
            ("model input", _outline(in_tile, ctx["mask"])),
            ("model mask (dilated)", ctx["mask"]),
            ("model output (raw)", _outline(out_tile, ctx["mask"])),
            ("final (crop_back)", out_full),
        ],
        os.path.join(out_dir, "grid.png"),
        cols=5, cell=args.cell,
        title=f"{name}   \"{prompt}\"   "
              f"seed {args.seed}  steps {args.steps}  cfg {args.guidance_scale:g}",
    )

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(dict(
            task="addition", model=args.flux,
            case=name, seed=args.seed,
            steps=args.steps, guidance_scale=args.guidance_scale,
            prompt=prompt,
            class_name=item["class_name"],
            source_image=item["img_path"],
            loc_index=args.loc_index if args.case and not args.loc_bbox else None,
            loc_bbox=item["loc_bbox"],
            source_size=[int(tar_image.shape[1]), int(tar_image.shape[0])],
            tar_box_yyxx_crop=[int(v) for v in ctx["box"]],
            H1=int(ctx["H1"]), W1=int(ctx["W1"]),
            H2=int(ctx["H2"]), W2=int(ctx["W2"]),
            tile_size=args.size,
        ), f, indent=2)

    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
