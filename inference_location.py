"""Run Insert-Anything over the Location_with_anno_test benchmark.

Input construction follows UniCombine_Ins_Rm/inference_location.py
(see `get_inputs` / `get_conditions` there), driven by
UniCombine_Ins_Rm/scripts/infer_location.sh:

    <test_dir>/Anno/<file>.json              class + origin "<case>/inpainted_turn_<t>.png"
    <test_dir>/Image/<file>.png              the input image (== OBJ_INS_FSC origin)
    <test_dir>/phase1_output_k3/<file>.json  predicted location, pred_box[0] (xyxy)

Only files whose origin turn equals --turn are kept.  Inputs handed to
Insert-Anything, per case:

    source_image : Image/<file>.png
    mask_image   : filled rectangle of phase1_output_k3 pred_box[0]  (loc bbox)
    ref_image    : the same image -- the exemplar is another instance still
                   present in it
    ref_mask     : OBJ_INS_FSC/{test,val}/<case>/mask_<turn+1>.png, i.e. the mask
                   of annotation["inpainted_bboxes"][turn]
                   (--ref_mask_source bbox uses that plain rectangle instead,
                   matching UniCombine's `--padding` / no-`--use_mask` subject
                   condition)

Cases with len(inpainted_bboxes) <= turn are skipped, exactly as upstream.
Outputs are named <case>.png (the OBJ_INS_FSC folder, e.g. 1050_b1), the same
names UniCombine writes, so results from both can be paired directly.

Every Insert-Anything stage (ref tile, target crop, diptych, paste-back), the
input loader and the grid renderer are imported from inference_fsc_final, so
the benchmarks run identical pre/post-processing.  Per case four images are
written:

    grid_image/        labelled visualisation of the whole pass
    input_model_image/ right panel of the diptych handed to the model  (768x768)
    output_model_image/right panel of the model's raw output           (768x768)
    output_full_image/ final result pasted back into the full source image

plus the two masks needed to score background preservation on those panels
(mask/ for the full image, input_model_mask/ for the 768x768 panels) and a
meta/<case>.json describing the crop geometry.
"""

import argparse
import json
import os
import re
import traceback

import numpy as np
from PIL import Image
from tqdm import tqdm

from inference_fsc_final import (
    OUT_DIRS,
    REQUIRED,
    _outline,
    build_diptych,
    build_ref_tile,
    build_target,
    load_case,
    make_grid,
    split_output,
)

REPO = os.path.dirname(os.path.abspath(__file__))

ORIGIN_RE = re.compile(r"inpainted_turn_(\d+)\.png$")


# ------------------------------------------------------- dataset / inputs --
# Ported from UniCombine_Ins_Rm/inference_location.py:50-99 and 176-225.

def get_inputs(test_dir, turn, root_path):
    """One entry per usable case of `turn` in `test_dir`, resolved against OBJ_INS_FSC."""
    img_test_path = os.path.join(test_dir, "Image")
    anno_class_test_path = os.path.join(test_dir, "Anno")
    anno_bbox_test_path = os.path.join(test_dir, "phase1_output_k3")

    inputs = []
    skipped = {"missing_case": 0, "missing_file": 0, "too_few_bboxes": 0}

    for file in sorted(os.listdir(anno_class_test_path)):
        if not file.endswith(".json"):
            continue

        # get turn_i and img_folder name
        with open(os.path.join(anno_class_test_path, file), "r") as f:
            anno_class = json.load(f)
        origin_detail = anno_class["origin"]
        m = ORIGIN_RE.search(origin_detail)
        if m is None or int(m.group(1)) != turn:
            continue
        turn_i = int(m.group(1))
        img_folder = origin_detail.split("/")[0]

        # get fill + location bbox
        img_path = os.path.join(img_test_path, file.replace(".json", ".png"))
        loc_bbox_path = os.path.join(anno_bbox_test_path, file)
        if not (os.path.exists(img_path) and os.path.exists(loc_bbox_path)):
            skipped["missing_file"] += 1
            continue
        with open(loc_bbox_path, "r") as f:
            anno_loc = json.load(f)
        loc_bbox = anno_loc["pred_box"][0]

        # get subject
        img_folder_path = os.path.join(root_path, "test", img_folder)
        if not os.path.exists(img_folder_path):
            img_folder_path = os.path.join(root_path, "val", img_folder)
        anno_path = os.path.join(img_folder_path, "annotation.json")
        if not os.path.exists(anno_path):
            skipped["missing_case"] += 1
            continue
        with open(anno_path, "r") as f:
            anno = json.load(f)

        # only keep cases that still have an exemplar instance left in the image
        if len(anno["inpainted_bboxes"]) <= turn_i:
            skipped["too_few_bboxes"] += 1
            continue
        exam_bbox = anno["inpainted_bboxes"][turn_i]
        exam_mask_path = os.path.join(img_folder_path, f"mask_{turn_i + 1}.png")

        inputs.append(dict(
            case=img_folder,
            file=file[:-len(".json")],
            class_name=anno_class["class"],
            img_path=img_path,
            loc_bbox=[float(v) for v in loc_bbox],
            loc_is_valid=anno_loc.get("is_valid"),
            exam_bbox=exam_bbox,
            exam_mask_path=exam_mask_path,
        ))

    return inputs, skipped


# ------------------------------------------------------------------ main --

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test_dir",
                   default="/mnt/disk1/aiotlab/hachi/data/Location_with_anno_test/"
                           "Location_with_anno_test",
                   help="folder holding Anno/, Image/ and phase1_output_k3/")
    p.add_argument("--obj_root", default="/mnt/disk1/aiotlab/hachi/data/OBJ_INS_FSC",
                   help="root with test/ and val/ case folders")
    p.add_argument("--work_dir",
                   default="/mnt/disk1/aiotlab/hachi/Output/Location_with_anno_test/"
                           "InsertAnything/Turn_1")
    p.add_argument("--turn", type=int, default=1)
    p.add_argument("--ref_mask_source", choices=("mask", "bbox"), default="mask",
                   help="mask: use mask_<turn+1>.png | bbox: use the exam_bbox rectangle")
    p.add_argument("--seed", type=int, default=666)
    p.add_argument("--size", type=int, default=768, help="tile size, inference.py uses 768")
    p.add_argument("--steps", type=int, default=None,
                   help="num_inference_steps (default: the pipeline's own, 50)")
    p.add_argument("--cell", type=int, default=340, help="grid thumbnail size in px")
    p.add_argument("--limit", type=int, default=None, help="first N cases")
    p.add_argument("--start", type=int, default=0, help="skip the first N cases")
    p.add_argument("--overwrite", action="store_true",
                   help="redo cases whose outputs already exist")
    p.add_argument("--dry_run", action="store_true",
                   help="list the work without loading FLUX")
    p.add_argument("--offload", choices=("none", "model", "sequential"), default="none",
                   help="none: all on GPU (~40GB, like inference.py) | "
                        "model: per-module offload + vae slicing (~26GB) | "
                        "sequential: per-submodule offload (~5GB, much slower)")
    p.add_argument("--flux", default="/mnt/disk1/aiotlab/hachi/checkpoints/FLUX/FLUX.1-Fill-dev")
    p.add_argument("--redux", default="/mnt/disk1/aiotlab/hachi/checkpoints/FLUX/FLUX.1-Redux-dev")
    p.add_argument("--lora", default=os.path.join(
        REPO, "ckpt/Insert-Anything/20250321_steps5000_pytorch_lora_weights.safetensors"))
    return p.parse_args()


def collect(args):
    """The case list after listing, slicing and resume filtering."""
    for d in ("Anno", "Image", "phase1_output_k3"):
        if not os.path.isdir(os.path.join(args.test_dir, d)):
            raise SystemExit(f"[error] no such directory: {os.path.join(args.test_dir, d)}")

    items, skipped = get_inputs(args.test_dir, args.turn, args.obj_root)
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
    from diffusers import FluxFillPipeline, FluxPriorReduxPipeline

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Load the pre-trained model and LoRA weights (inference.py:15-33).
    # Loaded once here and reused for every case in the sweep.
    print("loading FLUX.1-Fill-dev ...")
    pipe = FluxFillPipeline.from_pretrained(args.flux, torch_dtype=dtype)
    pipe.load_lora_weights(args.lora)
    print("loading FLUX.1-Redux-dev ...")
    redux = FluxPriorReduxPipeline.from_pretrained(args.redux).to(dtype=dtype)

    # the per-case denoising bars would drown out the sweep bar below
    pipe.set_progress_bar_config(disable=True)
    redux.set_progress_bar_config(disable=True)

    if args.offload == "model":
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
        redux.enable_model_cpu_offload()
    elif args.offload == "sequential":
        # Redux stays on plain model offload: sequential offload leaves its
        # SigLIP encoder on the meta device and the call blows up.
        pipe.enable_sequential_cpu_offload()
        pipe.enable_vae_slicing()
        redux.enable_model_cpu_offload()
    else:
        pipe.to(device)
        redux.to(device)

    extra = {"num_inference_steps": args.steps} if args.steps else {}
    done = failed = 0

    for d in OUT_DIRS:
        os.makedirs(os.path.join(args.work_dir, d), exist_ok=True)

    # one bar over every case, in place of the per-call denoising bars
    bar = tqdm(total=todo, unit="case", dynamic_ncols=True, smoothing=0.05)

    for item in items:
        case = item["case"]
        bar.set_description(case)
        try:
            ref_image, ref_mask, tar_image, tar_mask, tar_mask_pil, used = load_case(
                item, args.ref_mask_source)

            ref_tile = build_ref_tile(ref_image, ref_mask, size)
            ctx = build_target(tar_image, tar_mask, size)
            diptych_ref_tar, mask_diptych = build_diptych(ref_tile, ctx)

            # Extract the features of the reference image (inference.py:90)
            pipe_prior_output = redux(Image.fromarray(ref_tile))
            if args.offload != "none":
                pipe_prior_output = {k: v.to(device) for k, v in pipe_prior_output.items()}

            generator = torch.Generator(device).manual_seed(args.seed)
            edited_image = pipe(
                image=diptych_ref_tar,
                mask_image=mask_diptych,
                height=mask_diptych.size[1],
                width=mask_diptych.size[0],
                max_sequence_length=512,
                generator=generator,
                **extra,
                **pipe_prior_output,
            ).images[0]

            out_panel, out_full = split_output(edited_image, ctx)
            in_panel = Image.fromarray(ctx["tile"])
            in_panel_mask = Image.fromarray((ctx["mask"] > 0).astype(np.uint8) * 255)

            in_panel.save(os.path.join(args.work_dir, "input_model_image", f"{case}.png"))
            out_panel.save(os.path.join(args.work_dir, "output_model_image", f"{case}.png"))
            out_full.save(os.path.join(args.work_dir, "output_full_image", f"{case}.png"))
            in_panel_mask.save(os.path.join(args.work_dir, "input_model_mask", f"{case}.png"))
            tar_mask_pil.save(os.path.join(args.work_dir, "mask", f"{case}.png"))

            make_grid(
                [
                    ("source + loc bbox", _outline(tar_image, tar_mask)),
                    (f"ref tile -> Redux ({used})", ref_tile),
                    ("model input (right panel)", _outline(in_panel, ctx["mask"])),
                    ("model output (right panel)", out_panel),
                    ("final (pasted back)", out_full),
                ],
                os.path.join(args.work_dir, "grid_image", f"{case}.png"),
                cols=5, cell=args.cell,
                title=f"Location / {case}   [{item['class_name']}]   seed {args.seed}",
            )

            with open(os.path.join(args.work_dir, "meta", f"{case}.json"), "w") as f:
                json.dump(dict(
                    case=case, file=item["file"], turn=args.turn, seed=args.seed,
                    class_name=item["class_name"],
                    source_image=item["img_path"],
                    ref_mask_path=item["exam_mask_path"],
                    ref_mask_source=used,
                    loc_bbox=item["loc_bbox"], loc_is_valid=item["loc_is_valid"],
                    exam_bbox=item["exam_bbox"],
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
