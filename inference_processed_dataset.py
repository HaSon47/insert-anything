"""Run Insert-Anything over the `processed_dataset` benchmark.

Same task shape as inference_fsc_final.py -- insert one instance of a class into
an empty spot, using another instance still visible in the same photo as the
exemplar -- but over a flat dataset instead of FSC_final's four splits:

    <data_root>/Image/<case>.jpg          the input image
    <data_root>/Anno/<case>.json          the annotation
    <data_root>/masks/<case>/mask_NNN.png one mask per exemplar

Inputs handed to Insert-Anything, per case:

    source_image : Image/<case>.jpg
    mask_image   : filled rectangle of anno["loc_bbox"][--loc_index]
    ref_image    : the same image -- the exemplar is another instance of the
                   class that is already in frame
    ref_mask     : masks/<case>/mask_<--exemplar_index:03d>.png, which is the
                   segmentation of anno["counting_anno"]["exemplars"][i]
                   (--ref_mask_source bbox uses that plain rectangle instead)

The index pairing exemplars[i] <-> mask_{i:03d}.png was verified over the
dataset: off-diagonal IoU between exemplar boxes and mask boxes is 0.

Every Insert-Anything stage (ref tile, target crop, diptych, paste-back) and the
grid renderer are imported from inference_fsc_final, so the two benchmarks run
identical pre/post-processing.  Per case four images are written:

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
import traceback

import cv2
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
    create_mask_from_bbox,
    make_grid,
    split_output,
)

REPO = os.path.dirname(os.path.abspath(__file__))

IMG_EXTS = (".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG")


# ------------------------------------------------------- dataset / inputs --

def find_image(image_dir, case):
    for ext in IMG_EXTS:
        p = os.path.join(image_dir, case + ext)
        if os.path.exists(p):
            return p
    return None


def get_inputs(anno_dir, image_dir, mask_dir, loc_index, exemplar_index):
    """One entry per usable case under `anno_dir`."""
    inputs = []
    skipped = {"missing_image": 0, "missing_mask": 0, "too_few_loc_bboxes": 0,
               "too_few_exemplars": 0}

    for fn in sorted(os.listdir(anno_dir)):
        if not fn.endswith(".json"):
            continue
        case = fn[:-len(".json")]

        img_path = find_image(image_dir, case)
        if img_path is None:
            skipped["missing_image"] += 1
            continue

        with open(os.path.join(anno_dir, fn), "r") as f:
            anno = json.load(f)

        loc_bboxes = anno.get("loc_bbox") or []
        if len(loc_bboxes) <= loc_index:
            skipped["too_few_loc_bboxes"] += 1
            continue

        exemplars = (anno.get("counting_anno") or {}).get("exemplars") or []
        if len(exemplars) <= exemplar_index:
            skipped["too_few_exemplars"] += 1
            continue

        # exemplars[i] is the box of mask_{i:03d}.png
        exam_mask_path = os.path.join(mask_dir, case, f"mask_{exemplar_index:03d}.png")

        inputs.append(dict(
            case=case,
            class_name=anno.get("class_name", ""),
            img_path=img_path,
            loc_bbox=[float(v) for v in loc_bboxes[loc_index]],
            exam_bbox=[float(v) for v in exemplars[exemplar_index]],
            exam_mask_path=exam_mask_path,
        ))

    return inputs, skipped


def load_case(item, ref_mask_source):
    """Build the four Insert-Anything inputs, then load them as inference.py:45-51."""
    tar_image = cv2.imread(item["img_path"])
    if tar_image is None:
        raise RuntimeError(f"cannot read {item['img_path']}")
    tar_image = cv2.cvtColor(tar_image, cv2.COLOR_BGR2RGB)
    h, w = tar_image.shape[:2]

    # source_image == ref_image: the exemplar is another instance in the same photo
    ref_image = tar_image.copy()

    # mask_image: the location bbox, as UniCombine's `create_mask_from_bbox`
    tar_mask_pil = create_mask_from_bbox((w, h), item["loc_bbox"])
    tar_mask = (np.array(tar_mask_pil) > 128).astype(np.uint8)
    tar_mask = cv2.resize(tar_mask, (tar_image.shape[1], tar_image.shape[0]))

    # ref_mask: the exemplar's segmentation mask, or its bbox rectangle
    used = ref_mask_source
    ref_mask = None
    if ref_mask_source == "mask" and os.path.exists(item["exam_mask_path"]):
        m = cv2.imread(item["exam_mask_path"])
        if m is not None:
            ref_mask = (m > 128).astype(np.uint8)[:, :, 0]
            if ref_mask.shape != ref_image.shape[:2]:
                ref_mask = cv2.resize(ref_mask, (ref_image.shape[1], ref_image.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
            if ref_mask.sum() < 10:       # degenerate -- get_bbox_from_mask would
                ref_mask = None           # return the whole frame
    if ref_mask is None:
        used = "bbox"
        ref_mask = (np.array(create_mask_from_bbox((w, h), item["exam_bbox"])) > 128).astype(np.uint8)
        if ref_mask.sum() < 10:
            raise RuntimeError("empty reference mask")

    if tar_mask.sum() < 10:
        raise RuntimeError("empty source mask")

    return ref_image, ref_mask, tar_image, tar_mask, tar_mask_pil, used


# ------------------------------------------------------------------ main --

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", default="/home/user03/aiotlab/hachi/Image_Editing/data/processed_dataset",
                   help="root holding Anno/, Image/ and masks/")
    p.add_argument("--anno_dir", default=None, help="default: <data_root>/Anno")
    p.add_argument("--image_dir", default=None, help="default: <data_root>/Image")
    p.add_argument("--mask_dir", default=None, help="default: <data_root>/masks")
    p.add_argument("--work_dir",
                   default="/home/user03/aiotlab/hachi/Image_Editing/output/processed_dataset/InsertAnything")
    p.add_argument("--loc_index", type=int, default=0,
                   help="which loc_bbox to insert into (default 0)")
    p.add_argument("--exemplar_index", type=int, default=0,
                   help="which exemplar / mask_NNN.png to use as reference (default 0)")
    p.add_argument("--ref_mask_source", choices=("mask", "bbox"), default="mask",
                   help="mask: use mask_NNN.png | bbox: use the exemplar rectangle")
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
    p.add_argument("--flux", default="/home/user03/aiotlab/hachi/Image_Editing/ckpt/black-forest-labs/FLUX.1-Fill-dev")
    p.add_argument("--redux", default="/home/user03/aiotlab/hachi/Image_Editing/ckpt/black-forest-labs/FLUX.1-Redux-dev")
    p.add_argument("--lora", default=os.path.join(
        REPO, "ckpt/Insert-Anything/20250321_steps5000_pytorch_lora_weights.safetensors"))

    args = p.parse_args()
    args.anno_dir = args.anno_dir or os.path.join(args.data_root, "Anno")
    args.image_dir = args.image_dir or os.path.join(args.data_root, "Image")
    args.mask_dir = args.mask_dir or os.path.join(args.data_root, "masks")
    return args


def collect(args):
    """The case list after listing, slicing and resume filtering."""
    for d in (args.anno_dir, args.image_dir, args.mask_dir):
        if not os.path.isdir(d):
            raise SystemExit(f"[error] no such directory: {d}")

    items, skipped = get_inputs(args.anno_dir, args.image_dir, args.mask_dir,
                                args.loc_index, args.exemplar_index)
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

    print(f"processed_dataset: usable={total} selected={len(items)} todo={len(kept)} "
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
        bar.set_description(case[:40])
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
                title=f"{case}   [{item['class_name']}]   seed {args.seed}",
            )

            with open(os.path.join(args.work_dir, "meta", f"{case}.json"), "w") as f:
                json.dump(dict(
                    case=case, seed=args.seed,
                    class_name=item["class_name"],
                    source_image=item["img_path"],
                    ref_mask_path=item["exam_mask_path"],
                    ref_mask_source=used,
                    loc_index=args.loc_index, exemplar_index=args.exemplar_index,
                    loc_bbox=item["loc_bbox"], exam_bbox=item["exam_bbox"],
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
