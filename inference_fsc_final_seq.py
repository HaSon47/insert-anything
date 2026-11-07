"""Run Insert-Anything over FSC_final inserting N objects SEQUENTIALLY, one per pass.

Sequential counterpart of inference_fsc_final_multi.py.  Instead of filling all
`n` location boxes in a single FLUX.1-Fill call, the n objects are inserted one at
a time in n calls, pass k filling hole inpainted_bboxes[k-1] in the output of
pass k-1.  `--mode` picks what is carried from one pass to the next:

  full_image : the full image.  Pass k's panel is pasted back with crop_back and
               pass k+1 re-crops, pads and resizes it:
                 pass 1 : inpainted_turn_<turn>.png + hole bboxes[0]   -> out_1
                 pass 2 : out_1                     + hole bboxes[1]   -> out_2
                 ...
                 pass n : out_{n-1}                 + hole bboxes[n-1] -> final
  panel      : the 768x768 output panel, as-is.  No resize or paste-back between
               passes; crop_back runs once, after pass n.  Comparing the two modes
               isolates the background loss of the intermediate round trips.

Everything else is held identical to the simultaneous run so the two are directly
comparable -- the stages below are imported from inference_fsc_final_multi.py,
not copied:

  * inputs    : get_inputs / load_case, so the same cases, the same initial image,
                the same n target boxes and the same reference object
                (exemplar = inpainted_bboxes[turn] / mask_<turn+1>.png).
  * reference : the ref tile and its Redux embedding are computed once from the
                initial image and shared by all n passes.
  * crop      : the zoom-in crop is computed ONCE, by the multi script's own
                build_target on the union of all n boxes, and that crop box (and
                its pad / resize geometry) is reused by every pass; only the hole
                changes.  Pass 1's input panel is therefore pixel-identical to the
                multi run's input_model_image, and every panel of every pass covers
                the same image region at the same scale.
  * settings  : same FLUX.1-Fill-dev + LoRA + Redux, tile size, steps and seed
                (each pass re-seeds its generator with --seed, as the single
                multi pass does).

In full_image mode each pass is the unchanged Insert-Anything pass (diptych ->
pipe() -> right panel -> crop_back into the full image); the only new stage is
`build_target_fixed`, which is build_target minus the crop-box derivation.  Panel
mode differs only in that pass k>1 takes the previous output panel as its tile,
and passes before the last stop at the right panel (`right_panel`).

Output layout per split mirrors inference_fsc_final_multi.py.  The top-level
directories hold the FINAL result after all n insertions, so the same scoring
code runs on both work dirs:

    grid_image/          labelled visualisation of all n passes
    input_model_image/   pass-1 input panel (== the multi run's)          (768x768)
    output_model_image/  pass-n output panel                              (768x768)
    output_full_image/   full image after all n insertions
    mask/, input_model_mask/   union of all n targets (full image / panel)
    meta/<case>.json     crop geometry, the n boxes, per-pass info

and the intermediate result of every pass k = 1..n under step_<k>/:

    step_<k>/input_model_image/   pass-k input panel (objects 1..k-1 already in)
    step_<k>/output_model_image/  pass-k output panel
    step_<k>/output_full_image/   full image after k insertions (panel mode:
                                  pass n only, there is no full image before)
    step_<k>/mask/, step_<k>/input_model_mask/   target k only

The two modes default to separate work dirs (InsertAnything_seq<n>_<mode>).
"""

import argparse
import json
import os
import traceback

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from inference_fsc_final_multi import (
    OUT_DIRS,
    REPO,
    SPLITS,
    _outline_boxes,
    build_diptych,
    build_ref_tile,
    build_target,
    collect,
    create_mask_from_bboxes,
    load_case,
    make_grid,
    map_box_to_panel,
    split_output,
)
from utils.utils import pad_to_square

# per-pass intermediates, same names as the top-level dirs they mirror
STEP_DIRS = [
    "input_model_image",
    "output_model_image",
    "output_full_image",
    "mask",
    "input_model_mask",
]


# ------------------------------------------------- Insert-Anything stages --

def build_target_fixed(tar_image, tar_mask, crop, size):
    """build_target (inference.py:66-95) with the zoom-in crop held fixed.

    `crop` is the ctx build_target returned for the union of all n boxes.  Its
    crop box is reused as-is instead of being re-derived from `tar_mask`, so
    every pass is cropped, padded and resized exactly like the simultaneous
    pass; the remaining lines are build_target's own.
    """
    # Dilate the mask
    kernel = np.ones((7, 7), np.uint8)
    iterations = 2
    tar_mask = cv2.dilate(tar_mask, kernel, iterations=iterations)

    # zome in -- with the crop shared by all n passes
    tar_box_yyxx_crop = crop["box"]
    y1, y2, x1, x2 = tar_box_yyxx_crop

    old_tar_image = tar_image.copy()
    tar_image = tar_image[y1:y2, x1:x2, :]
    tar_mask = tar_mask[y1:y2, x1:x2]

    H1, W1 = tar_image.shape[0], tar_image.shape[1]

    tar_mask = pad_to_square(tar_mask, pad_value=0)
    tar_mask = cv2.resize(tar_mask, size)

    tar_image = pad_to_square(tar_image, pad_value=255)
    H2, W2 = tar_image.shape[0], tar_image.shape[1]

    tar_image = cv2.resize(tar_image, size)

    return dict(tile=tar_image, mask=tar_mask, old_tar_image=old_tar_image,
                H1=H1, W1=W1, H2=H2, W2=W2, box=tar_box_yyxx_crop, union=crop["union"])


def right_panel(edited_image):
    """split_output minus the paste-back: the right (target) panel, untouched."""
    width, height = edited_image.size
    return edited_image.crop((width // 2, 0, width, height))


def _panel_mask(ctx):
    return Image.fromarray((ctx["mask"] > 0).astype(np.uint8) * 255)


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
    p.add_argument("--mode", choices=("full_image", "panel"), default="full_image",
                   help="full_image: paste each pass back into the full image and "
                        "re-crop it for the next | panel: feed each 768x768 output "
                        "panel to the next pass as-is, crop_back once at the end")
    p.add_argument("--work_dir", default=None,
                   help="default: .../FSC_final/InsertAnything_seq<n>_<mode>/Turn_<turn>")
    p.add_argument("--n", type=int, default=3,
                   help="objects inserted one per pass, in order (loc bboxes [0:n])")
    p.add_argument("--turn", type=int, default=None,
                   help="inpainted_turn_<turn>.png to insert into (default: --n, the "
                        "smallest turn in which all n loc bboxes are empty holes)")
    p.add_argument("--ref_mask_source", choices=("mask", "bbox"), default="mask",
                   help="mask: use mask_<turn+1>.png | bbox: use the exam_bbox rectangle")
    p.add_argument("--seed", type=int, default=666,
                   help="generator seed, re-applied at the start of every pass")
    p.add_argument("--size", type=int, default=768, help="tile size, inference.py uses 768")
    p.add_argument("--steps", type=int, default=None,
                   help="num_inference_steps (default: the pipeline's own, 50)")
    p.add_argument("--cell", type=int, default=340, help="grid thumbnail size in px")
    p.add_argument("--limit", type=int, default=None, help="first N cases per split")
    p.add_argument("--start", type=int, default=0, help="skip the first N cases per split")
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
    p.add_argument("--lora", default="/home/user03/aiotlab/hachi/Image_Editing/ckpt/Insert-Anything/20250321_steps5000_pytorch_lora_weights.safetensors")

    args = p.parse_args()
    if args.n < 1:
        p.error("--n must be >= 1")
    if args.turn is None:
        args.turn = args.n
    if args.n > args.turn:
        p.error(f"--n {args.n} > --turn {args.turn}: inpainted_turn_{args.turn}.png only "
                f"has bboxes[0:{args.turn}] removed, so not every loc bbox is empty")
    if args.work_dir is None:
        args.work_dir = (f"/mnt/disk1/aiotlab/hachi/Output/FSC_final/"
                         f"InsertAnything_seq{args.n}_{args.mode}/Turn_{args.turn}")
    return args


def main():
    args = parse_args()
    size = (args.size, args.size)

    print(f"inserting n={args.n} objects one per pass ({args.mode} mode) into "
          f"inpainted_turn_{args.turn}.png, "
          f"exemplar = inpainted_bboxes[{args.turn}] / mask_{args.turn + 1}.png")

    # same case listing / resume rule as the multi run: a case is done once its
    # top-level (= final) outputs exist, and those are written last
    plan = collect(args)
    todo = sum(len(v) for _, v in plan)
    print(f"\ntotal cases to run: {todo}  ({todo * args.n} inference passes)")
    if args.dry_run or todo == 0:
        return

    import torch
    from diffusers import FluxFillPipeline, FluxPriorReduxPipeline

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Load the pre-trained model and LoRA weights (inference.py:15-33).
    # Loaded once here and reused for every split/case/pass in the sweep.
    print("loading FLUX.1-Fill-dev ...")
    pipe = FluxFillPipeline.from_pretrained(args.flux, torch_dtype=dtype)
    pipe.load_lora_weights(args.lora)
    print("loading FLUX.1-Redux-dev ...")
    redux = FluxPriorReduxPipeline.from_pretrained(args.redux).to(dtype=dtype)

    # the per-call denoising bars would drown out the sweep bar below
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

    # one bar over every case of every split, in place of the per-call denoising bars
    bar = tqdm(total=todo, unit="case", dynamic_ncols=True, smoothing=0.05)

    for split, items in plan:
        out_root = os.path.join(args.work_dir, split)
        for d in OUT_DIRS:
            os.makedirs(os.path.join(out_root, d), exist_ok=True)
        for k in range(1, args.n + 1):
            for d in STEP_DIRS:
                # panel mode has no full image before the final crop_back
                if d == "output_full_image" and args.mode == "panel" and k < args.n:
                    continue
                os.makedirs(os.path.join(out_root, f"step_{k}", d), exist_ok=True)

        for item in items:
            case = item["case"]
            bar.set_description(f"{split}/{case}")
            try:
                # tar_mask / tar_mask_pil are the union of all n boxes: they fix
                # the crop and are the final masks, exactly as in the multi run
                ref_image, ref_mask, tar_image, tar_mask, tar_mask_pil, used = load_case(
                    item, args.ref_mask_source)
                h, w = tar_image.shape[:2]

                ref_tile = build_ref_tile(ref_image, ref_mask, size)

                # the one crop shared by all n passes -- the multi run's crop
                crop = build_target(tar_image, tar_mask, item["loc_bboxes"], size)

                # the n targets in 768x768 panel coordinates (same crop for all)
                panel_boxes = [map_box_to_panel(b, crop, size) for b in item["loc_bboxes"]]

                # Extract the features of the reference image (inference.py:90).
                # One Redux embedding, shared by all n passes.
                pipe_prior_output = redux(Image.fromarray(ref_tile))
                if args.offload != "none":
                    pipe_prior_output = {k: v.to(device) for k, v in pipe_prior_output.items()}

                # insert object k into the output of pass k-1: the full image
                # (full_image mode) or the 768x768 panel (panel mode)
                cur_image, cur_panel = tar_image, None
                passes = []
                for k, loc_bbox in enumerate(item["loc_bboxes"], start=1):
                    step_mask_pil = create_mask_from_bboxes((w, h), [loc_bbox])
                    step_mask = (np.array(step_mask_pil) > 128).astype(np.uint8)

                    ctx = build_target_fixed(cur_image, step_mask, crop, size)
                    if cur_panel is not None:
                        # panel mode: keep hole k and the crop geometry, but the
                        # tile is the previous output panel as-is (cur_image stays
                        # the initial image, so the final crop_back pastes into it)
                        ctx["tile"] = cur_panel
                    diptych_ref_tar, mask_diptych = build_diptych(ref_tile, ctx)

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

                    if args.mode == "full_image" or k == args.n:
                        out_panel, out_full = split_output(edited_image, ctx)
                    else:
                        # panel mode, not the last pass: no paste-back
                        out_panel, out_full = right_panel(edited_image), None
                    in_panel = Image.fromarray(ctx["tile"])
                    in_mask = _panel_mask(ctx)

                    step_root = os.path.join(out_root, f"step_{k}")
                    in_panel.save(os.path.join(step_root, "input_model_image", f"{case}.png"))
                    out_panel.save(os.path.join(step_root, "output_model_image", f"{case}.png"))
                    if out_full is not None:
                        out_full.save(os.path.join(step_root, "output_full_image", f"{case}.png"))
                    in_mask.save(os.path.join(step_root, "input_model_mask", f"{case}.png"))
                    step_mask_pil.save(os.path.join(step_root, "mask", f"{case}.png"))

                    passes.append(dict(in_panel=in_panel, out_panel=out_panel,
                                       out_full=out_full, in_mask=in_mask))
                    if args.mode == "full_image":
                        cur_image = np.array(out_full)
                    else:
                        cur_panel = np.array(out_panel)

                # final result = pass n, written last so an interrupted case is redone
                final_panel, final_full = passes[-1]["out_panel"], passes[-1]["out_full"]
                Image.fromarray(crop["tile"]).save(
                    os.path.join(out_root, "input_model_image", f"{case}.png"))
                final_panel.save(os.path.join(out_root, "output_model_image", f"{case}.png"))
                final_full.save(os.path.join(out_root, "output_full_image", f"{case}.png"))
                _panel_mask(crop).save(os.path.join(out_root, "input_model_mask", f"{case}.png"))
                tar_mask_pil.save(os.path.join(out_root, "mask", f"{case}.png"))

                # row 0: inputs + final; row k: pass k input / output / pasted
                # back (panel mode before pass n: the mask handed to the model)
                stages = [
                    (f"source + {args.n} loc bboxes",
                     _outline_boxes(tar_image, item["loc_bboxes"])),
                    (f"ref tile -> Redux ({used})", ref_tile),
                    (f"final after {args.n} passes", final_full),
                ]
                for k, ps in enumerate(passes, start=1):
                    stages += [
                        (f"pass {k} input (hole {k})",
                         _outline_boxes(ps["in_panel"], [panel_boxes[k - 1]], label=False)),
                        (f"pass {k} output (right panel)", ps["out_panel"]),
                        (f"pass {k} pasted back", ps["out_full"]) if ps["out_full"] is not None
                        else (f"pass {k} mask (no paste-back)", ps["in_mask"]),
                    ]
                make_grid(
                    stages,
                    os.path.join(out_root, "grid_image", f"{case}.png"),
                    cols=3, cell=args.cell,
                    title=(f"{split} / {case}   [{item['class_name']}]   "
                           f"n={args.n} sequential/{args.mode} turn={args.turn} "
                           f"seed {args.seed}"),
                )

                # what pass k hands to pass k+1
                chain_dir = ("output_full_image" if args.mode == "full_image"
                             else "output_model_image")
                with open(os.path.join(out_root, "meta", f"{case}.json"), "w") as f:
                    json.dump(dict(
                        case=case, split=split, mode="sequential", seq_mode=args.mode,
                        n=args.n, turn=args.turn, seed=args.seed,
                        class_name=item["class_name"],
                        source_image=item["img_path"],
                        ref_mask_path=item["exam_mask_path"],
                        ref_mask_source=used,
                        exam_bbox_index=args.turn,
                        loc_bboxes=item["loc_bboxes"],
                        loc_bboxes_panel=panel_boxes,
                        loc_union_yyxx=[int(v) for v in crop["union"]],
                        exam_bbox=item["exam_bbox"],
                        source_size=[int(w), int(h)],
                        tar_box_yyxx_crop=[int(v) for v in crop["box"]],
                        H1=int(crop["H1"]), W1=int(crop["W1"]),
                        H2=int(crop["H2"]), W2=int(crop["W2"]),
                        tile_size=args.size,
                        # pass k fills loc_bboxes[k-1] in the output of pass k-1,
                        # always with the crop above
                        passes=[dict(
                            step=k,
                            loc_bbox_index=k - 1,
                            loc_bbox=item["loc_bboxes"][k - 1],
                            loc_bbox_panel=panel_boxes[k - 1],
                            source=(item["img_path"] if k == 1 else
                                    f"step_{k - 1}/{chain_dir}/{case}.png"),
                            output=f"step_{k}/{chain_dir}/{case}.png",
                        ) for k in range(1, args.n + 1)],
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
