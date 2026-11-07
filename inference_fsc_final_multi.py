"""Run Insert-Anything over FSC_final inserting N objects in ONE forward pass.

Multi-object variant of inference_fsc_final.py.  The single `loc_bbox` becomes
`n` location boxes that are all filled into one target mask, so FLUX.1-Fill sees
`n` holes and a single Redux reference, and fills all of them in one call.

    source_image : OBJ_INS_FSC/{test,val}/<case>/inpainted_turn_<turn>.png
    mask_image   : union of the filled rectangles of
                   annotation["inpainted_bboxes"][0:n]            (the n loc bboxes)
    ref_image    : the same inpainted_turn_<turn>.png -- the exemplar is another
                   instance still present in that image
    ref_mask     : <case>/mask_<turn+1>.png, i.e. the mask of
                   inpainted_bboxes[turn], the first instance NOT yet removed
                   (--ref_mask_source bbox uses that bbox's plain rectangle)

`inpainted_turn_T.png` has exactly `inpainted_bboxes[0:T]` removed, so `--turn`
defaults to `--n`: that is the smallest turn for which all n location boxes are
genuinely empty holes.  `inpainted_bboxes[turn]` is then the first instance still
visible, which is why the reference index follows the turn rather than being
pinned to 1.  At `--n 1` this reduces to inference_fsc_final.py exactly
(turn 1, loc = bbox[0], exemplar = bbox[1] / mask_2.png, 1439 usable cases);
`--n 3` leaves 1194.

Cases with len(inpainted_bboxes) <= turn are skipped, exactly as upstream.

The Insert-Anything half is still a line-for-line copy of inference.py.  The only
structural change is in `build_target`: the zoom-in crop is taken around the
union of the n boxes (which `get_bbox_from_mask` on the combined mask already
yields) and is explicitly held to contain that union, while the crop expansion,
square padding, resize and paste-back are untouched.  Per case four images are
written:

    grid_image/        labelled visualisation of the whole pass
    input_model_image/ right panel of the diptych handed to the model  (768x768)
    output_model_image/right panel of the model's raw output           (768x768)
    output_full_image/ final result pasted back into the full source image

plus the two masks needed to score background preservation on those panels
(mask/ for the full image, input_model_mask/ for the 768x768 panels; both are the
union of all n targets) and a meta/<case>.json describing the crop geometry, the
n boxes and their coordinates inside the 768x768 panel.
"""

import argparse
import json
import math
import os
import traceback

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from utils.utils import (
    box2squre,
    crop_back,
    expand_bbox,
    expand_image_mask,
    get_bbox_from_mask,
    pad_to_square,
)

REPO = os.path.dirname(os.path.abspath(__file__))

SPLITS = [
    "FSC_difficult_no_overlap",
    "FSC_difficult_overlap",
    "FSC_easy_no_overlap",
    "FSC_easy_overlap",
]

# the four required outputs + the masks that make the panels scorable
OUT_DIRS = [
    "grid_image",
    "input_model_image",
    "output_model_image",
    "output_full_image",
    "mask",
    "input_model_mask",
    "meta",
]
REQUIRED = OUT_DIRS[:4]


# ------------------------------------------------------- dataset / inputs --
# Ported from UniCombine_Ins_Rm/inference_fsc_final.py:50-88 and 216-237,
# generalised from one loc bbox to n.

def create_mask_from_bboxes(image_size, bboxes):
    """Create one mask image from the union of `bboxes` (each xyxy). PIL 'L'."""
    width, height = image_size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for x1, y1, x2, y2 in bboxes:
        draw.rectangle([x1, y1, x2, y2], fill=255)
    return mask


def get_inputs(test_dir, turn, n, root_path):
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

        # get img_path
        img_path = os.path.join(img_folder_path, f"inpainted_turn_{turn}.png")

        # get prompt, bbox
        anno_path = os.path.join(img_folder_path, "annotation.json")
        if not (os.path.exists(img_path) and os.path.exists(anno_path)):
            skipped["missing_file"] += 1
            continue
        with open(anno_path, "r") as f:
            anno = json.load(f)

        # only keep cases that still have an exemplar instance left in the image
        if len(anno["inpainted_bboxes"]) <= turn:
            skipped["too_few_bboxes"] += 1
            continue

        # bboxes[0:n] are holes in inpainted_turn_<turn>.png (n <= turn), and
        # bboxes[turn] is the first instance that is still there -> exemplar.
        loc_bboxes = anno["inpainted_bboxes"][:n]
        exam_bbox = anno["inpainted_bboxes"][turn]
        exam_mask_path = os.path.join(img_folder_path, f"mask_{turn + 1}.png")
        class_name = anno["class_based_caption"]

        inputs.append(dict(
            case=img_folder,
            class_name=class_name,
            img_path=img_path,
            loc_bboxes=loc_bboxes,
            exam_bbox=exam_bbox,
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

    # mask_image: all n location bboxes at once, unchanged otherwise
    tar_mask_pil = create_mask_from_bboxes((w, h), item["loc_bboxes"])
    tar_mask = (np.array(tar_mask_pil) > 128).astype(np.uint8)
    tar_mask = cv2.resize(tar_mask, (tar_image.shape[1], tar_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # ref_mask: the exemplar's segmentation mask, or its bbox rectangle.
    # One reference object conditions every one of the n insertions.
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
        ref_mask = (np.array(create_mask_from_bboxes((w, h), [item["exam_bbox"]])) > 128).astype(np.uint8)
        if ref_mask.sum() < 10:
            raise RuntimeError("empty reference mask")

    if tar_mask.sum() < 10:
        raise RuntimeError("empty source mask")

    return ref_image, ref_mask, tar_image, tar_mask, tar_mask_pil, used


# ------------------------------------------------- Insert-Anything stages --
# Verbatim from inference.py; only split into functions so the intermediates
# can be saved, plus the union handling in build_target.

def build_ref_tile(ref_image, ref_mask, size):
    """inference.py:53-64 + 89 -- the 768x768 tile that goes into Redux."""
    # Remove the background information of the reference picture
    ref_box_yyxx = get_bbox_from_mask(ref_mask)
    ref_mask_3 = np.stack([ref_mask, ref_mask, ref_mask], -1)
    masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1 - ref_mask_3)

    # Extract the box where the reference image is located, and place the
    # reference object at the center of the image
    y1, y2, x1, x2 = ref_box_yyxx
    masked_ref_image = masked_ref_image[y1:y2, x1:x2, :]
    ref_mask = ref_mask[y1:y2, x1:x2]
    ratio = 1.3
    masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)
    masked_ref_image = pad_to_square(masked_ref_image, pad_value=255, random=False)

    return cv2.resize(masked_ref_image.astype(np.uint8), size).astype(np.uint8)


def union_yyxx(bboxes_xyxy, shape):
    """Enclosing yyxx box of the n xyxy location boxes, clipped to the image."""
    H, W = shape[0], shape[1]
    x1 = min(b[0] for b in bboxes_xyxy)
    y1 = min(b[1] for b in bboxes_xyxy)
    x2 = max(b[2] for b in bboxes_xyxy)
    y2 = max(b[3] for b in bboxes_xyxy)
    return (max(0, int(y1)), min(H, int(y2)), max(0, int(x1)), min(W, int(x2)))


def grow_to_contain(box, must, shape):
    """Grow yyxx `box` just enough to cover yyxx `must`, staying in the image."""
    H, W = shape[0], shape[1]
    y1, y2, x1, x2 = box
    my1, my2, mx1, mx2 = must
    return (max(0, min(y1, my1)), min(H, max(y2, my2)),
            max(0, min(x1, mx1)), min(W, max(x2, mx2)))


def build_target(tar_image, tar_mask, loc_bboxes, size):
    """inference.py:66-95 -- the zoomed-in target tile plus the crop geometry.

    `tar_mask` already holds all n boxes, so `get_bbox_from_mask` returns their
    enclosing box and the upstream zoom-in works on the union as-is: both
    `expand_bbox` calls only ever grow a box (its internal ratio is
    sqrt(f(r)/r) >= 1) and `box2squre` only grows the shorter side, so the union
    stays inside the crop.  `grow_to_contain` makes that guarantee explicit
    rather than implicit -- in practice it is a no-op, since the dilated mask's
    bbox is already a superset of the raw union.
    """
    # Dilate the mask
    kernel = np.ones((7, 7), np.uint8)
    iterations = 2
    tar_mask = cv2.dilate(tar_mask, kernel, iterations=iterations)

    # the n boxes we must not crop away
    must = union_yyxx(loc_bboxes, tar_image.shape)

    # zome in -- now around the union of the n boxes
    tar_box_yyxx = get_bbox_from_mask(tar_mask)
    tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=1.2)

    tar_box_yyxx_crop = expand_bbox(tar_image, tar_box_yyxx, ratio=2)
    tar_box_yyxx_crop = grow_to_contain(tar_box_yyxx_crop, must, tar_image.shape)
    tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop)
    y1, y2, x1, x2 = tar_box_yyxx_crop

    old_tar_image = tar_image.copy()
    tar_image = tar_image[y1:y2, x1:x2, :]
    tar_mask = tar_mask[y1:y2, x1:x2]

    H1, W1 = tar_image.shape[0], tar_image.shape[1]

    tar_mask = pad_to_square(tar_mask, pad_value=0)
    tar_mask = cv2.resize(tar_mask, size, interpolation=cv2.INTER_NEAREST)

    tar_image = pad_to_square(tar_image, pad_value=255)
    H2, W2 = tar_image.shape[0], tar_image.shape[1]

    tar_image = cv2.resize(tar_image, size)

    return dict(tile=tar_image, mask=tar_mask, old_tar_image=old_tar_image,
                H1=H1, W1=W1, H2=H2, W2=W2, box=tar_box_yyxx_crop, union=must)


def map_box_to_panel(box_xyxy, ctx, size):
    """Where an xyxy source box lands in the size x size right panel.

    Inverts the crop -> pad_to_square -> resize chain of `build_target`, so the
    n targets can be outlined on the panel and stored per-object in the meta.
    """
    y1c, _, x1c, _ = ctx["box"]
    H1, W1, H2, W2 = ctx["H1"], ctx["W1"], ctx["H2"], ctx["W2"]

    # pad_to_square pads the shorter side by int(|H1-W1| / 2) at its start
    padd_1 = int(abs(H1 - W1) / 2)
    pad_x = padd_1 if H1 > W1 else 0
    pad_y = padd_1 if W1 > H1 else 0

    sx, sy = size[0] / W2, size[1] / H2
    x1, y1, x2, y2 = box_xyxy
    out = [(x1 - x1c + pad_x) * sx, (y1 - y1c + pad_y) * sy,
           (x2 - x1c + pad_x) * sx, (y2 - y1c + pad_y) * sy]
    return [float(np.clip(out[0], 0, size[0])), float(np.clip(out[1], 0, size[1])),
            float(np.clip(out[2], 0, size[0])), float(np.clip(out[3], 0, size[1]))]


def build_diptych(ref_tile, ctx):
    """inference.py:96-104 -- the `image=` and `mask_image=` arguments.

    The right half of `mask_diptych` carries all n holes, which is what makes the
    n insertions happen in a single pipe() call.
    """
    diptych_ref_tar = np.concatenate([ref_tile, ctx["tile"]], axis=1)

    tar_mask = np.stack([ctx["mask"], ctx["mask"], ctx["mask"]], -1)
    mask_black = np.ones_like(ctx["tile"]) * 0
    mask_diptych = np.concatenate([mask_black, tar_mask], axis=1)

    diptych_ref_tar = Image.fromarray(diptych_ref_tar)
    mask_diptych[mask_diptych == 1] = 255
    mask_diptych = Image.fromarray(mask_diptych)
    return diptych_ref_tar, mask_diptych


def split_output(edited_image, ctx):
    """inference.py:121-130 -- right panel, then paste back into the full image."""
    width, height = edited_image.size
    left = width // 2
    right = width
    top = 0
    bottom = height
    edited_image = edited_image.crop((left, top, right, bottom))

    right_panel = np.array(edited_image)
    final = crop_back(right_panel.copy(), ctx["old_tar_image"].copy(),
                      np.array([ctx["H1"], ctx["W1"], ctx["H2"], ctx["W2"]]),
                      np.array(ctx["box"]))
    return Image.fromarray(right_panel), Image.fromarray(final)


# ----------------------------------------------------------------- grid --

BG = (250, 250, 251)
CELL_BG = (214, 216, 222)
BORDER = (120, 124, 134)
FG = (24, 26, 32)
MUTED = (110, 114, 124)
GOLD = (255, 215, 0)

FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)


def _font(size, bold=False):
    paths = FONT_PATHS if bold else FONT_PATHS[1:] + FONT_PATHS[:1]
    for p in paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                pass
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def _to_rgb(arr):
    if isinstance(arr, Image.Image):
        arr = np.array(arr.convert("RGB"))
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        if a.max() <= 1:
            a = a * 255
        a = np.stack([a] * 3, -1)
    elif a.ndim == 3 and a.shape[2] == 4:
        a = a[:, :, :3]
    return np.ascontiguousarray(a)


def make_grid(stages, path, cols=5, cell=340, title=None):
    """stages: list of (label, array/PIL). Renders a labelled grid and saves it."""
    n = len(stages)
    cols = max(1, min(cols, n))
    rows = math.ceil(n / cols)
    gap, margin, cap_h = 18, 24, 44
    head_h = 62 if title else 0

    W = margin * 2 + cols * cell + (cols - 1) * gap
    H = margin * 2 + head_h + rows * (cell + cap_h) + (rows - 1) * gap

    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    f_title, f_lab, f_dim = _font(26, True), _font(15, True), _font(14)

    if title:
        d.text((margin, margin + 6), title, font=f_title, fill=FG)
        d.line([margin, margin + head_h - 12, W - margin, margin + head_h - 12],
               fill=(220, 222, 228), width=1)

    for i, (label, arr) in enumerate(stages):
        img = _to_rgb(arr)
        h, w = img.shape[:2]
        s = cell / max(h, w)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST
        thumb = Image.fromarray(cv2.resize(img, (nw, nh), interpolation=interp))

        r, c = divmod(i, cols)
        cx = margin + c * (cell + gap)
        cy = margin + head_h + r * (cell + cap_h + gap)

        d.rectangle([cx, cy, cx + cell - 1, cy + cell - 1], fill=CELL_BG)
        ox, oy = cx + (cell - nw) // 2, cy + (cell - nh) // 2
        canvas.paste(thumb, (ox, oy))
        d.rectangle([ox - 1, oy - 1, ox + nw, oy + nh], outline=BORDER, width=1)

        d.text((cx, cy + cell + 6), f"{i + 1:02d}  {label}", font=f_lab, fill=FG)
        d.text((cx, cy + cell + 25), f"{w} x {h}", font=f_dim, fill=MUTED)

    canvas.save(path)


def _outline_boxes(img, boxes_xyxy, color=GOLD, label=True):
    """Draw each of the n xyxy boxes onto a copy of img, numbered 1..n."""
    out = _to_rgb(img).copy()
    t = max(2, round(max(out.shape[:2]) / 320))
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        p1, p2 = (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))
        cv2.rectangle(out, p1, p2, color, t)
        if label:
            cv2.putText(out, str(i + 1), (p1[0] + 2 * t, p1[1] + 9 * t),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35 * t, color, max(1, t - 1),
                        cv2.LINE_AA)
    return out


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
    p.add_argument("--work_dir", default=None,
                   help="default: .../FSC_final/InsertAnything_multi<n>/Turn_<turn>")
    p.add_argument("--n", type=int, default=3,
                   help="objects inserted simultaneously in one pass (loc bboxes [0:n])")
    p.add_argument("--turn", type=int, default=None,
                   help="inpainted_turn_<turn>.png to insert into (default: --n, the "
                        "smallest turn in which all n loc bboxes are empty holes)")
    p.add_argument("--ref_mask_source", choices=("mask", "bbox"), default="mask",
                   help="mask: use mask_<turn+1>.png | bbox: use the exam_bbox rectangle")
    p.add_argument("--seed", type=int, default=666)
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
    p.add_argument("--flux", default="/mnt/disk1/aiotlab/hachi/checkpoints/FLUX/FLUX.1-Fill-dev")
    p.add_argument("--redux", default="/mnt/disk1/aiotlab/hachi/checkpoints/FLUX/FLUX.1-Redux-dev")
    p.add_argument("--lora", default=os.path.join(
        REPO, "ckpt/Insert-Anything/20250321_steps5000_pytorch_lora_weights.safetensors"))

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
                         f"InsertAnything_multi{args.n}/Turn_{args.turn}")
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
        items, skipped = get_inputs(test_dir, args.turn, args.n, args.obj_root)
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

    print(f"inserting n={args.n} objects per pass into inpainted_turn_{args.turn}.png, "
          f"exemplar = inpainted_bboxes[{args.turn}] / mask_{args.turn + 1}.png")

    plan = collect(args)
    todo = sum(len(v) for _, v in plan)
    print(f"\ntotal cases to run: {todo}")
    if args.dry_run or todo == 0:
        return

    import torch
    from diffusers import FluxFillPipeline, FluxPriorReduxPipeline

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Load the pre-trained model and LoRA weights (inference.py:15-33).
    # Loaded once here and reused for every split/case in the sweep.
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
                ref_image, ref_mask, tar_image, tar_mask, tar_mask_pil, used = load_case(
                    item, args.ref_mask_source)

                ref_tile = build_ref_tile(ref_image, ref_mask, size)
                ctx = build_target(tar_image, tar_mask, item["loc_bboxes"], size)
                diptych_ref_tar, mask_diptych = build_diptych(ref_tile, ctx)

                # the n targets in 768x768 panel coordinates
                panel_boxes = [map_box_to_panel(b, ctx, size) for b in item["loc_bboxes"]]

                # Extract the features of the reference image (inference.py:90).
                # One Redux embedding, shared by all n insertions.
                pipe_prior_output = redux(Image.fromarray(ref_tile))
                if args.offload != "none":
                    pipe_prior_output = {k: v.to(device) for k, v in pipe_prior_output.items()}

                # all n holes are filled in this single call
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

                in_panel.save(os.path.join(out_root, "input_model_image", f"{case}.png"))
                out_panel.save(os.path.join(out_root, "output_model_image", f"{case}.png"))
                out_full.save(os.path.join(out_root, "output_full_image", f"{case}.png"))
                in_panel_mask.save(os.path.join(out_root, "input_model_mask", f"{case}.png"))
                tar_mask_pil.save(os.path.join(out_root, "mask", f"{case}.png"))

                make_grid(
                    [
                        (f"source + {args.n} loc bboxes",
                         _outline_boxes(tar_image, item["loc_bboxes"])),
                        (f"ref tile -> Redux ({used})", ref_tile),
                        (f"model input ({args.n} holes)",
                         _outline_boxes(in_panel, panel_boxes)),
                        ("model output (right panel)", out_panel),
                        ("final (pasted back)", out_full),
                    ],
                    os.path.join(out_root, "grid_image", f"{case}.png"),
                    cols=5, cell=args.cell,
                    title=(f"{split} / {case}   [{item['class_name']}]   "
                           f"n={args.n} turn={args.turn} seed {args.seed}"),
                )

                with open(os.path.join(out_root, "meta", f"{case}.json"), "w") as f:
                    json.dump(dict(
                        case=case, split=split, n=args.n, turn=args.turn, seed=args.seed,
                        class_name=item["class_name"],
                        source_image=item["img_path"],
                        ref_mask_path=item["exam_mask_path"],
                        ref_mask_source=used,
                        exam_bbox_index=args.turn,
                        loc_bboxes=item["loc_bboxes"],
                        loc_bboxes_panel=panel_boxes,
                        loc_union_yyxx=[int(v) for v in ctx["union"]],
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
                bar.write(f"[fail] {split}/{case}: {type(e).__name__}: {e}")
                bar.write(traceback.format_exc().rstrip())

            bar.set_postfix(ok=done, fail=failed)
            bar.update(1)

    bar.close()
    print(f"\ndone={done} failed={failed} -> {args.work_dir}")


if __name__ == "__main__":
    main()
