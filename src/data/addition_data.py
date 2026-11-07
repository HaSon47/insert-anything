"""Object-addition data: ObjectStitch_V2 + OBJ_INS_FSC -> reference diptychs.

See addition_training_plan.md, sections 1-2. A sample <scene>_<s> (s = file suffix) is

    input = ObjectStitch images/<scene>_<s>.png    objects 0 .. s-1 removed, object s still present
    GT    = ObjectStitch inpaint/<scene>_<s>.png   object s-1 present
    box   = inpainted_bboxes[s-1]                  where to insert (inclusive [x1, y1, x2, y2])
    ref   = input * mask_{s+1}, cropped with inpainted_bboxes[s]: another instance, on white

and becomes, on the GPU (make_addition_diptych):

    src    = [ref | input tile, masked pixels filled]   3 x S x 2S, in [0, 1]
    mask   = [zeros | white rectangle]                  3 x S x 2S, 0/1
    result = [ref | GT tile]                            3 x S x 2S, in [0, 1]

The input mask is always a filled rectangle. Object-shaped masks are only used to cut out the reference.

Debug / checks (run from the repo root):
    python -m src.data.addition_data --check
    python -m src.data.addition_data --dump 32 --split train --out <dir>
"""

import argparse
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from functools import lru_cache
from multiprocessing import Pool

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.utils import expand_image_mask, get_bbox_from_mask, pad_to_square

from .removal_data import (
    BalancedEpochSampler,
    build_tiles,
    inference_crop_box,
    inference_mask,
    load_config,
    place_square,
    read_mask,
    read_rgb,
    repo_path,
    sample_rng,
    to_tile,
)

cv2.setNumThreads(0)

# <image id>_b<branch>_<suffix>.png
PATTERN = re.compile(r"^((\d+)_b\d+)_(\d+)\.png$")
MASKED_FILL = {"gray": 0.5, "black": 0.0}
# Expected kept samples (total, suffix 1, suffix 2), measured for addition_training_plan.md section 1.3.
EXPECTED_COUNTS = {"train": (7363, 4011, 3352), "val": (1106, 615, 491)}


# ---------------------------------------------------------------- geometry

def box_mask(hw, box, scale=1.0):
    """Filled-rectangle mask (uint8, 0/1) of an inclusive [x1, y1, x2, y2] box, scaled about its centre and
    clipped to the image. scale=1 gives the pixels of create_mask_from_bbox (inference_fsc_final.py)."""
    H, W = hw
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w, half_h = (x2 - x1 + 1) * scale / 2, (y2 - y1 + 1) * scale / 2
    out = np.zeros((H, W), np.uint8)
    out[max(0, math.floor(cy - half_h + 0.5)):min(H, math.ceil(cy + half_h + 0.5)),
        max(0, math.floor(cx - half_w + 0.5)):min(W, math.ceil(cx + half_w + 0.5))] = 1
    return out


def is_solid_rectangle(mask):
    """True if the non-zero pixels are exactly one filled axis-aligned rectangle."""
    ys, xs = np.nonzero(mask)
    return len(xs) > 0 and (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1) == len(xs)


def build_ref_tile(image, ref_mask, ref_box, ratio, size):
    """Reference tile: background white, crop with the inclusive box, white margin of `ratio`, centred
    white square, resize. Same steps as build_ref_tile in inference_fsc_final.py, except that the crop is
    the annotated box instead of the mask's bounding box (addition_training_plan.md section 2)."""
    x1, y1, x2, y2 = ref_box
    masked = np.where(ref_mask[..., None] > 0, image, 255).astype(np.uint8)
    crop, crop_mask = masked[y1:y2 + 1, x1:x2 + 1], ref_mask[y1:y2 + 1, x1:x2 + 1]
    crop, _ = expand_image_mask(crop, crop_mask, ratio=ratio)
    crop = pad_to_square(crop, pad_value=255, random=False)
    return cv2.resize(crop.astype(np.uint8), (size, size)).astype(np.uint8)


def eval_tiles(image, gt, ref_mask, item, dcfg):
    """Deterministic tiles: the exact box, inference dilation and crop, reference margin ref_expand_eval.
    `valid` marks tile pixels that come from the image, i.e. not the padding of a crop at the image edge."""
    size = dcfg["size"]
    mask = inference_mask(box_mask(image.shape[:2], item["ins_box"]))
    tiles = build_tiles(image, gt, mask, inference_crop_box(image, mask), size)
    tiles["ref"] = build_ref_tile(image, ref_mask, item["ref_box"], dcfg["ref_expand_eval"], size)
    tiles["valid"] = to_tile(np.ones(mask.shape, np.uint8), tiles["box"], size, 0, cv2.INTER_NEAREST) > 0
    tiles["aug"] = "eval"
    return tiles


def train_tiles(image, gt, ref_mask, item, dcfg, rng, keep_full_mask=False):
    """Scaled box + inference dilation, jittered inference crop, random reference margin (plan section 2).
    The crop is based on the unscaled box, so the jitter is the same whatever the box scale drew."""
    size = dcfg["size"]
    H, W = image.shape[:2]
    y1, y2, x1, x2 = inference_crop_box(image, inference_mask(box_mask((H, W), item["ins_box"])))
    scale = float(rng.uniform(*dcfg["box_scale"]))
    mask = inference_mask(box_mask((H, W), item["ins_box"], scale))

    jcfg = dcfg["crop_jitter"]
    side = max(y2 - y1, x2 - x1) * rng.uniform(*jcfg["scale"])
    cy = (y1 + y2) / 2 + rng.uniform(-jcfg["shift"], jcfg["shift"]) * side
    cx = (x1 + x2) / 2 + rng.uniform(-jcfg["shift"], jcfg["shift"]) * side
    box = place_square(cy, cx, side, get_bbox_from_mask(mask), H, W)

    tiles = build_tiles(image, gt, mask, box, size)
    tiles["ref"] = build_ref_tile(image, ref_mask, item["ref_box"], float(rng.choice(dcfg["ref_expand_train"])), size)
    tiles["aug"] = f"{scale:.2f}"
    if keep_full_mask:
        tiles["mask_full"] = mask
    return tiles


def _chw(tile):
    return torch.from_numpy(np.ascontiguousarray(tile.transpose(2, 0, 1)))


def tiles_to_tensors(tiles):
    """Compact uint8 CHW tensors for the DataLoader; make_addition_diptych() expands them on the GPU."""
    return dict(
        ref_tile=_chw(tiles["ref"]),
        src_tile=_chw(tiles["src"]),
        gt_tile=_chw(tiles["gt"]),
        mask_tile=torch.from_numpy(tiles["mask"][None].copy()),
    )


def make_addition_diptych(ref_tile, src_tile, gt_tile, mask_tile, masked_fill="gray"):
    """Model inputs from (batched) uint8 tiles; works on any device.
    gray fill = 0.5, i.e. 0 after FLUX's [-1, 1] normalization, as FluxFillPipeline masks."""
    ref = ref_tile.float() / 255
    src = src_tile.float() / 255
    gt = gt_tile.float() / 255
    m = (mask_tile > 0).float().expand_as(src)
    masked = torch.where(m > 0, torch.full_like(src, MASKED_FILL[masked_fill]), src)
    return dict(
        src=torch.cat([ref, masked], dim=-1),
        mask=torch.cat([torch.zeros_like(m), m], dim=-1),
        result=torch.cat([ref, gt], dim=-1),
    )


# ---------------------------------------------------------------- dataset

def fsc_index(fsc_root, splits):
    """{scene: folder}. A scene in two searched splits would make the lookup ambiguous, so it is an error."""
    index = {}
    for split in splits:
        for scene in os.listdir(os.path.join(fsc_root, split)):
            if scene in index:
                raise ValueError(f"scene {scene} is in more than one of {splits}")
            index[scene] = os.path.join(fsc_root, split, scene)
    return index


@lru_cache(maxsize=None)
def scan_items(objectstitch_root, fsc_root, fsc_splits, suffixes, split):
    """Kept samples of one ObjectStitch split: suffix in `suffixes` and len(inpainted_bboxes) > suffix.
    Boxes are read here once, so workers never re-read annotations. Treat the result as read-only."""
    index = fsc_index(fsc_root, fsc_splits)
    img_dir, gt_dir = (os.path.join(objectstitch_root, d, split) for d in ("images", "inpaint"))
    gts = set(os.listdir(gt_dir))
    items = []
    for f in sorted(os.listdir(img_dir)):
        m = PATTERN.match(f)
        if m is None:
            raise ValueError(f"unexpected file name {img_dir}/{f}")
        scene, image_id, s = m.group(1), m.group(2), int(m.group(3))
        if s not in suffixes:
            continue
        if scene not in index:
            raise FileNotFoundError(f"{f}: scene {scene} is in none of {fsc_splits} under {fsc_root}")
        folder = index[scene]
        with open(os.path.join(folder, "annotation.json")) as fh:
            anno = json.load(fh)
        boxes = anno["inpainted_bboxes"]
        if len(boxes) <= s:
            continue
        ref_mask = os.path.join(folder, f"mask_{s + 1}.png")
        if f not in gts or not os.path.exists(ref_mask):
            raise FileNotFoundError(f"{f}: missing GT or {ref_mask}")
        items.append(dict(
            name=f[:-4],
            input=os.path.join(img_dir, f),
            gt=os.path.join(gt_dir, f),
            ref_mask=ref_mask,
            ins_box=[int(v) for v in boxes[s - 1]],
            ref_box=[int(v) for v in boxes[s]],
            scene=scene,
            image_id=image_id,
            suffix=s,
            class_name=anno.get("class_based_caption", "unknown"),
        ))
    return items


class AdditionDataset(Dataset):
    name = "objectstitch"

    def __init__(self, items, dcfg, train=True):
        self.items = items
        self.dcfg = dcfg
        self.train = train

    @classmethod
    def from_split(cls, dcfg, split, train=True):
        items = scan_items(dcfg["objectstitch_root"], dcfg["fsc_root"], tuple(dcfg["fsc_splits"]),
                           tuple(dcfg["suffixes"]), split)
        return cls(items, dcfg, train)

    def subset(self, names, train=False):
        by_name = {it["name"]: it for it in self.items}
        return type(self)([by_name[n] for n in names], self.dcfg, train)

    def __len__(self):
        return len(self.items)

    def load(self, idx):
        it = self.items[idx]
        image, gt = read_rgb(it["input"]), read_rgb(it["gt"])
        if gt.shape != image.shape:
            raise ValueError(f"GT {gt.shape} != input {image.shape}: {it['gt']}")
        return image, gt, read_mask(it["ref_mask"], image.shape[:2])

    def tiles(self, idx, rng=None):
        image, gt, ref_mask = self.load(idx)
        if self.train:
            return train_tiles(image, gt, ref_mask, self.items[idx], self.dcfg, rng)
        return eval_tiles(image, gt, ref_mask, self.items[idx], self.dcfg)

    def __getitem__(self, idx):
        tiles = self.tiles(idx, sample_rng())
        out = tiles_to_tensors(tiles)
        out.update(dataset=self.name, name=self.items[idx]["name"], aug=tiles["aug"])
        return out


def build_train_data(cfg):
    """(AdditionDataset of ObjectStitch train, sampler drawing samples_per_epoch per epoch)."""
    dcfg = cfg["data"]
    ds = AdditionDataset.from_split(dcfg, "train", train=True)
    return ds, BalancedEpochSampler([len(ds)], [dcfg["samples_per_epoch"]], seed=cfg["train"].get("seed", 0))


def pick_fixed(items, n, rng):
    """n names, balanced over suffixes, from distinct source images, round-robin over shuffled classes."""
    by_class = defaultdict(list)
    for it in items:
        by_class[it["class_name"]].append(it)
    classes = sorted(by_class)
    rng.shuffle(classes)
    picked, used, count = [], set(), Counter()
    while len(picked) < n:
        progress = False
        for cls in classes:
            if len(picked) == n:
                break
            suffix = min(sorted({it["suffix"] for it in items}), key=lambda s: count[s])
            free = sorted(it["name"] for it in by_class[cls] if it["suffix"] == suffix and it["image_id"] not in used)
            if not free:
                continue
            name = rng.choice(free)
            used.add(next(it["image_id"] for it in by_class[cls] if it["name"] == name))
            count[suffix] += 1
            picked.append(name)
            progress = True
        if not progress:
            raise ValueError(f"only {len(picked)} of {n} fixed samples could be picked")
    return sorted(picked)


def make_eval_subsets(cfg):
    """Fixed evaluation names: {"val": ObjectStitch val names, "train": ObjectStitch train names}."""
    dcfg, vcfg = cfg["data"], cfg["val"]
    rng = random.Random(vcfg.get("seed", 42))
    return dict(
        val=pick_fixed(AdditionDataset.from_split(dcfg, "val", train=False).items, vcfg["n_val"], rng),
        train=pick_fixed(AdditionDataset.from_split(dcfg, "train", train=False).items, vcfg["n_train"], rng),
    )


def build_eval_data(cfg, subsets):
    """Datasets of the fixed evaluation names (deterministic eval tiles); a group of size 0 is left out."""
    return {split: AdditionDataset.from_split(cfg["data"], split, train=False).subset(names)
            for split, names in subsets.items() if names}


# ---------------------------------------------------------------- debug

def _thumb(img, side):
    h, w = img.shape[:2]
    s = side / max(h, w)
    small = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    out = np.zeros((side, side, 3), np.uint8)
    out[:small.shape[0], :small.shape[1]] = small
    return out, s


def dump_row(ds, idx, rng, side=256):
    """[full input: insertion box red, reference box green, crop blue | ref tile | model input | mask | GT tile]."""
    image, gt, ref_mask = ds.load(idx)
    it = ds.items[idx]
    tiles = train_tiles(image, gt, ref_mask, it, ds.dcfg, rng) if ds.train else eval_tiles(image, gt, ref_mask, it, ds.dcfg)
    tensors = tiles_to_tensors(tiles)
    d = make_addition_diptych(tensors["ref_tile"], tensors["src_tile"], tensors["gt_tile"], tensors["mask_tile"],
                              ds.dcfg["masked_fill"])
    S = ds.dcfg["size"]
    assert torch.equal(d["src"][:, :, :S], d["result"][:, :, :S]) and torch.all(d["mask"][:, :, :S] == 0)
    full, s = _thumb(image, side)
    for box, color in ((it["ins_box"], (255, 0, 0)), (it["ref_box"], (0, 255, 0))):
        cv2.rectangle(full, (round(box[0] * s), round(box[1] * s)), (round(box[2] * s), round(box[3] * s)), color, 1)
    y1, y2, x1, x2 = tiles["box"]
    cv2.rectangle(full, (round(x1 * s), round(y1 * s)), (round(x2 * s) - 1, round(y2 * s) - 1), (0, 128, 255), 1)
    model_in = (d["src"][:, :, S:].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    cols = [full, tiles["ref"], model_in, np.repeat(tiles["mask"][..., None] * 255, 3, axis=2), tiles["gt"]]
    cols = [c if c.shape[0] == side else cv2.resize(c, (side, side), interpolation=cv2.INTER_AREA) for c in cols]
    row = np.concatenate(cols, axis=1)
    label = f"{it['name']} [{it['class_name']}] box_scale={tiles['aug']} crop={y2 - y1}x{x2 - x1}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(row, (0, side - th - 10), (tw + 10, side), (0, 0, 0), -1)
    cv2.putText(row, label, (5, side - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    return row


def run_dump(cfg, n, out, split, seed):
    """split "train": first samples of epoch 0, with augmentation; split "eval": the fixed evaluation set."""
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(seed)
    if split == "train":
        ds, sampler = build_train_data(cfg)
        picks = [(ds, i) for i in list(iter(sampler))[:n]]
    else:
        data = build_eval_data(cfg, make_eval_subsets(cfg))
        picks = [(ds, i) for ds in data.values() for i in range(len(ds))][:n]
    rows = [dump_row(ds, i, rng) for ds, i in picks]
    for g in range(0, len(rows), 8):
        path = os.path.join(out, f"{split}_dump_{g // 8:02d}.jpg")
        cv2.imwrite(path, cv2.cvtColor(np.concatenate(rows[g:g + 8], axis=0), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(path)


def _check_one(job):
    """Properties of one kept sample (real pixels). Returns (name, list of problems, x2-or-y2-on-the-edge)."""
    it, dcfg, seed = job
    problems = []
    image, gt = read_rgb(it["input"]), read_rgb(it["gt"])
    H, W = image.shape[:2]
    if gt.shape != image.shape:
        problems.append("input/GT shape")
    ref_mask = read_mask(it["ref_mask"], (H, W))
    edge = False
    for key in ("ins_box", "ref_box"):
        x1, y1, x2, y2 = it[key]
        if not (0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H):
            problems.append(f"{key} {it[key]} outside {W}x{H}")
        edge |= x2 == W or y2 == H
    x1, y1, x2, y2 = it["ref_box"]
    if ref_mask[y1:y2 + 1, x1:x2 + 1].sum() == 0:
        problems.append("reference mask empty inside its box")

    box = box_mask((H, W), it["ins_box"])
    t = eval_tiles(image, gt, ref_mask, it, dcfg)
    if not is_solid_rectangle(t["mask"]):
        problems.append("eval mask tile is not one solid rectangle")
    if t["ref"].shape != (dcfg["size"], dcfg["size"], 3) or t["ref"].min() == 255:
        problems.append("eval reference tile")
    if np.any((t["mask"] > 0) & ~t["valid"]):
        problems.append("eval mask reaches into padding")

    rng = np.random.default_rng(seed)
    for _ in range(3):
        t = train_tiles(image, gt, ref_mask, it, dcfg, rng, keep_full_mask=True)
        full, (y1, y2, x1, x2) = t["mask_full"], t["box"]
        if np.any(box & (1 - full)):
            problems.append("train mask misses part of the box")
        if full[y1:y2, x1:x2].sum() != full.sum():
            problems.append("train mask leaves the crop")
        if not is_solid_rectangle(t["mask"]):
            problems.append("train mask tile is not one solid rectangle")
    return it["name"], problems, edge


def check_eval_parity(ds, n):
    """Evaluation tile, mask and crop box must equal the inference path pixel for pixel:
    create_mask_from_bbox + build_target in inference_fsc_final.py."""
    from inference_fsc_final import build_target, create_mask_from_bbox
    size = ds.dcfg["size"]
    for i in range(min(n, len(ds))):
        image, gt, ref_mask = ds.load(i)
        it = ds.items[i]
        H, W = image.shape[:2]
        mine = eval_tiles(image, gt, ref_mask, it, ds.dcfg)
        pil = (np.array(create_mask_from_bbox((W, H), it["ins_box"])) > 128).astype(np.uint8)
        assert np.array_equal(pil, box_mask((H, W), it["ins_box"])), f"{it['name']}: box mask differs"
        ref = build_target(image, pil, (size, size))
        assert np.array_equal(mine["src"], ref["tile"]), f"{it['name']}: tile differs"
        assert np.array_equal(mine["mask"], (ref["mask"] > 0).astype(np.uint8)), f"{it['name']}: mask differs"
        assert mine["box"] == tuple(int(v) for v in ref["box"]), f"{it['name']}: crop box differs"


def check_loader(cfg, n_batches=3):
    """Real DataLoader with workers: throughput, and identical batches for an identical main-process
    torch seed and epoch (what a resume relies on)."""
    tcfg = cfg["train"]

    def run():
        torch.manual_seed(1234)
        ds, sampler = build_train_data(cfg)
        sampler.set_epoch(3)
        loader = DataLoader(ds, batch_size=tcfg["batch_size"], sampler=sampler, num_workers=tcfg["dataloader_workers"])
        out, t0 = [], time.time()
        for b, batch in enumerate(loader):
            assert batch["ref_tile"].dtype == batch["src_tile"].dtype == torch.uint8
            out.append((batch["name"], batch["aug"], int(batch["ref_tile"].sum()), int(batch["src_tile"].sum()),
                        int(batch["mask_tile"].sum())))
            if b + 1 == n_batches:
                break
        return out, (time.time() - t0) / n_batches

    a, dt = run()
    b, _ = run()
    assert a == b, "same seed + epoch must give identical batches"
    print(f"DataLoader ok: {n_batches} batches x {tcfg['batch_size']} with {tcfg['dataloader_workers']} workers, "
          f"{dt:.2f} s/batch incl. worker start-up; reproducible across runs")


def run_check(cfg, processes=16):
    dcfg = cfg["data"]
    data = {split: AdditionDataset.from_split(dcfg, split, train=True) for split in EXPECTED_COUNTS}
    for split, ds in data.items():
        by_suffix = Counter(it["suffix"] for it in ds.items)
        got = (len(ds), by_suffix[1], by_suffix[2])
        print(f"{split}: {got[0]} kept (suffix 1: {got[1]}, suffix 2: {got[2]}), "
              f"{len({it['image_id'] for it in ds.items})} source images, {len({it['class_name'] for it in ds.items})} classes")
        assert got == EXPECTED_COUNTS[split], f"{split}: {got} != {EXPECTED_COUNTS[split]}"
    shared = {it["image_id"] for it in data["train"].items} & {it["image_id"] for it in data["val"].items}
    assert not shared, f"train and val share source images: {sorted(shared)[:5]}"
    print("train and val share no source images")

    jobs = [(it, dcfg, i) for ds in data.values() for i, it in enumerate(ds.items)]
    t0 = time.time()
    with Pool(processes) as pool:
        results = pool.map(_check_one, jobs, chunksize=32)
    bad = [(n, p) for n, p, _ in results if p]
    print(f"pixel checks on {len(results)} samples ({(time.time() - t0) / 60:.1f} min): {len(bad)} with problems; "
          f"{sum(e for _, _, e in results)} have a box reaching the far image edge (x2 = W or y2 = H)")
    assert not bad, bad[:10]

    _, sampler = build_train_data(cfg)
    k = dcfg["samples_per_epoch"]
    e0 = list(sampler.epoch_indices(0))
    assert len(e0) == len(set(e0)) == k and e0 == list(sampler.epoch_indices(0)) and e0 != list(sampler.epoch_indices(1))
    print(f"sampler ok: {k} distinct samples per epoch, deterministic per epoch, differs between epochs")

    a, b = make_eval_subsets(cfg), make_eval_subsets(cfg)
    assert a == b, "fixed evaluation set must be deterministic"
    evald = build_eval_data(cfg, a)
    for group, ds in evald.items():
        assert len(ds) == cfg["val"][f"n_{group}"]
        assert not ds.train
        assert len({it["image_id"] for it in ds.items}) == len(ds), f"{group}: repeated source image"
        by_suffix = Counter(it["suffix"] for it in ds.items)
        assert by_suffix[1] == by_suffix[2], f"{group}: suffixes not balanced: {by_suffix}"
        print(f"fixed set {group}: {sorted((it['name'], it['class_name']) for it in ds.items)}")
        check_eval_parity(ds, 20)
    check_eval_parity(AdditionDataset.from_split(dcfg, "val", train=False), 40)
    print("evaluation tile, mask and crop == create_mask_from_bbox + build_target")

    item = data["train"][0]
    S = dcfg["size"]
    assert all(item[k].shape == (3, S, S) and item[k].dtype == torch.uint8 for k in ("ref_tile", "src_tile", "gt_tile"))
    assert item["mask_tile"].shape == (1, S, S) and item["mask_tile"].dtype == torch.uint8
    d = make_addition_diptych(item["ref_tile"], item["src_tile"], item["gt_tile"], item["mask_tile"], dcfg["masked_fill"])
    assert d["src"].shape == d["mask"].shape == d["result"].shape == (3, S, 2 * S)
    assert set(torch.unique(d["mask"]).tolist()) == {0.0, 1.0}
    assert torch.all(d["mask"][..., :S] == 0) and torch.equal(d["src"][..., :S], d["result"][..., :S])
    assert torch.all(d["src"][..., S:][d["mask"][..., S:] > 0] == MASKED_FILL[dcfg["masked_fill"]])
    check_loader(cfg)
    print("all checks passed")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="experiments/config/addition.yaml")
    p.add_argument("--check", action="store_true", help="counts, per-sample pixel checks, parity, sampler, loader")
    p.add_argument("--dump", type=int, default=0, help="write N processed samples as image grids")
    p.add_argument("--split", choices=["train", "eval"], default="train")
    p.add_argument("--out", default="addition_data_dump")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    cfg = load_config(repo_path(args.config))
    if args.check:
        run_check(cfg)
    if args.dump:
        run_dump(cfg, args.dump, repo_path(args.out), args.split, args.seed)


if __name__ == "__main__":
    main()
