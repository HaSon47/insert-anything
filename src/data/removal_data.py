"""Object-removal data: ObjectStitch_V2 + omnipaint-bench -> white-reference diptychs.

See removal_training_plan.md, sections 1-2. Each training item is

    src    = [white | masked target tile]     3 x S x 2S, in [0, 1]
    mask   = [zeros | removal mask tile]      3 x S x 2S, 0/1
    result = [white | ground-truth tile]      3 x S x 2S, in [0, 1]

where the target tile is the inference crop (build_target in inference_fsc_final.py) around
the object, with random jitter and mask augmentation during training.

Debug / checks (run from the repo root):
    python -m src.data.removal_data --check
    python -m src.data.removal_data --dump 32 --out <dir>
"""

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, Dataset, Sampler

from utils.utils import box2squre, expand_bbox, get_bbox_from_mask, pad_to_square

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MIN_MASK_PIXELS = 10
AUG_BRANCHES = ("dilate", "bbox", "erode", "inference")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def repo_path(p):
    return p if os.path.isabs(p) else os.path.join(REPO, p)


# ---------------------------------------------------------------- loading

def read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask(path, hw=None):
    """Binary uint8 (0/1) mask; JPEG and soft-edged masks are thresholded at 128."""
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    if hw is not None and m.shape != tuple(hw):
        m = cv2.resize(m, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 128).astype(np.uint8)


# ---------------------------------------------------------------- geometry

def inference_mask(obj_mask):
    """The mask dilation the inference scripts apply before cropping."""
    return cv2.dilate(obj_mask, np.ones((7, 7), np.uint8), iterations=2)


def inference_crop_box(image, mask):
    """build_target's crop: two area-adaptive expansions, then a square clipped to the image."""
    box = get_bbox_from_mask(mask)
    box = expand_bbox(mask, box, ratio=1.2)
    box = expand_bbox(image, box, ratio=2)
    return box2squre(image, box)


def place_square(cy, cx, side, keep_box, H, W):
    """Square of `side` centred near (cy, cx) that contains `keep_box`, clipped to the image.

    `keep_box` is (y1, y2, x1, x2) with inclusive ends, as get_bbox_from_mask returns.
    """
    ky1, ky2, kx1, kx2 = keep_box
    side = max(side, (ky2 - ky1 + 1) * 1.05, (kx2 - kx1 + 1) * 1.05)
    half = side / 2
    cy = min(max(cy, ky2 + 1 - half), ky1 + half)
    cx = min(max(cx, kx2 + 1 - half), kx1 + half)
    y1, y2 = int(np.floor(cy - half)), int(np.ceil(cy + half))
    x1, x2 = int(np.floor(cx - half)), int(np.ceil(cx + half))
    return max(0, y1), min(H, y2), max(0, x1), min(W, x2)


def union_box(a, b):
    return min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3])


def ellipse(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def augment_mask(obj_mask, aug_cfg, rng, px_per_tile):
    """One of dilate / bbox / erode / inference. Sizes in aug_cfg are 768-tile pixels and are
    converted to original-image pixels with `px_per_tile`."""
    probs = np.array([aug_cfg.get(b, {}).get("p", 0.0) for b in AUG_BRANCHES], dtype=np.float64)
    branch = AUG_BRANCHES[rng.choice(len(AUG_BRANCHES), p=probs / probs.sum())]
    cfg = aug_cfg.get(branch, {})

    if branch == "dilate":
        r_tile = rng.integers(cfg["radius"][0], cfg["radius"][1] + 1)
        r = max(1, int(round(r_tile * px_per_tile)))
        return cv2.dilate(obj_mask, ellipse(r)), branch

    if branch == "bbox":
        H, W = obj_mask.shape
        y1, y2, x1, x2 = get_bbox_from_mask(obj_mask)
        s = rng.uniform(*cfg["scale"])
        cy, cx = (y1 + y2) / 2, (x1 + x2) / 2
        hh, hw = (y2 - y1 + 1) * s / 2, (x2 - x1 + 1) * s / 2
        out = np.zeros_like(obj_mask)
        out[max(0, int(np.floor(cy - hh + 0.5))):min(H, int(np.ceil(cy + hh + 0.5))),
            max(0, int(np.floor(cx - hw + 0.5))):min(W, int(np.ceil(cx + hw + 0.5)))] = 1
        return out, branch

    if branch == "erode":
        y1, y2, x1, x2 = get_bbox_from_mask(obj_mask)
        cap = int(cfg.get("max_frac_of_obj", 0.1) * min(y2 - y1 + 1, x2 - x1 + 1))
        r_tile = rng.integers(cfg["radius"][0], cfg["radius"][1] + 1)
        r = min(int(round(r_tile * px_per_tile)), cap)
        if r < 1:
            return obj_mask.copy(), "erode0"
        out = cv2.erode(obj_mask, ellipse(r))
        if out.sum() < MIN_MASK_PIXELS:
            return obj_mask.copy(), "erode0"
        return out, branch

    return inference_mask(obj_mask), branch


def to_tile(img, box, size, pad_value, interpolation=cv2.INTER_LINEAR):
    """Crop to `box`, pad to a centred square with `pad_value`, resize to size x size."""
    y1, y2, x1, x2 = box
    crop = pad_to_square(img[y1:y2, x1:x2], pad_value=pad_value)
    return cv2.resize(crop, (size, size), interpolation=interpolation)


def build_tiles(src, gt, mask, box, size):
    """src/gt/mask tiles (uint8) for one crop box; the GT gets exactly the src transform."""
    return dict(
        src=to_tile(src, box, size, 255),
        gt=to_tile(gt, box, size, 255),
        mask=(to_tile(mask, box, size, 0, cv2.INTER_NEAREST) > 0).astype(np.uint8),
        box=tuple(int(v) for v in box),
    )


def eval_tiles(src, gt, obj_mask, size):
    """Deterministic tiles for validation / evaluation: inference dilation and crop, no jitter."""
    mask = inference_mask(obj_mask)
    return build_tiles(src, gt, mask, inference_crop_box(src, mask), size)


def train_tiles(src, gt, obj_mask, dcfg, rng, keep_full_mask=False):
    """Jittered inference crop + augmented removal mask (plan section 2, steps 2-4)."""
    size = dcfg["size"]
    H, W = obj_mask.shape
    dil = inference_mask(obj_mask)
    keep = get_bbox_from_mask(dil)
    y1, y2, x1, x2 = inference_crop_box(src, dil)

    jcfg = dcfg["crop_jitter"]
    side = max(y2 - y1, x2 - x1) * rng.uniform(*jcfg["scale"])
    cy = (y1 + y2) / 2 + rng.uniform(-jcfg["shift"], jcfg["shift"]) * side
    cx = (x1 + x2) / 2 + rng.uniform(-jcfg["shift"], jcfg["shift"]) * side
    box = place_square(cy, cx, side, keep, H, W)

    px_per_tile = max(box[1] - box[0], box[3] - box[2]) / size
    mask, branch = augment_mask(obj_mask, dcfg["mask_aug"], rng, px_per_tile)
    if mask.sum() < MIN_MASK_PIXELS:
        mask, branch = dil, "inference"
    box = place_square(cy, cx, side, union_box(keep, get_bbox_from_mask(mask)), H, W)

    tiles = build_tiles(src, gt, mask, box, size)
    tiles["aug"] = branch
    if keep_full_mask:
        tiles["mask_full"] = mask
    return tiles


def tiles_to_tensors(tiles):
    """Compact uint8 CHW tensors for the DataLoader (about 4 MB per sample instead of the 42 MB of
    float32 diptychs, which matters for /dev/shm). make_diptych() expands them, on the GPU."""
    return dict(
        src_tile=torch.from_numpy(np.ascontiguousarray(tiles["src"].transpose(2, 0, 1))),
        gt_tile=torch.from_numpy(np.ascontiguousarray(tiles["gt"].transpose(2, 0, 1))),
        mask_tile=torch.from_numpy(tiles["mask"][None].copy()),
    )


def make_diptych(src_tile, gt_tile, mask_tile, masked_fill="gray"):
    """Model inputs from (batched) uint8 tiles (plan section 2, steps 5-6):

        src    = [white | target tile, masked pixels filled]   in [0, 1]
        mask   = [zeros | mask]                                0/1
        result = [white | GT tile]                             in [0, 1]

    Works on [3,S,S] / [1,S,S] or batched [B,3,S,S] / [B,1,S,S], on any device.
    gray fill = 0.5, i.e. 0 after FLUX's [-1, 1] normalization, as FluxFillPipeline masks;
    black = 0, as the original InsertAnything training did.
    """
    fill = {"gray": 0.5, "black": 0.0}[masked_fill]
    src = src_tile.float() / 255
    gt = gt_tile.float() / 255
    m = (mask_tile > 0).float().expand_as(src)
    masked = torch.where(m > 0, torch.full_like(src, fill), src)
    white, black = torch.ones_like(src), torch.zeros_like(src)
    return dict(
        src=torch.cat([white, masked], dim=-1),
        mask=torch.cat([black, m], dim=-1),
        result=torch.cat([white, gt], dim=-1),
    )


def sample_rng():
    """Per-sample generator seeded from torch's RNG, which DataLoader seeds per worker from the
    main-process RNG, so augmentation is reproducible (and survives a resume)."""
    return np.random.default_rng(int(torch.randint(0, 2**62, ()).item()))


# ---------------------------------------------------------------- datasets

class RemovalDataset(Dataset):
    """Items are dicts with src / gt / mask paths plus metadata; subclasses build them."""

    name = "removal"

    def __init__(self, items, dcfg, train=True):
        self.items = items
        self.dcfg = dcfg
        self.train = train

    def __len__(self):
        return len(self.items)

    def load(self, idx):
        it = self.items[idx]
        src = read_rgb(it["src"])
        gt = read_rgb(it["gt"])
        if gt.shape != src.shape:
            raise ValueError(f"GT {gt.shape} != input {src.shape}: {it['gt']}")
        return src, gt, read_mask(it["mask"], src.shape[:2])

    def __getitem__(self, idx):
        rng = sample_rng()
        for _ in range(10):
            src, gt, obj = self.load(idx)
            if obj.sum() >= MIN_MASK_PIXELS:
                break
            idx = int(rng.integers(len(self)))  # empty mask: take another sample
        else:
            raise RuntimeError(f"{self.name}: no usable mask after 10 tries")
        if self.train:
            tiles = train_tiles(src, gt, obj, self.dcfg, rng)
        else:
            tiles = eval_tiles(src, gt, obj, self.dcfg["size"])
            tiles["aug"] = "inference"
        out = tiles_to_tensors(tiles)
        out.update(dataset=self.name, name=self.items[idx]["name"], aug=tiles["aug"])
        return out

    def subset(self, names):
        by_name = {it["name"]: it for it in self.items}
        return type(self).from_items([by_name[n] for n in names], self.dcfg, self.train)

    @classmethod
    def from_items(cls, items, dcfg, train):
        obj = cls.__new__(cls)
        RemovalDataset.__init__(obj, items, dcfg, train)
        return obj


class ObjectStitchRemoval(RemovalDataset):
    """ObjectStitch_V2: inpaint/ = input with the object, images/ = GT without it.

    Files are <scene>_b<branch>_<round>.png; a round-j input is the round-(j-1) GT.
    """

    name = "objectstitch"
    PATTERN = re.compile(r"^(\d+)_b(\d+)_(\d+)\.png$")

    def __init__(self, root, split, dcfg, rounds=(1, 2), train=True):
        names = []
        for f in sorted(os.listdir(os.path.join(root, "mask", split))):
            m = self.PATTERN.match(f)
            if m and int(m.group(3)) in rounds:
                names.append((f, m))
        have = {d: set(os.listdir(os.path.join(root, d, split))) for d in ("inpaint", "images")}
        items = []
        for f, m in names:
            missing = [d for d in have if f not in have[d]]
            if missing:
                raise FileNotFoundError(f"ObjectStitch {split}/{f} missing in {missing}")
            items.append(dict(
                name=f[:-4],
                src=os.path.join(root, "inpaint", split, f),
                gt=os.path.join(root, "images", split, f),
                mask=os.path.join(root, "mask", split, f),
                scene=m.group(1), branch=int(m.group(2)), round=int(m.group(3)),
            ))
        super().__init__(items, dcfg, train)


class OmniPaintRemoval(RemovalDataset):
    """omnipaint-bench: img/ = input, gt/ = GT, mask/ = object mask (same file names)."""

    name = "omnipaint"
    SPLITS = {"train": "Train", "val": "Val", "test": "Test"}

    def __init__(self, root, split, dcfg, exclude_file=None, train=True):
        split = self.SPLITS.get(split.lower(), split)
        exclude = set()
        if exclude_file:
            with open(repo_path(exclude_file)) as f:
                exclude = {ln.strip() for ln in f if ln.strip() and not ln.startswith("#")}
        d = os.path.join(root, split)
        have = {k: set(os.listdir(os.path.join(d, k))) for k in ("gt", "mask")}
        items = []
        for f in sorted(os.listdir(os.path.join(d, "img"))):
            stem, ext = os.path.splitext(f)
            if ext.lower() not in (".jpg", ".jpeg", ".png") or stem in exclude:
                continue
            missing = [k for k in have if f not in have[k]]
            if missing:
                raise FileNotFoundError(f"omnipaint {split}/{f} missing in {missing}")
            items.append(dict(
                name=stem,
                src=os.path.join(d, "img", f),
                gt=os.path.join(d, "gt", f),
                mask=os.path.join(d, "mask", f),
            ))
        super().__init__(items, dcfg, train)


# ---------------------------------------------------------------- sampling

class BalancedEpochSampler(Sampler):
    """Indices into a ConcatDataset: per epoch, `per_epoch[i]` samples from dataset i (without
    replacement while possible), all shuffled together. The draw depends only on (seed, epoch), so a
    resumed run repeats it exactly.

    `epoch_fn` is read whenever an iterator is created. train_removal.py sets it to
    `global_step // steps_per_epoch` instead of relying on Lightning's set_epoch(): after a restart
    Lightning builds the first iterator before it advances its epoch counter, so the resumed epoch
    would replay an earlier epoch's samples (caught by tests/test_removal_resume.py)."""

    def __init__(self, sizes, per_epoch, seed=0, epoch_fn=None):
        self.sizes = list(sizes)
        self.per_epoch = list(per_epoch)
        self.offsets = np.cumsum([0] + self.sizes[:-1]).tolist()
        self.seed = seed
        self.epoch = 0
        self.epoch_fn = epoch_fn

    def set_epoch(self, epoch):
        if self.epoch_fn is None:  # with epoch_fn, `epoch` is the epoch of the latest iterator
            self.epoch = int(epoch)

    def __len__(self):
        return sum(self.per_epoch)

    def epoch_indices(self, epoch):
        g = torch.Generator().manual_seed(self.seed * 100003 + epoch)
        parts = []
        for off, n, k in zip(self.offsets, self.sizes, self.per_epoch):
            reps = -(-k // n)
            pick = torch.cat([torch.randperm(n, generator=g) for _ in range(reps)])[:k]
            parts.append(pick + off)
        idx = torch.cat(parts)
        return idx[torch.randperm(len(idx), generator=g)].tolist()

    def __iter__(self):
        if self.epoch_fn is not None:
            self.epoch = int(self.epoch_fn())
        return iter(self.epoch_indices(self.epoch))


def build_train_data(cfg):
    """(ConcatDataset [omnipaint, objectstitch], BalancedEpochSampler) from the config."""
    dcfg = cfg["data"]
    omni = OmniPaintRemoval(dcfg["omnipaint_root"], "train", dcfg, train=True)
    objs = ObjectStitchRemoval(dcfg["objectstitch_root"], "train", dcfg,
                               rounds=tuple(dcfg["objectstitch_rounds"]), train=True)
    per = dcfg["samples_per_epoch"]
    sampler = BalancedEpochSampler([len(omni), len(objs)],
                                   [per["omnipaint"], per["objectstitch"]],
                                   seed=cfg["train"].get("seed", 0))
    return ConcatDataset([omni, objs]), sampler


def build_eval_data(cfg):
    """Full evaluation sets: ObjectStitch test (configured rounds) and deduped omnipaint Val."""
    dcfg = cfg["data"]
    return dict(
        objectstitch=ObjectStitchRemoval(dcfg["objectstitch_root"], "test", dcfg,
                                         rounds=tuple(dcfg["objectstitch_rounds"]), train=False),
        omnipaint=OmniPaintRemoval(dcfg["omnipaint_root"], "val", dcfg,
                                   exclude_file=dcfg.get("omnipaint_val_exclude"), train=False),
    )


def objectstitch_class(root, split, name):
    with open(os.path.join(root, "box_json", split, name + ".json")) as f:
        return json.load(f).get("class_name", "unknown")


def make_val_subsets(cfg):
    """Fixed validation names for the periodic validation:
    - ObjectStitch `val.objectstitch_split` (default val), round 1 only, one image per scene,
      spread over as many classes as possible (round-robin over shuffled classes);
    - omnipaint Val (deduped)."""
    dcfg, vcfg = cfg["data"], cfg["val"]
    rng = random.Random(vcfg.get("seed", 42))
    split = vcfg.get("objectstitch_split", "val")
    objs = ObjectStitchRemoval(dcfg["objectstitch_root"], split, dcfg, rounds=(1,), train=False)
    by_class = defaultdict(lambda: defaultdict(list))
    for it in objs.items:
        cls = objectstitch_class(dcfg["objectstitch_root"], split, it["name"])
        by_class[cls][it["scene"]].append(it["name"])
    classes = sorted(by_class)
    rng.shuffle(classes)
    picked, used_scenes = [], set()
    while len(picked) < vcfg["objectstitch_n"]:
        progress = False
        for cls in classes:
            free = sorted(s for s in by_class[cls] if s not in used_scenes)
            if not free or len(picked) == vcfg["objectstitch_n"]:
                continue
            scene = rng.choice(free)
            used_scenes.add(scene)
            picked.append(rng.choice(sorted(by_class[cls][scene])))
            progress = True
        if not progress:
            raise ValueError(f"only {len(picked)} round-1 scenes in ObjectStitch {split}")
    omni = OmniPaintRemoval(dcfg["omnipaint_root"], "val", dcfg,
                            exclude_file=dcfg.get("omnipaint_val_exclude"), train=False)
    return dict(
        objectstitch=sorted(picked),
        omnipaint=sorted(rng.sample([it["name"] for it in omni.items], vcfg["omnipaint_val_n"])),
    )


def build_val_data(cfg, subsets):
    """Datasets restricted to the fixed validation names (deterministic eval tiles)."""
    dcfg, vcfg = cfg["data"], cfg["val"]
    full = dict(
        objectstitch=ObjectStitchRemoval(dcfg["objectstitch_root"], vcfg.get("objectstitch_split", "val"),
                                         dcfg, rounds=(1,), train=False),
        omnipaint=OmniPaintRemoval(dcfg["omnipaint_root"], "val", dcfg,
                                   exclude_file=dcfg.get("omnipaint_val_exclude"), train=False),
    )
    return {k: full[k].subset(subsets[k]) for k in full}


# ---------------------------------------------------------------- debug

def _thumb(img, side=256):
    h, w = img.shape[:2]
    s = side / max(h, w)
    small = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    out = np.zeros((side, side, 3), np.uint8)
    out[:small.shape[0], :small.shape[1]] = small
    return out, s


def _contour(img, mask, color, thickness=2):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return cv2.drawContours(img.copy(), cs, -1, color, thickness)


def dump_row(ds, idx, rng, side=256):
    """[full input + crop box | input tile + object | model input | mask | GT tile]."""
    src, gt, obj = ds.load(idx)
    if ds.train:
        tiles = train_tiles(src, gt, obj, ds.dcfg, rng)
    else:
        tiles = eval_tiles(src, gt, obj, ds.dcfg["size"])
    tt = tiles_to_tensors(tiles)
    t = make_diptych(tt["src_tile"], tt["gt_tile"], tt["mask_tile"], ds.dcfg.get("masked_fill", "gray"))
    S = ds.dcfg["size"]
    assert torch.all(t["src"][:, :, :S] == 1) and torch.all(t["result"][:, :, :S] == 1)
    assert torch.all(t["mask"][:, :, :S] == 0)
    assert torch.equal(t["mask"][0, :, S:], torch.from_numpy(tiles["mask"]).float())

    y1, y2, x1, x2 = tiles["box"]
    full, s = _thumb(_contour(src, obj, (0, 255, 0), max(1, round(max(src.shape[:2]) / side))), side)
    cv2.rectangle(full, (round(x1 * s), round(y1 * s)), (round(x2 * s) - 1, round(y2 * s) - 1), (255, 0, 0), 2)
    obj_tile = to_tile(obj, tiles["box"], S, 0, cv2.INTER_NEAREST)
    model_in = (t["src"][:, :, S:].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    cols = [
        full,
        _contour(tiles["src"], obj_tile, (0, 255, 0), 3),
        model_in,
        np.repeat(tiles["mask"][..., None] * 255, 3, axis=2),
        tiles["gt"],
    ]
    cols = [c if c.shape[0] == side else cv2.resize(c, (side, side), interpolation=cv2.INTER_AREA) for c in cols]
    row = np.concatenate(cols, axis=1)
    label = f"{ds.name}/{ds.items[idx]['name']}  aug={tiles.get('aug', 'eval')}  crop={y2 - y1}x{x2 - x1}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(row, (0, 0), (tw + 10, th + 10), (0, 0, 0), -1)
    cv2.putText(row, label, (5, th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
    return row, tiles


def run_dump(cfg, n, out, split, seed):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(seed)
    if split == "train":
        concat, sampler = build_train_data(cfg)
        sampler.set_epoch(0)
        order = list(iter(sampler))[:n]
        picks = []
        for i in order:
            k = 0 if i < len(concat.datasets[0]) else 1
            picks.append((concat.datasets[k], i - (0 if k == 0 else len(concat.datasets[0]))))
    else:
        subsets = make_val_subsets(cfg)
        val = build_val_data(cfg, subsets)
        picks = [(ds, i) for ds in val.values() for i in range(len(ds))][:n]
    rows, stats = [], Counter()
    for ds, i in picks:
        row, tiles = dump_row(ds, i, rng)
        rows.append(row)
        stats[(ds.name, tiles.get("aug", "eval"))] += 1
    for g in range(0, len(rows), 8):
        path = os.path.join(out, f"{split}_dump_{g // 8:02d}.jpg")
        cv2.imwrite(path, cv2.cvtColor(np.concatenate(rows[g:g + 8], axis=0), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(path)
    print("samples per (dataset, aug):", dict(sorted(stats.items())))


def check_augmentation(ds, n, seed=0):
    """Property test over n random samples of a training dataset:
    - the augmented mask and the dilated object lie fully inside the crop box;
    - dilate / bbox / inference masks cover the whole object; erode masks lie inside it;
    - the mask tile is never empty.
    Returns stats on branch frequencies, crop jitter and mask area."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    branches, jitter, area = Counter(), [], []
    for i in idx:
        src, gt, obj = ds.load(int(i))
        t = train_tiles(src, gt, obj, ds.dcfg, rng, keep_full_mask=True)
        m, (y1, y2, x1, x2) = t["mask_full"], t["box"]
        name = ds.items[int(i)]["name"]
        dil = inference_mask(obj)
        assert m[y1:y2, x1:x2].sum() == m.sum(), f"{name}: aug mask leaves the crop"
        assert dil[y1:y2, x1:x2].sum() == dil.sum(), f"{name}: object leaves the crop"
        if t["aug"] in ("dilate", "bbox", "inference"):
            assert not np.any(obj & (1 - m)), f"{name}: {t['aug']} mask misses object pixels"
        elif t["aug"] == "erode":
            assert not np.any(m & (1 - obj)) and m.sum() < obj.sum(), f"{name}: bad erosion"
        assert t["mask"].sum() > 0, f"{name}: empty mask tile"
        assert t["src"].shape == t["gt"].shape == (ds.dcfg["size"], ds.dcfg["size"], 3)
        branches[t["aug"]] += 1
        ib = inference_crop_box(src, dil)
        jitter.append(max(y2 - y1, x2 - x1) / max(ib[1] - ib[0], ib[3] - ib[2]))
        area.append(t["mask"].mean())
    return dict(
        n=len(idx),
        branches={k: round(v / len(idx), 3) for k, v in sorted(branches.items())},
        crop_side_vs_inference=np.percentile(jitter, [0, 5, 50, 95, 100]).round(2).tolist(),
        mask_area_in_tile=np.percentile(area, [0, 5, 50, 95, 100]).round(3).tolist(),
    )


def check_eval_parity(ds, n):
    """eval_tiles must reproduce build_target() of the inference scripts pixel for pixel."""
    from inference_fsc_final import build_target
    size = ds.dcfg["size"]
    for i in range(min(n, len(ds))):
        src, gt, obj = ds.load(i)
        mine = eval_tiles(src, gt, obj, size)
        ref = build_target(src, obj, (size, size))
        assert np.array_equal(mine["src"], ref["tile"]), f"{ds.items[i]['name']}: tile differs"
        assert np.array_equal(mine["mask"], (ref["mask"] > 0).astype(np.uint8)), f"{ds.items[i]['name']}: mask differs"
        assert mine["box"] == tuple(int(v) for v in ref["box"])


def check_loader(cfg, n_batches=3):
    """Real DataLoader with workers: throughput, and identical batches for an identical main-process
    torch seed and epoch (what a resume relies on)."""
    import time
    from torch.utils.data import DataLoader
    tcfg = cfg["train"]

    def run():
        torch.manual_seed(1234)
        concat, sampler = build_train_data(cfg)
        sampler.set_epoch(3)
        loader = DataLoader(concat, batch_size=tcfg["batch_size"], sampler=sampler,
                            num_workers=tcfg["dataloader_workers"])
        out, t0 = [], time.time()
        for b, batch in enumerate(loader):
            assert batch["src_tile"].dtype == torch.uint8
            out.append((batch["name"], batch["aug"], int(batch["src_tile"].sum()), int(batch["mask_tile"].sum())))
            if b + 1 == n_batches:
                break
        return out, (time.time() - t0) / n_batches

    a, dt = run()
    b, _ = run()
    assert a == b, "same seed + epoch must give identical batches"
    print(f"DataLoader ok: {n_batches} batches x {tcfg['batch_size']} with {tcfg['dataloader_workers']} workers, "
          f"{dt:.2f} s/batch incl. worker start-up; reproducible across runs")


def run_check(cfg, scan_masks=False, n_aug=300):
    dcfg = cfg["data"]
    concat, sampler = build_train_data(cfg)
    omni, objs = concat.datasets
    ev = build_eval_data(cfg)
    counts = dict(objectstitch_train=len(objs), objectstitch_test=len(ev["objectstitch"]),
                  omnipaint_train=len(omni), omnipaint_val=len(ev["omnipaint"]))
    print("counts:", counts)
    expect = dict(objectstitch_train=9553, objectstitch_test=980, omnipaint_train=1000, omnipaint_val=112)
    for k, v in expect.items():
        assert counts[k] == v, f"{k}: {counts[k]} != {v}"
    assert Counter(it["round"] for it in objs.items).keys() <= set(dcfg["objectstitch_rounds"])

    n_omni = len(omni)
    epochs = {}
    for e in (0, 1):
        sampler.set_epoch(e)
        idx = list(iter(sampler))
        a, b = [i for i in idx if i < n_omni], [i for i in idx if i >= n_omni]
        assert len(idx) == 2000 and len(a) == 1000 and len(b) == 1000, (len(a), len(b))
        assert sorted(a) == list(range(n_omni)), "every omnipaint sample exactly once per epoch"
        assert len(set(b)) == 1000, "objectstitch drawn without replacement"
        epochs[e] = idx
        sampler.set_epoch(e)
        assert list(iter(sampler)) == idx, "draw must depend only on (seed, epoch)"
    overlap = len(set(i for i in epochs[0] if i >= n_omni) & set(i for i in epochs[1] if i >= n_omni))
    print(f"sampler ok: 1000 + 1000 per epoch, deterministic; objectstitch overlap epoch0/1 = {overlap}")
    assert overlap < 300

    s1, s2 = make_val_subsets(cfg), make_val_subsets(cfg)
    assert s1 == s2
    assert len({n.split("_")[0] for n in s1["objectstitch"]}) == len(s1["objectstitch"])
    assert all(n.endswith("_1") for n in s1["objectstitch"])
    vsplit = cfg["val"].get("objectstitch_split", "val")
    classes = [objectstitch_class(dcfg["objectstitch_root"], vsplit, n) for n in s1["objectstitch"]]
    print(f"val subsets (ObjectStitch {vsplit}, {len(set(classes))} classes: {sorted(set(classes))}):", json.dumps(s1))

    torch.manual_seed(0)
    S = dcfg["size"]
    for ds in (omni, objs):
        item = ds[0]
        assert item["src_tile"].shape == item["gt_tile"].shape == (3, S, S) and item["mask_tile"].shape == (1, S, S)
        assert item["src_tile"].dtype == item["mask_tile"].dtype == torch.uint8
        print(f"{ds.name}[0]:", {k: (tuple(v.shape), v.dtype) if torch.is_tensor(v) else v for k, v in item.items()})
        d = make_diptych(item["src_tile"][None], item["gt_tile"][None], item["mask_tile"][None], dcfg["masked_fill"])
        assert d["src"].shape == d["mask"].shape == d["result"].shape == (1, 3, S, 2 * S)
        assert set(torch.unique(d["mask"]).tolist()) <= {0.0, 1.0}
        assert torch.all(d["src"][..., :S] == 1) and torch.all(d["mask"][..., :S] == 0)
        m = d["mask"][..., S:] > 0
        fill = {"gray": 0.5, "black": 0.0}[dcfg["masked_fill"]]
        assert torch.all(d["src"][..., S:][m] == fill)
        assert torch.equal(d["src"][..., S:][~m], (item["src_tile"][None].float() / 255).expand_as(m)[~m])

    for ds in (omni, objs):
        print(f"augmentation {ds.name}:", check_augmentation(ds, n_aug))
    for ds in ev.values():
        check_eval_parity(ds, 20)
    print("eval tiles == inference build_target (20 per eval set)")
    check_loader(cfg)

    if scan_masks:
        for ds in (omni, objs, ev["objectstitch"], ev["omnipaint"]):
            small = [it["name"] for it in ds.items if read_mask(it["mask"]).sum() < MIN_MASK_PIXELS]
            print(f"{ds.name} ({'train' if ds.train else 'eval'}): {len(small)} masks < {MIN_MASK_PIXELS} px", small[:5])
    print("all checks passed")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="experiments/config/removal.yaml")
    p.add_argument("--check", action="store_true", help="counts, sampler and tensor checks")
    p.add_argument("--scan_masks", action="store_true", help="with --check: look for empty masks")
    p.add_argument("--dump", type=int, default=0, help="write N processed samples as image grids")
    p.add_argument("--split", choices=["train", "val"], default="train")
    p.add_argument("--out", default="removal_data_dump")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    cfg = load_config(repo_path(args.config))
    if args.check:
        run_check(cfg, args.scan_masks)
    if args.dump:
        run_dump(cfg, args.dump, args.out, args.split, args.seed)


if __name__ == "__main__":
    main()
