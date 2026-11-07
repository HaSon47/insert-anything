"""Shared removal inference and metrics, used by the training callback and eval_removal.py.

The model sees a diptych [white | target tile] with mask [0 | removal mask] and the cached
Redux embedding of a white image as its "prompt" (removal_training_plan.md, sections 3.2 and 6).
Metrics compare the raw 768x768 output tile with the GT tile (same crop / pad / resize).
"""

import csv
import os

import cv2
import numpy as np
import torch
from PIL import Image

from ..models.image_project import image_output

COND_FILE = "white_ref_cond.pt"


def white_image(size):
    return Image.new("RGB", (size, size), (255, 255, 255))


@torch.no_grad()
def build_white_ref_cond(redux, size, device):
    """Redux conditioning of an all-white reference, exactly as the training step uses it:
    prompt_embeds [1, 512 + 729, 4096] (zero T5 tokens + Redux image tokens), pooled [1, 768] (zeros)."""
    prompt_embeds, pooled = image_output(redux, white_image(size), device)
    return dict(prompt_embeds=prompt_embeds.detach().cpu(), pooled_prompt_embeds=pooled.detach().cpu())


def load_or_build_white_ref_cond(cfg, run_dir=None, device="cuda"):
    """Cached conditioning from <run_dir>/white_ref_cond.pt, or computed with FLUX.1-Redux (then the
    Redux pipeline is freed) and saved there."""
    path = os.path.join(run_dir, COND_FILE) if run_dir else None
    if path and os.path.exists(path):
        return torch.load(path, map_location="cpu")
    from diffusers import FluxPriorReduxPipeline
    dtype = getattr(torch, cfg["dtype"])
    redux = FluxPriorReduxPipeline.from_pretrained(cfg["flux_redux_path"], torch_dtype=dtype).to(device)
    cond = build_white_ref_cond(redux, cfg["data"]["size"], device)
    del redux
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if path:
        os.makedirs(run_dir, exist_ok=True)
        torch.save(cond, path)
    return cond


@torch.no_grad()
def run_removal(pipe, cond, src_tile, mask_tile, num_inference_steps=28, guidance_scale=30.0, seed=42):
    """One removal on a deterministic eval tile; returns the raw output tile (uint8, S x S x 3).

    `src_tile` is the unmasked uint8 tile and `mask_tile` its 0/1 mask; FluxFillPipeline does the
    masking itself (masked pixels -> 0 in [-1, 1]), which training mirrors with masked_fill: gray.
    """
    S = src_tile.shape[0]
    image = np.concatenate([np.full_like(src_tile, 255), src_tile], axis=1)
    mask = np.concatenate([np.zeros((S, S), np.uint8), (mask_tile > 0).astype(np.uint8) * 255], axis=1)
    device = pipe._execution_device
    dtype = pipe.transformer.dtype
    generator = torch.Generator(device).manual_seed(seed)
    out = pipe(
        image=Image.fromarray(image),
        mask_image=Image.fromarray(mask),
        height=S,
        width=2 * S,
        prompt_embeds=cond["prompt_embeds"].to(device, dtype),
        pooled_prompt_embeds=cond["pooled_prompt_embeds"].to(device, dtype),
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
        max_sequence_length=512,
    ).images[0]
    return np.array(out)[:, S:]


# ---------------------------------------------------------------- metrics

def psnr(pred, gt):
    """PSNR (dB) of two uint8 images, data range 255."""
    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


class LPIPS:
    """LPIPS-Alex on uint8 HxWx3 images (scaled to [-1, 1]); the network is loaded once."""

    def __init__(self, device="cuda"):
        import lpips
        self.device = device
        self.net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    @staticmethod
    def _t(img):
        return torch.from_numpy(img).permute(2, 0, 1)[None].float() / 127.5 - 1

    @torch.no_grad()
    def __call__(self, pred, gt):
        return float(self.net(self._t(pred).to(self.device), self._t(gt).to(self.device)).item())


# ---------------------------------------------------------------- outputs

def outline(tile, mask, color=(255, 0, 0), thickness=3):
    """Tile with the outline of the 0/1 mask drawn on it."""
    cs, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return cv2.drawContours(np.ascontiguousarray(tile).copy(), cs, -1, color, thickness)


def put_label(img, text, org=(0, 0), scale=0.55, color=(255, 255, 0)):
    """Single-colour text on a black box (readable on any background); modifies img."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(img, (x, y), (x + tw + 10, y + th + 10), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 5, y + th + 4), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    return img


GRID_COLUMNS = ("input + mask", "output", "GT")


def grid_row(src, mask, out, gt, title="", note="", side=384):
    """[input tile with mask outline | raw output tile | GT tile], each resized to side x side."""
    S = src.shape[0]
    cells = [outline(src, mask, thickness=max(1, round(3 * S / 768))), out, gt]
    cells = [np.ascontiguousarray(c if c.shape[0] == side else cv2.resize(c, (side, side), interpolation=cv2.INTER_AREA))
             for c in cells]
    if title:
        put_label(cells[0], title)
    if note:
        put_label(cells[1], note)
    return np.concatenate(cells, axis=1)


def make_grid(rows, side=384, header=GRID_COLUMNS):
    """Stack grid_row outputs under a header bar naming the columns."""
    bar = np.full((28, side * len(header), 3), 32, np.uint8)
    for j, h in enumerate(header):
        cv2.putText(bar, h, (j * side + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([bar, *rows], axis=0)


def write_csv(path, rows, fields=None):
    """Write dict rows atomically (tmp file + rename)."""
    if fields is None:
        fields = list(rows[0]) if rows else []
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))
