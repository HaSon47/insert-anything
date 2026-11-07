"""Addition inference and the evaluation metric, used by the training callback and the model checks.

The model sees a diptych [reference | input tile] with mask [0 | white rectangle] and the Redux
embedding of the reference tile as its "prompt" (addition_training_plan.md, sections 3 and 5).
"""

import cv2
import numpy as np
import torch
from PIL import Image

from ..models.image_project import image_output
from .removal_eval import make_grid, outline, put_label

GRID_COLUMNS = ("reference", "input + mask", "output", "GT")


@torch.no_grad()
def reference_conditioning(redux, ref_tile, device, dtype=None):
    """Redux conditioning of one uint8 HxWx3 reference tile, as the training step uses it:
    prompt_embeds [1, 512 + N, 4096] (zero T5 tokens + Redux tokens), pooled [1, 768] (zeros)."""
    prompt_embeds, pooled = image_output(redux, Image.fromarray(ref_tile), device)
    return prompt_embeds.to(dtype=dtype), pooled.to(dtype=dtype)


@torch.no_grad()
def run_addition(pipe, redux, ref_tile, src_tile, mask_tile, num_inference_steps=28, guidance_scale=30.0, seed=42):
    """One insertion on deterministic eval tiles; returns the raw output tile (uint8, S x S x 3).

    `src_tile` is the unmasked input tile and `mask_tile` its 0/1 mask; FluxFillPipeline does the masking
    itself (masked pixels -> 0 in [-1, 1]), which training mirrors with masked_fill: gray.
    """
    S = src_tile.shape[0]
    device = pipe._execution_device
    prompt_embeds, pooled = reference_conditioning(redux, ref_tile, device, pipe.transformer.dtype)
    image = np.concatenate([ref_tile, src_tile], axis=1)
    mask = np.concatenate([np.zeros((S, S), np.uint8), (mask_tile > 0).astype(np.uint8) * 255], axis=1)
    out = pipe(
        image=Image.fromarray(image),
        mask_image=Image.fromarray(mask),
        height=S,
        width=2 * S,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator(device).manual_seed(seed),
        max_sequence_length=512,
    ).images[0]
    return np.array(out)[:, S:]


def psnr_bg(pred, gt, mask_tile, valid=None):
    """PSNR (dB, uint8 RGB, data range 255) over pixels outside the mask and, if `valid` is given, inside
    it (valid = not padding). Says how well the scene is kept; it does not measure the inserted object."""
    keep = mask_tile == 0
    if valid is not None:
        keep = keep & valid
    if not keep.any():
        raise ValueError("psnr_bg: no pixels outside the mask")
    mse = np.mean((pred[keep].astype(np.float64) - gt[keep].astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def grid_row(ref, src, mask, out, gt, title="", note="", side=384):
    """[reference | input tile with mask outline | raw output tile | GT tile], each side x side."""
    S = src.shape[0]
    cells = [ref, outline(src, mask, thickness=max(1, round(3 * S / 768))), out, gt]
    cells = [np.ascontiguousarray(c if c.shape[0] == side else cv2.resize(c, (side, side), interpolation=cv2.INTER_AREA))
             for c in cells]
    if title:
        put_label(cells[1], title)
    if note:
        put_label(cells[2], note)
    return np.concatenate(cells, axis=1)


def save_grid(path, rows, side=384):
    Image.fromarray(make_grid(rows, side=side, header=GRID_COLUMNS)).save(path, quality=90)
