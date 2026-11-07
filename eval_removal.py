"""Evaluate a removal LoRA on the raw 768x768 output tiles (removal_training_plan.md, section 6).

    python eval_removal.py --lora runs_removal/<run>/ckpt/<step> --dataset objectstitch_test \\
        --work_dir outputs/removal_eval/<name> [--steps 50 --guidance_scale 30 --seed 42 --limit N --start K]

--lora takes a ckpt/<step> directory, a .safetensors file (e.g. the InsertAnything LoRA for the
baseline), or "none" (plain FLUX.1-Fill). It is loaded with pipe.load_lora_weights, the standard
diffusers path. Each sample: inference crop and tiles exactly as in training validation, diptych
[white | tile] with mask [0 | tile mask], cached white-reference Redux conditioning, right half of
the output = raw output tile. PSNR / LPIPS-Alex compare it with the GT tile (same crop / pad / resize).

Outputs in --work_dir: input_model_image/ input_model_mask/ output_model_image/ gt_model_image/
grid_image/ meta/ metrics.csv metrics_summary.json. Resumable: samples whose output tile exists are
not re-generated (their metrics are recomputed from the saved tiles).
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image

from src.data.removal_data import ObjectStitchRemoval, OmniPaintRemoval, eval_tiles, load_config, repo_path
from src.eval.removal_eval import (LPIPS, grid_row, load_or_build_white_ref_cond, make_grid, psnr, run_removal,
                                   write_csv)
from src.models.removal_model import load_fill_pipeline

DATASETS = ("objectstitch_test", "objectstitch_val", "omnipaint_val")
DIRS = ("input_model_image", "input_model_mask", "output_model_image", "gt_model_image", "grid_image", "meta")


def build_dataset(cfg, name):
    dcfg = cfg["data"]
    if name.startswith("objectstitch"):
        return ObjectStitchRemoval(dcfg["objectstitch_root"], name.split("_")[1], dcfg,
                                   rounds=tuple(dcfg["objectstitch_rounds"]), train=False)
    return OmniPaintRemoval(dcfg["omnipaint_root"], "val", dcfg, exclude_file=dcfg.get("omnipaint_val_exclude"),
                            train=False)


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def summarize(rows, settings):
    def stats(rs):
        return dict(n=len(rs), psnr=float(np.mean([r["psnr"] for r in rs])), lpips=float(np.mean([r["lpips"] for r in rs])))
    summary = dict(settings=settings, overall=stats(rows) if rows else None)
    rounds = sorted({r["round"] for r in rows if r.get("round") != ""})
    if rounds:
        summary["per_round"] = {str(k): stats([r for r in rows if r["round"] == k]) for k in rounds}
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="experiments/config/removal.yaml")
    p.add_argument("--lora", default="/home/user03/aiotlab/hachi/Image_Editing/code/insert-anything/runs_removal/removal_v1/ckpt/8000/pytorch_lora_weights.safetensors",
                   help='ckpt/<step> dir, .safetensors file, or "none"')
    p.add_argument("--dataset", choices=DATASETS, default="objectstitch_test")
    p.add_argument("--work_dir", default="/home/user03/aiotlab/hachi/Image_Editing/output/Tune_remove/objectstitch_test")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--size", type=int, default=None, help="tile size (default: data.size of the config)")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--names", default=None,
                   help="only these sample names: comma-separated, or @file with one name per line "
                        "(e.g. the run's val/subsets.json lists, to reproduce training validation)")
    p.add_argument("--cond", default=None, help="white-reference conditioning cache (default <work_dir>/white_ref_cond.pt)")
    args = p.parse_args()

    cfg = load_config(repo_path(args.config))
    if args.size:
        cfg["data"]["size"] = args.size
    size = cfg["data"]["size"]
    work = args.work_dir
    for d in DIRS:
        os.makedirs(os.path.join(work, d), exist_ok=True)

    ds = build_dataset(cfg, args.dataset)
    if args.names:
        if args.names.startswith("@"):
            with open(args.names[1:]) as f:
                wanted = [line.strip() for line in f if line.strip()]
        else:
            wanted = [n for n in args.names.split(",") if n]
        ds = ds.subset(wanted)
    idx = list(range(len(ds)))[args.start:]
    idx = idx[:args.limit] if args.limit else idx
    settings = dict(lora=args.lora, dataset=args.dataset, n=len(idx), start=args.start, steps=args.steps,
                    guidance_scale=args.guidance_scale, seed=args.seed, size=size, config=args.config)
    print(f"{args.dataset}: {len(idx)} of {len(ds)} samples -> {work}", flush=True)

    device = "cuda"
    if args.cond and os.path.exists(args.cond):
        cond = torch.load(args.cond, map_location="cpu")
    elif args.cond:
        cond = load_or_build_white_ref_cond(cfg, None, device)
        os.makedirs(os.path.dirname(os.path.abspath(args.cond)), exist_ok=True)
        torch.save(cond, args.cond)
    else:
        cond = load_or_build_white_ref_cond(cfg, work, device)
    pipe = load_fill_pipeline(cfg, device)
    if args.lora != "none":
        if os.path.isfile(args.lora):
            pipe.load_lora_weights(os.path.dirname(os.path.abspath(args.lora)), weight_name=os.path.basename(args.lora))
        else:
            pipe.load_lora_weights(args.lora, weight_name="pytorch_lora_weights.safetensors")
    lpips_fn = LPIPS(device)

    rows, t0, n_new = [], time.time(), 0
    for k, i in enumerate(idx):
        it = ds.items[i]
        name = it["name"]
        paths = {d: os.path.join(work, d, f"{name}.png") for d in DIRS[:4]}
        src, gt, obj = ds.load(i)
        tiles = eval_tiles(src, gt, obj, size)
        if os.path.exists(paths["output_model_image"]):
            out = load_rgb(paths["output_model_image"])
        else:
            out = run_removal(pipe, cond, tiles["src"], tiles["mask"], args.steps, args.guidance_scale, args.seed)
            Image.fromarray(tiles["src"]).save(paths["input_model_image"])
            Image.fromarray(tiles["mask"] * 255).save(paths["input_model_mask"])
            Image.fromarray(tiles["gt"]).save(paths["gt_model_image"])
            with open(os.path.join(work, "meta", f"{name}.json"), "w") as f:
                json.dump(dict(name=name, box_yyxx=tiles["box"], image_hw=list(src.shape[:2]),
                               mask_pixels_tile=int(tiles["mask"].sum()), **settings), f, indent=2)
            Image.fromarray(out).save(paths["output_model_image"])  # last: its existence marks the sample done
            n_new += 1
        p_, l_ = psnr(out, tiles["gt"]), lpips_fn(out, tiles["gt"])
        grid = make_grid([grid_row(tiles["src"], tiles["mask"], out, tiles["gt"], name, f"PSNR {p_:.2f}  LPIPS {l_:.3f}")])
        Image.fromarray(grid).save(os.path.join(work, "grid_image", f"{name}.jpg"), quality=90)
        rows.append(dict(name=name, round=it.get("round", ""), psnr=p_, lpips=l_))
        if (k + 1) % 10 == 0 or k + 1 == len(idx):
            write_csv(os.path.join(work, "metrics.csv"),
                      [dict(r, psnr=f"{r['psnr']:.4f}", lpips=f"{r['lpips']:.5f}") for r in rows])
            el = time.time() - t0
            eta = el / max(n_new, 1) * (len(idx) - k - 1) / 3600 if n_new else 0
            print(f"[{k + 1}/{len(idx)}] PSNR {np.mean([r['psnr'] for r in rows]):.3f}  "
                  f"LPIPS {np.mean([r['lpips'] for r in rows]):.4f}  ({n_new} generated, ~{eta:.1f} h left)", flush=True)

    summary = summarize(rows, settings)
    with open(os.path.join(work, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "settings"}, indent=2))


if __name__ == "__main__":
    main()
