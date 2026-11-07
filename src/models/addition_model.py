"""AdditionModel: FLUX.1-Fill-dev + InsertAnything LoRA, fine-tuned for object addition.

addition_training_plan.md, section 3. It inherits LoRA setup and strict loading, float32 master weights,
Prodigy, LoRA export and the LoRA-only resume hooks from RemovalModel. Differences:
- each sample has its own reference, so conditioning is computed per sample with FLUX.1-Redux;
- the diptych's left half is that reference instead of white.

Existing removal files are imported, not edited, because the unfinished removal_v1 run may resume with them.

GPU checks of the real model, from the repo root:
    python -m src.models.addition_model --gpu_check --out runs_addition/_checks/m2
    python -m src.models.addition_model --probe_batch 8 --out runs_addition/_checks/m2
    python -m src.models.addition_model --reload_check <run_dir> --step <N>
Run the probe on its own: earlier models in the same process would inflate its memory figures.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import FluxPriorReduxPipeline
from PIL import Image

from ..data.addition_data import make_addition_diptych
from ..eval.addition_eval import reference_conditioning
from .pipeline_tools import Flux_fill_encode_masks_images, encode_images
from .removal_model import RemovalModel, load_fill_pipeline
from .transformer import tranformer_forward


def load_redux(cfg, device):
    """FLUX.1-Redux, frozen. A pipeline is not an nn.Module, so it stays out of the trainable parameters
    and of resume checkpoints."""
    redux = FluxPriorReduxPipeline.from_pretrained(cfg["flux_redux_path"], torch_dtype=getattr(torch, cfg["dtype"]))
    redux.set_progress_bar_config(disable=True)
    redux.image_encoder.requires_grad_(False).eval()
    redux.image_embedder.requires_grad_(False).eval()
    return redux.to(device)


class AdditionModel(RemovalModel):
    def __init__(self, cfg, redux=None, pipe=None, device=None, load_init_lora=True):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # RemovalModel requires a fixed conditioning; this model never reads it (see step).
        unused = dict(prompt_embeds=torch.zeros(1, 1, 1), pooled_prompt_embeds=torch.zeros(1, 1))
        super().__init__(cfg, pipe=pipe, cond=unused, device=device, load_init_lora=load_init_lora)
        self.flux_redux = redux if redux is not None else load_redux(cfg, device)

    def on_after_batch_transfer(self, batch, dataloader_idx=0):
        if "src_tile" in batch:
            batch.update(make_addition_diptych(batch["ref_tile"], batch["src_tile"], batch["gt_tile"],
                                               batch["mask_tile"], self.masked_fill))
        return batch

    def reference_batch(self, ref_tiles):
        """prompt_embeds [B, 512 + N, 4096], pooled [B, 768] from uint8 tiles [B, 3, S, S]."""
        conds = [reference_conditioning(self.flux_redux, t.permute(1, 2, 0).cpu().numpy(), self.device, self.dtype)
                 for t in ref_tiles]
        return torch.cat([c[0] for c in conds]), torch.cat([c[1] for c in conds])

    def step(self, batch):
        """RemovalModel.step with per-sample conditioning; the flow-matching math is the same."""
        pipe = self.flux_fill_pipe
        result, src, mask = batch["result"], batch["src"], batch["mask"]
        B = result.shape[0]
        prompt_embeds, pooled = self.reference_batch(batch["ref_tile"])
        text_ids = torch.zeros(prompt_embeds.shape[1], 3, device=self.device, dtype=self.dtype)

        with torch.no_grad():
            x_0, img_ids = encode_images(pipe, result)
            t = torch.sigmoid(torch.randn((B,), device=self.device))
            x_1 = torch.randn_like(x_0)
            t_ = t[:, None, None]
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype)
            src_latents, mask_latents = Flux_fill_encode_masks_images(pipe, src, mask)
            condition = torch.cat((src_latents, mask_latents), dim=-1)
            guidance = (torch.full((B,), self.train_guidance, device=self.device)
                        if self.transformer.config.guidance_embeds else None)

        pred = tranformer_forward(
            self.transformer,
            model_config=self.model_config,
            hidden_states=torch.cat((x_t, condition), dim=2),
            timestep=t,
            guidance=guidance,
            pooled_projections=pooled,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
        loss = F.mse_loss(pred.float(), (x_1 - x_0).float())
        self.last_t = t.mean().item()
        return loss


# ---------------------------------------------------------------- GPU checks (milestones M2, M3)

def inference_script_call(pipe, redux, tiles, cfg):
    """The output of the path the inference scripts use: build_diptych + redux(PIL) + pipe(**prior)."""
    from inference_fsc_final import build_diptych
    S = tiles["src"].shape[0]
    diptych, mask = build_diptych(tiles["ref"], dict(tile=tiles["src"], mask=tiles["mask"]))
    out = pipe(
        image=diptych,
        mask_image=mask,
        height=mask.size[1],
        width=mask.size[0],
        max_sequence_length=512,
        generator=torch.Generator(pipe._execution_device).manual_seed(cfg["val"]["seed"]),
        num_inference_steps=cfg["val"]["num_inference_steps"],
        guidance_scale=cfg["val"]["guidance_scale"],
        **redux(Image.fromarray(tiles["ref"])),
    ).images[0]
    return np.array(out)[:, S:]


def _psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else round(float(10 * np.log10(255.0 ** 2 / mse)), 2)


def gpu_check(cfg, out_dir, n_parity=2):
    """Real-weights checks; writes <out_dir>/m2_report.json and parity images."""
    from ..data.addition_data import build_eval_data, make_eval_subsets
    from ..eval.addition_eval import run_addition

    os.makedirs(out_dir, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    report, device = {}, "cuda"
    t0 = time.time()

    def save(stage):
        report["stage_done"] = stage
        with open(os.path.join(out_dir, "m2_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"[m2] {stage} done ({time.time() - t0:.0f}s)", flush=True)

    data = build_eval_data(cfg, make_eval_subsets(cfg))
    samples = [(group, ds.items[i]["name"], ds.tiles(i)) for group, ds in data.items() for i in range(min(n_parity, len(ds)))]

    # 1. Our per-sample conditioning vs the Redux pipeline call of the inference scripts, on real references.
    redux = load_redux(cfg, device)
    report["mem_redux_GB"] = round(torch.cuda.memory_allocated() / 2**30, 2)
    diffs = []
    for _, name, tiles in samples:
        ours = reference_conditioning(redux, tiles["ref"], device, getattr(torch, cfg["dtype"]))
        prior = redux(Image.fromarray(tiles["ref"]))
        diffs.append(dict(sample=name, prompt_max_abs=float((ours[0] - prior.prompt_embeds).abs().max()),
                          pooled_max_abs=float((ours[1] - prior.pooled_prompt_embeds).abs().max()),
                          shape=list(ours[0].shape)))
    report["conditioning"] = diffs
    conds = [reference_conditioning(redux, t["ref"], device, getattr(torch, cfg["dtype"]))[0] for _, _, t in samples[:2]]
    report["two_references_differ"] = bool((conds[0] != conds[1]).any())
    save("1_conditioning")

    # 2. Model with the pretrained LoRA.
    model = AdditionModel(cfg, redux=redux, device=device)
    report["n_lora_modules"] = model.n_lora_modules
    report["n_lora_params_M"] = round(sum(p.numel() for n, p in model.named_parameters() if p.requires_grad) / 1e6, 1)
    report["load_s"] = round(time.time() - t0, 1)
    report["mem_after_load_GB"] = round(torch.cuda.memory_allocated() / 2**30, 1)
    assert model.n_lora_modules == 343, model.n_lora_modules
    save("2_load")

    # 3. Evaluation path vs a fresh pipeline + pipe.load_lora_weights + the inference scripts' call.
    # The training model holds float32 master weights while a loaded LoRA is bf16, so the check runs both:
    # bf16 must be pixel-identical; float32 is what evaluation during training actually uses.
    model.transformer.eval()
    vcfg = cfg["val"]

    def run_all():
        return [run_addition(model.flux_fill_pipe, redux, t["ref"], t["src"], t["mask"], vcfg["num_inference_steps"],
                             vcfg["guidance_scale"], vcfg["seed"]) for _, _, t in samples]

    out_fp32 = run_all()
    lora = set(model.lora_names)
    for n, p in model.named_parameters():
        if n in lora:
            p.data = p.data.to(getattr(torch, cfg["dtype"]))
    out_same_dtype = run_all()
    del model
    torch.cuda.empty_cache()
    ref_pipe = load_fill_pipeline(cfg, device)
    ref_pipe.load_lora_weights(cfg["init_lora_path"])
    parity = []
    for (group, name, t), a, b in zip(samples, out_fp32, out_same_dtype):
        ref = inference_script_call(ref_pipe, redux, t, cfg)
        parity.append(dict(sample=f"{group}/{name}",
                           same_dtype_max_abs=int(np.abs(b.astype(int) - ref.astype(int)).max()),
                           fp32_vs_script_psnr=_psnr(a, ref),
                           fp32_max_abs=int(np.abs(a.astype(int) - ref.astype(int)).max())))
        Image.fromarray(np.concatenate([t["ref"], t["src"], a, ref, t["gt"]], axis=1)).save(
            os.path.join(out_dir, f"parity_{group}_{name}.jpg"))
    report["parity"] = parity
    del ref_pipe
    torch.cuda.empty_cache()
    save("3_parity")
    print(json.dumps(report, indent=2))


def probe(cfg, out_dir, batch_size):
    """Memory and speed of full training steps (forward, backward, Prodigy), from a clean process, on the
    allocator settings of the launcher. Writes <out_dir>/probe_report.json."""
    from ..data.addition_data import build_train_data

    os.makedirs(out_dir, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda"
    assert torch.cuda.memory_allocated() == 0
    model = AdditionModel(cfg, device=device)
    report = dict(batch=batch_size, mem_after_load_GB=round(torch.cuda.memory_allocated() / 2**30, 1))
    opt = model.configure_optimizers()
    ds, sampler = build_train_data(cfg)
    batch = torch.utils.data.default_collate([ds[j] for j in list(iter(sampler))[:batch_size]])
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    batch = model.on_after_batch_transfer(batch)
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(4):
        torch.cuda.synchronize()
        t1 = time.time()
        loss = model.step(batch)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(round(time.time() - t1, 2))
    t1 = time.time()
    model.reference_batch(batch["ref_tile"])
    report.update(loss=float(loss), s_per_step=times, redux_s_per_batch=round(time.time() - t1, 3),
                  peak_allocated_GB=round(torch.cuda.max_memory_allocated() / 2**30, 1),
                  peak_reserved_GB=round(torch.cuda.max_memory_reserved() / 2**30, 1))
    with open(os.path.join(out_dir, "probe_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


def reload_check(run_dir, step, out_dir=None):
    """A LoRA export reloaded with pipe.load_lora_weights must reproduce the outputs written during training
    at that step. The export is bf16 and training holds float32 master weights, so small differences are expected."""
    from ..data.addition_data import build_eval_data, load_config
    from ..eval.addition_eval import psnr_bg, run_addition
    from ..eval.removal_eval import read_csv

    cfg = load_config(os.path.join(run_dir, "config.yaml"))
    device, vcfg = "cuda", cfg["val"]
    with open(os.path.join(run_dir, "val", "subsets.json")) as f:
        data = build_eval_data(cfg, json.load(f))
    pipe = load_fill_pipeline(cfg, device)
    pipe.load_lora_weights(os.path.join(run_dir, "ckpt", str(step)))
    redux = load_redux(cfg, device)
    rows = []
    for group, ds in data.items():
        written = {r["name"]: float(r["psnr_bg"]) for r in read_csv(os.path.join(run_dir, "val", f"step_{step}", group, "metrics.csv"))}
        for i in range(len(ds)):
            name, t = ds.items[i]["name"], ds.tiles(i)
            out = run_addition(pipe, redux, t["ref"], t["src"], t["mask"], vcfg["num_inference_steps"],
                               vcfg["guidance_scale"], vcfg["seed"])
            saved = np.array(Image.open(os.path.join(run_dir, "val", f"step_{step}", group, f"{name}.png")).convert("RGB"))
            rows.append(dict(sample=f"{group}/{name}", output_psnr_vs_saved=_psnr(out, saved),
                             psnr_bg_reloaded=round(psnr_bg(out, t["gt"], t["mask"], t["valid"]), 4),
                             psnr_bg_written=written[name]))
    report = dict(step=step, rows=rows)
    out_dir = out_dir or run_dir
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"reload_check_{step}.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


def main():
    from ..data.removal_data import load_config, repo_path
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="experiments/config/addition.yaml")
    p.add_argument("--gpu_check", action="store_true")
    p.add_argument("--probe_batch", type=int, default=0, help="measure training steps at this batch size; run on its own")
    p.add_argument("--out", default="runs_addition/_checks/m2")
    p.add_argument("--reload_check", default=None, metavar="RUN_DIR")
    p.add_argument("--step", type=int, default=None)
    args = p.parse_args()
    if args.gpu_check and args.probe_batch:
        p.error("run --gpu_check and --probe_batch in separate processes")
    if args.gpu_check:
        gpu_check(load_config(repo_path(args.config)), repo_path(args.out))
    if args.probe_batch:
        probe(load_config(repo_path(args.config)), repo_path(args.out), args.probe_batch)
    if args.reload_check:
        reload_check(repo_path(args.reload_check), args.step)


if __name__ == "__main__":
    main()
