"""RemovalModel: FLUX.1-Fill-dev + InsertAnything LoRA fine-tuned for object removal.

removal_training_plan.md, sections 3 and 4.2. Differences from InsertAnything (my_model.py):
- no T5 / CLIP text encoders and no per-sample Redux: the conditioning is the cached Redux
  embedding of a white image (buffers `cond_prompt_embeds` / `cond_pooled`);
- batches arrive as uint8 tiles and are expanded to diptychs on the GPU (on_after_batch_transfer);
- the LoRA can start from a saved InsertAnything LoRA (strictly checked);
- Lightning checkpoints hold only the LoRA (plus optimizer / loop / RNG state) for resuming;
- the trainable LoRA weights are float32 master weights (train.lora_dtype) on the bf16 base model.
  With bf16 LoRA weights (the original code casts everything to bf16) the M3 smoke run did not
  train: starting from the InsertAnything LoRA (|w| ~ 4e-3, bf16 spacing ~ 1.5e-5), Prodigy's
  first updates (d * lr = 1e-6) rounded away, so its step-size estimate d never grew from d0.
  LoRA exports are still written in the model dtype (bf16), the InsertAnything file format.

GPU check of the real model (milestone M2), from the repo root:
    python -m src.models.removal_model --gpu_check --out runs_removal/_checks/m2
"""

import argparse
import json
import os
import random
import time

import lightning as L
import numpy as np
import prodigyopt
import torch
import torch.nn.functional as F
from diffusers import FluxFillPipeline
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors import safe_open

from ..data.removal_data import make_diptych
from .pipeline_tools import Flux_fill_encode_masks_images, encode_images
from .transformer import tranformer_forward


# ---------------------------------------------------------------- loading

def load_fill_pipeline(cfg, device):
    """FLUX.1-Fill-dev without text encoders: prompts are always passed as embeddings."""
    pipe = FluxFillPipeline.from_pretrained(
        cfg["flux_fill_path"], text_encoder=None, text_encoder_2=None, tokenizer=None, tokenizer_2=None,
        torch_dtype=getattr(torch, cfg["dtype"]),
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


def lora_file_shapes(path):
    """{key without the `transformer.` prefix: shape} from a LoRA .safetensors header (no tensor loading)."""
    with safe_open(path, framework="pt") as f:
        return {k.removeprefix("transformer."): tuple(f.get_slice(k).get_shape()) for k in f.keys()}


def check_lora_compatible(transformer, path):
    """Assert the LoRA file has exactly the adapter's keys and shapes; returns the module count."""
    expected = {k: tuple(v.shape) for k, v in get_peft_model_state_dict(transformer).items()}
    got = lora_file_shapes(path)
    missing, unexpected = sorted(set(expected) - set(got)), sorted(set(got) - set(expected))
    assert not missing and not unexpected, (
        f"LoRA {path}: {len(missing)} missing (e.g. {missing[:3]}), {len(unexpected)} unexpected (e.g. {unexpected[:3]})")
    bad = [(k, expected[k], got[k]) for k in expected if expected[k] != got[k]]
    assert not bad, f"LoRA {path}: shape mismatch, e.g. {bad[:3]}"
    return len(expected) // 2


def load_lora_into(transformer, path):
    """Load a saved LoRA (diffusers format, `transformer.` prefix) into the existing adapter."""
    from safetensors.torch import load_file
    n_modules = check_lora_compatible(transformer, path)
    sd = {k.removeprefix("transformer."): v for k, v in load_file(path).items()}
    result = set_peft_model_state_dict(transformer, sd, adapter_name="default")
    assert not getattr(result, "unexpected_keys", []), result.unexpected_keys
    return n_modules


def rng_state():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


# ---------------------------------------------------------------- model

class RemovalModel(L.LightningModule):
    def __init__(self, cfg, pipe=None, cond=None, device=None, load_init_lora=True):
        """
        cfg:            the removal.yaml dict.
        pipe:           a FluxFillPipeline (built from cfg if None).
        cond:           dict(prompt_embeds, pooled_prompt_embeds) of the white reference; required.
        load_init_lora: load cfg["init_lora_path"] into the fresh adapter (False when resuming:
                        the LoRA then comes from the checkpoint).
        """
        super().__init__()
        assert cond is not None, "pass the white-reference conditioning (removal_eval.load_or_build_white_ref_cond)"
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tcfg = cfg["train"]
        self.cfg = cfg
        self.masked_fill = cfg["data"].get("masked_fill", "gray")
        self.train_guidance = float(tcfg.get("train_guidance", 1.0))
        self.optimizer_config = tcfg["optimizer"]
        self.model_config = cfg.get("model", {}) or {}
        dtype = getattr(torch, cfg["dtype"])

        self.flux_fill_pipe = pipe if pipe is not None else load_fill_pipeline(cfg, device)
        self.transformer = self.flux_fill_pipe.transformer
        self.flux_fill_pipe.vae.requires_grad_(False).eval()
        self.transformer.requires_grad_(False)
        self.transformer.gradient_checkpointing = bool(tcfg.get("gradient_checkpointing", False))

        self.transformer.add_adapter(LoraConfig(**tcfg["lora_config"]))
        self.lora_names = [n for n, p in self.named_parameters() if "lora_" in n]
        lora = set(self.lora_names)
        for n, p in self.named_parameters():
            p.requires_grad_(n in lora)
        self.n_lora_modules = len(self.lora_names) // 2
        self.init_lora_path = cfg.get("init_lora_path") if load_init_lora else None
        if self.init_lora_path:
            n = load_lora_into(self.transformer, self.init_lora_path)
            assert n == self.n_lora_modules, (n, self.n_lora_modules)

        self.register_buffer("cond_prompt_embeds", cond["prompt_embeds"].reshape(1, *cond["prompt_embeds"].shape[-2:]), persistent=False)
        self.register_buffer("cond_pooled", cond["pooled_prompt_embeds"].reshape(1, -1), persistent=False)

        # Resume checkpoints only carry the LoRA; frozen weights come from flux_fill_path.
        self.strict_loading = False
        self.to(device).to(dtype)
        self.flux_fill_pipe.vae.to(device, dtype)
        # Trainable LoRA weights as float32 master weights (see module docstring); peft casts each
        # LoRA layer's input to the LoRA weight dtype and the result back to the base dtype.
        self.lora_dtype = getattr(torch, tcfg.get("lora_dtype", "float32"))
        for n, p in self.named_parameters():
            if n in lora:
                p.data = p.data.to(self.lora_dtype)
        self.transformer.train()

    # ------------------------------------------------------------ training

    def on_after_batch_transfer(self, batch, dataloader_idx=0):
        if "src_tile" in batch:
            batch.update(make_diptych(batch["src_tile"], batch["gt_tile"], batch["mask_tile"], self.masked_fill))
        return batch

    def configure_optimizers(self):
        params = [p for n, p in self.named_parameters() if p.requires_grad]
        assert len(params) == len(self.lora_names), "only LoRA parameters may be trainable"
        opt = self.optimizer_config
        if opt["type"] == "Prodigy":
            return prodigyopt.Prodigy(params, **opt["params"])
        if opt["type"] == "AdamW":
            return torch.optim.AdamW(params, **opt["params"])
        if opt["type"] == "SGD":
            return torch.optim.SGD(params, **opt["params"])
        raise NotImplementedError(opt["type"])

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        value = loss.item()
        self.last_loss = value
        self.log_loss = value if not hasattr(self, "log_loss") else self.log_loss * 0.95 + value * 0.05
        return loss

    def step(self, batch):
        pipe = self.flux_fill_pipe
        result, src, mask = batch["result"], batch["src"], batch["mask"]
        B = result.shape[0]
        prompt_embeds = self.cond_prompt_embeds.expand(B, -1, -1)
        pooled = self.cond_pooled.expand(B, -1)
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

    # ------------------------------------------------------------ saving / resuming

    def save_lora(self, path, dtype=None):
        """pytorch_lora_weights.safetensors in diffusers format (pipe.load_lora_weights can read it),
        in the model dtype by default (bf16, like the InsertAnything LoRA; inference runs in bf16)."""
        dtype = dtype or getattr(torch, self.cfg["dtype"])
        sd = {k: v.detach().to(dtype) for k, v in get_peft_model_state_dict(self.transformer).items()}
        FluxFillPipeline.save_lora_weights(save_directory=path, transformer_lora_layers=sd, safe_serialization=True)

    def on_save_checkpoint(self, checkpoint):
        lora = set(self.lora_names)
        checkpoint["state_dict"] = {k: v for k, v in checkpoint["state_dict"].items() if k in lora}
        assert len(checkpoint["state_dict"]) == len(lora), "LoRA parameters missing from state_dict"
        checkpoint["removal"] = dict(
            rng=rng_state(),
            init_lora_path=self.init_lora_path or self.cfg.get("init_lora_path"),
            n_lora_modules=self.n_lora_modules,
            log_loss=getattr(self, "log_loss", None),
        )

    def on_load_checkpoint(self, checkpoint):
        lora = set(self.lora_names)
        expected = {n: tuple(p.shape) for n, p in self.named_parameters() if n in lora}
        got = {k: tuple(v.shape) for k, v in checkpoint["state_dict"].items()}
        missing, unexpected = sorted(set(expected) - set(got)), sorted(set(got) - set(expected))
        assert not missing and not unexpected, (
            f"resume checkpoint does not match the LoRA: {len(missing)} missing {missing[:2]}, "
            f"{len(unexpected)} unexpected {unexpected[:2]}")
        bad = [k for k in expected if expected[k] != got[k]]
        assert not bad, f"resume checkpoint LoRA shape mismatch: {bad[:3]}"
        extra = checkpoint.get("removal", {})
        if extra.get("log_loss") is not None:
            self.log_loss = extra["log_loss"]
        if extra.get("rng") is not None:
            set_rng_state(extra["rng"])


# ---------------------------------------------------------------- GPU check (milestone M2)

def gpu_check(cfg, out_dir, probe_batch=0, n_parity=2):
    """Real-weights checks; writes <out_dir>/m2_report.json and parity images."""
    from diffusers import FluxPriorReduxPipeline
    from PIL import Image

    from ..data.removal_data import build_train_data, build_val_data, eval_tiles, make_val_subsets
    from ..eval.removal_eval import build_white_ref_cond, run_removal, white_image

    os.makedirs(out_dir, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True  # float32 LoRA matmuls on tensor cores
    report, device, dtype = {}, "cuda", getattr(torch, cfg["dtype"])
    t0 = time.time()

    def save(stage):  # write after every stage, so a crash later keeps the earlier results
        report["stage_done"] = stage
        with open(os.path.join(out_dir, "m2_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"[m2] {stage} done ({time.time() - t0:.0f}s)", flush=True)

    # 1. White-reference conditioning: our builder vs the Redux pipeline call the inference scripts use.
    redux = FluxPriorReduxPipeline.from_pretrained(cfg["flux_redux_path"], torch_dtype=dtype).to(device)
    cond = build_white_ref_cond(redux, cfg["data"]["size"], device)
    prior = redux(white_image(cfg["data"]["size"]))
    report["cond_shape"] = list(cond["prompt_embeds"].shape)
    report["cond_vs_redux_call_max_abs"] = float((cond["prompt_embeds"].to(device) - prior.prompt_embeds).abs().max())
    report["pooled_vs_redux_call_max_abs"] = float((cond["pooled_prompt_embeds"].to(device) - prior.pooled_prompt_embeds).abs().max())
    del redux
    torch.cuda.empty_cache()
    save("1_cond")

    # 2. Model with the InsertAnything LoRA.
    model = RemovalModel(cfg, cond=cond, device=device)
    report["n_lora_modules"] = model.n_lora_modules
    report["n_lora_params_M"] = round(sum(p.numel() for n, p in model.named_parameters() if p.requires_grad) / 1e6, 1)
    report["load_s"] = round(time.time() - t0, 1)
    report["mem_after_load_GB"] = round(torch.cuda.memory_allocated() / 2**30, 1)
    save("2_load")

    # 3. Parity: our eval path vs a fresh pipeline + pipe.load_lora_weights + Redux call.
    val = build_val_data(cfg, make_val_subsets(cfg))
    items = [(name, ds, i) for name, ds in val.items() for i in range(min(n_parity, len(ds)))]
    model.transformer.eval()
    ours = []
    for name, ds, i in items:
        src, gt, obj = ds.load(i)
        tiles = eval_tiles(src, gt, obj, cfg["data"]["size"])
        ours.append((tiles, run_removal(model.flux_fill_pipe, cond, tiles["src"], tiles["mask"],
                                        cfg["val"]["num_inference_steps"], cfg["val"]["guidance_scale"], cfg["val"]["seed"])))
    del model
    torch.cuda.empty_cache()
    ref_pipe = load_fill_pipeline(cfg, device)
    ref_pipe.load_lora_weights(cfg["init_lora_path"])
    ref_cond = dict(prompt_embeds=prior.prompt_embeds, pooled_prompt_embeds=prior.pooled_prompt_embeds)
    diffs = []
    for (tiles, out), (name, ds, i) in zip(ours, items):
        ref = run_removal(ref_pipe, ref_cond, tiles["src"], tiles["mask"],
                          cfg["val"]["num_inference_steps"], cfg["val"]["guidance_scale"], cfg["val"]["seed"])
        diffs.append(dict(sample=f"{name}/{ds.items[i]['name']}",
                          max_abs=int(np.abs(out.astype(int) - ref.astype(int)).max()),
                          mean_abs=round(float(np.abs(out.astype(int) - ref.astype(int)).mean()), 3)))
        Image.fromarray(np.concatenate([tiles["src"], out, ref, tiles["gt"]], axis=1)).save(
            os.path.join(out_dir, f"parity_{name}_{ds.items[i]['name']}.jpg"))
    report["parity"] = diffs
    del ref_pipe
    torch.cuda.empty_cache()
    save("3_parity")

    # 4. Optional memory / speed probe: forward + backward + one Prodigy step, halving the batch on OOM.
    if probe_batch:
        model = RemovalModel(cfg, cond=cond, device=device)
        opt = model.configure_optimizers()
        concat, sampler = build_train_data(cfg)
        sampler.set_epoch(0)
        order = list(iter(sampler))
        report["probe"] = []
        bs = probe_batch
        while bs >= 1:
            try:
                batch = torch.utils.data.default_collate([concat[j] for j in order[:bs]])
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                batch = model.on_after_batch_transfer(batch)
                torch.cuda.reset_peak_memory_stats()
                times = []
                for _ in range(3):
                    torch.cuda.synchronize()
                    t1 = time.time()
                    loss = model.step(batch)
                    loss.backward()
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    times.append(time.time() - t1)
                report["probe"].append(dict(batch=bs, loss=float(loss), s_per_step=[round(x, 2) for x in times],
                                            peak_mem_GB=round(torch.cuda.max_memory_allocated() / 2**30, 1)))
                break
            except torch.cuda.OutOfMemoryError:
                report["probe"].append(dict(batch=bs, oom=True))
                del batch
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                bs //= 2

    save("4_probe" if probe_batch else "3_parity")
    print(json.dumps(report, indent=2))


def main():
    from ..data.removal_data import load_config, repo_path
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="experiments/config/removal.yaml")
    p.add_argument("--gpu_check", action="store_true")
    p.add_argument("--probe_batch", type=int, default=0, help="also time one train step at this batch size")
    p.add_argument("--out", default="runs_removal/_checks/m2")
    args = p.parse_args()
    if args.gpu_check:
        gpu_check(load_config(repo_path(args.config)), repo_path(args.out), args.probe_batch)


if __name__ == "__main__":
    main()
