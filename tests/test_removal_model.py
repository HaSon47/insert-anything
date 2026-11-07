"""CPU tests for the removal model (milestone M2), run from the repo root:

    python -m tests.test_removal_model            # all tests
    python -m tests.test_removal_model -k forward  # tests whose name contains "forward"

Most tests use a tiny random-weight FLUX-Fill with the real layout (384 input channels, 16-channel
VAE with 8x downsampling, same LoRA target regex). Two use the real checkpoints without loading the
12B transformer: LoRA keys against the real config (empty weights) and the Redux white-reference
conditioning (Redux runs on CPU).
"""

import argparse
import copy
import os
import tempfile
import time
import traceback

import numpy as np
import torch

from src.data.removal_data import build_train_data, eval_tiles, load_config, make_diptych, repo_path
from src.eval.removal_eval import run_removal
from src.models.removal_model import RemovalModel, check_lora_compatible, load_lora_into, lora_file_shapes
from src.models.transformer import tranformer_forward

CFG = load_config(repo_path("experiments/config/removal.yaml"))
TINY = 64  # tile size; diptych 64 x 128 -> latents 8 x 16 -> 32 tokens


def tiny_cfg(**train):
    cfg = copy.deepcopy(CFG)
    cfg["dtype"] = "float32"
    cfg["init_lora_path"] = None
    cfg["data"]["size"] = TINY
    cfg["train"]["lora_config"]["r"] = cfg["train"]["lora_config"]["lora_alpha"] = 4
    cfg["train"]["batch_size"] = 2
    cfg["train"].update(train)
    return cfg


def tiny_pipe(seed=0):
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxFillPipeline, FluxTransformer2DModel
    torch.manual_seed(seed)
    transformer = FluxTransformer2DModel(
        patch_size=1, in_channels=384, out_channels=64, num_layers=2, num_single_layers=2,
        attention_head_dim=16, num_attention_heads=2, joint_attention_dim=32, pooled_projection_dim=32,
        guidance_embeds=True, axes_dims_rope=(4, 6, 6))
    vae = AutoencoderKL(
        in_channels=3, out_channels=3, latent_channels=16, block_out_channels=(8, 8, 8, 8),
        down_block_types=["DownEncoderBlock2D"] * 4, up_block_types=["UpDecoderBlock2D"] * 4,
        layers_per_block=1, norm_num_groups=4, sample_size=TINY, scaling_factor=0.3611, shift_factor=0.1159,
        use_quant_conv=False, use_post_quant_conv=False)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(CFG["flux_fill_path"], subfolder="scheduler")
    pipe = FluxFillPipeline(scheduler=scheduler, vae=vae, text_encoder=None, tokenizer=None,
                            text_encoder_2=None, tokenizer_2=None, transformer=transformer)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def tiny_cond(seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(prompt_embeds=torch.randn(1, 8, 32, generator=g), pooled_prompt_embeds=torch.randn(1, 32, generator=g))


def tiny_model(seed=0, **train):
    model = RemovalModel(tiny_cfg(**train), pipe=tiny_pipe(seed), cond=tiny_cond(seed), device="cpu")
    randomize_lora_B(model, seed)  # gaussian init leaves lora_B = 0, which would hide LoRA bugs
    return model


def randomize_lora_B(model, seed):
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.02)


def real_batch(cfg, n=2):
    concat, sampler = build_train_data(cfg)
    sampler.set_epoch(0)
    torch.manual_seed(0)
    items = [concat[i] for i in list(iter(sampler))[:n]]
    return torch.utils.data.default_collate(items)


def lora_values(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if n in set(model.lora_names)}


# ---------------------------------------------------------------- tests

def test_custom_forward_matches_diffusers():
    """src/models/transformer.py + block.py (written for diffusers 0.32) must equal diffusers' own
    FluxTransformer2DModel.forward in the installed version, with an active LoRA, in eval mode and in
    training mode with gradient checkpointing."""
    model = tiny_model()
    tr = model.transformer
    g = torch.Generator().manual_seed(3)
    B, n_img, n_txt = 2, 32, 8
    kw = dict(
        hidden_states=torch.randn(B, n_img, 384, generator=g),
        encoder_hidden_states=torch.randn(B, n_txt, 32, generator=g),
        pooled_projections=torch.randn(B, 32, generator=g),
        timestep=torch.rand(B, generator=g),
        guidance=torch.full((B,), 1.0),
        img_ids=torch.randn(n_img, 3, generator=g),
        txt_ids=torch.zeros(n_txt, 3),
    )
    for train_mode in (False, True):
        tr.train(train_mode)
        tr.gradient_checkpointing = train_mode
        with torch.no_grad():
            ref = tr(**kw, return_dict=False)[0]
        ours = tranformer_forward(tr, model_config={}, joint_attention_kwargs=None, return_dict=False, **kw)[0]
        err = (ours - ref).abs().max().item()
        assert err < 1e-5, f"custom forward differs from diffusers (train={train_mode}): {err}"
        print(f"    train={train_mode}: max |custom - diffusers| = {err:.2e}")


def test_trainable_params_are_lora_only():
    model = tiny_model()
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)
    assert len(trainable) == len(model.lora_names) == 2 * model.n_lora_modules
    opt = model.configure_optimizers()
    assert sum(len(gr["params"]) for gr in opt.param_groups) == len(trainable)
    print(f"    {model.n_lora_modules} LoRA modules, {len(trainable)} trainable tensors, optimizer {type(opt).__name__}")


def test_step_backward_on_real_batch():
    """Full training step on real data tiles: diptych on device, VAE encode, mask packing,
    custom forward, MSE; gradients reach every LoRA tensor and nothing else."""
    model = tiny_model()
    batch = model.on_after_batch_transfer(real_batch(model.cfg))
    S = TINY
    assert batch["src"].shape == batch["mask"].shape == batch["result"].shape == (2, 3, S, 2 * S)
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss), loss
    loss.backward()
    grads = {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
    no_grad = [n for n, gr in grads.items() if gr is None or not torch.isfinite(gr).all()]
    assert not no_grad, f"{len(no_grad)} LoRA tensors without a finite grad, e.g. {no_grad[:2]}"
    zero_A = [n for n, gr in grads.items() if "lora_A" in n and gr.abs().sum() == 0]
    assert not zero_A, f"lora_A tensors with zero grad: {zero_A[:3]}"
    frozen_with_grad = [n for n, p in model.named_parameters() if not p.requires_grad and p.grad is not None]
    assert not frozen_with_grad
    print(f"    loss {loss.item():.4f}, t {model.last_t:.3f}, grads on all {len(grads)} LoRA tensors")


def test_training_reduces_loss_on_fixed_batch():
    """A few optimizer steps on one fixed batch and fixed noise must lower the loss (sanity check
    that the LoRA, the target and the optimizer are wired correctly)."""
    model = tiny_model(optimizer={"type": "AdamW", "params": {"lr": 1e-2}})
    opt = model.configure_optimizers()
    batch = model.on_after_batch_transfer(real_batch(model.cfg))
    losses = []
    for _ in range(15):
        torch.manual_seed(0)
        loss = model.step(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.8 * losses[0], losses
    print(f"    loss {losses[0]:.4f} -> {losses[-1]:.4f} in 15 steps")


def test_lora_save_load_roundtrip():
    """save_lora writes a diffusers LoRA; load_lora_into (used for init_lora_path) restores it exactly,
    and pipe.load_lora_weights (the inference scripts' loader) gives the same transformer output."""
    model = tiny_model(seed=0)
    with tempfile.TemporaryDirectory() as d:
        model.save_lora(d)
        path = os.path.join(d, "pytorch_lora_weights.safetensors")
        shapes = lora_file_shapes(path)
        assert len(shapes) == 2 * model.n_lora_modules
        other = tiny_model(seed=0)
        randomize_lora_B(other, seed=7)
        assert check_lora_compatible(other.transformer, path) == model.n_lora_modules
        load_lora_into(other.transformer, path)
        a, b = lora_values(model), lora_values(other)
        assert all(torch.equal(a[k], b[k]) for k in a)

        pipe = tiny_pipe(seed=0)
        pipe.load_lora_weights(path)
        g = torch.Generator().manual_seed(5)
        kw = dict(hidden_states=torch.randn(1, 32, 384, generator=g), encoder_hidden_states=torch.randn(1, 8, 32, generator=g),
                  pooled_projections=torch.randn(1, 32, generator=g), timestep=torch.rand(1, generator=g),
                  guidance=torch.ones(1), img_ids=torch.randn(32, 3, generator=g), txt_ids=torch.zeros(8, 3))
        model.transformer.eval()
        with torch.no_grad():
            err = (model.transformer(**kw, return_dict=False)[0] - pipe.transformer(**kw, return_dict=False)[0]).abs().max().item()
        assert err < 1e-5, err

        # a LoRA of a different rank must be rejected, not silently half-loaded
        wrong = tiny_model(seed=0, lora_config={**tiny_cfg()["train"]["lora_config"], "r": 8, "lora_alpha": 8})
        try:
            check_lora_compatible(wrong.transformer, path)
        except AssertionError as e:
            msg = str(e)
        else:
            raise AssertionError("rank-8 adapter accepted a rank-4 LoRA file")
    print(f"    save -> load exact; pipe.load_lora_weights output diff {err:.1e}; wrong rank rejected ({msg[:40]}...)")


def test_checkpoint_hooks_lora_only_and_rng():
    """on_save_checkpoint keeps only the LoRA + RNG; on_load_checkpoint rejects mismatches and restores RNG."""
    model = tiny_model(seed=0)
    ckpt = {"state_dict": model.state_dict()}
    n_all = len(ckpt["state_dict"])
    model.on_save_checkpoint(ckpt)
    assert set(ckpt["state_dict"]) == set(model.lora_names), "state_dict must be LoRA only"
    expected_next = torch.rand(3)          # what the RNG produces right after saving
    saved = {k: v.clone() for k, v in ckpt["state_dict"].items()}

    other = tiny_model(seed=0)
    randomize_lora_B(other, seed=9)
    torch.rand(10)                          # move the RNG on
    other.on_load_checkpoint(ckpt)
    assert torch.equal(torch.rand(3), expected_next), "RNG state not restored"
    other.load_state_dict(ckpt["state_dict"], strict=other.strict_loading)
    b = lora_values(other)
    assert all(torch.equal(saved[k], b[k]) for k in saved)

    broken = {"state_dict": dict(list(saved.items())[1:]), "removal": ckpt["removal"]}
    try:
        other.on_load_checkpoint(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("checkpoint with a missing LoRA tensor accepted")
    print(f"    state_dict {n_all} -> {len(saved)} LoRA tensors; RNG restored; missing key rejected")


def test_run_removal_inference_path():
    """The shared eval path through FluxFillPipeline (white diptych, cached conditioning)."""
    model = tiny_model()
    model.transformer.eval()
    from src.data.removal_data import ObjectStitchRemoval
    ds = ObjectStitchRemoval(CFG["data"]["objectstitch_root"], "val", tiny_cfg()["data"], rounds=(1,), train=False)
    src, gt, obj = ds.load(0)
    tiles = eval_tiles(src, gt, obj, TINY)
    cond = dict(prompt_embeds=model.cond_prompt_embeds, pooled_prompt_embeds=model.cond_pooled)
    a = run_removal(model.flux_fill_pipe, cond, tiles["src"], tiles["mask"], num_inference_steps=2, seed=1)
    b = run_removal(model.flux_fill_pipe, cond, tiles["src"], tiles["mask"], num_inference_steps=2, seed=1)
    assert a.shape == (TINY, TINY, 3) and a.dtype == np.uint8
    assert np.array_equal(a, b), "same seed must give the same output"
    print(f"    output tile {a.shape}, deterministic for a fixed seed")


def test_gray_fill_matches_pipeline_masking():
    """Training's masked_fill=gray must give the exact image FluxFillPipeline VAE-encodes as its
    masked image at inference: init_image * (1 - mask), with init_image normalized to [-1, 1]."""
    pipe = tiny_pipe()
    g = torch.Generator().manual_seed(2)
    src_tile = torch.randint(0, 256, (1, 3, TINY, TINY), generator=g, dtype=torch.uint8)
    mask_tile = (torch.rand(1, 1, TINY, TINY, generator=g) > 0.7).to(torch.uint8)
    d = make_diptych(src_tile, src_tile, mask_tile, "gray")
    ours = pipe.image_processor.preprocess(d["src"])               # what training encodes
    image = torch.cat([torch.ones(1, 3, TINY, TINY), src_tile.float() / 255], dim=-1)
    init = pipe.image_processor.preprocess(image)                  # what the pipeline gets
    masked = init * (1 - pipe.mask_processor.preprocess(d["mask"]))  # pipeline_flux_fill.py masking
    err = (ours - masked).abs().max().item()
    assert err < 1e-6, err
    print(f"    gray-filled training input == pipeline's masked image (max diff {err:.1e})")


def test_real_lora_matches_real_config():
    """The real InsertAnything LoRA against the real FLUX-Fill transformer config with removal.yaml's
    lora_config, on empty weights (nothing large is loaded)."""
    from accelerate import init_empty_weights
    from diffusers import FluxTransformer2DModel
    from peft import LoraConfig
    config = FluxTransformer2DModel.load_config(os.path.join(CFG["flux_fill_path"], "transformer"))
    with init_empty_weights():
        tr = FluxTransformer2DModel.from_config(config)
    tr.add_adapter(LoraConfig(**CFG["train"]["lora_config"]))
    n = check_lora_compatible(tr, CFG["init_lora_path"])
    assert n == 343, n
    shapes = lora_file_shapes(CFG["init_lora_path"])
    ranks = {s[0] for k, s in shapes.items() if "lora_A" in k}
    print(f"    {n} modules, 0 missing / 0 unexpected keys, rank {ranks}")


def test_white_ref_cond_real_redux():
    """Real FLUX.1-Redux on CPU: our cached conditioning equals the Redux pipeline call the inference
    scripts use, with shape [1, 512 + 729, 4096]."""
    from diffusers import FluxPriorReduxPipeline
    from src.eval.removal_eval import build_white_ref_cond, white_image
    redux = FluxPriorReduxPipeline.from_pretrained(CFG["flux_redux_path"], torch_dtype=torch.float32)
    cond = build_white_ref_cond(redux, CFG["data"]["size"], "cpu")
    prior = redux(white_image(CFG["data"]["size"]))
    assert tuple(cond["prompt_embeds"].shape) == (1, 1241, 4096), cond["prompt_embeds"].shape
    assert tuple(cond["pooled_prompt_embeds"].shape) == (1, 768)
    e1 = (cond["prompt_embeds"] - prior.prompt_embeds).abs().max().item()
    e2 = (cond["pooled_prompt_embeds"] - prior.pooled_prompt_embeds).abs().max().item()
    assert e1 == 0 and e2 == 0, (e1, e2)
    assert cond["prompt_embeds"][0, :512].abs().max() == 0 and cond["prompt_embeds"][0, 512:].abs().max() > 0
    print(f"    shape {tuple(cond['prompt_embeds'].shape)}; identical to FluxPriorReduxPipeline(white)")


def _prodigy_run_bf16_base(lora_dtype, steps=40):
    """Tiny model with a bf16 base and a LoRA at the trained InsertAnything magnitude (|w| ~ 4e-3);
    returns (fraction of LoRA elements changed, Prodigy d) after `steps` optimizer steps."""
    cfg = tiny_cfg(lora_dtype=lora_dtype)
    cfg["dtype"] = "bfloat16"
    model = RemovalModel(cfg, pipe=tiny_pipe(0), cond=tiny_cond(0), device="cpu")
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in set(model.lora_names):
                p.copy_(torch.randn(p.shape, generator=g) * 5e-3)
    before = lora_values(model)
    opt = model.configure_optimizers()
    batch = model.on_after_batch_transfer(real_batch(model.cfg))
    torch.manual_seed(0)
    for _ in range(steps):
        loss = model.step(batch)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    after = lora_values(model)
    changed = sum((after[k] != before[k]).sum().item() for k in before) / sum(v.numel() for v in before.values())
    return changed, opt.param_groups[0]["d"]


def test_bf16_base_trains_with_fp32_lora_master_weights():
    """Regression test for the M3 smoke run: with a bf16 base, bf16 LoRA weights starting from a trained
    LoRA do not move (Prodigy's 1e-6 updates are below bf16 spacing) and d never leaves d0; float32
    master weights train. Exports stay bf16."""
    cfg = tiny_cfg()
    cfg["dtype"] = "bfloat16"
    model = RemovalModel(cfg, pipe=tiny_pipe(0), cond=tiny_cond(0), device="cpu")
    lora = set(model.lora_names)
    assert all(p.dtype == torch.float32 for n, p in model.named_parameters() if n in lora)
    assert all(p.dtype == torch.bfloat16 for n, p in model.transformer.named_parameters() if "lora_" not in n)
    with tempfile.TemporaryDirectory() as d:
        model.save_lora(d)
        from safetensors.torch import load_file
        assert {v.dtype for v in load_file(os.path.join(d, "pytorch_lora_weights.safetensors")).values()} == {torch.bfloat16}
    changed16, d16 = _prodigy_run_bf16_base("bfloat16")
    changed32, d32 = _prodigy_run_bf16_base("float32")
    print(f"    bf16 LoRA: {changed16:.1%} of elements changed, d {d16:.2e};  fp32 LoRA: {changed32:.1%}, d {d32:.2e}")
    assert changed32 > 0.9, changed32
    assert d32 > 2e-6, f"fp32 master weights: Prodigy d did not grow ({d32})"  # ~4e-6 after 40 steps
    assert d16 == 1e-6 and changed16 < 0.5 * changed32, (d16, changed16)


TESTS = [v for k, v in list(globals().items()) if k.startswith("test_")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-k", default="", help="run tests whose name contains this")
    args = p.parse_args()
    failed = 0
    for t in TESTS:
        if args.k not in t.__name__:
            continue
        t0 = time.time()
        try:
            t()
            print(f"PASS {t.__name__} ({time.time() - t0:.1f}s)")
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__} ({time.time() - t0:.1f}s)")
            traceback.print_exc()
    print("all passed" if not failed else f"{failed} failed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
