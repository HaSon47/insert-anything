"""CPU tests for the addition pipeline (milestone M2), run from the repo root:

    python -m tests.test_addition            # all tests
    python -m tests.test_addition -k mask    # tests whose name contains "mask"

Most tests use a tiny random-weight FLUX-Fill with the real conditioning sizes (4096-d text tokens, 768-d
pooled) and a stub Redux whose tokens depend on the image. Two use real checkpoints without loading the
12B transformer: the LoRA keys against the real config (empty weights) and the Redux conditioning (on CPU).
"""

import argparse
import copy
import os
import tempfile
import time
import traceback
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from PIL import Image

from src.data.addition_data import (
    AdditionDataset,
    box_mask,
    build_eval_data,
    build_ref_tile,
    build_train_data,
    is_solid_rectangle,
    load_config,
    make_addition_diptych,
    make_eval_subsets,
    repo_path,
    train_tiles,
)
from src.eval.addition_eval import psnr_bg, reference_conditioning, run_addition
from src.models.addition_model import AdditionModel
from src.models.removal_model import check_lora_compatible, load_lora_into, lora_file_shapes
from src.train.addition_callbacks import AdditionTrainingCallback

CFG = load_config(repo_path("experiments/config/addition.yaml"))
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
        attention_head_dim=16, num_attention_heads=2, joint_attention_dim=4096, pooled_projection_dim=768,
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


class TinyRedux:
    """encode_image + image_embedder of FluxPriorReduxPipeline: 4 tokens, deterministic in the image."""

    def __init__(self, seed=0):
        self.proj = torch.randn(4, 4096, generator=torch.Generator().manual_seed(seed)) * 0.05

    def encode_image(self, image, device, num_images_per_prompt):
        a = np.asarray(image.resize((2, 2), Image.BILINEAR), dtype=np.float32) / 255
        return torch.from_numpy(a).reshape(1, 4, 3).to(device)

    def image_embedder(self, x):
        return SimpleNamespace(image_embeds=x.sum(-1, keepdim=True) * self.proj[None].to(x.device))


def randomize_lora_B(model, seed):
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.02)


def tiny_model(seed=0, **train):
    model = AdditionModel(tiny_cfg(**train), redux=TinyRedux(seed), pipe=tiny_pipe(seed), device="cpu")
    randomize_lora_B(model, seed)  # gaussian init leaves lora_B = 0, which would hide LoRA bugs
    return model


def real_batch(cfg, n=2):
    ds, sampler = build_train_data(cfg)
    torch.manual_seed(0)
    return torch.utils.data.default_collate([ds[i] for i in list(iter(sampler))[:n]])


def lora_values(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if n in set(model.lora_names)}


# ---------------------------------------------------------------- data

def test_box_mask_is_white_rectangle_matching_inference():
    from inference_fsc_final import create_mask_from_bbox
    H, W = 90, 120
    for box in ([10, 20, 40, 50], [0, 0, 5, 5], [100, 70, 119, 89], [100, 70, 120, 90]):  # the last reaches past the edge
        ours = box_mask((H, W), box)
        pil = (np.array(create_mask_from_bbox((W, H), box)) > 128).astype(np.uint8)
        assert np.array_equal(ours, pil), box
        assert is_solid_rectangle(ours), box
    big = box_mask((H, W), [10, 20, 40, 50], 1.2)
    assert is_solid_rectangle(big) and np.all(big[box_mask((H, W), [10, 20, 40, 50]) > 0] == 1) and big.sum() > 31 * 31
    dil = cv2.dilate(big, np.ones((7, 7), np.uint8), iterations=2)
    assert is_solid_rectangle(dil), "the inference dilation must keep the mask a rectangle"
    assert not is_solid_rectangle(np.array([[1, 0], [1, 1]], np.uint8))
    print("    equals create_mask_from_bbox, including a box reaching past the image edge; scale and dilation keep it a rectangle")


def test_real_train_masks_are_rectangles_containing_the_box():
    ds = AdditionDataset.from_split(tiny_cfg()["data"], "train", train=True)
    rng = np.random.default_rng(0)
    scales = []
    for i in rng.choice(len(ds), 25, replace=False):
        image, gt, ref_mask = ds.load(int(i))
        t = train_tiles(image, gt, ref_mask, ds.items[int(i)], ds.dcfg, rng, keep_full_mask=True)
        assert is_solid_rectangle(t["mask"]) and is_solid_rectangle(t["mask_full"])
        assert not np.any(box_mask(image.shape[:2], ds.items[int(i)]["ins_box"]) & (1 - t["mask_full"]))
        scales.append(float(t["aug"]))
    assert 1.0 <= min(scales) and max(scales) <= 1.2 and len(set(scales)) > 1, scales
    print(f"    25 samples: solid rectangles containing the box; box scale in [{min(scales)}, {max(scales)}]")


def test_reference_tile_inclusive_crop_white_background():
    """Synthetic image: a red square cut out by its mask, with blue background and a green distractor."""
    image = np.zeros((100, 100, 3), np.uint8)
    image[:] = (0, 0, 255)
    image[30:51, 40:61] = (255, 0, 0)  # 21 x 21 object: inclusive box [40, 30, 60, 50]
    image[0:5, 0:5] = (0, 255, 0)
    mask = np.zeros((100, 100), np.uint8)
    mask[30:51, 40:61] = 1
    tile = build_ref_tile(image, mask, [40, 30, 60, 50], 1.0, 21)
    assert np.all(tile == np.array([255, 0, 0])), "ratio 1: the inclusive crop must hold only the object, last row and column included"
    tile = build_ref_tile(image, mask, [35, 25, 65, 55], 1.0, 31)   # box larger than the object
    assert np.array_equal(tile[0, 0], [255, 255, 255]) and np.array_equal(tile[15, 15], [255, 0, 0])
    assert not np.any(np.all(tile == np.array([0, 0, 255]), axis=-1)), "background must be white"
    wide = build_ref_tile(image, mask, [40, 30, 60, 50], 2.0, 42)    # white margin of ratio 2
    assert np.array_equal(wide[0, 0], [255, 255, 255]) and np.array_equal(wide[21, 21], [255, 0, 0])
    print("    inclusive box crop, background white, white margin by ratio")


def test_gray_fill_matches_pipeline_masking():
    """Training's gray-filled input must be the image FluxFillPipeline VAE-encodes as its masked image:
    init_image * (1 - mask), with init_image normalized to [-1, 1]."""
    pipe = tiny_pipe()
    g = torch.Generator().manual_seed(2)
    ref = torch.randint(0, 256, (1, 3, TINY, TINY), generator=g, dtype=torch.uint8)
    src = torch.randint(0, 256, (1, 3, TINY, TINY), generator=g, dtype=torch.uint8)
    mask = torch.zeros(1, 1, TINY, TINY, dtype=torch.uint8)
    mask[..., 10:30, 20:50] = 1
    d = make_addition_diptych(ref, src, src, mask, "gray")
    ours = pipe.image_processor.preprocess(d["src"])
    init = pipe.image_processor.preprocess(torch.cat([ref.float() / 255, src.float() / 255], dim=-1))
    masked = init * (1 - pipe.mask_processor.preprocess(d["mask"]))
    err = (ours - masked).abs().max().item()
    assert err < 1e-6, err
    assert torch.equal(d["src"][..., :TINY], ref.float() / 255) and torch.equal(d["result"][..., :TINY], ref.float() / 255)
    assert torch.all(d["mask"][..., :TINY] == 0)
    print(f"    gray-filled input == pipeline's masked image (max diff {err:.1e}); left half is the reference, unmasked")


def test_eval_subsets_are_fixed_and_balanced():
    a, b = make_eval_subsets(CFG), make_eval_subsets(CFG)
    assert a == b and len(a["val"]) == 12 and len(a["train"]) == 4
    data = build_eval_data(CFG, a)
    for group, ds in data.items():
        suffix = [it["suffix"] for it in ds.items]
        assert suffix.count(1) == suffix.count(2), group
        assert len({it["image_id"] for it in ds.items}) == len(ds), group
        assert not ds.train
    assert not {it["image_id"] for it in data["val"].items} & {it["image_id"] for it in data["train"].items}
    print(f"    deterministic; 12 val + 4 train, suffixes balanced, distinct source images; {len({it['class_name'] for it in data['val'].items})} val classes")


def test_psnr_bg_ignores_masked_and_padded_pixels():
    gt = np.full((8, 8, 3), 100, np.uint8)
    pred = gt.copy()
    mask = np.zeros((8, 8), np.uint8)
    mask[2:4, 2:4] = 1
    valid = np.ones((8, 8), bool)
    valid[7] = False
    assert psnr_bg(pred, gt, mask, valid) == float("inf")
    pred[2:4, 2:4] = 0      # only masked pixels wrong
    pred[7] = 255           # only padding wrong
    assert psnr_bg(pred, gt, mask, valid) == float("inf"), "masked and padded pixels must not count"
    assert psnr_bg(pred, gt, mask) != float("inf"), "padding counts when no valid area is given"
    pred[0, 0] = gt[0, 0] + 10   # a scene pixel wrong by 10 on 3 channels
    n = int(((mask == 0) & valid).sum())
    expected = 10 * np.log10(255.0 ** 2 / (100 / n))
    assert abs(psnr_bg(pred, gt, mask, valid) - expected) < 1e-9
    print(f"    masked and padded pixels ignored; value matches the formula ({expected:.2f} dB)")


# ---------------------------------------------------------------- model

def test_trainable_params_are_lora_only():
    model = tiny_model()
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable) and len(trainable) == len(model.lora_names)
    assert not [n for n, _ in model.named_parameters() if "flux_redux" in n or "image_encoder" in n], "Redux must not be a submodule"
    assert not [k for k in model.state_dict() if "redux" in k or "image_e" in k]
    opt = model.configure_optimizers()
    assert sum(len(gr["params"]) for gr in opt.param_groups) == len(trainable)
    print(f"    {model.n_lora_modules} LoRA modules; Redux not a submodule; optimizer {type(opt).__name__}")


def test_step_backward_on_real_batch():
    """Full step on real tiles: gradients reach every LoRA tensor and nothing else."""
    model = tiny_model()
    batch = model.on_after_batch_transfer(real_batch(model.cfg))
    assert batch["src"].shape == batch["mask"].shape == batch["result"].shape == (2, 3, TINY, 2 * TINY)
    assert torch.equal(batch["src"][..., :TINY], batch["ref_tile"].float() / 255)
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss), loss
    loss.backward()
    grads = {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
    missing = [n for n, gr in grads.items() if gr is None or not torch.isfinite(gr).all()]
    assert not missing, missing[:3]
    assert not [n for n, gr in grads.items() if "lora_A" in n and gr.abs().sum() == 0]
    assert not [n for n, p in model.named_parameters() if not p.requires_grad and p.grad is not None]
    print(f"    loss {loss.item():.4f}, t {model.last_t:.3f}, grads on all {len(grads)} LoRA tensors")


def test_conditioning_is_per_sample_and_changes_the_loss():
    """Different references must give different conditioning in the same batch, and it must reach the model."""
    model = tiny_model()
    batch = real_batch(model.cfg)
    assert not torch.equal(batch["ref_tile"][0], batch["ref_tile"][1])
    prompt, pooled = model.reference_batch(batch["ref_tile"])
    assert prompt.shape == (2, 516, 4096) and pooled.shape == (2, 768)
    assert prompt[:, :512].abs().max() == 0, "T5 tokens must stay zero"
    assert pooled.abs().max() == 0, "pooled must stay zero"
    assert (prompt[0, 512:] != prompt[1, 512:]).any()
    one, _ = reference_conditioning(model.flux_redux, batch["ref_tile"][1].permute(1, 2, 0).numpy(), "cpu", torch.float32)
    assert torch.equal(one[0], prompt[1]), "batch order must not change a sample's conditioning"
    a = model.on_after_batch_transfer(dict(batch))
    swapped = dict(a, ref_tile=a["ref_tile"].flip(0))
    torch.manual_seed(1)
    l1 = model.step(a).item()
    torch.manual_seed(1)
    l2 = model.step(swapped).item()
    assert l1 != l2, "swapping references must change the model's output"
    print(f"    per-sample tokens, zero T5 and pooled; swapping references changes the loss ({l1:.5f} vs {l2:.5f})")


def test_training_reduces_loss_on_fixed_batch():
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


def test_lora_save_load_and_init_from_file():
    """The pretrained-LoRA path: a saved LoRA loads through init_lora_path exactly, and pipe.load_lora_weights
    (the inference scripts' loader) gives the same transformer output."""
    model = tiny_model(seed=0)
    with tempfile.TemporaryDirectory() as d:
        model.save_lora(d)
        path = os.path.join(d, "pytorch_lora_weights.safetensors")
        assert len(lora_file_shapes(path)) == 2 * model.n_lora_modules
        cfg = tiny_cfg()
        cfg["init_lora_path"] = path
        other = AdditionModel(cfg, redux=TinyRedux(0), pipe=tiny_pipe(0), device="cpu")
        a, b = lora_values(model), lora_values(other)
        assert all(torch.equal(a[k], b[k]) for k in a), "init_lora_path did not load exactly"
        skipped = AdditionModel(cfg, redux=TinyRedux(0), pipe=tiny_pipe(0), device="cpu", load_init_lora=False)
        c = lora_values(skipped)
        assert any(not torch.equal(a[k], c[k]) for k in a), "load_init_lora=False must not load the file"
        pipe = tiny_pipe(seed=0)
        pipe.load_lora_weights(path)
        g = torch.Generator().manual_seed(5)
        kw = dict(hidden_states=torch.randn(1, 32, 384, generator=g), encoder_hidden_states=torch.randn(1, 8, 4096, generator=g),
                  pooled_projections=torch.randn(1, 768, generator=g), timestep=torch.rand(1, generator=g),
                  guidance=torch.ones(1), img_ids=torch.randn(32, 3, generator=g), txt_ids=torch.zeros(8, 3))
        model.transformer.eval()
        with torch.no_grad():
            err = (model.transformer(**kw, return_dict=False)[0] - pipe.transformer(**kw, return_dict=False)[0]).abs().max().item()
        assert err < 1e-5, err
    print(f"    init_lora_path loads exactly and is skipped on resume; pipe.load_lora_weights output diff {err:.1e}")


def test_run_addition_inference_path():
    model = tiny_model()
    model.transformer.eval()
    ds = AdditionDataset.from_split(tiny_cfg()["data"], "val", train=False)
    t = ds.tiles(0)
    run = lambda ref, seed=1: run_addition(model.flux_fill_pipe, model.flux_redux, ref, t["src"], t["mask"], num_inference_steps=2, seed=seed)
    a, b = run(t["ref"]), run(t["ref"])
    assert a.shape == (TINY, TINY, 3) and a.dtype == np.uint8
    assert np.array_equal(a, b), "same seed must give the same output"
    assert not np.array_equal(a, run(255 - t["ref"])), "a different reference must change the output"
    print(f"    output tile {a.shape}, deterministic for a fixed seed, follows the reference")


def test_every_save_is_followed_by_evaluation():
    """The requirement: evaluation at every checkpoint save, including the final one."""
    cfg = tiny_cfg()
    calls = []

    class Spy(AdditionTrainingCallback):
        def validate(self, trainer, pl_module, step):
            calls.append(step)

    with tempfile.TemporaryDirectory() as d:
        cb = Spy(cfg, d)
        assert cb.val_interval == 0, "evaluation must only be driven by saves"
        model = tiny_model()
        cb.save_lora(model, 500)
        assert calls == [500] and os.path.exists(os.path.join(d, "ckpt", "500", "pytorch_lora_weights.safetensors"))
        cb.state["last_save_step"] = 500
        trainer = SimpleNamespace(global_step=730, current_epoch=0)
        cb.save_resume = lambda tr: None
        cb.on_train_end(trainer, model)
        assert calls == [500, 730], calls
    print("    a save evaluates at that step; the final export at a non-multiple step does too")


# ---------------------------------------------------------------- real checkpoints, nothing large loaded

def test_real_lora_matches_real_config():
    from accelerate import init_empty_weights
    from diffusers import FluxTransformer2DModel
    from peft import LoraConfig
    config = FluxTransformer2DModel.load_config(os.path.join(CFG["flux_fill_path"], "transformer"))
    with init_empty_weights():
        tr = FluxTransformer2DModel.from_config(config)
    tr.add_adapter(LoraConfig(**CFG["train"]["lora_config"]))
    n = check_lora_compatible(tr, CFG["init_lora_path"])
    assert n == 343, n
    print(f"    {n} modules, 0 missing / 0 unexpected keys")


def test_conditioning_real_redux():
    """Real FLUX.1-Redux on CPU, real references: identical to the Redux pipeline call of the inference scripts."""
    from diffusers import FluxPriorReduxPipeline
    redux = FluxPriorReduxPipeline.from_pretrained(CFG["flux_redux_path"], torch_dtype=torch.float32)
    ds = AdditionDataset.from_split(CFG["data"], "val", train=False)
    refs = [ds.tiles(i)["ref"] for i in (0, 1)]
    outs = []
    for ref in refs:
        prompt, pooled = reference_conditioning(redux, ref, "cpu", torch.float32)
        prior = redux(Image.fromarray(ref))
        assert tuple(prompt.shape) == (1, 1241, 4096) and tuple(pooled.shape) == (1, 768), prompt.shape
        assert (prompt - prior.prompt_embeds).abs().max() == 0 and (pooled - prior.pooled_prompt_embeds).abs().max() == 0
        outs.append(prompt)
    assert (outs[0] != outs[1]).any()
    print(f"    shape {tuple(outs[0].shape)}; identical to FluxPriorReduxPipeline(reference)")


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
