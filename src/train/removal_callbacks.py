"""Training callback for removal fine-tuning (removal_training_plan.md, sections 4.2 and 5).

All step numbers are trainer.global_step (optimizer steps). Files under <run_dir>:
    train_metrics.csv          one row per optimizer step (loss, EMA, t, grad norms, Prodigy d, s/step, memory)
    batch_names.csv            dataset/name/aug of every sample, per step (for debugging and the resume test)
    ckpt/<step>/pytorch_lora_weights.safetensors   LoRA export every save_interval steps (and at the end)
    resume/last.ckpt, prev.ckpt                    full resume state after the last batch of every epoch, written atomically
    val/subsets.json           fixed validation names (written by train_removal.py)
    val/step_<N>/<dataset>/    raw output tiles, grid.jpg, metrics.csv
    val/metrics.csv            mean PSNR / LPIPS per step and dataset (one row per step and dataset)
On resume, rows past the resumed step are dropped from the CSVs so they match an uninterrupted run.
"""

import os
import time

import lightning as L
import numpy as np
import torch
from PIL import Image

from ..data.removal_data import eval_tiles
from ..eval.removal_eval import LPIPS, grid_row, make_grid, psnr, read_csv, run_removal, write_csv

TRAIN_FIELDS = ["step", "epoch", "loss", "loss_ema", "t", "grad_mean", "grad_max", "grad_total", "d", "s_per_step", "mem_GB"]
VAL_FIELDS = ["step", "dataset", "n", "psnr", "lpips", "minutes"]


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def append_csv(path, row, fields):
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new:
            f.write(",".join(fields) + "\n")
        f.write(",".join(str(row.get(k, "")) for k in fields) + "\n")


def truncate_csv(path, max_step):
    """Drop rows with step > max_step (they belong to the interrupted part of a run)."""
    rows = read_csv(path)
    keep = [r for r in rows if int(r["step"]) <= max_step]
    if len(keep) != len(rows):
        write_csv(path, keep, list(rows[0]))


class RemovalTrainingCallback(L.Callback):
    def __init__(self, cfg, run_dir, val_data=None, sampler=None):
        tcfg, vcfg = cfg["train"], cfg["val"]
        self.cfg, self.run_dir, self.val_data, self.sampler = cfg, run_dir, val_data, sampler
        self.print_every = int(tcfg.get("print_every_n_steps", 10))
        self.save_interval = int(tcfg.get("save_interval", 500))
        self.resume_every = int(tcfg.get("resume_every_n_epochs", 1))
        self.keep_resume = int(tcfg.get("keep_resume_ckpts", 2))
        self.val_interval = int(vcfg.get("interval", 0) or 0)
        self.val_at_start = bool(vcfg.get("at_start", True))
        # Saved in resume checkpoints (state_dict / load_state_dict).
        self.state = dict(last_val_step=None, last_save_step=None, last_resume_step=None, launches=0)
        self.metrics_path = os.path.join(run_dir, "train_metrics.csv")
        self.names_path = os.path.join(run_dir, "batch_names.csv")
        self.val_metrics_path = os.path.join(run_dir, "val", "metrics.csv")
        self._lpips = None
        self._val_tiles = None
        self._lora_params = None
        self._grad = None
        self._micro_losses, self._micro_names = [], []

    # ------------------------------------------------------------ state

    def state_dict(self):
        return dict(self.state)

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)

    # ------------------------------------------------------------ hooks

    def on_train_start(self, trainer, pl_module):
        step = trainer.global_step
        self.state["launches"] += 1
        for path in (self.metrics_path, self.names_path):
            truncate_csv(path, step)
        self._seen_step, self._t_step = step, time.time()
        self._lora_params = [p for p in pl_module.parameters() if p.requires_grad]
        n_epoch = len(self.sampler) // self.cfg["train"]["batch_size"] if self.sampler is not None else "?"
        log(f"train start: global_step {step}, epoch {trainer.current_epoch}, launch {self.state['launches']}, "
            f"max_steps {trainer.max_steps}, {n_epoch} batches/epoch, "
            f"{sum(p.numel() for p in self._lora_params) / 1e6:.1f}M trainable params")
        if step == 0 and self.val_at_start and self.state["last_val_step"] is None:
            self.validate(trainer, pl_module, 0)
        self._t_step = time.time()

    def on_train_epoch_start(self, trainer, pl_module):
        # Not called for the first epoch after a resume (Lightning restart semantics), hence hasattr below.
        self._t_epoch = time.time()

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        # Gradient norms before clipping (Lightning clips after this hook), one GPU sync.
        grads = [p.grad for p in self._lora_params if p.grad is not None]
        if grads:
            norms = torch.stack(torch._foreach_norm(grads)).float()
            self._grad = torch.stack([norms.mean(), norms.max(), norms.norm()]).tolist()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.sampler is not None:
            # The epoch's samples depend only on (seed, epoch): this is what makes resuming exact.
            assert self.sampler.epoch == trainer.current_epoch, (
                f"sampler drew epoch {self.sampler.epoch} for trainer epoch {trainer.current_epoch}")
        self._micro_losses.append(pl_module.last_loss)
        self._micro_names.extend(f"{d}/{n}/{a}" for d, n, a in zip(batch["dataset"], batch["name"], batch["aug"]))
        step = trainer.global_step
        if step == self._seen_step:  # still accumulating gradients
            return
        self._seen_step = step
        now = time.time()
        dt, self._t_step = now - self._t_step, now
        self._dt_avg = dt if getattr(self, "_dt_avg", None) is None else 0.9 * self._dt_avg + 0.1 * dt
        opt = trainer.optimizers[0]
        d = opt.param_groups[0].get("d")
        mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        grad = self._grad or [float("nan")] * 3
        row = dict(step=step, epoch=trainer.current_epoch, loss=f"{np.mean(self._micro_losses):.6f}",
                   loss_ema=f"{pl_module.log_loss:.6f}", t=f"{pl_module.last_t:.4f}",
                   grad_mean=f"{grad[0]:.5g}", grad_max=f"{grad[1]:.5g}", grad_total=f"{grad[2]:.5g}",
                   d="" if d is None else f"{d:.6g}", s_per_step=f"{dt:.2f}", mem_GB=f"{mem:.1f}")
        append_csv(self.metrics_path, row, TRAIN_FIELDS)
        append_csv(self.names_path, dict(step=step, names=";".join(self._micro_names)), ["step", "names"])
        self._micro_losses, self._micro_names = [], []

        if step % self.print_every == 0 or step == 1:
            left = (trainer.max_steps - step) * self._dt_avg / 3600 if trainer.max_steps > 0 else float("nan")
            log(f"step {step} epoch {trainer.current_epoch} | loss {row['loss']} ema {row['loss_ema']} | "
                f"grad mean {row['grad_mean']} max {row['grad_max']} total {row['grad_total']} | d {row['d']} | "
                f"{dt:.1f} s/step, ~{left:.1f} h left | peak {row['mem_GB']} GB")
        if self.save_interval and step % self.save_interval == 0 and step != self.state["last_save_step"]:
            self.save_lora(pl_module, step)
        if self.val_interval and step % self.val_interval == 0 and step != self.state["last_val_step"]:
            self.validate(trainer, pl_module, step)
        # Resume checkpoint after the epoch's last batch, not in on_train_epoch_end: Lightning 2.6 restores a
        # checkpoint taken at the last batch's on_train_batch_end as "epoch completed" and starts the next
        # epoch cleanly; one taken in on_train_epoch_end resumes inside the old epoch (an extra step with
        # a different sample order), which the M3b test caught.
        last_batch = batch_idx + 1 == trainer.num_training_batches
        if last_batch and self.resume_every and (trainer.current_epoch + 1) % self.resume_every == 0:
            log(f"epoch {trainer.current_epoch} done at step {step}"
                + (f" ({(time.time() - self._t_epoch) / 60:.1f} min)" if hasattr(self, "_t_epoch") else ""))
            self.save_resume(trainer)
        self._t_step = time.time()  # saving / validation time is not counted in the next step's s/step

    def on_train_end(self, trainer, pl_module):
        step = trainer.global_step
        if step != self.state["last_save_step"] and step > 0:
            self.save_lora(pl_module, step)
        if step != self.state["last_resume_step"] and step > 0:
            self.save_resume(trainer)
        log(f"train end at step {step}")

    # ------------------------------------------------------------ saving

    def save_lora(self, pl_module, step):
        path = os.path.join(self.run_dir, "ckpt", str(step))
        pl_module.save_lora(path)
        self.state["last_save_step"] = step
        log(f"saved LoRA -> {path}")

    def save_resume(self, trainer):
        """last.ckpt.tmp -> (last.ckpt -> prev.ckpt) -> last.ckpt; a crash never leaves a partial last.ckpt."""
        d = os.path.join(self.run_dir, "resume")
        os.makedirs(d, exist_ok=True)
        last, prev, tmp = (os.path.join(d, n) for n in ("last.ckpt", "prev.ckpt", "last.ckpt.tmp"))
        self.state["last_resume_step"] = trainer.global_step  # before saving, so the checkpoint records it
        t0 = time.time()
        trainer.save_checkpoint(tmp, weights_only=False)
        if os.path.exists(last):
            if self.keep_resume >= 2:
                os.replace(last, prev)
            else:
                os.remove(last)
        os.replace(tmp, last)
        log(f"saved resume checkpoint at step {trainer.global_step}, epoch {trainer.current_epoch} "
            f"({os.path.getsize(last) / 2**30:.2f} GB, {time.time() - t0:.0f} s)")

    # ------------------------------------------------------------ validation

    def val_tiles(self):
        """Deterministic eval tiles of the fixed validation set, computed once."""
        if self._val_tiles is None:
            size = self.cfg["data"]["size"]
            self._val_tiles = {}
            for name, ds in self.val_data.items():
                rows = []
                for i in range(len(ds)):
                    src, gt, obj = ds.load(i)
                    rows.append((ds.items[i]["name"], eval_tiles(src, gt, obj, size)))
                self._val_tiles[name] = rows
        return self._val_tiles

    def validate(self, trainer, pl_module, step):
        if not self.val_data:
            return
        vcfg = self.cfg["val"]
        t0 = time.time()
        if self._lpips is None:
            self._lpips = LPIPS(pl_module.device)
        pipe = pl_module.flux_fill_pipe
        cond = dict(prompt_embeds=pl_module.cond_prompt_embeds, pooled_prompt_embeds=pl_module.cond_pooled)
        out_root = os.path.join(self.run_dir, "val", f"step_{step}")
        was_training = pl_module.transformer.training
        pl_module.transformer.eval()
        # Validation must not move the training RNG streams (resumed and uninterrupted runs stay equal).
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        summary = []
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            for name, items in self.val_tiles().items():
                d = os.path.join(out_root, name)
                os.makedirs(d, exist_ok=True)
                grid, per = [], []
                t1 = time.time()
                for item, tiles in items:
                    out = run_removal(pipe, cond, tiles["src"], tiles["mask"], vcfg["num_inference_steps"],
                                      vcfg["guidance_scale"], vcfg["seed"])
                    p, lp = psnr(out, tiles["gt"]), self._lpips(out, tiles["gt"])
                    Image.fromarray(out).save(os.path.join(d, f"{item}.png"))
                    per.append(dict(name=item, psnr=f"{p:.4f}", lpips=f"{lp:.5f}"))
                    grid.append(grid_row(tiles["src"], tiles["mask"], out, tiles["gt"], f"{name}/{item}",
                                         f"step {step}  PSNR {p:.2f}  LPIPS {lp:.3f}"))
                Image.fromarray(make_grid(grid)).save(os.path.join(d, "grid.jpg"), quality=90)
                write_csv(os.path.join(d, "metrics.csv"), per)
                summary.append(dict(step=step, dataset=name, n=len(per),
                                    psnr=f"{np.mean([float(r['psnr']) for r in per]):.4f}",
                                    lpips=f"{np.mean([float(r['lpips']) for r in per]):.5f}",
                                    minutes=f"{(time.time() - t1) / 60:.2f}"))
        pl_module.transformer.train(was_training)
        # One row per (step, dataset); a re-reached step replaces its old rows.
        rows = [r for r in read_csv(self.val_metrics_path) if int(r["step"]) != step] + summary
        write_csv(self.val_metrics_path, sorted(rows, key=lambda r: (int(r["step"]), r["dataset"])), VAL_FIELDS)
        self.state["last_val_step"] = step
        log(f"validation step {step}: " + ", ".join(f"{r['dataset']} PSNR {r['psnr']} LPIPS {r['lpips']}" for r in summary)
            + f" ({(time.time() - t0) / 60:.1f} min) -> {out_root}")
