"""Removal fine-tuning entry point (removal_training_plan.md, sections 4 and 4.2). From the repo root:

    python -m src.train.train_removal --config experiments/config/removal.yaml --run_name removal_v1 --resume auto

--resume auto   resume from <run>/resume/last.ckpt (prev.ckpt if last is unreadable), else start fresh
--resume none   start fresh; refuses if <run> already holds checkpoints
--resume PATH   resume from that checkpoint
--max_steps N   the only setting that may change on resume (extends or shortens the run)

On resume the run's saved <run>/config.yaml is used; if --config differs (other than max_steps
and run_name) the script stops with the differences. Configuration / argument errors exit with
code 2, which scripts/train_removal.sh does not retry.
"""

import argparse
import copy
import json
import math
import os
import random
import socket
import sys
import time

import lightning as L
from lightning.fabric.plugins.environments import LightningEnvironment
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ..data.removal_data import build_train_data, build_val_data, load_config, make_val_subsets, repo_path
from ..eval.removal_eval import load_or_build_white_ref_cond
from ..models.removal_model import RemovalModel
from .removal_callbacks import RemovalTrainingCallback, log

EXIT_CONFIG = 2
RESUME_MUTABLE = {"train.max_steps", "train.run_name"}
CKPT_KEYS = ("state_dict", "optimizer_states", "loops", "global_step", "epoch", "callbacks")


def fail(msg):
    print(msg, file=sys.stderr, flush=True)
    raise SystemExit(EXIT_CONFIG)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=os.environ.get("XFL_CONFIG", "experiments/config/removal.yaml"))
    p.add_argument("--run_name", default=None, help="default: train.run_name from the config")
    p.add_argument("--resume", default="auto", help="auto | none | path to a .ckpt")
    p.add_argument("--max_steps", type=int, default=None)
    return p.parse_args(argv)


# ---------------------------------------------------------------- run directory / resume

def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def config_diff(a, b, ignore=RESUME_MUTABLE):
    fa, fb = flatten(a), flatten(b)
    return [(k, fa.get(k, "<missing>"), fb.get(k, "<missing>"))
            for k in sorted(set(fa) | set(fb)) if k not in ignore and fa.get(k, "<missing>") != fb.get(k, "<missing>")]


def checkpoint_ok(path):
    """Cheap readability check (memory-mapped, tensors not read): a truncated or garbage file fails."""
    try:
        ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        missing = [k for k in CKPT_KEYS if k not in ck]
        return (False, f"missing keys {missing}") if missing else (True, f"step {ck['global_step']}, epoch {ck['epoch']}")
    except Exception as e:  # noqa: BLE001 - any failure means "not usable"
        return False, f"{type(e).__name__}: {e}"


def has_checkpoints(run_dir):
    resume = os.path.join(run_dir, "resume")
    ckpt = os.path.join(run_dir, "ckpt")
    return (os.path.isdir(resume) and any(f.endswith(".ckpt") for f in os.listdir(resume))) or \
           (os.path.isdir(ckpt) and bool(os.listdir(ckpt)))


def prepare_run(args, cfg_in):
    """Returns (cfg, run_dir, ckpt_path or None, notes); writes <run_dir>/config.yaml.
    `notes` are messages to log once train.log is open."""
    run_name = args.run_name or cfg_in["train"]["run_name"]
    run_dir = repo_path(os.path.join(cfg_in["train"]["save_path"], run_name))
    resume_dir = os.path.join(run_dir, "resume")
    ckpt, notes = None, []
    if args.resume == "none":
        if has_checkpoints(run_dir):
            fail(f"{run_dir} already has checkpoints; use --resume auto or another --run_name")
    elif args.resume == "auto":
        for name in ("last.ckpt", "prev.ckpt"):
            path = os.path.join(resume_dir, name)
            if not os.path.exists(path):
                continue
            ok, info = checkpoint_ok(path)
            if ok:
                ckpt = path
                notes.append(f"resuming from {path} ({info})")
                break
            notes.append(f"WARNING: {path} is not usable ({info}); trying the next one")
        if ckpt is None and has_checkpoints(run_dir):
            notes.append(f"WARNING: no usable resume checkpoint in {resume_dir}; starting {run_name} from scratch")
    else:
        ok, info = checkpoint_ok(args.resume)
        if not ok:
            fail(f"--resume {args.resume}: {info}")
        ckpt = args.resume
        notes.append(f"resuming from {ckpt} ({info})")

    cfg_path = os.path.join(run_dir, "config.yaml")
    if ckpt:
        if not os.path.exists(cfg_path):
            fail(f"resuming needs {cfg_path}")
        cfg = load_config(cfg_path)
        diff = config_diff(cfg_in, cfg)
        if diff:
            print(f"--config {args.config} differs from the run's {cfg_path}:", file=sys.stderr)
            for k, new, old in diff:
                print(f"  {k}: run={old!r}  --config={new!r}", file=sys.stderr)
            print("Resume with the run's config (edit or remove the difference), or start a new --run_name.",
                  file=sys.stderr)
            raise SystemExit(EXIT_CONFIG)
    else:
        cfg = copy.deepcopy(cfg_in)
    cfg["train"]["run_name"] = run_name
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = args.max_steps
    os.makedirs(run_dir, exist_ok=True)
    with open(cfg_path + ".tmp", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    os.replace(cfg_path + ".tmp", cfg_path)
    return cfg, run_dir, ckpt, notes


def load_or_make_subsets(cfg, run_dir):
    path = os.path.join(run_dir, "val", "subsets.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    subsets = make_val_subsets(cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(subsets, f, indent=2)
    return subsets


class _Tee:
    """Mirror a stream into the run's train.log (appended across launches)."""

    def __init__(self, stream, f):
        self.stream, self.f = stream, f

    def write(self, s):
        self.stream.write(s)
        self.f.write(s)
        self.f.flush()
        return len(s)

    def flush(self):
        self.stream.flush()
        self.f.flush()

    def isatty(self):
        return False

    def fileno(self):
        return self.stream.fileno()


# ---------------------------------------------------------------- main

def main(argv=None, pipe_factory=None, cond_factory=None, extra_callbacks=()):
    """pipe_factory / cond_factory / extra_callbacks are hooks for tests (tiny model on CPU)."""
    args = parse_args(argv)
    cfg, run_dir, ckpt, notes = prepare_run(args, load_config(repo_path(args.config)))
    tcfg = cfg["train"]

    log_file = open(os.path.join(run_dir, "train.log"), "a")
    sys.stdout, sys.stderr = _Tee(sys.stdout, log_file), _Tee(sys.stderr, log_file)
    log("=" * 80)
    log(f"launch on {socket.gethostname()} (SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID', '-')}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '-')}): {' '.join(sys.argv)}")
    for note in notes:
        log(note)
    log(f"run dir {run_dir}; " + (f"resume from {ckpt}" if ckpt else "fresh start"))

    seed = int(tcfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # resumed runs restore all RNG states from the checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # The LoRA master weights are float32 (RemovalModel); TF32 runs their matmuls on tensor cores.
    # The bf16 base model is unaffected.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    t0 = time.time()
    cond = cond_factory() if cond_factory else load_or_build_white_ref_cond(cfg, run_dir, device)
    pipe = pipe_factory() if pipe_factory else None
    model = RemovalModel(cfg, pipe=pipe, cond=cond, device=device, load_init_lora=ckpt is None)
    log(f"model ready in {time.time() - t0:.0f} s: {model.n_lora_modules} LoRA modules, "
        f"init LoRA {model.init_lora_path if ckpt is None else '(from checkpoint)'}")

    concat, sampler = build_train_data(cfg)
    loader = DataLoader(concat, batch_size=tcfg["batch_size"], sampler=sampler, drop_last=True,
                        num_workers=tcfg.get("dataloader_workers", 4), pin_memory=device == "cuda",
                        persistent_workers=False)
    val_data = None
    if cfg["val"].get("interval") or cfg["val"].get("at_start"):
        val_data = build_val_data(cfg, load_or_make_subsets(cfg, run_dir))
    callback = RemovalTrainingCallback(cfg, run_dir, val_data=val_data, sampler=sampler)

    trainer = L.Trainer(
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        max_steps=tcfg["max_steps"],
        max_epochs=-1,
        accumulate_grad_batches=tcfg.get("accumulate_grad_batches", 1),
        gradient_clip_val=tcfg.get("gradient_clip_val", 0.5),
        callbacks=[callback, *extra_callbacks],
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        use_distributed_sampler=False,
        # Single process: keep Lightning's SLURM integration (SIGUSR1 -> `scontrol requeue` of the job) off,
        # since the job may be a shared allocation. Re-running the launcher resumes instead.
        plugins=[LightningEnvironment()],
    )
    # The sampler's epoch comes from global_step, which Lightning restores before it builds the first
    # iterator of a resumed run; current_epoch is only advanced afterwards (FitLoop.setup_data() calls
    # iter() before FitLoop.reset()), so set_epoch()/current_epoch would replay the previous epoch.
    steps_per_epoch = math.ceil(len(loader) / tcfg.get("accumulate_grad_batches", 1))
    sampler.epoch_fn = lambda: trainer.global_step // steps_per_epoch
    trainer.fit(model, loader, ckpt_path=ckpt, weights_only=False)
    log(f"finished: global_step {trainer.global_step} / max_steps {trainer.max_steps}")


if __name__ == "__main__":
    main()
