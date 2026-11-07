"""Addition fine-tuning entry point (addition_training_plan.md, sections 4 and 6). From the repo root:

    python -m src.train.train_addition --config experiments/config/addition.yaml --run_name addition_v1 --resume auto

--resume auto   resume from <run>/resume/last.ckpt (prev.ckpt if last is unreadable), else start fresh
--resume none   start fresh; refuses if <run> already holds checkpoints
--resume PATH   resume from that checkpoint
--max_steps N   the only setting that may change on resume (extends or shortens the run)

Run directory handling and the config guard come from train_removal.py: on resume the run's saved
<run>/config.yaml is used, and a different --config stops with the differences (exit code 2, which
scripts/train_addition.sh does not retry).
"""

import argparse
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
from torch.utils.data import DataLoader

from ..data.addition_data import build_eval_data, build_train_data, load_config, make_eval_subsets, repo_path
from ..models.addition_model import AdditionModel, load_redux
from .addition_callbacks import AdditionTrainingCallback
from .removal_callbacks import log
from .train_removal import _Tee, prepare_run


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=os.environ.get("XFL_CONFIG", "experiments/config/addition.yaml"))
    p.add_argument("--run_name", default=None, help="default: train.run_name from the config")
    p.add_argument("--resume", default="auto", help="auto | none | path to a .ckpt")
    p.add_argument("--max_steps", type=int, default=None)
    return p.parse_args(argv)


def load_or_make_subsets(cfg, run_dir):
    path = os.path.join(run_dir, "val", "subsets.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    subsets = make_eval_subsets(cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(subsets, f, indent=2)
    return subsets


def main(argv=None, pipe_factory=None, redux_factory=None, extra_callbacks=()):
    """pipe_factory / redux_factory / extra_callbacks are hooks for tests (tiny model on CPU)."""
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
    # The LoRA master weights are float32; TF32 runs their matmuls on tensor cores. The bf16 base is unaffected.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    t0 = time.time()
    redux = redux_factory() if redux_factory else load_redux(cfg, device)
    pipe = pipe_factory() if pipe_factory else None
    model = AdditionModel(cfg, redux=redux, pipe=pipe, device=device, load_init_lora=ckpt is None)
    log(f"model ready in {time.time() - t0:.0f} s: {model.n_lora_modules} LoRA modules, "
        f"init LoRA {model.init_lora_path if ckpt is None else '(from checkpoint)'}")

    dataset, sampler = build_train_data(cfg)
    log(f"training data: {len(dataset)} samples, {sampler.per_epoch[0]} per epoch")
    loader = DataLoader(dataset, batch_size=tcfg["batch_size"], sampler=sampler, drop_last=True,
                        num_workers=tcfg.get("dataloader_workers", 4), pin_memory=device == "cuda",
                        persistent_workers=False)
    eval_data = build_eval_data(cfg, load_or_make_subsets(cfg, run_dir))
    callback = AdditionTrainingCallback(cfg, run_dir, val_data=eval_data, sampler=sampler)

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
    # See train_removal.py: the sampler's epoch must come from global_step for a resumed run to be exact.
    steps_per_epoch = math.ceil(len(loader) / tcfg.get("accumulate_grad_batches", 1))
    sampler.epoch_fn = lambda: trainer.global_step // steps_per_epoch
    trainer.fit(model, loader, ckpt_path=ckpt, weights_only=False)
    log(f"finished: global_step {trainer.global_step} / max_steps {trainer.max_steps}")


if __name__ == "__main__":
    main()
