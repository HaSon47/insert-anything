"""Milestone M3b: kill-and-resume test of src/train/train_removal.py (CPU, tiny FLUX), from the repo root:

    python -m tests.test_removal_resume [--keep] [--gpu]

Runs on CPU (CUDA hidden) unless --gpu: CPU float32 gives exact loss equality; --gpu also exercises the
CUDA RNG save/restore path.

Tiny config: 16 OmniPaint + 16 ObjectStitch samples per epoch, batch 8, so 4 steps per epoch; max_steps 16;
resume checkpoint every epoch; validation at step 0 and every 8 steps (2 + 1 images, 2 sampling steps).
  A   uninterrupted run.
  B1  same run, killed with SIGKILL at step 10 (inside epoch 3, epoch index 2).
  B2  same command again: must resume at step 8 / epoch 2 from last.ckpt and finish at step 16.
  B3  last.ckpt overwritten with garbage: must fall back to prev.ckpt (step 12) and finish again.
  B4  --max_steps 20 on the finished run: continues to step 20.
  bad --config that changes a hyper-parameter -> exit 2;  --resume none on an existing run -> refused.
Checks: restored LoRA and Prodigy state (incl. d) bit-identical to the checkpoint file; per-step sample
names and losses identical to A; no step-0 validation after resuming. Outputs in runs_removal/_checks/m3b.
"""

import argparse
import copy
import json
import os
import shutil
import signal
import subprocess
import sys
import time

import lightning as L
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "runs_removal", "_checks", "m3b")


# ---------------------------------------------------------------- worker side

class Killer(L.Callback):
    def __init__(self, step):
        self.step = step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step == self.step:
            print(f"KILLING at step {trainer.global_step}", flush=True)
            os.kill(os.getpid(), signal.SIGKILL)


def deep_equal(a, b):
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(deep_equal(x, y) for x, y in zip(a, b))
    return a == b


class RestoreProbe(L.Callback):
    """After Lightning restored a checkpoint, compare the live LoRA and optimizer with the file."""

    def __init__(self, out):
        self.out = out

    def on_train_start(self, trainer, pl_module):
        rec = dict(global_step=trainer.global_step, epoch=trainer.current_epoch, ckpt_path=trainer.ckpt_path)
        if trainer.ckpt_path:
            ck = torch.load(trainer.ckpt_path, map_location="cpu", weights_only=False)
            params = dict(pl_module.named_parameters())
            rec["n_lora_tensors"] = len(ck["state_dict"])
            rec["lora_equal"] = all(torch.equal(params[k].detach().cpu(), v) for k, v in ck["state_dict"].items())
            live, saved = trainer.optimizers[0].state_dict(), ck["optimizer_states"][0]
            rec["opt_state_equal"] = deep_equal(live["state"], saved["state"])
            rec["opt_groups_equal"] = deep_equal(live["param_groups"], saved["param_groups"])
            rec["d_live"], rec["d_saved"] = live["param_groups"][0].get("d"), saved["param_groups"][0].get("d")
        with open(self.out, "w") as f:
            json.dump(rec, f, indent=2, default=str)


def worker(args):
    from src.train import train_removal
    from tests.test_removal_model import tiny_cond, tiny_pipe
    cbs = [RestoreProbe(args.probe)] + ([Killer(args.kill_at)] if args.kill_at else [])
    argv = ["--config", args.config, "--run_name", args.run_name, "--resume", args.resume]
    if args.max_steps:
        argv += ["--max_steps", str(args.max_steps)]
    train_removal.main(argv, pipe_factory=lambda: tiny_pipe(0), cond_factory=lambda: tiny_cond(0), extra_callbacks=cbs)


# ---------------------------------------------------------------- driver side

def tiny_run_config():
    from tests.test_removal_model import tiny_cfg
    cfg = tiny_cfg()
    cfg["data"]["samples_per_epoch"] = {"omnipaint": 16, "objectstitch": 16}
    cfg["train"].update(batch_size=8, max_steps=16, save_interval=4, print_every_n_steps=1, dataloader_workers=2,
                        save_path=OUT, run_name="A", resume_every_n_epochs=1, keep_resume_ckpts=2)
    cfg["val"].update(at_start=True, interval=8, objectstitch_n=2, omnipaint_val_n=1, num_inference_steps=2)
    return cfg


ENV = dict(os.environ)


def launch(tag, config, run_name, kill_at=None, resume="auto", max_steps=None):
    probe = os.path.join(OUT, f"probe_{tag}.json")
    if os.path.exists(probe):
        os.remove(probe)
    cmd = [sys.executable, "-m", "tests.test_removal_resume", "--worker", "--config", config,
           "--run_name", run_name, "--resume", resume, "--probe", probe]
    if kill_at:
        cmd += ["--kill_at", str(kill_at)]
    if max_steps:
        cmd += ["--max_steps", str(max_steps)]
    log_path = os.path.join(OUT, f"stdout_{tag}.log")
    t0 = time.time()
    with open(log_path, "w") as f:  # a file, not a pipe: orphaned DataLoader workers must not block us
        rc = subprocess.call(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=ENV)
    text = open(log_path).read()
    probe_rec = json.load(open(probe)) if os.path.exists(probe) else None
    print(f"  {tag}: exit {rc} in {time.time() - t0:.0f} s; probe {probe_rec and {k: probe_rec[k] for k in ('global_step', 'epoch')}}")
    return rc, text, probe_rec


def rows(run, name):
    from src.eval.removal_eval import read_csv
    return read_csv(os.path.join(OUT, run, name))


def ckpt_step(run, name):
    return torch.load(os.path.join(OUT, run, "resume", name), map_location="cpu", mmap=True, weights_only=False)["global_step"]


def compare_with_a(run, steps):
    a = {int(r["step"]): r for r in rows("A", "train_metrics.csv")}
    b = {int(r["step"]): r for r in rows(run, "train_metrics.csv")}
    an = {int(r["step"]): r["names"] for r in rows("A", "batch_names.csv")}
    bn = {int(r["step"]): r["names"] for r in rows(run, "batch_names.csv")}
    assert sorted(b) == list(range(1, max(b) + 1)), f"{run}: steps in train_metrics.csv {sorted(b)}"
    diffs = [abs(float(a[s]["loss"]) - float(b[s]["loss"])) for s in steps]
    names_equal = all(an[s] == bn[s] and a[s]["epoch"] == b[s]["epoch"] for s in steps)
    return max(diffs), names_equal


def check(cond, msg, failures):
    print(("    ok    " if cond else "    FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def driver(keep, gpu):
    failures = []
    ENV["TMPDIR"] = f"/dev/shm/{os.environ.get('USER', 'user')}_tmp"
    os.makedirs(ENV["TMPDIR"], exist_ok=True)
    if not gpu:
        ENV["CUDA_VISIBLE_DEVICES"] = ""
    print(f"device: {'cuda' if gpu else 'cpu'}")
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    config = os.path.join(OUT, "tiny.yaml")
    cfg = tiny_run_config()
    with open(config, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print("A: uninterrupted")
    rc, text_a, probe_a = launch("A", config, "A")
    check(rc == 0, "A exits 0", failures)
    check(probe_a["global_step"] == 0 and not probe_a["ckpt_path"], "A starts fresh at step 0", failures)
    check("validation step 0" in text_a, "A runs the step-0 validation", failures)
    check(ckpt_step("A", "last.ckpt") == 16 and ckpt_step("A", "prev.ckpt") == 12, "A: last.ckpt step 16, prev.ckpt step 12", failures)

    print("B1: killed at step 10")
    rc, text_b1, _ = launch("B1", config, "B", kill_at=10)
    check(rc == -signal.SIGKILL, f"B1 killed by SIGKILL (exit {rc})", failures)
    check(ckpt_step("B", "last.ckpt") == 8 and ckpt_step("B", "prev.ckpt") == 4, "B1 left last.ckpt step 8, prev.ckpt step 4", failures)

    print("B2: same command again")
    rc, text_b2, p = launch("B2", config, "B")
    check(rc == 0, "B2 exits 0", failures)
    check(p["global_step"] == 8 and p["epoch"] == 2, f"B2 resumes at step 8 / epoch 2 (got {p['global_step']} / {p['epoch']})", failures)
    check(p.get("lora_equal") and p.get("n_lora_tensors", 0) > 0, f"B2 LoRA tensors bit-identical to last.ckpt ({p.get('n_lora_tensors')} tensors)", failures)
    check(p.get("opt_state_equal") and p.get("opt_groups_equal"), f"B2 Prodigy state + groups identical (d live {p.get('d_live')} saved {p.get('d_saved')})", failures)
    check("validation step 0" not in text_b2, "B2 skips the step-0 validation", failures)
    check("validation step 16" in text_b2, "B2 validates at step 16", failures)
    loss_diff, names_equal = compare_with_a("B", range(1, 17))
    check(names_equal, "B: per-step epoch and sample names (incl. augmentation branch) identical to A for steps 1-16", failures)
    check(loss_diff < 1e-6, f"B: losses identical to A for steps 1-16 (max |diff| {loss_diff:.2e})", failures)
    check(ckpt_step("B", "last.ckpt") == 16, "B final global_step 16", failures)
    check(sorted(os.listdir(os.path.join(OUT, "B", "ckpt")), key=int) == ["4", "8", "12", "16"], "B LoRA exports at 4, 8, 12, 16", failures)
    va = [(r["step"], r["dataset"], r["psnr"], r["lpips"]) for r in rows("A", "val/metrics.csv")]
    vb = [(r["step"], r["dataset"], r["psnr"], r["lpips"]) for r in rows("B", "val/metrics.csv")]
    check(va == vb and len(va) == 6, f"validation metrics identical to A at steps 0, 8, 16 ({len(vb)} rows)", failures)

    print("B3: corrupted last.ckpt")
    with open(os.path.join(OUT, "B", "resume", "last.ckpt"), "wb") as f:
        f.write(b"garbage")
    rc, text_b3, p = launch("B3", config, "B")
    check(rc == 0 and "is not usable" in text_b3, "B3 warns that last.ckpt is unusable and exits 0", failures)
    check(p["global_step"] == 12 and p["ckpt_path"].endswith("prev.ckpt"), f"B3 resumes from prev.ckpt at step 12 (got {p['global_step']})", failures)
    loss_diff, names_equal = compare_with_a("B", range(13, 17))
    check(names_equal and loss_diff < 1e-6, f"B3 steps 13-16 identical to A (max |diff| {loss_diff:.2e})", failures)

    print("B4: --max_steps 20")
    rc, text_b4, p = launch("B4", config, "B", max_steps=20)
    check(rc == 0 and p["global_step"] == 16 and ckpt_step("B", "last.ckpt") == 20, "B4 continues 16 -> 20", failures)
    check(yaml.safe_load(open(os.path.join(OUT, "B", "config.yaml")))["train"]["max_steps"] == 20, "run config records max_steps 20", failures)

    print("guards")
    bad = copy.deepcopy(cfg)
    bad["train"]["gradient_clip_val"] = 1.0
    bad_path = os.path.join(OUT, "tiny_bad.yaml")
    with open(bad_path, "w") as f:
        yaml.safe_dump(bad, f, sort_keys=False)
    rc, text, _ = launch("bad_config", bad_path, "B")
    check(rc == 2 and "train.gradient_clip_val" in text, "changed hyper-parameter on resume -> exit 2 with the diff", failures)
    rc, text, _ = launch("resume_none", config, "B", resume="none")
    check(rc != 0 and "already has checkpoints" in text, "--resume none on an existing run is refused", failures)

    print(f"\n{'ALL PASSED' if not failures else f'{len(failures)} FAILED'}; outputs in {OUT}")
    if not keep and not failures:
        for run in ("A", "B"):
            shutil.rmtree(os.path.join(OUT, run, "resume"))
    return 1 if failures else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--keep", action="store_true", help="keep the resume checkpoints")
    p.add_argument("--gpu", action="store_true", help="run the tiny model on the GPU instead of the CPU")
    p.add_argument("--config")
    p.add_argument("--run_name")
    p.add_argument("--resume", default="auto")
    p.add_argument("--probe")
    p.add_argument("--kill_at", type=int)
    p.add_argument("--max_steps", type=int)
    a = p.parse_args()
    sys.exit(worker(a) if a.worker else driver(a.keep, a.gpu))
