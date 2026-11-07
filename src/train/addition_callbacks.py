"""Training callback for addition fine-tuning (addition_training_plan.md, section 5).

Logging, CSVs and resume checkpoints are inherited from RemovalTrainingCallback. The difference is that
every LoRA export is followed by evaluation on the fixed set: save_lora() runs it, including for the
final export, so there is no separate evaluation interval that could drift out of step with saving.

Files under <run_dir>/val/:
    subsets.json                       fixed evaluation names (written by train_addition.py)
    metrics.csv                        mean psnr_bg per (step, group)
    step_<N>/<group>/grid.jpg          [reference | input + mask | output | GT], one row per sample
    step_<N>/<group>/metrics.csv       psnr_bg per sample
    step_<N>/<group>/<name>.png        raw output tile
"""

import os
import time

import numpy as np
import torch
from PIL import Image

from ..eval.addition_eval import grid_row, psnr_bg, run_addition, save_grid
from ..eval.removal_eval import read_csv, write_csv
from .removal_callbacks import RemovalTrainingCallback, log

VAL_FIELDS = ["step", "group", "n", "psnr_bg", "minutes"]


class AdditionTrainingCallback(RemovalTrainingCallback):
    def save_lora(self, pl_module, step):
        super().save_lora(pl_module, step)
        self.validate(None, pl_module, step)

    def val_tiles(self):
        """Deterministic tiles of the fixed set, computed once."""
        if self._val_tiles is None:
            self._val_tiles = {group: [(ds.items[i]["name"], ds.tiles(i)) for i in range(len(ds))]
                               for group, ds in self.val_data.items()}
        return self._val_tiles

    def validate(self, trainer, pl_module, step):
        if not self.val_data:
            return
        vcfg = self.cfg["val"]
        t0 = time.time()
        out_root = os.path.join(self.run_dir, "val", f"step_{step}")
        was_training = pl_module.transformer.training
        pl_module.transformer.eval()
        # Evaluation must not move the training RNG streams (resumed and uninterrupted runs stay equal).
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        summary = []
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            for group, samples in self.val_tiles().items():
                d = os.path.join(out_root, group)
                os.makedirs(d, exist_ok=True)
                rows, per = [], []
                t1 = time.time()
                for name, t in samples:
                    out = run_addition(pl_module.flux_fill_pipe, pl_module.flux_redux, t["ref"], t["src"], t["mask"],
                                       vcfg["num_inference_steps"], vcfg["guidance_scale"], vcfg["seed"])
                    value = psnr_bg(out, t["gt"], t["mask"], t["valid"])
                    Image.fromarray(out).save(os.path.join(d, f"{name}.png"))
                    per.append(dict(name=name, psnr_bg=f"{value:.4f}"))
                    rows.append(grid_row(t["ref"], t["src"], t["mask"], out, t["gt"], f"{group}/{name}",
                                         f"step {step}  psnr_bg {value:.2f}"))
                save_grid(os.path.join(d, "grid.jpg"), rows)
                write_csv(os.path.join(d, "metrics.csv"), per)
                summary.append(dict(step=step, group=group, n=len(per),
                                    psnr_bg=f"{np.mean([float(r['psnr_bg']) for r in per]):.4f}",
                                    minutes=f"{(time.time() - t1) / 60:.2f}"))
        pl_module.transformer.train(was_training)
        # One row per (step, group); a re-reached step replaces its old rows.
        rows = [r for r in read_csv(self.val_metrics_path) if int(r["step"]) != step] + summary
        write_csv(self.val_metrics_path, sorted(rows, key=lambda r: (int(r["step"]), r["group"])), VAL_FIELDS)
        self.state["last_val_step"] = step
        log(f"evaluation step {step}: " + ", ".join(f"{r['group']} psnr_bg {r['psnr_bg']}" for r in summary)
            + f" ({(time.time() - t0) / 60:.1f} min) -> {out_root}")
