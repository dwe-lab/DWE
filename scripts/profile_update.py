"""Replay fixed trajectories through native TRL/DeepSpeed complete optimizer updates."""

import argparse
import copy
import hashlib
import json
import statistics
import time
import tomllib
from functools import cache
from pathlib import Path

import torch
from datasets import Dataset
from deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine import TorchCheckpointEngine
from transformers import AutoTokenizer, TrainerCallback
from trl import DistillationConfig, DistillationTrainer

from dwe.loss import LossConfig
from dwe.trainer import CouplingTrainer


def make_inputs(rows, positions, rank, world_size, pad_id, device):
    if positions % world_size:
        raise ValueError("Global token budget must divide evenly across ranks")
    remaining = positions // world_size
    selected = []
    for row in rows[rank::world_size]:
        completion = row["completion_ids"][:remaining]
        if completion:
            selected.append((row["prompt_ids"], completion))
            remaining -= len(completion)
        if not remaining:
            break
    if remaining:
        raise ValueError(f"Rank {rank} needs {remaining} more distinct completion tokens")
    np = max(len(p) for p, _ in selected)
    nc = max(len(c) for _, c in selected)
    fields = {k: [] for k in ("prompt_ids", "prompt_mask", "completion_ids", "completion_mask")}
    for p, c in selected:
        fields["prompt_ids"].append([pad_id] * (np - len(p)) + p)
        fields["prompt_mask"].append([0] * (np - len(p)) + [1] * len(p))
        fields["completion_ids"].append(c + [pad_id] * (nc - len(c)))
        fields["completion_mask"].append([1] * len(c) + [0] * (nc - len(c)))
    tensors = {k: torch.tensor(v, device=device) for k, v in fields.items()}
    tensors["num_items_in_batch"] = torch.tensor(float(positions), device=device)
    return tensors


class FixedInputs:
    def _prepare_inputs(self, inputs):
        return self.replay_inputs


class ExactReplay(FixedInputs, DistillationTrainer):
    pass


class SparseReplay(FixedInputs, CouplingTrainer):
    pass


class ReplayCheckpointEngine(TorchCheckpointEngine):
    """Read the native snapshot once; restore independent CPU copies thereafter."""

    @cache  # noqa: B019 -- one engine per run; retain its initial snapshot until exit.
    def _snapshot(self, path):
        return super().load(path, map_location="cpu")

    def load(self, path, map_location=None):
        # Native ZeRO/model loaders both request CPU storage (device or callable).
        return copy.deepcopy(self._snapshot(path))


class MeasureUpdate(TrainerCallback):
    def __init__(self, trainer, args, cfg, manifest):
        self.trainer, self.args, self.cfg, self.manifest = trainer, args, cfg, manifest
        self.records = []

    def on_train_begin(self, args, state, control, **kwargs):
        self.engine = self.trainer.model_wrapped
        if not hasattr(self.engine, "save_checkpoint"):
            raise RuntimeError("This protocol requires native DeepSpeed checkpoint restoration")
        self.engine.checkpoint_engine = ReplayCheckpointEngine()
        # Native checkpoint preserves partitioned model, master weights, optimizer and scheduler.
        self.engine.save_checkpoint(str(self.args.scratch), tag="initial")
        self.probe = next(self.engine.module.parameters()).detach().flatten()[:32].clone()

    def on_step_begin(self, args, state, control, **kwargs):
        path, _ = self.engine.load_checkpoint(
            str(self.args.scratch),
            tag="initial",
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
        )
        if path is None:
            raise RuntimeError("Failed to restore the initial optimizer state")
        assert torch.equal(
            next(self.engine.module.parameters()).detach().flatten()[:32], self.probe
        )
        self.engine.zero_grad()
        torch.manual_seed(self.cfg["protocol"]["seed"] + state.global_step)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.reset_peak_memory_stats()
        self.start = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start
        measured = torch.tensor(
            [elapsed, torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
            dtype=torch.float64,
            device=self.trainer.accelerator.device,
        )
        all_ranks = [
            torch.empty_like(measured) for _ in range(self.trainer.accelerator.num_processes)
        ]
        torch.distributed.all_gather(all_ranks, measured)
        if state.global_step > self.cfg["protocol"]["warmup"]:
            self.records.append([x.cpu().tolist() for x in all_ranks])
        if self.trainer.is_world_process_zero():
            result = dict(self.manifest, updates=self.records)
            if self.records:
                result["median_seconds"] = statistics.median(
                    max(x[0] for x in r) for r in self.records
                )
                result["peak_allocated_bytes_per_gpu"] = [
                    max(r[i][1] for r in self.records) for i in range(len(all_ranks))
                ]
            self.args.output.mkdir(parents=True, exist_ok=True)
            (self.args.output / "measurement.json").write_text(json.dumps(result, indent=2))
            print(
                json.dumps(
                    {
                        "step": state.global_step,
                        "seconds": max(x[0].item() for x in all_ranks),
                        "measured": len(self.records),
                    }
                ),
                flush=True,
            )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--trajectories", type=Path, required=True)
    p.add_argument("--student", required=True)
    p.add_argument("--teacher", required=True)
    p.add_argument("--deepspeed", required=True)
    p.add_argument("--backend", choices=["upstream", "quantile", "taopd"], required=True)
    p.add_argument("--vocabulary-order", choices=["principal", "natural"])
    p.add_argument("--domain", required=True)
    p.add_argument("--positions", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scratch", type=Path, required=True)
    a = p.parse_args()
    cfg = tomllib.loads(a.config.read_text())
    data = json.loads(a.trajectories.read_text())
    settings = cfg["training"] | {
        "max_steps": cfg["protocol"]["warmup"] + cfg["protocol"]["repeats"],
        "deepspeed": a.deepspeed,
    }
    args = DistillationConfig(output_dir=str(a.output), **settings)
    tokenizer = AutoTokenizer.from_pretrained(a.student, padding_side="left")
    prompts = Dataset.from_list([{"prompt": [{"role": "user", "content": "Replay"}]}] * 128)
    kwargs = dict(
        model=a.student,
        teacher_model=a.teacher,
        args=args,
        train_dataset=prompts,
        processing_class=tokenizer,
    )
    if a.backend == "upstream":
        trainer = ExactReplay(**kwargs)
    else:
        trainer = SparseReplay(
            **kwargs,
            loss_backend="taopd" if a.backend == "taopd" else "sparse",
            estimator=LossConfig(**cfg["estimator"], method="quantile"),
            taopd_config=cfg["taopd"],
            vocabulary_order=a.vocabulary_order,
        )
        trainer.prepare_sampling()
    device = trainer.accelerator.device
    rows = [r for r in data["trajectories"] if r["domain"] == a.domain]
    trainer.replay_inputs = make_inputs(
        rows,
        a.positions,
        trainer.accelerator.process_index,
        trainer.accelerator.num_processes,
        data["pad_token_id"],
        device,
    )
    manifest = {
        "scope": "Two-GPU ZeRO-2 full update; teacher forward, student forward/backward and optimizer; excludes generation/state restore/save",
        "config": cfg,
        "backend": a.backend,
        "estimator_state": trainer.estimator_state()
        if isinstance(trainer, CouplingTrainer)
        else None,
        "geometry_setup_seconds": trainer.geometry_setup_seconds
        if isinstance(trainer, CouplingTrainer)
        else 0.0,
        "domain": a.domain,
        "global_positions": a.positions,
        "trajectory_sha256": hashlib.sha256(a.trajectories.read_bytes()).hexdigest(),
        "student": a.student,
        "teacher": a.teacher,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "world_size": trainer.accelerator.num_processes,
    }
    trainer.add_callback(MeasureUpdate(trainer, a, cfg, manifest))
    trainer.train()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
