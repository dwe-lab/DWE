"""Configure upstream TRL training; no custom training or generation loop."""

import argparse
import hashlib
import json
import time
import tomllib
from pathlib import Path

from transformers import AutoTokenizer, TrainerCallback
from trl import DistillationConfig, DistillationTrainer

from .data import load_prompts
from .loss import LossConfig
from .trainer import CouplingTrainer


class StepTiming(TrainerCallback):
    """Measure Trainer update boundaries, including accumulation and optimizer."""

    def on_step_begin(self, args, state, control, **kwargs):
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.started = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if state.is_world_process_zero:
            record = dict(
                step=state.global_step,
                seconds=time.perf_counter() - self.started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated()
                if torch.cuda.is_available()
                else None,
            )
            path = Path(args.output_dir) / "update-timing.jsonl"
            with path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")


class StopAfterStep(TrainerCallback):
    """Save and pause at an update boundary without shortening the LR schedule."""

    def __init__(self, step):
        self.step = step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.step:
            control.should_save = True
            control.should_training_stop = True
        return control


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=[
            "upstream",
            "liger",
            "coordinate",
            "independent",
            "coupled",
            "quantile",
            "taopd",
            "sctaopd",
        ],
        required=True,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--vocabulary-order",
        choices=["principal", "natural"],
        help="Coupled/quantile default: principal; natural is the ordering ablation",
    )
    parser.add_argument(
        "--stop-after-step",
        type=int,
        help="Pause with a full checkpoint; preserve max_steps schedule",
    )
    cli = parser.parse_args()
    if cli.stop_after_step is not None and cli.stop_after_step < 1:
        parser.error("--stop-after-step must be positive")
    if cli.vocabulary_order is not None and cli.backend not in ("coupled", "quantile"):
        parser.error("--vocabulary-order applies to coupled and quantile backends")
    with cli.config.open("rb") as handle:
        cfg = tomllib.load(handle)
    training = DistillationConfig(output_dir=cli.output_dir, **cfg["training"])
    model, teacher = cfg["models"]["student"], cfg["models"]["teacher"]
    student_tokenizer = AutoTokenizer.from_pretrained(
        model,
        revision=(training.model_init_kwargs or {}).get("revision", "main"),
        padding_side="left",
    )
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        teacher, revision=(training.teacher_model_init_kwargs or {}).get("revision", "main")
    )
    if student_tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError("Full-distribution distillation requires identical token-to-ID mappings")
    with training.main_process_first(desc="Prepare prompts"):
        dataset = load_prompts(cfg["data"])
    kwargs = dict(
        model=model,
        teacher_model=teacher,
        args=training,
        train_dataset=dataset,
        processing_class=student_tokenizer,
    )
    if cli.backend == "upstream":
        trainer = DistillationTrainer(**kwargs)
    else:
        trainer = CouplingTrainer(
            **kwargs,
            loss_backend=cli.backend if cli.backend in ("liger", "taopd", "sctaopd") else "sparse",
            estimator=LossConfig(
                **cfg["estimator"],
                method=cli.backend
                if cli.backend in ("coordinate", "independent", "coupled", "quantile")
                else "quantile",
            ),
            taopd_config=cfg.get("taopd"),
            vocabulary_order=cli.vocabulary_order,
        )
        trainer.prepare_sampling(cli.resume)
    if cli.stop_after_step is not None:
        trainer.add_callback(StopAfterStep(cli.stop_after_step))
    trainer.add_callback(StepTiming())
    if trainer.is_world_process_zero():
        output = Path(cli.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        manifest = dict(
            config=cfg,
            backend=cli.backend,
            sampling_scheme={
                "quantile": "stratified_quantiles",
                "coupled": "iid_shared_quantiles",
            }.get(cli.backend, cli.backend),
            estimator_state=trainer.estimator_state()
            if isinstance(trainer, CouplingTrainer)
            else None,
            vocabulary_order=trainer.vocabulary_order
            if isinstance(trainer, CouplingTrainer)
            else None,
            geometry_setup_seconds=trainer.geometry_setup_seconds
            if isinstance(trainer, CouplingTrainer)
            else 0.0,
            geometry_sha256=hashlib.sha256(trainer.order.cpu().numpy().tobytes()).hexdigest()
            if isinstance(trainer, CouplingTrainer) and trainer.order is not None
            else None,
            correction_sampling="fresh_current_student" if cli.backend == "sctaopd" else None,
            stop_after_step=cli.stop_after_step,
            dataset_fingerprint=dataset._fingerprint,
            dataset_rows=len(dataset),
            resume=str(cli.resume) if cli.resume else None,
        )
        (output / "experiment.json").write_text(json.dumps(manifest, indent=2) + "\n")
    result = trainer.train(resume_from_checkpoint=str(cli.resume) if cli.resume else None)
    trainer.save_model()
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
