"""Cached full-distribution KD; Transformers owns optimization and checkpointing."""

import argparse
import json
import math
import os
import time
import tomllib
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from ..geometry import principal_order, validate_order
from ..loss import LossConfig, distillation_loss, project
from ..objective import kl_adjoint
from ..reference import exact_loss
from .cache import HiddenDataset, tokenizer_digest, write_json


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class KDTrainer(Trainer):
    def __init__(
        self, *args, head, estimator, method, order=None, head_checkpointing=True, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if self.args.world_size != 1:
            raise ValueError("This entry point runs on one allocated GPU")
        self.head = {k: None if v is None else v.to(self.args.device) for k, v in head.items()}
        self.estimator, self.method, self.order = estimator, method, order
        self.head_checkpointing = head_checkpointing
        # compute_loss normalizes across the whole accumulation window.
        self.model_accepts_loss_kwargs = True
        self._loss_shifts_labels = True
        self.measured_seconds = 0.0
        self.measured_tokens = 0

    def _get_train_sampler(self, train_dataset=None):
        return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        # Start before dataloader reads, rather than after Trainer's prefetch.
        synchronize(self.args.device)
        self.update_started = time.perf_counter()
        if self.args.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.args.device)
        batches, count = super().get_batch_samples(epoch_iterator, num_batches, device)
        self.update_tokens = sum(batch["input_ids"].numel() for batch in batches)
        self.input_seconds = time.perf_counter() - self.update_started
        return batches, count

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        hidden = model.base_model(input_ids=inputs["input_ids"], use_cache=False).last_hidden_state
        mask = inputs["labels"][:, 1:] != -100
        h, th = hidden[:, :-1][mask], inputs["teacher_hidden"][mask]
        output = model.get_output_embeddings()
        common = dict(config=self.estimator, bias=output.bias, teacher_bias=self.head["bias"])
        if self.method == "exact":
            loss = exact_loss(
                h,
                output.weight,
                th,
                self.head["weight"],
                checkpoint_chunks=self.head_checkpointing,
                **common,
            )
        else:
            loss = distillation_loss(
                h, output.weight, th, self.head["weight"], order=self.order, **common
            )
        if num_items_in_batch is not None:
            loss = loss * mask.sum() / num_items_in_batch
        return (loss, {"loss": loss}) if return_outputs else loss

    @torch.no_grad()
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        was_training = self.model.training
        self.model.eval()
        totals = torch.zeros(3, dtype=torch.float64, device=self.args.device)
        loader = DataLoader(
            self.eval_dataset if eval_dataset is None else eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=default_data_collator,
        )
        output = self.model.get_output_embeddings()
        for batch in loader:
            batch = self._prepare_inputs(batch)
            hidden = self.model.base_model(
                input_ids=batch["input_ids"], use_cache=False
            ).last_hidden_state
            labels = batch["labels"][:, 1:]
            mask = labels != -100
            h, th, targets = hidden[:, :-1][mask], batch["teacher_hidden"][mask], labels[mask]
            for start in range(0, len(h), self.estimator.chunk_size):
                stop = start + self.estimator.chunk_size
                logits = project(h[start:stop], output.weight, output.bias)
                teacher = project(th[start:stop], self.head["weight"], self.head["bias"])
                values, _ = kl_adjoint(logits, teacher, "forward", 1.0)
                totals[0] += values.double().sum()
                totals[1] += F.cross_entropy(
                    logits.float(), targets[start:stop], reduction="sum"
                ).double()
                totals[2] += len(logits)
        kl, nll = (totals[:2] / totals[2]).tolist()
        metrics = {
            f"{metric_key_prefix}_loss": kl,
            f"{metric_key_prefix}_kl": kl,
            f"{metric_key_prefix}_nll": nll,
            f"{metric_key_prefix}_ppl": math.exp(nll),
            "training_seconds": self.measured_seconds,
            "input_tokens": self.measured_tokens,
        }
        self.log(metrics)
        with (Path(self.args.output_dir) / "quality.jsonl").open("a") as handle:
            handle.write(json.dumps({"step": self.state.global_step, **metrics}) + "\n")
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        self.model.train(was_training)
        return metrics


class Measurements(TrainerCallback):
    """Record complete updates and export the token/time endpoints of one trajectory."""

    def __init__(self, trainer, token_budget, time_budget):
        self.trainer = trainer
        self.token_budget, self.time_budget = token_budget, time_budget
        self.endpoints = {}
        self.pending = []

    def on_step_end(self, args, state, control, **kwargs):
        trainer = self.trainer
        synchronize(args.device)
        seconds = time.perf_counter() - trainer.update_started
        trainer.measured_seconds += seconds
        trainer.measured_tokens += trainer.update_tokens
        record = {
            "step": state.global_step,
            "seconds": seconds,
            "input_seconds": trainer.input_seconds,
            "cumulative_seconds": trainer.measured_seconds,
            "input_tokens": trainer.measured_tokens,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device)
            if args.device.type == "cuda"
            else None,
        }
        with (Path(args.output_dir) / "updates.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        for name, reached in (
            ("tokens", trainer.measured_tokens >= self.token_budget),
            ("time", self.time_budget is not None and trainer.measured_seconds >= self.time_budget),
        ):
            if reached and name not in self.endpoints:
                self.endpoints[name] = record
                self.pending.append(name)
                control.should_save = True
        done = "tokens" in self.endpoints and (self.time_budget is None or "time" in self.endpoints)
        if done:
            control.should_training_stop = True
        return control

    def on_save(self, args, state, control, **kwargs):
        trainer = self.trainer
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        write_json(
            checkpoint / "measurement.json",
            {
                "seconds": trainer.measured_seconds,
                "tokens": trainer.measured_tokens,
                "endpoints": self.endpoints,
            },
        )
        if trainer.order is not None:
            torch.save(trainer.order.cpu(), checkpoint / "order.pt")
        for name in [f"snapshots/step-{state.global_step}", *self.pending]:
            # Keep evaluation models when Trainer rotates optimizer checkpoints.
            destination = Path(args.output_dir) / name
            destination.mkdir(parents=True, exist_ok=True)
            for path in checkpoint.iterdir():
                if path.name in {
                    "config.json",
                    "generation_config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "vocab.json",
                    "merges.txt",
                    "chat_template.jinja",
                    "order.pt",
                    "measurement.json",
                } or path.name.endswith((".safetensors", ".safetensors.index.json")):
                    os.link(path, destination / path.name)
        self.pending.clear()
        write_json(Path(args.output_dir) / "endpoints.json", self.endpoints)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=["exact", "coordinate", "topk", "sketch", "quantile", "independent", "coupled"],
        required=True,
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", help="Local copy of the pinned student checkpoint")
    parser.add_argument("--tokens", type=int, help="Override the token budget for a short pilot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--order", choices=["principal", "natural"], default="principal")
    parser.add_argument("--time-budget", type=float)
    parser.add_argument("--resume", type=Path)
    cli = parser.parse_args()
    settings = tomllib.loads(cli.config.read_text())
    execution = settings.get("execution", {})
    if "cpu_threads" in execution:
        torch.set_num_threads(execution["cpu_threads"])
    args_dict = settings["training"].copy()
    tokens = settings["experiment"]["tokens"] if cli.tokens is None else cli.tokens
    if tokens <= 0 or (cli.time_budget is not None and cli.time_budget <= 0):
        raise ValueError("Training budgets must be positive")
    dataset, validation = HiddenDataset(cli.cache), HiddenDataset(cli.validation_cache)
    if dataset.meta["teacher_revision"] != validation.meta["teacher_revision"]:
        raise ValueError("Training and validation caches must use the same teacher")
    if cli.output.exists() and any(cli.output.iterdir()) and cli.resume is None:
        raise ValueError("Output already exists; use --resume to continue it")
    cli.output.mkdir(parents=True, exist_ok=True)
    set_seed(cli.seed)
    model_info = settings["model"]
    if dataset.meta["teacher_revision"] != model_info["teacher_revision"]:
        raise ValueError("The cache does not match the configured teacher revision")
    source = cli.model or model_info["name"]
    identity = {
        "settings": settings,
        "method": cli.method,
        "seed": cli.seed,
        "pairs": cli.pairs,
        "order": cli.order,
        "cache": str(cli.cache.resolve()),
        "validation_cache": str(cli.validation_cache.resolve()),
        "token_budget": tokens,
    }
    if cli.resume:
        saved_run = json.loads((cli.output / "run.json").read_text())
        if any(saved_run[key] != value for key, value in identity.items()):
            raise ValueError("Resume requires the same model, estimator, data and token budget")
        if saved_run["time_budget"] not in (None, cli.time_budget):
            raise ValueError("Resume cannot change an existing time budget")
        resumed_step = json.loads((cli.resume / "trainer_state.json").read_text())["global_step"]
        if any(
            int(p.name.split("-")[-1]) > resumed_step
            for p in (cli.output / "snapshots").glob("step-*")
        ):
            raise ValueError(
                "Resume predates saved evaluation models; use a separate run directory"
            )
        endpoints_path = cli.output / "endpoints.json"
        if endpoints_path.exists():
            exported = json.loads(endpoints_path.read_text())
            if any(value["step"] > resumed_step for value in exported.values()):
                raise ValueError(
                    "This checkpoint predates an exported endpoint; use a separate run directory"
                )
        for name in ("updates.jsonl", "quality.jsonl"):
            path = cli.output / name
            if path.exists():
                lines = [
                    line
                    for line in path.read_text().splitlines()
                    if json.loads(line)["step"] <= resumed_step
                ]
                path.write_text("\n".join(lines) + "\n")
        write_json(cli.output / "run.json", {**saved_run, "time_budget": cli.time_budget})
    tokenizer = AutoTokenizer.from_pretrained(source, revision=model_info["revision"])
    if tokenizer_digest(tokenizer) != dataset.meta["tokenizer_sha256"]:
        raise ValueError("Student token IDs differ from the teacher cache")
    model = AutoModelForCausalLM.from_pretrained(
        source,
        revision=model_info["revision"],
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    batch = args_dict["per_device_train_batch_size"] * args_dict["gradient_accumulation_steps"]
    tokens_per_step = batch * dataset.ids.shape[1]
    args_dict.update(
        output_dir=str(cli.output),
        seed=cli.seed,
        data_seed=2026,
        max_steps=len(dataset) // batch if cli.time_budget else math.ceil(tokens / tokens_per_step),
    )
    args = TrainingArguments(**args_dict)
    if args.device.type != "cuda":
        raise RuntimeError("KD training requires an allocated GPU; preparation can run without one")
    if len(dataset) * dataset.ids.shape[1] < tokens:
        raise ValueError("The teacher cache is shorter than the requested token budget")
    model.to(args.device)
    order = None
    geometry_seconds = 0.0
    if cli.method in ("quantile", "coupled") and cli.order == "principal":
        if cli.resume:
            order = torch.load(cli.resume / "order.pt", weights_only=True).to(args.device)
        else:
            synchronize(args.device)
            started = time.perf_counter()
            order = principal_order(model.get_output_embeddings().weight)
            synchronize(args.device)
            geometry_seconds = time.perf_counter() - started
        validate_order(order, model.config.vocab_size)
        if not cli.resume:
            torch.save(order.cpu(), cli.output / "order.pt")
    estimator = LossConfig(
        method="quantile" if cli.method == "exact" else cli.method,
        pairs=cli.pairs,
        **settings["loss"],
    )
    from bitsandbytes.optim import Adam8bit

    trainer = KDTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=validation,
        data_collator=default_data_collator,
        processing_class=tokenizer,
        head=torch.load(cli.cache / "head.pt", map_location="cpu", weights_only=True),
        estimator=estimator,
        method=cli.method,
        order=order,
        head_checkpointing=execution.get("head_checkpointing", True),
        optimizer_cls_and_kwargs=(
            Adam8bit,
            {
                "lr": args.learning_rate,
                "betas": (args.adam_beta1, args.adam_beta2),
                "eps": args.adam_epsilon,
                "weight_decay": args.weight_decay,
            },
        ),
    )
    measurement = Measurements(trainer, tokens, cli.time_budget)
    trainer.add_callback(measurement)
    if cli.resume:
        saved = json.loads((cli.resume / "measurement.json").read_text())
        trainer.measured_seconds, trainer.measured_tokens = saved["seconds"], saved["tokens"]
        measurement.endpoints = saved["endpoints"]
    else:
        write_json(
            cli.output / "run.json",
            {
                **identity,
                "time_budget": cli.time_budget,
                "geometry_seconds": geometry_seconds,
                "release": os.environ.get("DWE_RELEASE"),
                "time_endpoint": "first complete update at or after budget; actual time is reported",
            },
        )
    trainer.train(resume_from_checkpoint=str(cli.resume) if cli.resume else None)
    if "tokens" not in measurement.endpoints or (
        cli.time_budget and "time" not in measurement.endpoints
    ):
        raise RuntimeError("Cached data exhausted before reaching both endpoints; extend the cache")


if __name__ == "__main__":
    main()
