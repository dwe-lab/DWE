"""A fixed-hidden-state benchmark; model capture and GPU training are separate."""

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
import tomllib
from dataclasses import replace
from pathlib import Path

import torch

from .geometry import principal_order
from .loss import LossConfig, distillation_loss
from .reference import exact_loss


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def run(settings: dict) -> dict:
    exp = settings["experiment"]
    device = torch.device(exp["device"])
    dtype = getattr(torch, exp["dtype"])
    torch.manual_seed(exp["seed"])
    if "threads" in exp:
        torch.set_num_threads(exp["threads"])
    if exp["repeats"] < 1 or exp["warmup"] < 0:
        raise ValueError("repeats must be positive and warmup nonnegative")
    source = settings["source"]
    if source["kind"] == "synthetic":
        n, v, d = source["positions"], source["vocabulary"], source["hidden_size"]
        w = torch.randn(v, d, device=device, dtype=dtype) / d**0.5
        h = torch.randn(n, d, device=device, dtype=dtype)
        tw = w + source["teacher_perturbation"] * torch.randn_like(w) / d**0.5
        th = h.clone()
        bias = teacher_bias = None
        metadata = {"kind": "synthetic", "quality_claim": False}
    elif source["kind"] == "capture":
        data = torch.load(source["path"], map_location="cpu", weights_only=True)
        h, w, th, tw = [
            data[k].to(device=device, dtype=dtype)
            for k in ("hidden", "weight", "teacher_hidden", "teacher_weight")
        ]
        bias = data["bias"]
        teacher_bias = data["teacher_bias"]
        bias = None if bias is None else bias.to(device=device, dtype=dtype)
        teacher_bias = None if teacher_bias is None else teacher_bias.to(device=device, dtype=dtype)
        metadata = data["metadata"]
    else:
        raise ValueError("source.kind must be synthetic or capture")
    h.requires_grad_()
    w.requires_grad_(exp["train_head"])
    if bias is not None:
        bias.requires_grad_(exp["train_head"])
    config = LossConfig(**settings["loss"])
    common = dict(bias=bias, teacher_bias=teacher_bias)
    parameters = (h, w) if w.requires_grad else (h,)
    if bias is not None and bias.requires_grad:
        parameters += (bias,)

    generator = torch.Generator(device=device).manual_seed(exp["seed"])
    orders = exp.get("vocabulary_orders", ["principal"])
    if not orders or any(order not in ("principal", "natural") for order in orders):
        raise ValueError("vocabulary_orders must contain principal and/or natural")
    order = None
    geometry_seconds = 0.0
    if "principal" in orders and any(m in ("coupled", "quantile") for m in exp["methods"]):
        synchronize(device)
        started = time.perf_counter()
        order = principal_order(w)
        synchronize(device)
        geometry_seconds = time.perf_counter() - started

    reference_value = exact_loss(h, w, th, tw, config=config, **common)
    reference = tuple(g.detach() for g in torch.autograd.grad(reference_value, parameters))
    reference_norms = [g.float().square().sum().item() for g in reference]

    def step(method, pair_count, vocabulary_order):
        cfg = replace(config, pairs=pair_count)
        if method == "exact":
            value = exact_loss(h, w, th, tw, config=cfg, **common)
        elif method == "liger":
            from .liger import liger_loss

            value = liger_loss(h, w, th, tw, config=cfg, **common)
        else:
            cfg = replace(cfg, method=method)
            value = distillation_loss(
                h,
                w,
                th,
                tw,
                config=cfg,
                generator=generator,
                order=order if vocabulary_order == "principal" else None,
                **common,
            )
        return value.detach(), torch.autograd.grad(value, parameters)

    records = []
    variants = [
        (method, vocabulary_order)
        for method in exp["methods"]
        for vocabulary_order in (orders if method in ("coupled", "quantile") else [None])
    ]
    for method, vocabulary_order in variants:
        budgets = [config.pairs] if method in ("exact", "liger") else exp["pair_budgets"]
        for pair_count in budgets:
            generator.manual_seed(exp["seed"])
            for _ in range(exp["warmup"]):
                step(method, pair_count, vocabulary_order)
            synchronize(device)
            if device.type == "cuda":
                baseline_memory = torch.cuda.memory_allocated(device)
            durations, errors, values, peaks = [], [], [], []
            for _ in range(exp["repeats"]):
                synchronize(device)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                start = time.perf_counter()
                value, gradients = step(method, pair_count, vocabulary_order)
                synchronize(device)
                durations.append(time.perf_counter() - start)
                if device.type == "cuda":
                    peaks.append(torch.cuda.max_memory_allocated(device))
                # Measurement work stays outside the timed head operation.
                errors.append(
                    [
                        (g.float() - ref.float()).square().sum().item()
                        for g, ref in zip(gradients, reference, strict=True)
                    ]
                )
                values.append(value.item())
                del gradients, value
            names = ["hidden", "head"][: len(parameters)]
            if len(parameters) == 3:
                names.append("bias")
            record = dict(
                method=method,
                vocabulary_order=vocabulary_order,
                pairs=None if method in ("exact", "liger") else pair_count,
                median_head_seconds=statistics.median(durations),
                mean_kl=statistics.mean(values),
            )
            for index, name in enumerate(names):
                mse = statistics.mean(row[index] for row in errors)
                norm = reference_norms[index]
                record[f"{name}_gradient_mse"] = mse
                record[f"{name}_relative_mse"] = mse / norm if norm else None
            if device.type == "cuda":
                record["peak_allocated_bytes"] = max(peaks)
                record["baseline_allocated_bytes"] = baseline_memory
            records.append(record)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    return dict(
        settings=settings,
        source=metadata,
        geometry_setup_seconds=geometry_seconds,
        geometry_sha256=hashlib.sha256(order.cpu().numpy().tobytes()).hexdigest()
        if order is not None
        else None,
        torch_version=str(torch.__version__),
        python_version=platform.python_version(),
        git_revision=revision.stdout.strip() if revision.returncode == 0 else None,
        git_dirty=bool(status.stdout.strip()) if status.returncode == 0 else None,
        device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        records=records,
        scope="head forward + backward; excludes model backbone and optimizer",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open("rb") as handle:
        result = run(tomllib.load(handle))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
