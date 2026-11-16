"""Matched head timing, estimator error and isolated incremental memory measurements."""

import argparse
import random
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from ..loss import LossConfig, distillation_loss, project, sparse_products
from ..objective import kl_adjoint
from ..reference import exact_loss
from ..sampling import sample_residual
from .cache import write_json


@dataclass(frozen=True)
class Variant:
    name: str
    method: str
    pairs: int = 8
    geometry: bool = False


VARIANTS = [
    Variant("full", "exact"),
    Variant("importance", "coordinate"),
    Variant("topg", "topk"),
    Variant("sketch", "sketch"),
    Variant("dwe", "quantile", geometry=True),
    Variant("independent_pairs", "independent"),
    Variant("natural", "quantile"),
    Variant("shared_iid", "coupled", geometry=True),
    *[Variant(f"dwe_k{k}", "quantile", k, True) for k in (1, 2, 4, 16)],
]


def load_capture(path, positions):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if len(data["hidden"]) < positions:
        raise ValueError("Capture is shorter than the requested measurement")
    for key in ("hidden", "teacher_hidden"):
        data[key] = data[key][:positions]
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            data[key] = value.cuda()
    data["hidden"].requires_grad_()
    data["weight"].requires_grad_()
    return data


def head_loss(data, variant):
    args = [data[k] for k in ("hidden", "weight", "teacher_hidden", "teacher_weight")]
    config = LossConfig(
        method="quantile" if variant.method == "exact" else variant.method,
        pairs=variant.pairs,
        chunk_size=256,
    )
    kwargs = dict(config=config, bias=data["bias"], teacher_bias=data["teacher_bias"])
    if variant.method == "exact":
        return exact_loss(*args, **kwargs)
    return distillation_loss(*args, order=data["order"] if variant.geometry else None, **kwargs)


def timed_step(data, variant):
    start, middle, end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    start.record()
    loss = head_loss(data, variant)
    middle.record()
    gradients = torch.autograd.grad(loss, (data["hidden"], data["weight"]))
    end.record()
    end.synchronize()
    result = [start.elapsed_time(middle), middle.elapsed_time(end), start.elapsed_time(end)]
    del loss, gradients
    return result


@torch.no_grad()
def quality(data, repeats):
    h, w = data["hidden"].float(), data["weight"].float()
    residuals = []
    for start in range(0, len(h), 256):
        _, g = kl_adjoint(
            project(data["hidden"][start : start + 256], data["weight"], data["bias"]),
            project(
                data["teacher_hidden"][start : start + 256],
                data["teacher_weight"],
                data["teacher_bias"],
            ),
            "forward",
            1.0,
        )
        residuals.append(g)
    g = torch.cat(residuals)
    reference = (g @ w / len(h), g.T @ h / len(h))
    norms = [r.square().sum().item() for r in reference]
    results = {
        "full": {
            "relative_mse": [0.0, 0.0],
            "nonzero_coordinates": (g != 0).sum(1).float().mean().item(),
        }
    }
    for variant in VARIANTS[1:]:
        errors, nonzeros = [], []
        for seed in range(repeats):
            generator = torch.Generator(device=h.device).manual_seed(seed)
            ids, coefficients = [], []
            for residual in residuals:
                i, c = sample_residual(
                    residual,
                    variant.pairs,
                    variant.method,
                    data["order"] if variant.geometry else None,
                    generator,
                )
                ids.append(i)
                coefficients.append(c / len(h))
            ids, coefficients = torch.cat(ids), torch.cat(coefficients)
            gh, gw, _ = sparse_products(
                h, w, ids, coefficients, 256, shared_columns=variant.method == "sketch"
            )
            errors.append(
                [
                    ((a - b).square().sum().item() / norm if norm else None)
                    for a, b, norm in zip((gh, gw), reference, norms, strict=True)
                ]
            )
            ordered = torch.where(coefficients != 0, ids, -1).sort(dim=1).values
            count = (ordered[:, :1] >= 0).sum(1)
            count += ((ordered[:, 1:] != ordered[:, :-1]) & (ordered[:, 1:] >= 0)).sum(1)
            nonzeros.append(count.float().mean().item())
            del gh, gw
        results[variant.name] = {
            "relative_mse": [
                statistics.mean(v) if v[0] is not None else None for v in zip(*errors, strict=True)
            ],
            "draws": errors,
            "nonzero_coordinates": statistics.mean(nonzeros),
        }
    return results


def memory(data, variant):
    for _ in range(3):
        timed_step(data, variant)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    peaks = []
    for _ in range(3):
        torch.cuda.reset_peak_memory_stats()
        timed_step(data, variant)
        peaks.append(torch.cuda.max_memory_allocated())
    return {
        "baseline_bytes": base,
        "peak_bytes": peaks,
        "incremental_peak_bytes": [peak - base for peak in peaks],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positions", type=int, nargs="+", default=[1024, 4096, 8192])
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--draws", type=int, default=128)
    parser.add_argument(
        "--memory-method", choices=[v.name for v in VARIANTS[:5]], help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if min(*args.positions, args.repeats, args.draws) <= 0 or args.warmup < 0:
        raise ValueError("Positions and repeat counts must be positive")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(1337)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.memory_method:
        variant = next(v for v in VARIANTS if v.name == args.memory_method)
        write_json(
            args.output / f"memory-{args.positions[0]}-{variant.name}.json",
            memory(load_capture(args.capture, args.positions[0]), variant),
        )
        return
    for n in args.positions:
        data = load_capture(args.capture, n)
        variants = VARIANTS if n == args.positions[0] else VARIANTS[:5]
        for variant in variants:
            for _ in range(args.warmup):
                timed_step(data, variant)
        raw = {v.name: [] for v in variants}
        rng = random.Random(1337)
        for _ in range(args.repeats):
            shuffled = list(variants)
            rng.shuffle(shuffled)
            for variant in shuffled:
                raw[variant.name].append(timed_step(data, variant))
        write_json(
            args.output / f"timing-{n}.json",
            {
                "positions": n,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "columns": ["forward_ms", "backward_ms", "total_ms"],
                "samples": raw,
                "means": {
                    key: [statistics.mean(x) for x in zip(*rows, strict=True)]
                    for key, rows in raw.items()
                },
                "capture": data["metadata"],
            },
        )
        del data
        torch.cuda.empty_cache()
        for variant in VARIANTS[:5]:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "dwe.kd.measure",
                    "--capture",
                    str(args.capture),
                    "--output",
                    str(args.output),
                    "--positions",
                    str(n),
                    "--memory-method",
                    variant.name,
                ],
                check=True,
            )
    data = load_capture(args.capture, args.positions[0])
    write_json(
        args.output / "quality.json",
        {
            "positions": args.positions[0],
            "repeats": args.draws,
            "precision": "BF16 projections; FP32 analytic adjoints and gradient products",
            "columns": ["hidden_relative_mse", "head_relative_mse"],
            "results": quality(data, args.draws),
            "capture": data["metadata"],
        },
    )


if __name__ == "__main__":
    main()
