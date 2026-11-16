"""Measure the actual TRL loss boundary on fixed captured states."""

import argparse
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from trl.trainer.distillation_trainer import _chunked_divergence_loss

from dwe.geometry import principal_order
from dwe.loss import LossConfig
from dwe.trainer import CouplingTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--positions", type=int, nargs="+", required=True)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--upstream-chunk-size", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocabulary-order", choices=["principal", "natural"], default="principal")
    parser.add_argument("--direction", choices=["forward", "reverse"], default="reverse")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if min(args.positions) < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error("positions/repeats must be positive and warmup nonnegative")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    data = torch.load(args.capture, map_location="cpu", weights_only=True)
    h0, w, ht0, wt = [
        data[k].to(device=device, dtype=torch.bfloat16)
        for k in ("hidden", "weight", "teacher_hidden", "teacher_weight")
    ]
    w.requires_grad_()
    b = None if data["bias"] is None else data["bias"].to(device).requires_grad_()
    bt = None if data["teacher_bias"] is None else data["teacher_bias"].to(device)
    torch.cuda.synchronize()
    started = time.perf_counter()
    order = principal_order(w) if args.vocabulary_order == "principal" else None
    torch.cuda.synchronize()
    geometry_seconds = time.perf_counter() - started
    result = dict(
        scope="Fixed captured states without token repetition; excludes backbone, rollout, optimizer and distributed communication. Not task-quality evidence.",
        vocabulary_order=args.vocabulary_order,
        geometry_setup_seconds=geometry_seconds,
        student_shape=list(w.shape),
        teacher_shape=list(wt.shape),
        torch_version=torch.__version__,
        gpu=torch.cuda.get_device_name(device),
        config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        records=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for n in args.positions:
        if n > len(h0):
            raise ValueError("Capture has fewer distinct positions than requested")
        ids = torch.arange(n, device=device)
        h = h0[ids].unsqueeze(0).detach().requires_grad_()
        ht = ht0[ids].unsqueeze(0)
        mask = torch.ones((1, n), device=device, dtype=torch.long)
        for method in ("upstream", "quantile", "taopd"):
            adapter = SimpleNamespace(
                estimator=LossConfig(
                    method="quantile",
                    pairs=args.pairs,
                    chunk_size=args.chunk_size,
                ),
                loss_backend="taopd" if method == "taopd" else "sparse",
                taopd_config={"top_k": args.top_k, "tail_epsilon": 1e-6},
                order=order,
            )

            def forward(h=h, ht=ht, mask=mask, method=method, adapter=adapter):
                common = dict(
                    student_lm_head_bias=b, teacher_lm_head_bias=bt, temperature=args.temperature
                )
                inputs = (
                    h,
                    ht,
                    w,
                    wt,
                    mask,
                    float(args.direction == "reverse"),
                    args.upstream_chunk_size,
                )
                if method == "upstream":
                    return _chunked_divergence_loss(*inputs, **common)[0]
                return CouplingTrainer.distribution_loss(adapter, *inputs, **common)[0]

            def clear(h=h):
                h.grad = None
                w.grad = None
                if b is not None:
                    b.grad = None

            for _ in range(args.warmup):
                clear()
                forward().backward()
            clear()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            baseline_memory = torch.cuda.memory_allocated(device)
            timings = []
            for _ in range(args.repeats):
                clear()
                begin_event, middle, end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                begin_event.record()
                loss = forward()
                middle.record()
                loss.backward()
                end.record()
                end.synchronize()
                timings.append(
                    [
                        begin_event.elapsed_time(middle),
                        middle.elapsed_time(end),
                        begin_event.elapsed_time(end),
                    ]
                )
            record = dict(
                method=method,
                positions=n,
                loss=float(loss.detach()),
                samples_ms=timings,
                median_ms={
                    key: statistics.median(row[i] for row in timings)
                    for i, key in enumerate(("forward", "backward", "total"))
                },
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                baseline_allocated_bytes=baseline_memory,
            )
            if args.profile:
                clear()
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=True,
                ) as prof:
                    forward().backward()
                    torch.cuda.synchronize()
                trace = args.output.parent / f"{args.output.stem}-{method}-{n}.trace.json"
                prof.export_chrome_trace(str(trace))
                record["trace"] = str(trace)
                record["operators"] = [
                    dict(
                        name=e.key,
                        device_us=e.self_device_time_total,
                        cpu_us=e.self_cpu_time_total,
                        calls=e.count,
                    )
                    for e in prof.key_averages()
                ]
            result["records"].append(record)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(method, n, record["median_ms"], flush=True)
            clear()


if __name__ == "__main__":
    main()
