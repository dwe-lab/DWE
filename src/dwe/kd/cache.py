"""Prepare packed text and cache the frozen teacher's final hidden states."""

import argparse
import hashlib
import importlib.util
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def tokenizer_digest(tokenizer):
    data = json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()
    return hashlib.sha256(data).hexdigest()


class HiddenDataset(Dataset):
    """Read each cached BF16 sequence in one contiguous file operation."""

    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "cache.json").read_text())
        if not self.meta["complete"]:
            raise ValueError("Teacher cache is incomplete")
        self.ids = np.load(self.path / "input_ids.npy", mmap_mode="r")
        self.labels = np.load(self.path / "labels.npy", mmap_mode="r")
        self.hidden_shape = tuple(self.meta["shape"][1:])
        self.hidden_size = int(np.prod(self.hidden_shape))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        hidden = np.fromfile(
            self.path / "hidden.bf16",
            dtype=np.uint16,
            count=self.hidden_size,
            offset=index * self.hidden_size * 2,
        ).reshape(self.hidden_shape)
        return {
            "input_ids": torch.tensor(np.array(self.ids[index]), dtype=torch.long),
            "labels": torch.tensor(np.array(self.labels[index]), dtype=torch.long),
            "teacher_hidden": torch.from_numpy(hidden).view(torch.bfloat16),
        }


def pack(args):
    """Reuse SE-KD's EOS/boundary masking, with document-disjoint fixed splits."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    spec = importlib.util.spec_from_file_location(
        "sekd_dataset", args.sekd / "sekd/data/dataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stream = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=args.subset,
        revision=args.data_revision,
        split="train",
        streaming=True,
    ).shuffle(seed=args.seed, buffer_size=10_000)
    texts = {name: [] for name in ("train", "validation", "test")}
    counts = dict.fromkeys(texts, 0)
    budgets = {"train": args.tokens, "validation": args.eval_tokens, "test": args.eval_tokens}
    seen = set()
    for row in stream:
        text = row["text"]
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        bucket = int(digest[:8], 16) % 100
        name = "validation" if bucket == 0 else "test" if bucket == 1 else "train"
        if counts[name] >= budgets[name]:
            continue
        texts[name].append(text)
        counts[name] += len(tokenizer.encode(text, add_special_tokens=False)) + 1
        if all(counts[k] >= budgets[k] for k in budgets):
            break
    if any(counts[k] < budgets[k] for k in budgets):
        raise ValueError("Dataset exhausted before reaching token budgets")
    args.output.mkdir(parents=True, exist_ok=False)
    for name in texts:
        packed = module.PackedTokenDataset(texts[name], tokenizer, args.length)
        rows = budgets[name] // args.length
        target = args.output / name
        target.mkdir()
        np.save(target / "input_ids.npy", packed.input_ids[:rows].numpy().astype(np.int32))
        np.save(target / "labels.npy", packed.labels[:rows].numpy().astype(np.int32))
        write_json(
            target / "tokens.json",
            {
                "dataset": "HuggingFaceFW/fineweb-edu",
                "revision": args.data_revision,
                "subset": args.subset,
                "seed": args.seed,
                "split": name,
                "tokenizer_sha256": tokenizer_digest(tokenizer),
                "rows": rows,
                "length": args.length,
                "packing": "SE-KD PackedTokenDataset; EOS and post-EOS targets masked",
            },
        )


@torch.no_grad()
def build(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("A GPU allocation is required to build teacher hidden-state caches")
    token_meta = json.loads((args.tokens / "tokens.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer_digest(tokenizer) != token_meta["tokenizer_sha256"]:
        raise ValueError("Teacher tokenizer differs from the packed-text tokenizer")
    ids = np.load(args.tokens / "input_ids.npy", mmap_mode="r")
    if getattr(args, "max_rows", None) is not None:
        if args.max_rows < 1:
            raise ValueError("max_rows must be positive")
        ids = ids[: args.max_rows]
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(args.model, revision=args.revision)
    shape = (len(ids), ids.shape[1] - 1, config.hidden_size)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output / "cache.json"
    meta = {
        **token_meta,
        "teacher": args.model,
        "teacher_revision": args.revision,
        "shape": list(shape),
        "rows": len(ids),
        "complete": False,
        "rows_written": 0,
        "seconds": 0.0,
    }
    for name in ("input_ids.npy", "labels.npy"):
        with (args.tokens / name).open("rb") as handle:
            meta[name + "_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    if metadata_path.exists():
        saved = json.loads(metadata_path.read_text())
        for key in (
            "shape",
            "teacher_revision",
            "teacher",
            "input_ids.npy_sha256",
            "labels.npy_sha256",
        ):
            if saved[key] != meta[key]:
                raise ValueError(f"Existing cache differs in {key}")
        if saved["complete"]:
            print("Teacher cache already complete")
            return
        meta = saved
    else:
        required = int(np.prod(shape)) * 2 + config.vocab_size * config.hidden_size * 2
        if shutil.disk_usage(args.output).free < required + 10 * 1024**3:
            raise RuntimeError(f"Need {required / 1e9:.1f} GB plus 10 GiB free to build this cache")
        np.save(args.output / "input_ids.npy", ids)
        labels = np.load(args.tokens / "labels.npy", mmap_mode="r")[: len(ids)]
        np.save(args.output / "labels.npy", labels)
        write_json(metadata_path, meta)
    teacher = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            revision=args.revision,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to(args.device)
        .eval()
        .requires_grad_(False)
    )
    head = teacher.get_output_embeddings()
    torch.save(
        {"weight": head.weight.cpu(), "bias": None if head.bias is None else head.bias.cpu()},
        args.output / "head.pt",
    )
    mode = "r+" if (args.output / "hidden.bf16").exists() else "w+"
    storage = np.memmap(args.output / "hidden.bf16", dtype=np.uint16, mode=mode, shape=shape)
    started = time.perf_counter()
    old_seconds = meta["seconds"]
    for start in range(meta["rows_written"], len(ids), args.batch_size):
        batch = torch.tensor(np.array(ids[start : start + args.batch_size]), device=args.device)
        hidden = teacher.base_model(input_ids=batch, use_cache=False).last_hidden_state[:, :-1]
        storage[start : start + len(batch)] = hidden.contiguous().cpu().view(torch.uint16).numpy()
        storage.flush()
        meta.update(
            rows_written=start + len(batch), seconds=old_seconds + time.perf_counter() - started
        )
        write_json(metadata_path, meta)
        if start % 1024 == 0:
            print(f"Cached {meta['rows_written']}/{len(ids)} sequences", flush=True)
    meta["complete"] = True
    write_json(metadata_path, meta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("pack")
    data.add_argument("--sekd", type=Path, required=True)
    data.add_argument("--data-revision", required=True)
    data.add_argument("--subset", default="sample-10BT")
    data.add_argument("--seed", type=int, default=2026)
    data.add_argument("--tokens", type=int, default=100_007_936)
    data.add_argument("--eval-tokens", type=int, default=1_048_576)
    data.add_argument("--length", type=int, default=512)
    teacher = commands.add_parser("teacher")
    teacher.add_argument("--tokens", type=Path, required=True)
    teacher.add_argument("--device", default="cuda")
    teacher.add_argument("--batch-size", type=int, default=8)
    teacher.add_argument("--max-rows", type=int, help="Cache a fixed prefix for a short pilot")
    for sub in (data, teacher):
        sub.add_argument("--model", required=True)
        sub.add_argument("--revision", required=True)
        sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (pack if args.command == "pack" else build)(args)


if __name__ == "__main__":
    main()
