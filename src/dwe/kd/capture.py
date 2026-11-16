"""Capture aligned output-head inputs from a checkpoint and the fixed teacher cache."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator

from ..geometry import validate_order
from .cache import HiddenDataset, tokenizer_digest


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--order", type=Path, required=True, help="Order saved at training initialization"
    )
    parser.add_argument("--positions", type=int, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = HiddenDataset(args.cache)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer_digest(tokenizer) != data.meta["tokenizer_sha256"]:
        raise ValueError("Checkpoint and teacher cache must use the same token IDs")
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .to("cuda")
        .eval()
    )
    head = model.get_output_embeddings()
    order = torch.load(args.order, map_location="cpu", weights_only=True)
    validate_order(order, head.weight.shape[0])
    remaining = args.positions
    states, teacher_states = [], []
    for batch in DataLoader(data, batch_size=2, collate_fn=default_data_collator):
        hidden = model.base_model(
            input_ids=batch["input_ids"].cuda(), use_cache=False
        ).last_hidden_state
        mask = (batch["labels"][:, 1:] != -100).cuda()
        h = hidden[:, :-1][mask][:remaining].cpu()
        states.append(h)
        teacher_states.append(batch["teacher_hidden"][mask.cpu()][:remaining])
        remaining -= len(h)
        if remaining == 0:
            break
    if remaining:
        raise ValueError("The cache has fewer valid positions than requested")
    teacher_head = torch.load(args.cache / "head.pt", weights_only=True, map_location="cpu")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "hidden": torch.cat(states),
            "teacher_hidden": torch.cat(teacher_states),
            "weight": head.weight.detach().cpu(),
            "teacher_weight": teacher_head["weight"],
            "bias": None if head.bias is None else head.bias.detach().cpu(),
            "teacher_bias": teacher_head["bias"],
            "order": order,
            "metadata": {
                "checkpoint": args.model,
                "cache": data.meta,
                "positions": args.positions,
                "order_source": str(args.order),
            },
        },
        args.output,
    )


if __name__ == "__main__":
    main()
