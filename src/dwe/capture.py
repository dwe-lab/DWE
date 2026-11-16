"""Capture aligned real-model hidden states through Hugging Face APIs."""

import argparse
import gc
import hashlib
import json
import tomllib
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with args.config.open("rb") as handle:
        cfg = tomllib.load(handle)
    data_cfg = cfg["data"].copy()
    text_field = data_cfg.pop("text_field")
    examples = data_cfg.pop("examples")
    data = load_dataset(**data_cfg)
    texts = [row[text_field] for row in data.select(range(min(examples, len(data))))]
    student_cfg, teacher_cfg = cfg["student"], cfg["teacher"]
    tokenizers = [
        AutoTokenizer.from_pretrained(c["id"], revision=c["revision"])
        for c in (student_cfg, teacher_cfg)
    ]
    if tokenizers[0].get_vocab() != tokenizers[1].get_vocab():
        raise ValueError("Student and teacher must use the identical token-to-id mapping")
    run = cfg["capture"]
    sequences = [
        tokenizers[0](text, return_tensors="pt", truncation=True, max_length=run["max_length"])
        for text in texts
    ]
    # The exact same token IDs and positions are used for both models.
    inputs = []
    remaining = run["positions"]
    for sequence in sequences:
        count = min(sequence["input_ids"].shape[1] - 1, remaining)
        if count > 0:
            inputs.append((sequence, count))
            remaining -= count
        if remaining == 0:
            break
    if not inputs:
        raise ValueError("No next-token positions in the selected texts")
    result, revisions = {}, {}
    for role, model_cfg in (("student", student_cfg), ("teacher", teacher_cfg)):
        model = (
            AutoModelForCausalLM
            .from_pretrained(
                model_cfg["id"],
                revision=model_cfg["revision"],
                dtype=getattr(torch, run["dtype"]),
                device_map={"": run["device"]},
            )
            .eval()
        )
        head = model.get_output_embeddings()
        if not isinstance(head, torch.nn.Linear):
            raise TypeError("This experiment requires an ordinary linear output head")
        if head.weight.shape[0] <= max(tokenizers[0].get_vocab().values()):
            raise ValueError("The output head does not cover the tokenizer's token IDs")
        states = []
        with torch.no_grad():
            for sequence, count in inputs:
                output = model(
                    **{k: v.to(run["device"]) for k, v in sequence.items()},
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                if output.hidden_states is None:
                    raise ValueError("Selected model API did not expose decoder hidden states")
                states.append(output.hidden_states[-1][0, :count].detach().cpu())
                del output
        prefix = "" if role == "student" else "teacher_"
        result[prefix + "hidden"] = torch.cat(states)
        result[prefix + "weight"] = head.weight.detach().cpu()
        result[prefix + "bias"] = None if head.bias is None else head.bias.detach().cpu()
        revisions[role] = getattr(model.config, "_commit_hash", None)
        del head, model
        gc.collect()
        if torch.device(run["device"]).type == "cuda":
            torch.cuda.empty_cache()
    if result["weight"].shape[0] != result["teacher_weight"].shape[0]:
        raise ValueError("Student and teacher output vocabularies require explicit alignment")
    result["metadata"] = dict(
        kind="real-model-capture",
        config=cfg,
        resolved_model_revisions=revisions,
        texts_sha256=hashlib.sha256(json.dumps(texts).encode()).hexdigest(),
        positions=result["hidden"].shape[0],
        prefix_protocol="raw text; predict token t+1 from hidden state t",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(f"Wrote {args.output}; {result['metadata']['positions']} aligned positions")


if __name__ == "__main__":
    main()
