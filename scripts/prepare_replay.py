"""Generate and save real student trajectories once for controlled update timing."""

import argparse
import hashlib
import json
import random
import tomllib
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--student", required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    cfg = tomllib.loads(a.config.read_text())
    spec, protocol = cfg["trajectories"], cfg["protocol"]
    rows = []
    sources = {}
    for f in sorted(a.data.glob("*.parquet")):
        with f.open("rb") as handle:
            sources[f.name] = hashlib.file_digest(handle, "sha256").hexdigest()
        rows.extend(pq.read_table(f, columns=["domain", "conversations"]).to_pylist())
    random.Random(protocol["seed"]).shuffle(rows)
    tokenizer = AutoTokenizer.from_pretrained(a.student, padding_side="left")
    chosen = []
    for domain in protocol["domains"]:
        subset = [r for r in rows if r["domain"] == domain][: spec["examples_per_domain"]]
        if len(subset) != spec["examples_per_domain"]:
            raise ValueError(f"Insufficient {domain} prompts")
        for row in subset:
            prompt = [x["value"] for x in row["conversations"] if x["from"] in ("human", "user")][
                -1
            ]
            messages = [
                {"role": "system", "content": spec["system_prompt"]},
                {"role": "user", "content": prompt},
            ]
            ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=spec["enable_thinking"],
                return_dict=False,
            )
            chosen.append({"domain": domain, "prompt_ids": ids})
    max_len = max(len(x["prompt_ids"]) for x in chosen) + spec["max_tokens"]
    llm = LLM(
        model=a.student,
        dtype="bfloat16",
        max_model_len=max_len,
        gpu_memory_utilization=0.8,
        max_num_seqs=32,
        enforce_eager=True,
        seed=protocol["seed"],
    )
    outputs = llm.generate(
        [{"prompt_token_ids": x["prompt_ids"]} for x in chosen],
        SamplingParams(
            temperature=spec["temperature"], max_tokens=spec["max_tokens"], seed=protocol["seed"]
        ),
    )
    for row, output in zip(chosen, outputs, strict=True):
        row["completion_ids"] = list(output.outputs[0].token_ids)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(
        json.dumps(
            {
                "student": a.student,
                "sources": sources,
                "config": cfg,
                "pad_token_id": tokenizer.pad_token_id,
                "trajectories": chosen,
            }
        )
    )
    for domain in protocol["domains"]:
        lengths = [len(x["completion_ids"]) for x in chosen if x["domain"] == domain]
        print(json.dumps({"domain": domain, "completion_lengths": lengths}), flush=True)


if __name__ == "__main__":
    main()
