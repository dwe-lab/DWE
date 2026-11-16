"""Evaluate a saved KD checkpoint with exact held-out KL/PPL and lm-eval tasks."""

import argparse
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from ..loss import LossConfig
from .cache import HiddenDataset, tokenizer_digest, write_json
from .train import KDTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, help="Held-out teacher cache for KL and perplexity")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--tasks", default="hellaswag,piqa,arc_easy,arc_challenge,gsm8k,lambada_openai,mmlu"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.cache:
        data = HiddenDataset(args.cache)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer_digest(tokenizer) != data.meta["tokenizer_sha256"]:
            raise ValueError("Evaluation cache and checkpoint have different token IDs")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        trainer = KDTrainer(
            model=model,
            args=TrainingArguments(
                output_dir=str(args.output),
                bf16=True,
                report_to="none",
                disable_tqdm=True,
                per_device_eval_batch_size=args.batch_size,
            ),
            eval_dataset=data,
            estimator=LossConfig(chunk_size=256),
            method="exact",
            head=torch.load(args.cache / "head.pt", map_location="cpu", weights_only=True),
        )
        write_json(args.output / "heldout.json", trainer.evaluate())
        del trainer, model
        torch.cuda.empty_cache()
    if args.tasks:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "lm_eval",
                "run",
                "--model",
                "hf",
                "--model_args",
                f"pretrained={args.model},dtype=bfloat16,attn_implementation=sdpa",
                "--tasks",
                args.tasks,
                "--num_fewshot",
                "0",
                "--batch_size",
                str(args.batch_size),
                "--device",
                "cuda:0",
                "--seed",
                "0,1234,1234,1234",
                "--gen_kwargs",
                "do_sample=False,max_gen_toks=512",
                "--output_path",
                str(args.output / "benchmarks"),
                "--log_samples",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
