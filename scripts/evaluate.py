"""Run pinned upstream benchmark adapters; preserve their prompts and scoring."""

import argparse
import hashlib
import importlib.util
import json
import logging
import pickle
import sys
import tomllib
from pathlib import Path


def load_task(config, name, evalchemy, opd2, data_root=None):
    # BaseBenchmark uses these registered model types for dispatch.
    importlib.import_module("lm_eval.models.huggingface")
    importlib.import_module("lm_eval.models.openai_completions")

    sys.path.insert(0, str(evalchemy / "eval/chat_benchmarks/HMMT"))
    entry = config["benchmark"][name]
    root = (
        opd2 / "evaluation/chat_benchmarks"
        if entry["provider"] == "opd2"
        else evalchemy / "eval/chat_benchmarks"
    )
    source = root / name / "eval_instruct.py"
    module_name = f"eval.chat_benchmarks.{name}.eval_instruct"
    spec = importlib.util.spec_from_file_location(module_name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    kwargs = dict(entry.get("kwargs", {}))
    if "data_file" in kwargs:
        kwargs["data_file"] = str(evalchemy / kwargs["data_file"])
    if data_root is not None and entry.get("dataset", False):
        kwargs["dataset_name"] = str((data_root / name).resolve())
    return getattr(module, entry["class"])(**kwargs), source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--evalchemy", type=Path, required=True)
    parser.add_argument("--opd2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--mode", choices=["preflight", "generate", "score"], required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    config = tomllib.loads(args.config.read_text())
    task, source = load_task(config, args.task, args.evalchemy, args.opd2, args.data_root)
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / f"protocol-{args.repeat:02d}.json"
    saved = json.loads(protocol_path.read_text()) if args.mode == "score" else None
    # Native scoring consumes saved examples. Only RG also needs its task generators.
    if args.mode != "score" or hasattr(task, "TASK_NAMES"):
        questions = task.load_questions()
        if not len(questions):
            raise ValueError("Empty benchmark")
        if hasattr(task, "TASK_NAMES"):
            present = {q["task_name"] for q in questions}
            if present != set(task.TASK_NAMES):
                raise ValueError(f"Incomplete upstream task set: {present} != {task.TASK_NAMES}")
        # One generation pass; preserve data seeds before setting the generation seed.
        task.load_questions = lambda: questions
        loaded_count = len(questions)
        count = config["benchmark"][args.task].get("expected_questions", loaded_count)
    else:
        count = saved["questions"]
        loaded_count = saved["loaded_questions"]
    protocol = {
        "task": args.task,
        "questions": count,
        "loaded_questions": loaded_count,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "checkpoint": args.checkpoint,
        "repeat": args.repeat,
        "config": config,
    }
    if args.data_root is not None:
        manifest = args.data_root / args.task / "manifest.json"
        if manifest.exists():
            protocol["data"] = json.loads(manifest.read_text())
    protocol["max_new_tokens"] = task.max_new_tokens
    if args.mode == "preflight":
        (args.output / "preflight.json").write_text(json.dumps(protocol, indent=2))
        print(json.dumps({"task": args.task, "questions": count}))
        return
    task.n_repeat = 1
    task.seed = [s + args.repeat for s in task.seed]
    artifact_format = config["benchmark"][args.task].get("artifact_format", "json")
    if artifact_format not in ("json", "pickle"):
        raise ValueError(f"Unsupported artifact format: {artifact_format}")
    suffix = "pkl" if artifact_format == "pickle" else "json"
    generation_path = args.output / f"generation-{args.repeat:02d}.{suffix}"
    if args.mode == "generate":
        from lm_eval.models.vllm_causallms import VLLM

        class SamplingVLLM(VLLM):
            def generate_until(self, requests, **kwargs):
                for request in requests:
                    request.args[1]["do_sample"] = request.args[1].get("temperature", 0) > 0
                return super().generate_until(requests, **kwargs)

        if generation_path.exists():
            raise FileExistsError(generation_path)
        model_config = config["model"] | config["benchmark"][args.task].get("model", {})
        model = SamplingVLLM(pretrained=args.checkpoint, **model_config)
        results = task.generate_responses(model)
        if len(results["examples"]) != count:
            raise ValueError("Generated example count differs from the configured benchmark")
        temporary = generation_path.with_suffix(".tmp")
        if artifact_format == "pickle":
            with temporary.open("wb") as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with temporary.open("w") as handle:
                json.dump(results, handle, ensure_ascii=False)
        temporary.replace(generation_path)
        (args.output / f"protocol-{args.repeat:02d}.json").write_text(
            json.dumps(protocol, indent=2)
        )
    else:
        for key in ("task", "questions", "source_sha256", "config", "data"):
            if saved.get(key) != protocol.get(key):
                raise ValueError(f"Scoring protocol differs from generation: {key}")
        if artifact_format == "pickle":
            # Only load trusted artifacts generated by this evaluation pipeline.
            with generation_path.open("rb") as handle:
                results = pickle.load(handle)
        else:
            with generation_path.open() as handle:
                results = json.load(handle)
        if len(results["examples"]) != count:
            raise ValueError("Generation does not cover the recorded benchmark")
        scored = task.evaluate_responses(results)
        if artifact_format == "pickle":
            with (args.output / f"result-{args.repeat:02d}.pkl").open("wb") as handle:
                pickle.dump(scored, handle, protocol=pickle.HIGHEST_PROTOCOL)
            scored = {k: v for k, v in scored.items() if k != "examples"}
        with (args.output / f"result-{args.repeat:02d}.json").open("w") as handle:
            json.dump(scored, handle, ensure_ascii=False)
        print(json.dumps({k: v for k, v in scored.items() if k != "examples"}))


if __name__ == "__main__":
    main()
