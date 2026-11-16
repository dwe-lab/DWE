import copy
import itertools
import json
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, TrainerCallback, TrainingArguments, set_seed

from dwe.kd.cache import HiddenDataset, build, tokenizer_digest, write_json
from dwe.kd.train import KDTrainer, Measurements
from dwe.loss import LossConfig, distillation_loss
from dwe.objective import kl_adjoint
from dwe.sampling import inclusion_probabilities, sample_residual


@pytest.mark.parametrize(
    "scores,budget",
    [
        ([20.0, 3.0, 2.0, 1.0, 0.0], 2),
        ([1.0, 0.0, 0.0, 0.0], 3),
        ([0.0, 0.0, 0.0, 0.0], 2),
        ([1.0, 2.0, 3.0, 4.0], 4),
    ],
)
def test_sketch_inclusion_and_exact_expectation(monkeypatch, scores, budget):
    # Integrate over the finite intervals on which Algorithm 2 is constant.
    r = torch.tensor([scores, [-s / 2 for s in scores]], dtype=torch.float64)
    p = inclusion_probabilities(r.abs().sum(0), budget)
    assert ((p >= 0) & (p <= 1)).all()
    torch.testing.assert_close(p.sum(), torch.tensor(float(budget), dtype=p.dtype))
    edges = sorted(set([0.0, 1.0] + p.cumsum(0).remainder(1).tolist()))
    mean = torch.zeros_like(r)
    inclusion = torch.zeros_like(p)
    for lo, hi in itertools.pairwise(edges):
        monkeypatch.setattr(
            torch,
            "rand",
            lambda *a, u=(lo + hi) / 2, **kw: torch.tensor(
                u, dtype=kw["dtype"], device=kw["device"]
            ),
        )
        from dwe.sampling import sketch_columns

        ids, coef = sketch_columns(r, budget)
        assert ids[0].unique().numel() == budget
        assert torch.equal(ids[0], ids[1])
        inclusion[ids[0]] += hi - lo
        mean.scatter_add_(1, ids, coef * (hi - lo))
    torch.testing.assert_close(inclusion, p, atol=1e-14, rtol=1e-13)
    torch.testing.assert_close(mean, r, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("method", ["coordinate", "topk", "sketch"])
def test_comparison_backward_matches_sampled_autograd(method):
    torch.manual_seed(12)
    h = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    w = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
    b = torch.randn(7, dtype=torch.float64, requires_grad=True)
    th, tw = torch.randn(5, 4).double(), torch.randn(7, 4).double()
    cfg = LossConfig(method=method, pairs=2, chunk_size=2)
    actual = distillation_loss(
        h, w, th, tw, bias=b, config=cfg, generator=torch.Generator().manual_seed(9)
    )
    got = torch.autograd.grad(actual * 1.7, (h, w, b))
    generator = torch.Generator().manual_seed(9)
    surrogate = h.new_zeros(())
    for start in range(0, len(h), 2):
        z = h[start : start + 2] @ w.T + b
        _, g = kl_adjoint(z.detach(), th[start : start + 2] @ tw.T, "forward", 1.0)
        ids, coef = sample_residual(g, 2, method, generator=generator)
        surrogate = surrogate + (z.gather(1, ids) * coef).sum() / len(h)
    expected = torch.autograd.grad(surrogate * 1.7, (h, w, b))
    for a, e in zip(got, expected, strict=True):
        torch.testing.assert_close(a, e, rtol=1e-12, atol=1e-12)


def test_top_magnitude_is_unrescaled_truncation():
    g = torch.tensor([[0.1, -0.3, 0.5, -0.4, 0.1]])
    ids, coef = sample_residual(g, 1, "topk")
    actual = torch.zeros_like(g).scatter_add_(1, ids, coef)
    torch.testing.assert_close(actual, torch.tensor([[0.0, 0.0, 0.5, -0.4, 0.0]]))


def test_coordinate_sampling_reweights_and_accumulates_repeated_draws(monkeypatch):
    residual = torch.tensor([[-0.1, 0.2, 0.0, -0.3, 0.2]], dtype=torch.float64)

    def draw(*shape, **kwargs):
        assert shape == (1, 4)
        return torch.tensor([[0.05, 0.2, 0.2, 0.95]], dtype=kwargs["dtype"])

    monkeypatch.setattr(torch, "rand", draw)
    ids, coefficients = sample_residual(residual, 2, "coordinate")
    assert ids.tolist() == [[0, 1, 1, 4]]
    torch.testing.assert_close(
        coefficients, torch.tensor([[-0.2, 0.2, 0.2, 0.2]], dtype=residual.dtype)
    )
    estimate = torch.zeros_like(residual).scatter_add_(1, ids, coefficients)
    torch.testing.assert_close(
        estimate, torch.tensor([[-0.2, 0.4, 0.0, 0.0, 0.2]], dtype=residual.dtype)
    )


def test_bf16_cache_preserves_hidden_bits(tmp_path):
    ids = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    hidden = torch.randn(2, 2, 4).bfloat16()
    np.save(tmp_path / "input_ids.npy", ids)
    np.save(tmp_path / "labels.npy", ids)
    hidden.view(torch.uint16).numpy().tofile(tmp_path / "hidden.bf16")
    write_json(tmp_path / "cache.json", {"shape": list(hidden.shape), "complete": True})
    dataset = HiddenDataset(tmp_path)
    assert torch.equal(dataset[1]["teacher_hidden"], hidden[1])
    assert dataset[1]["teacher_hidden"].dtype == torch.bfloat16
    assert dataset[1]["input_ids"].dtype == torch.long


def test_teacher_cache_reconstructs_online_logits(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    teacher, data, _ = tiny_problem()
    model_path, tokens, target = [tmp_path / n for n in ("teacher", "text", "cache")]
    teacher.save_pretrained(model_path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({f"t{i}": i for i in range(19)}, unk_token="t0")),
        eos_token="t18",
        unk_token="t0",
    )
    tokenizer.save_pretrained(model_path)
    tokens.mkdir()
    ids = torch.stack([r["input_ids"] for r in data[:2]])
    np.save(tokens / "input_ids.npy", ids.numpy().astype(np.int32))
    np.save(tokens / "labels.npy", ids.numpy().astype(np.int32))
    write_json(tokens / "tokens.json", {"tokenizer_sha256": tokenizer_digest(tokenizer)})
    build(
        SimpleNamespace(
            model=str(model_path),
            revision="unit-test",
            tokens=tokens,
            output=target,
            device="cpu",
            batch_size=2,
            max_rows=1,
        )
    )
    cached = HiddenDataset(target)
    assert len(cached) == cached.meta["rows"] == 1
    head = torch.load(target / "head.pt", weights_only=True)
    teacher = Qwen3ForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).eval()
    with torch.no_grad():
        online_hidden = teacher.base_model(ids[:1]).last_hidden_state[:, :-1]
        stored_hidden = cached[0]["teacher_hidden"].unsqueeze(0)
        torch.testing.assert_close(stored_hidden, online_hidden, rtol=0, atol=0)
        expected = teacher(ids[:1]).logits[:, :-1]
        actual = stored_hidden @ head["weight"].T
    torch.testing.assert_close(actual, expected)


def test_measurement_full_support_recovers_reference_gradients():
    from dwe.kd.measure import quality

    torch.manual_seed(12)
    data = {
        "hidden": torch.randn(5, 3),
        "weight": torch.randn(7, 3),
        "bias": None,
        "teacher_hidden": torch.randn(5, 4),
        "teacher_weight": torch.randn(7, 4),
        "teacher_bias": None,
        "order": torch.arange(7),
    }
    results = quality(data, repeats=2)
    for name in ("topg", "sketch"):
        assert max(results[name]["relative_mse"]) < 1e-11
        assert results[name]["nonzero_coordinates"] == 7


def tiny_problem():
    set_seed(7)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=19,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            tie_word_embeddings=True,
            attention_dropout=0.0,
        )
    )
    data = []
    for i in range(8):
        ids = torch.randint(0, 19, (6,))
        labels = ids.clone()
        labels[1 : 1 + i % 3] = -100
        data.append({"input_ids": ids, "labels": labels, "teacher_hidden": torch.randn(5, 12)})
    return model, data, {"weight": torch.randn(19, 12), "bias": None}


def make_trainer(model, data, head, output, microbatch, accumulation, method="exact", steps=1):
    args = TrainingArguments(
        output_dir=str(output),
        use_cpu=True,
        report_to="none",
        disable_tqdm=True,
        per_device_train_batch_size=microbatch,
        gradient_accumulation_steps=accumulation,
        per_device_eval_batch_size=2,
        max_steps=steps,
        max_grad_norm=0.0,
        learning_rate=0.001,
        lr_scheduler_type="constant",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        save_strategy="no",
        eval_strategy="no",
        seed=23,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    return KDTrainer(
        model=model,
        args=args,
        train_dataset=data,
        eval_dataset=data,
        head=head,
        estimator=LossConfig(
            method="quantile" if method == "exact" else method, pairs=2, chunk_size=4
        ),
        method=method,
        optimizers=(torch.optim.SGD(model.parameters(), lr=0.001), None),
    )


def test_trainer_accumulation_matches_full_batch_with_masked_targets(tmp_path):
    model, data, head = tiny_problem()
    a = make_trainer(copy.deepcopy(model), data, head, tmp_path / "a", 4, 1)
    b = make_trainer(copy.deepcopy(model), data, head, tmp_path / "b", 1, 4)
    a.train()
    b.train()
    for x, y in zip(a.model.parameters(), b.model.parameters(), strict=True):
        torch.testing.assert_close(x, y, atol=2e-7, rtol=2e-6)
    assert a.model.get_output_embeddings().weight is a.model.get_input_embeddings().weight


def test_trainer_exports_endpoint_and_exact_quality(tmp_path):
    model, data, head = tiny_problem()
    trainer = make_trainer(model, data, head, tmp_path, 2, 2, method="quantile", steps=3)
    recorder = Measurements(trainer, token_budget=24, time_budget=None)
    trainer.add_callback(recorder)
    trainer.train()
    assert trainer.state.global_step == 1
    assert recorder.endpoints["tokens"]["input_tokens"] == 24
    assert (tmp_path / "tokens/model.safetensors").exists()
    exported = Qwen3ForCausalLM.from_pretrained(tmp_path / "tokens")
    for x, y in zip(exported.parameters(), trainer.model.parameters(), strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    saved = json.loads((tmp_path / "checkpoint-1/measurement.json").read_text())
    assert saved["tokens"] == 24
    assert not (tmp_path / "quality.jsonl").exists()
    trainer.evaluate()
    quality = json.loads((tmp_path / "quality.jsonl").read_text().splitlines()[-1])
    assert quality["eval_kl"] >= 0 and quality["eval_ppl"] > 1
    assert quality["training_seconds"] > 0
    shutil.rmtree(tmp_path / "checkpoint-1")
    snapshot = tmp_path / "snapshots/step-1"
    assert json.loads((snapshot / "measurement.json").read_text())["tokens"] == 24
    reloaded = Qwen3ForCausalLM.from_pretrained(snapshot)
    for x, y in zip(reloaded.parameters(), trainer.model.parameters(), strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


@pytest.mark.parametrize("save_strategy,expected_steps", [("no", [3]), ("steps", [1, 2, 3])])
def test_snapshot_policy_keeps_model_nodes_and_one_resumable_checkpoint(
    tmp_path, save_strategy, expected_steps
):
    model, data, head = tiny_problem()
    trainer = make_trainer(model, data, head, tmp_path, 2, 2, method="coordinate", steps=3)
    trainer.args.save_strategy = save_strategy
    trainer.args.save_steps = 1
    trainer.args.save_total_limit = 1
    recorder = Measurements(trainer, token_budget=72, time_budget=None)
    trainer.add_callback(recorder)
    trainer.train()
    snapshots = tmp_path / "snapshots"
    assert sorted(int(path.name.removeprefix("step-")) for path in snapshots.iterdir()) == expected_steps
    assert [path.name for path in tmp_path.glob("checkpoint-*")] == ["checkpoint-3"]
    for step in expected_steps:
        snapshot = snapshots / f"step-{step}"
        assert (snapshot / "model.safetensors").is_file()
        assert json.loads((snapshot / "measurement.json").read_text())["tokens"] == 24 * step
    assert (tmp_path / "tokens/model.safetensors").is_file()


def test_stochastic_training_resumes_with_same_parameters(tmp_path):
    model, data, head = tiny_problem()
    reference = make_trainer(
        copy.deepcopy(model), data, head, tmp_path / "full", 2, 2, method="quantile", steps=3
    )
    reference.train()

    class StopAfterOne(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 1:
                control.should_save = control.should_training_stop = True
            return control

    partial = make_trainer(
        copy.deepcopy(model), data, head, tmp_path / "resume", 2, 2, method="quantile", steps=3
    )
    partial.add_callback(StopAfterOne())
    partial.train()
    resumed = make_trainer(
        copy.deepcopy(model), data, head, tmp_path / "resume", 2, 2, method="quantile", steps=3
    )
    resumed.train(resume_from_checkpoint=str(tmp_path / "resume/checkpoint-1"))
    for x, y in zip(reference.model.parameters(), resumed.model.parameters(), strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
