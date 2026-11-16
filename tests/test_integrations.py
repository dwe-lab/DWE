import json
import sys

import pytest
import torch

import dwe.loss as loss_module
from dwe.loss import LossConfig, distillation_loss
from dwe.objective import kl_adjoint
from dwe.reference import exact_loss, transport_oracle


def test_pot_oracle():
    pytest.importorskip("ot")
    a = torch.tensor([0.5, 0.5], dtype=torch.float64)
    positive = torch.tensor([[0.0], [10.0]], dtype=torch.float64)
    negative = torch.tensor([[1.0], [13.0]], dtype=torch.float64)
    plan = transport_oracle(a, a, positive, negative)
    torch.testing.assert_close(plan, torch.diag(a))


def test_capture_local_huggingface_models(tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    pytest.importorskip("datasets")
    pytest.importorskip("accelerate")
    from dwe.capture import main

    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({"[UNK]": 0, "one": 1, "two": 2, "three": 3}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")
    fast.model_input_names = ["input_ids", "attention_mask"]
    for role in ("student", "teacher"):
        cfg = transformers.Qwen3_5TextConfig(
            vocab_size=8,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            layer_types=["full_attention"],
            max_position_embeddings=32,
            use_cache=False,
        )
        model = transformers.Qwen3_5ForCausalLM(cfg)
        model.save_pretrained(tmp_path / role)
        fast.save_pretrained(tmp_path / role)
    data = tmp_path / "texts.jsonl"
    data.write_text(json.dumps({"text": "one two three one"}) + "\n")
    config = tmp_path / "capture.toml"
    config.write_text(f'''
[student]
id = "{tmp_path / "student"}"
revision = "main"
[teacher]
id = "{tmp_path / "teacher"}"
revision = "main"
[data]
path = "json"
data_files = "{data}"
cache_dir = "{tmp_path / "cache"}"
split = "train"
text_field = "text"
examples = 1
[capture]
device = "cpu"
dtype = "float32"
max_length = 8
positions = 3
''')
    output = tmp_path / "capture.pt"
    monkeypatch.setattr(
        sys, "argv", ["dwe-capture", "--config", str(config), "--output", str(output)]
    )
    main()
    captured = torch.load(output, weights_only=True)
    assert captured["hidden"].shape == captured["teacher_hidden"].shape == (3, 16)
    assert captured["weight"].shape == captured["teacher_weight"].shape == (8, 16)
    assert captured["metadata"]["positions"] == 3


@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_qwen35_backbone_and_tied_head_vjp(monkeypatch, direction):
    transformers = pytest.importorskip("transformers")
    # Random tiny architecture: tests real HF APIs, not pretrained capabilities.
    config = transformers.Qwen3_5TextConfig(
        vocab_size=19,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        layer_types=["full_attention", "full_attention"],
        tie_word_embeddings=True,
        max_position_embeddings=32,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(11)
    model = transformers.Qwen3_5ForCausalLM(config).eval()
    output = model(torch.tensor([[1, 2, 3, 4, 5]]), output_hidden_states=True, use_cache=False)
    hidden = output.hidden_states[-1][:, :-1].flatten(0, 1)
    head = model.get_output_embeddings()
    teacher_hidden = hidden.detach() + 0.1 * torch.randn_like(hidden)
    teacher_weight = head.weight.detach().clone()
    teacher_logits = teacher_hidden @ teacher_weight.T

    def exact_sample(residual, *args):
        ids = torch.arange(residual.shape[1]).expand(residual.shape[0], -1)
        return ids, residual

    monkeypatch.setattr(loss_module, "sample_residual", exact_sample)
    cfg = LossConfig(direction=direction, temperature=1.3, chunk_size=2)
    actual = distillation_loss(hidden, head.weight, teacher_hidden, teacher_weight, config=cfg)
    expected = kl_adjoint(
        output.logits[:, :-1].flatten(0, 1), teacher_logits, direction, cfg.temperature
    )[0].mean()
    params = tuple(model.parameters())
    ga = torch.autograd.grad(actual, params, retain_graph=True)
    ge = torch.autograd.grad(expected, params)
    torch.testing.assert_close(actual, expected)
    for a, e in zip(ga, ge, strict=True):
        torch.testing.assert_close(a, e, atol=2e-7, rtol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/Liger runtime required")
@pytest.mark.parametrize("direction", ["forward", "reverse"])
@pytest.mark.parametrize("temperature", [0.7, 1.0, 2.0])
def test_liger_kl_matches_autograd(direction, temperature):
    pytest.importorskip("liger_kernel")
    from dwe.liger import liger_loss

    torch.manual_seed(7)
    h = torch.randn(8, 16, device="cuda", requires_grad=True)
    w = torch.randn(64, 16, device="cuda", requires_grad=True)
    b = torch.randn(64, device="cuda", requires_grad=True)
    th, tw = torch.randn(8, 12, device="cuda"), torch.randn(64, 12, device="cuda")
    cfg = LossConfig(direction=direction, temperature=temperature, chunk_size=4)
    actual = liger_loss(h, w, th, tw, config=cfg, bias=b)
    expected = exact_loss(h, w, th, tw, config=cfg, bias=b)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    ga = torch.autograd.grad(actual, (h, w, b))
    ge = torch.autograd.grad(expected, (h, w, b))
    for a, e in zip(ga, ge, strict=True):
        torch.testing.assert_close(a, e, atol=1e-5, rtol=1e-4)
