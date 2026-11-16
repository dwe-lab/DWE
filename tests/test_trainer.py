"""Check the upstream loss boundary and a real Trainer save/resume cycle."""

import pytest
import torch

pytest.importorskip("trl")
from trl import DistillationConfig
from trl.trainer.distillation_trainer import _chunked_divergence_loss

from dwe.loss import LossConfig
from dwe.trainer import CouplingTrainer


@pytest.mark.parametrize("beta", [0.0, 1.0])
@pytest.mark.parametrize("temperature", [0.7, 1.0, 2.0])
def test_loss_boundary(beta, temperature):
    torch.manual_seed(23)
    h = torch.randn(2, 3, 8, requires_grad=True)
    ht = torch.randn(2, 3, 6)
    w = torch.randn(16, 8, requires_grad=True)
    wt = torch.randn(16, 6)
    mask = torch.tensor([[1, 1, 0], [0, 1, 0]])
    trainer = object.__new__(CouplingTrainer)
    trainer.estimator = LossConfig(pairs=16, chunk_size=2)
    trainer.loss_backend = "sparse"
    trainer.order = torch.randperm(16)
    args = (h, ht, w, wt, mask, beta, 2)
    options = dict(num_items_in_batch=13, temperature=temperature)
    actual = trainer.distribution_loss(*args, **options)
    expected = _chunked_divergence_loss(*args, **options)
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e, atol=2e-6, rtol=2e-6)
    actual[0].backward()
    assert torch.equal(h.grad[~mask.bool()], torch.zeros_like(h.grad[~mask.bool()]))
    assert torch.isfinite(w.grad).all()
    h.grad = w.grad = None
    zero, entropy, count = trainer.distribution_loss(h, ht, w, wt, mask * 0, beta, 2)
    zero.backward()
    assert zero == entropy == count == 0
    assert h.grad.abs().sum() == w.grad.abs().sum() == 0


@pytest.mark.parametrize("vocabulary_order", ["principal", "natural"])
def test_training_checkpoint_resume(tmp_path, monkeypatch, vocabulary_order):
    from datasets import Dataset
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(42)
    vocab = {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "one": 3, "two": 4, "user": 5}
    raw = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=raw, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]"
    )
    tok.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
    tok.model_input_names = ["input_ids", "attention_mask"]
    config = Qwen3Config(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
        pad_token_id=1,
        eos_token_id=2,
        bos_token_id=None,
    )
    data = Dataset.from_dict({"prompt": [[{"role": "user", "content": "one two"}]] * 4})

    def build(output):
        torch.manual_seed(42)
        return CouplingTrainer(
            model=Qwen3ForCausalLM(config),
            teacher_model=Qwen3ForCausalLM(config),
            processing_class=tok,
            train_dataset=data,
            args=DistillationConfig(
                output_dir=str(output),
                use_cpu=True,
                bf16=False,
                max_steps=2,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=2,
                gradient_checkpointing=False,
                max_completion_length=2,
                beta=0.0,
                temperature=0.7,
                logging_steps=1,
                save_steps=1,
                report_to="none",
                dataloader_pin_memory=False,
                disable_tqdm=True,
            ),
            estimator=LossConfig(method="quantile", pairs=4, chunk_size=2),
            loss_backend="sparse",
            vocabulary_order=vocabulary_order,
        )

    from dwe.train import StopAfterStep

    full = build(tmp_path / "full")
    full.train()
    trainer = build(tmp_path / "split")
    trainer.add_callback(StopAfterStep(1))
    before = trainer.model.get_input_embeddings().weight.detach().clone()
    trainer.train()
    assert not torch.equal(before, trainer.model.get_input_embeddings().weight)
    checkpoint = tmp_path / "split" / "checkpoint-1"
    assert (checkpoint / "optimizer.pt").is_file()
    assert (checkpoint / "rng_state.pth").is_file()
    assert (checkpoint / "estimator.json").is_file()
    assert (checkpoint / "geometry.pt").exists() == (vocabulary_order == "principal")
    if vocabulary_order == "principal":
        original_order = torch.load(checkpoint / "geometry.pt", weights_only=True)
        assert torch.equal(original_order, trainer.order.cpu())
    assert "entropy" in trainer.state.log_history[0]

    def unexpected_pca(*args):
        raise AssertionError("Resume must restore the original order without PCA")

    monkeypatch.setattr("dwe.trainer.principal_order", unexpected_pca)
    resumed = build(tmp_path / "split")
    resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed.state.global_step == 2
    if vocabulary_order == "principal":
        assert torch.equal(resumed.order.cpu(), original_order)
    for actual, expected in zip(resumed.model.parameters(), full.model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    resumed.estimator = LossConfig(method="coupled", pairs=4, chunk_size=2)
    with pytest.raises(ValueError, match="settings differ"):
        resumed.prepare_sampling(checkpoint)
    resumed.estimator = trainer.estimator
    if vocabulary_order == "principal":
        torch.save(torch.zeros(16, dtype=torch.long), checkpoint / "geometry.pt")
        with pytest.raises(ValueError, match="every vocabulary index"):
            resumed.prepare_sampling(checkpoint)
        (checkpoint / "geometry.pt").unlink()
        with pytest.raises(ValueError, match="missing its fixed"):
            resumed.prepare_sampling(checkpoint)
    (checkpoint / "estimator.json").unlink()
    with pytest.raises(ValueError, match="original pinned release"):
        resumed.prepare_sampling(checkpoint)
