import pytest
import torch

from dwe.taopd import sctaopd_values, taopd_values


@pytest.mark.parametrize("temperature", [0.7, 1.0, 2.0])
@pytest.mark.parametrize("top_k", [1, 4])
def test_sample_corrected_exact_expected_value_and_gradient(temperature, top_k):
    torch.manual_seed(81)
    z = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    teacher = torch.randn_like(z, requires_grad=True)
    lp = (z / temperature).log_softmax(-1)
    lq = (teacher.detach() / temperature).log_softmax(-1)
    probabilities = lp.detach().exp()
    expected = (lp.exp() * (lp - lq)).sum(-1)
    mean = 0
    for index in range(z.shape[-1]):
        ids = torch.full((len(z), 1), index, dtype=torch.long)
        value, _ = sctaopd_values(z, teacher, top_k=top_k,
                                 temperature=temperature, sample_ids=ids)
        mean = mean + probabilities[:, index] * value
    torch.testing.assert_close(mean, expected, atol=1e-13, rtol=1e-12)
    actual_grad = torch.autograd.grad(mean.sum(), z, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.sum(), z)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-13, rtol=1e-12)
    assert torch.autograd.grad(mean.sum(), teacher, allow_unused=True)[0] is None


def test_sample_corrected_checkpoint_preserves_sampled_backward():
    from dwe.loss import LossConfig, project
    from dwe.taopd import taopd_loss

    torch.manual_seed(21)
    h = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    w = torch.randn(7, 4, dtype=torch.float64, requires_grad=True)
    ht, wt = torch.randn(3, 5).double(), torch.randn(7, 5).double()
    cfg = LossConfig(direction="reverse", temperature=0.7, chunk_size=3)
    torch.manual_seed(55)
    actual, _ = taopd_loss(h, w, ht, wt, config=cfg, top_k=2,
                           tail_epsilon=1e-6, sample_corrected=True)
    gradients = torch.autograd.grad(actual, (h, w))
    torch.manual_seed(55)
    values, _ = sctaopd_values(project(h, w, None), project(ht, wt, None),
                              top_k=2, temperature=0.7)
    reference = values.mean() * 0.7**2
    torch.testing.assert_close(actual, reference)
    for a, b in zip(gradients, torch.autograd.grad(reference, (h, w)), strict=True):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-11)


@pytest.mark.parametrize("temperature", [0.7, 1.0, 2.0])
def test_matches_grouped_distribution_and_gradient(temperature):
    torch.manual_seed(17)
    z = torch.randn(7, 19, dtype=torch.float64, requires_grad=True)
    t = torch.randn_like(z, requires_grad=True)
    value, entropy = taopd_values(z, t, top_k=4, temperature=temperature, tail_epsilon=1e-6)
    p, q = (z / temperature).softmax(-1), (t.detach() / temperature).softmax(-1)
    ids = q.topk(4, dim=-1).indices
    ph, qh = p.gather(-1, ids), q.gather(-1, ids)
    pa = torch.cat([ph, 1 - ph.sum(-1, keepdim=True)], -1)
    qa = torch.cat([qh, 1 - qh.sum(-1, keepdim=True)], -1)
    expected = (pa * (pa.log() - qa.log())).sum(-1)
    torch.testing.assert_close(value, expected)
    actual_grad = torch.autograd.grad(value.sum(), z, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.sum(), z)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    tail = torch.ones_like(z, dtype=torch.bool).scatter(-1, ids, False)
    assert actual_grad[tail].abs().sum() > 0
    assert not entropy.requires_grad
    assert torch.autograd.grad(value.sum(), t, allow_unused=True)[0] is None


def test_concentrated_teacher_is_finite():
    z = torch.tensor([[100.0, 0.0, -100.0]], requires_grad=True)
    t = torch.tensor([[150.0, -20.0, -80.0]])
    value, _ = taopd_values(z, t, top_k=1, temperature=0.7, tail_epsilon=1e-6)
    value.sum().backward()
    assert torch.isfinite(value).all() and torch.isfinite(z.grad).all()


def test_trainer_mask_denominator_and_gradients():
    pytest.importorskip("trl")
    from dwe.loss import LossConfig, project
    from dwe.trainer import CouplingTrainer

    torch.manual_seed(31)
    h = torch.randn(2, 3, 5, requires_grad=True)
    w = torch.randn(17, 5, requires_grad=True)
    ht, wt = torch.randn(2, 3, 4), torch.randn(17, 4)
    mask = torch.tensor([[1, 0, 1], [0, 1, 1]])
    trainer = object.__new__(CouplingTrainer)
    trainer.estimator = LossConfig(chunk_size=2)
    trainer.loss_backend = "taopd"
    trainer.taopd_config = dict(top_k=3, tail_epsilon=1e-6)
    actual, entropy, count = trainer.distribution_loss(
        h, ht, w, wt, mask, 1.0, 2, num_items_in_batch=13, temperature=0.7
    )
    expected, expected_entropy = taopd_values(
        project(h[mask.bool()], w, None),
        project(ht[mask.bool()], wt, None),
        temperature=0.7,
        **trainer.taopd_config,
    )
    torch.testing.assert_close(actual, expected.sum() / 13)
    torch.testing.assert_close(entropy, expected_entropy)
    grads = torch.autograd.grad(actual, (h, w), retain_graph=True)
    reference = torch.autograd.grad(expected.sum() / 13, (h, w))
    for a, b in zip(grads, reference, strict=True):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    assert count == 4
    zero, _, _ = trainer.distribution_loss(h, ht, w, wt, mask * 0, 1.0, 2)
    zero.backward()
    assert h.grad.abs().sum() == w.grad.abs().sum() == 0
