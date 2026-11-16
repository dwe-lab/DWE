"""Validate blocked CUDA sampling against a full high-precision inverse CDF."""

import pytest
import torch

from dwe.loss import LossConfig
from dwe.reference import exact_loss
from dwe.sampling import sample_residual


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("vocab", [7, 513, 151936])
@pytest.mark.parametrize("method", ["independent", "coupled", "quantile"])
def test_blocked_matches_full_cdf(vocab, method):
    torch.manual_seed(42)
    g = torch.randn(8, vocab, device="cuda") / vocab
    order = torch.randperm(vocab, device="cuda")
    actual_rng = torch.Generator(device="cuda").manual_seed(13)
    reference_rng = torch.Generator(device="cuda").manual_seed(13)
    ids, coef = sample_residual(g, 8, method, order, actual_rng)
    u = torch.rand(8, 8, device="cuda", generator=reference_rng)
    if method == "quantile":
        u = (u + torch.arange(8, device="cuda")) / 8
    v = torch.rand(8, 8, device="cuda", generator=reference_rng) if method == "independent" else u
    ordered = g if method == "independent" else g[:, order]
    expected_ids, expected_coef = [], []
    for sign, uniform in [(1, u), (-1, v)]:
        cdf = (sign * ordered.double()).clamp_min(0).cumsum(-1)
        mass = cdf[:, -1:]
        query = torch.minimum(
            uniform.double() * mass, torch.nextafter(mass, torch.zeros_like(mass))
        )
        index = torch.searchsorted(cdf, query.contiguous(), right=True)
        expected_ids.append(index if method == "independent" else order[index])
        expected_coef.append((sign * mass / 8).expand(-1, 8).float())
    assert torch.equal(ids, torch.cat(expected_ids, 1))
    torch.testing.assert_close(coef, torch.cat(expected_coef, 1), rtol=1e-6, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_blocked_exact_expectation_and_zero_mass(monkeypatch):
    g = torch.zeros(1, 1031, device="cuda")
    g[0, [0, 511, 512, 1029]] = torch.tensor([0.25, -0.125, 0.125, -0.25], device="cuda")
    pos, neg = g.double().clamp_min(0), (-g.double()).clamp_min(0)
    breaks = torch.unique(
        torch.cat(
            (
                torch.zeros(1, device="cuda"),
                (pos.cumsum(-1) / pos.sum()).flatten(),
                (neg.cumsum(-1) / neg.sum()).flatten(),
            )
        )
    )
    midpoint = ((breaks[1:] + breaks[:-1]) / 2).float().unsqueeze(1)
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: midpoint)
    inputs = g.expand(len(midpoint), -1).contiguous()
    ids, coef = sample_residual(inputs, 1, "quantile")
    estimates = torch.zeros_like(inputs, dtype=torch.float64).scatter_add_(1, ids, coef.double())
    expected = (estimates * (breaks[1:] - breaks[:-1])[:, None]).sum(0)
    torch.testing.assert_close(expected, g[0].double(), rtol=0, atol=0)
    _, zeros = sample_residual(torch.zeros_like(inputs), 1, "quantile")
    assert (zeros == 0).all()


@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_exact_head_recomputation_preserves_loss_and_gradients(direction):
    torch.manual_seed(9)
    h = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    w = torch.randn(11, 3, dtype=torch.float64, requires_grad=True)
    th, tw = torch.randn(5, 4).double(), torch.randn(11, 4).double()
    cfg = LossConfig(direction=direction, chunk_size=3)
    retained = exact_loss(h, w, th, tw, config=cfg, checkpoint_chunks=False)
    recomputed = exact_loss(h, w, th, tw, config=cfg, checkpoint_chunks=True)
    torch.testing.assert_close(retained, recomputed, rtol=0, atol=0)
    for a, b in zip(
        torch.autograd.grad(retained, (h, w)), torch.autograd.grad(recomputed, (h, w)), strict=True
    ):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
