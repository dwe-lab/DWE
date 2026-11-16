import itertools

import pytest
import torch

import dwe.loss as loss_module
from dwe.geometry import principal_order, validate_order
from dwe.loss import LossConfig, distillation_loss
from dwe.objective import kl_adjoint
from dwe.reference import exact_loss
from dwe.sampling import sample_residual


@pytest.mark.parametrize(
    "direction,temperature", list(itertools.product(["forward", "reverse"], [0.7, 1.0, 2.0]))
)
def test_analytic_adjoint(direction, temperature):
    torch.manual_seed(4)
    z = torch.randn(3, 7, dtype=torch.float64, requires_grad=True)
    teacher = torch.randn_like(z)
    value, residual = kl_adjoint(z, teacher, direction, temperature)
    (expected,) = torch.autograd.grad(value.sum(), z)
    torch.testing.assert_close(residual, expected, atol=1e-13, rtol=1e-12)


@pytest.mark.parametrize("direction", ["forward", "reverse"])
@pytest.mark.parametrize("train_head", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 3, 10])
def test_custom_vjp_matches_exact_with_enumerated_adjoint(
    monkeypatch, direction, train_head, chunk_size
):
    # Replace only randomness with its exact expectation; exercise chunking, bias,
    # loss scaling and the entire custom VJP against independent autograd.
    def exact_sample(r, *args):
        return torch.arange(r.shape[1]).expand(r.shape[0], -1), r

    monkeypatch.setattr(loss_module, "sample_residual", exact_sample)
    torch.manual_seed(9)
    h = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    w = torch.randn(7, 3, dtype=torch.float64, requires_grad=train_head)
    b = torch.randn(7, dtype=torch.float64, requires_grad=train_head)
    th, tw, tb = torch.randn(5, 4).double(), torch.randn(7, 4).double(), torch.randn(7).double()
    parameters = (h, w, b) if train_head else (h,)
    cfg = LossConfig(direction=direction, temperature=1.7, chunk_size=chunk_size)
    actual = distillation_loss(h, w, th, tw, config=cfg, bias=b, teacher_bias=tb)
    expected = exact_loss(h, w, th, tw, config=cfg, bias=b, teacher_bias=tb)
    torch.testing.assert_close(actual, expected)
    ga = torch.autograd.grad(actual * 2.3, parameters)
    ge = torch.autograd.grad(expected * 2.3, parameters)
    for a, e in zip(ga, ge, strict=True):
        torch.testing.assert_close(a, e, atol=1e-12, rtol=1e-11)


@pytest.mark.parametrize("method", ["independent", "coupled", "quantile", "coordinate"])
def test_sampler_exact_expectation(monkeypatch, method):
    r = torch.tensor([[0.2, 0.1, -0.15, -0.15, 0.0]], dtype=torch.float64)
    order = torch.tensor([3, 1, 4, 0, 2])
    ordered = r[0, order] if method in ("coupled", "quantile") else r[0]
    if method == "coordinate":
        cdf = ordered.abs().cumsum(0) / ordered.abs().sum()
        intervals = sorted(set([0.0] + cdf.tolist()))
        outcomes = [((a + b) / 2, None, b - a) for a, b in itertools.pairwise(intervals)]
    else:
        ap, bp = ordered.clamp_min(0), (-ordered).clamp_min(0)
        ac, bc = ap.cumsum(0) / ap.sum(), bp.cumsum(0) / bp.sum()
        aa, bb = sorted(set([0.0] + ac.tolist())), sorted(set([0.0] + bc.tolist()))
        if method in ("coupled", "quantile"):
            breaks = sorted(set(aa + bb))
            outcomes = [((a + b) / 2, None, b - a) for a, b in itertools.pairwise(breaks)]
        else:
            outcomes = [
                ((a + b) / 2, (c + d) / 2, (b - a) * (d - c))
                for a, b in itertools.pairwise(aa)
                for c, d in itertools.pairwise(bb)
            ]
    mean = torch.zeros_like(r)
    second = torch.zeros(r.numel(), r.numel(), dtype=r.dtype)
    for u, v, probability in outcomes:
        draws = iter([u, v])

        def draw(*shape, draws=draws, **kwargs):
            kwargs.pop("generator", None)
            return torch.full(shape, next(draws), **kwargs)

        monkeypatch.setattr(torch, "rand", draw)
        ids, coefficients = sample_residual(r, 1, method, order)
        sample = torch.zeros_like(r).scatter_add_(1, ids, coefficients)
        mean += probability * sample
        second += probability * torch.outer(sample[0], sample[0])
    torch.testing.assert_close(mean, r, atol=1e-15, rtol=1e-14)
    if method != "coordinate":
        mass = r.clamp_min(0).sum()
        covariance = second - torch.outer(r[0], r[0])
        # Coupling cannot improve logit trace, even when it changes covariance.
        torch.testing.assert_close(covariance.trace(), 2 * mass.square() - r.square().sum())
        if method == "independent":
            positive, negative = r[0].clamp_min(0), (-r[0]).clamp_min(0)
            expected = mass * torch.diag(r[0].abs())
            expected -= torch.outer(positive, positive) + torch.outer(negative, negative)
            torch.testing.assert_close(covariance, expected, atol=1e-15, rtol=1e-14)


@pytest.mark.parametrize("method", ["independent", "coupled", "quantile", "coordinate"])
def test_zero_and_tiny_residuals(method):
    r = torch.tensor([[0.0, 0.0], [1e-40, -1e-40]], dtype=torch.float64)
    ids, weights = sample_residual(r, 4, method, torch.arange(2))
    assert (weights[0] == 0).all()
    assert weights[1].abs().sum() > 0
    assert ((ids >= 0) & (ids < 2)).all()


def test_geometry_and_order_validation():
    weight = torch.tensor([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    torch.manual_seed(1)
    order = principal_order(weight)
    validate_order(order, 4)
    assert order.tolist() in ([0, 1, 2, 3], [3, 2, 1, 0])
    with pytest.raises(ValueError):
        validate_order(torch.tensor([0, 0, 2, 3]), 4)


def test_geometry_is_cached_without_advancing_training_rng():
    weight = torch.randn(31, 7)
    rng = torch.get_rng_state().clone()
    order = principal_order(weight)
    assert torch.equal(torch.get_rng_state(), rng)
    torch.manual_seed(123)
    assert torch.equal(order, principal_order(weight))


def test_geometry_reduces_hidden_error_for_aligned_rows():
    # All rows lie on one line, so projected transport is also optimal in hidden space.
    weight = torch.tensor([[0.0, 0.0], [10.0, 20.0], [1.0, 2.0], [11.0, 22.0]])
    order = principal_order(weight)
    positive = order[order < 2]
    negative = order[order >= 2]
    coupled = (weight[positive] - weight[negative]).square().sum(-1).mean()
    independent = torch.cdist(weight[:2], weight[2:]).square().mean()
    assert coupled < independent


@pytest.mark.parametrize("method", ["independent", "coupled", "quantile"])
def test_paired_samples_preserve_mass_and_row_budget(method):
    r = torch.tensor([[0.25, -0.125, 0.125, -0.25, 0.0]]).expand(100, -1)
    ids, coefficients = sample_residual(r, 2, method, torch.tensor([3, 1, 0, 4, 2]))
    sample = torch.zeros_like(r).scatter_add_(1, ids, coefficients)
    assert (sample.sum(-1) == 0).all()
    assert ((sample != 0).sum(-1) <= 4).all()


def test_coupled_draws_are_iid_quantiles_not_stratified(monkeypatch):
    r = torch.tensor([[0.2, -0.2, 0.3, -0.3]], dtype=torch.float64)
    order = torch.tensor([2, 3, 0, 1])

    def draw(*shape, **kwargs):
        return torch.tensor([[0.2, 0.3]], dtype=kwargs["dtype"])

    monkeypatch.setattr(torch, "rand", draw)
    ids, _ = sample_residual(r, 2, "coupled", order)
    assert ids.tolist() == [[2, 2, 3, 3]]
    ids, _ = sample_residual(r, 2, "quantile", order)
    assert ids.tolist() == [[2, 0, 3, 1]]


def test_projection_can_worsen_full_hidden_error():
    weight = torch.tensor([[0.0, 0.0], [1.0, 100.0], [0.0, 100.0], [1.0, 0.0]])
    independent = torch.cdist(weight[:2], weight[2:]).square().mean()
    projected = (weight[:2] - weight[2:]).square().sum(-1).mean()
    assert projected > independent


def test_real_sampling_backward_reproducible():
    h = torch.randn(4, 3, requires_grad=True)
    w = torch.randn(7, 3, requires_grad=True)
    th, tw = torch.randn(4, 3), torch.randn(7, 3)
    cfg = LossConfig(method="quantile", pairs=3, chunk_size=2)
    order = principal_order(w)
    gradients = []
    for _ in range(2):
        generator = torch.Generator().manual_seed(42)
        value = distillation_loss(h, w, th, tw, config=cfg, order=order, generator=generator)
        gradients.append(torch.autograd.grad(value, (h, w)))
    for a, b in zip(*gradients, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_unnormalized_inverse_cdf_boundaries(dtype):
    from dwe.sampling import _cdf, _invert

    weights = torch.tensor([[0.0, 2.0, 0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]], dtype=dtype)
    cdf, mass = _cdf(weights)
    u = torch.tensor([[0.0, 0.2, 0.4, 0.8, 1.0], [0.0, 0.2, 0.4, 0.8, 1.0]], dtype=dtype)
    ids = _invert(cdf, mass, u)
    assert ids[0].tolist() == [1, 1, 3, 3, 3]
    assert ids[1].tolist() == [0, 0, 0, 0, 0]


@pytest.mark.parametrize(
    "method,order_ids",
    [
        ("independent", None),
        ("coupled", [2, 0, 5, 3, 1, 4]),
        ("quantile", None),
        ("quantile", [2, 0, 5, 3, 1, 4]),
    ],
)
@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_actual_pair_backward_matches_selected_logit_surrogate(method, order_ids, direction):
    # Same random choices, independent autograd path through a nonlinear backbone.
    # Covers full parameter gradients rather than just the hidden-state estimate.
    torch.manual_seed(27)
    x = torch.randn(3, 2, dtype=torch.float64)
    backbone = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    w = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
    b = torch.randn(6, dtype=torch.float64, requires_grad=True)
    teacher_h, teacher_w = torch.randn(3, 5).double(), torch.randn(6, 5).double()
    h = (x @ backbone).tanh()
    config = LossConfig(direction=direction, method=method, pairs=2, temperature=0.7, chunk_size=3)
    order = None if order_ids is None else torch.tensor(order_ids)
    loss = distillation_loss(
        h,
        w,
        teacher_h,
        teacher_w,
        bias=b,
        config=config,
        order=order,
        generator=torch.Generator().manual_seed(8),
    )
    actual = torch.autograd.grad(loss, (backbone, w, b))
    h = (x @ backbone).tanh()
    z = h @ w.T + b
    _, residual = kl_adjoint(z.detach(), teacher_h @ teacher_w.T, direction, 0.7)
    ids, coefficients = sample_residual(
        residual, 2, method, order, torch.Generator().manual_seed(8)
    )
    surrogate = (z.gather(1, ids) * coefficients).sum() / len(x)
    expected = torch.autograd.grad(surrogate, (backbone, w, b))
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e, atol=1e-12, rtol=1e-11)


@pytest.mark.parametrize("pairs", [1, 2, 4])
@pytest.mark.parametrize("natural_order", [False, True])
def test_stratified_coupling_exact_covariance(monkeypatch, pairs, natural_order):
    # Enumerate every constant-CDF cell, not a noisy Monte Carlo assertion.
    r = torch.tensor([[0.3, -0.2, 0.1, -0.2, 0.0]], dtype=torch.float64)
    order = torch.arange(5) if natural_order else torch.tensor([2, 1, 4, 0, 3])
    ordered = r[0, order]
    ac = ordered.clamp_min(0).cumsum(0) / ordered.clamp_min(0).sum()
    bc = (-ordered).clamp_min(0).cumsum(0) / (-ordered).clamp_min(0).sum()
    breaks = sorted(set([0.0, 1.0] + ac.tolist() + bc.tolist()))
    eye = torch.eye(r.numel(), dtype=r.dtype)
    mass = r.clamp_min(0).sum()

    def pair_at(u):
        point = torch.tensor(u, dtype=r.dtype)
        i = order[torch.searchsorted(ac, point, right=True)]
        j = order[torch.searchsorted(bc, point, right=True)]
        return mass * (eye[i] - eye[j])

    strata, means = [], []
    iid_second = torch.zeros(5, 5, dtype=r.dtype)
    for lo, hi in itertools.pairwise(breaks):
        z = pair_at((lo + hi) / 2)
        iid_second += (hi - lo) * torch.outer(z, z)
    for k in range(pairs):
        edges = sorted(
            set(
                [k / pairs, (k + 1) / pairs]
                + [u for u in breaks if k / pairs < u < (k + 1) / pairs]
            )
        )
        cells = [((lo + hi) / 2, pairs * (hi - lo)) for lo, hi in itertools.pairwise(edges)]
        strata.append(cells)
        means.append(sum(prob * pair_at(u) for u, prob in cells))
    mean = torch.zeros(5, dtype=r.dtype)
    second = torch.zeros(5, 5, dtype=r.dtype)
    for outcome in itertools.product(*strata):
        jitter = torch.tensor([[u * pairs - k for k, (u, _) in enumerate(outcome)]], dtype=r.dtype)
        probability = 1.0
        for _, prob in outcome:
            probability *= prob

        def draw(*shape, jitter=jitter, **kwargs):
            assert shape == (1, pairs)
            return jitter.clone()

        monkeypatch.setattr(torch, "rand", draw)
        ids, coefficients = sample_residual(r, pairs, "quantile", None if natural_order else order)
        z = torch.zeros_like(r).scatter_add_(1, ids, coefficients)[0]
        mean += probability * z
        second += probability * torch.outer(z, z)
    torch.testing.assert_close(mean, r[0], atol=1e-14, rtol=1e-13)
    covariance = second - torch.outer(mean, mean)
    iid_covariance = (iid_second - torch.outer(r[0], r[0])) / pairs
    reduction = sum(torch.outer(mu - r[0], mu - r[0]) for mu in means) / pairs**2
    torch.testing.assert_close(iid_covariance - covariance, reduction, atol=1e-14, rtol=1e-13)
    # Any fixed Jacobian mapping inherits the covariance improvement.
    mapping = torch.tensor([[1.0, -2.0, 0.0, 3.0, 1.0], [0.0, 1.0, 4.0, -1.0, 2.0]], dtype=r.dtype)
    eigenvalues = torch.linalg.eigvalsh(mapping @ (iid_covariance - covariance) @ mapping.T)
    assert eigenvalues.min() >= -1e-13
