"""Unbiased residual samplers and a deterministic top-magnitude control."""

import torch
from torch import Tensor

from ._pointwise import ordered_parts


def inclusion_probabilities(scores: Tensor, budget: int) -> Tensor:
    """Water filling: p_i=min(1, score_i/t), with sum(p)=budget.

    Scores are sqrt(w_i) in Bakong et al. (2026), Algorithm 1.
    Algorithm 6 sets w_i=||G[:,i]||_1², hence scores=||G[:,i]||_1.
    Zero-score coordinates fill unused slots when support is smaller than budget.
    """
    if scores.ndim != 1 or not 1 <= budget <= scores.numel():
        raise ValueError("scores must be a vector and budget between 1 and its length")
    scores = scores.double()
    ranked = scores.sort(descending=True).values
    suffix = ranked.flip(0).cumsum(0).flip(0)
    k = torch.arange(len(scores), device=scores.device)
    thresholds = suffix / (budget - k).clamp_min(1)
    eligible = (k < budget) & (ranked <= thresholds)
    first = torch.where(eligible, k, len(scores)).amin().clamp_max(len(scores) - 1)
    threshold = thresholds[first]
    p = (scores / threshold.clamp_min(torch.finfo(scores.dtype).tiny)).clamp_max(1)
    support = (scores > 0).sum()
    zero_probability = (budget - support).clamp_min(0).to(scores.dtype)
    zero_probability = zero_probability / (len(scores) - support).clamp_min(1)
    return torch.where(scores > 0, p, zero_probability)


def sketch_columns(residual: Tensor, budget: int, generator=None) -> tuple[Tensor, Tensor]:
    """Algorithms 1, 2 and 6: one correlated column mask shared by this token chunk."""
    budget = min(budget, residual.shape[1])
    p = inclusion_probabilities(residual.abs().sum(0), budget)
    cdf = p.cumsum(0).clamp_max(budget)
    cdf[-1] = budget
    u = torch.rand((), device=p.device, dtype=p.dtype, generator=generator)
    offsets = torch.arange(budget, device=p.device, dtype=p.dtype)
    # Adding an offset must not round a draw below one to the next integer.
    queries = torch.minimum(u + offsets, torch.nextafter(offsets + 1, offsets))
    ids = torch.searchsorted(cdf, queries, right=True)
    coef = residual[:, ids] / p[ids].to(residual.dtype)
    return ids.expand(residual.shape[0], -1), coef


def _cdf(weights: Tensor) -> tuple[Tensor, Tensor]:
    cdf = weights.cumsum(-1)
    return cdf, cdf[:, -1:]


def _invert(cdf: Tensor, mass: Tensor, uniform: Tensor) -> Tensor:
    # Scale K queries instead of normalizing all V cumulative weights.
    # Rounding must not turn a draw below one into the inclusive upper endpoint.
    upper = torch.nextafter(mass, torch.zeros_like(mass))
    query = torch.minimum(uniform * mass, upper)
    indices = torch.searchsorted(cdf, query.contiguous(), right=True)
    return torch.where(mass > 0, indices, torch.zeros_like(indices))


@torch.no_grad()
def sample_residual(
    residual: Tensor,
    pairs: int,
    method: str,
    order: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Sample a detached adjoint; marginals fix its mean, coupling fixes covariance.

    Quantile sampling draws once per equal-width quantile stratum and shares
    that quantile within each pair. The average, not each stratum individually,
    has the required residual marginals. Strata and positions use independent jitter.
    ``coupled`` uses iid shared quantiles with the same order: it isolates the
    effect of stratification. ``independent`` samples the two sides independently.
    Pass a cached geometry permutation for coupled/quantile sampling; None uses
    natural vocabulary order for the ordering ablation.
    Separate side masses preserve the input residual's expectation even when
    floating-point roundoff makes its sum slightly nonzero. No residual threshold
    or additional noise is applied. ``coordinate`` is the equal-row-budget control.
    """
    if pairs < 1:
        raise ValueError("pairs must be positive")
    if residual.ndim != 2:
        raise ValueError("residual must have shape [positions, vocabulary]")
    n = residual.shape[0]
    kwargs = dict(device=residual.device, dtype=residual.dtype, generator=generator)
    if method == "topk":
        ids = residual.abs().topk(min(2 * pairs, residual.shape[1]), dim=-1).indices
        return ids, residual.gather(1, ids)
    if method == "sketch":
        return sketch_columns(residual, 2 * pairs, generator)
    if method == "coordinate":
        cdf, mass = _cdf(residual.abs())
        ids = _invert(cdf, mass, torch.rand(n, 2 * pairs, **kwargs))
        return ids, mass * residual.gather(1, ids).sign() / (2 * pairs)
    if method not in ("independent", "coupled", "quantile"):
        raise ValueError(f"Unknown sampler: {method}")
    coupled = method in ("coupled", "quantile")
    if coupled and order is not None:
        if order.shape != (residual.shape[1],):
            raise ValueError("order must be a vocabulary permutation")
    if residual.is_cuda and residual.dtype == torch.float32 and residual.is_contiguous():
        from ._quantile import sample_pairs

        return sample_pairs(residual, pairs, method, order if coupled else None, generator)
    pos, neg = ordered_parts(residual, order if coupled else None)
    positive, alpha = _cdf(pos)
    del pos
    negative, beta = _cdf(neg)
    del neg
    u = torch.rand(n, pairs, **kwargs)
    if method == "quantile":
        u = (u + torch.arange(pairs, device=residual.device, dtype=residual.dtype)) / pairs
    v = u if coupled else torch.rand(n, pairs, **kwargs)
    i, j = _invert(positive, alpha, u), _invert(negative, beta, v)
    ids = torch.cat((i, j), dim=1)
    if coupled and order is not None:
        ids = order[ids]
    coefficients = torch.cat((alpha.expand(-1, pairs), -beta.expand(-1, pairs)), dim=1)
    return ids, coefficients / pairs
