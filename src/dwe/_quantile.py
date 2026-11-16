"""CUDA inverse CDF using block masses and scans only inside selected blocks."""

import torch
import triton
import triton.language as tl


@triton.jit
def _masses(
    G, ORDER, POS, NEG, V: tl.constexpr, NB: tl.constexpr, B: tl.constexpr, ORDERED: tl.constexpr
):
    row = tl.program_id(1)
    block = tl.program_id(0)
    j = block * B + tl.arange(0, B)
    ids = tl.load(ORDER + j, j < V, other=0) if ORDERED else j
    g = tl.load(G + row * V + ids, j < V, other=0).to(tl.float64)
    tl.store(POS + row * NB + block, tl.sum(tl.maximum(g, 0), 0))
    tl.store(NEG + row * NB + block, tl.sum(tl.maximum(-g, 0), 0))


@triton.jit
def _within(
    G,
    ORDER,
    BLOCKS,
    QUERIES,
    IDS,
    V: tl.constexpr,
    K: tl.constexpr,
    B: tl.constexpr,
    ORDERED: tl.constexpr,
    NEGATIVE: tl.constexpr,
):
    row = tl.program_id(1)
    pair = tl.program_id(0)
    block = tl.load(BLOCKS + row * K + pair)
    j = block * B + tl.arange(0, B)
    ids = tl.load(ORDER + j, j < V, other=0) if ORDERED else j
    g = tl.load(G + row * V + ids, j < V, other=0).to(tl.float64)
    mass = tl.maximum(-g if NEGATIVE else g, 0)
    cdf = tl.cumsum(mass, 0)
    query = tl.load(QUERIES + row * K + pair)
    selected = tl.min(tl.where((cdf > query) & (mass > 0) & (j < V), j, V), 0)
    # Numerical endpoint guard: keep a valid support coordinate within the selected block.
    last = tl.max(tl.where((mass > 0) & (j < V), j, -1), 0)
    selected = tl.where(selected < V, selected, tl.maximum(last, 0))
    token = tl.load(ORDER + selected) if ORDERED else selected
    tl.store(IDS + row * K + pair, token)


@torch.no_grad()
def sample_pairs(residual, pairs, method, order=None, generator=None):
    block_size = 512
    # FP64 block sums keep inverse-CDF boundaries stable without large CDF buffers.
    n, v = residual.shape
    assert residual.is_cuda and residual.dtype == torch.float32 and residual.is_contiguous()
    nb = triton.cdiv(v, block_size)
    pos = torch.empty((n, nb), device=residual.device, dtype=torch.float64)
    neg = torch.empty_like(pos)
    _masses[(nb, n)](residual, order, pos, neg, v, nb, block_size, order is not None, num_warps=4)
    cp = pos.cumsum(-1)
    cn = neg.cumsum(-1)
    a = cp[:, -1:]
    b = cn[:, -1:]
    u = torch.rand(n, pairs, device=residual.device, dtype=residual.dtype, generator=generator)
    if method == "quantile":
        u = (u + torch.arange(pairs, device=residual.device, dtype=residual.dtype)) / pairs
    v_uniform = (
        u
        if method != "independent"
        else torch.rand(n, pairs, device=residual.device, dtype=residual.dtype, generator=generator)
    )
    u = u.to(torch.float64)
    v_uniform = v_uniform.to(torch.float64)
    selected = []
    for masses, cdf, total, uniform, negative in [
        (pos, cp, a, u, False),
        (neg, cn, b, v_uniform, True),
    ]:
        query = torch.minimum(uniform * total, torch.nextafter(total, torch.zeros_like(total)))
        blocks = torch.searchsorted(cdf, query.contiguous(), right=True).clamp_max(nb - 1)
        before = torch.where(blocks > 0, cdf.gather(1, (blocks - 1).clamp_min(0)), 0.0)
        local_mass = masses.gather(1, blocks)
        local_query = torch.minimum(
            (query - before).clamp_min(0), torch.nextafter(local_mass, torch.zeros_like(local_mass))
        )
        ids = torch.empty((n, pairs), device=residual.device, dtype=torch.int64)
        _within[(pairs, n)](
            residual,
            order,
            blocks,
            local_query,
            ids,
            v,
            pairs,
            block_size,
            order is not None,
            negative,
            num_warps=4,
        )
        selected.append(ids)
    ids = torch.cat(selected, dim=1)
    coef = (
        torch.cat((a.expand(-1, pairs), -b.expand(-1, pairs)), dim=1).div(pairs).to(residual.dtype)
    )
    return ids, coef
