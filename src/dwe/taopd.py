"""Tail-aware top-k RKL (arXiv:2608.14728, Eq. 4 and Appendix D.2).

Equation implementation; upstream provenance is recorded in upstream.json.
Chunk checkpointing bounds vocabulary activations, as in the exact TRL control.
"""

import math

import torch
from torch.utils.checkpoint import checkpoint

from .loss import project


def tail_log_probability(log_mass, epsilon):
    x = log_mass.clamp_max(-epsilon)
    # Evaluate each stable branch only on its domain to avoid inactive NaN gradients.
    low = x < -math.log(2)
    out = torch.empty_like(x)
    out[low] = torch.log1p(-x[low].exp())
    out[~low] = torch.log(-x[~low].expm1())
    return out


def taopd_values(student_logits, teacher_logits, *, top_k, temperature, tail_epsilon):
    """Per-position RKL on teacher top-k plus its collapsed tail; no T² factor."""
    if not 1 <= top_k < student_logits.shape[-1]:
        raise ValueError("top_k must be positive and smaller than the vocabulary")
    if temperature <= 0 or tail_epsilon <= 0:
        raise ValueError("temperature and tail_epsilon must be positive")
    lp = (student_logits / temperature).log_softmax(-1)
    with torch.no_grad():
        lq = (teacher_logits / temperature).log_softmax(-1)
        q_head, indices = lq.topk(top_k, dim=-1)
        q_tail = tail_log_probability(q_head.logsumexp(-1), tail_epsilon)
        q = torch.cat((q_head, q_tail.unsqueeze(-1)), dim=-1)
    p_head = lp.gather(-1, indices)
    p_tail = tail_log_probability(p_head.logsumexp(-1), tail_epsilon)
    p = torch.cat((p_head, p_tail.unsqueeze(-1)), dim=-1)
    values = (p.exp() * (p - q)).sum(-1)
    entropy = -(lp.detach().exp() * lp.detach()).sum()
    return values, entropy


def sctaopd_values(student_logits, teacher_logits, *, top_k, temperature, sample_ids=None):
    """Eq. 7: detached coarse ratios plus one current-student tail correction.

    Samples are fresh draws from the current objective distribution, not rollout
    tokens. Exact masked tail sums avoid introducing a tail-floor approximation.
    ``sample_ids`` enables exact enumeration of the estimator in correctness tests.
    """
    if not 1 <= top_k < student_logits.shape[-1] or temperature <= 0:
        raise ValueError("Require 0 < top_k < vocabulary and positive temperature")
    lp = (student_logits / temperature).log_softmax(-1)
    with torch.no_grad():
        lq = (teacher_logits / temperature).log_softmax(-1)
        q_head, indices = lq.topk(top_k, dim=-1)
        in_head = torch.zeros_like(lq, dtype=torch.bool).scatter_(-1, indices, True)
        q_tail = lq.masked_fill(in_head, -torch.inf).logsumexp(-1)
        if sample_ids is None:
            sample_ids = torch.multinomial(lp.detach().exp(), 1)
    p_head = lp.gather(-1, indices)
    p_tail = lp.masked_fill(in_head, -torch.inf).logsumexp(-1)
    p_coarse = torch.cat((p_head, p_tail.unsqueeze(-1)), -1)
    q_coarse = torch.cat((q_head, q_tail.unsqueeze(-1)), -1)
    coarse = (p_coarse.exp() * (p_coarse - q_coarse).detach()).sum(-1)
    sample_lp = lp.gather(-1, sample_ids).squeeze(-1)
    sample_lq = lq.gather(-1, sample_ids).squeeze(-1)
    outside = ~in_head.gather(-1, sample_ids).squeeze(-1)
    ratio = (sample_lp - p_tail - sample_lq + q_tail).detach()
    correction = outside * (sample_lp - sample_lp.detach()).exp() * ratio
    entropy = -(lp.detach().exp() * lp.detach()).sum()
    return coarse + correction, entropy


def taopd_loss(
    hidden,
    weight,
    teacher_hidden,
    teacher_weight,
    *,
    config,
    top_k,
    tail_epsilon,
    bias=None,
    teacher_bias=None,
    sample_corrected=False,
):
    if config.direction != "reverse":
        raise ValueError("TA-OPD is a reverse-KL baseline")

    def chunk(h, w, ht, wt):
        with torch.no_grad():
            teacher_logits = project(ht, wt, teacher_bias)
        values_fn = sctaopd_values if sample_corrected else taopd_values
        options = {} if sample_corrected else {"tail_epsilon": tail_epsilon}
        values, entropy = values_fn(
            project(h, w, bias),
            teacher_logits,
            top_k=top_k,
            temperature=config.temperature,
            **options,
        )
        return values.sum(), entropy

    total = hidden.new_zeros((), dtype=torch.float32)
    entropy = total.clone()
    for start in range(0, hidden.shape[0], config.chunk_size):
        stop = start + config.chunk_size
        value, ent = checkpoint(
            chunk,
            hidden[start:stop],
            weight,
            teacher_hidden[start:stop],
            teacher_weight,
            use_reentrant=False,
        )
        total = total + value
        entropy = entropy + ent
    # Match the shared loss adapter's T²-scaled mean convention.
    return total * config.temperature**2 / hidden.shape[0], entropy
