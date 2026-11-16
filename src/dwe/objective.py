"""Exact KL values and logit adjoints; no sampling or model dependencies."""

import torch
from torch import Tensor

from ._pointwise import forward_residual, reverse_residual


def kl_adjoint(
    student_logits: Tensor,
    teacher_logits: Tensor,
    direction: str,
    temperature: float,
    *,
    return_entropy: bool = False,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
    """Return per-position T² KL and its derivative w.r.t. unscaled student logits."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    dtype = torch.float64 if student_logits.dtype == torch.float64 else torch.float32
    lp = (student_logits.to(dtype) / temperature).log_softmax(-1)
    lq = (teacher_logits.to(dtype) / temperature).log_softmax(-1)
    p = lp.exp()
    if direction == "forward":
        q = lq.exp()
        value = (q * (lq - lp)).sum(-1)
        residual = forward_residual(p, q, temperature)
    elif direction == "reverse":
        ratio = lp - lq
        value = (p * ratio).sum(-1)
        residual = reverse_residual(p, ratio, value, temperature)
    else:
        raise ValueError(f"Unknown KL direction: {direction}")
    if return_entropy:
        return temperature**2 * value, residual, -(p * lp).sum()
    return temperature**2 * value, residual
