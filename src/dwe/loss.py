"""Sparse KL head backward. The forward returns actual KL, not a sampled loss value."""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from .objective import kl_adjoint
from .sampling import sample_residual


@dataclass(frozen=True)
class LossConfig:
    direction: str = "forward"
    temperature: float = 1.0
    method: str = "quantile"
    pairs: int = 4
    chunk_size: int = 128

    def __post_init__(self):
        if self.direction not in ("forward", "reverse"):
            raise ValueError("direction must be forward or reverse")
        if self.method not in (
            "independent",
            "coupled",
            "quantile",
            "coordinate",
            "topk",
            "sketch",
        ):
            raise ValueError("unknown sampling method")
        if self.temperature <= 0 or self.pairs < 1 or self.chunk_size < 1:
            raise ValueError("temperature, pairs and chunk_size must be positive")


def project(hidden: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    logits = hidden @ weight.T
    return logits if bias is None else logits + bias


def sparse_products(
    hidden,
    weight,
    indices,
    coefficients,
    chunk_size,
    *,
    shared_columns=False,
    needs=(True, True, False),
):
    """Apply selected logit adjoints to the two output-head gradient products."""
    dtype = coefficients.dtype
    need_h, need_w, need_b = needs
    gh = torch.empty_like(hidden) if need_h else None
    # One accumulation buffer for the entire head, rather than one per chunk.
    gw = torch.zeros(weight.shape, device=weight.device, dtype=dtype) if need_w else None
    gb = torch.zeros(weight.shape[0], device=weight.device, dtype=dtype) if need_b else None
    for start in range(0, hidden.shape[0], chunk_size):
        stop = start + chunk_size
        ids = indices[start:stop]
        coef = coefficients[start:stop]
        if shared_columns:
            # The paper's mask is shared across the chunk: use small GEMMs.
            selected = ids[0]
            if need_h:
                gh[start:stop] = (coef @ weight[selected].to(dtype)).to(hidden.dtype)
            if need_w:
                gw.index_add_(0, selected, coef.T @ hidden[start:stop].to(dtype))
            if need_b:
                gb.index_add_(0, selected, coef.sum(0))
            continue
        if need_h:
            rows = weight[ids].to(dtype)
            gh[start:stop] = (coef.unsqueeze(-1) * rows).sum(1).to(hidden.dtype)
        if need_w:
            updates = coef.unsqueeze(-1) * hidden[start:stop].to(dtype).unsqueeze(1)
            gw.index_add_(0, ids.flatten(), updates.flatten(0, 1))
        if need_b:
            gb.index_add_(0, ids.flatten(), coef.flatten())
    return gh, gw, gb


class _SparseHead(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden,
        weight,
        teacher_hidden,
        teacher_weight,
        bias,
        teacher_bias,
        config,
        order,
        generator,
        need_entropy,
    ):
        n = hidden.shape[0]
        indices, coefficients = [], []
        value = None
        entropy = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, n, config.chunk_size):
            stop = start + config.chunk_size
            terms = kl_adjoint(
                project(hidden[start:stop], weight, bias),
                project(teacher_hidden[start:stop], teacher_weight, teacher_bias),
                config.direction,
                config.temperature,
                return_entropy=need_entropy,
            )
            losses, residual = terms[:2]
            if need_entropy:
                entropy = entropy + terms[2]
            ids, coef = sample_residual(residual, config.pairs, config.method, order, generator)
            indices.append(ids)
            coefficients.append(coef / n)
            value = losses.sum() if value is None else value + losses.sum()
        ctx.save_for_backward(hidden, weight, torch.cat(indices), torch.cat(coefficients))
        ctx.chunk_size = config.chunk_size
        ctx.shared_columns = config.method == "sketch"
        ctx.bias_dtype = None if bias is None else bias.dtype
        ctx.mark_non_differentiable(entropy)
        return value / n, entropy

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output, grad_entropy):
        hidden, weight, indices, coefficients = ctx.saved_tensors
        coefficients = coefficients * grad_output.to(coefficients.dtype)
        gh, gw, gb = sparse_products(
            hidden,
            weight,
            indices,
            coefficients,
            ctx.chunk_size,
            shared_columns=ctx.shared_columns,
            needs=(*ctx.needs_input_grad[:2], ctx.needs_input_grad[4]),
        )
        return (
            gh,
            None if gw is None else gw.to(weight.dtype),
            None,
            None,
            None if gb is None else gb.to(ctx.bias_dtype),
            None,
            None,
            None,
            None,
            None,
        )


def distillation_loss(
    hidden: Tensor,
    weight: Tensor,
    teacher_hidden: Tensor,
    teacher_weight: Tensor,
    *,
    config: LossConfig,
    bias: Tensor | None = None,
    teacher_bias: Tensor | None = None,
    order: Tensor | None = None,
    generator: torch.Generator | None = None,
    return_entropy: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Mean KL for selected positions, with the configured sparse first-order backward.

    The caller handles causal shifting, masks, and task-specific weighting. Teacher
    inputs are frozen. All tensors must share a device and compatible projection dtypes.
    The stochastic methods are unbiased; deterministic ``topk`` is a biased control.
    """
    if hidden.ndim != 2 or teacher_hidden.ndim != 2 or hidden.shape[0] == 0:
        raise ValueError("hidden states must be nonempty [positions, hidden_size] matrices")
    if hidden.shape[0] != teacher_hidden.shape[0] or weight.shape[0] != teacher_weight.shape[0]:
        raise ValueError("student and teacher must share positions and vocabulary")
    if teacher_hidden.requires_grad or teacher_weight.requires_grad:
        raise ValueError("teacher hidden states and weight must be frozen")
    if teacher_bias is not None and teacher_bias.requires_grad:
        raise ValueError("teacher bias must be frozen")
    value, entropy = _SparseHead.apply(
        hidden,
        weight,
        teacher_hidden,
        teacher_weight,
        bias,
        teacher_bias,
        config,
        order,
        generator,
        return_entropy,
    )
    return (value, entropy) if return_entropy else value
