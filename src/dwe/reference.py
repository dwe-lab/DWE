"""Exact controls use upstream autograd/checkpointing and POT, not a new solver."""

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .loss import LossConfig, project


def exact_loss(
    hidden: Tensor,
    weight: Tensor,
    teacher_hidden: Tensor,
    teacher_weight: Tensor,
    *,
    config: LossConfig,
    checkpoint_chunks: bool = True,
    bias=None,
    teacher_bias=None,
) -> Tensor:
    def chunk(h, w, th, tw):
        dtype = torch.float64 if h.dtype == torch.float64 else torch.float32
        temperature = config.temperature
        with torch.no_grad():
            lq = (project(th, tw, teacher_bias).to(dtype) / temperature).log_softmax(-1)
        lp = (project(h, w, bias).to(dtype) / temperature).log_softmax(-1)
        # Autograd constructs the adjoint; do not also compute an unused analytic one.
        value = lq.exp() * (lq - lp) if config.direction == "forward" else lp.exp() * (lp - lq)
        return temperature**2 * value.sum()

    total = hidden.new_zeros(
        (), dtype=torch.float64 if hidden.dtype == torch.float64 else torch.float32
    )
    for start in range(0, hidden.shape[0], config.chunk_size):
        stop = start + config.chunk_size
        args = (hidden[start:stop], weight, teacher_hidden[start:stop], teacher_weight)
        value = checkpoint(chunk, *args, use_reentrant=False) if checkpoint_chunks else chunk(*args)
        total = total + value
    return total / hidden.shape[0]


def transport_oracle(a: Tensor, b: Tensor, positive_rows: Tensor, negative_rows: Tensor) -> Tensor:
    """Small-support diagnostic only. POT owns the transport optimization."""
    import ot

    cost = torch.cdist(positive_rows.double(), negative_rows.double()).square()
    plan = ot.emd(
        a.detach().double().cpu().numpy(),
        b.detach().double().cpu().numpy(),
        cost.detach().cpu().numpy(),
    )
    return torch.as_tensor(plan, device=a.device, dtype=a.dtype)
