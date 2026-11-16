"""A cached one-dimensional geometry, using PyTorch's existing PCA implementation."""

import torch
from torch import Tensor


@torch.no_grad()
def principal_order(weight: Tensor) -> Tensor:
    """Build a fixed PCA order without consuming the training RNG stream."""
    # MPS PCA runs on CPU; only the small permutation is returned to the device.
    matrix = weight.detach().to(device="cpu" if weight.is_mps else weight.device).float()
    devices = [matrix.device] if matrix.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(0)
        if matrix.is_cuda:
            with torch.cuda.device(matrix.device):
                torch.cuda.manual_seed(0)
        _, _, directions = torch.pca_lowrank(matrix, q=1, center=True)
    direction = directions[:, 0]
    # Fix PCA's sign ambiguity so ties and orientation are reproducible.
    direction = direction * direction[direction.abs().argmax()].sign()
    return (matrix @ direction).argsort(stable=True).to(weight.device)


def validate_order(order: Tensor, vocabulary: int) -> None:
    """Validate once on loading a cache, outside the loss's per-chunk hot path."""
    if order.dtype != torch.long or order.shape != (vocabulary,):
        raise ValueError("order must be a long tensor of length vocabulary")
    if not torch.equal(order.sort().values, torch.arange(vocabulary, device=order.device)):
        raise ValueError("order must contain every vocabulary index exactly once")
