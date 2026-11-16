"""Fuse bandwidth-bound CUDA operations without changing reductions or sampling."""

from functools import wraps

import torch


def _cuda_fused(function):
    compiled = torch.compile(function, fullgraph=True)

    @wraps(function)
    def call(*args):
        return compiled(*args) if args[0].is_cuda else function(*args)

    return call


@_cuda_fused
def reverse_residual(probability, ratio, value, temperature):
    return temperature * probability * (ratio - value.unsqueeze(-1))


@_cuda_fused
def forward_residual(student, teacher, temperature):
    return temperature * (student - teacher)


@_cuda_fused
def ordered_parts(residual, order):
    ordered = residual.index_select(1, order) if order is not None else residual
    return ordered.clamp_min(0), ordered.neg().clamp_min(0)
