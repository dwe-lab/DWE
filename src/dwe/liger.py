"""Thin KL adapter over Liger's upstream chunked distillation implementation."""

import torch
from liger_kernel.chunked_loss.fused_linear_distillation import LigerFusedLinearDistillationBase

from .objective import kl_adjoint


class _KL(LigerFusedLinearDistillationBase):
    @staticmethod
    def distillation_loss_fn(
        student_logits,
        teacher_logits,
        target=None,
        ignore_index=-100,
        direction="forward",
        loss_scale=1.0,
        **kwargs,
    ):
        values, _ = kl_adjoint(student_logits, teacher_logits, direction, 1.0)
        return values.sum() * loss_scale

    @staticmethod
    def forward(ctx, hidden, weight, teacher_hidden, teacher_weight, bias, teacher_bias, config):
        target = torch.zeros(hidden.shape[0], device=hidden.device, dtype=torch.long)
        return LigerFusedLinearDistillationBase.forward(
            _KL,
            ctx,
            hidden,
            weight,
            teacher_hidden,
            teacher_weight,
            target,
            student_bias=bias,
            teacher_bias=teacher_bias,
            chunk_size=config.chunk_size,
            compute_ce_loss=False,
            weight_hard_loss=0.0,
            weight_soft_loss=1.0,
            temperature=config.temperature,
            direction=config.direction,
            loss_scale=config.temperature**2,
        )

    @staticmethod
    def backward(ctx, grad_output):
        gh, gw, _, _, _, gb = LigerFusedLinearDistillationBase.backward(ctx, grad_output)
        return gh, gw, None, None, gb, None, None


def liger_loss(
    hidden, weight, teacher_hidden, teacher_weight, *, config, bias=None, teacher_bias=None
):
    return _KL.apply(hidden, weight, teacher_hidden, teacher_weight, bias, teacher_bias, config)
