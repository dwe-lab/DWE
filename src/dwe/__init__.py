"""Sparse output-head gradients with the original distillation objective."""

from .loss import LossConfig, distillation_loss

__all__ = ["LossConfig", "distillation_loss"]
