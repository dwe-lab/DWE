"""A loss adapter for TRL; generation, training and reductions remain upstream."""

import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
from transformers.trainer_utils import get_last_checkpoint
from trl import DistillationTrainer
from trl.trainer.distillation_trainer import maybe_gather_lm_head_ctx

from .geometry import principal_order, validate_order
from .loss import LossConfig, distillation_loss


class CouplingTrainer(DistillationTrainer):
    """Replace only the distribution loss through the reviewed upstream hook."""

    def __init__(
        self,
        *args,
        estimator: LossConfig,
        loss_backend: str,
        taopd_config=None,
        vocabulary_order: str | None = None,
        **kwargs,
    ):
        if not hasattr(DistillationTrainer, "distribution_loss"):
            raise RuntimeError("Run scripts/patch_trl.py against locked TRL 1.13.0")
        if loss_backend not in ("sparse", "liger", "taopd", "sctaopd"):
            raise ValueError("loss_backend must be sparse, liger, taopd, or sctaopd")
        self.estimator = estimator
        self.loss_backend = loss_backend
        self.taopd_config = taopd_config
        coupled = loss_backend == "sparse" and estimator.method in ("coupled", "quantile")
        self.vocabulary_order = vocabulary_order or ("principal" if coupled else "natural")
        if self.vocabulary_order not in ("natural", "principal"):
            raise ValueError("vocabulary_order must be natural or principal")
        if self.vocabulary_order == "principal" and not coupled:
            raise ValueError("Principal ordering requires coupled or quantile sampling")
        self.order = None
        self.geometry_setup_seconds = 0.0
        if loss_backend in ("taopd", "sctaopd") and taopd_config is None:
            raise ValueError("TA-OPD requires its baseline configuration")
        super().__init__(*args, **kwargs)
        if loss_backend in ("taopd", "sctaopd") and self.beta != 1.0:
            raise ValueError("TA-OPD requires beta=1 (reverse KL)")
        if self.beta not in (0.0, 1.0):
            raise ValueError("The estimator supports forward/reverse KL, not mixture JSD")

    def estimator_state(self):
        """Resolved method settings saved with every checkpoint."""
        return dict(
            loss_backend=self.loss_backend,
            estimator=asdict(
                replace(
                    self.estimator,
                    direction="forward" if self.beta == 0 else "reverse",
                    temperature=self.args.temperature,
                )
            ),
            vocabulary_order=self.vocabulary_order,
            taopd_config=self.taopd_config,
        )

    def prepare_sampling(self, checkpoint=None):
        """Initialize once before timing, or restore the exact checkpoint order.

        Call before train only when the setup metadata is needed in a manifest.
        Native train also calls this, including for DeepSpeed/FSDP resumes.
        """
        if checkpoint is not None:
            checkpoint = Path(checkpoint)
            state_path = checkpoint / "estimator.json"
            if not state_path.is_file():
                raise ValueError("Checkpoint lacks estimator.json; use its original pinned release")
            if json.loads(state_path.read_text()) != self.estimator_state():
                raise ValueError("Checkpoint estimator settings differ from this run")
            path = checkpoint / "geometry.pt"
            if self.vocabulary_order == "principal":
                if not path.is_file():
                    raise ValueError("Checkpoint is missing its fixed geometry.pt")
                order = torch.load(path, map_location="cpu", weights_only=True)
                head = self.model.get_output_embeddings().weight
                vocabulary = head.ds_shape[0] if hasattr(head, "ds_shape") else head.shape[0]
                validate_order(order, vocabulary)
                self.order = order.to(self.accelerator.device)
            elif path.exists():
                raise ValueError("Natural-order checkpoint unexpectedly contains geometry.pt")
            return
        if self.vocabulary_order == "natural" or self.order is not None:
            return
        from torch.distributed.tensor import DTensor

        weight = self.model.get_output_embeddings().weight
        weight = weight.full_tensor() if isinstance(weight, DTensor) else weight
        started = time.perf_counter()
        with maybe_gather_lm_head_ctx(weight, None):
            if self.is_world_process_zero():
                self.order = principal_order(weight).to(self.accelerator.device)
            else:
                self.order = torch.empty(
                    weight.shape[0], device=self.accelerator.device, dtype=torch.long
                )
            if torch.distributed.is_initialized():
                torch.distributed.broadcast(self.order, src=0)
        if self.order.is_cuda:
            torch.cuda.synchronize(self.order.device)
        self.geometry_setup_seconds = time.perf_counter() - started

    def train(self, resume_from_checkpoint=None, *args, **kwargs):
        if resume_from_checkpoint is True:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError("No checkpoint found in output_dir")
        checkpoint = resume_from_checkpoint or None
        self.prepare_sampling(checkpoint)
        return super().train(checkpoint, *args, **kwargs)

    def save_model(self, output_dir=None, _internal_call=False):
        self.prepare_sampling()
        super().save_model(output_dir, _internal_call=_internal_call)
        if self.args.should_save:
            output = Path(output_dir or self.args.output_dir)
            (output / "estimator.json").write_text(
                json.dumps(self.estimator_state(), indent=2) + "\n"
            )
            if self.order is not None:
                torch.save(self.order.cpu(), output / "geometry.pt")

    def distribution_loss(
        self,
        student_hidden_states,
        teacher_hidden_states,
        student_lm_head_weight,
        teacher_lm_head_weight,
        completion_mask,
        beta,
        chunk_size,
        num_items_in_batch=None,
        student_lm_head_bias=None,
        teacher_lm_head_bias=None,
        student_logit_scale=1.0,
        teacher_logit_scale=1.0,
        student_final_logit_softcapping=None,
        teacher_final_logit_softcapping=None,
        temperature=1.0,
    ):
        if (
            student_final_logit_softcapping is not None
            or teacher_final_logit_softcapping is not None
        ):
            raise ValueError(
                "Sparse linear-head estimation does not implement nonlinear softcapping"
            )
        # TRL supplies already-shifted completion states and the global accumulation denominator.
        valid = completion_mask.reshape(-1).bool()
        count = valid.sum()
        h = student_hidden_states.reshape(-1, student_hidden_states.shape[-1])[valid]
        ht = teacher_hidden_states.reshape(-1, teacher_hidden_states.shape[-1])[valid].detach()
        weights = [
            student_lm_head_weight,
            teacher_lm_head_weight,
            student_lm_head_bias,
            teacher_lm_head_bias,
        ]
        from torch.distributed.tensor import DTensor

        weights = [w.full_tensor() if isinstance(w, DTensor) else w for w in weights]
        w, wt, b, bt = weights
        with maybe_gather_lm_head_ctx(w, b), maybe_gather_lm_head_ctx(wt, bt):
            # Casts remain differentiable for the student; teacher parameters stay frozen.
            w = w.to(h.dtype)
            if student_logit_scale != 1.0:
                w = w * student_logit_scale
            wt = wt.detach().to(ht.dtype)
            if teacher_logit_scale != 1.0:
                wt = wt * teacher_logit_scale
            b = None if b is None else b.to(h.dtype) * student_logit_scale
            bt = None if bt is None else bt.detach().to(ht.dtype) * teacher_logit_scale
            if h.shape[0] == 0:
                zero = h.sum() * 0 + w.sum() * 0
                if b is not None:
                    zero = zero + b.sum() * 0
                return zero, zero.detach(), count
            cfg = replace(
                self.estimator,
                direction="forward" if beta == 0 else "reverse",
                temperature=temperature,
            )
            if self.loss_backend == "sparse":
                value, entropy = distillation_loss(
                    h,
                    w,
                    ht,
                    wt,
                    config=cfg,
                    bias=b,
                    teacher_bias=bt,
                    order=self.order,
                    return_entropy=True,
                )
            elif self.loss_backend in ("taopd", "sctaopd"):
                from .taopd import taopd_loss

                value, entropy = taopd_loss(
                    h,
                    w,
                    ht,
                    wt,
                    config=cfg,
                    bias=b,
                    teacher_bias=bt,
                    sample_corrected=self.loss_backend == "sctaopd",
                    **self.taopd_config,
                )
            else:
                from .liger import liger_loss

                value = liger_loss(h, w, ht, wt, config=cfg, bias=b, teacher_bias=bt)
                # Preserve TRL's native entropy metric. This additional forward is included in timings.
                with torch.no_grad():
                    entropy = torch.zeros((), device=h.device, dtype=torch.float32)
                    for hs in h.split(cfg.chunk_size):
                        logits = (hs @ w.T).float()
                        if b is not None:
                            logits = logits + b.float()
                        lp = (logits / temperature).log_softmax(-1)
                        entropy -= (lp.exp() * lp).sum()
        denominator = count if num_items_in_batch is None else num_items_in_batch
        denominator = torch.as_tensor(denominator, device=h.device).clamp(min=1)
        # Core kernels return T² * mean(KL); TRL's objective omits T².
        return value * count / denominator / temperature**2, entropy, count
