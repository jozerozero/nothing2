"""Learning rate scheduler."""

from __future__ import annotations

from transformers import (
    get_constant_schedule,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)


import math
from functools import partial
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler


class Muon(Optimizer):
    """Muon optimizer with AdamW fallback for non-matrix parameters.

    Matrix parameters use momentum followed by Newton-Schulz orthogonalization.
    Vectors and scalars use a small AdamW-style update so biases and LayerNorm
    parameters remain trainable under a single optimizer/scheduler interface.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        ns_steps: int = 5,
        cautious_weight_decay: bool = False,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            ns_steps=ns_steps,
            cautious_weight_decay=cautious_weight_decay,
            adam_betas=adam_betas,
            eps=eps,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _zeropower_via_newton_schulz(update: torch.Tensor, steps: int, eps: float) -> torch.Tensor:
        original_shape = update.shape
        x = update.reshape(update.shape[0], -1).float()
        if x.numel() == 0:
            return update

        transposed = x.shape[0] > x.shape[1]
        if transposed:
            x = x.T
        x = x / x.norm().clamp_min(eps)

        # Coefficients from common Muon implementations for fast orthogonalization.
        a, b, c = 3.4445, -4.7750, 2.0315
        for _ in range(int(steps)):
            gram = x @ x.T
            x = a * x + (b * gram + c * gram @ gram) @ x
        if transposed:
            x = x.T
        return x.reshape(original_shape).to(dtype=update.dtype, device=update.device)

    @staticmethod
    def _apply_weight_decay(
        param: torch.Tensor,
        update: torch.Tensor,
        lr: float,
        weight_decay: float,
        cautious: bool,
    ) -> None:
        if weight_decay == 0.0:
            return
        if cautious:
            mask = (param * update) > 0
            param.addcmul_(param, mask.to(dtype=param.dtype), value=-lr * weight_decay)
        else:
            param.mul_(1.0 - lr * weight_decay)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]
            cautious_weight_decay = group["cautious_weight_decay"]
            beta1, beta2 = group["adam_betas"]
            eps = group["eps"]

            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")

                state = self.state[param]
                if param.ndim >= 2:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(param)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad)
                    update = grad.add(buf, alpha=momentum)
                    update = self._zeropower_via_newton_schulz(update, ns_steps, eps)
                    self._apply_weight_decay(param, update, lr, weight_decay, cautious_weight_decay)
                    param.add_(update, alpha=-lr)
                else:
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(param)
                        state["exp_avg_sq"] = torch.zeros_like(param)
                    state["step"] += 1
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    update = (exp_avg / bias_correction1) / ((exp_avg_sq / bias_correction2).sqrt() + eps)
                    self._apply_weight_decay(param, update, lr, weight_decay, cautious_weight_decay)
                    param.add_(update, alpha=-lr)

        return loss


class CosineWarmupLR(LRScheduler):
    """Cosine warmup scheduler equivalent to transformers' cosine_warmup lambda."""

    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int | float,
        num_training_steps: int,
        num_cycles: float = 0.5,
        lr_floor: float = 0.0,
        last_epoch: int = -1,
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_cycles = num_cycles
        self.lr_floor = lr_floor
        super().__init__(optimizer, last_epoch)

    def _factor(self, current_step: int) -> float:
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        progress = float(current_step - self.num_warmup_steps) / float(
            max(1, self.num_training_steps - self.num_warmup_steps)
        )
        factor = 0.5 * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress))
        return max(0.0, factor)

    def get_lr(self):
        factor = self._factor(self.last_epoch)
        if self.lr_floor > 0 and self.last_epoch >= self.num_warmup_steps:
            return [max(self.lr_floor, base_lr * factor) for base_lr in self.base_lrs]
        return [base_lr * factor for base_lr in self.base_lrs]


def _get_cosine_with_restarts_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int,
    amplitude_decay: float,
    lr_end: float = 0.0,
    lr_init: float = 1.0,
):
    """
    Compute the learning rate factor for a cosine schedule with warmup, hard restarts, and amplitude scaling.
    """
    if current_step < num_warmup_steps:
        # Warmup phase: Linearly increase learning rate
        return float(current_step) / float(max(1, num_warmup_steps))

    # After warmup: Apply cosine schedule with hard restarts and amplitude scaling
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    if progress >= 1.0:
        return lr_end / lr_init  # as LambdaLR multiplies by lr_init

    # Determine which cycle the current step is in
    cycle_progress = (float(num_cycles) * progress) % 1.0
    current_cycle = int(float(num_cycles) * progress)
    amplitude = amplitude_decay**current_cycle  # Exponentially decay amplitude per cycle

    # Calculate the current learning rate with proper scaling
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * cycle_progress))
    current_lr = lr_end + (lr_init - lr_end) * cosine_factor * amplitude
    return current_lr / lr_init  # as LambdaLR multiplies by lr_init


def get_cosine_with_restarts(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int = 1,
    amplitude_decay: float = 1.0,
    lr_end: float = 0.0,
    last_epoch: int = -1,
):
    """Create a learning rate scheduler with warmup, cosine decay, hard restarts, and amplitude scaling.

    Parameters
    ----------
    optimizer : Optimizer
        The optimizer for which to schedule the learning rate.

    num_warmup_steps : int
        Number of warmup steps.

    num_training_steps : int
        Total number of training steps.

    num_cycles : int, default=1
        Number of hard restarts.

    amplitude_decay : float, default=1.0
        Factor to exponentially decay the max LR per cycle.

    lr_end : float, default=0.0
        Minimum learning rate at the end of each cycle.

    last_epoch : int, default=-1
        The index of the last epoch.

    Returns
    -------
    LambdaLR
        A learning rate scheduler.
    """
    lr_init = optimizer.defaults["lr"]
    if lr_end > lr_init:
        raise ValueError(f"lr_end ({lr_end}) must be smaller than initial lr ({lr_init})")

    lr_lambda = partial(
        _get_cosine_with_restarts_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        amplitude_decay=amplitude_decay,
        lr_end=lr_end,
        lr_init=lr_init,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_scheduler(config, optimizer):
    """Get the learning rate scheduler based on the training configuration."""

    if config.warmup_proportion >= 0:
        warmup_steps = config.max_steps * config.warmup_proportion
    else:
        warmup_steps = config.warmup_steps
    lr_floor = float(getattr(config, "lr_floor", 0.0))

    if config.scheduler == "constant":
        scheduler = get_constant_schedule(optimizer=optimizer)
    elif config.scheduler == "linear_warmup":
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=config.max_steps
        )
    elif config.scheduler == "cosine_warmup":
        if bool(getattr(config, "fast_cosine_scheduler", False)) or lr_floor > 0:
            scheduler = CosineWarmupLR(
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=config.max_steps,
                lr_floor=lr_floor,
            )
        else:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=config.max_steps
            )
    elif config.scheduler == "cosine_with_restarts":
        scheduler = get_cosine_with_restarts(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=config.max_steps,
            num_cycles=config.cosine_num_cycles,
            amplitude_decay=config.cosine_amplitude_decay,
            lr_end=config.cosine_lr_end,
        )
    elif config.scheduler == "polynomial_decay_warmup":
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=config.max_steps,
            lr_end=config.poly_decay_lr_end,
            power=config.poly_decay_power,
        )
    else:
        raise NotImplementedError

    return scheduler
