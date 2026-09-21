"""Leakage-safe preprocessing for synthetic in-context regression tasks."""

from __future__ import annotations

import torch
from torch import Tensor


class NonFinitePriorError(ValueError):
    """A generated synthetic task is invalid and must be resampled."""


def _validate_matrix(values: Tensor, train_size: int) -> None:
    if values.ndim != 2:
        raise ValueError(f"expected a rank-2 tensor, got shape={tuple(values.shape)}")
    if not 2 <= train_size < values.shape[0]:
        raise ValueError(
            f"invalid train_size={train_size} for sequence length {values.shape[0]}"
        )
    if not torch.isfinite(values).all():
        raise NonFinitePriorError("support-only preprocessing requires finite inputs")


def _population_location_scale(support: Tensor) -> tuple[Tensor, Tensor]:
    mean = support.mean(dim=0)
    std = support.std(dim=0, correction=0)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise NonFinitePriorError("support statistics are non-finite")
    # Match sklearn StandardScaler: constant columns use scale_=1.  Query
    # residuals are therefore retained instead of being silently zeroed.
    scale = torch.where(std > 0, std, torch.ones_like(std))
    return mean, scale


def support_only_outlier_remove_and_standardize(
    values: Tensor,
    train_size: int,
    *,
    threshold: float = 4.0,
    clip_value: float = 100.0,
) -> Tensor:
    """Fit robust bounds and StandardScaler statistics on support only.

    The fitted bounds and population mean/scale are then applied unchanged to
    both support and query rows.  Query rows never contribute to any fitted
    statistic.  This mirrors the information boundary available at inference.
    """

    _validate_matrix(values, train_size)
    if threshold <= 0 or clip_value <= 0:
        raise ValueError("threshold and clip_value must be positive")

    support = values[:train_size]
    initial_mean, initial_scale = _population_location_scale(support)
    initial_lower = initial_mean - threshold * initial_scale
    initial_upper = initial_mean + threshold * initial_scale

    inlier_mask = (support >= initial_lower) & (support <= initial_upper)
    inlier_count = inlier_mask.sum(dim=0)
    inlier_sum = torch.where(inlier_mask, support, torch.zeros_like(support)).sum(dim=0)
    robust_mean = inlier_sum / inlier_count.clamp_min(1).to(support.dtype)

    centered = torch.where(
        inlier_mask,
        support - robust_mean,
        torch.zeros_like(support),
    )
    robust_var = (centered.square().sum(dim=0) / inlier_count.clamp_min(1).to(support.dtype))
    robust_scale = robust_var.sqrt()

    # If a column has no inliers, fall back to the first-pass support statistics.
    # Constant support columns use unit scale for both bounds and StandardScaler,
    # matching sklearn semantics and retaining bounded query residuals.
    robust_mean = torch.where(inlier_count > 0, robust_mean, initial_mean)
    robust_scale = torch.where(inlier_count > 0, robust_scale, initial_scale)
    robust_bound_scale = torch.where(
        robust_scale > 0,
        robust_scale,
        torch.ones_like(robust_scale),
    )
    lower = robust_mean - threshold * robust_bound_scale
    upper = robust_mean + threshold * robust_bound_scale
    bounded = torch.clamp(values, min=lower, max=upper)

    bounded_support = bounded[:train_size]
    mean, scale = _population_location_scale(bounded_support)
    standardized = (bounded - mean) / scale
    if not torch.isfinite(standardized).all():
        raise NonFinitePriorError("support-only preprocessing produced non-finite outputs")
    return torch.clamp(standardized, min=-clip_value, max=clip_value)
