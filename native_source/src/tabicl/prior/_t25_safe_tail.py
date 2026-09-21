"""Runtime patch for the registered T25/TL regression-generator arms.

The patch changes only two generator surfaces:

* RW target-template tails become support-only, monotone and bounded, followed
  by a loss-space clip at +/-8.
* TL optionally replaces the fixed 4096 episode length and the uniform
  support fraction with the frozen, bounded curriculum registered in the plan.

No benchmark data, query target, domain id, or E4 cross-table signal is read.
"""

from __future__ import annotations

import math
import os
from typing import Any, Final

import numpy as np
try:
    import torch
    from torch import Tensor
except ModuleNotFoundError:  # Local contract checks may not bundle PyTorch.
    torch = None
    Tensor = Any


SEQ_BUCKETS: Final = (
    (256, 1024, 0.15),
    (1025, 2048, 0.25),
    (2049, 3072, 0.20),
    (3073, 4096, 0.40),
)
SUPPORT_BUCKETS: Final = (
    (0.30, 0.50, 0.15),
    (0.50, 0.70, 0.35),
    (0.70, 0.90, 0.50),
)

_INSTALLED = False


def _draw_bucket(buckets):
    value = float(np.random.random())
    cumulative = 0.0
    for low, high, probability in buckets:
        cumulative += probability
        if value < cumulative:
            return low, high
    return buckets[-1][0], buckets[-1][1]


def sample_tl_sequence_length(
    min_seq_len,
    max_seq_len: int,
    log: bool = False,
    replay_small: bool = False,
) -> int:
    """Sample the registered TL length distribution, always <=4096."""

    del min_seq_len, log, replay_small
    if int(max_seq_len) != 4096:
        raise ValueError(f"TL requires max_seq_len=4096, got {max_seq_len}")
    low, high = _draw_bucket(SEQ_BUCKETS)
    return int(np.random.randint(int(low), int(high) + 1))


def sample_tl_train_size(min_train_size, max_train_size, seq_len: int) -> int:
    """Sample the registered support-fraction strata using support-only bounds."""

    if (float(min_train_size), float(max_train_size)) != (0.3, 0.9):
        raise ValueError(
            "TL requires the launcher guard min_train_size=0.3 and max_train_size=0.9"
        )
    low, high = _draw_bucket(SUPPORT_BUCKETS)
    fraction = float(np.random.uniform(float(low), float(high)))
    return min(max(int(int(seq_len) * fraction), 2), int(seq_len) - 1)


def _safe_support_only_standardize(y: Tensor, train_size: int) -> Tensor:
    from tabicl.prior import _regression_target_prior as target_prior

    target_prior._validate_target(y, train_size)
    support = y[:train_size]
    mean = torch.mean(support, dim=0)
    std = torch.std(support, dim=0, correction=0)
    if not torch.isfinite(std).all():
        raise target_prior.NonFinitePriorError(
            "support target standard deviation is non-finite"
        )
    safe_std = torch.where(std > 0, std, torch.ones_like(std))
    scaled = (y - mean) / safe_std
    return torch.clip(scaled, min=-8.0, max=8.0)


def _interpolate_inside(probabilities: Tensor, levels: Tensor, template: Tensor) -> Tensor:
    levels = levels.to(device=probabilities.device, dtype=probabilities.dtype)
    template = template.to(device=probabilities.device, dtype=probabilities.dtype)
    probabilities = probabilities.clamp(levels[0], levels[-1])
    indices = torch.searchsorted(levels, probabilities, right=True)
    lo = indices.clamp(1, levels.numel() - 1) - 1
    hi = lo + 1
    width = (levels[hi] - levels[lo]).clamp_min(torch.finfo(levels.dtype).eps)
    weight = (probabilities - levels[lo]) / width
    return template[lo] + weight * (template[hi] - template[lo])


def _safe_map_to_template_from_support(
    y: Tensor,
    train_size: int,
    levels: Tensor,
    template: Tensor,
) -> Tensor:
    """Support-ECDF interior mapping with saturating target-space tails."""

    from tabicl.prior import _regression_target_prior as target_prior

    target_prior._validate_target(y, train_size)
    support = torch.sort(y[:train_size, 0]).values.contiguous()
    values = y[:, 0].contiguous()
    left = torch.searchsorted(support, values, right=False)
    right = torch.searchsorted(support, values, right=True)
    probabilities = (
        left.to(dtype=values.dtype) + right.to(dtype=values.dtype)
    ) / (2.0 * float(train_size))
    mapped = _interpolate_inside(probabilities, levels, template)

    support_q25, support_q75 = torch.quantile(
        support,
        torch.tensor((0.25, 0.75), device=support.device, dtype=support.dtype),
    )
    support_iqr = support_q75 - support_q25
    support_std = torch.std(support, correction=0)
    robust_scale = torch.maximum(support_iqr / 1.349, support_std)
    robust_scale = torch.clamp(robust_scale, min=torch.finfo(support.dtype).eps)

    level_tensor = levels.to(device=values.device, dtype=values.dtype)
    template_tensor = template.to(device=values.device, dtype=values.dtype)
    template_quartiles = _interpolate_inside(
        torch.tensor((0.25, 0.75), device=values.device, dtype=values.dtype),
        level_tensor,
        template_tensor,
    )
    template_iqr = (template_quartiles[1] - template_quartiles[0]).clamp_min(0.0)
    gamma = 0.20 * min(1.0, math.sqrt(float(train_size) / 512.0))
    tail_budget = float(gamma) * template_iqr

    lower_tail = template_tensor[0] - tail_budget * torch.tanh(
        (support[0] - values) / robust_scale
    )
    upper_tail = template_tensor[-1] + tail_budget * torch.tanh(
        (values - support[-1]) / robust_scale
    )
    mapped = torch.where(values < support[0], lower_tail, mapped)
    mapped = torch.where(values > support[-1], upper_tail, mapped)
    if not torch.isfinite(mapped).all():
        raise target_prior.NonFinitePriorError(
            "safe-tail template mapping produced non-finite targets"
        )
    return mapped.view(-1, 1)


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from tabicl.prior import _dataset
    from tabicl.prior import _regression_target_prior as target_prior

    target_prior.support_only_standardize = _safe_support_only_standardize
    target_prior.map_to_template_from_support = _safe_map_to_template_from_support

    length_enabled = os.environ.get(
        "REGRESSION_TL_LENGTH_CURRICULUM_ENABLED", "false"
    ).lower() == "true"
    if length_enabled:
        _dataset.Prior.sample_seq_len = staticmethod(sample_tl_sequence_length)
        _dataset.Prior.sample_train_size = staticmethod(sample_tl_train_size)
    _INSTALLED = True


def contract() -> dict:
    return {
        "safe_tail": {
            "fit_split": "support_only",
            "gamma": "0.20*min(1,sqrt(n_support/512))",
            "shape": "template_IQR*tanh(query_distance/support_robust_scale)",
            "loss_clip": [-8.0, 8.0],
            "query_targets_fit_statistics": False,
        },
        "length_curriculum": {
            "enabled": os.environ.get(
                "REGRESSION_TL_LENGTH_CURRICULUM_ENABLED", "false"
            ).lower() == "true",
            "sequence_buckets": [list(item) for item in SEQ_BUCKETS],
            "support_buckets": [list(item) for item in SUPPORT_BUCKETS],
            "maximum_sequence_length": 4096,
        },
    }
