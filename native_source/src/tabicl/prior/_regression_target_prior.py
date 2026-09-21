"""Leakage-safe target processing for the RW-style regression prior."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch
from torch import Tensor

from ._support_only_preprocessing import NonFinitePriorError


@lru_cache(maxsize=8)
def load_anonymous_quantile_profile(path: str) -> tuple[Tensor, Tensor]:
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != 2:
        raise ValueError("regression target profile must use schema version 2")
    if payload.get("profile_type") != "anonymous_train_only_regression_quantile_templates":
        raise ValueError("unexpected regression target profile type")
    if payload.get("source_scope") != "deduplicated_GT_regression_suites":
        raise ValueError("regression target profile must cover deduplicated GT suites")
    privacy = payload.get("privacy_and_leakage", {})
    required_false = (
        "contains_dataset_names",
        "contains_raw_targets",
        "reads_validation_targets",
        "reads_test_targets",
    )
    if any(privacy.get(key) is not False for key in required_false):
        raise ValueError("regression target profile violates the train-only anonymous contract")
    if payload.get("source_split") != "official_train_only":
        raise ValueError("regression target profile must use official training splits only")
    preprocessing = payload.get("model_target_preprocessing", {})
    if (
        preprocessing.get("order") != ["GT identity/asinh transform", "StandardScaler"]
        or preprocessing.get("standard_scaling_ddof") != 0
        or preprocessing.get("fit_split") != "support_only"
        or preprocessing.get("query_targets_never_fit_statistics") is not True
    ):
        raise ValueError("regression target profile preprocessing contract is invalid")
    postprocessing = payload.get("model_target_postprocessing", {})
    if (
        postprocessing.get("order") != ["inverse StandardScaler", "inverse GT identity/asinh transform"]
        or postprocessing.get("before_metrics") is not True
        or postprocessing.get("prediction_units") != "original_target_units"
    ):
        raise ValueError("regression target profile postprocessing contract is invalid")
    levels = torch.tensor(payload["quantile_levels"], dtype=torch.float32)
    templates = torch.tensor(payload["templates"], dtype=torch.float32)
    if levels.ndim != 1 or templates.ndim != 2 or templates.shape[1] != levels.numel():
        raise ValueError("invalid quantile profile shapes")
    if templates.shape[0] < 1 or levels.numel() < 3:
        raise ValueError("quantile profile is empty")
    if payload.get("template_count") != templates.shape[0]:
        raise ValueError("quantile profile template_count does not match templates")
    if not torch.isfinite(levels).all() or not torch.isfinite(templates).all():
        raise ValueError("quantile profile contains non-finite values")
    if not torch.all(levels[1:] > levels[:-1]):
        raise ValueError("quantile levels must be strictly increasing")
    if not torch.all(templates[:, 1:] >= templates[:, :-1]):
        raise ValueError("quantile templates must be monotone")
    kinds = payload.get("template_transform_kinds")
    if kinds is not None and (len(kinds) != templates.shape[0] or any(k not in {"identity", "asinh"} for k in kinds)):
        raise ValueError("invalid target transform metadata in quantile profile")
    suites = payload.get("template_source_suites")
    suite_counts = payload.get("source_suite_template_counts")
    if not isinstance(suites, list) or len(suites) != templates.shape[0]:
        raise ValueError("invalid source-suite metadata in quantile profile")
    if not isinstance(suite_counts, dict):
        raise ValueError("missing source-suite counts in quantile profile")
    actual_suite_counts = {suite: suites.count(suite) for suite in suite_counts}
    if actual_suite_counts != suite_counts or any(suite not in suite_counts for suite in suites):
        raise ValueError("source-suite counts do not match template metadata")
    return levels.contiguous(), templates.contiguous()


def _validate_target(y: Tensor, train_size: int) -> None:
    if y.ndim != 2 or y.shape[1] != 1:
        raise ValueError(f"expected y shape (T, 1), got {tuple(y.shape)}")
    if not torch.isfinite(y).all():
        raise NonFinitePriorError("regression target must contain only finite values")
    if not 2 <= train_size < y.shape[0]:
        raise ValueError(f"invalid train_size={train_size} for sequence length {y.shape[0]}")


def support_only_standardize(y: Tensor, train_size: int) -> Tensor:
    """Apply sklearn-StandardScaler-equivalent statistics from support only."""

    _validate_target(y, train_size)
    support = y[:train_size]
    mean = torch.mean(support, dim=0)
    # StandardScaler uses population variance (ddof/correction=0).  Divide by
    # every representable positive scale and reserve zeros only for constants.
    std = torch.std(support, dim=0, correction=0)
    if not torch.isfinite(std).all():
        raise NonFinitePriorError("support target standard deviation is non-finite")
    # sklearn StandardScaler sets scale_=1 for a constant feature.  Therefore
    # constant support values map to zero while a different query value must
    # remain different from zero; zeroing the full sequence would silently
    # erase the query supervision signal.
    safe_std = torch.where(std > 0, std, torch.ones_like(std))
    scaled = (y - mean) / safe_std
    # This is a loss-space numerical guard, not a statistic fitted on query.
    return torch.clip(scaled, min=-100.0, max=100.0)


def support_only_gt_transform_and_standardize(y: Tensor, train_size: int) -> Tensor:
    """Mirror GT preprocessing using support-only fit statistics.

    The transform decision and its center/scale use support targets only.  The
    same monotone transform is applied to support+query labels, followed by the
    exact population standardization used by ``TabICLRegressor``.
    """

    _validate_target(y, train_size)
    support = y[:train_size, 0]
    quantiles = torch.quantile(
        support,
        torch.tensor((0.01, 0.25, 0.50, 0.75, 0.99), device=y.device, dtype=y.dtype),
    )
    q01, q25, q50, q75, q99 = quantiles.unbind()
    iqr = torch.clamp(q75 - q25, min=1e-12)
    std = torch.clamp(torch.std(support, correction=0), min=1e-12)
    mean = torch.mean(support)
    skew = torch.mean(((support - mean) / std) ** 3)
    tail_ratio = (q99 - q01) / iqr
    use_asinh = bool(((torch.abs(skew) > 2.0) | (tail_ratio > 25.0)).item())
    if use_asinh:
        scale = torch.maximum(torch.maximum(iqr / 1.349, 0.05 * std), torch.tensor(1e-12, device=y.device, dtype=y.dtype))
        transformed = torch.asinh((y - q50) / scale)
    else:
        transformed = y
    return support_only_standardize(transformed, train_size)


def support_only_clip_and_scale(y: Tensor, train_size: int) -> Tensor:
    """Backward-compatible alias for the old experimental helper.

    New RW runs deliberately use the GT-aligned transform/standardization path
    below; callers retaining this name receive support-only StandardScaler
    semantics rather than the retired two-pass clipping experiment.
    """

    return support_only_standardize(y, train_size)


def _linear_template_interpolate_extrapolate(
    probabilities: Tensor,
    levels: Tensor,
    template: Tensor,
) -> Tensor:
    """Interpolate a monotone template and retain finite tail variation.

    The v4 mapper silently assumed a uniform quantile grid and clipped all
    values outside the support range to the two template endpoints.  That is
    especially harmful for regression because distinct query tail values then
    become identical loss targets.  This helper uses the stored quantile grid
    and linearly extrapolates beyond it with a conservative positive endpoint
    slope.
    """

    levels = levels.to(device=probabilities.device, dtype=probabilities.dtype)
    template = template.to(device=probabilities.device, dtype=probabilities.dtype)
    indices = torch.searchsorted(levels, probabilities, right=True)
    lo = indices.clamp(1, levels.numel() - 1) - 1
    hi = lo + 1
    level_width = (levels[hi] - levels[lo]).clamp_min(torch.finfo(levels.dtype).eps)
    weight = (probabilities - levels[lo]) / level_width
    mapped = template[lo] + weight * (template[hi] - template[lo])

    # Quantile templates may have repeated endpoint values (for rounded or
    # discrete real targets).  A tiny slope derived from the template's central
    # spread avoids reintroducing endpoint collapse without fitting query y.
    q25_index = int(
        torch.searchsorted(levels, torch.tensor(0.25, device=levels.device, dtype=levels.dtype)).clamp(
            0, levels.numel() - 1
        ).item()
    )
    q75_index = int(
        torch.searchsorted(levels, torch.tensor(0.75, device=levels.device, dtype=levels.dtype)).clamp(
            0, levels.numel() - 1
        ).item()
    )
    central_level_width = (levels[q75_index] - levels[q25_index]).clamp_min(
        torch.finfo(levels.dtype).eps
    )
    central_slope = (template[q75_index] - template[q25_index]).abs() / central_level_width
    minimum_tail_slope = torch.clamp(0.05 * central_slope, min=1.0e-4)
    lower_slope = torch.maximum(
        (template[1] - template[0]) / (levels[1] - levels[0]),
        minimum_tail_slope,
    )
    upper_slope = torch.maximum(
        (template[-1] - template[-2]) / (levels[-1] - levels[-2]),
        minimum_tail_slope,
    )
    mapped = torch.where(
        probabilities < levels[0],
        template[0] + (probabilities - levels[0]) * lower_slope,
        mapped,
    )
    mapped = torch.where(
        probabilities > levels[-1],
        template[-1] + (probabilities - levels[-1]) * upper_slope,
        mapped,
    )
    return mapped


def map_to_template_from_support(
    y: Tensor,
    train_size: int,
    levels: Tensor,
    template: Tensor,
) -> Tensor:
    """Map y through a support-only empirical CDF into an anonymous template.

    Interior probabilities use support mid-ranks.  Query values outside the
    support range receive bounded probability extrapolation based only on the
    support IQR/standard deviation.  Query targets affect only their own mapped
    loss target and never any fitted statistic.
    """

    _validate_target(y, train_size)
    support = torch.sort(y[:train_size, 0]).values.contiguous()
    values = y[:, 0].contiguous()
    left = torch.searchsorted(support, values, right=False)
    right = torch.searchsorted(support, values, right=True)
    probabilities = (
        left.to(dtype=values.dtype) + right.to(dtype=values.dtype)
    ) / (2.0 * float(train_size))

    q25, q75 = torch.quantile(
        support,
        torch.tensor((0.25, 0.75), device=support.device, dtype=support.dtype),
    )
    support_std = support.std(correction=0)
    robust_scale = torch.maximum((q75 - q25) / 1.349, support_std)
    robust_scale = torch.clamp(robust_scale, min=torch.finfo(support.dtype).eps)
    edge_probability = 0.5 / float(train_size)
    tail_probability_per_scale = 0.05
    lower = edge_probability + tail_probability_per_scale * (
        values - support[0]
    ) / robust_scale
    upper = 1.0 - edge_probability + tail_probability_per_scale * (
        values - support[-1]
    ) / robust_scale
    probabilities = torch.where(values < support[0], lower, probabilities)
    probabilities = torch.where(values > support[-1], upper, probabilities)

    # Keep extreme synthetic draws finite while allowing meaningful mass beyond
    # both template endpoints.  The bound is a numerical guard, not query-fit
    # preprocessing.
    probabilities = probabilities.clamp(-0.20, 1.20)
    mapped = _linear_template_interpolate_extrapolate(
        probabilities,
        levels,
        template,
    )
    if not torch.isfinite(mapped).all():
        raise NonFinitePriorError("tail-preserving template mapping produced non-finite targets")
    return mapped.view(-1, 1)


def process_rw_sample_regression_target(
    y: Tensor,
    train_size: int,
    profile_path: str,
    mix_probability: float,
) -> Tensor:
    """Apply the RW target prior under the GT pre/postprocessing contract.

    With ``mix_probability`` probability, a train-only empirical-CDF mapping
    transfers the model-space marginal shape of one anonymous GT training
    target.  The template already represents GT-transform + StandardScaler
    space, so it is only re-standardized on the sampled support.  Otherwise the
    synthetic target itself receives the exact support-only GT transform and
    StandardScaler path.  Query targets never fit preprocessing statistics.
    """

    if not 0.0 <= mix_probability <= 1.0:
        raise ValueError("regression target mix probability must be in [0, 1]")
    if not profile_path:
        raise ValueError("rw_sample50 regression target prior requires a profile path")
    levels, templates = load_anonymous_quantile_profile(profile_path)
    if torch.rand((), device=y.device).item() < mix_probability:
        index = int(torch.randint(templates.shape[0], (), device=y.device).item())
        template = templates[index].to(device=y.device, dtype=y.dtype)
        y = map_to_template_from_support(y, train_size, levels, template)
        return support_only_standardize(y, train_size)
    return support_only_gt_transform_and_standardize(y, train_size)
