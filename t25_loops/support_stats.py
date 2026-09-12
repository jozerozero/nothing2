from __future__ import annotations
import math
from typing import Optional
import torch
from torch import Tensor, nn

class SupportSchemaEncoder(nn.Module):
    """Encode support-set table statistics into a compact schema embedding."""

    stats_dim = 51

    def __init__(self, max_classes: int, hidden_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.max_classes = int(max_classes)
        self.net = nn.Sequential(
            nn.Linear(self.stats_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.GELU(),
        )

    @staticmethod
    def _feature_summary(values: Tensor, feature_mask: Tensor) -> Tensor:
        mask = feature_mask.to(dtype=torch.bool)
        denom = mask.sum(dim=-1).clamp_min(1).to(dtype=values.dtype)
        masked = values.masked_fill(~mask, 0.0)
        mean = masked.sum(dim=-1) / denom
        centered = (values - mean.unsqueeze(-1)).masked_fill(~mask, 0.0)
        std = (centered.square().sum(dim=-1) / denom).clamp_min(0.0).sqrt()
        min_value = values.masked_fill(~mask, torch.inf).min(dim=-1).values
        max_value = values.masked_fill(~mask, -torch.inf).max(dim=-1).values
        has_feature = mask.any(dim=-1)
        min_value = torch.where(has_feature, min_value, torch.zeros_like(min_value))
        max_value = torch.where(has_feature, max_value, torch.zeros_like(max_value))
        return torch.stack([mean, std, min_value, max_value], dim=-1)

    def _label_stats(self, y_train: Tensor, dtype: torch.dtype) -> Tensor:
        B, train_size = y_train.shape
        device = y_train.device
        if self.max_classes <= 0:
            return torch.zeros(B, 4, dtype=dtype, device=device)

        labels = y_train.long().clamp(min=0, max=max(self.max_classes - 1, 0))
        counts = torch.zeros(B, self.max_classes, dtype=dtype, device=device)
        counts.scatter_add_(1, labels, torch.ones(B, train_size, dtype=dtype, device=device))
        probs = counts / float(max(train_size, 1))
        present = counts > 0
        present_count = present.sum(dim=-1).to(dtype=dtype)
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
        if self.max_classes > 1:
            entropy = entropy / math.log(float(self.max_classes))
        majority = probs.max(dim=-1).values
        minority = probs.masked_fill(~present, torch.inf).min(dim=-1).values
        minority = torch.where(present.any(dim=-1), minority, torch.zeros_like(minority))
        return torch.stack(
            [
                present_count / float(max(self.max_classes, 1)),
                entropy,
                majority,
                minority,
            ],
            dim=-1,
        )

    def compute_stats(
        self,
        X: Tensor,
        y_train: Tensor,
        d: Optional[Tensor] = None,
        total_seq_len: Optional[int] = None,
    ) -> Tensor:
        B, T, H = X.shape
        train_size = y_train.shape[1]
        support = X[:, :train_size].float()
        dtype = support.dtype
        device = support.device
        if total_seq_len is None:
            total_seq_len = T

        if d is None:
            valid_features = torch.full((B,), H, dtype=torch.long, device=device)
        else:
            valid_features = d.to(device=device, dtype=torch.long).clamp(min=0, max=H)
        feature_idx = torch.arange(H, device=device).view(1, H)
        feature_mask = feature_idx < valid_features.view(B, 1)

        finite_mask = torch.isfinite(support)
        value_mask = finite_mask & feature_mask.view(B, 1, H)
        counts = value_mask.sum(dim=1).clamp_min(1).to(dtype=dtype)
        safe_support = support.masked_fill(~value_mask, 0.0)

        col_mean = safe_support.sum(dim=1) / counts
        col_abs_mean = support.abs().masked_fill(~value_mask, 0.0).sum(dim=1) / counts
        centered = (support - col_mean.unsqueeze(1)).masked_fill(~value_mask, 0.0)
        col_std = (centered.square().sum(dim=1) / counts).clamp_min(0.0).sqrt()
        col_min = support.masked_fill(~value_mask, torch.inf).min(dim=1).values
        col_max = support.masked_fill(~value_mask, -torch.inf).max(dim=1).values
        has_value = value_mask.any(dim=1)
        col_min = torch.where(has_value, col_min, torch.zeros_like(col_min))
        col_max = torch.where(has_value, col_max, torch.zeros_like(col_max))
        col_range = col_max - col_min
        col_zero_ratio = (support == 0).masked_fill(~value_mask, False).sum(dim=1).to(dtype=dtype) / counts
        col_near_zero_ratio = (
            (support.abs() < 1e-6).masked_fill(~value_mask, False).sum(dim=1).to(dtype=dtype) / counts
        )
        col_integer_like_ratio = (
            ((support - support.round()).abs() < 1e-4).masked_fill(~value_mask, False).sum(dim=1).to(dtype=dtype)
            / counts
        )
        near_zero_or_one = (support.abs() < 1e-4) | ((support - 1.0).abs() < 1e-4)
        col_binary_like_ratio = near_zero_or_one.masked_fill(~value_mask, False).sum(dim=1).to(dtype=dtype) / counts

        feature_summaries = torch.cat(
            [
                self._feature_summary(metric, feature_mask)
                for metric in (
                    col_mean,
                    col_std,
                    col_abs_mean,
                    col_min,
                    col_max,
                    col_range,
                    col_zero_ratio,
                    col_near_zero_ratio,
                    col_integer_like_ratio,
                    col_binary_like_ratio,
                )
            ],
            dim=-1,
        )

        valid_feature_float = valid_features.to(dtype=dtype)
        valid_values = (valid_feature_float * float(max(train_size, 1))).clamp_min(1.0)
        finite_ratio = value_mask.sum(dim=(1, 2)).to(dtype=dtype) / valid_values
        label_stats = self._label_stats(y_train, dtype=dtype)
        base_stats = torch.stack(
            [
                torch.log1p(valid_feature_float),
                torch.full((B,), math.log1p(float(train_size)), dtype=dtype, device=device),
                torch.full((B,), math.log1p(float(total_seq_len)), dtype=dtype, device=device),
                torch.full((B,), float(train_size) / float(max(total_seq_len, 1)), dtype=dtype, device=device),
                valid_feature_float / float(max(H, 1)),
                1.0 - (valid_feature_float / float(max(H, 1))),
                finite_ratio,
            ],
            dim=-1,
        )
        stats = torch.cat([base_stats, label_stats, feature_summaries], dim=-1)
        return torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)

    def forward(
        self,
        X: Tensor,
        y_train: Tensor,
        d: Optional[Tensor] = None,
        total_seq_len: Optional[int] = None,
    ) -> Tensor:
        stats = self.compute_stats(X, y_train, d=d, total_seq_len=total_seq_len)
        weight = self.net[0].weight
        return self.net(stats.to(dtype=weight.dtype))


class SupportSchemaStatistics(SupportSchemaEncoder):
    """Parameter-free support-only dataset statistics for conditioned gates."""

    def __init__(self, max_classes: int) -> None:
        nn.Module.__init__(self)
        self.max_classes = int(max_classes)

    def forward(
        self,
        X: Tensor,
        y_train: Tensor,
        d: Optional[Tensor] = None,
        total_seq_len: Optional[int] = None,
    ) -> Tensor:
        return self.compute_stats(X, y_train, d=d, total_seq_len=total_seq_len)


