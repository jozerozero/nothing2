from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn


class SupportGroupRFF(nn.Module):
    """Fixed RFF summary computed strictly from the labelled support set."""

    input_dim = 24
    output_dim = 64
    seed = 20260817

    def __init__(self) -> None:
        super().__init__()
        rng = np.random.default_rng(self.seed)
        weight = rng.normal(0.0, 1.0, size=(self.output_dim // 2, self.input_dim)).astype(np.float32)
        phase = rng.uniform(0.0, 2.0 * math.pi, size=(self.output_dim // 2,)).astype(np.float32)
        self.register_buffer("weight", torch.from_numpy(weight), persistent=True)
        self.register_buffer("phase", torch.from_numpy(phase), persistent=True)

    @staticmethod
    def _summary(values: Tensor, mask: Tensor) -> Tensor:
        masked = values.masked_fill(~mask, 0.0)
        count = mask.sum(dim=-1).clamp_min(1).to(values.dtype)
        mean = masked.sum(dim=-1) / count
        centered = (values - mean.unsqueeze(-1)).masked_fill(~mask, 0.0)
        std = (centered.square().sum(dim=-1) / count).clamp_min(0.0).sqrt()
        minimum = values.masked_fill(~mask, torch.inf).amin(dim=-1)
        maximum = values.masked_fill(~mask, -torch.inf).amax(dim=-1)
        valid = mask.any(dim=-1)
        minimum = torch.where(valid, minimum, torch.zeros_like(minimum))
        maximum = torch.where(valid, maximum, torch.zeros_like(maximum))
        return torch.stack((mean, std, minimum, maximum), dim=-1)

    def _raw(self, X: Tensor, y_train: Tensor, d: Tensor | None) -> Tensor:
        batch, _, features = X.shape
        train_size = y_train.shape[1]
        support = X[:, :train_size].float()
        finite = torch.isfinite(support)
        if d is None:
            valid_features = torch.full((batch,), features, device=X.device, dtype=torch.long)
        else:
            valid_features = d.to(device=X.device, dtype=torch.long).clamp(0, features)
        feature_mask = torch.arange(features, device=X.device)[None, :] < valid_features[:, None]
        value_mask = finite & feature_mask[:, None, :]
        count = value_mask.sum(dim=1).clamp_min(1).to(support.dtype)
        safe = support.masked_fill(~value_mask, 0.0)
        means = safe.sum(dim=1) / count
        centered_values = (support - means[:, None, :]).masked_fill(~value_mask, 0.0)
        stds = (centered_values.square().sum(dim=1) / count).clamp_min(1.0e-12).sqrt()
        z = centered_values / stds[:, None, :]
        skew = (z.pow(3).sum(dim=1) / count).clamp(-8.0, 8.0)
        kurt = (z.pow(4).sum(dim=1) / count - 3.0).clamp(-8.0, 8.0)

        # Quantiles are computed per table and only over valid support values.
        iqrs, medians = [], []
        for batch_idx in range(batch):
            active = int(valid_features[batch_idx].item())
            values = support[batch_idx, :, :active]
            if active == 0:
                iqrs.append(support.new_zeros((0,)))
                medians.append(support.new_zeros((0,)))
                continue
            values = torch.nan_to_num(values, nan=0.0, posinf=8.0, neginf=-8.0)
            q10 = torch.quantile(values, 0.10, dim=0)
            q50 = torch.quantile(values, 0.50, dim=0)
            q90 = torch.quantile(values, 0.90, dim=0)
            iqrs.append(q90 - q10)
            medians.append(q50)

        raw_rows = []
        for batch_idx in range(batch):
            mask = feature_mask[batch_idx : batch_idx + 1]
            parts = [
                self._summary(means[batch_idx : batch_idx + 1], mask),
                self._summary(stds[batch_idx : batch_idx + 1], mask),
                self._summary(skew[batch_idx : batch_idx + 1], mask),
                self._summary(kurt[batch_idx : batch_idx + 1], mask),
            ]
            iqr = iqrs[batch_idx]
            median = medians[batch_idx]
            if iqr.numel():
                iqr_mean = iqr.mean()
                iqr_std = iqr.std(unbiased=False)
                median_mean = median.mean()
            else:
                iqr_mean = iqr_std = median_mean = support.new_zeros(())
            active = max(int(valid_features[batch_idx].item()), 1)
            zero_ratio = (support[batch_idx, :, :active].abs() < 1.0e-8).float().mean()
            labels, class_counts = torch.unique(y_train[batch_idx].long(), return_counts=True)
            probabilities = class_counts.float() / class_counts.sum().clamp_min(1)
            entropy = -(probabilities * probabilities.clamp_min(1.0e-8).log()).sum()
            entropy = entropy / math.log(max(2, int(labels.numel())))
            tail = torch.stack(
                (
                    iqr_mean,
                    iqr_std,
                    median_mean,
                    zero_ratio,
                    valid_features[batch_idx].float() / 100.0,
                    support.new_tensor(float(train_size) / float(max(1, X.shape[1]))),
                    support.new_tensor(float(labels.numel()) / 20.0),
                    entropy,
                )
            )[None, :]
            raw_rows.append(torch.cat(parts + [tail], dim=-1).squeeze(0))
        raw = torch.stack(raw_rows, dim=0)
        if raw.shape[-1] != self.input_dim:
            raise RuntimeError(f"expected {self.input_dim} support statistics, got {raw.shape[-1]}")
        return torch.nan_to_num(raw, nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0)

    def forward(self, X: Tensor, y_train: Tensor, d: Tensor | None = None) -> Tensor:
        raw = self._raw(X, y_train, d)
        weight = self.weight.to(device=raw.device, dtype=raw.dtype)
        phase = self.phase.to(device=raw.device, dtype=raw.dtype)
        angle = raw @ weight.t() + phase
        return math.sqrt(1.0 / float(self.output_dim // 2)) * torch.cat((angle.cos(), angle.sin()), dim=-1)


class DatasetConditionedAttentionGate(nn.Module):
    """Query- and dataset-conditioned multiplicative gate on SDPA head outputs."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        context_dim: int,
        shape: str,
        rho: float,
        bias_free_ln: bool = False,
    ) -> None:
        super().__init__()
        if shape not in {"scalar", "channel"}:
            raise ValueError("attention gate shape must be 'scalar' or 'channel'")
        if not 0.0 < rho <= 1.0:
            raise ValueError("attention gate rho must be in (0, 1]")
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.head_dim = self.d_model // self.nhead
        self.shape = shape
        self.rho = float(rho)
        output_dim = self.nhead if shape == "scalar" else self.d_model
        self.query_norm = nn.LayerNorm(self.d_model, bias=not bias_free_ln)
        self.context_norm = nn.LayerNorm(context_dim, bias=not bias_free_ln)
        self.query_projection = nn.Linear(self.d_model, output_dim, bias=False)
        self.context_projection = nn.Linear(context_dim, output_dim, bias=True)
        nn.init.zeros_(self.query_projection.weight)
        nn.init.zeros_(self.context_projection.weight)
        nn.init.zeros_(self.context_projection.bias)

    def forward(self, query: Tensor, dataset_context: Tensor) -> Tensor:
        if query.ndim != 3 or dataset_context.ndim != 2 or query.shape[0] != dataset_context.shape[0]:
            raise ValueError(
                "attention gate requires query [B,T,D] and dataset_context [B,C], "
                f"got {tuple(query.shape)} and {tuple(dataset_context.shape)}"
            )
        query_logits = self.query_projection(self.query_norm(query))
        context_logits = self.context_projection(self.context_norm(dataset_context))[:, None, :]
        logits = query_logits + context_logits
        if self.shape == "scalar":
            logits = logits.view(query.shape[0], query.shape[1], self.nhead, 1)
        else:
            logits = logits.view(query.shape[0], query.shape[1], self.nhead, self.head_dim)
        return 1.0 + self.rho * torch.tanh(logits).permute(0, 2, 1, 3)
