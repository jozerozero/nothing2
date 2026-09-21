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


class AdapterExpert(nn.Module):
    def __init__(self, d_model: int, bottleneck: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, d_model),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SchemaExpertConditioner(nn.Module):
    """Apply schema-conditioned routed adapters and optional FiLM to row representations."""

    def __init__(
        self,
        d_model: int,
        max_classes: int,
        num_experts: int = 32,
        top_k: int = 2,
        bottleneck: int = 32,
        hidden_dim: int = 128,
        adapter_scale: float = 0.1,
        film_enabled: bool = False,
        film_scale: float = 0.1,
        dropout: float = 0.0,
        adapter_enabled: bool = True,
        router_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.d_model = int(d_model)
        self.num_experts = int(num_experts)
        self.top_k = min(int(top_k), self.num_experts)
        self.adapter_scale = float(adapter_scale)
        self.film_scale = float(film_scale)
        self.adapter_enabled = bool(adapter_enabled)
        self.film_enabled = bool(film_enabled)
        self.router_temperature = max(float(router_temperature), 1e-6)

        self.schema_encoder = SupportSchemaEncoder(max_classes=max_classes, hidden_dim=hidden_dim, embedding_dim=hidden_dim)
        if self.adapter_enabled:
            self.router = nn.Linear(hidden_dim, self.num_experts)
            self.experts = nn.ModuleList(
                [AdapterExpert(d_model=d_model, bottleneck=bottleneck, dropout=dropout) for _ in range(self.num_experts)]
            )
        else:
            self.router = None
            self.experts = nn.ModuleList()
        if self.film_enabled:
            self.film = nn.Linear(hidden_dim, d_model * 2)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        else:
            self.film = None
        self.last_metrics: dict[str, Tensor] = {}

    def compute_context(
        self,
        X: Tensor,
        y_train: Tensor,
        d: Optional[Tensor] = None,
        total_seq_len: Optional[int] = None,
    ) -> Tensor:
        return self.schema_encoder(X, y_train, d=d, total_seq_len=total_seq_len)

    def _zero_touch_experts(self, reference: Tensor) -> Tensor:
        if not self.adapter_enabled:
            return reference.new_zeros(())
        touched = reference.new_zeros(())
        for expert in self.experts:
            for param in expert.parameters():
                touched = touched + param.sum() * 0.0
        return touched

    def _apply_adapter(self, representations: Tensor, schema_context: Tensor) -> Tensor:
        assert self.router is not None
        logits = self.router(schema_context)
        probs = torch.softmax(logits / self.router_temperature, dim=-1)
        top_values, top_indices = torch.topk(logits, k=self.top_k, dim=-1)
        top_weights = torch.softmax(top_values / self.router_temperature, dim=-1)

        adapter_update = torch.zeros_like(representations)
        for expert_idx, expert in enumerate(self.experts):
            selected = top_indices == expert_idx
            batch_selected = selected.any(dim=-1)
            if bool(batch_selected.any().detach().item()):
                selected_weights = torch.where(
                    selected[batch_selected],
                    top_weights[batch_selected],
                    torch.zeros_like(top_weights[batch_selected]),
                ).sum(dim=-1)
                expert_dtype = next(expert.parameters()).dtype
                expert_in = representations[batch_selected].to(dtype=expert_dtype)
                expert_out = expert(expert_in).to(dtype=representations.dtype)
                adapter_update[batch_selected] = adapter_update[batch_selected] + (
                    selected_weights.view(-1, 1, 1).to(dtype=representations.dtype) * expert_out
                )

        usage = torch.zeros(self.num_experts, dtype=representations.dtype, device=representations.device)
        usage.scatter_add_(0, top_indices.reshape(-1), torch.ones_like(top_indices, dtype=representations.dtype).reshape(-1))
        usage = usage / float(max(top_indices.numel(), 1))
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
        entropy_norm = entropy / math.log(float(self.num_experts)) if self.num_experts > 1 else entropy
        self.last_metrics = {
            "schema_router_entropy": entropy_norm.mean().detach(),
            "schema_router_active_experts": (usage > 0).sum().to(dtype=representations.dtype).detach(),
            "schema_router_max_usage": usage.max().detach(),
        }

        if self.training:
            adapter_update = adapter_update + self._zero_touch_experts(representations)
        return representations + self.adapter_scale * adapter_update

    def forward(
        self,
        representations: Tensor,
        X: Optional[Tensor] = None,
        y_train: Optional[Tensor] = None,
        d: Optional[Tensor] = None,
        schema_context: Optional[Tensor] = None,
        total_seq_len: Optional[int] = None,
    ) -> Tensor:
        if schema_context is None:
            if X is None or y_train is None:
                raise ValueError("X and y_train are required when schema_context is not provided")
            schema_context = self.compute_context(X, y_train, d=d, total_seq_len=total_seq_len)
        else:
            first_weight = next(self.schema_encoder.parameters())
            schema_context = schema_context.to(device=representations.device, dtype=first_weight.dtype)

        out = representations
        if self.adapter_enabled:
            out = self._apply_adapter(out, schema_context)
        else:
            self.last_metrics = {}
        if self.film is not None:
            gamma, beta = self.film(schema_context).chunk(2, dim=-1)
            gamma = torch.tanh(gamma).to(dtype=out.dtype).unsqueeze(1)
            beta = beta.to(dtype=out.dtype).unsqueeze(1)
            out = out * (1.0 + self.film_scale * gamma) + self.film_scale * beta
        return out
