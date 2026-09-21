from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn

from .layers import OneHotAndLinear


class CrossAttentionBlock(nn.Module):
    """Small pre-norm cross-attention block used by function tokens."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float = 0.0,
        bias_free_ln: bool = False,
    ) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(d_model, bias=not bias_free_ln)
        self.kv_norm = nn.LayerNorm(d_model, bias=not bias_free_ln)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(d_model, bias=not bias_free_ln)
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(
        self,
        query: Tensor,
        memory: Tensor,
        need_weights: bool = False,
    ) -> tuple[Tensor, Optional[Tensor]]:
        q = self.q_norm(query)
        kv = self.kv_norm(memory)
        attn_out, attn_weights = self.attn(
            q,
            kv,
            kv,
            need_weights=need_weights,
            average_attn_weights=True,
        )
        out = query + self.dropout(attn_out)
        out = out + self.dropout(self.ff(self.ff_norm(out)))
        return out, attn_weights


class FunctionTokenConditioner(nn.Module):
    """Apply support-conditioned dataset/query/latent function-token updates.

    The module keeps the row-representation sequence length unchanged. It builds
    support-side function context from labeled support rows, then uses query
    tokens to produce a residual update for query rows only.
    """

    def __init__(
        self,
        d_model: int,
        max_classes: int,
        dataset_count: int = 1,
        query_count: int = 1,
        latent_count: int = 8,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 1,
        scale: float = 0.1,
        dropout: float = 0.0,
        use_dataset_token: bool = True,
        use_query_token: bool = True,
        use_latent_tokens: bool = False,
        bias_free_ln: bool = False,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by function_token_num_heads ({num_heads})")
        if query_count <= 0:
            raise ValueError("function_token_query_count must be positive")
        if hidden_dim <= 0:
            raise ValueError("function_token_hidden_dim must be positive")
        if num_layers <= 0:
            raise ValueError("function_token_num_layers must be positive")
        if use_dataset_token and dataset_count <= 0:
            raise ValueError("function_token_dataset_count must be positive when dataset token is enabled")
        if use_latent_tokens and latent_count <= 0:
            raise ValueError("function_token_latent_count must be positive when latent tokens are enabled")

        self.d_model = int(d_model)
        self.max_classes = int(max_classes)
        self.dataset_count = int(dataset_count)
        self.query_count = int(query_count)
        self.latent_count = int(latent_count)
        self.scale = float(scale)
        self.use_dataset_token = bool(use_dataset_token)
        self.use_query_token = bool(use_query_token)
        self.use_latent_tokens = bool(use_latent_tokens)

        if self.max_classes > 0:
            self.y_encoder = OneHotAndLinear(self.max_classes, d_model)
        else:
            self.y_encoder = nn.Linear(1, d_model)

        self.support_norm = nn.LayerNorm(d_model, bias=not bias_free_ln)
        control_count = self.control_token_count
        if control_count > 0:
            self.dataset_tokens = (
                nn.Parameter(torch.empty(dataset_count, d_model)) if self.use_dataset_token else None
            )
            self.latent_tokens = nn.Parameter(torch.empty(latent_count, d_model)) if self.use_latent_tokens else None
            self.context_blocks = nn.ModuleList(
                [
                    CrossAttentionBlock(
                        d_model=d_model,
                        num_heads=num_heads,
                        hidden_dim=hidden_dim,
                        dropout=dropout,
                        bias_free_ln=bias_free_ln,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.control_norm = nn.LayerNorm(d_model, bias=not bias_free_ln)
        else:
            self.dataset_tokens = None
            self.latent_tokens = None
            self.context_blocks = nn.ModuleList()
            self.control_norm = nn.Identity()

        self.query_tokens = nn.Parameter(torch.empty(query_count, d_model)) if self.use_query_token else None
        self.query_input = nn.Linear(d_model, d_model)
        self.query_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    bias_free_ln=bias_free_ln,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(d_model, bias=not bias_free_ln)
        self.output = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.dropout = nn.Dropout(dropout)
        self.last_metrics: dict[str, Tensor] = {}
        self.reset_parameters()

    @property
    def active_dataset_count(self) -> int:
        return self.dataset_count if self.use_dataset_token else 0

    @property
    def active_latent_count(self) -> int:
        return self.latent_count if self.use_latent_tokens else 0

    @property
    def control_token_count(self) -> int:
        return self.active_dataset_count + self.active_latent_count

    def reset_parameters(self) -> None:
        if self.dataset_tokens is not None:
            nn.init.normal_(self.dataset_tokens, mean=0.0, std=0.02)
        if self.latent_tokens is not None:
            nn.init.normal_(self.latent_tokens, mean=0.0, std=0.02)
        if self.query_tokens is not None:
            nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)

    def _label_embedding(self, y_train: Tensor, dtype: torch.dtype) -> Tensor:
        if self.max_classes > 0:
            y = y_train.long().clamp(min=0, max=max(self.max_classes - 1, 0))
            encoded = self.y_encoder(y)
        else:
            encoded = self.y_encoder(y_train.unsqueeze(-1))
        return encoded.to(dtype=dtype)

    def _control_seed(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Optional[Tensor]:
        tokens = []
        if self.dataset_tokens is not None:
            tokens.append(self.dataset_tokens.to(device=device, dtype=dtype))
        if self.latent_tokens is not None:
            tokens.append(self.latent_tokens.to(device=device, dtype=dtype))
        if not tokens:
            return None
        seed = torch.cat(tokens, dim=0)
        return seed.unsqueeze(0).expand(batch_size, -1, -1)

    def compute_context(self, representations: Tensor, y_train: Tensor) -> Tensor:
        train_size = y_train.shape[1]
        if train_size <= 0:
            return representations.new_empty(representations.shape[0], 0, representations.shape[-1])

        compute_dtype = self.support_norm.weight.dtype
        support = representations[:, :train_size].to(dtype=compute_dtype)
        support = support + self._label_embedding(y_train, dtype=compute_dtype)
        support = self.support_norm(support)

        control = self._control_seed(support.shape[0], support.device, support.dtype)
        if control is None:
            return support

        for block in self.context_blocks:
            control, _ = block(control, support, need_weights=False)
        control = self.control_norm(control)
        return torch.cat([control, support], dim=1)

    def _empty_metrics(self, reference: Tensor) -> None:
        zero = reference.new_zeros(())
        self.last_metrics = {
            "function_token_support_attn_entropy": zero.detach(),
            "function_token_latent_usage_entropy": zero.detach(),
            "function_token_query_update_norm": zero.detach(),
            "function_token_update_scale": reference.new_tensor(self.scale).detach(),
        }

    def _update_metrics(self, attn_weights: Optional[Tensor], update: Tensor, memory_len: int) -> None:
        if attn_weights is None or attn_weights.numel() == 0:
            self._empty_metrics(update)
            return

        weights = attn_weights.detach().float()
        control_count = self.control_token_count
        support_start = min(control_count, memory_len)
        support_weights = weights[..., support_start:]
        if support_weights.numel() == 0:
            support_entropy = weights.new_zeros(())
        else:
            support_probs = support_weights / support_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            support_entropy = -(support_probs * support_probs.clamp_min(1e-12).log()).sum(dim=-1)
            support_len = support_weights.shape[-1]
            if support_len > 1:
                support_entropy = support_entropy / math.log(float(support_len))
            support_entropy = support_entropy.mean()

        latent_entropy = weights.new_zeros(())
        if self.use_latent_tokens and self.active_latent_count > 0:
            latent_start = self.active_dataset_count
            latent_end = min(latent_start + self.active_latent_count, memory_len)
            latent_weights = weights[..., latent_start:latent_end]
            if latent_weights.numel() > 0:
                latent_usage = latent_weights.sum(dim=tuple(range(latent_weights.ndim - 1)))
                latent_probs = latent_usage / latent_usage.sum().clamp_min(1e-12)
                latent_entropy = -(latent_probs * latent_probs.clamp_min(1e-12).log()).sum()
                if latent_weights.shape[-1] > 1:
                    latent_entropy = latent_entropy / math.log(float(latent_weights.shape[-1]))

        update_norm = update.detach().float().norm(dim=-1).mean() if update.numel() else weights.new_zeros(())
        self.last_metrics = {
            "function_token_support_attn_entropy": torch.nan_to_num(support_entropy).detach(),
            "function_token_latent_usage_entropy": torch.nan_to_num(latent_entropy).detach(),
            "function_token_query_update_norm": torch.nan_to_num(update_norm).detach(),
            "function_token_update_scale": update.new_tensor(self.scale).detach(),
        }

    def _query_seed(self, query_rows: Tensor) -> Tensor:
        B, Q, D = query_rows.shape
        base = self.query_input(query_rows).reshape(B * Q, 1, D)
        if self.query_tokens is None:
            return base
        tokens = self.query_tokens.to(device=query_rows.device, dtype=query_rows.dtype)
        return base + tokens.unsqueeze(0)

    def _apply_query_update(self, query_rows: Tensor, function_context: Tensor) -> Tensor:
        B, Q, D = query_rows.shape
        if Q == 0 or function_context.shape[1] == 0:
            self._empty_metrics(query_rows)
            return query_rows

        query = self._query_seed(query_rows)
        memory = function_context.unsqueeze(1).expand(B, Q, function_context.shape[1], D).reshape(
            B * Q,
            function_context.shape[1],
            D,
        )

        attn_weights = None
        for layer_idx, block in enumerate(self.query_blocks):
            query, attn_weights = block(query, memory, need_weights=layer_idx == len(self.query_blocks) - 1)

        pooled = query.mean(dim=1)
        update = self.output(self.output_norm(pooled)).reshape(B, Q, D)
        update = self.scale * self.dropout(update)
        self._update_metrics(attn_weights, update, memory_len=function_context.shape[1])
        return query_rows + update

    def forward(
        self,
        representations: Tensor,
        y_train: Optional[Tensor] = None,
        function_context: Optional[Tensor] = None,
        query_start: Optional[int] = None,
    ) -> Tensor:
        output_dtype = representations.dtype
        compute_dtype = self.output.weight.dtype
        internal_representations = representations.to(dtype=compute_dtype)
        if function_context is None:
            if y_train is None:
                raise ValueError("y_train is required when function_context is not provided")
            function_context = self.compute_context(internal_representations, y_train)
            query_start = y_train.shape[1] if query_start is None else query_start
        else:
            function_context = function_context.to(device=representations.device, dtype=compute_dtype)
            query_start = 0 if query_start is None else query_start

        if query_start < 0 or query_start > representations.shape[1]:
            raise ValueError(
                f"query_start must be in [0, {representations.shape[1]}], got {query_start}"
            )

        query_rows = internal_representations[:, query_start:]
        updated_query = self._apply_query_update(query_rows, function_context)
        if query_start == 0:
            return updated_query.to(dtype=output_dtype)
        output = torch.cat([internal_representations[:, :query_start], updated_query], dim=1)
        return output.to(dtype=output_dtype)
