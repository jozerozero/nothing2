from __future__ import annotations

import math

import torch
from torch import nn


def _logn(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Compute :math:`\\log(n)` safely, avoiding fp16 overflow for large ``n``."""
    return torch.tensor(math.log(max(n, 1)), device=device, dtype=dtype)


class SSMax(nn.Module):
    """Scalable Softmax with learnable per-head scaling factors.

    Applies scaling to queries:
    :math:`q_{\\text{scaled}} = q \\cdot (s \\cdot \\log n)`,
    where :math:`s` is a learnable per-head parameter.

    Parameters
    ----------
    num_heads : int
        Number of attention heads.
    """

    def __init__(self, num_heads: int):
        super().__init__()
        self.scales = nn.Parameter(torch.ones(num_heads))

    def forward(self, q: torch.Tensor, n: int) -> torch.Tensor:
        """Apply SSMax scaling to queries.

        Parameters
        ----------
        q : torch.Tensor
            Query tensor after projection, shape ``(bs, n_heads, seq_len, head_dim)``.

        n : int
            Source sequence length.

        Returns
        -------
        torch.Tensor
            Scaled query tensor, same shape as ``q``.
        """
        logn = _logn(n, q.device, q.dtype)
        scales = self.scales.view(1, -1, 1, 1) * logn
        return q * scales


class SSMaxMLP(nn.Module):
    """Scalable Softmax using an MLP to compute scaling factors.

    Applies scaling to queries:
    :math:`q_{\\text{scaled}} = q \\cdot \\text{mlp}(\\log n)`,
    where the MLP learns to map sequence length to scaling factors.

    Parameters
    ----------
    num_heads : int
        Number of attention heads.

    n_hidden : int, default=64
        Number of hidden units in the MLP.

    elementwise : bool, default=False
        If True, apply elementwise scaling per head dimension, allowing
        different scaling for each element in the head dimension.

    head_dim : int, optional
        Dimension of each attention head. Required if ``elementwise=True``.
    """

    def __init__(
        self,
        num_heads: int,
        n_hidden: int = 64,
        elementwise: bool = False,
        head_dim: int = None,
    ):
        super().__init__()
        self.elementwise = elementwise
        if elementwise:
            if head_dim is None:
                raise ValueError("head_dim must be provided when elementwise=True")
            out_dim = num_heads * head_dim
        else:
            out_dim = num_heads
        self.mlp = nn.Sequential(nn.Linear(1, n_hidden), nn.GELU(), nn.Linear(n_hidden, out_dim))
        self.num_heads = num_heads

    def forward(self, q: torch.Tensor, n: int) -> torch.Tensor:
        """Apply SSMax scaling to queries.

        Parameters
        ----------
        q : torch.Tensor
            Query tensor after projection, shape ``(bs, n_heads, seq_len, head_dim)``.

        n : int
            Source sequence length.

        Returns
        -------
        torch.Tensor
            Scaled query tensor, same shape as ``q``.
        """
        logn = _logn(n, q.device, q.dtype).reshape(1, 1)
        scales = self.mlp(logn)
        if self.elementwise:
            # scales: (1, num_heads * head_dim) -> (1, num_heads, 1, head_dim)
            head_dim = q.shape[-1]
            scales = scales.view(1, self.num_heads, 1, head_dim)
        else:
            scales = scales.view(1, self.num_heads, 1, 1)
        return q * scales


class QASSMaxMLP(nn.Module):
    """Query-Aware Scalable Softmax using MLPs to compute scaling factors.

    Applies scaling to queries:

    .. math::

        q_{\\text{scaled}} = q \\cdot \\text{base\\_mlp}(\\log n)
        \\cdot \\bigl(1 + \\tanh(\\text{query\\_mlp}(q))\\bigr)

    where the base MLP learns length-dependent scaling and the query MLP
    learns query-dependent modulation.

    Parameters
    ----------
    num_heads : int
        Number of attention heads.

    head_dim : int
        Dimension of each attention head.

    n_hidden : int, default=64
        Number of hidden units in the MLPs.

    elementwise : bool, default=False
        If True, apply elementwise scaling per head dimension, allowing
        different scaling for each element in the head dimension.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        n_hidden: int = 64,
        elementwise: bool = False,
        max_abs_base_scale: float = 16.0,
        max_abs_scale: float = 16.0,
        max_abs_query_logit: float = 8.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.elementwise = elementwise
        self.max_abs_base_scale = max_abs_base_scale
        self.max_abs_scale = max_abs_scale
        self.max_abs_query_logit = max_abs_query_logit

        if elementwise:
            base_out_dim = num_heads * head_dim
            query_out_dim = head_dim
        else:
            base_out_dim = num_heads
            query_out_dim = 1

        self.base_mlp = nn.Sequential(nn.Linear(1, n_hidden), nn.GELU(), nn.Linear(n_hidden, base_out_dim))
        self.query_mlp = nn.Sequential(nn.Linear(head_dim, n_hidden), nn.GELU(), nn.Linear(n_hidden, query_out_dim))

        # ensures initial modulation is zero
        nn.init.zeros_(self.query_mlp[-1].weight)
        nn.init.zeros_(self.query_mlp[-1].bias)

    def forward(self, q: torch.Tensor, n: int) -> torch.Tensor:
        """Apply QASSMax scaling to queries.

        Parameters
        ----------
        q : torch.Tensor
            Query tensor after projection, shape ``(bs, n_heads, seq_len, head_dim)``.

        n : int
            Source sequence length.

        Returns
        -------
        torch.Tensor
            Scaled query tensor, same shape as ``q``.
        """
        if q.dtype == torch.float32:
            return self._forward_float(q, n)

        q_dtype = q.dtype
        with torch.amp.autocast(device_type=q.device.type, enabled=False):
            return self._forward_float(q.float(), n).to(q_dtype)

    def _forward_float(self, q_float: torch.Tensor, n: int) -> torch.Tensor:
        logn = _logn(n, q_float.device, torch.float32).reshape(1, 1)

        if self.elementwise:
            # base_scales: (1, num_heads * head_dim) -> (1, num_heads, 1, head_dim)
            base_scales = self.base_mlp(logn).view(1, self.num_heads, 1, self.head_dim)
        else:
            base_scales = self.base_mlp(logn).view(1, self.num_heads, 1, 1)

        base_scales = torch.nan_to_num(
            base_scales,
            nan=0.0,
            posinf=self.max_abs_base_scale,
            neginf=-self.max_abs_base_scale,
        ).clamp(-self.max_abs_base_scale, self.max_abs_base_scale)

        query_logits = self.query_mlp(q_float)
        query_logits = torch.nan_to_num(
            query_logits,
            nan=0.0,
            posinf=self.max_abs_query_logit,
            neginf=-self.max_abs_query_logit,
        ).clamp(-self.max_abs_query_logit, self.max_abs_query_logit)
        modulation = 1 + torch.tanh(query_logits)

        scales = torch.nan_to_num(
            base_scales * modulation,
            nan=0.0,
            posinf=self.max_abs_scale,
            neginf=-self.max_abs_scale,
        ).clamp(-self.max_abs_scale, self.max_abs_scale)

        return q_float * scales


def create_ssmax_layer(ssmax_type: str, num_heads: int, embed_dim: int):
    """Factory function to create SSMax layer based on type."""

    if ssmax_type == "none":
        return None
    elif ssmax_type == "ssmax":
        return SSMax(num_heads)
    elif ssmax_type == "ssmax-mlp":
        return SSMaxMLP(num_heads)
    elif ssmax_type == "ssmax-mlp-elementwise":
        return SSMaxMLP(num_heads, head_dim=embed_dim // num_heads, elementwise=True)
    elif ssmax_type == "qassmax-mlp":
        return QASSMaxMLP(num_heads, embed_dim // num_heads)
    elif ssmax_type == "qassmax-mlp-elementwise":
        return QASSMaxMLP(num_heads, embed_dim // num_heads, elementwise=True)
    else:
        raise ValueError(f"Unknown {ssmax_type=}")
