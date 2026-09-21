from __future__ import annotations

from typing import Optional, Union
from functools import partial

import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint

from .rope import RotaryEmbedding
from .layers import MultiheadAttentionBlock, InducedSelfAttentionBlock
from .kv_cache import KVCacheEntry, KVCache
from .attention_gate import DatasetConditionedAttentionGate


class Encoder(nn.Module):
    """Stack of multihead attention blocks.

    Parameters
    ----------
    num_blocks : int
        Number of multihead attention blocks in the stack.

    d_model : int
        Model dimension.

    nhead : int
        Number of attention heads and should be a divisor of ``d_model``.

    dim_feedforward : int
        Dimension of the feedforward network in each block.

    dropout : float, default=0.0
        Dropout probability.

    activation : str or unary callable, default="gelu"
        The activation function used in the feedforward network, can be
        either string ("relu" or "gelu") or unary callable.

    norm_first : bool, default=True
        If True, uses pre-norm architecture (LayerNorm before attention and feedforward).

    bias_free_ln : bool, default=False
        If True, removes bias from all LayerNorm layers.

    use_rope : bool, default=False
        Whether to use rotary positional encoding.

    rope_base : int, default=100000
        A base scaling factor for rotary position encoding.

    rope_interleaved : bool, default=True
        If True, uses interleaved rotation where dimension pairs are (0,1), (2,3), etc.
        If False, uses non-interleaved rotation where the embedding is split into
        first half [0:d//2] and second half [d//2:d].

    ssmax : bool or str, default=False
        Type of scalable softmax to use.
        If True, equivalent to "qassmax-mlp-elementwise".
        If False, equivalent to "none".
        If a string, uses the specified scalable softmax type.
        Options include:

        - "none": No scaling applied.
        - "ssmax": :math:`q_{\\text{scaled}} = q \\cdot (s \\cdot \\log n)` where
          :math:`s` is a learnable per-head parameter.
        - "ssmax-mlp": Uses MLP to compute scaling factors based on sequence length.
        - "ssmax-mlp-elementwise": Elementwise scaling per head dimension using MLP.
        - "qassmax-mlp": Query-aware scaling:
          :math:`\\text{scale} = \\text{base\\_mlp}(\\log n) \\cdot (1 + \\tanh(\\text{query\\_mlp}(q)))`.
        - "qassmax-mlp-elementwise": Elementwise query-aware scaling.

    recompute : bool, default=False
        If True, uses gradient checkpointing to save memory at the cost of
        additional computation.
    """

    def __init__(
        self,
        num_blocks: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
        use_rope: bool = False,
        rope_base: int = 100000,
        rope_interleaved: bool = True,
        ssmax: Union[bool, str] = False,
        recompute: bool = False,
        attention_gate_enabled: bool = False,
        attention_gate_shape: str = "scalar",
        attention_gate_layers: Optional[list[int]] = None,
        attention_gate_context_dim: int = 192,
        attention_gate_rho: float = 0.25,
        swiglu_enabled: bool = False,
        swiglu_conditioned: bool = False,
        swiglu_context_dim: int = 192,
        swiglu_rho: float = 0.25,
        swiglu_output_scale: float = 1.0,
        swiglu_product_tanh_rms_multiple: float = 0.0,
        swiglu_product_tanh_last_n_layers: int = 0,
        swiglu_init_seed_base: int = 2026082601,
        reference_dim_feedforward: Optional[int] = None,
        qk_pds_enabled: bool = False,
        shared_depth_enabled: bool = False,
        shared_depth_rho: float = 1.0,
        shared_depth_dataset_conditioned: bool = False,
        shared_depth_num_passes: int = 2,
        shared_depth_context_dim: int = 51,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        if not 0 <= int(swiglu_product_tanh_last_n_layers) <= num_blocks:
            raise ValueError(
                "swiglu_product_tanh_last_n_layers must be within "
                f"[0, {num_blocks}], got {swiglu_product_tanh_last_n_layers}"
            )
        if (float(swiglu_product_tanh_rms_multiple) > 0.0) != (
            int(swiglu_product_tanh_last_n_layers) > 0
        ):
            raise ValueError(
                "SwiGLU product smoothing multiple and last-N depth must be enabled together"
            )

        self.blocks = nn.ModuleList(
            [
                MultiheadAttentionBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    norm_first=norm_first,
                    bias_free_ln=bias_free_ln,
                    ssmax=ssmax,
                    swiglu_enabled=swiglu_enabled,
                    swiglu_conditioned=swiglu_conditioned,
                    swiglu_context_dim=swiglu_context_dim,
                    swiglu_rho=swiglu_rho,
                    swiglu_output_scale=swiglu_output_scale,
                    swiglu_product_tanh_rms_multiple=(
                        swiglu_product_tanh_rms_multiple
                        if layer_index >= num_blocks - swiglu_product_tanh_last_n_layers
                        else 0.0
                    ),
                    swiglu_init_seed=int(swiglu_init_seed_base) + layer_index,
                    reference_dim_feedforward=reference_dim_feedforward,
                    qk_pds_enabled=qk_pds_enabled,
                )
                for layer_index in range(num_blocks)
            ]
        )

        self.rope = (
            RotaryEmbedding(dim=d_model // nhead, theta=rope_base, interleaved=rope_interleaved) if use_rope else None
        )
        self.recompute = recompute
        self.shared_depth_enabled = bool(shared_depth_enabled)
        self.shared_depth_rho = float(shared_depth_rho)
        self.shared_depth_dataset_conditioned = bool(shared_depth_dataset_conditioned)
        self.shared_depth_num_passes = int(shared_depth_num_passes)
        if self.shared_depth_enabled and self.shared_depth_num_passes < 2:
            raise ValueError("shared_depth_num_passes must be >= 2 when shared depth is enabled")
        if not torch.isfinite(torch.tensor(self.shared_depth_rho)) or self.shared_depth_rho <= 0.0:
            raise ValueError("shared_depth_rho must be finite and positive")
        if self.shared_depth_dataset_conditioned and not self.shared_depth_enabled:
            raise ValueError("shared_depth_dataset_conditioned requires shared_depth_enabled")
        if self.shared_depth_dataset_conditioned and self.shared_depth_rho != 1.0:
            raise ValueError("dataset-conditioned shared depth requires shared_depth_rho=1.0")
        if self.shared_depth_enabled:
            # G5 reuses every existing ICL block. A zero scalar makes every
            # pass after the first an exact identity-gated no-op at
            # initialization and consumes no RNG, preserving every
            # pre-existing E4 tensor.
            self.shared_depth_gate = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("shared_depth_gate", None)
        if self.shared_depth_dataset_conditioned:
            # This is w in alpha_D. Zero initialization consumes no RNG and
            # makes the conditioned model exactly equal to G5 at step zero.
            self.shared_depth_condition_weight = nn.Parameter(
                torch.zeros(int(shared_depth_context_dim))
            )
        else:
            self.register_parameter("shared_depth_condition_weight", None)
        self.attention_gate_enabled = bool(attention_gate_enabled)
        selected_layers = set(attention_gate_layers or [])
        if any(layer < 0 or layer >= num_blocks for layer in selected_layers):
            raise ValueError(f"attention gate layers {sorted(selected_layers)} outside [0, {num_blocks})")
        # Keep every pre-existing E4 parameter bitwise identical under the same
        # global seed: new gate initialization must not advance the caller RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2026082203)
            self.attention_gates = nn.ModuleDict(
                {
                    str(layer): DatasetConditionedAttentionGate(
                        d_model=d_model,
                        nhead=nhead,
                        context_dim=attention_gate_context_dim,
                        shape=attention_gate_shape,
                        rho=attention_gate_rho,
                        bias_free_ln=bias_free_ln,
                    )
                    for layer in sorted(selected_layers)
                }
                if self.attention_gate_enabled
                else {}
            )

    def _shared_depth_alpha(
        self,
        dataset_context: Optional[Tensor],
        reference: Tensor,
    ) -> Tensor:
        """Return G5 alpha, optionally conditioned by support-only c_D."""
        if not self.shared_depth_dataset_conditioned:
            return self.shared_depth_rho * torch.tanh(self.shared_depth_gate)
        if dataset_context is None:
            raise ValueError("dataset_context is required for dataset-conditioned shared depth")
        weight = self.shared_depth_condition_weight
        context = dataset_context.to(device=weight.device, dtype=weight.dtype)
        if context.ndim != 2 or context.shape[-1] != weight.numel():
            raise ValueError(
                "shared-depth dataset context must be [B,C] with "
                f"C={weight.numel()}, got {tuple(context.shape)}"
            )
        support_logit = (context * weight).sum(dim=-1)
        bounded_support = 0.1 * (2.0 * torch.sigmoid(support_logit) - 1.0)
        return torch.tanh(self.shared_depth_gate + bounded_support).view(-1, 1, 1).to(
            dtype=reference.dtype
        )

    def forward(
        self,
        src: Tensor,
        train_size: Optional[int] = None,
        return_intermediate_layers: Optional[set[int]] = None,
        dataset_context: Optional[Tensor] = None,
    ) -> Tensor | tuple[Tensor, dict[int, Tensor]]:
        """Process input through the stacked blocks.

        Parameters
        ----------
        src : Tensor
            Input tensor of shape (..., seq_len, d_model).

        train_size : Optional[int], default=None
            Positive integer indicating the number of training samples.
            When provided, queries attend only to the first ``train_size``
            positions. Useful in the ICL transformer where only training
            samples serve as context.

        Returns
        -------
        Tensor
            Output tensor with same shape as ``src``.
        """
        out = src
        collect_intermediates = return_intermediate_layers is not None
        intermediate_layers = set(return_intermediate_layers or ())
        intermediates: dict[int, Tensor] = {}
        for layer_idx, block in enumerate(self.blocks):
            gate_module = self.attention_gates[str(layer_idx)] if str(layer_idx) in self.attention_gates else None
            if gate_module is not None and dataset_context is None:
                raise ValueError("dataset_context is required when dataset-conditioned attention gates are enabled")
            sdpa_gate = gate_module(out, dataset_context) if gate_module is not None else None
            if self.recompute:
                kwargs = {
                    "train_size": train_size,
                    "rope": self.rope,
                    "sdpa_gate": sdpa_gate,
                    "ffn_context": dataset_context,
                }
                out = checkpoint(partial(block, **kwargs), out, use_reentrant=False)
            else:
                out = block(
                    q=out,
                    train_size=train_size,
                    rope=self.rope,
                    sdpa_gate=sdpa_gate,
                    ffn_context=dataset_context,
                )
            if collect_intermediates and layer_idx in intermediate_layers:
                intermediates[layer_idx] = out

        if self.shared_depth_enabled:
            for _pass_idx in range(1, self.shared_depth_num_passes):
                previous = out
                for layer_idx, block in enumerate(self.blocks):
                    gate_module = (
                        self.attention_gates[str(layer_idx)]
                        if str(layer_idx) in self.attention_gates
                        else None
                    )
                    if gate_module is not None and dataset_context is None:
                        raise ValueError(
                            "dataset_context is required when dataset-conditioned attention gates are enabled"
                        )
                    sdpa_gate = gate_module(out, dataset_context) if gate_module is not None else None
                    if self.recompute:
                        kwargs = {
                            "train_size": train_size,
                            "rope": self.rope,
                            "sdpa_gate": sdpa_gate,
                            "ffn_context": dataset_context,
                        }
                        out = checkpoint(partial(block, **kwargs), out, use_reentrant=False)
                    else:
                        out = block(
                            q=out,
                            train_size=train_size,
                            rope=self.rope,
                            sdpa_gate=sdpa_gate,
                            ffn_context=dataset_context,
                        )
                gate = self._shared_depth_alpha(dataset_context, out)
                out = previous + gate * (out - previous)

        if collect_intermediates:
            return out, intermediates
        return out

    def forward_with_cache(
        self,
        src: Tensor,
        icl_cache: KVCache,
        train_size: Optional[int] = None,
        use_cache: bool = False,
        store_cache: bool = True,
        dataset_context: Optional[Tensor] = None,
    ) -> Tensor:
        """Process input through the stacked blocks with KV caching support.

        1. If ``store_cache=True``, this method processes the full sequence and
           stores K/V projections from training data (positions
           ``[0:train_size]``) at each layer.

        2. If ``use_cache=True``, this method assumes ``src`` only contains test
           data and uses cached K/V from training data for attention at each layer.

        Parameters
        ----------
        src : Tensor
            Input tensor of shape (..., seq_len, d_model).

        icl_cache : KVCache
            Cache object for storing/retrieving K/V projections per layer.

        train_size : Optional[int], default=None
            Positive integer indicating the number of training samples.
            When provided, queries attend only to the first ``train_size``
            positions.

        use_cache : bool, default=False
            Whether to use cached values to avoid redundant computation.

        store_cache : bool, default=True
            Whether to store computed values in cache.

        Returns
        -------
        Tensor
            Output tensor with same shape as ``src``.
        """

        if use_cache == store_cache:
            raise ValueError("Exactly one of use_cache or store_cache must be True")

        if store_cache and train_size is None:
            raise ValueError("train_size must be provided when store_cache=True")

        out = src
        num_passes = self.shared_depth_num_passes if self.shared_depth_enabled else 1
        for pass_idx in range(num_passes):
            previous = out if pass_idx > 0 else None
            for layer_idx, block in enumerate(self.blocks):
                cache_idx = pass_idx * len(self.blocks) + layer_idx
                gate_module = (
                    self.attention_gates[str(layer_idx)]
                    if str(layer_idx) in self.attention_gates
                    else None
                )
                if gate_module is not None and dataset_context is None:
                    raise ValueError(
                        "dataset_context is required when dataset-conditioned attention gates are enabled"
                    )
                sdpa_gate = gate_module(out, dataset_context) if gate_module is not None else None
                if use_cache:
                    out = block(
                        q=out,
                        rope=self.rope,
                        cached_kv=icl_cache.kv[cache_idx],
                        sdpa_gate=sdpa_gate,
                        ffn_context=dataset_context,
                    )
                else:
                    out, k_proj, v_proj = block(
                        q=out,
                        train_size=train_size,
                        rope=self.rope,
                        sdpa_gate=sdpa_gate,
                        ffn_context=dataset_context,
                        need_kv=True,
                    )
                    icl_cache.kv[cache_idx] = KVCacheEntry(key=k_proj, value=v_proj)
            if pass_idx > 0:
                gate = self._shared_depth_alpha(dataset_context, out)
                out = previous + gate * (out - previous)
        return out


class SetTransformer(nn.Module):
    """Stack of induced self-attention blocks.

    A set transformer uses induced self-attention mechanism to efficiently
    process variable-sized sets while maintaining permutation invariance.

    Parameters
    ----------
    num_blocks : int
        Number of induced self-attention blocks in the stack.

    d_model : int
        Model dimension.

    nhead : int
        Number of attention heads and should be a divisor of ``d_model``.

    dim_feedforward : int
        Dimension of the feedforward network in each block.

    num_inds : int, default=16
        Number of inducing points used in self-attention blocks.

    dropout : float, default=0.0
        Dropout probability.

    activation : str or unary callable, default="gelu"
        The activation function used in the feedforward network, can be
        either string ("relu" or "gelu") or unary callable.

    norm_first : bool, default=True
        If True, uses pre-norm architecture (LayerNorm before attention and feedforward).

    bias_free_ln : bool, default=False
        If True, removes bias from all LayerNorm layers.

    ssmax : bool or str, default=False
        Type of scalable softmax to use in attention. Note that only the first
        attention layer of the induced self-attention blocks uses SSMax.
        If True, equivalent to "qassmax-mlp-elementwise".
        If False, equivalent to "none".
        If a string, uses the specified scalable softmax type.
        Options include:

        - "none": No scaling applied.
        - "ssmax": :math:`q_{\\text{scaled}} = q \\cdot (s \\cdot \\log n)` where
          :math:`s` is a learnable per-head parameter.
        - "ssmax-mlp": Uses MLP to compute scaling factors based on sequence length.
        - "ssmax-mlp-elementwise": Elementwise scaling per head dimension using MLP.
        - "qassmax-mlp": Query-aware scaling:
          :math:`\\text{scale} = \\text{base\\_mlp}(\\log n) \\cdot (1 + \\tanh(\\text{query\\_mlp}(q)))`.
        - "qassmax-mlp-elementwise": Elementwise query-aware scaling.

    recompute : bool, default=False
        If True, uses gradient checkpointing to save memory at the cost of
        additional computation.

    References
    ----------
    .. [1] Lee et al. "Set Transformer: A Framework for Attention-based
           Permutation-Invariant Neural Networks", ICML 2019
    """

    def __init__(
        self,
        num_blocks: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_inds: int = 16,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
        ssmax: Union[bool, str] = False,
        recompute: bool = False,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.blocks = nn.ModuleList(
            [
                InducedSelfAttentionBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    num_inds=num_inds,
                    dropout=dropout,
                    activation=activation,
                    norm_first=norm_first,
                    bias_free_ln=bias_free_ln,
                    ssmax=ssmax,
                )
                for _ in range(num_blocks)
            ]
        )
        self.recompute = recompute

    def forward(self, src: Tensor, train_size: Optional[int] = None) -> Tensor:
        """Process input through the stacked blocks.

        Parameters
        ----------
        src : Tensor
            Input tensor of shape (..., seq_len, d_model).

        train_size : Optional[int], default=None
            Position to split the input into training and test data. When provided,
            inducing points will only attend to training data in the first attention
            stage of induced self-attention blocks to prevent information leakage.

        Returns
        -------
        Tensor
            Output tensor with same shape as ``src``.
        """
        out = src
        for block in self.blocks:
            if self.recompute:
                out = checkpoint(partial(block, train_size=train_size), out, use_reentrant=False)
            else:
                out = block(out, train_size)

        return out

    def forward_with_cache(
        self,
        src: Tensor,
        col_cache: KVCache,
        train_size: Optional[int] = None,
        use_cache: bool = False,
        store_cache: bool = True,
    ) -> Tensor:
        """Process input through the stacked ISAB blocks with KV caching support.

        Each block has two attention stages:

        1. Stage 1: Inducing points attend to training data, producing ``hidden``.
        2. Stage 2: Input attends to ``hidden``, producing the output.

        We cache the K/V projections of ``hidden`` for Stage 2, which allows test
        samples to reuse the cached K/V without recomputing Stage 1.

        If ``store_cache=True``, this method:

        - Runs Stage 1: inducing points attend to training data to produce ``hidden``.
        - Caches K/V projections of ``hidden`` for each block.
        - Runs Stage 2: all samples attend to ``hidden``.

        If ``use_cache=True``, this method:

        - Skips Stage 1 (uses cached K/V from ``hidden``).
        - Runs Stage 2: test samples attend to cached K/V.

        Parameters
        ----------
        src : Tensor
            Input tensor of shape (..., seq_len, d_model).

        col_cache : KVCache
            Cache object for storing/retrieving K/V projections of ``hidden``.

        train_size : Optional[int], default=None
            Position to split the input into training and test data. If storing
            cache, it must be provided to ensure the cache is populated with
            training data correctly. If using cache, it is ignored.

        use_cache : bool, default=False
            Whether to use cached values to avoid redundant computation.

        store_cache : bool, default=True
            Whether to store computed values in cache.

        Returns
        -------
        Tensor
            Output tensor with same shape as ``src``.
        """

        if use_cache == store_cache:
            raise ValueError("Exactly one of use_cache or store_cache must be True")

        if store_cache and train_size is None:
            raise ValueError("train_size must be provided when store_cache=True")

        out = src
        for block_idx, block in enumerate(self.blocks):
            out = block.forward_with_cache(out, col_cache, block_idx, train_size, use_cache, store_cache)

        return out
