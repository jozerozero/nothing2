from __future__ import annotations
from typing import Optional, Tuple, Union
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .ssmax import create_ssmax_layer
from .rope import RotaryEmbedding
from .attention import multi_head_attention_forward
from .kv_cache import KVCacheEntry, KVCache


class ClassNode:
    """Node in the hierarchical classification tree for handling many-class problems.

    Attributes
    ----------
    depth : int
        Current depth level in the hierarchical tree.

    is_leaf : bool
        Whether this node handles a small enough subset of classes directly.

    classes_ : Tensor
        List of unique class indices this node is responsible for.

    child_nodes : list
        Child nodes for non-leaf nodes, each handling a subset of classes.

    class_mapping : dict
        Maps original class indices to group indices for internal nodes.

    group_indices : Tensor
        Transformed labels after mapping original classes to their group indices.

    R : Tensor
        Feature data associated with this node.

    y : Tensor
        Target labels associated with this node.
    """

    def __init__(self, depth=0):
        self.depth = depth
        self.is_leaf = False
        self.classes_ = None
        self.child_nodes = []
        self.class_mapping = {}
        self.group_indices = None
        self.R = None
        self.y = None


class OneHotAndLinear(nn.Linear):
    """Combines one-hot encoding and linear projection in a single efficient operation
    to convert categorical indices to embeddings.

    Parameters
    ----------
    num_classes : int
        Number of distinct categories for one-hot encoding.

    embed_dim : int
        Output embedding dimension.
    """

    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__(num_classes, embed_dim)
        self.num_classes = num_classes
        self.embed_dim = embed_dim

    def forward(self, src: Tensor) -> Tensor:
        """Transform integer indices to dense embeddings.

        Parameters
        ----------
        src : Tensor
            Integer tensor of shape (batch_size, sequence_length) containing
            category indices.

        Returns
        -------
        Tensor
            Embedded representation of shape (batch_size, sequence_length, embed_dim).
        """
        out = F.embedding(src.long(), self.weight.t())
        if self.bias is not None:
            out = out + self.bias
        return out


class SkippableLinear(nn.Linear):
    """Linear layer that handles inputs where all values equal ``skip_value``.

    First applies the linear transformation to all inputs, then replaces outputs
    for inputs where all values equal ``skip_value`` with the ``skip_value``.

    Parameters
    ----------
    in_features : int
        Size of each input sample.

    out_features : int
        Size of each output sample.

    bias : bool, default=True
        If set to False, the layer will not learn an additive bias.

    skip_value : float, default=-100.0
        Value used to mark inputs that should be skipped.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, skip_value: float = -100.0):
        super().__init__(in_features, out_features, bias)
        self.skip_value = skip_value

    def forward(self, src: Tensor) -> Tensor:
        """Forward pass that handles inputs flagged with ``skip_value``.

        Parameters
        ----------
        src : Tensor
            Input tensor of shape (..., in_features).

        Returns
        -------
        Tensor
            Output tensor of shape (..., out_features) where rows corresponding
            to skipped inputs are filled with ``skip_value``.
        """

        out = F.linear(src, self.weight, self.bias)
        skip_mask = (src == self.skip_value).all(dim=-1)
        if skip_mask.any():
            out[skip_mask] = self.skip_value

        return out


class MultiheadAttention(nn.MultiheadAttention):
    """Enhanced multi-head attention with RoPE, scalable softmax, and KV caching.

    Parameters
    ----------
    embed_dim : int
        Model dimension (total size of each attention head combined).

    num_heads : int
        Number of attention heads.

    dropout : float, default=0.0
        Dropout probability applied to attention weights.

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

    Notes
    -----
    The implementation always uses ``batch_first=True``, so input tensors have
    shape (..., seq_len, embed_dim).

    References
    ----------
    .. [1] Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
           https://arxiv.org/abs/2104.09864

    .. [2] Liu et al., "Scalable-Softmax Is Superior for Attention"
           https://arxiv.org/abs/2501.19399
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        ssmax: Union[bool, str] = False,
        qk_pds_enabled: bool = False,
    ):
        super().__init__(embed_dim, num_heads, dropout, batch_first=True)
        if isinstance(ssmax, bool):
            ssmax = "qassmax-mlp-elementwise" if ssmax else "none"
        self.ssmax_layer = create_ssmax_layer(ssmax_type=ssmax, num_heads=num_heads, embed_dim=embed_dim)
        self.qk_pds_enabled = bool(qk_pds_enabled)
        if self.qk_pds_enabled:
            # Construct SSMax first to preserve the reference RNG stream, then
            # remove it: G3 is a replacement, not a combination with QASSMax.
            self.ssmax_layer = None
            self.qk_pds_log_scale = nn.Parameter(torch.zeros(num_heads, embed_dim // num_heads))
        else:
            self.register_parameter("qk_pds_log_scale", None)

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        cached_kv: Optional[KVCacheEntry] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor | int] = None,
        rope: Optional[RotaryEmbedding] = None,
        sdpa_gate: Optional[Tensor] = None,
        need_kv: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        """Compute multi-head attention.

        Parameters
        ----------
        query : Tensor
            Query tensor of shape (..., tgt_len, embed_dim).

        key : Optional[Tensor], default=None
            Key tensor of shape (..., src_len, embed_dim).
            Required when ``cached_kv`` is None.

        value : Optional[Tensor], default=None
            Value tensor of shape (..., src_len, embed_dim).
            Required when ``cached_kv`` is None.

        cached_kv : Optional[KVCacheEntry], default=None
            Pre-computed key and value projections for caching. When provided,
            key and value parameters are ignored.

        key_padding_mask : Optional[Tensor], default=None
            Mask of shape (..., src_len) that identifies padding elements
            in the key sequence to be ignored:

            - For binary masks: True values indicate positions to ignore.
            - For float masks: Values are directly added to attention scores.

        attn_mask : Optional[Tensor], default=None
            Attention mask of shape (tgt_len, src_len) or
            (..., num_heads, tgt_len, src_len).

        rope : Optional[RotaryEmbedding]
            Rotary positional encoding.

        need_kv : bool, default=False
            If True and ``cached_kv`` is None, also returns the computed K and V
            projections along with the attention output. Useful for caching K/V
            for subsequent calls.

        Returns
        -------
        Union[Tensor, Tuple[Tensor, Tensor, Tensor]]
            If ``need_kv`` is False or ``cached_kv`` is provided:
                Attention output of shape (..., tgt_len, embed_dim).
            If ``need_kv`` is True and ``cached_kv`` is None:
                Tuple of (attn_output, k, v) where:

                - attn_output: shape (..., tgt_len, embed_dim)
                - k: shape (..., num_heads, src_len, head_dim)
                - v: shape (..., num_heads, src_len, head_dim)
        """

        if key_padding_mask is not None:
            key_padding_mask = F._canonical_mask(
                mask=key_padding_mask,
                mask_name="key_padding_mask",
                other_type=F._none_or_dtype(attn_mask),
                other_name="src_mask",
                target_type=query.dtype,
            )

        if attn_mask is not None:
            attn_mask = F._canonical_mask(
                mask=attn_mask,
                mask_name="attn_mask",
                other_type=None,
                other_name="",
                target_type=query.dtype,
                check_other=False,
            )

        return multi_head_attention_forward(
            query,
            self.num_heads,
            self.in_proj_weight,
            self.in_proj_bias,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            key=key,
            value=value,
            cached_kv=cached_kv,
            training=self.training,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            rope=rope,
            ssmax_layer=self.ssmax_layer,
            qk_pds_log_scale=self.qk_pds_log_scale,
            sdpa_gate=sdpa_gate,
            need_kv=need_kv,
        )


class MultiheadAttentionBlock(nn.TransformerEncoderLayer):
    """Attention block supporting RoPE, scalable softmax, and KV caching.

    Parameters
    ----------
    d_model : int
        Model dimension.

    nhead : int
        Number of attention heads.

    dim_feedforward : int
        Dimension of the feedforward network.

    dropout : float, default=0.0
        Dropout probability.

    activation : str or unary callable, default="gelu"
        The activation function used in the feedforward network, can be
        either string ("relu", "gelu") or unary callable.

    norm_first : bool, default=True
        If True, uses pre-norm architecture (LayerNorm before attention and feedforward).

    bias_free_ln : bool, default=False
        If True, removes bias from all LayerNorm layers.

    ssmax : bool or str, default=False
        Type of scalable softmax to use in attention.
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
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
        ssmax: Union[bool, str] = False,
        swiglu_enabled: bool = False,
        swiglu_conditioned: bool = False,
        swiglu_context_dim: int = 192,
        swiglu_rho: float = 0.25,
        swiglu_output_scale: float = 1.0,
        swiglu_product_tanh_rms_multiple: float = 0.0,
        swiglu_init_seed: int = 2026082601,
        reference_dim_feedforward: Optional[int] = None,
        qk_pds_enabled: bool = False,
    ):
        # Preserve the exact RNG consumption of the E4 GELU reference even
        # when a parameter-matched SwiGLU uses a narrower hidden dimension.
        # The reference width is supplied explicitly by TabICL so changing
        # ff_factor cannot silently break paired RNG consumption.
        # We intentionally construct that reference-sized superclass first;
        # architecture-specific replacements are constructed in a forked RNG
        # stream below and therefore cannot perturb later shared parameters or
        # the DataLoader seed sampled after model construction.
        reference_dim_feedforward = (
            int(reference_dim_feedforward)
            if swiglu_enabled and reference_dim_feedforward is not None
            else dim_feedforward
        )
        if reference_dim_feedforward <= 0:
            raise ValueError("reference_dim_feedforward must be positive")
        super().__init__(
            d_model,
            nhead,
            reference_dim_feedforward,
            dropout,
            activation=activation,
            norm_first=norm_first,
            batch_first=True,
        )
        if bias_free_ln:
            self.norm1 = nn.LayerNorm(d_model, bias=False)
            self.norm2 = nn.LayerNorm(d_model, bias=False)

        del self.self_attn
        self.attn = MultiheadAttention(
            d_model, nhead, dropout, ssmax, qk_pds_enabled=qk_pds_enabled
        )
        self.swiglu_enabled = bool(swiglu_enabled)
        self.swiglu_conditioned = bool(swiglu_conditioned)
        self.swiglu_rho = float(swiglu_rho)
        self.swiglu_output_scale = float(swiglu_output_scale)
        self.swiglu_product_tanh_rms_multiple = float(swiglu_product_tanh_rms_multiple)
        self.swiglu_init_seed = int(swiglu_init_seed)
        if self.swiglu_conditioned and not self.swiglu_enabled:
            raise ValueError("swiglu_conditioned requires swiglu_enabled")
        if not (0.0 <= self.swiglu_rho <= 1.0):
            raise ValueError("swiglu_rho must be in [0, 1]")
        if not math.isfinite(self.swiglu_output_scale) or self.swiglu_output_scale <= 0.0:
            raise ValueError("swiglu_output_scale must be finite and positive")
        if (
            not math.isfinite(self.swiglu_product_tanh_rms_multiple)
            or self.swiglu_product_tanh_rms_multiple < 0.0
        ):
            raise ValueError("swiglu_product_tanh_rms_multiple must be finite and non-negative")
        if self.swiglu_product_tanh_rms_multiple > 0.0 and not self.swiglu_enabled:
            raise ValueError("SwiGLU product smoothing requires swiglu_enabled")
        if self.swiglu_enabled:
            # All SwiGLU-only allocations live in a forked RNG stream.  For a
            # narrower, parameter-matched arm, replace the reference-sized
            # gate/down projections here as well.  fork_rng restores the
            # caller RNG on exit while still giving different blocks distinct
            # weights because each block enters from a different RNG state.
            isolated = torch.Generator(device="cpu")
            isolated.manual_seed(self.swiglu_init_seed)
            with torch.random.fork_rng(devices=[]):
                torch.random.set_rng_state(isolated.get_state())
                if dim_feedforward != reference_dim_feedforward:
                    self.linear1 = nn.Linear(d_model, dim_feedforward)
                    self.linear2 = nn.Linear(dim_feedforward, d_model)
                self.swiglu_value_proj = nn.Linear(d_model, dim_feedforward)
            if self.swiglu_conditioned:
                # The zero-initialized conditional branch must not perturb RNG
                # state for the shared D-SG1-U/D-SG1-C parameters.
                conditioner_rng = torch.Generator(device="cpu")
                conditioner_rng.manual_seed(2026082401)
                with torch.random.fork_rng(devices=[]):
                    torch.random.set_rng_state(conditioner_rng.get_state())
                    self.swiglu_conditioner = nn.Linear(swiglu_context_dim, 2 * dim_feedforward)
                nn.init.zeros_(self.swiglu_conditioner.weight)
                nn.init.zeros_(self.swiglu_conditioner.bias)
            else:
                self.swiglu_conditioner = None
        self.init_weights()

    def init_weights(self):
        """Initialize projection layers to zero for stable training."""
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(
        self,
        q: Tensor,
        k: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
        cached_kv: Optional[KVCacheEntry] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        train_size: Optional[int] = None,
        rope: Optional[RotaryEmbedding] = None,
        sdpa_gate: Optional[Tensor] = None,
        ffn_context: Optional[Tensor] = None,
        need_kv: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        """Process input through attention.

        Parameters
        ----------
        q : Tensor
            Query tensor of shape (..., tgt_len, d_model).

        k : Optional[Tensor], default=None
            Key tensor of shape (..., src_len, d_model).
            If None, uses ``q`` for self-attention.

        v : Optional[Tensor], default=None
            Value tensor of shape (..., src_len, d_model).
            If None, uses ``q`` for self-attention.

        cached_kv : Optional[KVCacheEntry], default=None
            Pre-computed K/V projections for caching. When provided,
            ``k`` and ``v`` parameters are ignored.

        key_padding_mask : Optional[Tensor], default=None
            Mask of shape (..., src_len) that identifies padding elements
            in the key sequence to be ignored:

            - For binary masks: True values indicate positions to ignore.
            - For float masks: Values are directly added to attention scores.

        attn_mask : Optional[Tensor], default=None
            Attention mask of shape (tgt_len, src_len) or
            (..., num_heads, tgt_len, src_len).

        train_size : Optional[int], default=None
            When provided (requires k=None and v=None for self-attention), the full
            sequence is used as query while only the first ``train_size`` positions
            serve as key/value. Useful in the ICL transformer where only training
            samples provide context.

        rope : Optional[RotaryEmbedding]
            Rotary positional encoding.

        need_kv : bool, default=False
            If True, also returns the computed K and V projections along with the
            output. Useful for caching K/V for subsequent calls.

        Returns
        -------
        Union[Tensor, Tuple[Tensor, Tensor, Tensor]]
            If ``need_kv`` is False:
                Output tensor of shape (..., tgt_len, d_model).
            If ``need_kv`` is True:
                Tuple of (output, k, v) where:

                - output: shape (..., tgt_len, d_model)
                - k: shape (..., num_heads, src_len, head_dim)
                - v: shape (..., num_heads, src_len, head_dim)
        """

        if train_size is None:
            k = q if k is None else k
            v = q if v is None else v
        else:
            assert k is None and v is None, "k and v must be None when train_size is provided"
            k = v = q[..., :train_size, :]

        k_proj, v_proj = None, None
        use_cache = cached_kv is not None

        if self.norm_first:
            # Pre-norm: normalize first, then apply attention
            q_normed = self.norm1(q)
            if use_cache:
                attn = self._attn_block(
                    q_normed, cached_kv=cached_kv, key_padding_mask=key_padding_mask, attn_mask=attn_mask,
                    rope=rope, sdpa_gate=sdpa_gate
                )
            else:
                if train_size is None:
                    k_normed = self.norm1(k) if k is not q else q_normed
                    v_normed = self.norm1(v) if v is not k else k_normed
                else:
                    k_normed = v_normed = q_normed[..., :train_size, :]

                attn_result = self._attn_block(
                    q_normed,
                    k_normed,
                    v_normed,
                    key_padding_mask=key_padding_mask,
                    attn_mask=attn_mask,
                    rope=rope,
                    sdpa_gate=sdpa_gate,
                    need_kv=need_kv,
                )

                if need_kv and isinstance(attn_result, tuple):
                    attn, k_proj, v_proj = attn_result
                else:
                    attn = attn_result

            x = q + attn
            x = x + self._ff_block(self.norm2(x), ffn_context=ffn_context)
        else:
            # Post-norm: attention first, then normalize
            if use_cache:
                attn = self._attn_block(
                    q, cached_kv=cached_kv, key_padding_mask=key_padding_mask, attn_mask=attn_mask,
                    rope=rope, sdpa_gate=sdpa_gate
                )
            else:
                attn_result = self._attn_block(
                    q, k, v, key_padding_mask=key_padding_mask, attn_mask=attn_mask, rope=rope,
                    sdpa_gate=sdpa_gate, need_kv=need_kv
                )

                if need_kv and isinstance(attn_result, tuple):
                    attn, k_proj, v_proj = attn_result
                else:
                    attn = attn_result

            x = self.norm1(q + attn)
            x = self.norm2(x + self._ff_block(x, ffn_context=ffn_context))

        if need_kv and k_proj is not None:
            return x, k_proj, v_proj

        return x

    def _attn_block(
        self,
        q: Tensor,
        k: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
        cached_kv: Optional[KVCacheEntry] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        rope: Optional[RotaryEmbedding] = None,
        sdpa_gate: Optional[Tensor] = None,
        need_kv: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        result = self.attn(
            q,
            k,
            v,
            cached_kv=cached_kv,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            rope=rope,
            sdpa_gate=sdpa_gate,
            need_kv=need_kv,
        )
        if need_kv and isinstance(result, tuple):
            attn, k_proj, v_proj = result
            return self.dropout1(attn), k_proj, v_proj
        return self.dropout1(result)

    def _ff_block(self, x: Tensor, ffn_context: Optional[Tensor] = None) -> Tensor:
        if not self.swiglu_enabled:
            x = self.linear2(self.dropout(self.activation(self.linear1(x))))
            return self.dropout2(x)

        gate = torch.nn.functional.silu(self.linear1(x))
        if self.swiglu_conditioned:
            if ffn_context is None:
                raise ValueError("support-only ffn_context is required for conditioned SwiGLU")
            modulation = self.swiglu_conditioner(ffn_context)
            gamma_raw, beta_raw = modulation.chunk(2, dim=-1)
            gamma = 1.0 + self.swiglu_rho * torch.tanh(gamma_raw)
            beta = self.swiglu_rho * torch.tanh(beta_raw)
            while gamma.ndim < gate.ndim:
                gamma = gamma.unsqueeze(-2)
                beta = beta.unsqueeze(-2)
            gate = gamma * gate + beta
        product = gate * self.swiglu_value_proj(x)
        if self.swiglu_product_tanh_rms_multiple > 0.0:
            # Smooth-SwiGLU, localized by the Encoder to selected deep blocks.
            # The detached per-token cap prevents the model from weakening the
            # regularizer merely by increasing its own product RMS.  The
            # disabled branch is absent rather than algebraically equivalent,
            # preserving the historical raw path bitwise.
            product_fp32 = product.float()
            token_rms = product_fp32.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-12)
            cap = self.swiglu_product_tanh_rms_multiple * token_rms.detach()
            product = (cap * torch.tanh(product_fp32 / cap)).to(product.dtype)
        x = self.linear2(self.dropout(product * self.swiglu_output_scale))
        return self.dropout2(x)


class InducedSelfAttentionBlock(nn.Module):
    """Induced Self-Attention for efficient :math:`O(n)` attention on large sets.

    This module implements a bottleneck attention mechanism using a small set of
    learned inducing points that mediate interactions between input elements.
    The complexity is reduced from :math:`O(n^2)` to :math:`O(n)` by:

    1. Projecting inputs onto inducing points (size :math:`m \\ll n`)
    2. Propagating information through these inducing points
    3. Projecting back to the original sequence

    Parameters
    ----------
    d_model : int
        Model dimension.

    nhead : int
        Number of attention heads.

    dim_feedforward : int
        Dimension of the feedforward network.

    num_inds : int
        Number of inducing points (controls capacity vs. efficiency).

    dropout : float, default=0.0
        Dropout probability.

    activation : str or unary callable, default="gelu"
        The activation function used in the feedforward network, can be
        either string ("relu" or "gelu") or unary callable.

    norm_first : bool, default=True
        If True, uses pre-norm architecture (LayerNorm before attention and feedforward).

    bias_free_ln : bool, default=False
        If True, removes bias from all LayerNorm layers.

    skip_value : float, default=-100.0
        Value used to mark inputs that should be skipped.

    ssmax : bool or str, default=False
        Type of scalable softmax to use in attention. Note that only the first
        attention layer uses SSMax.
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

    References
    ----------
    .. [1] Lee et al. "Set Transformer: A Framework for Attention-based
           Permutation-Invariant Neural Networks", ICML 2019
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_inds: int,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
        ssmax: Union[bool, str] = False,
        skip_value: float = -100.0,
    ):
        super().__init__()
        self.skip_value = skip_value

        if isinstance(ssmax, bool):
            ssmax = "qassmax-mlp-elementwise" if ssmax else "none"

        # Two-stage attention mechanism
        self.multihead_attn1 = MultiheadAttentionBlock(
            d_model, nhead, dim_feedforward, dropout, activation, norm_first, bias_free_ln, ssmax
        )
        self.multihead_attn2 = MultiheadAttentionBlock(
            d_model, nhead, dim_feedforward, dropout, activation, norm_first, bias_free_ln
        )

        # Learnable inducing points
        self.num_inds = num_inds
        self.ind_vectors = nn.Parameter(torch.empty(num_inds, d_model))
        nn.init.trunc_normal_(self.ind_vectors, std=0.02)

    def induced_attention(self, src: Tensor, train_size: Optional[int] = None) -> Tensor:
        """Apply induced self-attention to input sequence.

        Parameters
        ----------
        src : Tensor
            Input tensor of shape (..., seq_len, d_model).

        train_size : Optional[int], default=None
            Position to split the input into training and test data.

        Returns
        -------
        Tensor
            Output tensor with same shape as input.
        """

        *batch_shape, _, d_model = src.shape
        ind_vectors = self.ind_vectors.expand(*batch_shape, self.num_inds, d_model)

        if train_size is None:
            hidden = self.multihead_attn1(ind_vectors, src, src)
        else:
            hidden = self.multihead_attn1(ind_vectors, src[..., :train_size, :], src[..., :train_size, :])

        out = self.multihead_attn2(src, hidden, hidden)

        return out

    def forward(self, src: Tensor, train_size: Optional[int] = None) -> Tensor:
        """Apply induced self-attention to input sequence.

        Parameters
        ----------
        src : Tensor
            Input tensor of shape (..., seq_len, d_model).

        train_size : Optional[int], default=None
            Position to split the input into training and test data. When provided,
            inducing points will only attend to training data in the first attention
            stage to prevent information leakage from test data during evaluation.

        Returns
        -------
        Tensor
            Output tensor with same shape as input.
        """

        skip_mask = (src == self.skip_value).all(dim=(-2, -1))  # batch shape
        if skip_mask.any():
            if skip_mask.all():
                out = torch.full_like(src, self.skip_value)
            else:
                out = torch.empty_like(src)
                out[~skip_mask] = self.induced_attention(src[~skip_mask], train_size)
                out[skip_mask] = self.skip_value
        else:
            out = self.induced_attention(src, train_size)

        return out

    def induced_attention_with_cache(
        self,
        src: Tensor,
        col_cache: KVCache,
        block_idx: int,
        train_size: Optional[int] = None,
        use_cache: bool = False,
        store_cache: bool = True,
    ) -> Tensor:
        """Apply induced self-attention with optional caching.

        Parameters
        ----------
        src : Tensor
            Input tensor of shape (..., seq_len, d_model).

        col_cache : KVCache
            Cache object for storing/retrieving K/V of the second attention layer.

        block_idx : int
            Index of this block for cache key.

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
            Output tensor with same shape as input.
        """

        *batch_shape, _, d_model = src.shape
        ind_vectors = self.ind_vectors.expand(*batch_shape, self.num_inds, d_model)

        if use_cache:
            assert block_idx in col_cache.kv, f"Cache miss for kv at ISAB {block_idx}"
            out = self.multihead_attn2(src, cached_kv=col_cache.kv[block_idx])

        if store_cache:
            assert train_size is not None, "train_size must be provided when store_cache=True"
            hidden = self.multihead_attn1(ind_vectors, src[..., :train_size, :], src[..., :train_size, :])
            out, k_proj, v_proj = self.multihead_attn2(src, hidden, hidden, need_kv=True)
            col_cache.kv[block_idx] = KVCacheEntry(key=k_proj, value=v_proj)

        return out

    def forward_with_cache(
        self,
        src: Tensor,
        col_cache: KVCache,
        block_idx: int,
        train_size: Optional[int] = None,
        use_cache: bool = False,
        store_cache: bool = True,
    ) -> Tensor:
        """Apply induced self-attention with optional caching, handling skip values.

        Parameters
        ----------
        src : Tensor
            Input tensor of shape (..., seq_len, d_model).

        col_cache : KVCache
            Cache object for storing/retrieving hidden tensors and K/V.

        block_idx : int
            Index of this block for cache key.

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
            Output tensor with same shape as input.
        """

        if use_cache == store_cache:
            raise ValueError("Exactly one of use_cache or store_cache must be True")

        if store_cache and train_size is None:
            raise ValueError("train_size must be provided when store_cache=True")

        # When using cache, we need consistent batch dimensions, so we don't apply skip_mask
        # The cache was populated with the full batch shape
        skip_mask = (src == self.skip_value).all(dim=(-2, -1))
        if skip_mask.all():
            return torch.full_like(src, self.skip_value)
        else:
            out = self.induced_attention_with_cache(src, col_cache, block_idx, train_size, use_cache, store_cache)
            # Restore skip values in output
            if skip_mask.any():
                out[skip_mask] = self.skip_value
            return out
