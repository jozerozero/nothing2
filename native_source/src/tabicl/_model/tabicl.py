from __future__ import annotations
from typing import Optional, List, Union, Literal, Sequence
import math
import os

import torch
from torch import nn, Tensor

from .embedding import ColEmbedding
from .interaction import RowInteraction
from .learning import ICLearning
from .schema_expert import SchemaExpertConditioner, SupportSchemaEncoder, SupportSchemaStatistics
from .attention_gate import SupportGroupRFF
from .function_tokens import FunctionTokenConditioner
from .quantile_dist import QuantileToDistribution
from .kv_cache import TabICLCache
from .inference_config import InferenceConfig


class TabICL(nn.Module):
    """A Tabular In-Context Learning Foundation Model.

    TabICL is a transformer-based architecture for in-context learning on tabular data to make
    predictions without fine-tuning. It processes tabular data through three sequential stages:

    1. Column-wise embedding creates distribution-aware embeddings
    2. Row-wise interaction captures interactions between features within each row
    3. Dataset-wise in-context learning to learn patterns from labeled examples and make predictions

    This class is the underlying raw PyTorch module for TabICL. It is not
    intended to be used directly. Instead, use the classes from the top-level
    `tabicl` package such as :class:`tabicl.TabICLClassifier` or
    :class:`tabicl.TabICLRegressor` that wrap this class to include the
    necessary preprocessing of input features and postprocessing of
    predictions.

    Parameters
    ----------
    max_classes : int, default=10
        Determines the task type and output behavior:
        - If max_classes=0: The model performs regression using quantile prediction.
        - If max_classes>0: The model performs classification. This value specifies
          the number of classes the model supports natively. If the number of classes
          in the dataset exceeds this value, mixed-radix ensembling is used during
          column-wise embedding and hierarchical classification is used during in-context learning.

    num_quantiles : int, default=999
        Number of quantiles to predict for regression tasks. Only used when max_classes=0.
        The model directly predicts these quantile values.

    embed_dim : int, default=128
        Model dimension used in the column / row embedding transformers. For the in-context
        learning transformer, the dimension is this value multiplied by the number of CLS tokens.

    col_num_blocks : int, default=3
        Number of induced self-attention blocks in the column embedding transformer.

    col_nhead : int, default=8
        Number of attention heads in the column embedding transformer.

    col_num_inds : int, default=128
        Number of inducing points in the column embedding transformer.

    col_affine : bool, default=False
        If True, computes embeddings as: :math:`\\text{features} \\times W + b`.
        If False, directly uses the set transformer output as embeddings.

    col_feature_group : bool or Literal["same", "valid"], default="same"
        Feature grouping mode:
        - False: No grouping
        - True or "same": Group through circular permutation (output has same number of groups as features)
        - "valid": Group through padding and reshaping (output may have fewer groups)

    col_feature_group_size : int, default=3
        Number of features per group when feature grouping is enabled.

    col_target_aware : bool, default=True
        If True, incorporates target information into column-wise embeddings.

    col_ssmax : bool or str, default="qassmax-mlp-elementwise"
        Type of scalable softmax to use in the column embedding transformer. Note that only the first
        attention layer of the induced self-attention blocks uses SSMax.
        If True, equivalent to "qassmax-mlp-elementwise".
        If False, equivalent to "none".
        If a string, uses the specified scalable softmax type.
        Options include:
            - "none": No scaling applied
            - "ssmax": :math:`q_{\\text{scaled}} = q \\cdot (s \\cdot \\log n)` where s is learnable per-head parameter
            - "ssmax-mlp": Uses MLP to compute scaling factors based on sequence length
            - "ssmax-mlp-elementwise": Elementwise scaling per head dimension using MLP
            - "qassmax-mlp": Query-aware scaling: :math:`\\text{scale} = \\text{base\\_mlp}(\\log n) \\cdot (1 + \\tanh(\\text{query\\_mlp}(q)))`
            - "qassmax-mlp-elementwise": Elementwise query-aware scaling

    row_num_blocks : int, default=3
        Number of attention blocks in the row interaction transformer.

    row_nhead : int, default=8
        Number of attention heads in the row interaction transformer.

    row_num_cls : int, default=4
        Number of learnable CLS tokens used to aggregate feature information per row.

    row_rope_base : float, default=100000
        Base scaling factor for rotary position encoding in the row interaction transformer.

    row_rope_interleaved : bool, default=False
        If True, uses interleaved rotation where dimension pairs are (0,1), (2,3), etc.
        If False, uses non-interleaved rotation where the embedding is split into
        first half [0:d//2] and second half [d//2:d].

    row_use_rope : bool, default=True
        If True, uses rotary positional encoding in the row interaction transformer.

    col_output_layer_norm : bool, default=False
        If True, applies LayerNorm to the column embedder output before row interaction.

    icl_num_blocks : int, default=12
        Number of transformer blocks in the in-context learning transformer.

    icl_nhead : int, default=8
        Number of attention heads in the in-context learning transformer.

    icl_ssmax : bool or str, default="qassmax-mlp-elementwise"
        Type of scalable softmax to use in the in-context learning transformer.
        If True, equivalent to "qassmax-mlp-elementwise".
        If False, equivalent to "none".
        If a string, uses the specified scalable softmax type.
        Options include:
            - "none": No scaling applied
            - "ssmax": :math:`q_{\\text{scaled}} = q \\cdot (s \\cdot \\log n)` where s is learnable per-head parameter
            - "ssmax-mlp": Uses MLP to compute scaling factors based on sequence length
            - "ssmax-mlp-elementwise": Elementwise scaling per head dimension using MLP
            - "qassmax-mlp": Query-aware scaling: :math:`\\text{scale} = \\text{base\\_mlp}(\\log n) \\cdot (1 + \\tanh(\\text{query\\_mlp}(q)))`
            - "qassmax-mlp-elementwise": Elementwise query-aware scaling

    ff_factor : int, default=2
        Expansion factor for feedforward networks across all components.

    dropout : float, default=0.0
        Dropout probability across all components.

    activation : str or unary callable, default="gelu"
        Activation function used throughout the model.

    norm_first : bool, default=True
        If True, uses pre-norm architecture across all components.

    bias_free_ln : bool, default=False
        If True, removes bias from all LayerNorm layers (sets bias=False in nn.LayerNorm).

    recompute : bool, default=False
        If True, uses gradient checkpointing to save memory at the cost of additional computation.
    """

    def __init__(
        self,
        max_classes: int = 10,
        num_quantiles: int = 999,
        embed_dim: int = 128,
        col_num_blocks: int = 3,
        col_nhead: int = 8,
        col_num_inds: int = 128,
        col_affine: bool = False,
        col_feature_group: Union[bool, Literal["same", "valid"]] = "same",
        col_feature_group_size: int = 3,
        col_target_aware: bool = True,
        col_ssmax: Union[
            bool,
            Literal[
                "none",
                "ssmax",
                "ssmax-mlp",
                "ssmax-mlp-elementwise",
                "qassmax-mlp",
                "qassmax-mlp-elementwise",
            ],
        ] = "qassmax-mlp-elementwise",
        row_num_blocks: int = 3,
        row_nhead: int = 8,
        row_num_cls: int = 4,
        row_rope_base: float = 100000,
        row_rope_interleaved: bool = False,
        row_use_rope: bool = True,
        col_output_layer_norm: bool = False,
        icl_num_blocks: int = 12,
        icl_nhead: int = 8,
        icl_ssmax: Union[
            bool,
            Literal[
                "none",
                "ssmax",
                "ssmax-mlp",
                "ssmax-mlp-elementwise",
                "qassmax-mlp",
                "qassmax-mlp-elementwise",
            ],
        ] = "qassmax-mlp-elementwise",
        icl_qassmax_cap_enabled: bool = False,
        icl_qassmax_cap_layers: str | Sequence[int] = "7;8;9;10",
        icl_qassmax_cap_base_scale: float = 16.0,
        icl_qassmax_cap_scale: float = 16.0,
        icl_qassmax_cap_query_logit: float = 8.0,
        layer_gate_enabled: bool = False,
        layer_gate_layers: str | Sequence[int] = "4;6;8;10;11",
        layer_gate_hidden_dim: int = 128,
        layer_gate_temperature: float = 1.0,
        layer_gate_low_layer_floor: float = 0.0,
        layer_gate_low_layer_max: int = 6,
        layer_gate_use_confidence_features: bool = False,
        layer_gate_repr_source: str = "deepest",
        layer_gate_confidence_source: str = "deepest",
        layer_gate_max_weight_target: float = 0.85,
        attention_gate_enabled: bool = False,
        attention_gate_shape: Literal["scalar", "channel"] = "scalar",
        attention_gate_layers: str | Sequence[int] = "8;9;10;11",
        attention_gate_hidden_dim: int = 128,
        attention_gate_rho: float = 0.25,
        swiglu_enabled: bool = False,
        swiglu_conditioned: bool = False,
        swiglu_hidden_dim: int = 1024,
        swiglu_context_hidden_dim: int = 128,
        swiglu_rho: float = 0.25,
        swiglu_output_scale: float = 1.0,
        swiglu_product_tanh_rms_multiple: float = 0.0,
        swiglu_product_tanh_last_n_layers: int = 0,
        swiglu_init_seed_base: int = 2026082601,
        cr2_shared_refinement_enabled: bool = False,
        qk_pds_attention_enabled: bool = False,
        cls8_pooled_enabled: bool = False,
        shared_depth_icl_enabled: bool = False,
        shared_depth_icl_rho: float = 1.0,
        shared_depth_icl_dataset_conditioned: bool = False,
        shared_depth_icl_num_passes: int = 2,
        cls8_width_enabled: bool = False,
        schema_expert_enabled: bool = False,
        schema_expert_num_experts: int = 32,
        schema_expert_top_k: int = 2,
        schema_expert_bottleneck: int = 32,
        schema_expert_hidden_dim: int = 128,
        schema_expert_router_temperature: float = 1.0,
        schema_expert_adapter_scale: float = 0.1,
        schema_film_enabled: bool = False,
        schema_film_scale: float = 0.1,
        schema_expert_dropout: float = 0.0,
        schema_context_source: str = "support",
        function_tokens_enabled: bool = False,
        function_token_use_dataset_token: bool = True,
        function_token_use_query_token: bool = True,
        function_token_use_latent_tokens: bool = False,
        function_token_dataset_count: int = 1,
        function_token_query_count: int = 1,
        function_token_latent_count: int = 8,
        function_token_num_heads: int = 8,
        function_token_num_layers: int = 1,
        function_token_hidden_dim: int = 128,
        function_token_scale: float = 0.1,
        function_token_dropout: float = 0.0,
        function_token_context_source: str = "support",
        ff_factor: int = 2,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
        recompute: bool = False,
    ):
        super().__init__()
        if (cls8_pooled_enabled or cls8_width_enabled) and row_num_cls != 4:
            raise ValueError("CLS8 variants require row_num_cls=4 in the reference E4 architecture")
        if cls8_pooled_enabled and cls8_width_enabled:
            raise ValueError("cls8_pooled_enabled and cls8_width_enabled are mutually exclusive")
        row_token_count = 8 if (cls8_pooled_enabled or cls8_width_enabled) else row_num_cls
        row_output_count = 8 if cls8_width_enabled else row_num_cls
        icl_dim = embed_dim * row_output_count

        # Determine task type
        if max_classes == 0:  # Regression
            if num_quantiles <= 0:
                raise ValueError("For regression (max_classes=0), num_quantiles must be greater than 0.")
            out_dim = num_quantiles
            self.quantile_dist = QuantileToDistribution(num_quantiles=num_quantiles)
        else:  # Classification
            out_dim = max_classes

        self.max_classes = max_classes
        self.num_quantiles = num_quantiles
        self.embed_dim = embed_dim
        self.col_num_blocks = col_num_blocks
        self.col_nhead = col_nhead
        self.col_num_inds = col_num_inds
        self.col_affine = col_affine
        self.col_feature_group = col_feature_group
        self.col_feature_group_size = col_feature_group_size
        self.col_target_aware = col_target_aware
        self.col_ssmax = col_ssmax
        self.row_num_blocks = row_num_blocks
        self.row_nhead = row_nhead
        self.row_num_cls = row_num_cls
        self.row_rope_base = row_rope_base
        self.row_rope_interleaved = row_rope_interleaved
        self.row_use_rope = row_use_rope
        self.col_output_layer_norm = col_output_layer_norm
        self.icl_num_blocks = icl_num_blocks
        self.icl_nhead = icl_nhead
        self.icl_ssmax = icl_ssmax
        self.icl_qassmax_cap_enabled = icl_qassmax_cap_enabled
        self.icl_qassmax_cap_layers = icl_qassmax_cap_layers
        self.icl_qassmax_cap_base_scale = icl_qassmax_cap_base_scale
        self.icl_qassmax_cap_scale = icl_qassmax_cap_scale
        self.icl_qassmax_cap_query_logit = icl_qassmax_cap_query_logit
        self.layer_gate_enabled = layer_gate_enabled
        self.layer_gate_layers = layer_gate_layers
        self.layer_gate_hidden_dim = layer_gate_hidden_dim
        self.layer_gate_temperature = layer_gate_temperature
        self.layer_gate_low_layer_floor = layer_gate_low_layer_floor
        self.layer_gate_low_layer_max = layer_gate_low_layer_max
        self.layer_gate_use_confidence_features = layer_gate_use_confidence_features
        self.layer_gate_repr_source = layer_gate_repr_source
        self.layer_gate_confidence_source = layer_gate_confidence_source
        self.layer_gate_max_weight_target = layer_gate_max_weight_target
        self.attention_gate_enabled = attention_gate_enabled
        self.attention_gate_shape = attention_gate_shape
        self.attention_gate_layers = attention_gate_layers
        self.attention_gate_hidden_dim = attention_gate_hidden_dim
        self.attention_gate_rho = attention_gate_rho
        self.swiglu_enabled = bool(swiglu_enabled)
        self.swiglu_conditioned = bool(swiglu_conditioned)
        self.swiglu_hidden_dim = int(swiglu_hidden_dim)
        self.swiglu_context_hidden_dim = int(swiglu_context_hidden_dim)
        self.swiglu_rho = float(swiglu_rho)
        self.swiglu_output_scale = float(swiglu_output_scale)
        self.swiglu_product_tanh_rms_multiple = float(swiglu_product_tanh_rms_multiple)
        self.swiglu_product_tanh_last_n_layers = int(swiglu_product_tanh_last_n_layers)
        self.swiglu_init_seed_base = int(swiglu_init_seed_base)
        self.cr2_shared_refinement_enabled = bool(cr2_shared_refinement_enabled)
        self.qk_pds_attention_enabled = bool(qk_pds_attention_enabled)
        self.cls8_pooled_enabled = bool(cls8_pooled_enabled)
        self.shared_depth_icl_enabled = bool(shared_depth_icl_enabled)
        self.shared_depth_icl_rho = float(shared_depth_icl_rho)
        self.shared_depth_icl_dataset_conditioned = bool(shared_depth_icl_dataset_conditioned)
        self.shared_depth_icl_num_passes = int(shared_depth_icl_num_passes)
        self.cls8_width_enabled = bool(cls8_width_enabled)
        if self.shared_depth_icl_enabled and self.shared_depth_icl_num_passes < 2:
            raise ValueError("shared_depth_icl_num_passes must be >= 2 when shared depth is enabled")
        if self.shared_depth_icl_dataset_conditioned and not self.shared_depth_icl_enabled:
            raise ValueError(
                "shared_depth_icl_dataset_conditioned requires shared_depth_icl_enabled"
            )
        if self.shared_depth_icl_dataset_conditioned and (
            self.attention_gate_enabled or self.swiglu_conditioned
        ):
            raise ValueError(
                "dataset-conditioned shared depth is isolated from attention/SwiGLU conditioning"
            )
        if self.swiglu_conditioned and not self.swiglu_enabled:
            raise ValueError("swiglu_conditioned requires swiglu_enabled")
        if self.swiglu_enabled and self.swiglu_hidden_dim <= 0:
            raise ValueError("swiglu_hidden_dim must be positive")
        if not math.isfinite(self.swiglu_output_scale) or self.swiglu_output_scale <= 0.0:
            raise ValueError("swiglu_output_scale must be finite and positive")
        if (
            not math.isfinite(self.swiglu_product_tanh_rms_multiple)
            or self.swiglu_product_tanh_rms_multiple < 0.0
        ):
            raise ValueError("swiglu_product_tanh_rms_multiple must be finite and non-negative")
        if not 0 <= self.swiglu_product_tanh_last_n_layers <= icl_num_blocks:
            raise ValueError(
                "swiglu_product_tanh_last_n_layers must be within "
                f"[0, {icl_num_blocks}]"
            )
        if (self.swiglu_product_tanh_rms_multiple > 0.0) != (
            self.swiglu_product_tanh_last_n_layers > 0
        ):
            raise ValueError(
                "SwiGLU product smoothing multiple and last-N depth must be enabled together"
            )
        if self.swiglu_product_tanh_last_n_layers > 0 and not self.swiglu_enabled:
            raise ValueError("SwiGLU product smoothing requires swiglu_enabled")
        self.schema_expert_enabled = schema_expert_enabled
        self.schema_expert_num_experts = schema_expert_num_experts
        self.schema_expert_top_k = schema_expert_top_k
        self.schema_expert_bottleneck = schema_expert_bottleneck
        self.schema_expert_hidden_dim = schema_expert_hidden_dim
        self.schema_expert_router_temperature = schema_expert_router_temperature
        self.schema_expert_adapter_scale = schema_expert_adapter_scale
        self.schema_film_enabled = schema_film_enabled
        self.schema_film_scale = schema_film_scale
        self.schema_expert_dropout = schema_expert_dropout
        self.schema_context_source = schema_context_source
        self.function_tokens_enabled = function_tokens_enabled
        self.function_token_use_dataset_token = function_token_use_dataset_token
        self.function_token_use_query_token = function_token_use_query_token
        self.function_token_use_latent_tokens = function_token_use_latent_tokens
        self.function_token_dataset_count = function_token_dataset_count
        self.function_token_query_count = function_token_query_count
        self.function_token_latent_count = function_token_latent_count
        self.function_token_num_heads = function_token_num_heads
        self.function_token_num_layers = function_token_num_layers
        self.function_token_hidden_dim = function_token_hidden_dim
        self.function_token_scale = function_token_scale
        self.function_token_dropout = function_token_dropout
        self.function_token_context_source = function_token_context_source
        self.ff_factor = ff_factor
        self.dropout = dropout
        self.activation = activation
        self.norm_first = norm_first
        self.bias_free_ln = bias_free_ln

        if schema_context_source != "support":
            raise ValueError(f"Unsupported schema_context_source={schema_context_source!r}; expected 'support'.")
        if function_token_context_source != "support":
            raise ValueError(
                f"Unsupported function_token_context_source={function_token_context_source!r}; expected 'support'."
            )

        self.col_embedder = ColEmbedding(
            embed_dim=embed_dim,
            num_blocks=col_num_blocks,
            nhead=col_nhead,
            num_inds=col_num_inds,
            dim_feedforward=embed_dim * ff_factor,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            bias_free_ln=bias_free_ln,
            affine=col_affine,
            feature_group=col_feature_group,
            feature_group_size=col_feature_group_size,
            target_aware=col_target_aware,
            max_classes=max_classes,
            reserve_cls_tokens=row_token_count,
            ssmax=col_ssmax,
            recompute=recompute,
        )

        self.row_interactor = RowInteraction(
            embed_dim=embed_dim,
            num_blocks=row_num_blocks,
            nhead=row_nhead,
            dim_feedforward=embed_dim * ff_factor,
            num_cls=row_token_count,
            output_num_cls=row_output_count,
            reference_num_cls=row_num_cls,
            rope_base=row_rope_base,
            rope_interleaved=row_rope_interleaved,
            use_rope=row_use_rope,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            bias_free_ln=bias_free_ln,
            recompute=recompute,
        )

        # CR2 reuses the existing column and row stacks.  The only additional
        # parameters are two zero-initialized scalar gates, so enabling CR2
        # neither consumes RNG nor perturbs any shared E4 initialization.
        if self.cr2_shared_refinement_enabled:
            self.cr2_row_gate = nn.Parameter(torch.zeros(()))
            self.cr2_col_gate = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("cr2_row_gate", None)
            self.register_parameter("cr2_col_gate", None)

        self.col_output_ln = (
            nn.LayerNorm(embed_dim, bias=not bias_free_ln) if col_output_layer_norm else nn.Identity()
        )

        self.schema_conditioner = (
            SchemaExpertConditioner(
                d_model=icl_dim,
                max_classes=max_classes,
                num_experts=schema_expert_num_experts,
                top_k=schema_expert_top_k,
                bottleneck=schema_expert_bottleneck,
                hidden_dim=schema_expert_hidden_dim,
                adapter_scale=schema_expert_adapter_scale,
                film_enabled=schema_film_enabled,
                film_scale=schema_film_scale,
                dropout=schema_expert_dropout,
                adapter_enabled=schema_expert_enabled,
                router_temperature=schema_expert_router_temperature,
            )
            if schema_expert_enabled or schema_film_enabled
            else None
        )

        self.function_token_conditioner = (
            FunctionTokenConditioner(
                d_model=icl_dim,
                max_classes=max_classes,
                dataset_count=function_token_dataset_count,
                query_count=function_token_query_count,
                latent_count=function_token_latent_count,
                hidden_dim=function_token_hidden_dim,
                num_heads=function_token_num_heads,
                num_layers=function_token_num_layers,
                scale=function_token_scale,
                dropout=function_token_dropout,
                use_dataset_token=function_token_use_dataset_token,
                use_query_token=function_token_use_query_token,
                use_latent_tokens=function_token_use_latent_tokens,
                bias_free_ln=bias_free_ln,
            )
            if function_tokens_enabled
            else None
        )

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2026082202)
            self.attention_gate_schema_encoder = (
                SupportSchemaEncoder(
                    max_classes=max_classes,
                    hidden_dim=(swiglu_context_hidden_dim if swiglu_conditioned else attention_gate_hidden_dim),
                    embedding_dim=(swiglu_context_hidden_dim if swiglu_conditioned else attention_gate_hidden_dim),
                )
                if attention_gate_enabled or swiglu_conditioned
                else None
            )
        self.attention_gate_group_rff = (
            SupportGroupRFF() if attention_gate_enabled or swiglu_conditioned else None
        )
        self.shared_depth_condition_stats = (
            SupportSchemaStatistics(max_classes=max_classes)
            if self.shared_depth_icl_dataset_conditioned
            else None
        )

        self.icl_predictor = ICLearning(
            out_dim=out_dim,
            max_classes=max_classes,
            d_model=icl_dim,
            num_blocks=icl_num_blocks,
            nhead=icl_nhead,
            dim_feedforward=(swiglu_hidden_dim if swiglu_enabled else icl_dim * ff_factor),
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            bias_free_ln=bias_free_ln,
            ssmax=icl_ssmax,
            recompute=recompute,
            layer_gate_enabled=layer_gate_enabled,
            layer_gate_layers=layer_gate_layers,
            layer_gate_hidden_dim=layer_gate_hidden_dim,
            layer_gate_temperature=layer_gate_temperature,
            layer_gate_low_layer_floor=layer_gate_low_layer_floor,
            layer_gate_low_layer_max=layer_gate_low_layer_max,
            layer_gate_use_confidence_features=layer_gate_use_confidence_features,
            layer_gate_repr_source=layer_gate_repr_source,
            layer_gate_confidence_source=layer_gate_confidence_source,
            layer_gate_max_weight_target=layer_gate_max_weight_target,
            attention_gate_enabled=attention_gate_enabled,
            attention_gate_shape=attention_gate_shape,
            attention_gate_layers=attention_gate_layers,
            attention_gate_context_dim=attention_gate_hidden_dim + SupportGroupRFF.output_dim,
            attention_gate_rho=attention_gate_rho,
            swiglu_enabled=swiglu_enabled,
            swiglu_conditioned=swiglu_conditioned,
            swiglu_context_dim=swiglu_context_hidden_dim + SupportGroupRFF.output_dim,
            swiglu_rho=swiglu_rho,
            swiglu_output_scale=swiglu_output_scale,
            swiglu_product_tanh_rms_multiple=swiglu_product_tanh_rms_multiple,
            swiglu_product_tanh_last_n_layers=swiglu_product_tanh_last_n_layers,
            swiglu_init_seed_base=swiglu_init_seed_base,
            reference_dim_feedforward=icl_dim * ff_factor,
            qk_pds_enabled=self.qk_pds_attention_enabled,
            shared_depth_enabled=self.shared_depth_icl_enabled,
            shared_depth_rho=self.shared_depth_icl_rho,
            shared_depth_dataset_conditioned=self.shared_depth_icl_dataset_conditioned,
            shared_depth_num_passes=self.shared_depth_icl_num_passes,
            shared_depth_context_dim=SupportSchemaStatistics.stats_dim,
        )
        self._apply_icl_qassmax_caps()

        # E4 intentionally creates no model parameters and therefore remains
        # exactly the E3 architecture.  E5 introduces only a rank-2 domain
        # correction at the column/row boundary.  E6 additionally introduces a
        # shared group-statistic residual at the ICL input.  With absent side
        # information both paths are exactly zero, which is also the default
        # one-domain inference behavior.
        self.cross_table_arm = os.environ.get("CROSS_TABLE_ARM", "").upper()
        self.cross_table_domain_enabled = self.cross_table_arm in {"E5", "E6"}
        self.cross_table_stats_enabled = self.cross_table_arm == "E6"
        if self.cross_table_domain_enabled:
            self.cross_table_domain_projection = nn.Linear(2, embed_dim, bias=False)
            nn.init.normal_(self.cross_table_domain_projection.weight, mean=0.0, std=0.02)
            self.cross_table_domain_gate = nn.Parameter(torch.zeros(()))
        else:
            self.cross_table_domain_projection = None
            self.register_parameter("cross_table_domain_gate", None)
        if self.cross_table_stats_enabled:
            self.cross_table_stats_norm = nn.LayerNorm(64, bias=not bias_free_ln)
            self.cross_table_stats_projection = nn.Linear(64, icl_dim, bias=False)
            nn.init.normal_(self.cross_table_stats_projection.weight, mean=0.0, std=0.01)
            self.cross_table_stats_gate = nn.Parameter(torch.zeros(()))
        else:
            self.cross_table_stats_norm = None
            self.cross_table_stats_projection = None
            self.register_parameter("cross_table_stats_gate", None)

        # KV cache for efficient inference
        self._cache: Optional[TabICLCache] = None

    @staticmethod
    def _parse_icl_layer_indices(value: str | Sequence[int]) -> list[int]:
        if isinstance(value, str):
            items = value.replace(",", ";").replace(":", ";").split(";")
        else:
            items = list(value)
        layers: list[int] = []
        for item in items:
            if item == "":
                continue
            layers.append(int(item))
        return layers

    def _apply_icl_qassmax_caps(self) -> None:
        if not bool(self.icl_qassmax_cap_enabled):
            return
        blocks = getattr(getattr(self.icl_predictor, "tf_icl", None), "blocks", None)
        if blocks is None:
            raise ValueError("Could not locate icl_predictor.tf_icl.blocks for ICL QASSMax cap.")
        for cap_name, cap_value in (
            ("icl_qassmax_cap_base_scale", self.icl_qassmax_cap_base_scale),
            ("icl_qassmax_cap_scale", self.icl_qassmax_cap_scale),
            ("icl_qassmax_cap_query_logit", self.icl_qassmax_cap_query_logit),
        ):
            if float(cap_value) <= 0:
                raise ValueError(f"{cap_name} must be positive when icl_qassmax_cap_enabled=True.")

        for layer_idx in sorted(set(self._parse_icl_layer_indices(self.icl_qassmax_cap_layers))):
            if layer_idx < 0 or layer_idx >= len(blocks):
                continue
            ssmax_layer = getattr(getattr(blocks[layer_idx], "attn", None), "ssmax_layer", None)
            if ssmax_layer is None:
                continue
            if not all(
                hasattr(ssmax_layer, attr)
                for attr in ("max_abs_base_scale", "max_abs_scale", "max_abs_query_logit")
            ):
                continue
            ssmax_layer.max_abs_base_scale = float(self.icl_qassmax_cap_base_scale)
            ssmax_layer.max_abs_scale = float(self.icl_qassmax_cap_scale)
            ssmax_layer.max_abs_query_logit = float(self.icl_qassmax_cap_query_logit)

    @property
    def schema_expert_metrics(self) -> dict[str, Tensor]:
        if self.schema_conditioner is None:
            return {}
        return self.schema_conditioner.last_metrics

    @property
    def function_token_metrics(self) -> dict[str, Tensor]:
        if self.function_token_conditioner is None:
            return {}
        return self.function_token_conditioner.last_metrics

    @property
    def layer_gate_metrics(self) -> dict[str, Tensor]:
        return getattr(self.icl_predictor, "layer_gate_metrics", {}) or {}

    def _apply_col_output_ln(self, col_embeddings: Tensor) -> Tensor:
        if isinstance(self.col_output_ln, nn.Identity):
            return col_embeddings

        input_dtype = col_embeddings.dtype
        weight = getattr(self.col_output_ln, "weight", None)
        norm_dtype = weight.dtype if weight is not None else input_dtype
        if input_dtype != norm_dtype:
            return self.col_output_ln(col_embeddings.to(dtype=norm_dtype)).to(dtype=input_dtype)
        return self.col_output_ln(col_embeddings)

    def _centered_cross_table_domain_code(
        self,
        domain_ids: Tensor,
        reference: Tensor,
    ) -> Tensor:
        if domain_ids.ndim != 2 or domain_ids.shape[:2] != reference.shape[:2]:
            raise ValueError(
                "domain_ids must have shape [B,T] matching row positions, "
                f"got {tuple(domain_ids.shape)} and {tuple(reference.shape)}"
            )
        domain_ids = domain_ids.to(device=reference.device, dtype=reference.dtype)
        angle = (2.0 * math.pi / 4.0) * domain_ids
        code = torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)
        # Equal-domain centering, rather than row-count centering, prevents a
        # slightly larger integer partition from receiving a systematic offset.
        centered = torch.empty_like(code)
        for batch_index in range(code.shape[0]):
            ids = domain_ids[batch_index].long()
            unique_ids = torch.unique(ids, sorted=True)
            domain_codes = torch.stack(
                [code[batch_index][ids == domain_id][0] for domain_id in unique_ids], dim=0
            )
            centered[batch_index] = code[batch_index] - domain_codes.mean(dim=0)
        return centered

    def _apply_cross_table_domain_conditioning(
        self,
        col_embeddings: Tensor,
        domain_ids: Optional[Tensor],
    ) -> Tensor:
        if self.cross_table_domain_projection is None or domain_ids is None:
            return col_embeddings
        centered = self._centered_cross_table_domain_code(domain_ids, col_embeddings)
        correction = self.cross_table_domain_projection(centered)
        gate = 0.01 * torch.tanh(self.cross_table_domain_gate)
        correction = correction * gate
        # ColEmbedding outputs [B,T,H,E]; the same tiny domain correction is
        # broadcast over columns before row attention.
        return col_embeddings + correction.unsqueeze(-2)

    def _apply_cross_table_domain_row_conditioning(
        self,
        representations: Tensor,
        domain_ids: Optional[Tensor],
    ) -> Tensor:
        """Keep the tiny domain signal observable at the row-encoder boundary.

        The row encoder replaces its reserved CLS inputs and can strongly
        attenuate a small feature-token offset.  Reusing the same 2D code,
        projection, and bounded gate on the concatenated row CLS outputs makes
        the intended attention correction trainable without adding parameters
        or increasing the domain-code dimensionality.
        """
        if self.cross_table_domain_projection is None or domain_ids is None:
            return representations
        centered = self._centered_cross_table_domain_code(domain_ids, representations)
        correction = self.cross_table_domain_projection(centered).repeat(1, 1, self.row_num_cls)
        gate = 0.01 * torch.tanh(self.cross_table_domain_gate)
        return representations + gate * correction

    def _apply_cross_table_stats_conditioning(
        self,
        representations: Tensor,
        group_rff: Optional[Tensor],
    ) -> Tensor:
        if self.cross_table_stats_projection is None or group_rff is None:
            return representations
        if group_rff.ndim != 2 or group_rff.shape != (representations.shape[0], 64):
            raise ValueError(
                "group_rff must have shape [B,64], "
                f"got {tuple(group_rff.shape)} for B={representations.shape[0]}"
            )
        group_rff = group_rff.to(device=representations.device, dtype=representations.dtype)
        condition = self.cross_table_stats_projection(self.cross_table_stats_norm(group_rff))
        gate = 0.01 * torch.tanh(self.cross_table_stats_gate)
        # One group-level continuous condition is shared by every support/query
        # row and enters the ICL attention stack through its input tokens.
        return representations + gate * condition.unsqueeze(1)

    @property
    def has_cache(self) -> bool:
        """Check if a valid cache is stored."""
        return self._cache is not None and not self._cache.is_empty()

    def clear_cache(self) -> None:
        """Clear the stored cache."""
        self._cache = None

    def _compute_schema_context(
        self,
        X: Tensor,
        y_train: Tensor,
        d: Optional[Tensor] = None,
        total_seq_len: Optional[int] = None,
    ) -> Optional[Tensor]:
        if self.schema_conditioner is None:
            return None
        return self.schema_conditioner.compute_context(X, y_train, d=d, total_seq_len=total_seq_len)

    def _apply_schema_conditioner(
        self,
        representations: Tensor,
        X: Optional[Tensor] = None,
        y_train: Optional[Tensor] = None,
        d: Optional[Tensor] = None,
        schema_context: Optional[Tensor] = None,
        total_seq_len: Optional[int] = None,
    ) -> Tensor:
        if self.schema_conditioner is None:
            return representations
        return self.schema_conditioner(
            representations,
            X=X,
            y_train=y_train,
            d=d,
            schema_context=schema_context,
            total_seq_len=total_seq_len,
        )

    def _compute_function_context(
        self,
        representations: Tensor,
        y_train: Tensor,
    ) -> Optional[Tensor]:
        if self.function_token_conditioner is None:
            return None
        return self.function_token_conditioner.compute_context(representations, y_train)

    def _compute_attention_gate_context(
        self,
        X: Tensor,
        y_train: Tensor,
        d: Optional[Tensor] = None,
        total_seq_len: Optional[int] = None,
    ) -> Optional[Tensor]:
        if self.shared_depth_condition_stats is not None:
            return self.shared_depth_condition_stats(
                X, y_train, d=d, total_seq_len=total_seq_len
            )
        if self.attention_gate_schema_encoder is not None and self.attention_gate_group_rff is not None:
            schema = self.attention_gate_schema_encoder(X, y_train, d=d, total_seq_len=total_seq_len)
            group_rff = self.attention_gate_group_rff(X, y_train, d=d)
            return torch.cat((schema, group_rff.to(dtype=schema.dtype)), dim=-1)
        return None

    def _apply_cr2_shared_refinement(
        self,
        col_embeddings: Tensor,
        *,
        train_size: int,
        d: Optional[Tensor] = None,
        row_mgr_config=None,
    ) -> Tensor:
        """Apply identity-gated C1-R1-C1-R1 with shared C/R weights.

        The first row pass returns full tokens.  A second call to the existing
        column encoder refines each token across rows, and the final shared row
        pass returns CLS representations.  Both residual gates start at exact
        identity, making the initial G1 function bitwise equal to G0/E4.
        """

        if not self.cr2_shared_refinement_enabled:
            return self.row_interactor(col_embeddings, d=d, mgr_config=row_mgr_config)

        row_full = self.row_interactor.refine_full_tokens(col_embeddings, d=d)
        row_gate = torch.tanh(self.cr2_row_gate).to(dtype=col_embeddings.dtype)
        row_refined = col_embeddings + row_gate * (row_full - col_embeddings)

        # Encoder treats the penultimate axis as sequence.  Move rows there so
        # the exact same C1 weights perform the second column refinement.
        col_full = self.col_embedder.tf_col(
            row_refined.transpose(1, 2),
            train_size=train_size,
        ).transpose(1, 2)
        col_gate = torch.tanh(self.cr2_col_gate).to(dtype=col_embeddings.dtype)
        col_refined = row_refined + col_gate * (col_full - row_refined)

        return self.row_interactor(col_refined, d=d, mgr_config=row_mgr_config)

    def _apply_function_token_conditioner(
        self,
        representations: Tensor,
        y_train: Optional[Tensor] = None,
        function_context: Optional[Tensor] = None,
        query_start: Optional[int] = None,
    ) -> Tensor:
        if self.function_token_conditioner is None:
            return representations
        return self.function_token_conditioner(
            representations,
            y_train=y_train,
            function_context=function_context,
            query_start=query_start,
        )

    def _train_forward(
        self,
        X: Tensor,
        y_train: Tensor,
        d: Optional[Tensor] = None,
        embed_with_test: bool = False,
        return_row_representations: bool = False,
        return_icl_representations: bool = False,
        icl_intermediate_layers: Optional[Sequence[int]] = None,
        domain_ids: Optional[Tensor] = None,
        group_rff: Optional[Tensor] = None,
    ) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, dict[str, Tensor | dict[int, Tensor]]]:
        """Column-wise embedding -> row-wise interaction -> dataset-wise in-context learning for training.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (B, T, H) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features (columns)
            The first train_size positions contain training samples, and the remaining positions contain test samples.

        y_train : Tensor
            Training labels of shape (B, train_size) where:
             - B is the number of tables
             - train_size is the number of training samples provided for in-context learning

        d : Optional[Tensor], default=None
            The number of features per dataset.

        embed_with_test : bool, default=False
            If True, allow training samples to attend to test samples during embedding.

        return_row_representations : bool, default=False
            If True, also return the raw row-interactor output before optional
            schema/function conditioning. This is used by training-only auxiliary
            representation losses and leaves the default model API unchanged.

        return_icl_representations : bool, default=False
            If True, also return selected ICL block outputs for training-only
            auxiliary representation losses.

        Returns
        -------
        Tensor
            Predictions of shape (B, test_size, out_dim):

            - For regression (max_classes=0): out_dim = num_quantiles
            - For classification (max_classes>0): out_dim = max_classes
        """

        B, T, H = X.shape
        train_size = y_train.shape[1]
        assert train_size <= T, "Number of training samples exceeds total samples"
        attention_gate_context = self._compute_attention_gate_context(
            X, y_train, d=d, total_seq_len=T
        )

        # Check if d is provided and has the same length as the number of features
        if d is not None and len(d.unique()) == 1 and d[0] == H:
            d = None
        col_d = None if self.col_feature_group else d
        row_key_d = None if self.col_feature_group else d

        # Column-wise embedding -> optional output normalization -> row-wise interaction
        col_embeddings = self.col_embedder(
            X,
            y_train=y_train,
            d=col_d,
            embed_with_test=embed_with_test,
        )
        col_embeddings = self._apply_col_output_ln(col_embeddings)
        col_embeddings = self._apply_cross_table_domain_conditioning(col_embeddings, domain_ids)
        representations = self._apply_cr2_shared_refinement(
            col_embeddings,
            train_size=train_size,
            d=row_key_d,
        )
        representations = self._apply_cross_table_domain_row_conditioning(representations, domain_ids)
        row_representations = representations
        representations = self._apply_schema_conditioner(
            representations,
            X=X,
            y_train=y_train,
            d=d,
            total_seq_len=T,
        )
        representations = self._apply_function_token_conditioner(
            representations,
            y_train=y_train,
        )
        representations = self._apply_cross_table_stats_conditioning(representations, group_rff)

        # Dataset-wise in-context learning
        if return_icl_representations:
            predictions, icl_representations = self.icl_predictor(
                representations,
                y_train=y_train,
                return_intermediate_layers=icl_intermediate_layers,
                dataset_context=attention_gate_context,
            )
        else:
            predictions = self.icl_predictor(
                representations, y_train=y_train, dataset_context=attention_gate_context
            )
            icl_representations = None
        if return_row_representations and return_icl_representations:
            return predictions, {"row": row_representations, "icl": icl_representations}
        if return_row_representations:
            return predictions, row_representations
        if return_icl_representations:
            return predictions, icl_representations
        return predictions

    def _inference_forward(
        self,
        X: Tensor,
        y_train: Tensor,
        feature_shuffles: Optional[List[List[int]]] = None,
        embed_with_test: bool = False,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        inference_config: Optional[InferenceConfig] = None,
    ) -> Tensor:
        """Column-wise embedding -> row-wise interaction -> dataset-wise in-context learning.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (B, T, H) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features (columns)
            The first train_size positions contain training samples, and the remaining positions contain test samples.

        y_train : Tensor
            Training labels of shape (B, train_size) where:
             - B is the number of tables
             - train_size is the number of training samples provided for in-context learning

        feature_shuffles : Optional[List[List[int]]], default=None
            A list of feature shuffle patterns for each table in the batch.
            When provided, indicates that X contains the same table with different feature orders.
            In this case, column-wise embeddings are computed once and then shuffled accordingly.

        embed_with_test : bool, default=False
            If True, allow training samples to attend to test samples during embedding.

        return_logits : bool, default=True
            If True, return raw logits instead of probabilities.

        softmax_temperature : float, default=0.9
            Temperature for the softmax function.

        inference_config : Optional[InferenceConfig], default=None
            Inference configuration.

        Returns
        -------
        Tensor
            For regression (max_classes=0):
                Predictions of shape (B, test_size, num_quantiles), where test_size = T - train_size

            For classification (max_classes>0):
                If return_logits=True: Logits of shape (B, test_size, num_classes)
                If return_logits=False: Probabilities of shape (B, test_size, num_classes)
        """

        train_size = y_train.shape[1]
        assert train_size <= X.shape[1], "Number of training samples exceeds total samples"
        attention_gate_context = self._compute_attention_gate_context(
            X, y_train, total_seq_len=X.shape[1]
        )

        if inference_config is None:
            inference_config = InferenceConfig()

        # Column-wise embedding -> optional output normalization -> row-wise interaction
        col_embeddings = self.col_embedder(
            X,
            y_train=y_train,
            embed_with_test=embed_with_test,
            feature_shuffles=feature_shuffles,
            mgr_config=inference_config.COL_CONFIG,
        )
        col_embeddings = self._apply_col_output_ln(col_embeddings)
        representations = self._apply_cr2_shared_refinement(
            col_embeddings,
            train_size=train_size,
            row_mgr_config=inference_config.ROW_CONFIG,
        )
        representations = self._apply_schema_conditioner(
            representations,
            X=X,
            y_train=y_train,
            total_seq_len=X.shape[1],
        )
        representations = self._apply_function_token_conditioner(
            representations,
            y_train=y_train,
        )

        # Dataset-wise in-context learning
        out = self.icl_predictor(
            representations,
            y_train=y_train,
            return_logits=return_logits,
            softmax_temperature=softmax_temperature,
            mgr_config=inference_config.ICL_CONFIG,
            dataset_context=attention_gate_context,
        )

        return out

    def forward(
        self,
        X: Tensor,
        y_train: Tensor,
        d: Optional[Tensor] = None,
        embed_with_test: bool = False,
        feature_shuffles: Optional[List[List[int]]] = None,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        inference_config: Optional[InferenceConfig] = None,
        return_row_representations: bool = False,
        return_icl_representations: bool = False,
        icl_intermediate_layers: Optional[Sequence[int]] = None,
        domain_ids: Optional[Tensor] = None,
        group_rff: Optional[Tensor] = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Column-wise embedding -> row-wise interaction -> dataset-wise in-context learning.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (B, T, H) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features (columns)
            The first train_size positions contain training samples, and the remaining positions contain test samples.

        y_train : Tensor
            Training labels of shape (B, train_size) where:
             - B is the number of tables
             - train_size is the number of training samples provided for in-context learning

        d : Optional[Tensor], default=None
            The number of features per dataset. Used only in training mode.

        embed_with_test : bool, default=False
            If True, allow training samples to attend to test samples during embedding.

        feature_shuffles : Optional[List[List[int]]], default=None
            A list of feature shuffle patterns for each table in the batch. Used only in inference mode.
            When provided, indicates that X contains the same table with different feature orders.
            In this case, column-wise embeddings are computed once and then shuffled accordingly.

        return_logits : bool, default=True
            If True, return raw logits instead of probabilities. Used only in inference mode.

        softmax_temperature : float, default=0.9
            Temperature for the softmax function. Used only in inference mode.

        inference_config : Optional[InferenceConfig], default=None
            Inference configuration. Used only in inference mode.

        return_row_representations : bool, default=False
            Training-path hook for auxiliary representation losses. When enabled,
            returns (predictions, row_representations).

        return_icl_representations : bool, default=False
            Training-path hook for auxiliary losses on selected ICL block outputs.

        Returns
        -------
        Tensor
            For training mode:
                Predictions of shape (B, test_size, out_dim):

                - For regression (max_classes=0): out_dim = num_quantiles
                - For classification (max_classes>0): out_dim = max_classes

            For inference mode:
                For regression (max_classes=0):
                    Predictions of shape (B, test_size, num_quantiles)

                For classification (max_classes>0):
                    If return_logits=True: Logits of shape (B, test_size, num_classes)
                    If return_logits=False: Probabilities of shape (B, test_size, num_classes)
        """

        if self.training:
            out = self._train_forward(
                X,
                y_train,
                d=d,
                embed_with_test=embed_with_test,
                return_row_representations=return_row_representations,
                return_icl_representations=return_icl_representations,
                icl_intermediate_layers=icl_intermediate_layers,
                domain_ids=domain_ids,
                group_rff=group_rff,
            )
        else:
            if return_row_representations or return_icl_representations or icl_intermediate_layers is not None:
                out = self._train_forward(
                    X,
                    y_train,
                    d=d,
                    embed_with_test=embed_with_test,
                    return_row_representations=return_row_representations,
                    return_icl_representations=return_icl_representations,
                    icl_intermediate_layers=icl_intermediate_layers,
                    domain_ids=domain_ids,
                    group_rff=group_rff,
                )
                return out
            out = self._inference_forward(
                X,
                y_train,
                feature_shuffles=feature_shuffles,
                embed_with_test=embed_with_test,
                return_logits=return_logits,
                softmax_temperature=softmax_temperature,
                inference_config=inference_config,
            )

        return out

    def predict_stats(
        self,
        X: Tensor,
        y_train: Tensor,
        output_type: str = "mean",
        alphas: Optional[List[float]] = None,
        embed_with_test: bool = False,
        inference_config: InferenceConfig = None,
    ) -> Tensor:
        """Compute summary statistics from predicted quantiles.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (B, T, H) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features (columns)
            The first train_size positions contain training samples, and the remaining
            positions contain test samples.

        y_train : Tensor
            Training labels of shape (B, train_size) where:
             - B is the number of tables
             - train_size is the number of training samples provided for in-context learning

        output_type : str or list of str, default="mean"
            Determines the type of output to return. Supported values:
            - "mean": Mean of the predicted quantiles (fast, no tail modeling).
            - "variance": Variance of the predicted quantiles (fast, no tail modeling).
            - "median": Median via inverse CDF interpolation.
            - "quantiles": Specific quantiles via inverse CDF. Use `alphas` to specify levels.
            If a list, returns a dict with the requested statistics.

        alphas : Optional[List[float]], default=None
            Probability levels for quantile output. Only used when "quantiles" is in `output_type`.
            Default: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].

        embed_with_test : bool, default=False
            If True, allow training samples to attend to test samples during embedding.

        inference_config : InferenceConfig
            Inference configuration.

        Returns
        -------
        Tensor or dict of Tensors
            - If `output_type` is a single string: returns the corresponding tensor.
            - If `output_type` is a list: returns a dict mapping names to tensors.

            Output shapes:

            - "mean", "variance", "median": (B, test_size)
            - "quantiles": (B, test_size, len(alphas))
            - "raw_quantiles": (B, test_size, num_quantiles), where `num_quantiles` denotes 
                the number of quantile levels configured in the model architecture.
        """
        assert self.max_classes == 0, "predict_stats is only applicable for regression tasks"

        raw_quantiles = self._inference_forward(
            X, y_train, embed_with_test=embed_with_test, inference_config=inference_config
        )  # (B, test_size, num_quantiles)

        dist = self.quantile_dist(raw_quantiles)
        raw_quantiles = dist.quantiles  # dist ensures that quantiles are monotonic

        output_type = [output_type] if isinstance(output_type, str) else output_type
        results = {}

        if "mean" in output_type:
            results["mean"] = raw_quantiles.mean(dim=-1)
        if "variance" in output_type:
            results["variance"] = raw_quantiles.var(dim=-1)
        if "median" in output_type:
            results["median"] = dist.icdf(
                alpha=torch.tensor(0.5, device=raw_quantiles.device, dtype=raw_quantiles.dtype)
            )
        if "quantiles" in output_type:
            if alphas is None:
                alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            results["quantiles"] = dist.icdf(
                alpha=torch.tensor(alphas, device=raw_quantiles.device, dtype=raw_quantiles.dtype)
            )
        if "raw_quantiles" in output_type:
            results["raw_quantiles"] = raw_quantiles

        if len(output_type) == 1:
            return results[output_type[0]]

        return results

    def forward_with_cache(
        self,
        X_train: Optional[Tensor] = None,
        y_train: Optional[Tensor] = None,
        X_test: Optional[Tensor] = None,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        use_cache: bool = False,
        store_cache: bool = True,
        cache: Optional[TabICLCache] = None,
        cache_mode: str = "kv",
        inference_config: Optional[InferenceConfig] = None,
    ) -> Optional[Tensor]:
        """Forward pass with caching support for efficient inference.

        This method enables caching of training data computations to speed up
        repeated inference on the same training context. Two caching modes are
        supported:

        - ``"kv"``: Cache KV projections from both column embedding and ICL
          transformer layers. Fastest inference but uses more memory.
        - ``"repr"``: Cache column embedding KV projections and row interaction
          outputs (representations with y_train baked in). Uses ~24x less memory
          for the ICL part, at the cost of re-running the ICL transformer.

        Exactly one of `use_cache` or `store_cache` must be True.

        When ``store_cache=True``:
        - Requires X_train and y_train to be provided
        - Processes training data and stores cached values in self._cache
        - If X_test is also provided, returns predictions for test samples
        - If X_test is None, returns None (cache-only mode)

        When ``use_cache=True``:
        - Requires X_test and a populated self._cache
        - Uses cached values for training data

        Parameters
        ----------
        X_train : Optional[Tensor], default=None
            Training input of shape (B, train_size, H). Required when store_cache=True.

        y_train : Optional[Tensor], default=None
            Training target of shape (B, train_size). Required when store_cache=True.

        X_test : Optional[Tensor], default=None
            Test input of shape (B, test_size, H). Required when use_cache=True and optional
            when store_cache=True.

        return_logits : bool, default=True
            If True, return raw logits instead of probabilities.

        softmax_temperature : float, default=0.9
            Temperature for the softmax function.

        use_cache : bool, default=False
            Whether to use cached values to avoid redundant computation.

        store_cache : bool, default=True
            Whether to store computed values in cache.

        cache : Optional[TabICLCache], default=None
            External cache to use for inference. If provided, equivalent to
            setting use_cache=True and store_cache=False, but uses the provided
            cache instead of the model's internal self._cache.

        cache_mode : str, default="kv"
            Caching strategy: ``"kv"`` for KV projection caching, ``"repr"`` for
            representation caching. Ignored when ``use_cache=True`` (auto-detected
            from cache contents).

        inference_config : Optional[InferenceConfig], default=None
            Inference configuration.

        Returns
        -------
        Optional[Tensor]
            Predictions of shape (B, test_size, out_dim), or None if store_cache=True
            and X_test is not provided.

        Raises
        ------
        ValueError
            If use_cache == store_cache (exactly one must be True),
            if store_cache=True but X_train or y_train is None, or
            if use_cache=True but X_test is None or no cache exists.
        """

        if cache is not None:
            use_cache = True
            store_cache = False
            self._cache = cache

        if use_cache == store_cache:
            raise ValueError("Exactly one of use_cache or store_cache must be True")

        if cache_mode not in ("kv", "repr"):
            raise ValueError(f"cache_mode must be 'kv' or 'repr', got '{cache_mode}'")

        if inference_config is None:
            inference_config = InferenceConfig()

        # Auto-detect cache mode from cache contents
        if use_cache and self._cache is not None and self._cache.cache_type == "repr":
            cache_mode = "repr"

        schema_context = None
        function_context = None
        attention_gate_context = None
        if store_cache:
            if X_train is None or y_train is None:
                raise ValueError("X_train and y_train are required when store_cache=True")

            # Initialize cache based on training data
            num_classes = len(torch.unique(y_train[0])) if self.max_classes > 0 else 0
            self._cache = TabICLCache(train_shape=X_train.shape, num_classes=num_classes)

            if X_test is None:
                X = X_train
            else:
                X = torch.cat([X_train, X_test], dim=1)
            schema_context = self._compute_schema_context(
                X_train,
                y_train,
                total_seq_len=X.shape[1],
            )
            if schema_context is not None:
                self._cache.schema_context = schema_context.detach()
            attention_gate_context = self._compute_attention_gate_context(
                X_train,
                y_train,
                total_seq_len=X.shape[1],
            )
            if attention_gate_context is not None:
                self._cache.attention_gate_context = attention_gate_context.detach()

        if use_cache:
            if X_test is None:
                raise ValueError("X_test is required when use_cache=True")

            if self._cache is None or self._cache.is_empty():
                raise ValueError("No cache available. Call with store_cache=True first.")

            X = X_test
            y_train = None
            schema_context = self._cache.schema_context
            function_context = self._cache.function_context
            attention_gate_context = self._cache.attention_gate_context
            if self.schema_conditioner is not None and schema_context is None:
                raise ValueError("Schema-conditioned cache is missing schema_context. Rebuild the cache.")
            if self.function_token_conditioner is not None and function_context is None:
                raise ValueError("Function-token cache is missing function_context. Rebuild the cache.")
            if (
                self.attention_gate_enabled or self.shared_depth_icl_dataset_conditioned
            ) and attention_gate_context is None:
                raise ValueError("Attention-gated cache is missing attention_gate_context. Rebuild the cache.")

        # Column-wise embedding with cache support -> optional output normalization -> row-wise interaction
        col_embeddings = self.col_embedder.forward_with_cache(
            X,
            col_cache=self._cache.col_cache,
            y_train=y_train,
            use_cache=use_cache,
            store_cache=store_cache,
            mgr_config=inference_config.COL_CONFIG,
        )
        col_embeddings = self._apply_col_output_ln(col_embeddings)
        representations = self.row_interactor(col_embeddings, mgr_config=inference_config.ROW_CONFIG)
        representations = self._apply_schema_conditioner(representations, schema_context=schema_context)
        if store_cache:
            train_size = y_train.shape[1]
            function_context = self._compute_function_context(representations, y_train)
            if function_context is not None:
                self._cache.function_context = function_context.detach()
            representations = self._apply_function_token_conditioner(
                representations,
                function_context=function_context,
                query_start=train_size,
            )
        else:
            representations = self._apply_function_token_conditioner(
                representations,
                function_context=function_context,
                query_start=0,
            )

        # Dataset-wise in-context learning
        if cache_mode == "repr":
            if store_cache:
                train_size = y_train.shape[1]
                # Bake y_train into train portion of representations
                representations = self.icl_predictor.prepare_repr_cache(representations, y_train)
                self._cache.row_repr = representations[:, :train_size]

                if X_test is None:
                    return None
            else:
                # Concatenate cached train representations with test representations
                train_repr = self._cache.row_repr
                train_size = train_repr.shape[1]
                representations = torch.cat([train_repr.to(representations.device), representations], dim=1)

            out = self.icl_predictor.forward_with_repr_cache(
                representations,
                train_size=train_size,
                num_classes=self._cache.num_classes,
                return_logits=return_logits,
                softmax_temperature=softmax_temperature,
                mgr_config=inference_config.ICL_CONFIG,
                dataset_context=attention_gate_context,
            )
        else:
            out = self.icl_predictor.forward_with_cache(
                representations,
                icl_cache=self._cache.icl_cache,
                y_train=y_train,
                num_classes=self._cache.num_classes,
                return_logits=return_logits,
                softmax_temperature=softmax_temperature,
                use_cache=use_cache,
                store_cache=store_cache,
                mgr_config=inference_config.ICL_CONFIG,
                dataset_context=attention_gate_context,
            )

            if X_test is None:
                return None

        return out

    def predict_stats_with_cache(
        self,
        X_train: Optional[Tensor] = None,
        y_train: Optional[Tensor] = None,
        X_test: Optional[Tensor] = None,
        output_type: str = "mean",
        alphas: Optional[List[float]] = None,
        use_cache: bool = False,
        store_cache: bool = True,
        cache: Optional[TabICLCache] = None,
        cache_mode: str = "kv",
        inference_config: Optional[InferenceConfig] = None,
    ) -> Optional[Tensor]:
        """Compute summary statistics from predicted quantiles with KV caching.

        Parameters
        ----------
        X_train : Optional[Tensor], default=None
            Training input of shape (B, train_size, H). Required when store_cache=True.

        y_train : Optional[Tensor], default=None
            Training target of shape (B, train_size). Required when store_cache=True.

        X_test : Optional[Tensor], default=None
            Test input of shape (B, test_size, H). Required when use_cache=True and
            optional when store_cache=True.

        output_type : str or list of str, default="mean"
            Determines the type of output to return. Supported values:
            - "mean": Mean of the predicted quantiles (fast, no tail modeling).
            - "variance": Variance of the predicted quantiles (fast, no tail modeling).
            - "median": Median via inverse CDF interpolation.
            - "quantiles": Specific quantiles via inverse CDF. Use `alphas` to specify levels.
            If a list, returns a dict with the requested statistics.

        alphas : Optional[List[float]], default=None
            Probability levels for quantile output. Only used when "quantiles" is in
            `output_type`. Default: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].

        use_cache : bool, default=False
            Whether to use cached values to avoid redundant computation.

        store_cache : bool, default=True
            Whether to store computed values in cache.

        cache : Optional[TabICLCache], default=None
            External cache to use for inference. If provided, equivalent to
            setting use_cache=True and store_cache=False.

        cache_mode : str, default="kv"
            Caching strategy: ``"kv"`` for KV projection caching, ``"repr"`` for
            representation caching. Ignored when ``use_cache=True`` (auto-detected
            from cache contents).

        inference_config : Optional[InferenceConfig], default=None
            Inference configuration.

        Returns
        -------
        Tensor or dict of Tensors or None
            None if store_cache=True and X_test is not provided. Otherwise:

            - If `output_type` is a single string: returns the corresponding tensor.
            - If `output_type` is a list: returns a dict mapping names to tensors.

            Output shapes:

            - "mean", "variance", "median": (B, test_size)
            - "quantiles": (B, test_size, len(alphas))
            - "raw_quantiles": (B, test_size, num_quantiles), where `num_quantiles` denotes 
                the number of quantile levels configured in the model architecture.
        """
        assert self.max_classes == 0, "predict_stats_with_cache is only applicable for regression tasks"

        raw_quantiles = self.forward_with_cache(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            use_cache=use_cache,
            store_cache=store_cache,
            cache=cache,
            cache_mode=cache_mode,
            inference_config=inference_config,
        )

        if raw_quantiles is None:
            return None

        dist = self.quantile_dist(raw_quantiles)
        raw_quantiles = dist.quantiles

        output_type = [output_type] if isinstance(output_type, str) else output_type
        results = {}

        if "mean" in output_type:
            results["mean"] = raw_quantiles.mean(dim=-1)
        if "variance" in output_type:
            results["variance"] = raw_quantiles.var(dim=-1)
        if "median" in output_type:
            results["median"] = dist.icdf(
                alpha=torch.tensor(0.5, device=raw_quantiles.device, dtype=raw_quantiles.dtype)
            )
        if "quantiles" in output_type:
            if alphas is None:
                alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            results["quantiles"] = dist.icdf(
                alpha=torch.tensor(alphas, device=raw_quantiles.device, dtype=raw_quantiles.dtype)
            )
        if "raw_quantiles" in output_type:
            results["raw_quantiles"] = raw_quantiles

        if len(output_type) == 1:
            return results[output_type[0]]

        return results
