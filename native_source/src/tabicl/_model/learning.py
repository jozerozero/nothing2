from __future__ import annotations

from typing import Optional, Union, Sequence
from collections import OrderedDict
import math
import torch
from torch import nn, Tensor

from .layers import ClassNode, OneHotAndLinear
from .encoders import Encoder
from .kv_cache import KVCache
from .inference import InferenceManager
from .inference_config import MgrConfig, InferenceConfig


class ICLLayerLogitGate(nn.Module):
    """Query-wise gate over logits decoded from selected ICL layers."""

    def __init__(
        self,
        d_model: int,
        layers: Sequence[int],
        max_classes: int,
        hidden_dim: int = 128,
        gate_repr_dim: int | None = None,
        temperature: float = 1.0,
        low_layer_floor: float = 0.0,
        low_layer_max: int = 6,
        use_confidence_features: bool = False,
        confidence_source: str = "deepest",
        max_weight_target: float = 0.85,
        bias_free_ln: bool = False,
    ):
        super().__init__()
        if not layers:
            raise ValueError("ICLLayerLogitGate requires at least one layer.")
        if temperature <= 0.0:
            raise ValueError("layer gate temperature must be positive.")
        if low_layer_floor < 0.0 or low_layer_floor >= 1.0:
            raise ValueError("layer gate low_layer_floor must be in [0, 1).")
        self.layers = [int(layer) for layer in layers]
        self.max_classes = int(max_classes)
        self.temperature = float(temperature)
        self.low_layer_floor = float(low_layer_floor)
        self.low_layer_max = int(low_layer_max)
        self.use_confidence_features = bool(use_confidence_features)
        confidence_source = str(confidence_source or "none").lower()
        if not self.use_confidence_features:
            confidence_source = "none"
        elif confidence_source == "none":
            confidence_source = "deepest"
        if confidence_source not in {"none", "deepest", "all_layers"}:
            raise ValueError(f"Unsupported layer gate confidence_source={confidence_source!r}")
        self.confidence_source = confidence_source
        self.max_weight_target = float(max_weight_target)
        self.gate_repr_dim = int(gate_repr_dim or d_model)
        confidence_dim = 0
        if self.confidence_source == "deepest":
            confidence_dim = 2
        elif self.confidence_source == "all_layers":
            confidence_dim = 2 * len(self.layers)
        self.confidence_dim = confidence_dim
        self.norm = nn.LayerNorm(self.gate_repr_dim, bias=not bias_free_ln)
        self.gate = nn.Sequential(
            nn.Linear(self.gate_repr_dim + 6 + confidence_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(self.layers)),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.last_metrics: dict[str, Tensor] = {}
        self.last_regularization: dict[str, Tensor] = {}
        self.last_layer_logits: Tensor | None = None

    def _support_features(self, y_train: Tensor, query_len: int, dtype: torch.dtype, device: torch.device) -> Tensor:
        B, train_size = y_train.shape
        labels = y_train.long().clamp(min=0, max=max(0, self.max_classes - 1))
        counts = torch.zeros(B, self.max_classes, device=device, dtype=torch.float32)
        counts.scatter_add_(1, labels.to(device), torch.ones_like(labels, device=device, dtype=torch.float32))
        present = counts > 0
        prior = counts / max(1, int(train_size))
        present_count = present.sum(dim=-1).float().clamp_min(1.0)
        entropy = -(prior * prior.clamp_min(1e-12).log()).sum(dim=-1)
        entropy_norm = entropy / present_count.log().clamp_min(1e-12)
        max_prior = prior.max(dim=-1).values
        min_prior = torch.where(present, prior, torch.ones_like(prior)).min(dim=-1).values
        imbalance = torch.log1p(max_prior / min_prior.clamp_min(1e-6)) / 10.0
        support_features = torch.stack(
            [
                torch.full((B,), math.log1p(float(train_size)) / 10.0, device=device),
                present_count / max(1, self.max_classes),
                entropy_norm.clamp(max=1.0),
                max_prior,
                min_prior,
                imbalance,
            ],
            dim=-1,
        ).to(dtype=dtype)
        return support_features[:, None, :].expand(B, query_len, -1)

    def _confidence_features(self, layer_logits: Tensor) -> Tensor | None:
        if self.confidence_dim <= 0:
            return None
        logits = layer_logits.float()
        if self.confidence_source == "deepest":
            selected = logits[:, -1:]
        elif self.confidence_source == "all_layers":
            selected = logits
        else:
            return None
        probs = torch.softmax(selected, dim=-1)
        pmax = probs.max(dim=-1).values
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
        entropy = entropy / math.log(max(2, int(logits.shape[-1])))
        if self.confidence_source == "deepest":
            features = torch.stack([pmax[:, 0], entropy[:, 0]], dim=-1)
        else:
            features = torch.stack([pmax, entropy], dim=-1).permute(0, 2, 1, 3).flatten(start_dim=-2)
        return features.to(dtype=layer_logits.dtype)

    def forward(
        self,
        query_repr: Tensor,
        layer_logits: Tensor,
        y_train: Tensor,
    ) -> Tensor:
        """Return mixed logits with shape (B, Q, C)."""

        B, query_len, _ = query_repr.shape
        support_features = self._support_features(y_train, query_len, query_repr.dtype, query_repr.device)
        confidence_features = self._confidence_features(layer_logits)
        gate_parts = [self.norm(query_repr), support_features]
        if confidence_features is not None:
            gate_parts.append(confidence_features.to(dtype=query_repr.dtype))
        gate_input = torch.cat(gate_parts, dim=-1)
        gate_logits = self.gate(gate_input)
        weights = torch.softmax(gate_logits.float() / self.temperature, dim=-1).to(dtype=layer_logits.dtype)
        if self.low_layer_floor > 0.0:
            low_mask = torch.tensor(
                [layer <= self.low_layer_max for layer in self.layers],
                device=weights.device,
                dtype=weights.dtype,
            )
            low_count = low_mask.sum().clamp_min(1.0)
            if bool((low_mask.sum() > 0).detach().item()):
                weights = weights * (1.0 - self.low_layer_floor) + low_mask * (self.low_layer_floor / low_count)

        mixed = (weights.permute(0, 2, 1).unsqueeze(-1) * layer_logits).sum(dim=1)
        entropy_for_reg = -(weights.float() * weights.float().clamp_min(1e-12).log()).sum(dim=-1)
        max_weight = weights.float().max(dim=-1).values
        entropy_floor = math.log(max(1, len(self.layers)))
        entropy_reg = (entropy_floor - entropy_for_reg).mean()
        max_weight_reg = (max_weight - self.max_weight_target).clamp_min(0.0).pow(2).mean()
        self.last_regularization = {
            "layer_gate_entropy_reg": entropy_reg,
            "layer_gate_max_weight_reg": max_weight_reg,
        }
        self.last_layer_logits = layer_logits
        with torch.no_grad():
            entropy = entropy_for_reg
            low_mask = torch.tensor(
                [layer <= self.low_layer_max for layer in self.layers],
                device=weights.device,
                dtype=torch.bool,
            )
            low_mass = weights[..., low_mask].sum(dim=-1) if bool(low_mask.any().item()) else torch.zeros_like(entropy)
            metrics: dict[str, Tensor] = {
                "layer_gate_entropy": entropy.mean().detach(),
                "layer_gate_low_mass": low_mass.mean().detach(),
                "layer_gate_max_weight": max_weight.mean().detach(),
                "layer_gate_entropy_reg": entropy_reg.detach(),
                "layer_gate_max_weight_reg": max_weight_reg.detach(),
            }
            mean_weights = weights.float().mean(dim=(0, 1))
            for idx, layer in enumerate(self.layers):
                metrics[f"layer_gate_weight_icl{layer}"] = mean_weights[idx].detach()
            self.last_metrics = metrics
        return mixed


class ICLearning(nn.Module):
    """Dataset-wise in-context learning.

    Parameters
    ----------
    out_dim : int
        Output dimension of the model.

    max_classes : int
        Determines the task type and output behavior:
        - If max_classes=0: The model performs regression using quantile prediction.
        - If max_classes>0: The model performs classification. This value specifies
          the number of classes the model supports natively. If the number of classes
          in the dataset exceeds this value, hierarchical classification is used.

    d_model : int
        Model dimension.

    num_blocks : int
        Number of blocks used in the ICL encoder.

    nhead : int
        Number of attention heads of the ICL encoder.

    dim_feedforward : int
        Dimension of the feedforward network of the ICL encoder.

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
        Type of scalable softmax to use in the ICL encoder.
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

    recompute : bool, default=False
        If True, uses gradient checkpointing to save memory at the cost of additional computation.
    """

    def __init__(
        self,
        max_classes: int,
        out_dim: int,
        d_model: int,
        num_blocks: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
        ssmax: Union[bool, str] = False,
        recompute: bool = False,
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
        attention_gate_shape: str = "scalar",
        attention_gate_layers: str | Sequence[int] = "8;9;10;11",
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

        self.max_classes = max_classes
        self.norm_first = norm_first

        self.attention_gate_enabled = bool(attention_gate_enabled)
        self.attention_gate_layers = self._parse_layer_indices(attention_gate_layers)
        self.tf_icl = Encoder(
            num_blocks=num_blocks,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            bias_free_ln=bias_free_ln,
            ssmax=ssmax,
            recompute=recompute,
            attention_gate_enabled=self.attention_gate_enabled,
            attention_gate_shape=attention_gate_shape,
            attention_gate_layers=self.attention_gate_layers,
            attention_gate_context_dim=attention_gate_context_dim,
            attention_gate_rho=attention_gate_rho,
            swiglu_enabled=swiglu_enabled,
            swiglu_conditioned=swiglu_conditioned,
            swiglu_context_dim=swiglu_context_dim,
            swiglu_rho=swiglu_rho,
            swiglu_output_scale=swiglu_output_scale,
            swiglu_product_tanh_rms_multiple=swiglu_product_tanh_rms_multiple,
            swiglu_product_tanh_last_n_layers=swiglu_product_tanh_last_n_layers,
            swiglu_init_seed_base=swiglu_init_seed_base,
            reference_dim_feedforward=reference_dim_feedforward,
            qk_pds_enabled=qk_pds_enabled,
            shared_depth_enabled=shared_depth_enabled,
            shared_depth_rho=shared_depth_rho,
            shared_depth_dataset_conditioned=shared_depth_dataset_conditioned,
            shared_depth_num_passes=shared_depth_num_passes,
            shared_depth_context_dim=shared_depth_context_dim,
        )
        if self.norm_first:
            self.ln = nn.LayerNorm(d_model, bias=not bias_free_ln)

        if max_classes > 0:  # Classification
            self.y_encoder = OneHotAndLinear(max_classes, d_model)
        else:  # Regression
            self.y_encoder = nn.Linear(1, d_model)

        self.decoder = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, out_dim))
        self.inference_mgr = InferenceManager(enc_name="tf_icl", out_dim=out_dim)
        self.layer_gate_enabled = bool(layer_gate_enabled)
        self.layer_gate_layers = self._parse_layer_indices(layer_gate_layers)
        self.layer_gate_repr_source = str(layer_gate_repr_source or "deepest").lower()
        if self.layer_gate_repr_source not in {"deepest", "multi_mean", "multi_delta"}:
            raise ValueError(f"Unsupported layer_gate_repr_source={self.layer_gate_repr_source!r}")
        self.layer_gate: Optional[ICLLayerLogitGate]
        if self.layer_gate_enabled:
            if max_classes <= 0:
                raise ValueError("layer_gate_enabled currently supports classification only.")
            valid_layers = [layer for layer in self.layer_gate_layers if 0 <= layer < num_blocks]
            if not valid_layers:
                raise ValueError(
                    f"layer_gate_layers={self.layer_gate_layers} has no valid layers for num_blocks={num_blocks}."
                )
            self.layer_gate_layers = valid_layers
            gate_repr_dim = d_model
            if self.layer_gate_repr_source == "multi_delta":
                gate_repr_dim = d_model * 3
            self.layer_gate = ICLLayerLogitGate(
                d_model=d_model,
                layers=self.layer_gate_layers,
                max_classes=max_classes,
                hidden_dim=layer_gate_hidden_dim,
                gate_repr_dim=gate_repr_dim,
                temperature=layer_gate_temperature,
                low_layer_floor=layer_gate_low_layer_floor,
                low_layer_max=layer_gate_low_layer_max,
                use_confidence_features=layer_gate_use_confidence_features,
                confidence_source=layer_gate_confidence_source,
                max_weight_target=layer_gate_max_weight_target,
                bias_free_ln=bias_free_ln,
            )
        else:
            self.layer_gate = None

    @staticmethod
    def _parse_layer_indices(value: str | Sequence[int]) -> list[int]:
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

    @property
    def layer_gate_metrics(self) -> dict[str, Tensor]:
        if self.layer_gate is None:
            return {}
        return self.layer_gate.last_metrics

    @property
    def layer_gate_regularization(self) -> dict[str, Tensor]:
        if self.layer_gate is None:
            return {}
        return self.layer_gate.last_regularization

    @property
    def layer_gate_last_layer_logits(self) -> Tensor | None:
        if self.layer_gate is None:
            return None
        return self.layer_gate.last_layer_logits

    def _build_layer_gate_repr(self, layer_reprs: list[Tensor]) -> Tensor:
        if not layer_reprs:
            raise ValueError("Layer gate needs at least one representation")
        if self.layer_gate_repr_source == "deepest":
            return layer_reprs[-1]
        if self.layer_gate_repr_source == "multi_mean":
            return torch.stack(layer_reprs, dim=0).mean(dim=0)
        low_layers = [
            repr_tensor
            for repr_tensor, layer in zip(layer_reprs, self.layer_gate_layers)
            if layer <= getattr(self.layer_gate, "low_layer_max", 6)
        ]
        low_mean = torch.stack(low_layers or layer_reprs, dim=0).mean(dim=0)
        deep = layer_reprs[-1]
        return torch.cat([low_mean, deep, deep - low_mean], dim=-1)

    def _grouping(self, num_classes: int) -> tuple[Tensor, int]:
        """Divide classes into balanced groups for hierarchical classification.

        This method implements a balanced partitioning strategy that divides classes
        into approximately equal-sized groups to minimize tree depth. The number of
        groups formed at this level will not exceed `max_classes`.

        Parameters
        ----------
        num_classes : int
            Total number of unique classes to partition into groups.

        Returns
        -------
        group_assignments : Tensor
            Tensor mapping each class index to its assigned group (0-indexed).

        num_groups : int
            Total number of groups created (will be <= max_classes).

        Notes
        -----
        For example, with max_classes=10 and num_classes=25:
        - Distributes 25 classes into 3 groups. Sizes: [9, 8, 8].
        - Returns assignments tensor and num_groups = 3.

        With max_classes=10 and num_classes=101:
        - Distributes 101 classes into 10 groups. Sizes: [11, 10, 10, 10, 10, 10, 10, 10, 10, 10].
        - Returns assignments tensor and num_groups = 10.
        - The child node receiving 11 classes will be further divided into 2 groups: [6, 5].
        """

        if num_classes <= self.max_classes:
            return torch.zeros(num_classes, dtype=torch.int), 1

        num_groups = min(math.ceil(num_classes / self.max_classes), self.max_classes)
        group_assignments = torch.zeros(num_classes, dtype=torch.int)
        current_pos = 0

        remaining_classes = num_classes
        remaining_groups = num_groups
        for i in range(num_groups):
            group_size = math.ceil(remaining_classes / remaining_groups)
            group_assignments[current_pos : current_pos + group_size] = i
            current_pos += group_size
            remaining_classes -= group_size
            remaining_groups -= 1

        return group_assignments, num_groups

    def _fit_node(self, node: ClassNode, R: Tensor, y: Tensor, current_depth: int):
        """Recursively build a node in the hierarchical classification tree.

        For each node, this method either:

        1. Creates a leaf node if the number of classes is small enough to handle directly
        2. Splits classes into groups and recursively creates child nodes for each group

        Parameters
        ----------
        node : ClassNode
            Current node being constructed in the tree.

        R : Tensor
            Row representations of shape (num_samples, D) where num_samples is the number of
            examples assigned to this node.

        y : Tensor
            Targets of shape (num_samples,) corresponding to the samples in R.

        current_depth : int
            Current depth in the hierarchical tree (root = 0).
        """

        unique_classes = torch.unique(y).int()
        node.classes_ = unique_classes

        if len(unique_classes) <= self.max_classes:
            # Create leaf node for direct classification
            node.is_leaf = True
            node.R = R
            node.y = y
            return

        # Merge classes into groups
        group_assignments, num_groups = self._grouping(len(unique_classes))

        # Create mapping from original class labels to their corresponding group numbers
        node.class_mapping = {c.item(): g.item() for c, g in zip(unique_classes, group_assignments)}
        node.group_indices = torch.tensor([node.class_mapping[c.item()] for c in y], dtype=torch.int)
        node.R = R
        node.y = y
        node.is_leaf = False

        # Create child nodes for each group
        for group in range(num_groups):
            mask = node.group_indices == group
            child_node = ClassNode(current_depth + 1)
            self._fit_node(child_node, R[mask], y[mask], current_depth + 1)
            node.child_nodes.append(child_node)

    def _fit_hierarchical(self, R_train: Tensor, y_train: Tensor):
        """Initialize the hierarchical classification tree.

        Parameters
        ----------
        R_train : Tensor
            Row representations of training data of shape (train_size, D).

        y_train : Tensor
            Training targets of shape (train_size,).
        """

        self.root = ClassNode(depth=0)
        self._fit_node(self.root, R_train, y_train, current_depth=0)

    def _label_encoding(self, y: Tensor) -> Tensor:
        """Remapping target values to contiguous integers starting from 0."""

        unique_vals, _ = torch.unique(y, return_inverse=True)
        indices = unique_vals.argsort()
        return indices[torch.searchsorted(unique_vals, y)]

    def _icl_predictions(
        self,
        R: Tensor,
        y_train: Tensor,
        decode_test_only: bool = False,
        return_intermediate_layers: Optional[Sequence[int]] = None,
        dataset_context: Optional[Tensor] = None,
    ) -> Tensor | tuple[Tensor, dict[int, Tensor]]:
        """In-context learning predictions.

        Parameters
        ----------
        R : Tensor
            Row representations of shape (B, T, D) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - D is the dimension of row representations

        y_train : Tensor
            Training targets of shape (B, train_size), where train_size is the position
            to split the input into training and test data.

        Returns
        -------
        Tensor
            Predictions of shape (B, T, out_dim):

            - For regression (max_classes=0): out_dim = num_quantiles
            - For classification (max_classes>0): out_dim = max_classes
        """

        train_size = y_train.shape[1]
        if self.max_classes > 0:  # Classification
            Ry_train = self.y_encoder(y_train.float())
        else:  # Regression
            Ry_train = self.y_encoder(y_train.unsqueeze(-1))
        R[:, :train_size] = R[:, :train_size] + Ry_train

        should_return_intermediates = return_intermediate_layers is not None
        requested_intermediate_layers = set(int(x) for x in (return_intermediate_layers or ()))
        gate_layers = set(self.layer_gate_layers) if self.layer_gate_enabled else set()
        effective_intermediate_layers = requested_intermediate_layers | gate_layers
        if effective_intermediate_layers:
            src, intermediates = self.tf_icl(
                R,
                train_size=train_size,
                return_intermediate_layers=effective_intermediate_layers,
                dataset_context=dataset_context,
            )
        else:
            src = self.tf_icl(R, train_size=train_size, dataset_context=dataset_context)
            intermediates = None
        decode_start = train_size if decode_test_only else 0
        if self.layer_gate_enabled and self.layer_gate is not None and intermediates is not None:
            decoded_layers: list[Tensor] = []
            gate_reprs: list[Tensor] = []
            for layer_idx in self.layer_gate_layers:
                layer_src = src if layer_idx == len(self.tf_icl.blocks) - 1 else intermediates[layer_idx]
                layer_src = layer_src[:, decode_start:]
                if self.norm_first:
                    layer_src = self.ln(layer_src)
                gate_reprs.append(layer_src)
                decoded_layers.append(self.decoder(layer_src))
            layer_logits = torch.stack(decoded_layers, dim=1)
            gate_query_repr = self._build_layer_gate_repr(gate_reprs)
            out = self.layer_gate(gate_query_repr, layer_logits, y_train)
        else:
            if decode_test_only:
                src = src[:, train_size:]
            if self.norm_first:
                src = self.ln(src)
            out = self.decoder(src)

        if should_return_intermediates and intermediates is not None:
            returned_intermediates = {
                layer: tensor for layer, tensor in intermediates.items() if layer in requested_intermediate_layers
            }
            return out, returned_intermediates
        return out

    def _predict_standard(
        self,
        R: Tensor,
        y_train: Tensor,
        return_logits: bool = False,
        softmax_temperature: float = 0.9,
        auto_batch: bool = True,
        dataset_context: Optional[Tensor] = None,
    ) -> Tensor:
        """Generate predictions for standard classification with up to `max_classes` classes.

        Parameters
        ----------
        R : Tensor
            Row representations of shape (B, T, D) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - D is the dimension of row representations

        y_train : Tensor
            Training targets of shape (B, train_size), where train_size is the position
            to split the input into training and test data.

        return_logits : bool, default=False
            If True, return logits instead of probabilities.

        softmax_temperature : float, default=0.9
            Temperature for the softmax function.

        auto_batch : bool, default=True
            Whether to use InferenceManager to automatically split inputs into smaller batches.

        Returns
        -------
        Tensor
            For regression (max_classes=0):
                Predictions of shape (B, test_size, num_quantiles), where test_size = T - train_size

            For classification (max_classes>0):
                If return_logits=True: Logits of shape (B, test_size, num_classes)
                If return_logits=False: Probabilities of shape (B, test_size, num_classes)
        """

        inputs = OrderedDict([("R", R), ("y_train", y_train), ("dataset_context", dataset_context)])
        out = self.inference_mgr(self._icl_predictions, inputs=inputs, auto_batch=auto_batch)

        train_size = y_train.shape[1]
        if self.max_classes == 0:
            out = out[:, train_size:]
        else:
            num_classes = len(torch.unique(y_train[0]))
            out = out[:, train_size:, :num_classes]
            if not return_logits:
                out = torch.softmax(out / softmax_temperature, dim=-1)

        return out

    def _predict_hierarchical(
        self, R_test: Tensor, dataset_context: Optional[Tensor] = None,
        softmax_temperature: float = 0.9, inference_recurrence: Optional[int] = None
    ) -> Tensor:
        """Generate predictions using the hierarchical classification tree.

        This method traverses the tree from leaves to root, computing probabilities at each level
        and combining them according to the probability chain rule.

        Parameters
        ----------
        R_test : Tensor
            Row representations of test data of shape (test_size, D).

        softmax_temperature : float, default=0.9
            Temperature for the softmax function.

        Returns
        -------
        Tensor
            Probability over all classes, shape (test_size, C).
        """

        test_size = R_test.shape[0]
        device = R_test.device
        num_classes = len(self.root.classes_)

        def process_node(node, R_test):
            """Recursively process a node in the hierarchical tree.

            For leaf nodes: Directly predict class probabilities within the node's subset
            For internal nodes: Combine predictions from child nodes weighted by group probabilities
            """

            # Concatenate test data with node data
            node_R = torch.cat([node.R.to(device), R_test], dim=0)

            # Case 1: Leaf node - direct classification
            if node.is_leaf:
                node_y = self._label_encoding(node.y.to(device))
                # Get predictions for this leaf
                leaf_preds = self._predict_standard(
                    R=node_R.unsqueeze(0),
                    y_train=node_y.unsqueeze(0),
                    softmax_temperature=softmax_temperature,
                    auto_batch=False,
                    dataset_context=dataset_context,
                ).squeeze(0)
                # Map leaf predictions to the global class space
                global_preds = torch.zeros((test_size, num_classes), device=device)
                for local_idx, global_idx in enumerate(node.classes_):
                    global_preds[:, global_idx] = leaf_preds[:, local_idx]

                return global_preds

            # Case 2: Internal node - classification into groups
            # Initialize output tensor for all classes
            final_probs = torch.zeros((test_size, num_classes), device=device)

            # Get group probabilities for this node
            node_y = node.group_indices.to(device)
            group_probs = self._predict_standard(
                R=node_R.unsqueeze(0),
                y_train=node_y.unsqueeze(0),
                softmax_temperature=softmax_temperature,
                auto_batch=False,
                dataset_context=dataset_context,
            ).squeeze(0)

            # Recursively process child nodes and combine predictions
            for group_idx, child_node in enumerate(node.child_nodes):
                child_probs = process_node(child_node, R_test)
                final_probs += child_probs * group_probs[:, group_idx : group_idx + 1]

            return final_probs

        return process_node(self.root, R_test)

    def _inference_forward(
        self,
        R: Tensor,
        y_train: Tensor,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        mgr_config: MgrConfig = None,
        dataset_context: Optional[Tensor] = None,
    ) -> Tensor:
        """In-context learning based on learned row representations for inference.

        Parameters
        ----------
        R : Tensor
            Row representations of shape (B, T, D) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - D is the dimension of row representations

        y_train : Tensor
            Training targets of shape (B, train_size), where train_size is the position
            to split the input into training and test data.

        return_logits : bool, default=True
            If True, return logits instead of probabilities.

        softmax_temperature : float, default=0.9
            Temperature for the softmax function.

        mgr_config : MgrConfig, default=None
            Configuration for InferenceManager.

        Returns
        -------
        Tensor
            For regression (max_classes=0):
                Predictions of shape (B, test_size, num_quantiles), where test_size = T - train_size

            For classification (max_classes>0):
                If return_logits=True: Logits of shape (B, test_size, num_classes)
                If return_logits=False: Probabilities of shape (B, test_size, num_classes)
        """
        # Configure inference parameters
        if mgr_config is None:
            mgr_config = InferenceConfig().ICL_CONFIG
        self.inference_mgr.configure(**mgr_config)

        if self.max_classes == 0:  # Regression
            out = self._predict_standard(R, y_train, dataset_context=dataset_context)
        else:  # Classification
            num_classes = len(torch.unique(y_train[0]))
            assert all(
                len(torch.unique(yi)) == num_classes for yi in y_train
            ), "All tables must have the same number of classes"

            if num_classes <= self.max_classes:
                # Standard classification
                out = self._predict_standard(
                    R, y_train, return_logits=return_logits, softmax_temperature=softmax_temperature,
                    dataset_context=dataset_context
                )
            else:
                # Hierarchical classification
                out = []
                train_size = y_train.shape[1]
                contexts = dataset_context if dataset_context is not None else [None] * len(R)
                for ri, yi, context in zip(R, y_train, contexts):
                    if mgr_config.offload:
                        ri, yi = ri.cpu(), yi.cpu()
                    else:
                        ri, yi = ri.to(mgr_config.device), yi.to(mgr_config.device)
                    self._fit_hierarchical(ri[:train_size], yi)
                    node_context = None if context is None else context.unsqueeze(0).to(ri.device)
                    probs = self._predict_hierarchical(ri[train_size:], dataset_context=node_context)
                    out.append(probs)
                out = torch.stack(out, dim=0)
                if return_logits:
                    out = softmax_temperature * torch.log(out + 1e-6)

        return out

    def forward(
        self,
        R: Tensor,
        y_train: Tensor,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        mgr_config: MgrConfig = None,
        return_intermediate_layers: Optional[Sequence[int]] = None,
        dataset_context: Optional[Tensor] = None,
    ) -> Tensor:
        """In-context learning based on learned row representations.

        Parameters
        ----------
        R : Tensor
            Row representations of shape (B, T, D) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - D is the dimension of row representations

        y_train : Tensor
            Training targets of shape (B, train_size), where train_size is the position
            to split the input into training and test data.

        return_logits : bool, default=True
            If True, return logits instead of probabilities. Used only in inference mode.

        softmax_temperature : float, default=0.9
            Temperature for the softmax function. Used only in inference mode.

        mgr_config : MgrConfig, default=None
            Configuration for InferenceManager. Used only in inference mode.

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
            out = self._icl_predictions(
                R,
                y_train,
                decode_test_only=True,
                return_intermediate_layers=return_intermediate_layers,
                dataset_context=dataset_context,
            )
        else:
            if return_intermediate_layers is not None:
                out = self._icl_predictions(
                    R,
                    y_train,
                    decode_test_only=True,
                    return_intermediate_layers=return_intermediate_layers,
                    dataset_context=dataset_context,
                )
            else:
                out = self._inference_forward(
                    R, y_train, return_logits, softmax_temperature, mgr_config, dataset_context=dataset_context
                )

        return out

    def prepare_repr_cache(self, R: Tensor, y_train: Tensor) -> Tensor:
        """Add target embedding to train representations.

        Parameters
        ----------
        R : Tensor
            Row representations of shape (B, T, D) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - D is the dimension of row representations

        y_train : Tensor
            Training targets of shape (B, train_size), where train_size is the position
            to split the input into training and test data.

        Returns
        -------
        Tensor
            Full representations with y_train baked into the train portion,
            shape (B, T, D).
        """

        train_size = y_train.shape[1]
        if self.max_classes > 0:
            Ry_train = self.y_encoder(y_train.float())
        else:
            Ry_train = self.y_encoder(y_train.unsqueeze(-1))
        R[:, :train_size] = R[:, :train_size] + Ry_train

        return R

    def _icl_predictions_repr_cache(
        self, R: Tensor, train_size: int, dataset_context: Optional[Tensor] = None
    ) -> Tensor:
        """In-context learning predictions with representation cache.

        This method does not add target embedding because it is already
        baked into the cached train representations.

        Parameters
        ----------
        R : Tensor
            Full representations of shape (B, T, D) where
            R[:, :train_size] has y_train already baked in.

        train_size : int
            Number of training samples.

        Returns
        -------
        Tensor
            Predictions of shape (B, T, out_dim).
        """

        src = self.tf_icl(R, train_size=train_size, dataset_context=dataset_context)
        if self.norm_first:
            src = self.ln(src)
        out = self.decoder(src)

        return out

    def forward_with_repr_cache(
        self,
        R: Tensor,
        train_size: int,
        num_classes: Optional[int] = None,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        mgr_config: MgrConfig = None,
        dataset_context: Optional[Tensor] = None,
    ) -> Tensor:
        """In-context learning with representation cache.

        Runs the ICL transformer on pre-assembled representations where
        the training portion already has y_train baked in.

        Parameters
        ----------
        R : Tensor
            Full representations of shape (B, T, D) where
            R[:, :train_size] has y_train already baked in.

        train_size : int
            Number of training samples.

        num_classes : Optional[int], default=None
            Number of classes for classification tasks.

        return_logits : bool, default=True
            If True, return raw logits instead of probabilities.

        softmax_temperature : float, default=0.9
            Temperature for the softmax function.

        mgr_config : MgrConfig, default=None
            Configuration for InferenceManager. If None, uses the default
            ICL_CONFIG from InferenceConfig.

        Returns
        -------
        Tensor
            For regression (max_classes=0):
                Predictions of shape (B, test_size, num_quantiles)

            For classification (max_classes>0):
                If return_logits=True: Logits of shape (B, test_size, num_classes)
                If return_logits=False: Probabilities of shape (B, test_size, num_classes)
        """

        if mgr_config is None:
            mgr_config = InferenceConfig().ICL_CONFIG
        self.inference_mgr.configure(**mgr_config)

        out = self.inference_mgr(
            self._icl_predictions_repr_cache,
            inputs=OrderedDict(
                [("R", R), ("train_size", train_size), ("dataset_context", dataset_context)]
            ),
        )

        out = out[:, train_size:]
        if self.max_classes > 0:
            assert num_classes is not None, "num_classes must be provided for classification"
            out = out[..., :num_classes]
            if not return_logits:
                out = torch.softmax(out / softmax_temperature, dim=-1)

        return out

    def _icl_predictions_with_cache(
        self,
        R: Tensor,
        icl_cache: KVCache,
        y_train: Optional[Tensor] = None,
        use_cache: bool = False,
        store_cache: bool = True,
        dataset_context: Optional[Tensor] = None,
    ) -> Tensor:
        """In-context learning predictions with KV caching.

        Parameters
        ----------
        R : Tensor
            Row representations of shape (B, T, D).

        icl_cache : KVCache
            Cache object for storing/retrieving K/V projections.

        y_train : Optional[Tensor], default=None
            Training targets of shape (B, train_size). Required when store_cache=True;
            ignored when use_cache=True.

        use_cache : bool, default=False
            Whether to use cached values to avoid redundant computation.

        store_cache : bool, default=True
            Whether to store computed values in cache.

        Returns
        -------
        Tensor
            Predictions of shape (B, T, out_dim) or (B, test_size, out_dim) when use_cache=True:

            - For regression (max_classes=0): out_dim = num_quantiles
            - For classification (max_classes>0): out_dim = max_classes
        """
        # When using cache, skip y_train embedding — it's already baked
        # into the cached K/V projections from the store_cache pass.
        if store_cache:
            assert y_train is not None, "y_train must be provided when store_cache=True"
            train_size = y_train.shape[1]

            if self.max_classes > 0:  # Classification
                Ry_train = self.y_encoder(y_train.float())
            else:  # Regression
                Ry_train = self.y_encoder(y_train.unsqueeze(-1))
            R[:, :train_size] = R[:, :train_size] + Ry_train

        src = self.tf_icl.forward_with_cache(
            R,
            icl_cache=icl_cache,
            train_size=train_size if store_cache else None,
            use_cache=use_cache,
            store_cache=store_cache,
            dataset_context=dataset_context,
        )
        if self.norm_first:
            src = self.ln(src)
        out = self.decoder(src)

        return out

    def forward_with_cache(
        self,
        R: Tensor,
        icl_cache: KVCache,
        y_train: Optional[Tensor] = None,
        num_classes: Optional[int] = None,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        use_cache: bool = False,
        store_cache: bool = True,
        mgr_config: MgrConfig = None,
        dataset_context: Optional[Tensor] = None,
    ) -> Tensor:
        """In-context learning with KV caching support.

        Parameters
        ----------
        R : Tensor
            Row representations of shape (B, T, D).

        icl_cache : KVCache
            Cache object for storing/retrieving K/V projections.

        y_train : Optional[Tensor], default=None
            Training targets of shape (B, train_size). Required when store_cache=True;
            ignored when use_cache=True.

        num_classes : Optional[int], default=None
            Number of classes for classification. If None, computed from y_train.
            When use_cache=True, this should be provided from the cache.

        return_logits : bool, default=True
            If True, return logits instead of probabilities.

        softmax_temperature : float, default=0.9
            Temperature for the softmax function.

        use_cache : bool, default=False
            Whether to use cached values to avoid redundant computation.

        store_cache : bool, default=True
            Whether to store computed values in cache.

        mgr_config : MgrConfig, default=None
            Configuration for InferenceManager. If None, uses the default
            ICL_CONFIG from InferenceConfig.

        Returns
        -------
        Tensor
            For regression (max_classes=0):
                Predictions of shape (B, test_size, num_quantiles)

            For classification (max_classes>0):
                If return_logits=True: Logits of shape (B, test_size, num_classes)
                If return_logits=False: Probabilities of shape (B, test_size, num_classes)
        """

        if use_cache == store_cache:
            raise ValueError("Exactly one of use_cache or store_cache must be True")

        if store_cache:
            assert y_train is not None, "y_train must be provided when store_cache=True"
            # many-class classification is not supported with caching
            if self.max_classes > 0:
                num_classes = len(torch.unique(y_train[0]))
                if num_classes > self.max_classes:
                    raise ValueError(
                        f"KV caching is not supported for classification with more classes "
                        f"({num_classes}) than max_classes ({self.max_classes}). Hierarchical classification "
                        f"requires multiple forward passes which is incompatible with caching."
                    )
        else:
            assert num_classes is not None, "num_classes must be provided when use_cache=True"

        if mgr_config is None:
            mgr_config = InferenceConfig().ICL_CONFIG
        self.inference_mgr.configure(**mgr_config)

        out = self.inference_mgr(
            self._icl_predictions_with_cache,
            inputs=OrderedDict(
                [
                    ("R", R),
                    ("icl_cache", icl_cache),
                    ("y_train", y_train),
                    ("use_cache", use_cache),
                    ("store_cache", store_cache),
                    ("dataset_context", dataset_context),
                ]
            ),
        )

        if store_cache:
            train_size = y_train.shape[1]
            out = out[:, train_size:]

        if self.max_classes > 0:
            out = out[..., :num_classes]
            if not return_logits:
                out = torch.softmax(out / softmax_temperature, dim=-1)

        return out
