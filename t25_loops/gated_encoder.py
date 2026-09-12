"""G5SC recurrence, isolated from the frozen T25 column/row encoders."""
from functools import partial

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .encoders import Encoder
from .kv_cache import KVCacheEntry
from .g5sc_support_stats import SupportSchemaStatistics


class GatedLoopEncoder(Encoder):
    def __init__(self, *args, shared_depth_num_passes=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_depth_num_passes = int(shared_depth_num_passes)
        if self.shared_depth_num_passes not in (1, 3, 4):
            raise ValueError("T25 experiment requires 1, 3 or 4 total passes")
        if self.shared_depth_num_passes > 1:
            self.shared_depth_gate = nn.Parameter(torch.zeros(()))
            self.shared_depth_condition_weight = nn.Parameter(torch.zeros(51))
        else:
            self.register_parameter("shared_depth_gate", None)
            self.register_parameter("shared_depth_condition_weight", None)

    def _shared_depth_alpha(self, dataset_context, reference):
        if dataset_context is None:
            raise ValueError("support-only dataset_context is required")
        weight = self.shared_depth_condition_weight
        context = dataset_context.to(device=weight.device, dtype=weight.dtype)
        if context.shape != (reference.shape[0], 51):
            raise ValueError(f"invalid support context shape: {context.shape}")
        support_logit = (context * weight).sum(dim=-1)
        bounded = 0.1 * (2.0 * torch.sigmoid(support_logit) - 1.0)
        return torch.tanh(self.shared_depth_gate + bounded).view(-1, 1, 1).to(reference.dtype)

    def forward(self, src, train_size=None, dataset_context=None):
        # The first pass is exactly the original T25 implementation.
        out = super().forward(src, train_size=train_size)
        for _ in range(1, self.shared_depth_num_passes):
            previous = out
            for block in self.blocks:
                # Added passes use recomputation only, preserving the function,
                # RNG and microbatch while bounding their activation memory.
                if self.training and torch.is_grad_enabled():
                    out = checkpoint(partial(block, train_size=train_size, rope=self.rope),
                                     out, use_reentrant=False)
                else:
                    out = block(q=out, train_size=train_size, rope=self.rope)
            out = previous + self._shared_depth_alpha(dataset_context, out) * (out - previous)
        return out

    def forward_with_cache(self, src, icl_cache, train_size=None,
                           use_cache=False, store_cache=True, dataset_context=None):
        if use_cache == store_cache:
            raise ValueError("Exactly one cache mode must be true")
        if store_cache and train_size is None:
            raise ValueError("train_size required for cache prefill")
        out = src
        for pass_idx in range(self.shared_depth_num_passes):
            previous = out
            for layer_idx, block in enumerate(self.blocks):
                key = pass_idx * len(self.blocks) + layer_idx
                if use_cache:
                    out = block(q=out, rope=self.rope, cached_kv=icl_cache.kv[key])
                else:
                    out, k, v = block(q=out, train_size=train_size, rope=self.rope, need_kv=True)
                    icl_cache.kv[key] = KVCacheEntry(key=k, value=v)
            if pass_idx:
                out = previous + self._shared_depth_alpha(dataset_context, out) * (out - previous)
        return out
