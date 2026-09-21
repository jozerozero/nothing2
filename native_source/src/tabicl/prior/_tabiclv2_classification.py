from __future__ import annotations

import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import expit
from torch import Tensor
from torch.nested import nested_tensor


def _as_float_tensor(x: np.ndarray | Tensor, device: str | torch.device) -> Tensor:
    if torch.is_tensor(x):
        return x.to(device=device, dtype=torch.float32)
    return torch.tensor(x, dtype=torch.float32, device=device)


def _safe_std(x: Tensor, dim: int = 0, keepdim: bool = False) -> Tensor:
    std = torch.std(x, dim=dim, keepdim=keepdim, unbiased=False)
    return torch.nan_to_num(std, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-6)


def _standardize(x: Tensor) -> Tensor:
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return (x - x.mean(dim=0, keepdim=True)) / _safe_std(x, dim=0, keepdim=True)


def _remove_outliers(x: Tensor, threshold: float = 4.0) -> Tensor:
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    mean = x.mean(dim=0, keepdim=True)
    std = _safe_std(x, dim=0, keepdim=True)
    return x.clamp(mean - threshold * std, mean + threshold * std)


def _ordinal_encode(y: Tensor) -> Tensor:
    unique = torch.unique(y)
    unique, _ = torch.sort(unique)
    return torch.searchsorted(unique, y).long()


def _pad_features(x: Tensor, max_features: int) -> Tensor:
    if x.shape[1] > max_features:
        return x[:, :max_features]
    if x.shape[1] < max_features:
        return F.pad(x, (0, max_features - x.shape[1]), value=0.0)
    return x


def _parse_bin_weights(
    weights: Optional[str | Sequence[float] | np.ndarray],
    expected: int,
    name: str,
) -> Optional[np.ndarray]:
    if weights is None:
        return None
    if isinstance(weights, str):
        text = weights.strip()
        if not text:
            return None
        values = [float(part.strip()) for part in re.split(r"[,;:\s]+", text) if part.strip()]
    else:
        values = [float(value) for value in weights]
    if len(values) != expected:
        raise ValueError(f"{name} must contain exactly {expected} comma-separated weights; got {len(values)}")
    arr = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite weights")
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if total <= 0.0:
        raise ValueError(f"{name} must have positive total weight")
    return arr / total


def _blend_bin_weights(
    base: Optional[np.ndarray],
    dynamic: Optional[np.ndarray],
    blend: float,
) -> Optional[np.ndarray]:
    if dynamic is None:
        return base
    if base is None:
        return dynamic
    alpha = float(np.clip(blend, 0.0, 1.0))
    arr = (1.0 - alpha) * base + alpha * dynamic
    total = float(arr.sum())
    return arr / total if total > 0.0 else base


def _sample_int_from_weighted_bins(
    bins: Sequence[Tuple[int, int]],
    weights: Optional[np.ndarray],
    low_limit: int,
    high_limit: int,
) -> int:
    low_limit = int(low_limit)
    high_limit = int(high_limit)
    if high_limit <= low_limit:
        return low_limit
    if weights is None:
        return int(np.random.randint(low_limit, high_limit + 1))

    valid_bins: List[Tuple[int, int]] = []
    valid_weights: List[float] = []
    for (lo, hi), weight in zip(bins, weights):
        lo = max(int(lo), low_limit)
        hi = min(int(hi), high_limit)
        if lo <= hi and float(weight) > 0.0:
            valid_bins.append((lo, hi))
            valid_weights.append(float(weight))
    if not valid_bins:
        return int(np.random.randint(low_limit, high_limit + 1))
    probs = np.asarray(valid_weights, dtype=np.float64)
    probs = probs / probs.sum()
    idx = int(np.random.choice(len(valid_bins), p=probs))
    lo, hi = valid_bins[idx]
    return int(np.random.randint(lo, hi + 1))


def _sample_cat_ratio_from_weighted_bins(weights: np.ndarray) -> float:
    idx = int(np.random.choice(len(weights), p=weights))
    if idx == 0:
        return 0.0
    if idx == 1:
        return float(np.random.uniform(0.01, 0.25))
    if idx == 2:
        return float(np.random.uniform(0.25, 0.75))
    if idx == 3:
        return float(np.random.uniform(0.75, 0.99))
    return 1.0


class CorrelatedScalarSampler:
    """Sample scalar hyperparameters with per-name correlated distributions.

    The sampler follows Appendix E.2: all numeric values with the same name share
    beta-distribution parameters, and categorical values with the same name share
    a sampled weight vector.
    """

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng(
            int(np.random.randint(0, 2**32 - 1))
        )
        self._numeric: Dict[str, Tuple[float, float]] = {}
        self._categorical: Dict[Tuple[str, int], np.ndarray] = {}

    def _beta_params(self, name: str) -> Tuple[float, float]:
        if name not in self._numeric:
            t = self.rng.uniform(0.0, 1.0)
            s = math.exp(self.rng.uniform(math.log(0.1), math.log(10000.0)))
            self._numeric[name] = (max(s * t, 1e-6), max(s * (1.0 - t), 1e-6))
        return self._numeric[name]

    def unit(self, name: str) -> float:
        alpha, beta = self._beta_params(name)
        return float(self.rng.beta(alpha, beta))

    def num(self, name: str, low: float, high: float) -> float:
        return low + (high - low) * self.unit(name)

    def integer(self, name: str, low: int, high: int) -> int:
        if high <= low:
            return int(low)
        value = low + (high + 1 - low) * self.unit(name)
        return int(np.clip(math.floor(value), low, high))

    def lognum(self, name: str, low: float, high: float) -> float:
        low = max(low, 1e-12)
        high = max(high, low * (1.0 + 1e-12))
        return math.exp(self.num(name, math.log(low), math.log(high)))

    def logint(self, name: str, low: int, high: int) -> int:
        if high <= low:
            return int(low)
        value = math.exp(self.num(name, math.log(max(low, 1)), math.log(max(high + 1, low + 1))))
        return int(np.clip(math.floor(value), low, high))

    def choice(self, name: str, values: Sequence[Any]) -> Any:
        if not values:
            raise ValueError("choice values must be non-empty")
        key = (name, len(values))
        if key not in self._categorical:
            self._categorical[key] = self.positive_weights(f"{name}:choice", len(values))
        idx = int(self.rng.choice(len(values), p=self._categorical[key]))
        return values[idx]

    def positive_weights(self, name: str, dim: int) -> np.ndarray:
        """Sample the Appendix E.11 random positive vector in numpy."""
        if dim <= 0:
            return np.empty(0, dtype=np.float64)
        q_low = 0.1 / math.log(dim + 1.0)
        q = self.lognum("categorical-choice-weights:q", q_low, 6.0)
        sigma = self.lognum("categorical-choice-weights:sigma", 1e-4, 10.0)
        ranks = np.arange(1, dim + 1, dtype=np.float64)
        noise = self.rng.normal(loc=0.0, scale=sigma, size=dim)
        w = np.power(ranks, -q) * np.exp(noise)
        w = w / max(float(w.sum()), 1e-12)
        return self.rng.permutation(w)


def random_weights(dim: int, sampler: CorrelatedScalarSampler, device: str | torch.device, name: str = "weights") -> Tensor:
    if dim <= 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    q_low = 0.1 / math.log(dim + 1.0)
    q = sampler.lognum(f"{name}:q", q_low, 6.0)
    sigma = sampler.lognum(f"{name}:sigma", 1e-4, 10.0)
    ranks = torch.arange(1, dim + 1, dtype=torch.float32, device=device)
    noise = torch.randn(dim, dtype=torch.float32, device=device) * sigma
    w = ranks.pow(-q) * torch.exp(noise)
    w = w / w.sum().clamp_min(1e-12)
    return w[torch.randperm(dim, device=device)]


def random_activation(x: Tensor, sampler: CorrelatedScalarSampler, rescale: bool = True) -> Tensor:
    if rescale:
        x = _standardize(x)
        scale = sampler.lognum("activation-scale", 1.0, 10.0)
        if x.shape[0] > 0:
            anchor = x[torch.randint(0, x.shape[0], (1,), device=x.device)]
        else:
            anchor = torch.zeros_like(x[:1])
        x = scale * (x - anchor)

    fixed = [
        "tanh",
        "leaky_relu",
        "elu",
        "identity",
        "selu",
        "silu",
        "relu",
        "softplus",
        "relu6",
        "hardtanh",
        "sign",
        "heaviside",
        "gaussian",
        "exp",
        "indicator01",
        "sin",
        "square",
        "abs",
        "softmax",
        "argmax_onehot",
        "argsort",
        "logsigmoid",
        "logabs",
        "rank",
        "sigmoid",
        "round",
        "mod1",
    ]
    parametric = ["relu_power", "signed_power", "inverse_power", "integer_power"]
    kind = sampler.choice("activation-kind", fixed) if random.random() < 2.0 / 3.0 else sampler.choice("param-activation-kind", parametric)

    if kind == "tanh":
        y = torch.tanh(x)
    elif kind == "leaky_relu":
        y = F.leaky_relu(x)
    elif kind == "elu":
        y = F.elu(x)
    elif kind == "identity":
        y = x
    elif kind == "selu":
        y = F.selu(x)
    elif kind == "silu":
        y = F.silu(x)
    elif kind == "relu":
        y = F.relu(x)
    elif kind == "softplus":
        y = F.softplus(x)
    elif kind == "relu6":
        y = F.relu6(x)
    elif kind == "hardtanh":
        y = F.hardtanh(x)
    elif kind == "sign":
        y = torch.sign(x)
    elif kind == "heaviside":
        y = (x > 0).float()
    elif kind == "gaussian":
        y = torch.exp(-x.square())
    elif kind == "exp":
        y = torch.exp(x.clamp(-8.0, 8.0))
    elif kind == "indicator01":
        y = ((x >= 0.0) & (x <= 1.0)).float()
    elif kind == "sin":
        y = torch.sin(x)
    elif kind == "square":
        y = x.square()
    elif kind == "abs":
        y = x.abs()
    elif kind == "softmax":
        y = torch.softmax(x, dim=-1)
    elif kind == "argmax_onehot":
        idx = x.argmax(dim=-1)
        y = F.one_hot(idx, num_classes=x.shape[-1]).float()
    elif kind == "argsort":
        y = torch.argsort(torch.argsort(x, dim=-1), dim=-1).float()
    elif kind == "logsigmoid":
        y = F.logsigmoid(x)
    elif kind == "logabs":
        y = torch.log(x.abs().clamp_min(1e-6))
    elif kind == "rank":
        y = torch.argsort(torch.argsort(x, dim=0), dim=0).float()
    elif kind == "sigmoid":
        y = torch.sigmoid(x)
    elif kind == "round":
        y = torch.round(x)
    elif kind == "mod1":
        y = torch.remainder(x, 1.0)
    elif kind == "relu_power":
        q = sampler.lognum("activation-power", 0.1, 10.0)
        y = F.relu(x).pow(q)
    elif kind == "signed_power":
        q = sampler.lognum("activation-power", 0.1, 10.0)
        y = torch.sign(x) * x.abs().pow(q)
    elif kind == "inverse_power":
        q = sampler.lognum("activation-power", 0.1, 10.0)
        y = (x.abs() + 1e-3).pow(-q).clamp(max=1e4)
    else:
        m = sampler.integer("activation-int-power", 2, 5)
        y = x.clamp(-10.0, 10.0).pow(m)

    y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return _standardize(y) if rescale else y


class RandomMatrixFactory:
    def __init__(self, sampler: CorrelatedScalarSampler, device: str | torch.device):
        self.sampler = sampler
        self.device = device

    def sample_kind(self, allow_activation: bool = True) -> str:
        return self.sampler.choice(
            "matrix-type",
            ["gaussian", "weights", "singular_values", "kernel", "activation"] if allow_activation else ["gaussian", "weights", "singular_values", "kernel"],
        )

    def __call__(self, rows: int, cols: int, allow_activation: bool = True, kind: Optional[str] = None) -> Tensor:
        kind = kind or self.sample_kind(allow_activation=allow_activation)
        if kind == "gaussian":
            mat = torch.randn(rows, cols, dtype=torch.float32, device=self.device)
        elif kind == "weights":
            w = torch.stack(
                [random_weights(cols, self.sampler, self.device, name="matrix-column-weights") for _ in range(rows)]
            )
            mat = torch.randn(rows, cols, dtype=torch.float32, device=self.device) * w
        elif kind == "singular_values":
            rank = min(rows, cols)
            u = torch.randn(rows, rank, dtype=torch.float32, device=self.device)
            v = torch.randn(cols, rank, dtype=torch.float32, device=self.device)
            w = random_weights(rank, self.sampler, self.device, name="matrix-singular-values")
            mat = (u * w.unsqueeze(0)) @ v.T
        elif kind == "kernel":
            p = torch.randn(rows + cols, 3, dtype=torch.float32, device=self.device)
            gamma = self.sampler.lognum("kernel-matrix-gamma", 0.1, 10.0)
            dist = torch.cdist(p[:rows], p[rows:])
            signs = torch.randint(0, 2, (rows, cols), device=self.device, dtype=torch.float32) * 2.0 - 1.0
            mat = torch.exp(-gamma * dist) * signs
        else:
            mat = self(rows, cols, allow_activation=False)
            mat = random_activation(mat.reshape(1, -1), self.sampler, rescale=False).reshape(rows, cols)
            mat = mat + 1e-3 * torch.randn_like(mat)

        mat = mat + 1e-6 * torch.randn_like(mat)
        return mat / mat.norm(dim=1, keepdim=True).clamp_min(1e-6)


class RandomFunctionFactory:
    def __init__(
        self,
        sampler: CorrelatedScalarSampler,
        device: str | torch.device,
        trace: Optional[List[Dict[str, Any]]] = None,
    ):
        self.sampler = sampler
        self.device = device
        self.matrix = RandomMatrixFactory(sampler, device)
        self.trace = trace

    def __call__(
        self,
        x: Tensor,
        out_dim: int,
        allow_product: bool = True,
        exclude: Optional[set[str]] = None,
    ) -> Tensor:
        options = ["nn", "tree", "discretization", "gp", "linear", "quadratic", "em"]
        if allow_product:
            options.append("product")
        if exclude:
            options = [option for option in options if option not in exclude]
        kind = self.sampler.choice("random-function-type", options)
        if self.trace is not None:
            self.trace.append(
                {
                    "event": "random_function",
                    "kind": kind,
                    "in_dim": int(x.shape[1]),
                    "out_dim": int(out_dim),
                    "allow_product": bool(allow_product),
                }
            )
        if kind == "nn":
            return self.nn(x, out_dim)
        if kind == "tree":
            return self.tree(x, out_dim)
        if kind == "discretization":
            return self.discretization(x, out_dim)
        if kind == "gp":
            return self.gp(x, out_dim)
        if kind == "linear":
            return self.linear(x, out_dim)
        if kind == "quadratic":
            return self.quadratic(x, out_dim)
        if kind == "em":
            return self.em_assignment(x, out_dim)
        f = self(x, out_dim, allow_product=False, exclude={"nn", "em"})
        g = self(x, out_dim, allow_product=False, exclude={"nn", "em"})
        return _standardize(f * g)

    def linear(self, x: Tensor, out_dim: int) -> Tensor:
        return _standardize(x @ self.matrix(out_dim, x.shape[1]).T)

    def nn(self, x: Tensor, out_dim: int) -> Tensor:
        depth = self.sampler.logint("nn-depth", 1, 3)
        hidden = self.sampler.logint("nn-hidden", 1, 127)
        hidden = max(1, min(hidden, 127))
        y = x
        if random.random() < 0.5:
            y = random_activation(y, self.sampler)
        in_dim = y.shape[1]
        for layer_idx in range(depth):
            width = out_dim if layer_idx == depth - 1 else hidden
            y = y @ self.matrix(width, in_dim).T
            if layer_idx != depth - 1 or random.random() < 0.5:
                y = random_activation(y, self.sampler)
            in_dim = width
        return _standardize(y)

    def tree(self, x: Tensor, out_dim: int) -> Tensor:
        n, d = x.shape
        n_trees = self.sampler.logint("tree-count", 1, 128)
        depth = self.sampler.integer("tree-depth", 1, 7)
        x_std = _safe_std(x, dim=0)
        dim_probs = (x_std / x_std.sum().clamp_min(1e-12)).detach().cpu().numpy()
        out = torch.zeros(n, out_dim, dtype=torch.float32, device=self.device)
        max_leaves = 2**depth
        for _ in range(n_trees):
            leaf_idx = torch.zeros(n, dtype=torch.long, device=self.device)
            for level in range(depth):
                split_dim = int(np.random.choice(d, p=dim_probs))
                split_at = x[torch.randint(0, n, (1,), device=self.device), split_dim].item()
                bit = (x[:, split_dim] > split_at).long()
                leaf_idx = leaf_idx * 2 + bit
            leaves = torch.randn(max_leaves, out_dim, dtype=torch.float32, device=self.device)
            out = out + leaves[leaf_idx]
        return _standardize(out / float(n_trees))

    def discretization(self, x: Tensor, out_dim: int) -> Tensor:
        n = x.shape[0]
        n_centers = min(self.sampler.logint("discretization-centers", 2, 255), n)
        centers = x[torch.randperm(n, device=self.device)[:n_centers]]
        p = self.sampler.lognum("discretization-p", 0.5, 4.0)
        dist = (x[:, None, :] - centers[None, :, :]).abs().pow(p).sum(dim=-1)
        nearest = centers[dist.argmin(dim=1)]
        return self.linear(nearest, out_dim)

    def gp(self, x: Tensor, out_dim: int) -> Tensor:
        n, d = x.shape
        p_features = 256
        if random.random() < 0.5:
            a = self.sampler.lognum("gp-radial-tail", 2.0, 20.0)
            z = torch.randn(p_features, d, dtype=torch.float32, device=self.device)
            radius_u = torch.rand(p_features, 1, dtype=torch.float32, device=self.device).clamp(1e-6, 1.0 - 1e-6)
            radius = radius_u.pow(1.0 / (1.0 - a)) - 1.0
            w_fourier = radius * z / z.norm(dim=1, keepdim=True).clamp_min(1e-6)
            scale = self.sampler.lognum("gp-input-scale", 0.5, 10.0)
            weights = random_weights(d, self.sampler, self.device, name="gp-feature-weights")
            a_mat = torch.randn(d, d, dtype=torch.float32, device=self.device)
            projected = (x @ (torch.diag(weights) @ a_mat).T) * scale
        else:
            a = self.sampler.lognum("gp-product-tail", 2.0, 20.0)
            u = torch.rand(p_features, d, dtype=torch.float32, device=self.device).clamp(1e-6, 1.0 - 1e-6)
            signs = torch.randint(0, 2, (p_features, d), device=self.device, dtype=torch.float32) * 2.0 - 1.0
            w_fourier = signs * (u.pow(1.0 / (1.0 - a)) - 1.0)
            projected = x
        b = 2.0 * math.pi * torch.rand(p_features, dtype=torch.float32, device=self.device)
        phi = torch.cos(projected @ w_fourier.T + b) / math.sqrt(float(p_features))
        z_out = torch.randn(out_dim, p_features, dtype=torch.float32, device=self.device)
        return _standardize(phi @ z_out.T)

    def quadratic(self, x: Tensor, out_dim: int) -> Tensor:
        if x.shape[1] > 20:
            idx = torch.randperm(x.shape[1], device=self.device)[:20]
            x = x[:, idx]
        x_aug = torch.cat([x, torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)], dim=1)
        dim = x_aug.shape[1]
        kind = self.matrix.sample_kind(allow_activation=True)
        mat = torch.stack([self.matrix(dim, dim, kind=kind) for _ in range(out_dim)], dim=0)
        return _standardize(torch.einsum("nd,odk,nk->no", x_aug, mat, x_aug))

    def em_assignment(self, x: Tensor, out_dim: int) -> Tensor:
        n, d = x.shape
        m = min(self.sampler.logint("em-components", 2, max(16, 2 * out_dim)), n)
        centers = x[torch.randperm(n, device=self.device)[:m]] + torch.randn(m, d, device=self.device)
        sigma = torch.exp(0.1 * torch.randn(m, device=self.device)).clamp_min(1e-3)
        p = self.sampler.lognum("em-p", 1.0, 4.0)
        q = self.sampler.lognum("em-q", 1.0, 2.0)
        norm = (x[:, None, :] - centers[None, :, :]).abs().pow(p).sum(dim=-1).pow(1.0 / p)
        logits = -0.5 * torch.log(2.0 * math.pi * sigma.square()).unsqueeze(0) - (norm / sigma.unsqueeze(0)).pow(q)
        probs = torch.softmax(logits, dim=-1)
        return self.linear(probs, out_dim)


@dataclass
class Converter:
    is_categorical: bool
    categories: Optional[int]
    dim: int
    kind: str

    def apply(self, x: Tensor, sampler: CorrelatedScalarSampler, fn_factory: RandomFunctionFactory) -> Tuple[Tensor, Tensor]:
        if not self.is_categorical:
            v = x[:, 0]
            if self.kind == "kumaraswamy":
                x_min = x[:, :1].min(dim=0, keepdim=True).values
                x_max = x[:, :1].max(dim=0, keepdim=True).values
                z = ((x[:, :1] - x_min) / (x_max - x_min).clamp_min(1e-6)).clamp(0.0, 1.0)
                a = sampler.lognum("kumaraswamy-a", 0.2, 5.0)
                b = sampler.lognum("kumaraswamy-b", 0.2, 5.0)
                x = x.clone()
                x[:, :1] = 1.0 - (1.0 - z.pow(a)).pow(b)
            return x, v

        c = int(self.categories or 2)
        if self.kind.startswith("softmax"):
            z = _standardize(x[:, :c])
            a = sampler.lognum("softmax-cat-scale", 0.1, 10.0)
            w = random_weights(c, sampler, x.device, name="softmax-cat-weights")
            logits = a * z + torch.log(w + 1e-4).unsqueeze(0)
            idx = torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(1)
            centers = None
        else:
            n_centers = min(c, x.shape[0])
            centers = x[torch.randperm(x.shape[0], device=x.device)[:n_centers], : self.dim]
            if n_centers < c:
                centers = F.pad(centers, (0, 0, 0, c - n_centers), value=0.0)
            p = sampler.lognum("cat-neighbor-p", 0.5, 4.0)
            dist = (x[:, None, : self.dim] - centers[None, :, :]).abs().pow(p).sum(dim=-1)
            idx = dist.argmin(dim=1)

        if self.kind.endswith("input"):
            x_prime = x
        elif self.kind.endswith("index"):
            x_prime = idx.float().unsqueeze(1).repeat(1, self.dim)
        elif self.kind.endswith("center") and centers is not None:
            x_prime = centers[idx]
        elif self.kind.endswith("function_center") and centers is not None:
            x_prime = fn_factory(centers, self.dim, allow_product=True)[idx]
        else:
            z = torch.randn(c, self.dim, dtype=torch.float32, device=x.device)
            x_prime = z[idx]
        return x_prime, idx.float()


@dataclass
class ColumnSpec:
    index: int
    node: int
    converter: Converter
    is_target: bool = False


class TabICLv2ClassificationGenerator:
    def __init__(
        self,
        seq_len: int,
        train_size: int,
        num_features: int,
        max_features: int,
        max_classes: int,
        device: str = "cpu",
        extra_trees_filter: bool = True,
        filter_bootstrap_samples: int = 200,
        max_attempts: int = 2048,
        return_metadata: bool = False,
        class_conditional_cauchy_multiclass: bool = False,
        base_class_bin_weights: Optional[str | Sequence[float] | np.ndarray] = None,
        base_cat_ratio_bin_weights: Optional[str | Sequence[float] | np.ndarray] = None,
        categorical_max_cardinality: int = 9,
        graph_nodes_min: int = 2,
        graph_nodes_max: int = 32,
        node_extra_dim_min: int = 1,
        node_extra_dim_max: int = 32,
        latent_needed_nodes_min: Optional[int] = None,
        latent_needed_nodes_max: Optional[int] = None,
    ):
        self.seq_len = seq_len
        self.train_size = train_size
        self.num_features = num_features
        self.max_features = max_features
        self.max_classes = max(2, int(max_classes))
        self.device = device
        self.extra_trees_filter = extra_trees_filter
        self.filter_bootstrap_samples = filter_bootstrap_samples
        self.max_attempts = max_attempts
        self.return_metadata = return_metadata
        self.class_conditional_cauchy_multiclass = bool(class_conditional_cauchy_multiclass)
        self.categorical_max_cardinality = max(2, int(categorical_max_cardinality))
        self.graph_nodes_min = max(2, int(graph_nodes_min))
        self.graph_nodes_max = max(self.graph_nodes_min, int(graph_nodes_max))
        self.node_extra_dim_min = max(0, int(node_extra_dim_min))
        self.node_extra_dim_max = max(self.node_extra_dim_min, int(node_extra_dim_max))
        self.latent_needed_nodes_min = (
            None if latent_needed_nodes_min is None else max(0, int(latent_needed_nodes_min))
        )
        self.latent_needed_nodes_max = (
            None if latent_needed_nodes_max is None else max(0, int(latent_needed_nodes_max))
        )
        self.base_class_bin_weights = _parse_bin_weights(
            base_class_bin_weights, 5, "base_class_bin_weights"
        )
        self.base_cat_ratio_bin_weights = _parse_bin_weights(
            base_cat_ratio_bin_weights, 5, "base_cat_ratio_bin_weights"
        )
        self._trace: Optional[List[Dict[str, Any]]] = None
        self._last_metadata: Optional[Dict[str, Any]] = None
        self._reset_candidate_sampler()

    def _reset_candidate_sampler(self) -> None:
        self.sampler = CorrelatedScalarSampler()
        self._trace = [] if self.return_metadata else None
        self.fn_factory = RandomFunctionFactory(self.sampler, self.device, trace=self._trace)

    @staticmethod
    def _count_values(values: Iterable[Any]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for value in values:
            key = str(value)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _trace_counts(self, event: str, field: str) -> Dict[str, int]:
        if not self._trace:
            return {}
        return self._count_values(item.get(field) for item in self._trace if item.get("event") == event)

    def _trace_count(self, event: str) -> int:
        if not self._trace:
            return 0
        return sum(1 for item in self._trace if item.get("event") == event)

    def _trace_event(self, event: str, **fields: Any) -> None:
        if self._trace is not None:
            record = {"event": event}
            record.update(fields)
            self._trace.append(record)

    def sample_dataset(self):
        rejections = {"sample": 0, "accept": 0, "postprocess": 0, "split": 0, "filter": 0}
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_attempts + 1):
            self._reset_candidate_sampler()
            self._last_metadata = None
            try:
                sampled = self._sample_once()
            except Exception as exc:
                rejections["sample"] += 1
                last_error = exc
                continue
            x, y = sampled
            if not self._accept_dataset(x, y):
                rejections["accept"] += 1
                continue
            try:
                x, y = self._postprocess(x, y)
            except Exception as exc:
                rejections["postprocess"] += 1
                last_error = exc
                continue
            if not self._fix_split_coverage(x, y):
                rejections["split"] += 1
                continue
            if self.extra_trees_filter and not self._extra_trees_filter(x, y):
                rejections["filter"] += 1
                continue
            d = torch.tensor(x.shape[1], dtype=torch.long, device=self.device)
            padded_x = _pad_features(x, self.max_features)
            if self.return_metadata:
                metadata = dict(self._last_metadata or {})
                metadata.update(
                    {
                        "accepted_attempt": int(attempt),
                        "rejections_before_accept": {key: int(value) for key, value in rejections.items()},
                        "final_num_features": int(d.item()),
                        "final_num_classes": int(torch.unique(y).numel()),
                        "extra_trees_filter": bool(self.extra_trees_filter),
                    }
                )
                return padded_x, y.float(), d, metadata
            return padded_x, y.float(), d
        raise RuntimeError(
            "TabICLv2 classification prior failed to sample an accepted dataset "
            f"after {self.max_attempts} attempts; rejections={rejections}; "
            f"last_error={last_error!r}"
        ) from last_error

    @staticmethod
    def _coerce_class_probs(class_probs: Sequence[float] | np.ndarray, num_classes: int) -> np.ndarray:
        """Normalize empirical class probabilities to the active class cap."""

        num_classes = max(2, int(num_classes))
        probs = np.asarray(class_probs, dtype=np.float64).reshape(-1)
        probs = probs[np.isfinite(probs)]
        if probs.size == 0 or float(probs.sum()) <= 0.0:
            return np.full(num_classes, 1.0 / num_classes, dtype=np.float64)
        probs = np.clip(probs, 0.0, None)
        if probs.size > num_classes:
            compressed = np.zeros(num_classes, dtype=np.float64)
            for idx, prob in enumerate(probs):
                compressed[idx % num_classes] += float(prob)
            probs = compressed
        elif probs.size < num_classes:
            pad_mass = max(1e-8, float(probs.sum()) * 0.01)
            probs = np.concatenate([probs, np.full(num_classes - probs.size, pad_mass, dtype=np.float64)])
        probs = np.clip(probs, 1e-8, None)
        probs = probs / probs.sum()
        return probs

    def _sample_base_num_classes(self) -> int:
        bins = [(2, 2), (3, 3), (4, 5), (6, 10), (11, self.max_classes)]
        return _sample_int_from_weighted_bins(bins, self.base_class_bin_weights, 2, self.max_classes)

    def _sample_base_cat_ratio(self) -> float:
        if self.base_cat_ratio_bin_weights is not None:
            return _sample_cat_ratio_from_weighted_bins(self.base_cat_ratio_bin_weights)
        return float(np.clip(np.random.uniform(-0.5, 1.2), 0.0, 1.0))

    def _sample_once(self) -> Tuple[Tensor, Tensor]:
        num_classes = self._sample_base_num_classes()
        if self.class_conditional_cauchy_multiclass and num_classes >= 3:
            cat_ratio = self._sample_base_cat_ratio() if self.base_cat_ratio_bin_weights is not None else None
            return self._sample_class_conditional_cauchy_once(num_classes, categorical_ratio=cat_ratio)

        cat_ratio = self._sample_base_cat_ratio()
        max_cardinality = self.sampler.logint(
            "categorical-max-cardinality", 2, self.categorical_max_cardinality
        )
        correlated_cardinality_fraction = float(np.random.random())
        num_nodes = self.sampler.logint("graph-nodes", self.graph_nodes_min, self.graph_nodes_max)
        adjacency, x_nodes, y_node = self._sample_accepted_graph_and_assignments(num_nodes)

        specs: List[ColumnSpec] = []
        for col_idx, node_idx in enumerate(x_nodes):
            is_cat = random.random() < cat_ratio
            categories = (
                self._sample_categorical_cardinality(max_cardinality, correlated_cardinality_fraction)
                if is_cat
                else None
            )
            converter = self._make_converter(is_cat, max_cardinality, categories=categories)
            specs.append(ColumnSpec(index=col_idx, node=node_idx, converter=converter))
        target_converter = self._make_target_converter(num_classes)
        specs.append(ColumnSpec(index=0, node=y_node, converter=target_converter, is_target=True))

        if self.return_metadata:
            input_specs = [spec for spec in specs if not spec.is_target]
            input_categories = [
                int(spec.converter.categories)
                for spec in input_specs
                if spec.converter.is_categorical and spec.converter.categories is not None
            ]
            self._last_metadata = {
                "prior_kind": "tabiclv2_cls",
                "seq_len": int(self.seq_len),
                "train_size": int(self.train_size),
                "requested_num_features": int(self.num_features),
                "sampled_num_classes": int(num_classes),
                "categorical_ratio": float(cat_ratio),
                "max_categorical_cardinality": int(max_cardinality),
                "correlated_cardinality_fraction": float(correlated_cardinality_fraction),
                "num_graph_nodes": int(num_nodes),
                "num_graph_edges": int(adjacency.sum()),
                "num_needed_graph_nodes": int(len(self._needed_nodes(adjacency, [spec.node for spec in specs]))),
                "num_observed_assigned_graph_nodes": int(len(set(x_nodes) | {y_node})),
                "num_latent_needed_graph_nodes": int(
                    len(self._needed_nodes(adjacency, [spec.node for spec in specs]) - (set(x_nodes) | {y_node}))
                ),
                "num_unique_x_nodes": int(len(set(x_nodes))),
                "target_node": int(y_node),
                "input_categorical_features": int(sum(spec.converter.is_categorical for spec in input_specs)),
                "input_numerical_features": int(sum(not spec.converter.is_categorical for spec in input_specs)),
                "input_converter_kinds": self._count_values(spec.converter.kind for spec in input_specs),
                "input_categorical_cardinalities": self._count_values(input_categories),
                "target_converter": {
                    "kind": target_converter.kind,
                    "categories": int(target_converter.categories or num_classes),
                    "dim": int(target_converter.dim),
                },
            }

        needed_nodes = self._needed_nodes(adjacency, [spec.node for spec in specs])
        node_outputs: Dict[int, Tensor] = {}
        extracted_x: Dict[int, Tensor] = {}
        extracted_y: Optional[Tensor] = None
        specs_by_node: Dict[int, List[ColumnSpec]] = {}
        for spec in specs:
            specs_by_node.setdefault(spec.node, []).append(spec)

        for node in range(num_nodes):
            if node not in needed_nodes:
                continue
            parents = [p for p in range(node) if adjacency[p, node] and p in node_outputs]
            assigned = specs_by_node.get(node, [])
            required_dim = sum(spec.converter.dim for spec in assigned)
            out_dim = required_dim + self.sampler.logint(
                "node-extra-dim", self.node_extra_dim_min, self.node_extra_dim_max
            )
            if parents:
                parent_data = [node_outputs[p] for p in parents]
                node_x = self._random_multi_function(parent_data, out_dim)
            else:
                node_x = self._random_points(out_dim)
            node_x = self._node_post_function_transform(node_x)

            offset = 0
            for spec in assigned:
                dim = spec.converter.dim
                segment = node_x[:, offset : offset + dim]
                segment_prime, values = spec.converter.apply(segment, self.sampler, self.fn_factory)
                node_x[:, offset : offset + dim] = segment_prime
                if spec.is_target:
                    extracted_y = values
                else:
                    extracted_x[spec.index] = values
                offset += dim
            node_outputs[node] = node_x * self.sampler.lognum("node-rescale", 0.1, 10.0)

        if extracted_y is None or len(extracted_x) != self.num_features:
            raise ValueError("failed to extract all columns")
        x = torch.stack([extracted_x[i] for i in range(self.num_features)], dim=1)
        if self.return_metadata and self._last_metadata is not None:
            self._last_metadata.update(
                {
                    "random_function_call_count": int(self._trace_count("random_function")),
                    "random_function_kinds": self._trace_counts("random_function", "kind"),
                    "random_points_bases": self._trace_counts("random_points", "kind"),
                    "multi_function_aggregations": self._trace_counts("multi_function", "aggregation"),
                }
            )
        return x, extracted_y

    def _sample_class_conditional_cauchy_once(
        self,
        num_classes: int,
        semantic_submode: Optional[str] = None,
        class_probs: Optional[Sequence[float]] = None,
        categorical_ratio: Optional[float] = None,
        num_categorical_features: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Generate a multiclass task where y is a semantic root in a Cauchy DAG."""

        num_classes = max(2, min(int(num_classes), self.max_classes))
        semantic_submode = semantic_submode or str(
            np.random.choice(
                [
                    "latent_factor",
                    "template",
                    "categorical_prototype",
                    "derived_feature",
                ],
                p=[0.35, 0.22, 0.25, 0.18],
            )
        )

        if class_probs is None:
            probs = np.random.dirichlet(np.full(num_classes, 4.0, dtype=np.float64))
            probs = 0.70 * probs + 0.30 / num_classes
            probs = probs / probs.sum()
        else:
            probs = self._coerce_class_probs(class_probs, num_classes)
        min_per_class = 2
        if self.seq_len < min_per_class * num_classes:
            num_classes = max(2, self.seq_len // min_per_class)
            probs = self._coerce_class_probs(probs, num_classes)
        remaining = max(self.seq_len - min_per_class * num_classes, 0)
        counts = np.random.multinomial(remaining, probs) + min_per_class
        counts[0] += self.seq_len - int(counts.sum())
        y_np = np.concatenate([np.full(int(count), cls, dtype=np.int64) for cls, count in enumerate(counts)])
        np.random.shuffle(y_np)
        y = torch.tensor(y_np, dtype=torch.float32, device=self.device)
        y_long = y.long()

        min_nodes = max(4, min(10, self.num_features + 1))
        max_nodes = max(min_nodes, min(64, max(12, self.num_features + 8)))
        min_nodes = max(min_nodes, self.graph_nodes_min)
        max_nodes = max(min_nodes, min(max_nodes, self.graph_nodes_max))
        num_nodes = self.sampler.logint("class-conditional-cauchy-nodes", min_nodes, max_nodes)
        num_nodes = max(4, min(int(num_nodes), 64))
        y_root = 0
        adjacency = self._sample_cauchy_dag(num_nodes)
        root_child_count = int(
            np.random.randint(2, min(num_nodes - 1, max(3, self.num_features // 3 + 2)) + 1)
        )
        root_children = np.random.choice(np.arange(1, num_nodes), size=root_child_count, replace=False)
        adjacency[y_root, root_children] = True

        def descendants_of_root() -> List[int]:
            descendants: set[int] = set()
            frontier = [y_root]
            while frontier:
                current = frontier.pop()
                for child in np.where(adjacency[current])[0].tolist():
                    child = int(child)
                    if child not in descendants:
                        descendants.add(child)
                        frontier.append(child)
            descendants.discard(y_root)
            return sorted(descendants)

        y_descendants = descendants_of_root()
        if not y_descendants:
            fallback = int(root_children[0])
            adjacency[y_root, fallback] = True
            y_descendants = [fallback]

        all_non_root = list(range(1, num_nodes))
        non_descendants = [node for node in all_non_root if node not in set(y_descendants)]
        signal_fraction = float(np.random.uniform(0.68, 0.94))
        x_nodes: List[int] = []
        informative_cols: List[bool] = []
        for _ in range(self.num_features):
            use_signal = np.random.random() < signal_fraction or not non_descendants
            if use_signal:
                node = int(np.random.choice(y_descendants))
            else:
                node = int(np.random.choice(non_descendants))
            x_nodes.append(node)
            informative_cols.append(bool(node in set(y_descendants)))
        if not any(informative_cols):
            x_nodes[0] = int(np.random.choice(y_descendants))
            informative_cols[0] = True

        needed_nodes = self._needed_nodes(adjacency, x_nodes)
        needed_nodes.add(y_root)
        root_dim = max(num_classes, min(16, num_classes + 4))
        root_embedding = torch.randn(num_classes, root_dim, dtype=torch.float32, device=self.device) * float(
            np.random.uniform(0.8, 1.8)
        )
        one_hot = F.one_hot(y_long, num_classes=num_classes).float()
        node_outputs: Dict[int, Tensor] = {
            y_root: _standardize(root_embedding[y_long] + 1.5 * F.pad(one_hot, (0, root_dim - num_classes)))
        }
        for node in range(1, num_nodes):
            if node not in needed_nodes:
                continue
            parents = [p for p in range(node) if adjacency[p, node] and p in node_outputs]
            out_dim = self.sampler.logint("class-conditional-node-dim", 2, 32)
            if parents:
                node_x = self._random_multi_function([node_outputs[p] for p in parents], out_dim)
            else:
                node_x = self._random_points(out_dim)
            node_x = self._node_post_function_transform(node_x)
            node_outputs[node] = node_x * self.sampler.lognum("node-rescale", 0.1, 10.0)

        template = None
        if semantic_submode == "template":
            grid = torch.linspace(0.0, 1.0, self.num_features, dtype=torch.float32, device=self.device)
            templates = []
            for _ in range(num_classes):
                curve = torch.zeros(self.num_features, dtype=torch.float32, device=self.device)
                for _basis in range(int(np.random.randint(2, 5))):
                    kind = str(np.random.choice(["sine", "bump", "step", "slope"]))
                    weight = float(np.random.uniform(-1.4, 1.4))
                    if kind == "sine":
                        curve = curve + weight * torch.sin(
                            2.0
                            * math.pi
                            * float(np.random.uniform(0.6, 3.0))
                            * grid
                            + float(np.random.uniform(0.0, 2.0 * math.pi))
                        )
                    elif kind == "bump":
                        center = float(np.random.uniform(0.15, 0.85))
                        width = float(np.random.uniform(0.05, 0.20))
                        curve = curve + weight * torch.exp(-0.5 * ((grid - center) / width).pow(2))
                    elif kind == "step":
                        curve = curve + weight * ((grid > float(np.random.uniform(0.20, 0.80))).float() * 2.0 - 1.0)
                    else:
                        curve = curve + weight * (grid * 2.0 - 1.0)
                curve = curve - curve.mean()
                curve = curve / curve.std().clamp_min(0.2)
                templates.append(curve)
            template = torch.stack(templates, dim=0)

        columns: List[Tensor] = []
        max_cardinality = self.sampler.logint(
            "categorical-max-cardinality", 2, self.categorical_max_cardinality
        )
        correlated_cardinality_fraction = float(np.random.random())
        if categorical_ratio is None:
            base_cat_ratio = float(np.clip(np.random.uniform(-0.2, 1.0), 0.0, 1.0))
        else:
            base_cat_ratio = float(np.clip(categorical_ratio, 0.0, 1.0))
        if num_categorical_features is None and semantic_submode == "categorical_prototype":
            base_cat_ratio = max(base_cat_ratio, float(np.random.uniform(0.55, 0.90)))
        categorical_columns: set[int] | None = None
        if num_categorical_features is not None:
            n_cat = int(max(0, min(self.num_features, int(num_categorical_features))))
            if n_cat > 0:
                categorical_columns = set(np.random.choice(self.num_features, size=n_cat, replace=False).tolist())
            else:
                categorical_columns = set()
        for col_idx, node in enumerate(x_nodes):
            base = node_outputs[node]
            if base.shape[1] < 2:
                base = torch.cat([base, torch.randn(self.seq_len, 1, dtype=torch.float32, device=self.device)], dim=1)
            is_signal = informative_cols[col_idx]
            is_cat = col_idx in categorical_columns if categorical_columns is not None else np.random.random() < base_cat_ratio
            if semantic_submode == "template":
                base_value = base[:, col_idx % base.shape[1]]
                signal = template[y_long, col_idx] if template is not None else 0.0
                values = 0.55 * _standardize(base_value.unsqueeze(1)).squeeze(1) + signal
                values = values + torch.randn(self.seq_len, dtype=torch.float32, device=self.device) * float(
                    np.random.uniform(0.12, 0.45)
                )
            elif semantic_submode == "categorical_prototype" and (is_cat or is_signal):
                categories = self._sample_categorical_cardinality(max_cardinality, correlated_cardinality_fraction)
                converter = self._make_converter(True, max_cardinality, categories=categories)
                segment = base[:, : converter.dim]
                if segment.shape[1] < converter.dim:
                    segment = F.pad(segment, (0, converter.dim - segment.shape[1]), value=0.0)
                _, base_values = converter.apply(segment, self.sampler, self.fn_factory)
                class_codes = torch.randint(0, categories, (num_classes,), dtype=torch.int64, device=self.device)
                use_proto = torch.rand(self.seq_len, device=self.device) < float(np.random.uniform(0.55, 0.86))
                values = torch.where(use_proto, class_codes[y_long].float(), base_values.float())
            elif semantic_submode == "derived_feature":
                raw = base[:, col_idx % base.shape[1]]
                aux = base[:, (col_idx + 1) % base.shape[1]]
                transform = col_idx % 7
                if transform == 0:
                    values = raw
                elif transform == 1:
                    values = raw.abs()
                elif transform == 2:
                    values = raw.pow(2)
                elif transform == 3:
                    values = torch.sigmoid(raw * float(np.random.uniform(0.7, 1.8)))
                elif transform == 4:
                    values = raw - aux
                elif transform == 5:
                    values = raw * aux
                else:
                    levels = int(np.random.randint(3, 10))
                    score = raw + torch.randn(self.seq_len, dtype=torch.float32, device=self.device) * 0.15
                    cuts = torch.quantile(
                        score.detach(),
                        torch.linspace(0.12, 0.88, levels - 1, dtype=torch.float32, device=self.device),
                    )
                    values = torch.bucketize(score, cuts).float()
                values = values + torch.randn(self.seq_len, dtype=torch.float32, device=self.device) * float(
                    np.random.uniform(0.03, 0.22)
                )
            elif is_cat:
                categories = self._sample_categorical_cardinality(max_cardinality, correlated_cardinality_fraction)
                converter = self._make_converter(True, max_cardinality, categories=categories)
                segment = base[:, : converter.dim]
                if segment.shape[1] < converter.dim:
                    segment = F.pad(segment, (0, converter.dim - segment.shape[1]), value=0.0)
                _, values = converter.apply(segment, self.sampler, self.fn_factory)
            else:
                values = base[:, col_idx % base.shape[1]]
                if np.random.random() < 0.20:
                    values = torch.tanh(values) * float(np.random.uniform(1.0, 4.0))
                values = values + torch.randn(self.seq_len, dtype=torch.float32, device=self.device) * float(
                    np.random.uniform(0.03, 0.20)
                )
            if not is_signal and np.random.random() < 0.50:
                values = torch.randn(self.seq_len, dtype=torch.float32, device=self.device)
            columns.append(values.float())

        x = torch.stack(columns, dim=1)
        label_noise = float(np.random.uniform(0.0, 0.025))
        if label_noise > 0.0:
            noise_mask = torch.rand(self.seq_len, device=self.device) < label_noise
            noise_count = int(noise_mask.sum().item())
            if noise_count > 0:
                y[noise_mask] = torch.randint(0, num_classes, (noise_count,), dtype=torch.int64, device=self.device).float()

        if self.return_metadata:
            self._last_metadata = {
                "prior_kind": "tabiclv2_cls_class_conditional_cauchy",
                "seq_len": int(self.seq_len),
                "train_size": int(self.train_size),
                "requested_num_features": int(self.num_features),
                "sampled_num_classes": int(num_classes),
                "class_conditional_cauchy_multiclass": True,
                "class_conditional_submode": semantic_submode,
                "num_graph_nodes": int(num_nodes),
                "num_graph_edges": int(adjacency.sum()),
                "num_y_root_children": int(len(root_children)),
                "num_y_descendants": int(len(y_descendants)),
                "num_y_descendant_feature_columns": int(sum(informative_cols)),
                "class_counts_before_shuffle": {str(i): int(counts[i]) for i in range(num_classes)},
                "categorical_ratio": float(base_cat_ratio),
                "n_cat_features_requested": (
                    int(num_categorical_features) if num_categorical_features is not None else None
                ),
                "label_noise": float(label_noise),
                "random_function_call_count": int(self._trace_count("random_function")),
                "random_function_kinds": self._trace_counts("random_function", "kind"),
                "random_points_bases": self._trace_counts("random_points", "kind"),
                "multi_function_aggregations": self._trace_counts("multi_function", "aggregation"),
            }

        return x, y

    def _sample_cauchy_dag(self, num_nodes: int) -> np.ndarray:
        a = np.random.standard_cauchy()
        b = np.random.standard_cauchy(num_nodes)
        c = np.random.standard_cauchy(num_nodes)
        adjacency = np.zeros((num_nodes, num_nodes), dtype=bool)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                adjacency[i, j] = np.random.random() < expit(a + b[i] + c[j])
        return adjacency

    def _sample_accepted_graph_and_assignments(self, num_nodes: int, attempts: int = 256) -> Tuple[np.ndarray, List[int], int]:
        for _ in range(attempts):
            adjacency = self._sample_cauchy_dag(num_nodes)
            x_nodes, y_node = self._assign_columns(num_nodes)
            if not self._has_common_ancestor(adjacency, x_nodes, y_node):
                continue
            needed_nodes = self._needed_nodes(adjacency, [*x_nodes, y_node])
            observed_nodes = set(x_nodes)
            observed_nodes.add(y_node)
            latent_needed = len(needed_nodes - observed_nodes)
            if self.latent_needed_nodes_min is not None and latent_needed < self.latent_needed_nodes_min:
                continue
            if self.latent_needed_nodes_max is not None and latent_needed > self.latent_needed_nodes_max:
                continue
            return adjacency, x_nodes, y_node
        raise ValueError("x/y graph filter rejected dataset")

    def _assign_columns(self, num_nodes: int) -> Tuple[List[int], int]:
        def eligible_nodes() -> np.ndarray:
            size = np.random.randint(1, num_nodes + 1)
            return np.random.choice(num_nodes, size=size, replace=False)

        x_eligible = eligible_nodes()
        y_eligible = eligible_nodes()
        x_nodes = [int(np.random.choice(x_eligible)) for _ in range(self.num_features)]
        y_node = int(np.random.choice(y_eligible))
        return x_nodes, y_node

    @staticmethod
    def _ancestors(adjacency: np.ndarray, node: int) -> set[int]:
        ancestors = {node}
        frontier = [node]
        while frontier:
            current = frontier.pop()
            for parent in np.where(adjacency[:, current])[0].tolist():
                if parent not in ancestors:
                    ancestors.add(parent)
                    frontier.append(parent)
        return ancestors

    def _has_common_ancestor(self, adjacency: np.ndarray, x_nodes: Sequence[int], y_node: int) -> bool:
        y_ancestors = self._ancestors(adjacency, y_node)
        return any(self._ancestors(adjacency, node) & y_ancestors for node in x_nodes)

    @staticmethod
    def _ancestors_of_set(adjacency: np.ndarray, nodes: Iterable[int]) -> set[int]:
        ancestors: set[int] = set()
        frontier = [int(node) for node in nodes]
        while frontier:
            current = frontier.pop()
            if current in ancestors:
                continue
            ancestors.add(current)
            frontier.extend(int(parent) for parent in np.where(adjacency[:, current])[0].tolist())
        return ancestors

    @classmethod
    def _d_connected(cls, adjacency: np.ndarray, src: int, dst: int, conditioned: set[int]) -> bool:
        if src in conditioned or dst in conditioned:
            return False

        ancestral_nodes = cls._ancestors_of_set(adjacency, set(conditioned) | {src, dst})
        undirected: Dict[int, set[int]] = {node: set() for node in ancestral_nodes}

        for parent, child in zip(*np.where(adjacency)):
            parent = int(parent)
            child = int(child)
            if parent in ancestral_nodes and child in ancestral_nodes:
                undirected[parent].add(child)
                undirected[child].add(parent)

        for child in ancestral_nodes:
            parents = [
                int(parent)
                for parent in np.where(adjacency[:, child])[0].tolist()
                if int(parent) in ancestral_nodes
            ]
            for i, parent_i in enumerate(parents):
                for parent_j in parents[i + 1 :]:
                    undirected[parent_i].add(parent_j)
                    undirected[parent_j].add(parent_i)

        blocked = set(conditioned)
        frontier = [src]
        visited = {src}
        while frontier:
            current = frontier.pop()
            if current == dst:
                return True
            for neighbor in undirected.get(current, set()):
                if neighbor in blocked or neighbor in visited:
                    continue
                visited.add(neighbor)
                frontier.append(neighbor)
        return False

    def _needed_nodes(self, adjacency: np.ndarray, assigned_nodes: Iterable[int]) -> set[int]:
        needed: set[int] = set()
        for node in assigned_nodes:
            needed |= self._ancestors(adjacency, int(node))
        return needed

    @staticmethod
    def _independent_logint(low: int, high: int) -> int:
        if high <= low:
            return int(low)
        value = math.exp(np.random.uniform(math.log(max(low, 1)), math.log(max(high + 1, low + 1))))
        return int(np.clip(math.floor(value), low, high))

    def _sample_categorical_cardinality(self, max_categories: int, correlated_fraction: float) -> int:
        max_categories = max(2, min(int(max_categories), 10))
        if random.random() < correlated_fraction:
            return self.sampler.logint("categorical-cardinality", 2, max_categories)
        return self._independent_logint(2, max_categories)

    def _make_converter(
        self, categorical: bool, max_categories: int, categories: Optional[int] = None
    ) -> Converter:
        if not categorical:
            kind = self.sampler.choice("numerical-converter", ["identity", "kumaraswamy"])
            return Converter(is_categorical=False, categories=None, dim=1, kind=kind)
        categories = max(2, min(int(categories if categories is not None else max_categories), 10))
        return self._make_categorical_converter(
            categories,
            [
                "neighbor-input",
                "neighbor-index",
                "neighbor-center",
                "neighbor-function_center",
                "softmax-input",
                "softmax-index",
                "softmax-random_points",
            ],
            choice_name="categorical-converter",
        )

    def _make_target_converter(self, num_classes: int) -> Converter:
        categories = max(2, min(int(num_classes), self.max_classes))
        kinds = [
            "neighbor-input",
            "neighbor-index",
            "neighbor-center",
            "neighbor-function_center",
            "softmax-input",
            "softmax-index",
            "softmax-random_points",
        ]
        if categories >= 4:
            kinds = [kind for kind in kinds if kind != "softmax-random_points"]
        return self._make_categorical_converter(
            categories,
            kinds,
            choice_name="target-categorical-converter-high" if categories >= 4 else "target-categorical-converter",
        )

    def _make_categorical_converter(
        self, categories: int, kinds: Sequence[str], choice_name: str
    ) -> Converter:
        kind = self.sampler.choice(
            choice_name,
            kinds,
        )
        if kind.startswith("softmax"):
            dim = categories
        else:
            dim = categories if random.random() < 0.5 else self.sampler.integer("categorical-neighbor-dim", 1, categories - 1)
        return Converter(is_categorical=True, categories=categories, dim=int(dim), kind=kind)

    def _random_multi_function(self, parent_data: List[Tensor], out_dim: int) -> Tensor:
        if len(parent_data) == 1 or random.random() < 0.5:
            self._trace_event(
                "multi_function",
                aggregation="single",
                num_parents=int(len(parent_data)),
                out_dim=int(out_dim),
            )
            return self.fn_factory(torch.cat(parent_data, dim=1), out_dim)
        transformed = torch.stack([self.fn_factory(parent, out_dim) for parent in parent_data], dim=0)
        agg = self.sampler.choice("multi-function-aggregation", ["sum", "product", "max", "logsumexp"])
        self._trace_event(
            "multi_function",
            aggregation=agg,
            num_parents=int(len(parent_data)),
            out_dim=int(out_dim),
        )
        if agg == "sum":
            return transformed.sum(dim=0)
        if agg == "product":
            return transformed.prod(dim=0)
        if agg == "max":
            return transformed.max(dim=0).values
        return torch.logsumexp(transformed, dim=0)

    def _random_points(self, dim: int) -> Tensor:
        kind = self.sampler.choice("random-points-base", ["normal", "uniform", "ball", "covariance"])
        self._trace_event("random_points", kind=kind, dim=int(dim))
        if kind == "normal":
            x = torch.randn(self.seq_len, dim, dtype=torch.float32, device=self.device)
        elif kind == "uniform":
            x = torch.rand(self.seq_len, dim, dtype=torch.float32, device=self.device) * 2.0 - 1.0
        elif kind == "ball":
            z = torch.randn(self.seq_len, dim, dtype=torch.float32, device=self.device)
            z = z / z.norm(dim=1, keepdim=True).clamp_min(1e-6)
            radius = torch.rand(self.seq_len, 1, dtype=torch.float32, device=self.device).pow(1.0 / max(dim, 1))
            x = z * radius
        else:
            z = torch.randn(self.seq_len, dim, dtype=torch.float32, device=self.device)
            w = random_weights(dim, self.sampler, self.device, name="points-covariance-weights")
            a = torch.randn(dim, dim, dtype=torch.float32, device=self.device)
            x = (z * w.unsqueeze(0)) @ a.T
        return self.fn_factory(x, dim, allow_product=True)

    def _node_post_function_transform(self, x: Tensor) -> Tensor:
        x = _standardize(x)
        weights = random_weights(x.shape[1], self.sampler, self.device, name="node-feature-importance")
        x = x * weights.unsqueeze(0)
        avg_l2 = x.norm(dim=1).mean().clamp_min(1e-6)
        return x / avg_l2

    def _accept_dataset(self, x: Tensor, y: Tensor) -> bool:
        if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
            return False
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            return False
        if torch.unique(y).numel() < 2:
            return False
        return True

    def _fix_split_coverage(self, x: Tensor, y: Tensor) -> bool:
        y_enc = _ordinal_encode(y)
        classes = torch.unique(y_enc)
        num_classes = int(classes.numel())
        if num_classes < 2 or self.train_size < num_classes or (y.shape[0] - self.train_size) < num_classes:
            return False

        train_indices: List[Tensor] = []
        test_indices: List[Tensor] = []
        remaining: List[Tensor] = []
        for cls in classes.tolist():
            cls_idx = torch.where(y_enc == int(cls))[0]
            if cls_idx.numel() < 2:
                return False
            cls_idx = cls_idx[torch.randperm(cls_idx.numel(), device=cls_idx.device)]
            train_indices.append(cls_idx[:1])
            test_indices.append(cls_idx[1:2])
            if cls_idx.numel() > 2:
                remaining.append(cls_idx[2:])

        remaining_idx = (
            torch.cat(remaining) if remaining else torch.empty(0, dtype=torch.long, device=y.device)
        )
        if remaining_idx.numel() > 0:
            remaining_idx = remaining_idx[torch.randperm(remaining_idx.numel(), device=y.device)]

        train_base = torch.cat(train_indices)
        test_base = torch.cat(test_indices)
        train_needed = self.train_size - train_base.numel()
        if train_needed < 0 or train_needed > remaining_idx.numel():
            return False
        train_extra = remaining_idx[:train_needed]
        test_extra = remaining_idx[train_needed:]
        train_idx = torch.cat([train_base, train_extra])
        test_idx = torch.cat([test_base, test_extra])
        train_idx = train_idx[torch.randperm(train_idx.numel(), device=y.device)]
        test_idx = test_idx[torch.randperm(test_idx.numel(), device=y.device)]
        perm = torch.cat([train_idx, test_idx])
        if perm.numel() != y.shape[0]:
            return False
        x[:] = x[perm]
        y[:] = y[perm]
        return True

    def _extra_trees_filter(self, x: Tensor, y: Tensor) -> bool:
        from sklearn.ensemble import ExtraTreesRegressor

        x_np = torch.nan_to_num(x).detach().cpu().numpy().astype(np.float32)
        y_enc = _ordinal_encode(y)
        num_classes = int(y_enc.max().item()) + 1
        y_np = F.one_hot(y_enc, num_classes=num_classes).float().detach().cpu().numpy()
        try:
            model = ExtraTreesRegressor(
                n_estimators=25,
                bootstrap=True,
                oob_score=False,
                max_depth=6,
                random_state=np.random.randint(0, 2**31 - 1),
                n_jobs=1,
            )
            model.fit(x_np, y_np)
            oob_sum = np.zeros_like(y_np, dtype=np.float32)
            oob_count = np.zeros(x_np.shape[0], dtype=np.int32)
            for estimator, in_bag in zip(model.estimators_, model.estimators_samples_):
                in_bag_mask = np.zeros(x_np.shape[0], dtype=bool)
                in_bag_mask[np.asarray(in_bag, dtype=np.intp)] = True
                oob_mask = ~in_bag_mask
                if not oob_mask.any():
                    continue
                tree_pred = np.asarray(estimator.predict(x_np[oob_mask]), dtype=np.float32)
                if tree_pred.ndim == 1:
                    tree_pred = tree_pred[:, None]
                if tree_pred.shape != oob_sum[oob_mask].shape or not np.isfinite(tree_pred).all():
                    return False
                oob_sum[oob_mask] += tree_pred
                oob_count[oob_mask] += 1
        except Exception:
            return False

        if np.any(oob_count == 0):
            return False
        pred = oob_sum / oob_count[:, None]
        if not np.isfinite(pred).all():
            return False
        labels = y_np
        baseline = np.repeat(labels.mean(axis=0, keepdims=True), labels.shape[0], axis=0)
        rng = np.random.default_rng(
            int(np.random.randint(0, 2**32 - 1))
        )
        wins = 0
        sample_count = int(self.filter_bootstrap_samples)
        n = labels.shape[0]
        for _ in range(sample_count):
            idx = rng.integers(0, n, size=n)
            model_mse = np.mean((labels[idx] - pred[idx]) ** 2)
            baseline_mse = np.mean((labels[idx] - baseline[idx]) ** 2)
            wins += int(model_mse < baseline_mse)
        return wins >= math.ceil(0.95 * sample_count)

    def _postprocess(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        y = _ordinal_encode(y)
        perm = torch.randperm(x.shape[0], device=x.device)
        x, y = x[perm], y[perm]
        keep = [idx for idx in range(x.shape[1]) if torch.unique(x[:, idx]).numel() > 1]
        if not keep:
            raise ValueError("no non-constant features")
        x = x[:, keep]
        x = _standardize(_remove_outliers(x))
        col_perm = torch.randperm(x.shape[1], device=x.device)
        x = x[:, col_perm]
        class_perm = torch.randperm(int(y.max().item()) + 1, device=x.device)
        y = class_perm[y.long()]
        return x.float(), y.float()

class TabICLv2ClassificationPrior:
    """Iterable prior implementing the TabICLv2 classification data generator."""

    def __init__(
        self,
        batch_size: int = 64,
        batch_size_per_gp: int = 4,
        min_features: int = 2,
        max_features: int = 100,
        max_classes: int = 10,
        min_seq_len: Optional[int] = None,
        max_seq_len: int = 1024,
        log_seq_len: bool = False,
        seq_len_per_gp: bool = False,
        min_train_size: int | float = 0.3,
        max_train_size: int | float = 0.9,
        replay_small: bool = False,
        device: str = "cpu",
        extra_trees_filter: bool = True,
        filter_bootstrap_samples: int = 200,
        class_conditional_cauchy_multiclass: bool = False,
        base_class_bin_weights: Optional[str | Sequence[float] | np.ndarray] = None,
        base_feature_bin_weights: Optional[str | Sequence[float] | np.ndarray] = None,
        base_cat_ratio_bin_weights: Optional[str | Sequence[float] | np.ndarray] = None,
        categorical_max_cardinality: int = 9,
        graph_nodes_min: int = 2,
        graph_nodes_max: int = 32,
        node_extra_dim_min: int = 1,
        node_extra_dim_max: int = 32,
        latent_needed_nodes_min: Optional[int] = None,
        latent_needed_nodes_max: Optional[int] = None,
        dynamic_reweight_path: Optional[str] = None,
        dynamic_reweight_reload_sec: float = 30.0,
        dynamic_reweight_blend: float = 1.0,
        return_metadata: bool = False,
    ):
        self.batch_size = batch_size
        self.batch_size_per_gp = batch_size_per_gp
        self.min_features = min_features
        self.max_features = max_features
        self.max_classes = max_classes
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.log_seq_len = log_seq_len
        self.seq_len_per_gp = seq_len_per_gp
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.replay_small = replay_small
        self.device = device
        self.extra_trees_filter = extra_trees_filter
        self.filter_bootstrap_samples = filter_bootstrap_samples
        self.class_conditional_cauchy_multiclass = bool(class_conditional_cauchy_multiclass)
        self.base_class_bin_weights = _parse_bin_weights(
            base_class_bin_weights, 5, "base_class_bin_weights"
        )
        self.base_feature_bin_weights = _parse_bin_weights(
            base_feature_bin_weights, 5, "base_feature_bin_weights"
        )
        self.base_cat_ratio_bin_weights = _parse_bin_weights(
            base_cat_ratio_bin_weights, 5, "base_cat_ratio_bin_weights"
        )
        self.categorical_max_cardinality = max(2, int(categorical_max_cardinality))
        self.graph_nodes_min = max(2, int(graph_nodes_min))
        self.graph_nodes_max = max(self.graph_nodes_min, int(graph_nodes_max))
        self.node_extra_dim_min = max(0, int(node_extra_dim_min))
        self.node_extra_dim_max = max(self.node_extra_dim_min, int(node_extra_dim_max))
        self.latent_needed_nodes_min = (
            None if latent_needed_nodes_min is None else max(0, int(latent_needed_nodes_min))
        )
        self.latent_needed_nodes_max = (
            None if latent_needed_nodes_max is None else max(0, int(latent_needed_nodes_max))
        )
        self.dynamic_reweight_path = Path(dynamic_reweight_path) if dynamic_reweight_path else None
        self.dynamic_reweight_reload_sec = max(0.0, float(dynamic_reweight_reload_sec))
        self.dynamic_reweight_blend = float(np.clip(dynamic_reweight_blend, 0.0, 1.0))
        self._dynamic_reweight_last_check = 0.0
        self._dynamic_reweight_mtime: Optional[float] = None
        self._dynamic_class_bin_weights: Optional[np.ndarray] = None
        self._dynamic_feature_bin_weights: Optional[np.ndarray] = None
        self._dynamic_cat_ratio_bin_weights: Optional[np.ndarray] = None
        self.return_metadata = return_metadata

    def _maybe_reload_dynamic_reweight(self) -> None:
        if self.dynamic_reweight_path is None:
            return
        now = time.monotonic()
        if self._dynamic_reweight_last_check and now - self._dynamic_reweight_last_check < self.dynamic_reweight_reload_sec:
            return
        self._dynamic_reweight_last_check = now
        try:
            stat = self.dynamic_reweight_path.stat()
        except FileNotFoundError:
            return
        if self._dynamic_reweight_mtime == stat.st_mtime:
            return
        payload = json.loads(self.dynamic_reweight_path.read_text(encoding="utf-8"))
        self._dynamic_class_bin_weights = _parse_bin_weights(
            payload.get("class_bin_weights"), 5, "dynamic class_bin_weights"
        )
        self._dynamic_feature_bin_weights = _parse_bin_weights(
            payload.get("feature_bin_weights"), 5, "dynamic feature_bin_weights"
        )
        self._dynamic_cat_ratio_bin_weights = _parse_bin_weights(
            payload.get("cat_ratio_bin_weights"), 5, "dynamic cat_ratio_bin_weights"
        )
        self._dynamic_reweight_mtime = stat.st_mtime

    def _effective_class_bin_weights(self) -> Optional[np.ndarray]:
        self._maybe_reload_dynamic_reweight()
        return _blend_bin_weights(self.base_class_bin_weights, self._dynamic_class_bin_weights, self.dynamic_reweight_blend)

    def _effective_feature_bin_weights(self) -> Optional[np.ndarray]:
        self._maybe_reload_dynamic_reweight()
        return _blend_bin_weights(self.base_feature_bin_weights, self._dynamic_feature_bin_weights, self.dynamic_reweight_blend)

    def _effective_cat_ratio_bin_weights(self) -> Optional[np.ndarray]:
        self._maybe_reload_dynamic_reweight()
        return _blend_bin_weights(
            self.base_cat_ratio_bin_weights,
            self._dynamic_cat_ratio_bin_weights,
            self.dynamic_reweight_blend,
        )

    @staticmethod
    def sample_seq_len(min_seq_len: Optional[int], max_seq_len: int, log: bool = False, replay_small: bool = False) -> int:
        if min_seq_len is None:
            seq_len = max_seq_len
        elif log:
            seq_len = int(math.exp(np.random.uniform(math.log(min_seq_len), math.log(max_seq_len))))
        else:
            seq_len = int(np.random.randint(min_seq_len, max_seq_len + 1))
        if replay_small:
            p = np.random.random()
            if p < 0.05:
                return int(np.random.randint(200, 1001))
            if p < 0.3:
                return int(math.exp(np.random.uniform(math.log(1000), math.log(10000))))
        return seq_len

    @staticmethod
    def sample_train_size(min_train_size: int | float, max_train_size: int | float, seq_len: int) -> int:
        if isinstance(min_train_size, int) and isinstance(max_train_size, int):
            if min_train_size == max_train_size:
                return min_train_size
            return int(np.random.randint(min_train_size, max_train_size + 1))
        if isinstance(min_train_size, float) and isinstance(max_train_size, float):
            if min_train_size == max_train_size:
                ratio = min_train_size
            else:
                ratio = float(np.random.uniform(min_train_size, max_train_size))
            return int(np.clip(round(seq_len * ratio), 1, seq_len - 1))
        raise ValueError("train sizes must both be int or both be float")

    def _sample_base_num_features(self, max_features: Optional[int] = None) -> int:
        feature_cap = self.max_features if max_features is None else int(max_features)
        bins = [(2, 5), (6, 10), (11, 20), (21, 50), (51, 100)]
        return _sample_int_from_weighted_bins(
            bins,
            self._effective_feature_bin_weights(),
            self.min_features,
            min(feature_cap, self.max_features),
        )

    def _generate_one(self, seq_len: int, train_size: int, num_features: int) -> Tuple[Tensor, Tensor, Tensor]:
        generator = TabICLv2ClassificationGenerator(
            seq_len=seq_len,
            train_size=train_size,
            num_features=num_features,
            max_features=self.max_features,
            max_classes=self.max_classes,
            device=self.device,
            extra_trees_filter=self.extra_trees_filter,
            filter_bootstrap_samples=self.filter_bootstrap_samples,
            return_metadata=self.return_metadata,
            class_conditional_cauchy_multiclass=self.class_conditional_cauchy_multiclass,
            base_class_bin_weights=self._effective_class_bin_weights(),
            base_cat_ratio_bin_weights=self._effective_cat_ratio_bin_weights(),
            categorical_max_cardinality=self.categorical_max_cardinality,
            graph_nodes_min=self.graph_nodes_min,
            graph_nodes_max=self.graph_nodes_max,
            node_extra_dim_min=self.node_extra_dim_min,
            node_extra_dim_max=self.node_extra_dim_max,
            latent_needed_nodes_min=self.latent_needed_nodes_min,
            latent_needed_nodes_max=self.latent_needed_nodes_max,
        )
        return generator.sample_dataset()

    def _generate_class_conditional_cauchy_stress(self, seq_len: int, train_size: int):
        """Generate a class-conditional Cauchy-DAG multiclass replay batch."""

        max_classes = max(3, int(self.max_classes))
        num_classes = int(np.random.randint(3, max_classes + 1))
        high_features = max(int(self.min_features), min(int(self.max_features), 100))
        low_features = min(max(12, int(self.min_features)), high_features)
        num_features = int(np.random.randint(low_features, high_features + 1))
        generator = TabICLv2ClassificationGenerator(
            seq_len=seq_len,
            train_size=train_size,
            num_features=num_features,
            max_features=self.max_features,
            max_classes=self.max_classes,
            device=self.device,
            extra_trees_filter=False,
            filter_bootstrap_samples=self.filter_bootstrap_samples,
            return_metadata=self.return_metadata,
            class_conditional_cauchy_multiclass=True,
            categorical_max_cardinality=self.categorical_max_cardinality,
        )
        rejections = {"sample": 0, "accept": 0, "postprocess": 0, "split": 0}
        last_error: Optional[Exception] = None
        for attempt in range(1, 129):
            generator._reset_candidate_sampler()
            generator._last_metadata = None
            try:
                sampled = generator._sample_class_conditional_cauchy_once(num_classes)
            except Exception as exc:
                rejections["sample"] += 1
                last_error = exc
                continue
            x, y = sampled
            if not generator._accept_dataset(x, y):
                rejections["accept"] += 1
                continue
            try:
                x, y = generator._postprocess(x, y)
            except Exception as exc:
                rejections["postprocess"] += 1
                last_error = exc
                continue
            if not generator._fix_split_coverage(x, y):
                rejections["split"] += 1
                continue

            d = torch.tensor(x.shape[1], dtype=torch.long, device=self.device)
            padded_x = _pad_features(x, self.max_features)
            if self.return_metadata:
                metadata = dict(generator._last_metadata or {})
                metadata.update(
                    {
                        "prior_kind": "tabiclv2_cls_hard_diversity_class_conditional_cauchy",
                        "accepted_attempt": int(attempt),
                        "rejections_before_accept": {key: int(value) for key, value in rejections.items()},
                        "final_num_features": int(d.item()),
                        "final_num_classes": int(torch.unique(y).numel()),
                    }
                )
                return padded_x, y.float(), d, metadata
            return padded_x, y.float(), d
        raise RuntimeError(
            "class-conditional Cauchy DAG stress replay failed "
            f"after 128 attempts; rejections={rejections}; last_error={last_error!r}"
        ) from last_error

    def _generate_hard_diversity_stress(self, seq_len: int, train_size: int):
        """Generate generic hard tabular tasks matching common low-performing regimes.

        This intentionally does not read or encode any data178 dataset. Multiclass
        replay is kept mostly learnable; the harsh weak-signal/sparse/noisy-rule
        regimes are kept binary so Stage-3 is not dominated by near-impossible
        synthetic multiclass batches.
        """

        max_classes = max(2, int(self.max_classes))
        if self.class_conditional_cauchy_multiclass and max_classes >= 3 and np.random.random() < 0.90:
            return self._generate_class_conditional_cauchy_stress(seq_len, train_size)

        semantic_modes = {
            "semantic_latent_factor",
            "semantic_class_template",
            "semantic_categorical_rule",
            "semantic_derived_feature",
        }
        if max_classes >= 3 and np.random.random() < 0.90:
            mode = str(
                np.random.choice(
                    [
                        "semantic_latent_factor",
                        "semantic_class_template",
                        "semantic_categorical_rule",
                        "semantic_derived_feature",
                        "ordinal_survey",
                        "many_class_continuous",
                        "low_dim_count_multiclass",
                        "trend_indicator",
                    ],
                    p=[0.18, 0.16, 0.14, 0.12, 0.12, 0.10, 0.12, 0.06],
                )
            )
        else:
            mode = str(
                np.random.choice(
                    [
                        "weak_signal_continuous",
                        "sparse_symbolic",
                        "mixed_high_cardinality",
                        "noisy_boolean_rule",
                        "sparse_weak_binary",
                    ],
                    p=[0.22, 0.18, 0.22, 0.18, 0.20],
                )
            )

        if mode == "semantic_latent_factor":
            num_classes = int(np.random.randint(3, min(max_classes, 8) + 1)) if max_classes >= 3 else 2
        elif mode == "semantic_class_template":
            num_classes = int(np.random.randint(3, min(max_classes, 6) + 1)) if max_classes >= 3 else 2
        elif mode == "semantic_categorical_rule":
            num_classes = int(np.random.randint(3, min(max_classes, 8) + 1)) if max_classes >= 3 else 2
        elif mode == "semantic_derived_feature":
            num_classes = int(np.random.randint(3, min(max_classes, 7) + 1)) if max_classes >= 3 else 2
        elif mode == "many_class_continuous":
            low = min(4, max_classes)
            high = min(max_classes, 8)
            num_classes = int(np.random.randint(low, high + 1)) if max_classes >= low else max_classes
        elif mode == "ordinal_survey":
            num_classes = int(np.random.randint(3, min(max_classes, 7) + 1)) if max_classes >= 3 else 2
        elif mode == "mixed_high_cardinality":
            num_classes = 2
        elif mode == "sparse_symbolic":
            num_classes = 2
        elif mode == "low_dim_count_multiclass":
            num_classes = int(np.random.randint(3, min(max_classes, 5) + 1)) if max_classes >= 3 else 2
        elif mode == "bio_multiclass_small_margin":
            num_classes = int(np.random.randint(3, max_classes + 1)) if max_classes >= 3 else 2
        elif mode == "noisy_boolean_rule":
            num_classes = 2
        elif mode == "trend_indicator":
            num_classes = int(np.random.randint(2, min(max_classes, 4) + 1))
        elif mode == "sparse_weak_binary":
            num_classes = 2
        else:
            num_classes = 2

        min_per_class = 2
        if seq_len < min_per_class * num_classes:
            num_classes = max(2, seq_len // min_per_class)
        if mode in semantic_modes:
            probs = np.random.dirichlet(np.full(num_classes, 4.0, dtype=np.float64))
            probs = 0.70 * probs + 0.30 / num_classes
            probs = probs / probs.sum()
        elif mode in {"ordinal_survey", "mixed_high_cardinality"}:
            probs = self._sample_count_stress_class_probs(num_classes)
            probs = 0.5 * probs + 0.5 / num_classes
        elif mode == "many_class_continuous":
            probs = np.random.dirichlet(np.full(num_classes, 4.0, dtype=np.float64))
        elif mode == "sparse_symbolic":
            probs = self._sample_count_stress_class_probs(num_classes)
        elif mode == "low_dim_count_multiclass":
            if num_classes >= 4:
                majority = float(np.random.uniform(0.45, 0.68))
                mid_a = float(np.random.uniform(0.12, 0.28))
                mid_b = float(np.random.uniform(0.10, 0.26))
                rare = float(math.exp(np.random.uniform(math.log(0.001), math.log(0.035))))
                probs = np.array([majority, mid_a, mid_b, rare], dtype=np.float64)
                if num_classes > 4:
                    extras = np.random.dirichlet(np.ones(num_classes - 4)) * float(np.random.uniform(0.02, 0.12))
                    probs = np.concatenate([probs, extras])
                probs = probs[:num_classes]
                probs = probs / probs.sum()
            else:
                majority = float(np.random.uniform(0.48, 0.72))
                mid = float(np.random.uniform(0.15, 0.34))
                probs = np.array([majority, mid, max(1e-4, 1.0 - majority - mid)], dtype=np.float64)[:num_classes]
                probs = probs / probs.sum()
        elif mode == "bio_multiclass_small_margin":
            probs = np.random.dirichlet(np.full(num_classes, float(np.random.uniform(0.8, 2.2)), dtype=np.float64))
        elif mode == "noisy_boolean_rule":
            probs = self._sample_count_stress_class_probs(num_classes)
            probs = 0.55 * probs + 0.45 / num_classes
            probs = probs / probs.sum()
        elif mode == "trend_indicator":
            logits = np.random.normal(0.0, 0.75, size=num_classes)
            probs = np.exp(logits - logits.max())
            probs = probs / probs.sum()
        elif mode == "sparse_weak_binary":
            majority = float(np.random.uniform(0.55, 0.88))
            probs = np.array([majority, 1.0 - majority], dtype=np.float64)
        else:
            logits = np.random.normal(0.0, 0.45, size=num_classes)
            probs = np.exp(logits - logits.max())
            probs = probs / probs.sum()

        remaining = max(seq_len - min_per_class * num_classes, 0)
        counts = np.random.multinomial(remaining, probs) + min_per_class
        counts[0] += seq_len - int(counts.sum())
        y_np = np.concatenate([np.full(int(count), cls, dtype=np.int64) for cls, count in enumerate(counts)])
        np.random.shuffle(y_np)
        y = torch.tensor(y_np, dtype=torch.float32, device=self.device)
        y_long = y.long()

        if mode == "semantic_latent_factor":
            high_features = max(int(self.min_features), min(int(self.max_features), 96))
            low_features = min(max(12, int(self.min_features)), high_features)
            num_features = int(np.random.randint(low_features, high_features + 1))
            max_latent = max(2, min(6, num_features // 3))
            latent_dim = int(np.random.randint(2, max_latent + 1))
            center_scale = float(np.random.uniform(1.0, 2.0))
            centers = torch.randn(num_classes, latent_dim, dtype=torch.float32, device=self.device) * center_scale
            latent_noise = float(np.random.uniform(0.35, 0.85))
            latent = centers[y_long] + torch.randn(seq_len, latent_dim, dtype=torch.float32, device=self.device) * latent_noise
            x = torch.zeros(seq_len, num_features, dtype=torch.float32, device=self.device)
            feature_factor = torch.randint(0, latent_dim, (num_features,), dtype=torch.int64, device=self.device)
            for col in range(num_features):
                primary = int(feature_factor[col].item())
                values = latent[:, primary] * float(np.random.uniform(0.55, 1.45))
                if latent_dim > 1 and np.random.random() < 0.28:
                    secondary = int((primary + np.random.randint(1, latent_dim)) % latent_dim)
                    values = values + latent[:, secondary] * float(np.random.uniform(-0.35, 0.35))
                values = values + torch.randn(seq_len, dtype=torch.float32, device=self.device) * float(
                    np.random.uniform(0.12, 0.50)
                )
                if np.random.random() < 0.18:
                    values = torch.tanh(values) * float(np.random.uniform(1.0, 4.0))
                elif np.random.random() < 0.18:
                    levels = int(np.random.randint(3, 9))
                    cuts = torch.quantile(
                        values.detach(),
                        torch.linspace(0.15, 0.85, levels - 1, dtype=torch.float32, device=self.device),
                    )
                    values = torch.bucketize(values, cuts).float()
                x[:, col] = values

            redundant_cols = int(num_features * np.random.uniform(0.08, 0.25))
            if redundant_cols > 0 and num_features > 1:
                dst_cols = torch.randperm(num_features, device=self.device)[:redundant_cols]
                src_cols = torch.randint(0, num_features, (redundant_cols,), device=self.device)
                x[:, dst_cols] = (
                    x[:, src_cols] * float(np.random.uniform(0.65, 1.15))
                    + torch.randn(seq_len, redundant_cols, dtype=torch.float32, device=self.device)
                    * float(np.random.uniform(0.04, 0.20))
                )

        elif mode == "semantic_class_template":
            high_features = max(int(self.min_features), min(int(self.max_features), 96))
            low_features = min(max(16, int(self.min_features)), high_features)
            num_features = int(np.random.randint(low_features, high_features + 1))
            grid = torch.linspace(0.0, 1.0, num_features, dtype=torch.float32, device=self.device)
            templates = []
            for _ in range(num_classes):
                template = torch.zeros(num_features, dtype=torch.float32, device=self.device)
                for _basis in range(int(np.random.randint(2, 5))):
                    basis_type = str(np.random.choice(["sine", "bump", "step", "slope"]))
                    weight = float(np.random.uniform(-1.5, 1.5))
                    if basis_type == "sine":
                        freq = float(np.random.uniform(0.6, 3.2))
                        phase = float(np.random.uniform(0.0, 2.0 * math.pi))
                        basis = torch.sin(2.0 * math.pi * freq * grid + phase)
                    elif basis_type == "bump":
                        center = float(np.random.uniform(0.15, 0.85))
                        width = float(np.random.uniform(0.05, 0.20))
                        basis = torch.exp(-0.5 * ((grid - center) / width).pow(2))
                    elif basis_type == "step":
                        split = float(np.random.uniform(0.20, 0.80))
                        basis = (grid > split).float() * 2.0 - 1.0
                    else:
                        basis = grid * 2.0 - 1.0
                    template = template + weight * basis
                template = template - template.mean()
                template = template / template.std().clamp_min(0.2)
                templates.append(template)
            templates_t = torch.stack(templates, dim=0)
            amplitude = torch.empty(seq_len, 1, dtype=torch.float32, device=self.device).uniform_(0.75, 1.25)
            shift = torch.randn(seq_len, 1, dtype=torch.float32, device=self.device) * float(np.random.uniform(0.00, 0.25))
            x = amplitude * templates_t[y_long] + shift
            x = x + torch.randn(seq_len, num_features, dtype=torch.float32, device=self.device) * float(
                np.random.uniform(0.20, 0.65)
            )
            if np.random.random() < 0.45:
                noise_cols = max(1, int(num_features * np.random.uniform(0.08, 0.30)))
                cols = torch.randperm(num_features, device=self.device)[:noise_cols]
                x[:, cols] = torch.randn(seq_len, noise_cols, dtype=torch.float32, device=self.device)

        elif mode == "semantic_categorical_rule":
            high_features = max(int(self.min_features), min(int(self.max_features), 120))
            low_features = min(max(12, int(self.min_features)), high_features)
            num_features = int(np.random.randint(low_features, high_features + 1))
            x = torch.zeros(seq_len, num_features, dtype=torch.float32, device=self.device)
            info_cols = set(
                torch.randperm(num_features, device=self.device)[: max(3, int(num_features * np.random.uniform(0.18, 0.42)))]
                .cpu()
                .numpy()
                .tolist()
            )
            for col in range(num_features):
                if col in info_cols:
                    levels = int(np.random.randint(max(3, num_classes), max(5, min(64, 4 * num_classes)) + 1))
                    class_codes = torch.randint(0, levels, (num_classes,), dtype=torch.int64, device=self.device)
                    base = torch.randint(0, levels, (seq_len,), dtype=torch.int64, device=self.device)
                    use_proto = torch.rand(seq_len, device=self.device) < float(np.random.uniform(0.58, 0.88))
                    values = torch.where(use_proto, class_codes[y_long], base).float()
                    if np.random.random() < 0.18:
                        values = values + torch.randn(seq_len, dtype=torch.float32, device=self.device) * float(
                            np.random.uniform(0.05, 0.25)
                        )
                elif np.random.random() < 0.55:
                    levels = int(np.random.randint(2, 80))
                    values = torch.randint(0, levels, (seq_len,), dtype=torch.int64, device=self.device).float()
                else:
                    values = torch.randn(seq_len, dtype=torch.float32, device=self.device) * float(
                        np.random.uniform(0.7, 2.5)
                    )
                if np.random.random() < 0.08:
                    values[torch.rand(seq_len, device=self.device) < float(np.random.uniform(0.04, 0.25))] = 0.0
                x[:, col] = values

        elif mode == "semantic_derived_feature":
            high_features = max(int(self.min_features), min(int(self.max_features), 96))
            low_features = min(max(12, int(self.min_features)), high_features)
            num_features = int(np.random.randint(low_features, high_features + 1))
            num_sources = int(np.random.randint(2, min(6, max(2, num_features // 4)) + 1))
            centers = torch.randn(num_classes, num_sources, dtype=torch.float32, device=self.device) * float(
                np.random.uniform(0.9, 1.8)
            )
            raw = centers[y_long] + torch.randn(seq_len, num_sources, dtype=torch.float32, device=self.device) * float(
                np.random.uniform(0.35, 0.95)
            )
            x = torch.zeros(seq_len, num_features, dtype=torch.float32, device=self.device)
            for col in range(num_features):
                src = col % num_sources
                aux = (src + 1) % num_sources
                transform = col % 7
                if transform == 0:
                    values = raw[:, src]
                elif transform == 1:
                    values = raw[:, src].abs()
                elif transform == 2:
                    values = raw[:, src].pow(2)
                elif transform == 3:
                    values = torch.sigmoid(raw[:, src] * float(np.random.uniform(0.7, 1.8)))
                elif transform == 4:
                    values = raw[:, src] - raw[:, aux]
                elif transform == 5:
                    values = raw[:, src] * raw[:, aux]
                else:
                    levels = int(np.random.randint(3, 10))
                    score = raw[:, src] + torch.randn(seq_len, dtype=torch.float32, device=self.device) * 0.15
                    cuts = torch.quantile(
                        score.detach(),
                        torch.linspace(0.12, 0.88, levels - 1, dtype=torch.float32, device=self.device),
                    )
                    values = torch.bucketize(score, cuts).float()
                values = values + torch.randn(seq_len, dtype=torch.float32, device=self.device) * float(
                    np.random.uniform(0.03, 0.25)
                )
                x[:, col] = values

        elif mode == "ordinal_survey":
            high_features = max(int(self.min_features), min(int(self.max_features), 48))
            low_features = min(max(8, int(self.min_features)), high_features)
            num_features = int(np.random.randint(low_features, high_features + 1))
            x = torch.zeros(seq_len, num_features, dtype=torch.float32, device=self.device)
            class_bias = torch.randn(num_classes, num_features, dtype=torch.float32, device=self.device) * 0.9
            for col in range(num_features):
                levels = int(np.random.randint(2, 13))
                score = class_bias[y_long, col] + torch.randn(seq_len, dtype=torch.float32, device=self.device) * 1.2
                score = score + torch.randn((), dtype=torch.float32, device=self.device) * 0.2
                buckets = torch.bucketize(score, torch.linspace(-2.0, 2.0, levels - 1, device=self.device))
                values = buckets.float()
                if np.random.random() < 0.18:
                    # Add a high-cardinality ordinal code column.
                    card = int(np.random.randint(16, 128))
                    values = torch.remainder(values * int(np.random.randint(3, 17)) + torch.randint(0, card, (seq_len,), device=self.device), card).float()
                if np.random.random() < 0.08:
                    miss = torch.rand(seq_len, device=self.device) < float(np.random.uniform(0.05, 0.35))
                    values[miss] = 0.0
                x[:, col] = values

        elif mode == "many_class_continuous":
            high_features = max(int(self.min_features), min(int(self.max_features), 100))
            low_features = min(max(20, int(self.min_features)), high_features)
            num_features = int(np.random.randint(low_features, high_features + 1))
            latent_dim = int(np.random.randint(3, min(12, num_features) + 1))
            centers = torch.randn(num_classes, latent_dim, dtype=torch.float32, device=self.device) * float(
                np.random.uniform(1.2, 2.4)
            )
            latent = centers[y_long] + torch.randn(seq_len, latent_dim, dtype=torch.float32, device=self.device) * float(
                np.random.uniform(0.45, 1.05)
            )
            proj = torch.randn(latent_dim, num_features, dtype=torch.float32, device=self.device)
            x = latent @ proj
            x = x + torch.randn_like(x) * float(np.random.uniform(0.12, 0.55))
            if np.random.random() < 0.25:
                x = torch.sin(x * float(np.random.uniform(0.25, 1.0))) + 0.65 * x
            noise_cols = max(1, int(num_features * np.random.uniform(0.10, 0.40)))
            cols = torch.randperm(num_features, device=self.device)[:noise_cols]
            x[:, cols] = torch.randn(seq_len, noise_cols, dtype=torch.float32, device=self.device)

        elif mode == "sparse_symbolic":
            high_features = max(int(self.min_features), min(int(self.max_features), 100))
            low_features = min(max(20, int(self.min_features)), high_features)
            num_features = int(np.random.randint(low_features, high_features + 1))
            x = torch.zeros(seq_len, num_features, dtype=torch.float32, device=self.device)
            base_density = float(np.random.uniform(0.03, 0.22))
            for col in range(num_features):
                active = torch.rand(seq_len, device=self.device) < base_density
                values = torch.zeros(seq_len, dtype=torch.float32, device=self.device)
                if np.random.random() < 0.55:
                    values[active] = torch.randint(1, int(np.random.randint(2, 16)), (int(active.sum().item()),), device=self.device).float()
                else:
                    rate = math.exp(np.random.uniform(math.log(0.5), math.log(20.0)))
                    values[active] = torch.poisson(torch.full((int(active.sum().item()),), rate, dtype=torch.float32, device=self.device))
                x[:, col] = values
            for cls in range(num_classes):
                cls_mask = y_long == cls
                rows = cls_mask.nonzero(as_tuple=False).flatten()
                if rows.numel() == 0:
                    continue
                cols = torch.randperm(num_features, device=self.device)[: int(np.random.randint(2, min(8, num_features) + 1))]
                x[rows[:, None], cols[None, :]] += torch.randint(
                    1, int(np.random.randint(3, 32)), (rows.numel(), cols.numel()), device=self.device
                ).float()

        elif mode == "mixed_high_cardinality":
            high_features = max(int(self.min_features), min(int(self.max_features), 80))
            low_features = min(max(10, int(self.min_features)), high_features)
            num_features = int(np.random.randint(low_features, high_features + 1))
            x = torch.zeros(seq_len, num_features, dtype=torch.float32, device=self.device)
            cat_cols = set(np.random.choice(num_features, size=max(1, int(num_features * np.random.uniform(0.35, 0.75))), replace=False).tolist())
            class_pref = torch.randint(0, 256, (num_classes, num_features), dtype=torch.int64, device=self.device)
            for col in range(num_features):
                if col in cat_cols:
                    card = int(np.random.randint(4, 256))
                    base = torch.randint(0, card, (seq_len,), dtype=torch.int64, device=self.device)
                    prefer = torch.remainder(class_pref[y_long, col], card)
                    use_pref = torch.rand(seq_len, device=self.device) < float(np.random.uniform(0.20, 0.60))
                    values = torch.where(use_pref, prefer, base).float()
                else:
                    shift = torch.randn(num_classes, dtype=torch.float32, device=self.device) * float(np.random.uniform(0.25, 1.2))
                    values = shift[y_long] + torch.randn(seq_len, dtype=torch.float32, device=self.device) * float(
                        np.random.uniform(0.8, 2.5)
                    )
                    if np.random.random() < 0.35:
                        values = torch.exp(values.clamp(-5.0, 5.0))
                if np.random.random() < 0.08:
                    values[torch.rand(seq_len, device=self.device) < float(np.random.uniform(0.05, 0.30))] = 0.0
                x[:, col] = values

        elif mode == "low_dim_count_multiclass":
            high_features = max(int(self.min_features), min(int(self.max_features), 12))
            low_features = min(max(5, int(self.min_features)), high_features)
            num_features = int(np.random.randint(low_features, high_features + 1))
            x = torch.zeros(seq_len, num_features, dtype=torch.float32, device=self.device)
            categorical_cols = set(
                np.random.choice(num_features, size=max(1, int(num_features * np.random.uniform(0.25, 0.45))), replace=False).tolist()
            )
            zero_cols = set(
                np.random.choice(num_features, size=max(1, int(num_features * np.random.uniform(0.25, 0.55))), replace=False).tolist()
            )
            for col in range(num_features):
                if col in categorical_cols:
                    levels = int(np.random.randint(4, 32))
                    logits = torch.randn(levels, dtype=torch.float32, device=self.device) * float(np.random.uniform(0.4, 1.6))
                    base_prob = torch.softmax(logits, dim=0)
                    idx = torch.multinomial(base_prob, seq_len, replacement=True)
                    values = idx.float()
                else:
                    rate = math.exp(np.random.uniform(math.log(0.5), math.log(500.0)))
                    if np.random.random() < 0.45:
                        shape = float(np.random.uniform(0.4, 2.8))
                        gamma_rate = torch.distributions.Gamma(shape, shape / max(rate, 1e-3)).sample((seq_len,)).to(self.device)
                        values = torch.poisson(gamma_rate.clamp_min(1e-3))
                    else:
                        values = torch.poisson(torch.full((seq_len,), rate, dtype=torch.float32, device=self.device))
                    if np.random.random() < 0.30:
                        values = torch.round(values * float(np.random.uniform(0.5, 4.0)))
                if col in zero_cols:
                    zero_prob = float(np.random.uniform(0.15, 0.65))
                    values[torch.rand(seq_len, device=self.device) < zero_prob] = 0.0
                x[:, col] = values

            # Give mid-frequency classes broad, partially overlapping signatures;
            # keep the rare class more extreme but not tied to any real schema.
            for cls in range(1, num_classes):
                rows = (y_long == cls).nonzero(as_tuple=False).flatten()
                if rows.numel() == 0:
                    continue
                k = int(np.random.randint(1, min(4, num_features) + 1))
                cols = torch.randperm(num_features, device=self.device)[:k]
                if cls == num_classes - 1 and float(counts[cls]) / max(int(counts.sum()), 1) < 0.05:
                    multiplier = float(math.exp(np.random.uniform(math.log(3.0), math.log(60.0))))
                    additive = float(math.exp(np.random.uniform(math.log(32.0), math.log(250_000.0))))
                    noise_scale = float(np.random.uniform(0.05, 0.25))
                else:
                    multiplier = float(math.exp(np.random.uniform(math.log(1.05), math.log(3.5))))
                    additive = float(math.exp(np.random.uniform(math.log(1.0), math.log(512.0))))
                    noise_scale = float(np.random.uniform(0.2, 0.8))
                x[rows[:, None], cols[None, :]] = (
                    x[rows[:, None], cols[None, :]] * multiplier
                    + additive
                    + torch.randn(rows.numel(), cols.numel(), dtype=torch.float32, device=self.device) * noise_scale
                )

            if num_classes >= 3 and np.random.random() < 0.75:
                # Make two mid-frequency classes hard to separate by sharing
                # part of the same low-dimensional signature.
                mid_classes = np.argsort(np.abs(probs - np.median(probs)))[:2]
                a = int(mid_classes[0])
                b = int(mid_classes[1])
                rows_a = (y_long == a).nonzero(as_tuple=False).flatten()
                rows_b = (y_long == b).nonzero(as_tuple=False).flatten()
                common = min(int(rows_a.numel()), int(rows_b.numel()))
                if common > 0:
                    cols = torch.randperm(num_features, device=self.device)[: max(1, min(3, num_features))]
                    rows_a = rows_a[:common]
                    rows_b = rows_b[:common]
                    x[rows_b[:, None], cols[None, :]] = (
                        0.65 * x[rows_b[:, None], cols[None, :]]
                        + 0.35 * x[rows_a[:, None], cols[None, :]]
                        + torch.randn(common, cols.numel(), dtype=torch.float32, device=self.device) * 0.1
                    )

            anomaly_prob = float(np.random.uniform(0.001, 0.018))
            anomaly_mask = torch.rand(seq_len, num_features, device=self.device) < anomaly_prob
            anomaly_count = int(anomaly_mask.sum().item())
            if anomaly_count > 0:
                    x[anomaly_mask] = torch.exp(
                        torch.empty(anomaly_count, dtype=torch.float32, device=self.device).uniform_(
                            math.log(64.0), math.log(1_000_000.0)
                        )
                    )

        elif mode == "bio_multiclass_small_margin":
            high_features = max(int(self.min_features), min(int(self.max_features), 80))
            low_features = min(max(8, int(self.min_features)), high_features)
            num_features = int(np.random.randint(low_features, high_features + 1))
            latent_dim = int(np.random.randint(3, min(10, num_features) + 1))
            centers = torch.randn(num_classes, latent_dim, dtype=torch.float32, device=self.device) * float(
                np.random.uniform(0.12, 0.45)
            )
            latent = centers[y_long] + torch.randn(seq_len, latent_dim, dtype=torch.float32, device=self.device) * float(
                np.random.uniform(0.8, 1.8)
            )
            proj = torch.randn(latent_dim, num_features, dtype=torch.float32, device=self.device)
            x = latent @ proj
            x = x + torch.randn(seq_len, num_features, dtype=torch.float32, device=self.device) * float(
                np.random.uniform(0.7, 2.2)
            )
            if np.random.random() < 0.65:
                block = max(2, min(num_features, int(num_features * np.random.uniform(0.15, 0.45))))
                src = torch.randperm(num_features, device=self.device)[:block]
                dst = torch.randperm(num_features, device=self.device)[:block]
                x[:, dst] = 0.55 * x[:, dst] + 0.45 * x[:, src] + torch.randn(seq_len, block, device=self.device) * 0.15
            if np.random.random() < 0.35:
                noisy_cols = torch.randperm(num_features, device=self.device)[: max(1, int(num_features * 0.25))]
                x[:, noisy_cols] = torch.randn(seq_len, noisy_cols.numel(), dtype=torch.float32, device=self.device)

        elif mode == "noisy_boolean_rule":
            high_features = max(int(self.min_features), min(int(self.max_features), 64))
            low_features = min(max(8, int(self.min_features)), high_features)
            num_features = int(np.random.randint(low_features, high_features + 1))
            x = torch.zeros(seq_len, num_features, dtype=torch.float32, device=self.device)
            class_bit_logits = torch.randn(num_classes, num_features, dtype=torch.float32, device=self.device) * float(
                np.random.uniform(0.2, 1.0)
            )
            for col in range(num_features):
                if np.random.random() < 0.72:
                    probs_t = torch.sigmoid(class_bit_logits[:, col])[y_long]
                    x[:, col] = (torch.rand(seq_len, device=self.device) < probs_t).float()
                else:
                    levels = int(np.random.randint(3, 16))
                    base = torch.randint(0, levels, (seq_len,), dtype=torch.int64, device=self.device).float()
                    offset = torch.remainder(torch.randint(0, levels, (num_classes,), device=self.device)[y_long], levels).float()
                    use_offset = torch.rand(seq_len, device=self.device) < float(np.random.uniform(0.15, 0.55))
                    x[:, col] = torch.where(use_offset, offset, base)
            if num_features >= 4:
                cols = torch.randperm(num_features, device=self.device)[:4]
                parity = torch.remainder(x[:, cols].sum(dim=1), 2.0)
                for cls in range(num_classes):
                    rows = y_long == cls
                    if bool(rows.any()):
                        x[rows, cols[0]] = torch.where(
                            torch.rand(int(rows.sum().item()), device=self.device) < float(np.random.uniform(0.35, 0.70)),
                            torch.full((int(rows.sum().item()),), float(cls % 2), device=self.device),
                            parity[rows],
                        )

        elif mode == "trend_indicator":
            high_features = max(int(self.min_features), min(int(self.max_features), 48))
            low_features = min(max(6, int(self.min_features)), high_features)
            num_features = int(np.random.randint(low_features, high_features + 1))
            t = torch.rand(seq_len, 1, dtype=torch.float32, device=self.device)
            class_slope = torch.randn(num_classes, 1, dtype=torch.float32, device=self.device) * float(np.random.uniform(0.25, 0.9))
            class_phase = torch.rand(num_classes, 1, dtype=torch.float32, device=self.device) * (2.0 * math.pi)
            base = class_slope[y_long] * t + torch.sin(t * float(np.random.uniform(2.0, 8.0)) + class_phase[y_long])
            proj = torch.randn(1, num_features, dtype=torch.float32, device=self.device) * float(np.random.uniform(0.4, 1.2))
            x = base @ proj + torch.randn(seq_len, num_features, dtype=torch.float32, device=self.device) * float(
                np.random.uniform(0.75, 2.0)
            )
            for col in range(num_features):
                if np.random.random() < 0.35:
                    x[:, col] = torch.sign(x[:, col]) * torch.log1p(x[:, col].abs())
                if np.random.random() < 0.20:
                    x[:, col] = torch.round(x[:, col] * float(np.random.uniform(2.0, 8.0)))

        elif mode == "sparse_weak_binary":
            high_features = max(int(self.min_features), min(int(self.max_features), 100))
            low_features = min(max(20, int(self.min_features)), high_features)
            num_features = int(np.random.randint(low_features, high_features + 1))
            base_density = float(np.random.uniform(0.01, 0.12))
            x = (torch.rand(seq_len, num_features, device=self.device) < base_density).float()
            signal_cols = torch.randperm(num_features, device=self.device)[: max(2, int(num_features * np.random.uniform(0.05, 0.18)))]
            pos = y_long == 1
            neg = ~pos
            if bool(pos.any()):
                rows = pos.nonzero(as_tuple=False).flatten()
                x[rows[:, None], signal_cols[None, :]] = (
                    torch.rand(rows.numel(), signal_cols.numel(), device=self.device)
                    < float(np.random.uniform(0.12, 0.38))
                ).float()
            if bool(neg.any()):
                rows = neg.nonzero(as_tuple=False).flatten()
                x[rows[:, None], signal_cols[None, :]] = (
                    torch.rand(rows.numel(), signal_cols.numel(), device=self.device)
                    < float(np.random.uniform(0.02, 0.18))
                ).float()
            count_cols = torch.randperm(num_features, device=self.device)[: max(1, int(num_features * 0.15))]
            x[:, count_cols] = x[:, count_cols] * torch.poisson(
                torch.full((seq_len, count_cols.numel()), float(np.random.uniform(1.0, 12.0)), device=self.device)
            )

        else:
            high_features = max(int(self.min_features), min(int(self.max_features), 40))
            low_features = min(max(4, int(self.min_features)), high_features)
            num_features = int(np.random.randint(low_features, high_features + 1))
            centers = torch.randn(num_classes, num_features, dtype=torch.float32, device=self.device) * float(
                np.random.uniform(0.08, 0.35)
            )
            x = centers[y_long] + torch.randn(seq_len, num_features, dtype=torch.float32, device=self.device) * float(
                np.random.uniform(0.9, 1.8)
            )
            rule_cols = torch.randperm(num_features, device=self.device)[: max(1, min(4, num_features))]
            nonlinear = torch.sin(x[:, rule_cols].sum(dim=1) * float(np.random.uniform(0.5, 2.5)))
            x[:, rule_cols[0]] = x[:, rule_cols[0]] + nonlinear

        label_noise = float(np.random.uniform(0.00, 0.12))
        if mode in {"weak_signal_continuous", "noisy_boolean_rule", "sparse_weak_binary"}:
            label_noise = float(np.random.uniform(0.02, 0.10))
        elif mode in {"sparse_symbolic", "mixed_high_cardinality"}:
            label_noise = float(np.random.uniform(0.00, 0.08))
        elif mode == "many_class_continuous":
            label_noise = float(np.random.uniform(0.00, 0.035))
        elif mode in semantic_modes:
            label_noise = float(np.random.uniform(0.00, 0.025))
        elif mode == "trend_indicator":
            label_noise = float(np.random.uniform(0.00, 0.06))
        elif mode == "bio_multiclass_small_margin":
            label_noise = float(np.random.uniform(0.01, 0.08))
        if label_noise > 0.0:
            noise_mask = torch.rand(seq_len, device=self.device) < label_noise
            noise_count = int(noise_mask.sum().item())
            if noise_count > 0:
                y[noise_mask] = torch.randint(0, num_classes, (noise_count,), dtype=torch.int64, device=self.device).float()

        x = torch.nan_to_num(x, nan=0.0, posinf=1e9, neginf=-1e9).clamp(min=-1e6, max=1e9)
        post = TabICLv2ClassificationGenerator(
            seq_len=seq_len,
            train_size=train_size,
            num_features=num_features,
            max_features=self.max_features,
            max_classes=self.max_classes,
            device=self.device,
            extra_trees_filter=False,
            return_metadata=False,
        )
        x, y = post._postprocess(x, y)
        if not post._fix_split_coverage(x, y):
            raise ValueError("hard diversity stress replay failed train/test class coverage")

        d = torch.tensor(x.shape[1], dtype=torch.long, device=self.device)
        padded_x = _pad_features(x, self.max_features)
        metadata = {
            "prior_kind": "tabiclv2_cls_hard_diversity_stress",
            "hard_diversity_mode": mode,
            "seq_len": int(seq_len),
            "train_size": int(train_size),
            "requested_num_features": int(num_features),
            "final_num_features": int(d.item()),
            "sampled_num_classes": int(num_classes),
            "final_num_classes": int(torch.unique(y).numel()),
            "class_counts_before_shuffle": {str(i): int(counts[i]) for i in range(num_classes)},
            "label_noise": float(label_noise),
        }
        if self.return_metadata:
            return padded_x, y.float(), d, metadata
        return padded_x, y.float(), d

    def _generate_classic_waveform_stress(self, seq_len: int, train_size: int):
        """Generate a classic waveform-style continuous three-class task.

        This is a generic template-mixture generator: three smooth base waves,
        three classes formed by pairwise sums of those bases, and Gaussian
        measurement noise. It targets the common waveform benchmark shape
        without reading or memorizing any evaluation dataset rows.
        """

        num_classes = 3 if int(self.max_classes) >= 3 else 2
        informative = min(21, max(int(self.min_features), int(self.max_features)))
        add_noise_features = np.random.random() < 0.35 and int(self.max_features) >= informative + 6
        if add_noise_features:
            noise_features = int(np.random.randint(6, min(20, int(self.max_features) - informative) + 1))
        else:
            noise_features = 0
        num_features = min(int(self.max_features), informative + noise_features)

        min_per_class = 2
        if seq_len < min_per_class * num_classes:
            num_classes = max(2, seq_len // min_per_class)
        if np.random.random() < 0.75:
            probs = np.random.dirichlet(np.full(num_classes, 12.0, dtype=np.float64))
        else:
            logits = np.random.normal(0.0, 0.35, size=num_classes)
            probs = np.exp(logits - logits.max())
            probs = probs / probs.sum()
        counts = np.random.multinomial(max(seq_len - min_per_class * num_classes, 0), probs) + min_per_class
        counts[0] += seq_len - int(counts.sum())

        y_np = np.concatenate([np.full(int(count), cls, dtype=np.int64) for cls, count in enumerate(counts)])
        np.random.shuffle(y_np)
        y = torch.tensor(y_np, dtype=torch.float32, device=self.device)
        y_long = y.long()

        grid = torch.arange(informative, dtype=torch.float32, device=self.device)

        def triangular(center: float, width: float) -> Tensor:
            return torch.clamp(1.0 - torch.abs(grid - center) / max(width, 1e-3), min=0.0)

        scale = float(np.random.uniform(3.0, 5.5))
        h1 = scale * triangular(float(np.random.uniform(5.0, 7.0)), float(np.random.uniform(4.5, 6.5)))
        h2 = scale * triangular(float(np.random.uniform(13.0, 15.0)), float(np.random.uniform(4.5, 6.5)))
        h3 = scale * triangular(float(np.random.uniform(9.0, 11.0)), float(np.random.uniform(5.0, 8.0)))
        bases = torch.stack([h1, h2, h3], dim=0)
        bases = bases + torch.randn_like(bases) * float(np.random.uniform(0.01, 0.08))

        if num_classes == 2:
            templates = torch.stack([bases[0] + bases[1], bases[1] + bases[2]], dim=0)
        else:
            templates = torch.stack([bases[0] + bases[1], bases[0] + bases[2], bases[1] + bases[2]], dim=0)

        row_scale = torch.empty(seq_len, 1, dtype=torch.float32, device=self.device).uniform_(0.90, 1.10)
        row_shift = torch.randn(seq_len, 1, dtype=torch.float32, device=self.device) * float(np.random.uniform(0.0, 0.12))
        noise_sigma = float(np.random.uniform(0.75, 1.35))
        if np.random.random() < 0.20:
            noise_sigma = float(np.random.uniform(1.35, 1.80))
        x_info = templates[y_long] * row_scale + row_shift
        x_info = x_info + torch.randn(seq_len, informative, dtype=torch.float32, device=self.device) * noise_sigma

        if num_features > informative:
            nuisance = torch.randn(seq_len, num_features - informative, dtype=torch.float32, device=self.device)
            nuisance = nuisance * float(np.random.uniform(0.8, 1.6))
            if np.random.random() < 0.35:
                weak_cols = min(nuisance.shape[1], int(np.random.randint(1, min(5, nuisance.shape[1]) + 1)))
                src = torch.randint(0, informative, (weak_cols,), dtype=torch.long, device=self.device)
                coef = torch.empty(weak_cols, dtype=torch.float32, device=self.device).uniform_(0.05, 0.20)
                nuisance[:, :weak_cols] = nuisance[:, :weak_cols] + x_info[:, src] * coef
            x = torch.cat([x_info, nuisance], dim=1)
        else:
            x = x_info

        label_noise = float(np.random.uniform(0.0, 0.025))
        noise_mask = torch.rand(seq_len, device=self.device) < label_noise
        noise_count = int(noise_mask.sum().item())
        if noise_count > 0:
            y[noise_mask] = torch.randint(0, num_classes, (noise_count,), dtype=torch.int64, device=self.device).float()

        perm = torch.randperm(num_features, device=self.device)
        x = x[:, perm]

        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6).clamp(min=-1e6, max=1e6)
        post = TabICLv2ClassificationGenerator(
            seq_len=seq_len,
            train_size=train_size,
            num_features=num_features,
            max_features=self.max_features,
            max_classes=self.max_classes,
            device=self.device,
            extra_trees_filter=False,
            return_metadata=False,
        )
        x, y = post._postprocess(x, y)
        if not post._fix_split_coverage(x, y):
            raise ValueError("classic waveform stress replay failed train/test class coverage")

        d = torch.tensor(x.shape[1], dtype=torch.long, device=self.device)
        padded_x = _pad_features(x, self.max_features)
        metadata = {
            "prior_kind": "tabiclv2_cls_classic_waveform_stress",
            "seq_len": int(seq_len),
            "train_size": int(train_size),
            "requested_num_features": int(num_features),
            "final_num_features": int(d.item()),
            "informative_features": int(informative),
            "sampled_num_classes": int(num_classes),
            "final_num_classes": int(torch.unique(y).numel()),
            "class_counts_before_shuffle": {str(i): int(counts[i]) for i in range(num_classes)},
            "noise_sigma": float(noise_sigma),
            "label_noise": float(label_noise),
            "classic_waveform_stress": True,
        }
        if self.return_metadata:
            return padded_x, y.float(), d, metadata
        return padded_x, y.float(), d

    def _generate_waveform_low_snr_stress(self, seq_len: int, train_size: int):
        """Generate generic waveform-like low-SNR continuous multiclass tasks.

        This covers broad low-signal regimes such as overlapping continuous
        class templates, many nuisance dimensions, correlated redundant columns,
        and modest label noise. It intentionally does not read or encode any
        real evaluation dataset.
        """

        max_classes = max(2, min(int(self.max_classes), 5))
        num_classes = int(np.random.randint(3, max_classes + 1)) if max_classes >= 3 else 2
        max_features = max(int(self.min_features), min(int(self.max_features), 80))
        low_features = min(max(18, int(self.min_features)), max_features)
        num_features = int(np.random.randint(low_features, max_features + 1))
        informative = int(np.random.randint(min(12, num_features), min(32, num_features) + 1))

        min_per_class = 2
        if seq_len < min_per_class * num_classes:
            num_classes = max(2, seq_len // min_per_class)
        if np.random.random() < 0.55:
            probs = np.random.dirichlet(np.full(num_classes, 2.5, dtype=np.float64))
        else:
            probs = self._sample_count_stress_class_probs(num_classes)
            probs = 0.65 * probs + 0.35 / num_classes
            probs = probs / probs.sum()
        remaining = max(seq_len - min_per_class * num_classes, 0)
        counts = np.random.multinomial(remaining, probs) + min_per_class
        counts[0] += seq_len - int(counts.sum())
        y_np = np.concatenate([np.full(int(count), cls, dtype=np.int64) for cls, count in enumerate(counts)])
        np.random.shuffle(y_np)
        y = torch.tensor(y_np, dtype=torch.float32, device=self.device)
        y_long = y.long()

        grid = torch.linspace(-1.0, 1.0, informative, dtype=torch.float32, device=self.device)
        templates = []
        base_centers = torch.linspace(-0.65, 0.65, num_classes, dtype=torch.float32, device=self.device)
        for cls in range(num_classes):
            center = base_centers[cls] + torch.randn((), dtype=torch.float32, device=self.device) * 0.18
            width = float(np.random.uniform(0.28, 0.75))
            phase = float(np.random.uniform(-math.pi, math.pi))
            freq = float(np.random.uniform(0.7, 2.8))
            amp = float(np.random.uniform(0.45, 1.25))
            bump = torch.exp(-0.5 * ((grid - center) / width) ** 2)
            shoulder = torch.exp(-0.5 * ((grid + center * 0.55) / (width * 1.35)) ** 2)
            wave = torch.sin(grid * math.pi * freq + phase)
            template = amp * (bump - 0.55 * shoulder) + float(np.random.uniform(0.05, 0.30)) * wave
            templates.append(template)
        template_t = torch.stack(templates, dim=0)

        row_scale = torch.empty(seq_len, 1, dtype=torch.float32, device=self.device).uniform_(0.65, 1.35)
        row_shift = torch.randn(seq_len, 1, dtype=torch.float32, device=self.device) * float(np.random.uniform(0.02, 0.18))
        signal = template_t[y_long] * row_scale + row_shift
        noise_sigma = float(np.random.uniform(0.9, 2.6))
        x_info = signal + torch.randn(seq_len, informative, dtype=torch.float32, device=self.device) * noise_sigma

        if np.random.random() < 0.55:
            warp = float(np.random.uniform(0.15, 0.55))
            x_info = x_info + warp * torch.sin(x_info * float(np.random.uniform(0.7, 1.8)))

        x = torch.randn(seq_len, num_features, dtype=torch.float32, device=self.device) * float(np.random.uniform(0.8, 2.4))
        x[:, :informative] = x_info

        # Redundant columns are noisy linear combinations of weak informative signals.
        redundant = max(0, min(num_features - informative, int(num_features * np.random.uniform(0.10, 0.35))))
        if redundant > 0:
            src = torch.randint(0, informative, (redundant,), dtype=torch.long, device=self.device)
            dst = torch.arange(informative, informative + redundant, dtype=torch.long, device=self.device)
            coef = torch.empty(redundant, dtype=torch.float32, device=self.device).uniform_(0.35, 1.25)
            x[:, dst] = x[:, src] * coef + torch.randn(seq_len, redundant, dtype=torch.float32, device=self.device) * float(
                np.random.uniform(0.5, 1.6)
            )

        # Some columns are discretized/ordinal views, mimicking lossy extracted waveform features.
        if np.random.random() < 0.45:
            num_disc = int(np.random.randint(1, max(2, min(8, num_features)) + 1))
            cols = torch.randperm(num_features, device=self.device)[:num_disc]
            levels = float(np.random.randint(4, 16))
            vals = x[:, cols]
            vals = torch.round((vals - vals.mean(dim=0, keepdim=True)) / vals.std(dim=0, keepdim=True).clamp_min(1e-3) * 1.5 + levels / 2)
            x[:, cols] = vals.clamp(0.0, levels - 1.0)

        # Missing-like zeros and mild heavy-tailed outliers make the task less clean.
        if np.random.random() < 0.35:
            zero_cols = torch.randperm(num_features, device=self.device)[: max(1, int(num_features * np.random.uniform(0.05, 0.18)))]
            zero_mask = torch.rand(seq_len, zero_cols.numel(), device=self.device) < float(np.random.uniform(0.03, 0.18))
            x[:, zero_cols] = torch.where(zero_mask, torch.zeros_like(x[:, zero_cols]), x[:, zero_cols])
        outlier_prob = float(np.random.uniform(0.0005, 0.008))
        outlier_mask = torch.rand(seq_len, num_features, device=self.device) < outlier_prob
        outlier_count = int(outlier_mask.sum().item())
        if outlier_count > 0:
            x[outlier_mask] = x[outlier_mask] + torch.randn(outlier_count, dtype=torch.float32, device=self.device) * float(
                np.random.uniform(8.0, 35.0)
            )

        label_noise = float(np.random.uniform(0.02, 0.16))
        noise_mask = torch.rand(seq_len, device=self.device) < label_noise
        noise_count = int(noise_mask.sum().item())
        if noise_count > 0:
            y[noise_mask] = torch.randint(0, num_classes, (noise_count,), dtype=torch.int64, device=self.device).float()

        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6).clamp(min=-1e6, max=1e6)
        post = TabICLv2ClassificationGenerator(
            seq_len=seq_len,
            train_size=train_size,
            num_features=num_features,
            max_features=self.max_features,
            max_classes=self.max_classes,
            device=self.device,
            extra_trees_filter=False,
            return_metadata=False,
        )
        x, y = post._postprocess(x, y)
        if not post._fix_split_coverage(x, y):
            raise ValueError("waveform/low-SNR stress replay failed train/test class coverage")

        d = torch.tensor(x.shape[1], dtype=torch.long, device=self.device)
        padded_x = _pad_features(x, self.max_features)
        metadata = {
            "prior_kind": "tabiclv2_cls_waveform_low_snr_stress",
            "seq_len": int(seq_len),
            "train_size": int(train_size),
            "requested_num_features": int(num_features),
            "final_num_features": int(d.item()),
            "informative_features": int(informative),
            "sampled_num_classes": int(num_classes),
            "final_num_classes": int(torch.unique(y).numel()),
            "class_counts_before_shuffle": {str(i): int(counts[i]) for i in range(num_classes)},
            "noise_sigma": float(noise_sigma),
            "label_noise": float(label_noise),
            "waveform_low_snr_stress": True,
        }
        if self.return_metadata:
            return padded_x, y.float(), d, metadata
        return padded_x, y.float(), d

    def _generate_count_imbalance_stress(self, seq_len: int, train_size: int):
        max_classes = max(2, int(self.max_classes))
        num_classes = int(np.random.randint(2, max_classes + 1))

        max_features = max(int(self.min_features), min(int(self.max_features), 40))
        min_features = min(max(3, int(self.min_features)), max_features)
        num_features = int(np.random.randint(min_features, max_features + 1))

        min_per_class = 2
        if seq_len < min_per_class * num_classes:
            num_classes = max(2, seq_len // min_per_class)
        probs = self._sample_count_stress_class_probs(num_classes)
        remaining = max(seq_len - min_per_class * num_classes, 0)
        counts = np.random.multinomial(remaining, probs) + min_per_class
        counts[0] += seq_len - int(counts.sum())

        y_np = np.concatenate([np.full(int(count), cls, dtype=np.int64) for cls, count in enumerate(counts)])
        np.random.shuffle(y_np)
        y = torch.tensor(y_np, dtype=torch.float32, device=self.device)
        y_long = y.long()

        x = torch.zeros(seq_len, num_features, dtype=torch.float32, device=self.device)
        class_scales = torch.exp(
            torch.randn(num_classes, num_features, dtype=torch.float32, device=self.device) * 0.8
        )
        class_offsets = torch.randn(num_classes, num_features, dtype=torch.float32, device=self.device) * 0.25

        for col in range(num_features):
            family = np.random.choice(["poisson", "gamma_poisson", "lognormal", "ordinal"])
            zero_prob = float(np.random.beta(1.2, 3.0))
            if np.random.random() < 0.30:
                zero_prob = float(np.random.uniform(0.35, 0.80))
            base_rate = math.exp(np.random.uniform(math.log(0.4), math.log(80.0)))
            values = torch.empty(seq_len, dtype=torch.float32, device=self.device)

            for cls in range(num_classes):
                cls_mask = y_long == cls
                cls_count = int(cls_mask.sum().item())
                if cls_count == 0:
                    continue
                rate = base_rate * float(class_scales[cls, col].item())
                if family == "poisson":
                    cls_values = torch.poisson(torch.full((cls_count,), rate, dtype=torch.float32, device=self.device))
                elif family == "gamma_poisson":
                    gamma_shape = float(np.random.uniform(0.6, 3.0))
                    gamma_rate = torch.distributions.Gamma(gamma_shape, gamma_shape / max(rate, 1e-3)).sample(
                        (cls_count,)
                    ).to(device=self.device)
                    cls_values = torch.poisson(gamma_rate.clamp_min(1e-3))
                elif family == "lognormal":
                    sigma = float(np.random.uniform(0.5, 1.8))
                    mean = math.log(max(rate, 1e-3)) - 0.5 * sigma * sigma
                    cls_values = torch.exp(
                        torch.randn(cls_count, dtype=torch.float32, device=self.device) * sigma + mean
                    )
                    if np.random.random() < 0.7:
                        cls_values = torch.round(cls_values)
                else:
                    levels = int(np.random.randint(3, 12))
                    cls_values = torch.randint(0, levels, (cls_count,), dtype=torch.int64, device=self.device).float()
                    cls_values = cls_values * float(np.random.uniform(1.0, 16.0))

                cls_values = cls_values + class_offsets[cls, col]
                values[cls_mask] = cls_values

            zero_mask = torch.rand(seq_len, device=self.device) < zero_prob
            values[zero_mask] = 0.0
            outlier_prob = float(np.random.uniform(0.001, 0.02))
            outlier_mask = torch.rand(seq_len, device=self.device) < outlier_prob
            outlier_count = int(outlier_mask.sum().item())
            if outlier_count > 0:
                tail = torch.exp(
                    torch.empty(outlier_count, dtype=torch.float32, device=self.device).uniform_(
                        math.log(50.0), math.log(2_000_000.0)
                    )
                )
                values[outlier_mask] = values[outlier_mask].abs() + tail
            x[:, col] = values

        num_signal_classes = int(np.random.randint(1, num_classes + 1))
        for cls in np.random.choice(num_classes, size=num_signal_classes, replace=False):
            cls_mask = y_long == int(cls)
            cls_count = int(cls_mask.sum().item())
            if cls_count == 0:
                continue
            k = int(np.random.randint(1, min(5, num_features) + 1))
            cols = np.random.choice(num_features, size=k, replace=False)
            for col in cols:
                if np.random.random() < 0.5:
                    x[cls_mask, col] = x[cls_mask, col] * float(math.exp(np.random.uniform(-0.7, 1.2)))
                else:
                    shift = math.exp(np.random.uniform(math.log(0.5), math.log(128.0)))
                    x[cls_mask, col] = x[cls_mask, col] + shift

        if num_classes >= 3 and np.random.random() < 0.35:
            # Sometimes make two minority classes deliberately overlapping so
            # the stress task is not only a rare-class memorization exercise.
            rare_classes = np.argsort(counts)[:2]
            source_mask = y_long == int(rare_classes[0])
            target_mask = y_long == int(rare_classes[1])
            if bool(source_mask.any()) and bool(target_mask.any()):
                common = min(int(source_mask.sum().item()), int(target_mask.sum().item()))
                cols = np.random.choice(num_features, size=max(1, min(3, num_features)), replace=False)
                noise = torch.randn(common, len(cols), dtype=torch.float32, device=self.device) * 0.05
                source_rows = source_mask.nonzero(as_tuple=False).flatten()[:common]
                target_rows = target_mask.nonzero(as_tuple=False).flatten()[:common]
                col_idx = torch.tensor(cols, dtype=torch.long, device=self.device)
                x[target_rows[:, None], col_idx[None, :]] = (
                    x[source_rows[:, None], col_idx[None, :]] * (1.0 + noise)
                )

        x = torch.nan_to_num(x, nan=0.0, posinf=1e9, neginf=-1e9).clamp(min=-1e6, max=1e9)
        post = TabICLv2ClassificationGenerator(
            seq_len=seq_len,
            train_size=train_size,
            num_features=num_features,
            max_features=self.max_features,
            max_classes=self.max_classes,
            device=self.device,
            extra_trees_filter=False,
            return_metadata=False,
        )
        x, y = post._postprocess(x, y)
        if not post._fix_split_coverage(x, y):
            raise ValueError("count/imbalance stress replay failed train/test class coverage")

        d = torch.tensor(x.shape[1], dtype=torch.long, device=self.device)
        padded_x = _pad_features(x, self.max_features)
        metadata = {
            "prior_kind": "tabiclv2_cls_count_imbalance_stress",
            "seq_len": int(seq_len),
            "train_size": int(train_size),
            "requested_num_features": int(num_features),
            "final_num_features": int(d.item()),
            "sampled_num_classes": int(num_classes),
            "final_num_classes": int(torch.unique(y).numel()),
            "class_counts_before_shuffle": {str(i): int(counts[i]) for i in range(num_classes)},
            "min_class_fraction": float(counts.min() / max(int(counts.sum()), 1)),
            "max_class_fraction": float(counts.max() / max(int(counts.sum()), 1)),
            "zero_inflated_count_stress": True,
        }
        if self.return_metadata:
            return padded_x, y.float(), d, metadata
        return padded_x, y.float(), d

    def _generate_network_count_stress(self, seq_len: int, train_size: int):
        """Generate generic network-flow-like count data without reading real datasets."""

        max_classes = max(2, min(int(self.max_classes), 5))
        num_classes = int(np.random.choice(np.arange(3, max_classes + 1))) if max_classes >= 3 else 2
        max_features = max(2, int(self.max_features))
        high_features = min(max_features, 12)
        min_features = min(max(5, int(self.min_features)), high_features)
        num_features = int(np.random.randint(min_features, high_features + 1))

        if num_classes >= 4:
            majority = float(np.random.uniform(0.45, 0.72))
            mid_a = float(np.random.uniform(0.10, 0.30))
            mid_b = float(np.random.uniform(0.08, 0.28))
            rare = float(math.exp(np.random.uniform(math.log(0.0008), math.log(0.025))))
            probs = np.array([majority, mid_a, mid_b, rare], dtype=np.float64)
            if num_classes > 4:
                extras = np.random.dirichlet(np.ones(num_classes - 4)) * float(np.random.uniform(0.03, 0.18))
                probs = np.concatenate([probs, extras])
        else:
            majority = float(np.random.uniform(0.50, 0.78))
            minority = float(np.random.uniform(0.10, 0.35))
            probs = np.array([majority, minority, max(1e-4, 1.0 - majority - minority)], dtype=np.float64)
        probs = np.clip(probs[:num_classes], 1e-6, None)
        probs = probs / probs.sum()

        min_per_class = 2
        if seq_len < min_per_class * num_classes:
            num_classes = max(2, seq_len // min_per_class)
            probs = probs[:num_classes] / probs[:num_classes].sum()
        counts = np.random.multinomial(max(seq_len - min_per_class * num_classes, 0), probs) + min_per_class
        counts[0] += seq_len - int(counts.sum())
        y_np = np.concatenate([np.full(int(count), cls, dtype=np.int64) for cls, count in enumerate(counts)])
        np.random.shuffle(y_np)
        y = torch.tensor(y_np, dtype=torch.float32, device=self.device)
        y_long = y.long()

        x = torch.zeros(seq_len, num_features, dtype=torch.float32, device=self.device)
        protocol_cols = set(np.random.choice(num_features, size=max(1, num_features // 4), replace=False).tolist())
        zero_cols = set(np.random.choice(num_features, size=max(1, num_features // 3), replace=False).tolist())
        for col in range(num_features):
            if col in protocol_cols:
                levels = torch.tensor(
                    sorted(np.random.choice(np.arange(1, 257), size=int(np.random.randint(4, 11)), replace=False)),
                    dtype=torch.float32,
                    device=self.device,
                )
                logits = torch.randn(levels.numel(), dtype=torch.float32, device=self.device) * 1.5
                probs_t = torch.softmax(logits, dim=0)
                idx = torch.multinomial(probs_t, seq_len, replacement=True)
                values = levels[idx]
            else:
                rate = math.exp(np.random.uniform(math.log(0.5), math.log(400.0)))
                if np.random.random() < 0.45:
                    shape = float(np.random.uniform(0.5, 3.0))
                    gamma_rate = torch.distributions.Gamma(shape, shape / rate).sample((seq_len,)).to(self.device)
                    values = torch.poisson(gamma_rate.clamp_min(1e-3))
                else:
                    values = torch.poisson(torch.full((seq_len,), rate, dtype=torch.float32, device=self.device))
                if np.random.random() < 0.35:
                    values = values + torch.randint(0, int(np.random.randint(2, 32)), (seq_len,), device=self.device).float()

            if col in zero_cols:
                zero_prob = float(np.random.uniform(0.20, 0.70))
                values[torch.rand(seq_len, device=self.device) < zero_prob] = 0.0
            x[:, col] = values

        # Medium-frequency minority classes get separable but overlapping shifts.
        for cls in range(1, num_classes):
            cls_mask = y_long == cls
            cls_count = int(cls_mask.sum().item())
            if cls_count == 0:
                continue
            k = int(np.random.randint(1, min(4, num_features) + 1))
            cols = np.random.choice(num_features, size=k, replace=False)
            col_idx = torch.tensor(cols, dtype=torch.long, device=self.device)
            if cls == num_classes - 1 and counts[cls] / max(seq_len, 1) < 0.04:
                multiplier = float(math.exp(np.random.uniform(math.log(4.0), math.log(80.0))))
                additive = float(math.exp(np.random.uniform(math.log(64.0), math.log(200_000.0))))
            else:
                multiplier = float(math.exp(np.random.uniform(math.log(1.15), math.log(4.0))))
                additive = float(math.exp(np.random.uniform(math.log(2.0), math.log(512.0))))
            rows = cls_mask.nonzero(as_tuple=False).flatten()
            x[rows[:, None], col_idx[None, :]] = x[rows[:, None], col_idx[None, :]] * multiplier + additive

        # Add independent heavy-tailed anomalies to avoid a single fixed signature.
        anomaly_prob = float(np.random.uniform(0.002, 0.02))
        anomaly_mask = torch.rand(seq_len, num_features, device=self.device) < anomaly_prob
        anomaly_count = int(anomaly_mask.sum().item())
        if anomaly_count > 0:
            x[anomaly_mask] = torch.exp(
                torch.empty(anomaly_count, dtype=torch.float32, device=self.device).uniform_(
                    math.log(128.0), math.log(1_000_000.0)
                )
            )
        post = TabICLv2ClassificationGenerator(
            seq_len=seq_len,
            train_size=train_size,
            num_features=num_features,
            max_features=self.max_features,
            max_classes=self.max_classes,
            device=self.device,
            extra_trees_filter=False,
            return_metadata=False,
        )
        x, y = post._postprocess(x, y)
        if not post._fix_split_coverage(x, y):
            raise ValueError("network count stress replay failed train/test class coverage")

        d = torch.tensor(x.shape[1], dtype=torch.long, device=self.device)
        padded_x = _pad_features(x, self.max_features)
        metadata = {
            "prior_kind": "tabiclv2_cls_network_count_stress",
            "seq_len": int(seq_len),
            "train_size": int(train_size),
            "requested_num_features": int(num_features),
            "final_num_features": int(d.item()),
            "sampled_num_classes": int(num_classes),
            "final_num_classes": int(torch.unique(y).numel()),
            "class_counts_before_shuffle": {str(i): int(counts[i]) for i in range(num_classes)},
            "min_class_fraction": float(counts.min() / max(int(counts.sum()), 1)),
            "max_class_fraction": float(counts.max() / max(int(counts.sum()), 1)),
            "network_count_stress": True,
        }
        if self.return_metadata:
            return padded_x, y.float(), d, metadata
        return padded_x, y.float(), d

    def _generate_firewall_like(self, seq_len: int, train_size: int):
        num_classes = min(4, int(self.max_classes))
        if num_classes < 2:
            raise ValueError("firewall-like replay requires at least 2 classes")

        if num_classes == 4:
            probs = np.array([0.57435, 0.22872, 0.19609, 0.00084], dtype=np.float64)
        else:
            probs = np.geomspace(1.0, 0.1, num_classes).astype(np.float64)
        probs = probs / probs.sum()

        min_per_class = 2
        remaining = max(seq_len - min_per_class * num_classes, 0)
        counts = np.random.multinomial(remaining, probs) + min_per_class
        counts[0] += seq_len - int(counts.sum())
        y_np = np.concatenate([np.full(int(count), cls, dtype=np.int64) for cls, count in enumerate(counts)])
        np.random.shuffle(y_np)
        y = torch.tensor(y_np, dtype=torch.float32, device=self.device)
        y_long = y.long()

        x = torch.zeros(seq_len, 7, dtype=torch.float32, device=self.device)

        def sample_values(values: Sequence[float], probs_: Sequence[float], n: int) -> Tensor:
            value_t = torch.tensor(values, dtype=torch.float32, device=self.device)
            prob_t = torch.tensor(probs_, dtype=torch.float32, device=self.device)
            prob_t = prob_t / prob_t.sum().clamp_min(1e-12)
            idx = torch.multinomial(prob_t, n, replacement=True)
            return value_t[idx]

        n = int(seq_len)
        x[:, 0] = sample_values([60, 62, 66, 70, 146, 168, 177, 199, 256, 512], [0.04, 0.14, 0.22, 0.26, 0.04, 0.03, 0.02, 0.02, 0.12, 0.11], n)
        x[:, 1] = sample_values([60, 62, 66, 70, 86, 94, 102, 110, 256, 512], [0.04, 0.13, 0.22, 0.26, 0.04, 0.09, 0.08, 0.03, 0.06, 0.05], n)
        zero_payload = torch.rand(n, device=self.device) < 0.48
        x[:, 2] = sample_values([74, 83, 89, 90, 91, 93, 97, 146, 512, 1024], [0.09, 0.18, 0.12, 0.16, 0.11, 0.09, 0.12, 0.05, 0.04, 0.04], n)
        x[zero_payload, 2] = 0.0
        x[:, 3] = sample_values([1, 2, 4, 6, 10, 18, 19, 20, 64, 128], [0.46, 0.26, 0.03, 0.02, 0.02, 0.01, 0.015, 0.015, 0.09, 0.09], n)
        zero_duration = torch.rand(n, device=self.device) < 0.43
        x[:, 4] = sample_values([5, 8, 15, 16, 29, 30, 31, 64, 128, 256], [0.04, 0.03, 0.05, 0.03, 0.07, 0.31, 0.10, 0.12, 0.13, 0.12], n)
        x[zero_duration, 4] = 0.0
        x[:, 5] = sample_values([1, 2, 3, 5, 6, 7, 9, 11, 32, 128], [0.69, 0.045, 0.025, 0.02, 0.018, 0.018, 0.018, 0.02, 0.08, 0.066], n)
        x[:, 6] = sample_values([1, 2, 3, 5, 7, 8, 9, 16, 64, 256], [0.31, 0.04, 0.025, 0.02, 0.02, 0.025, 0.025, 0.08, 0.18, 0.275], n)
        x[zero_payload, 6] = 0.0

        class1 = y_long == 1
        class2 = y_long == 2
        class3 = y_long == 3
        if class1.any():
            x[class1, 3] += sample_values([1, 2, 8, 16, 64], [0.35, 0.25, 0.15, 0.15, 0.10], int(class1.sum().item()))
            x[class1, 4] += sample_values([0, 15, 30, 60, 120], [0.35, 0.20, 0.25, 0.10, 0.10], int(class1.sum().item()))
        if class2.any():
            x[class2, 0] += sample_values([0, 8, 64, 128, 512], [0.45, 0.20, 0.18, 0.12, 0.05], int(class2.sum().item()))
            x[class2, 2] += sample_values([0, 16, 64, 256, 1024], [0.45, 0.22, 0.18, 0.10, 0.05], int(class2.sum().item()))
            x[class2, 6] += sample_values([0, 1, 8, 64, 256], [0.40, 0.25, 0.15, 0.12, 0.08], int(class2.sum().item()))
        if class3.any():
            m = int(class3.sum().item())
            x[class3, 0] = sample_values([4096, 65535, 1000000, 10000000], [0.30, 0.35, 0.25, 0.10], m)
            x[class3, 1] = sample_values([4096, 65535, 1000000, 10000000], [0.30, 0.35, 0.25, 0.10], m)
            x[class3, 3] = sample_values([64, 128, 512, 2048], [0.35, 0.30, 0.25, 0.10], m)
            x[class3, 5] = sample_values([32, 128, 512, 2048], [0.40, 0.30, 0.20, 0.10], m)

        for col, prob, low, high in [
            (0, 0.010, 2000, 1_300_000_000),
            (1, 0.007, 2000, 950_000_000),
            (2, 0.006, 1000, 330_000_000),
            (3, 0.004, 128, 1_100_000),
            (4, 0.004, 128, 12_000),
            (5, 0.003, 128, 760_000),
            (6, 0.004, 128, 340_000),
        ]:
            mask = torch.rand(n, device=self.device) < prob
            count = int(mask.sum().item())
            if count > 0:
                x[mask, col] = torch.randint(low, high + 1, (count,), dtype=torch.int64, device=self.device).float()
        post = TabICLv2ClassificationGenerator(
            seq_len=seq_len,
            train_size=train_size,
            num_features=7,
            max_features=self.max_features,
            max_classes=self.max_classes,
            device=self.device,
            extra_trees_filter=False,
            return_metadata=False,
        )
        x, y = post._postprocess(x, y)
        if not post._fix_split_coverage(x, y):
            raise ValueError("firewall-like replay failed train/test class coverage")

        d = torch.tensor(x.shape[1], dtype=torch.long, device=self.device)
        padded_x = _pad_features(x, self.max_features)
        metadata = {
            "prior_kind": "tabiclv2_cls_firewall_like",
            "seq_len": int(seq_len),
            "train_size": int(train_size),
            "requested_num_features": 7,
            "final_num_features": int(d.item()),
            "sampled_num_classes": int(num_classes),
            "final_num_classes": int(torch.unique(y).numel()),
            "class_counts_before_shuffle": {str(i): int(counts[i]) for i in range(num_classes)},
            "integer_spike_features": True,
            "zero_inflated_features": [2, 4, 6],
            "rare_class_fraction": float(probs[-1]),
        }
        if self.return_metadata:
            return padded_x, y.float(), d, metadata
        return padded_x, y.float(), d

    def get_batch(self, batch_size: Optional[int] = None):
        batch_size = batch_size or self.batch_size
        x_list: List[Tensor] = []
        y_list: List[Tensor] = []
        d_list: List[Tensor] = []
        seq_lens: List[int] = []
        train_sizes: List[int] = []
        metadata_list: List[Dict[str, Any]] = []

        size_per_gp = min(self.batch_size_per_gp, batch_size)
        num_groups = math.ceil(batch_size / size_per_gp)
        global_seq_len = None
        global_train_size = None
        global_num_features = None
        if not self.seq_len_per_gp:
            global_seq_len = self.sample_seq_len(self.min_seq_len, self.max_seq_len, self.log_seq_len, self.replay_small)
            global_train_size = self.sample_train_size(self.min_train_size, self.max_train_size, global_seq_len)
            global_num_features = self._sample_base_num_features()

        for gp_idx in range(num_groups):
            group_size = min(size_per_gp, batch_size - gp_idx * size_per_gp)
            if self.seq_len_per_gp:
                seq_len = self.sample_seq_len(self.min_seq_len, self.max_seq_len, self.log_seq_len, self.replay_small)
                train_size = self.sample_train_size(self.min_train_size, self.max_train_size, seq_len)
                num_features = self._sample_base_num_features()
            else:
                seq_len = int(global_seq_len)
                train_size = int(global_train_size)
                num_features = int(global_num_features)
            for _ in range(group_size):
                generated = self._generate_one(seq_len, train_size, num_features)
                if self.return_metadata:
                    x, y, d, metadata = generated
                    metadata_list.append(metadata)
                else:
                    x, y, d = generated
                x_list.append(x)
                y_list.append(y)
                d_list.append(d)
                seq_lens.append(seq_len)
                train_sizes.append(train_size)

        if self.seq_len_per_gp:
            x_batch = nested_tensor([x.to(self.device) for x in x_list], device=self.device)
            y_batch = nested_tensor([y.to(self.device) for y in y_list], device=self.device)
        else:
            x_batch = torch.stack(x_list).to(self.device)
            y_batch = torch.stack(y_list).to(self.device)
        d_batch = torch.stack(d_list).to(self.device)
        seq_lens_t = torch.tensor(seq_lens, dtype=torch.long, device=self.device)
        train_sizes_t = torch.tensor(train_sizes, dtype=torch.long, device=self.device)
        if self.return_metadata:
            return x_batch, y_batch, d_batch, seq_lens_t, train_sizes_t, metadata_list
        return x_batch, y_batch, d_batch, seq_lens_t, train_sizes_t
