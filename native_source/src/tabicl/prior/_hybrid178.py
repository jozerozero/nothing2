"""Benchmark-conditioned statistical prior for the 178 datasets.

The official TabICLv2 GraphSCM is the causal backbone.  Profiles are scheduled
globally so every data178 template is represented in each sufficiently large
DDP batch, but the dominant ``official_shape`` branch deliberately treats that
profile as coverage metadata only.  This prevents the finite set of 178
statistical templates from defining the entire training distribution.  The
profile-conditioned branches use either:

* an official GraphSCM copula plus class-conditional empirical marginals, or
* a fitted class-conditional Gaussian copula.

An optional, explicitly disclosed empirical-atom mixture can replace a small
number of generated rows with exact preprocessed GT-train rows.  This is the
only mathematically valid way to give exact GT rows positive probability.  It
is disabled by default and is never described as synthetic-only when enabled.

The production Stage-1 mode is stricter: an offline compiler first creates a
frozen synthetic surrogate bank, then ``runtime_isolated`` generation reads
only that bank.  The isolated mode verifies the compiler manifest, rejects
overlapping/symlinked roots, guards every NumPy load against the forbidden GT
root, disables empirical atoms, and applies supervised statistical quality
checks to the profile-conditioned branches.  The checks cover class-conditional
marginals, feature-label mutual information, higher-order dependence, task
difficulty and cross-task diversity.  Thus no real row is available to the
training-time generator.

Datasets with more than ten labels are deterministically merged into ten
synthetic class groups.  Generated tasks are also hard-limited to one hundred
features.  Wider source profiles are represented by rotating row-free
statistical views whose union covers every source feature.  Test files are
never discovered or opened, and the Stage-1 configuration uses train only.
Results produced with this prior must be reported as benchmark-conditioned
synthetic augmentation.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm
import torch
from torch import Tensor
from torch.nested import nested_tensor
from torch.utils.data import get_worker_info

from ._graph_scm import GraphSCM
from .graph_lib._config import PriorConfig
from ._protected_batch_mix import (
    PROTECTED_DATASET_NAMES as HYBRID178_PROTECTED_PROFILE_NAMES,
)
from ._tabiclv2_classification import TabICLv2ClassificationPrior


BRANCH_NAMES = ("official_shape", "official_profile", "copula")
ALLOWED_SPLITS = ("train", "val")
ISOLATED_MANIFEST = "hybrid178_compiled_manifest.json"
BLUEPRINT_FILENAME = "hybrid178_blueprint_v2.npz"
BLUEPRINT_V3_FILENAME = "hybrid178_blueprint_v3.npz"
HYBRID178_MAX_CLASSES = 10
HYBRID178_MAX_FEATURES = 100
HYBRID178_TARGET_PROFILE_NAMES = frozenset(
    {
        "artificial-characters",
        "kr-vs-k",
        "kropt",
        "volkert",
        "compass",
        "Credit_c",
        "eye_movements_bin",
        "in_vehicle_coupon_recommendation",
        "GesturePhaseSegmentationProcessed",
        "eye_movements",
        "electricity",
        "walking-activity",
        "jungle_chess_2pcs_raw_endgame_complete",
    }
)

# These scores are derived once from the row-free frozen blueprint V3, not
# from source rows at training time.  They combine the reconstructed task
# difficulty proxy (0.55), normalized class entropy (0.30), and normalized
# log class count (0.15).  The optional hardness-weighted scheduler uses them
# only for surplus protected slots; every one of the 178 profiles still gets
# its base conditioned coverage slot and every protected profile still gets
# one surplus slot before any weighted repetition.
HYBRID178_PROTECTED_HARDNESS_SCORES = {
    "internet_firewall": 0.846781,
    "ada": 0.798036,
    "ada_agnostic": 0.817703,
    "MIC": 0.768887,
    "thyroid-dis": 0.776296,
    "mfeat-zernike": 0.680951,
    "waveform_database_generator": 0.921033,
    "autoUniv-au4-2500": 0.868851,
    "pc1": 0.654361,
    "pc4": 0.716057,
    "phoneme": 0.816875,
    "UJI_Pen_Characters": 0.954864,
    "splice": 0.866723,
    "naticusdroid+android+permissions+dataset": 0.873955,
    "mfeat-pixel": 0.727506,
    "vehicle": 0.826884,
    "allbp": 0.664726,
    "baseball": 0.570709,
    "National_Health_and_Nutrition_Health_Survey": 0.779615,
    "Customer_Personality_Analysis": 0.737612,
}
HYBRID178_PROTECTED_HARDNESS_TEMPERATURE = 4.0

# Optional late-curriculum weights for protected surplus slots.  These values
# are computed offline from frozen, row-free profile statistics only:
#   0.40 z(log rows) + 0.20 z(zero ratio) + 0.15 z(binary-column ratio)
#   + 0.15 z(low-cardinality-column ratio)
#   - 0.10 z(normalized log cardinality), followed by exp(0.5 * score).
# They target the profile regime associated with the official run's late
# protected collapse without reading benchmark rows during training.  The
# scheduler still gives every protected profile one surplus slot before using
# these weights for repetitions.
HYBRID178_PROTECTED_COLLAPSE_RISK_WEIGHTS = {
    "internet_firewall": 1.375,
    "ada": 1.494,
    "ada_agnostic": 1.522,
    "MIC": 1.203,
    "thyroid-dis": 1.012,
    "mfeat-zernike": 0.619,
    "waveform_database_generator": 0.771,
    "autoUniv-au4-2500": 0.892,
    "pc1": 0.599,
    "pc4": 0.676,
    "phoneme": 0.752,
    "UJI_Pen_Characters": 0.586,
    "splice": 1.004,
    "naticusdroid+android+permissions+dataset": 2.393,
    "mfeat-pixel": 0.956,
    "vehicle": 0.568,
    "allbp": 1.282,
    "baseball": 0.611,
    "National_Health_and_Nutrition_Health_Survey": 0.771,
    "Customer_Personality_Analysis": 0.916,
}


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _assert_non_symlink_path(path: Path, boundary: Path) -> None:
    """Reject symlinks from ``boundary`` down to ``path`` (inclusive)."""

    unresolved = path.expanduser().absolute()
    boundary_unresolved = boundary.expanduser().absolute()
    if not _path_is_within(unresolved, boundary_unresolved):
        raise ValueError(f"path {unresolved} is outside expected root {boundary_unresolved}")
    relative = unresolved.relative_to(boundary_unresolved)
    current = boundary_unresolved
    if current.is_symlink():
        raise ValueError(f"isolated Hybrid-178 root may not be a symlink: {current}")
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"isolated Hybrid-178 artifacts may not use symlinks: {current}")


@dataclass(frozen=True)
class DatasetProfile:
    """Small immutable profile. Raw rows stay lazily loaded."""

    name: str
    directory: Path
    n_rows: int
    n_num_features: int
    n_cat_features: int
    n_classes: int
    class_probs: np.ndarray

    @property
    def n_features(self) -> int:
        return self.n_num_features + self.n_cat_features


@dataclass
class ProfileData:
    """Bounded-cache representation of the allowed rows for one profile."""

    x: np.ndarray
    y: np.ndarray
    kinds: np.ndarray
    class_probs: np.ndarray
    class_indices: Tuple[np.ndarray, ...]


@dataclass(frozen=True)
class FrozenDatasetBlueprint:
    """Row-free GT-derived parameters used by runtime-isolated generation."""

    name: str
    class_probs: np.ndarray
    source_class_count: int
    column_types: np.ndarray
    quantile_grid: np.ndarray
    class_quantiles: np.ndarray
    class_zero_rates: np.ndarray
    class_zero_cdf_lower: np.ndarray
    copula_modes: Tuple[str, ...]
    copula_factors: Tuple[np.ndarray, ...]
    copula_residuals: Tuple[np.ndarray, ...]
    target_quantile_levels: np.ndarray
    target_quantiles: np.ndarray
    target_scales: np.ndarray
    target_zero_rates: np.ndarray
    correlation_columns: np.ndarray
    target_rank_correlation: np.ndarray
    feature_view_columns: Tuple[np.ndarray, ...] = ()
    feature_view_modes: Tuple[Tuple[str, ...], ...] = ()
    feature_view_factors: Tuple[Tuple[np.ndarray, ...], ...] = ()
    feature_view_residuals: Tuple[Tuple[np.ndarray, ...], ...] = ()

    @property
    def n_features(self) -> int:
        return int(self.column_types.size)

    @property
    def n_classes(self) -> int:
        return int(self.class_probs.size)


def _stable_u32(text: str) -> int:
    digest = hashlib.blake2s(text.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little")


def _build_feature_view_columns(
    name: str,
    width: int,
    max_features: int,
    seed: int,
) -> Tuple[np.ndarray, ...]:
    """Build a deterministic cycle of bounded views covering every column.

    Every view has ``min(width, max_features)`` unique columns.  For a wide
    profile, consecutive windows traverse a fixed per-profile permutation;
    the final window wraps only to keep a constant task width.  Consequently
    the union of one complete cycle is exactly the full source schema.
    """

    width = int(width)
    cap = int(max_features)
    if width < 1 or cap < 1:
        raise ValueError("feature view width and cap must be positive")
    if cap > HYBRID178_MAX_FEATURES:
        raise ValueError(
            f"Hybrid-178 feature views are capped at {HYBRID178_MAX_FEATURES}"
        )
    if width <= cap:
        return (np.arange(width, dtype=np.int64),)
    view_width = cap
    view_count = int(np.ceil(width / view_width))
    view_rng = np.random.default_rng(
        (int(seed) ^ _stable_u32(name)) % (2**32)
    )
    permutation = view_rng.permutation(width).astype(np.int64, copy=False)
    offsets = np.arange(view_width, dtype=np.int64)
    return tuple(
        permutation[(view * view_width + offsets) % width]
        for view in range(view_count)
    )


def _slice_frozen_blueprint(
    blueprint: FrozenDatasetBlueprint,
    columns: np.ndarray,
    *,
    modes: Optional[Tuple[str, ...]] = None,
    factors: Optional[Tuple[np.ndarray, ...]] = None,
    residuals: Optional[Tuple[np.ndarray, ...]] = None,
) -> FrozenDatasetBlueprint:
    """Create a row-free feature view, retaining its exact factor copula."""

    columns = np.asarray(columns, dtype=np.int64)
    if columns.ndim != 1 or columns.size < 1:
        raise ValueError("feature view must contain at least one column")
    if np.unique(columns).size != columns.size:
        raise ValueError("feature view columns must be unique")
    if columns.min() < 0 or columns.max() >= blueprint.n_features:
        raise ValueError("feature view column lies outside the source schema")

    selected_modes = modes or blueprint.copula_modes
    selected_factors = factors or tuple(
        factor[columns] for factor in blueprint.copula_factors
    )
    selected_residuals = residuals or tuple(
        residual[columns] if residual.size else residual
        for residual in blueprint.copula_residuals
    )
    if not (
        len(selected_modes)
        == len(selected_factors)
        == len(selected_residuals)
        == blueprint.n_classes
    ):
        raise ValueError("feature view copula class count mismatch")

    local_lookup = {int(source): local for local, source in enumerate(columns)}
    corr_positions: List[int] = []
    local_corr_columns: List[int] = []
    for position, source in enumerate(blueprint.correlation_columns.tolist()):
        local = local_lookup.get(int(source))
        if local is not None:
            corr_positions.append(position)
            local_corr_columns.append(local)
    positions = np.asarray(corr_positions, dtype=np.int64)
    target_corr = (
        blueprint.target_rank_correlation[np.ix_(positions, positions)]
        if positions.size
        else np.empty((0, 0), dtype=np.float64)
    )
    return FrozenDatasetBlueprint(
        name=blueprint.name,
        class_probs=blueprint.class_probs,
        source_class_count=blueprint.source_class_count,
        column_types=blueprint.column_types[columns],
        quantile_grid=blueprint.quantile_grid,
        class_quantiles=blueprint.class_quantiles[:, columns, :],
        class_zero_rates=blueprint.class_zero_rates[:, columns],
        class_zero_cdf_lower=blueprint.class_zero_cdf_lower[:, columns],
        copula_modes=tuple(selected_modes),
        copula_factors=tuple(selected_factors),
        copula_residuals=tuple(selected_residuals),
        target_quantile_levels=blueprint.target_quantile_levels,
        target_quantiles=blueprint.target_quantiles[:, columns],
        target_scales=blueprint.target_scales[columns],
        target_zero_rates=blueprint.target_zero_rates[columns],
        correlation_columns=np.asarray(local_corr_columns, dtype=np.int64),
        target_rank_correlation=target_corr,
    )


def _frozen_feature_view(
    blueprint: FrozenDatasetBlueprint,
    max_features: int,
    view_index: int,
    seed: int,
) -> Tuple[FrozenDatasetBlueprint, np.ndarray, int, int, bool]:
    """Select one bounded view and report deterministic cycle metadata."""

    cap = min(int(max_features), HYBRID178_MAX_FEATURES)
    compiled_views = blueprint.feature_view_columns
    use_compiled = bool(
        compiled_views
        and compiled_views[0].size == min(blueprint.n_features, cap)
    )
    views = (
        compiled_views
        if use_compiled
        else _build_feature_view_columns(
            blueprint.name, blueprint.n_features, cap, seed
        )
    )
    resolved = int(view_index) % len(views)
    columns = np.asarray(views[resolved], dtype=np.int64)
    precompiled_copula = bool(
        use_compiled
        and len(blueprint.feature_view_modes) == len(views)
        and len(blueprint.feature_view_factors) == len(views)
        and len(blueprint.feature_view_residuals) == len(views)
    )
    if precompiled_copula:
        view = _slice_frozen_blueprint(
            blueprint,
            columns,
            modes=blueprint.feature_view_modes[resolved],
            factors=blueprint.feature_view_factors[resolved],
            residuals=blueprint.feature_view_residuals[resolved],
        )
    else:
        view = _slice_frozen_blueprint(blueprint, columns)
    return view, columns, resolved, len(views), precompiled_copula


def _encode_labels(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(values).reshape(-1)
    if flat.dtype.kind in {"O", "U", "S"}:
        flat = np.asarray([str(value) for value in flat], dtype=str)
    unique, inverse = np.unique(flat, return_inverse=True)
    return inverse.astype(np.int64, copy=False), unique


def _as_2d(values: np.ndarray, n_rows: int) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or arr.shape[0] != n_rows:
        raise ValueError(f"feature array has shape {arr.shape}, expected ({n_rows}, d)")
    return arr


def _encode_categorical_matrix(values: np.ndarray) -> np.ndarray:
    if values.shape[1] == 0:
        return np.empty((values.shape[0], 0), dtype=np.float32)
    encoded = np.empty(values.shape, dtype=np.float32)
    for col in range(values.shape[1]):
        raw = np.asarray(values[:, col])
        if raw.dtype.kind in {"f", "i", "u", "b"}:
            text = np.asarray(
                ["<missing>" if not np.isfinite(float(value)) else str(value) for value in raw]
            )
        else:
            text = np.asarray(["<missing>" if value is None else str(value) for value in raw])
        _, inverse = np.unique(text, return_inverse=True)
        encoded[:, col] = inverse.astype(np.float32, copy=False)
    return encoded


def _sanitize_numeric(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32).copy()
    for col in range(x.shape[1]):
        finite = np.isfinite(x[:, col])
        fill = float(np.median(x[finite, col])) if finite.any() else 0.0
        x[~finite, col] = fill
    return x


def _is_missing_scalar(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(np.asarray(np.isnat(value)).item())
    except (TypeError, ValueError):
        pass
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def _evaluation_compatible_split(
    num: np.ndarray,
    cat: np.ndarray,
    y_raw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror the data178 evaluation conversion for one source split.

    Numeric columns use split-local mean imputation. Numeric-looking
    categorical columns remain numeric. Rows with actual missing values in a
    string column are removed, and remaining string categories are factorized
    in sorted order. The evaluation script performs these operations separately
    for train and val before concatenating them.
    """

    n_rows = int(y_raw.shape[0])
    keep = np.ones(n_rows, dtype=bool)
    columns: List[Tuple[str, np.ndarray]] = []
    for values in [num[:, col] for col in range(num.shape[1])] + [
        cat[:, col] for col in range(cat.shape[1])
    ]:
        raw = np.asarray(values)
        try:
            numeric = raw.astype(np.float64)
            finite = np.isfinite(numeric)
            fill = float(np.mean(numeric[finite])) if finite.any() else 0.0
            numeric[~finite] = fill
            columns.append(("numeric", numeric))
        except (TypeError, ValueError):
            missing = np.asarray([_is_missing_scalar(value) for value in raw])
            keep &= ~missing
            columns.append(("categorical", raw))

    encoded: List[np.ndarray] = []
    kinds: List[str] = []
    for kind, values in columns:
        selected = values[keep]
        if kind == "numeric":
            encoded.append(selected.astype(np.float32, copy=False))
            kinds.append("numeric")
        else:
            text = np.asarray([str(value) for value in selected], dtype=str)
            _, inverse = np.unique(text, return_inverse=True)
            encoded.append(inverse.astype(np.float32, copy=False))
            kinds.append("categorical")
    x = (
        np.column_stack(encoded).astype(np.float32, copy=False)
        if encoded
        else np.empty((int(keep.sum()), 0), dtype=np.float32)
    )
    return x, np.asarray(y_raw).reshape(-1)[keep], np.asarray(kinds, dtype=object)


class ProfileBank:
    """Discover joint profiles and lazily cache explicitly allowed splits."""

    def __init__(
        self,
        root: str | Path,
        *,
        include_val: bool = False,
        max_cached_datasets: int = 2,
        max_rows_per_dataset: int = 200_000,
        seed: int = 0,
        evaluation_compatible: bool = False,
        forbidden_root: str | Path | None = None,
        reject_symlinks: bool = False,
    ):
        unresolved_root = Path(root).expanduser().absolute()
        if reject_symlinks and unresolved_root.is_symlink():
            raise ValueError(
                f"isolated Hybrid-178 root may not be a symlink: {unresolved_root}"
            )
        self.root = unresolved_root.resolve()
        self.forbidden_root = (
            Path(forbidden_root).expanduser().resolve()
            if forbidden_root is not None
            else None
        )
        self.reject_symlinks = bool(reject_symlinks)
        if self.forbidden_root is not None and (
            _path_is_within(self.root, self.forbidden_root)
            or _path_is_within(self.forbidden_root, self.root)
        ):
            raise ValueError(
                "hybrid178_data_root and forbidden GT root must be disjoint: "
                f"data={self.root}, forbidden={self.forbidden_root}"
            )
        self.splits = ("train", "val") if include_val else ("train",)
        if any(split not in ALLOWED_SPLITS for split in self.splits):
            raise ValueError(f"only these profile splits are permitted: {ALLOWED_SPLITS}")
        self.max_cached_datasets = max(1, int(max_cached_datasets))
        self.max_rows_per_dataset = max(128, int(max_rows_per_dataset))
        self.seed = int(seed)
        self.evaluation_compatible = bool(evaluation_compatible)
        self.profiles = self._discover()
        self._cache: OrderedDict[str, ProfileData] = OrderedDict()

    def _checked_path(self, path: Path) -> Path:
        if self.reject_symlinks:
            _assert_non_symlink_path(path, self.root)
        resolved = path.expanduser().resolve()
        if self.forbidden_root is not None and _path_is_within(
            resolved, self.forbidden_root
        ):
            raise PermissionError(
                f"isolated Hybrid-178 blocked access to GT path: {resolved}"
            )
        if not _path_is_within(resolved, self.root):
            raise PermissionError(
                f"Hybrid-178 profile path escaped its data root: {resolved}"
            )
        return resolved

    def _load_array(self, path: Path) -> np.ndarray:
        return np.load(
            self._checked_path(path), allow_pickle=True, mmap_mode=None
        )

    def _feature_width(self, directory: Path, filename: str) -> int:
        path = directory / filename
        if not path.exists():
            return 0
        arr = self._load_array(path)
        return int(arr.shape[1]) if arr.ndim > 1 else 1

    def _discover(self) -> Tuple[DatasetProfile, ...]:
        if not self.root.is_dir():
            raise FileNotFoundError(f"hybrid178_data_root is not a directory: {self.root}")
        profiles: List[DatasetProfile] = []
        for directory in sorted(path for path in self.root.iterdir() if path.is_dir()):
            y_path = directory / "y_train.npy"
            if not y_path.exists():
                continue
            n_num = self._feature_width(directory, "N_train.npy")
            n_cat = self._feature_width(directory, "C_train.npy")
            if n_num + n_cat == 0:
                continue
            y, _ = _encode_labels(self._load_array(y_path))
            if y.size < 4:
                continue
            counts = np.bincount(y).astype(np.float64)
            counts = counts[counts > 0]
            if counts.size < 2:
                continue
            profiles.append(
                DatasetProfile(
                    name=directory.name,
                    directory=directory,
                    n_rows=int(y.size),
                    n_num_features=n_num,
                    n_cat_features=n_cat,
                    n_classes=int(counts.size),
                    class_probs=counts / counts.sum(),
                )
            )
        if not profiles:
            raise FileNotFoundError(
                "no classification datasets with y_train.npy and N_train.npy/C_train.npy "
                f"found under {self.root}"
            )
        return tuple(profiles)

    def _load_split(
        self, profile: DatasetProfile, split: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if split not in self.splits:
            raise ValueError(f"split {split!r} was not enabled for this profile bank")
        y_path = profile.directory / f"y_{split}.npy"
        if not y_path.exists():
            return (
                np.empty((0, profile.n_num_features), dtype=np.float32),
                np.empty((0, profile.n_cat_features), dtype=object),
                np.empty(0, dtype=np.int64),
            )
        y_raw = self._load_array(y_path).reshape(-1)
        n_rows = int(y_raw.shape[0])
        num_path = profile.directory / f"N_{split}.npy"
        cat_path = profile.directory / f"C_{split}.npy"
        num = (
            _as_2d(self._load_array(num_path), n_rows)
            if num_path.exists()
            else np.empty((n_rows, 0), dtype=np.float32)
        )
        cat = (
            _as_2d(self._load_array(cat_path), n_rows)
            if cat_path.exists()
            else np.empty((n_rows, 0), dtype=object)
        )
        return num, cat, y_raw

    def load(self, profile: DatasetProfile) -> ProfileData:
        cached = self._cache.pop(profile.name, None)
        if cached is not None:
            self._cache[profile.name] = cached
            return cached

        chunks = [self._load_split(profile, split) for split in self.splits]
        chunks = [chunk for chunk in chunks if chunk[2].size > 0]
        if not chunks:
            raise ValueError(f"profile {profile.name!r} has no rows in splits {self.splits}")
        if self.evaluation_compatible:
            processed = [_evaluation_compatible_split(*chunk) for chunk in chunks]
            x = np.concatenate([chunk[0] for chunk in processed], axis=0)
            y_raw = np.concatenate([chunk[1] for chunk in processed], axis=0)
            kinds = processed[0][2]
        else:
            num = np.concatenate([chunk[0] for chunk in chunks], axis=0)
            cat = np.concatenate([chunk[1] for chunk in chunks], axis=0)
            y_raw = np.concatenate([chunk[2] for chunk in chunks], axis=0)
            num = _sanitize_numeric(num)
            cat_encoded = _encode_categorical_matrix(cat)
            x = np.concatenate([num, cat_encoded], axis=1).astype(
                np.float32, copy=False
            )
            kinds = np.asarray(
                ["numeric"] * num.shape[1] + ["categorical"] * cat.shape[1],
                dtype=object,
            )
        y, _ = _encode_labels(y_raw)

        if x.shape[0] > self.max_rows_per_dataset:
            rng = np.random.default_rng(self.seed ^ _stable_u32(profile.name))
            selected: List[np.ndarray] = []
            classes = np.unique(y)
            budget = self.max_rows_per_dataset
            for cls in classes:
                cls_idx = np.flatnonzero(y == cls)
                take = min(cls_idx.size, max(2, budget // max(len(classes), 1)))
                selected.append(rng.choice(cls_idx, size=take, replace=False))
            idx = np.unique(np.concatenate(selected))
            if idx.size < budget:
                remaining = np.setdiff1d(np.arange(y.size), idx, assume_unique=False)
                extra = rng.choice(
                    remaining, size=min(budget - idx.size, remaining.size), replace=False
                )
                idx = np.concatenate([idx, extra])
            rng.shuffle(idx)
            x, y = x[idx], y[idx]

        counts = np.bincount(y).astype(np.float64)
        class_indices = tuple(
            np.flatnonzero(y == cls).astype(np.int64) for cls in range(counts.size)
        )
        data = ProfileData(
            x=x,
            y=y,
            kinds=kinds,
            class_probs=counts / max(counts.sum(), 1.0),
            class_indices=class_indices,
        )
        self._cache[profile.name] = data
        while len(self._cache) > self.max_cached_datasets:
            self._cache.popitem(last=False)
        return data

    def __len__(self) -> int:
        return len(self.profiles)


class FrozenBlueprintBank:
    """Load row-free per-profile statistical blueprints under path isolation."""

    def __init__(
        self,
        root: str | Path,
        manifest: Dict[str, object],
        *,
        forbidden_root: str | Path,
        max_cached_datasets: int = 2,
    ):
        unresolved_root = Path(root).expanduser().absolute()
        _assert_non_symlink_path(unresolved_root, unresolved_root)
        self.root = unresolved_root.resolve()
        self.forbidden_root = Path(forbidden_root).expanduser().resolve()
        if _path_is_within(self.root, self.forbidden_root) or _path_is_within(
            self.forbidden_root, self.root
        ):
            raise ValueError("blueprint and forbidden GT roots must be disjoint")
        self.splits = ("compiled_blueprint",)
        self.max_cached_datasets = max(1, int(max_cached_datasets))
        self.blueprint_filename = str(
            manifest.get("blueprint_filename", BLUEPRINT_FILENAME)
        )
        if (
            Path(self.blueprint_filename).name != self.blueprint_filename
            or self.blueprint_filename not in {BLUEPRINT_FILENAME, BLUEPRINT_V3_FILENAME}
        ):
            raise ValueError(f"invalid blueprint filename: {self.blueprint_filename!r}")
        records = manifest.get("datasets")
        if not isinstance(records, list) or not records:
            raise ValueError("blueprint manifest has no dataset records")
        profiles: List[DatasetProfile] = []
        self._records: Dict[str, Dict[str, object]] = {}
        for raw_record in records:
            if not isinstance(raw_record, dict):
                raise ValueError("invalid blueprint dataset record")
            record = dict(raw_record)
            name = str(record["name"])
            probs = np.asarray(record["class_probs"], dtype=np.float64)
            if probs.ndim != 1 or probs.size < 2 or not np.isfinite(probs).all():
                raise ValueError(f"invalid blueprint class probabilities for {name}")
            probs = probs / probs.sum()
            n_num = int(record["num_features"])
            n_cat = int(record["cat_features"])
            profiles.append(
                DatasetProfile(
                    name=name,
                    directory=self.root / name,
                    n_rows=int(record["source_rows"]),
                    n_num_features=n_num,
                    n_cat_features=n_cat,
                    n_classes=int(probs.size),
                    class_probs=probs,
                )
            )
            self._records[name] = record
        self.profiles = tuple(profiles)
        self._cache: OrderedDict[str, FrozenDatasetBlueprint] = OrderedDict()

    def _checked_path(self, path: Path) -> Path:
        _assert_non_symlink_path(path, self.root)
        resolved = path.expanduser().resolve()
        if not _path_is_within(resolved, self.root):
            raise PermissionError(f"blueprint path escaped artifact root: {resolved}")
        if _path_is_within(resolved, self.forbidden_root):
            raise PermissionError(f"blocked GT access from blueprint bank: {resolved}")
        return resolved

    def load(self, profile: DatasetProfile) -> FrozenDatasetBlueprint:
        cached = self._cache.pop(profile.name, None)
        if cached is not None:
            self._cache[profile.name] = cached
            return cached
        path = self._checked_path(profile.directory / self.blueprint_filename)
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen blueprint: {path}")
        with np.load(path, allow_pickle=False) as arrays:
            class_probs = arrays["class_probs"].astype(np.float64)
            column_types = arrays["column_types"].astype(str)
            quantile_grid = arrays["quantile_grid"].astype(np.float64)
            class_quantiles = arrays["class_quantiles"].astype(np.float32)
            class_zero_rates = arrays["class_zero_rates"].astype(np.float64)
            class_zero_cdf_lower = arrays["class_zero_cdf_lower"].astype(
                np.float64
            )
            modes = tuple(arrays["copula_modes"].astype(str).tolist())
            factors = tuple(
                arrays[f"copula_factor_{index}"].astype(np.float64)
                for index in range(class_probs.size)
            )
            residuals = tuple(
                arrays[f"copula_residual_{index}"].astype(np.float64)
                for index in range(class_probs.size)
            )
            if "feature_view_columns" in arrays.files:
                view_columns_array = arrays["feature_view_columns"].astype(np.int64)
                view_modes_array = arrays["feature_view_modes"].astype(str)
                if view_columns_array.ndim != 2 or view_modes_array.shape != (
                    view_columns_array.shape[0],
                    class_probs.size,
                ):
                    raise ValueError(f"invalid compiled feature views for {profile.name}")
                view_columns = tuple(view_columns_array[index] for index in range(view_columns_array.shape[0]))
                view_modes = tuple(
                    tuple(view_modes_array[index].tolist())
                    for index in range(view_modes_array.shape[0])
                )
                view_factors = tuple(
                    tuple(
                        arrays[f"feature_view_factor_{view}_{local_class}"].astype(
                            np.float64
                        )
                        for local_class in range(class_probs.size)
                    )
                    for view in range(view_columns_array.shape[0])
                )
                view_residuals = tuple(
                    tuple(
                        arrays[
                            f"feature_view_residual_{view}_{local_class}"
                        ].astype(np.float64)
                        for local_class in range(class_probs.size)
                    )
                    for view in range(view_columns_array.shape[0])
                )
            else:
                view_columns = ()
                view_modes = ()
                view_factors = ()
                view_residuals = ()
            blueprint = FrozenDatasetBlueprint(
                name=profile.name,
                class_probs=class_probs / class_probs.sum(),
                source_class_count=int(arrays["source_class_count"].item()),
                column_types=column_types,
                quantile_grid=quantile_grid,
                class_quantiles=class_quantiles,
                class_zero_rates=class_zero_rates,
                class_zero_cdf_lower=class_zero_cdf_lower,
                copula_modes=modes,
                copula_factors=factors,
                copula_residuals=residuals,
                target_quantile_levels=arrays[
                    "target_quantile_levels"
                ].astype(np.float64),
                target_quantiles=arrays["target_quantiles"].astype(np.float64),
                target_scales=arrays["target_scales"].astype(np.float64),
                target_zero_rates=arrays["target_zero_rates"].astype(np.float64),
                correlation_columns=arrays["correlation_columns"].astype(np.int64),
                target_rank_correlation=arrays[
                    "target_rank_correlation"
                ].astype(np.float64),
                feature_view_columns=view_columns,
                feature_view_modes=view_modes,
                feature_view_factors=view_factors,
                feature_view_residuals=view_residuals,
            )
        expected_shape = (
            blueprint.n_classes,
            blueprint.n_features,
            blueprint.quantile_grid.size,
        )
        if blueprint.class_quantiles.shape != expected_shape:
            raise ValueError(
                f"invalid class quantile shape for {profile.name}: "
                f"{blueprint.class_quantiles.shape}, expected {expected_shape}"
            )
        if blueprint.n_features != profile.n_features:
            raise ValueError(f"blueprint schema mismatch for {profile.name}")
        if blueprint.n_classes > HYBRID178_MAX_CLASSES:
            raise ValueError(
                f"blueprint {profile.name} has {blueprint.n_classes} classes; "
                f"maximum is {HYBRID178_MAX_CLASSES}"
            )
        self._cache[profile.name] = blueprint
        while len(self._cache) > self.max_cached_datasets:
            self._cache.popitem(last=False)
        return blueprint

    def __len__(self) -> int:
        return len(self.profiles)


class RollingProfileScheduler:
    """Shuffled queue that covers every profile before any repetition."""

    def __init__(self, n_profiles: int, rng: np.random.Generator):
        if n_profiles <= 0:
            raise ValueError("n_profiles must be positive")
        self.n_profiles = int(n_profiles)
        self.rng = rng
        self._queue = np.empty(0, dtype=np.int64)
        self.cycle = 0

    def take(self, count: int) -> np.ndarray:
        out: List[np.ndarray] = []
        needed = int(count)
        while needed > 0:
            if self._queue.size == 0:
                self._queue = self.rng.permutation(self.n_profiles).astype(np.int64)
                self.cycle += 1
            take = min(needed, self._queue.size)
            out.append(self._queue[:take])
            self._queue = self._queue[take:]
            needed -= take
        return np.concatenate(out) if out else np.empty(0, dtype=np.int64)


def coordinated_global_schedule(
    *,
    n_profiles: int,
    branch_probs: np.ndarray,
    base_seed: int,
    global_batch_index: int,
    world_size: int,
    rank: int,
    local_batch_size: int,
    priority_profile_ids: Optional[Sequence[int]] = None,
    priority_profile_weights: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Return one rank's slice of a deterministic DDP-global schedule.

    Prior generation happens after the configured batch has been divided over
    DDP ranks.  Scheduling independently inside each rank would therefore lose
    the global guarantees: a 1% branch rounds to zero for a 16-task local batch,
    and 64 independent local profile queues need not cover all 178 templates.
    This helper constructs the complete effective global batch first and then
    returns a disjoint rank-local slice.
    """

    n_profiles = int(n_profiles)
    world_size = int(world_size)
    rank = int(rank)
    local_batch_size = int(local_batch_size)
    global_batch_index = int(global_batch_index)
    if n_profiles <= 0:
        raise ValueError("n_profiles must be positive")
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    if local_batch_size <= 0 or global_batch_index < 0:
        raise ValueError("batch sizes must be positive and batch index non-negative")

    raw_priority_ids = np.asarray(
        () if priority_profile_ids is None else tuple(priority_profile_ids),
        dtype=np.int64,
    )
    if raw_priority_ids.ndim != 1:
        raise ValueError("priority_profile_ids must be one-dimensional")
    priority_weights: Optional[np.ndarray] = None
    if raw_priority_ids.size:
        if np.unique(raw_priority_ids).size != raw_priority_ids.size:
            raise ValueError("priority profile IDs must be unique")
        order = np.argsort(raw_priority_ids, kind="stable")
        priority_ids = raw_priority_ids[order]
        if (priority_ids < 0).any() or (priority_ids >= n_profiles).any():
            raise ValueError("priority profile IDs must be in [0, n_profiles)")
        if priority_profile_weights is not None:
            raw_weights = np.asarray(tuple(priority_profile_weights), dtype=np.float64)
            if raw_weights.ndim != 1 or raw_weights.size != raw_priority_ids.size:
                raise ValueError(
                    "priority_profile_weights must align one-to-one with "
                    "priority_profile_ids"
                )
            if not np.isfinite(raw_weights).all() or (raw_weights <= 0).any():
                raise ValueError("priority profile weights must be finite and positive")
            priority_weights = raw_weights[order]
    else:
        priority_ids = raw_priority_ids
        if priority_profile_weights is not None and tuple(priority_profile_weights):
            raise ValueError("priority weights require priority profile IDs")

    global_batch_size = world_size * local_batch_size
    seed = np.random.SeedSequence(
        [int(base_seed) % (2**32), global_batch_index % (2**32), 178]
    )
    rng = np.random.default_rng(seed)

    profile_cycles: List[np.ndarray] = []
    remaining = global_batch_size
    while remaining > 0:
        cycle = rng.permutation(n_profiles).astype(np.int64)
        take = min(remaining, n_profiles)
        profile_cycles.append(cycle[:take])
        remaining -= take
    global_profile_ids = np.concatenate(profile_cycles)

    expected = np.asarray(branch_probs, dtype=np.float64) * global_batch_size
    counts = np.floor(expected).astype(np.int64)
    remainder = global_batch_size - int(counts.sum())
    if remainder:
        order = np.argsort(-(expected - counts), kind="stable")
        counts[order[:remainder]] += 1
    # Keep official GraphSCM as the default task distribution.  If the global
    # batch contains at least one conditioned slot per profile, explicitly put
    # one official-profile/copula task on every profile.  The remaining slots
    # stay pure official-shape tasks.  This turns the 178 bank into a coverage
    # constraint instead of making all generated tasks template-conditioned.
    global_branches = np.full(
        global_batch_size, "official_shape", dtype=object
    )
    conditioned = [
        name
        for name, count in zip(BRANCH_NAMES[1:], counts[1:])
        for _ in range(int(count))
    ]
    rng.shuffle(conditioned)
    if len(conditioned) >= n_profiles:
        coverage_slots = np.asarray(
            [
                int(rng.choice(np.flatnonzero(global_profile_ids == profile_id)))
                for profile_id in range(n_profiles)
            ],
            dtype=np.int64,
        )
        global_branches[coverage_slots] = conditioned[:n_profiles]
        remaining_names = conditioned[n_profiles:]
        if remaining_names:
            available = np.setdiff1d(
                np.arange(global_batch_size, dtype=np.int64),
                coverage_slots,
                assume_unique=False,
            )
            extra_slots: List[int] = []
            # Every profile already owns one conditioned coverage slot.  When
            # requested, spend only the surplus conditioned capacity on the
            # priority group.  A shuffled round-robin gives every priority
            # profile one extra exposure before any receives a second, while
            # retaining deterministic DDP-global scheduling.
            if priority_ids.size:
                shuffled_priority = rng.permutation(priority_ids)
                available_by_profile: Dict[int, List[int]] = {}
                for profile_id in shuffled_priority.tolist():
                    slots = available[global_profile_ids[available] == profile_id]
                    slots = rng.permutation(slots).astype(np.int64)
                    available_by_profile[int(profile_id)] = slots.tolist()
                weight_by_profile = (
                    {
                        int(profile_id): float(weight)
                        for profile_id, weight in zip(priority_ids, priority_weights)
                    }
                    if priority_weights is not None
                    else None
                )
                cycle_index = 0
                while len(extra_slots) < len(remaining_names):
                    active = np.asarray(
                        [
                            int(profile_id)
                            for profile_id in shuffled_priority.tolist()
                            if available_by_profile[int(profile_id)]
                        ],
                        dtype=np.int64,
                    )
                    if active.size == 0:
                        break
                    # Cycle zero preserves the one-extra-per-protected-profile
                    # guarantee.  Only subsequent repetitions are weighted.
                    if weight_by_profile is not None and cycle_index > 0:
                        active_weights = np.asarray(
                            [weight_by_profile[int(profile_id)] for profile_id in active],
                            dtype=np.float64,
                        )
                        active_weights /= active_weights.sum()
                        cycle_priority = rng.choice(
                            active,
                            size=active.size,
                            replace=False,
                            p=active_weights,
                        )
                    else:
                        cycle_priority = active
                    made_progress = False
                    for profile_id in cycle_priority.tolist():
                        profile_slots = available_by_profile[int(profile_id)]
                        if profile_slots:
                            extra_slots.append(int(profile_slots.pop()))
                            made_progress = True
                            if len(extra_slots) == len(remaining_names):
                                break
                    if not made_progress:
                        break
                    cycle_index += 1
            if len(extra_slots) < len(remaining_names):
                unused = np.setdiff1d(
                    available,
                    np.asarray(extra_slots, dtype=np.int64),
                    assume_unique=False,
                )
                fill = rng.choice(
                    unused,
                    size=len(remaining_names) - len(extra_slots),
                    replace=False,
                )
                extra_slots.extend(int(slot) for slot in fill.tolist())
            extra_slots_array = np.asarray(extra_slots, dtype=np.int64)
            global_branches[extra_slots_array] = remaining_names
    elif conditioned:
        slots = rng.choice(
            global_batch_size, size=len(conditioned), replace=False
        )
        global_branches[slots] = conditioned

    start = rank * local_batch_size
    stop = start + local_batch_size
    return global_profile_ids[start:stop], global_branches[start:stop].tolist()


def _split_counts(n_rows: int, probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    probs = probs / max(probs.sum(), 1e-12)
    if n_rows < probs.size:
        raise ValueError("split has fewer rows than classes")
    counts = np.ones(probs.size, dtype=np.int64)
    counts += rng.multinomial(n_rows - probs.size, probs)
    return counts


def _resize_class_probabilities(probs: np.ndarray, n_classes: int) -> np.ndarray:
    """Resize an ordered class-probability profile without flattening imbalance."""

    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    probs = probs / max(float(probs.sum()), 1e-12)
    n_classes = int(n_classes)
    if n_classes < 2:
        raise ValueError("n_classes must be at least two")
    if probs.size == n_classes:
        return probs.copy()
    source_edges = np.linspace(0.0, 1.0, probs.size + 1)
    target_edges = np.linspace(0.0, 1.0, n_classes + 1)
    target_cdf = np.interp(target_edges, source_edges, np.r_[0.0, np.cumsum(probs)])
    resized = np.maximum(np.diff(target_cdf), 1e-12)
    return resized / resized.sum()


def _closest_proportional_counts(n_rows: int, probs: np.ndarray) -> np.ndarray:
    """Deterministic integer counts closest to empirical class proportions."""

    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    probs = probs / max(probs.sum(), 1e-12)
    if n_rows < probs.size:
        raise ValueError("split has fewer rows than classes")
    target = probs * n_rows
    counts = np.floor(target).astype(np.int64)
    counts[counts < 1] = 1
    while int(counts.sum()) > n_rows:
        candidates = np.flatnonzero(counts > 1)
        remove = candidates[np.argmax(counts[candidates] - target[candidates])]
        counts[remove] -= 1
    if int(counts.sum()) < n_rows:
        order = np.argsort(-(target - counts), kind="stable")
        for idx in order[: n_rows - int(counts.sum())]:
            counts[idx] += 1
    return counts


def _rank_stratified_uniforms(values: np.ndarray) -> np.ndarray:
    """Convert every column to its exact midpoint rank grid.

    This retains the sampled copula's column order while removing avoidable
    Monte-Carlo noise from finite-sample marginals.
    """

    n_rows, width = values.shape
    uniforms = np.empty((n_rows, width), dtype=np.float64)
    midpoint_grid = (np.arange(n_rows, dtype=np.float64) + 0.5) / n_rows
    for col in range(width):
        order = np.argsort(values[:, col], kind="mergesort")
        uniforms[order, col] = midpoint_grid
    return uniforms


def _task_quality_metrics(
    generated_x: np.ndarray,
    generated_y: np.ndarray,
    template: ProfileData,
    class_probs: np.ndarray,
) -> Dict[str, float]:
    """Cheap raw-space gate against the frozen surrogate, never against GT."""

    width = generated_x.shape[1]
    template_x = template.x[:, :width].astype(np.float64, copy=False)
    generated_x = generated_x.astype(np.float64, copy=False)
    columns = np.arange(width)
    if columns.size > 32:
        columns = columns[
            np.linspace(0, columns.size - 1, num=32, dtype=np.int64)
        ]
    reference = template_x[:, columns]
    candidate = generated_x[:, columns]
    quantiles = np.asarray([0.10, 0.25, 0.50, 0.75, 0.90])
    ref_q = np.quantile(reference, quantiles, axis=0)
    candidate_q = np.quantile(candidate, quantiles, axis=0)
    q25, q75 = np.quantile(reference, [0.25, 0.75], axis=0)
    scale = np.maximum((q75 - q25) / 1.349, np.std(reference, axis=0))
    informative = scale > 1e-6
    marginal_error = (
        float(
            np.mean(
                np.abs(candidate_q[:, informative] - ref_q[:, informative])
                / scale[informative]
            )
        )
        if informative.any()
        else 0.0
    )
    zero_rate_error = float(
        np.mean(
            np.abs(
                np.isclose(candidate, 0.0, atol=1e-8).mean(axis=0)
                - np.isclose(reference, 0.0, atol=1e-8).mean(axis=0)
            )
        )
    )
    generated_probs = np.bincount(
        generated_y.astype(np.int64), minlength=class_probs.size
    ).astype(np.float64)
    generated_probs /= max(generated_probs.sum(), 1.0)
    class_tv = float(0.5 * np.abs(generated_probs - class_probs).sum())
    variable = np.flatnonzero(
        (np.std(reference, axis=0) > 1e-6)
        & (np.std(candidate, axis=0) > 1e-6)
    )
    if variable.size > 16:
        variable = variable[
            np.linspace(0, variable.size - 1, num=16, dtype=np.int64)
        ]
    if variable.size >= 2:
        reference_scores = _normal_scores(reference[:, variable])
        candidate_scores = _normal_scores(candidate[:, variable])
        reference_corr = np.nan_to_num(
            np.corrcoef(reference_scores, rowvar=False)
        )
        candidate_corr = np.nan_to_num(
            np.corrcoef(candidate_scores, rowvar=False)
        )
        upper = np.triu_indices(variable.size, k=1)
        correlation_error = float(
            np.mean(np.abs(reference_corr[upper] - candidate_corr[upper]))
        )
    else:
        correlation_error = 0.0
    return {
        "quality_class_tv": class_tv,
        "quality_marginal_quantile_scaled_mae": marginal_error,
        "quality_zero_rate_mae": zero_rate_error,
        "quality_rank_correlation_mae": correlation_error,
    }


def _blueprint_quality_metrics(
    generated_x: np.ndarray,
    generated_y: np.ndarray,
    blueprint: FrozenDatasetBlueprint,
) -> Dict[str, float]:
    """Compare a candidate directly with frozen GT statistics, not proxy rows."""

    generated_x = generated_x.astype(np.float64, copy=False)
    width = generated_x.shape[1]
    columns = np.arange(width)
    if columns.size > 32:
        columns = columns[
            np.linspace(0, columns.size - 1, num=32, dtype=np.int64)
        ]
    candidate = generated_x[:, columns]
    candidate_q = np.quantile(
        candidate, blueprint.target_quantile_levels, axis=0
    )
    target_q = blueprint.target_quantiles[:, columns]
    scales = blueprint.target_scales[columns]
    informative = scales > 1e-6
    marginal_error = (
        float(
            np.mean(
                np.abs(candidate_q[:, informative] - target_q[:, informative])
                / scales[informative]
            )
        )
        if informative.any()
        else 0.0
    )
    zero_rate_error = float(
        np.mean(
            np.abs(
                np.isclose(candidate, 0.0, atol=1e-8).mean(axis=0)
                - blueprint.target_zero_rates[columns]
            )
        )
    )
    generated_probs = np.bincount(
        generated_y.astype(np.int64), minlength=blueprint.n_classes
    ).astype(np.float64)
    generated_probs /= max(generated_probs.sum(), 1.0)
    class_tv = float(
        0.5 * np.abs(generated_probs - blueprint.class_probs).sum()
    )

    corr_columns = blueprint.correlation_columns
    corr_columns = corr_columns[corr_columns < width]
    variable = np.asarray(
        [
            col
            for col in corr_columns
            if np.std(generated_x[:, col]) > 1e-6
        ],
        dtype=np.int64,
    )
    if variable.size >= 2:
        correlation_lookup = {
            int(column): position
            for position, column in enumerate(blueprint.correlation_columns)
        }
        positions = np.asarray(
            [correlation_lookup[int(column)] for column in variable],
            dtype=np.int64,
        )
        candidate_scores = _normal_scores(generated_x[:, variable])
        candidate_corr = np.nan_to_num(
            np.corrcoef(candidate_scores, rowvar=False)
        )
        target_corr = blueprint.target_rank_correlation[
            np.ix_(positions, positions)
        ]
        upper = np.triu_indices(variable.size, k=1)
        correlation_error = float(
            np.mean(np.abs(candidate_corr[upper] - target_corr[upper]))
        )
    else:
        correlation_error = 0.0

    covered_bins = 0
    total_bins = 0
    for local_col, source_col in enumerate(columns):
        if blueprint.column_types[source_col] != "continuous":
            continue
        edges = np.unique(target_q[:, local_col])
        if edges.size < 3:
            continue
        counts = np.bincount(
            np.searchsorted(edges, candidate[:, local_col], side="right"),
            minlength=edges.size + 1,
        )
        covered_bins += int(np.count_nonzero(counts))
        total_bins += int(counts.size)
    quantile_bin_coverage = (
        float(covered_bins / total_bins) if total_bins else 1.0
    )
    return {
        "quality_class_tv": class_tv,
        "quality_marginal_quantile_scaled_mae": marginal_error,
        "quality_zero_rate_mae": zero_rate_error,
        "quality_rank_correlation_mae": correlation_error,
        "quality_quantile_bin_coverage": quantile_bin_coverage,
    }


def _quality_columns(width: int, limit: int = 24) -> np.ndarray:
    """Deterministically bound supervised quality work for wide tasks."""

    columns = np.arange(int(width), dtype=np.int64)
    if columns.size > limit:
        columns = columns[
            np.linspace(0, columns.size - 1, num=limit, dtype=np.int64)
        ]
    return columns


def _feature_label_nmi(
    values: np.ndarray,
    labels: np.ndarray,
    bin_edges: Optional[Tuple[np.ndarray, ...]] = None,
) -> np.ndarray:
    """Histogram mutual information per feature, normalized by label entropy."""

    values = np.asarray(values, dtype=np.float64)
    _, encoded = np.unique(labels.astype(np.int64), return_inverse=True)
    n_classes = int(encoded.max()) + 1
    class_counts = np.bincount(encoded, minlength=n_classes).astype(np.float64)
    class_probs = class_counts / max(class_counts.sum(), 1.0)
    positive = class_probs > 0
    label_entropy = float(
        -np.sum(class_probs[positive] * np.log(class_probs[positive]))
    )
    if label_entropy <= 1e-12:
        return np.zeros(values.shape[1], dtype=np.float64)

    output = np.zeros(values.shape[1], dtype=np.float64)
    default_levels = np.linspace(0.0, 1.0, 9)[1:-1]
    for col in range(values.shape[1]):
        if bin_edges is None:
            edges = np.unique(np.quantile(values[:, col], default_levels))
        else:
            edges = np.unique(np.asarray(bin_edges[col], dtype=np.float64))
        edges = edges[np.isfinite(edges)]
        bins = np.searchsorted(edges, values[:, col], side="right")
        n_bins = int(edges.size + 1)
        joint = np.bincount(
            encoded * n_bins + bins,
            minlength=n_classes * n_bins,
        ).reshape(n_classes, n_bins).astype(np.float64)
        joint /= max(joint.sum(), 1.0)
        p_class = joint.sum(axis=1, keepdims=True)
        p_bin = joint.sum(axis=0, keepdims=True)
        expected = p_class * p_bin
        mask = (joint > 0) & (expected > 0)
        mi = float(np.sum(joint[mask] * np.log(joint[mask] / expected[mask])))
        output[col] = float(np.clip(mi / label_entropy, 0.0, 1.0))
    return output


def _higher_order_correlation_signature(correlation: np.ndarray) -> np.ndarray:
    """Triangle closure and spectral concentration of a dependence matrix."""

    correlation = np.nan_to_num(np.asarray(correlation, dtype=np.float64))
    if correlation.ndim != 2 or correlation.shape[0] < 2:
        return np.zeros(2, dtype=np.float64)
    correlation = 0.5 * (correlation + correlation.T)
    np.fill_diagonal(correlation, 1.0)
    width = correlation.shape[0]
    off_diagonal = correlation - np.eye(width)
    triangle_denominator = max(width * (width - 1) * (width - 2), 1)
    triangle_closure = float(
        np.trace(off_diagonal @ off_diagonal @ off_diagonal)
        / triangle_denominator
    )
    eigenvalues = np.linalg.eigvalsh(correlation)
    spectral_concentration = float(
        np.sum(np.square(eigenvalues)) / max(width * width, 1)
    )
    return np.asarray(
        [triangle_closure, spectral_concentration], dtype=np.float64
    )


def _task_structure_metrics(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    columns: Optional[np.ndarray] = None,
    bin_edges: Optional[Tuple[np.ndarray, ...]] = None,
) -> Dict[str, object]:
    """Supervised structure and intrinsic diversity for one generated task."""

    values = np.asarray(values, dtype=np.float64)
    if columns is None:
        columns = _quality_columns(values.shape[1])
    selected = values[:, columns]
    labels = labels.astype(np.int64, copy=False)
    classes, encoded = np.unique(labels, return_inverse=True)
    counts = np.bincount(encoded, minlength=classes.size).astype(np.float64)
    probs = counts / max(counts.sum(), 1.0)
    positive = probs > 0
    entropy = float(-np.sum(probs[positive] * np.log(probs[positive])))
    normalized_entropy = (
        float(entropy / np.log(classes.size)) if classes.size > 1 else 0.0
    )

    nmi = _feature_label_nmi(selected, labels, bin_edges)
    global_mean = selected.mean(axis=0)
    total_variance = selected.var(axis=0)
    between = np.zeros(selected.shape[1], dtype=np.float64)
    for local_class in range(classes.size):
        rows = encoded == local_class
        if rows.any():
            between += probs[local_class] * np.square(
                selected[rows].mean(axis=0) - global_mean
            )
    eta_squared = np.divide(
        between,
        total_variance,
        out=np.zeros_like(between),
        where=total_variance > 1e-12,
    )
    strongest = np.sort(np.clip(eta_squared, 0.0, 1.0))[
        -min(8, eta_squared.size) :
    ]
    signal_strength = float(strongest.mean()) if strongest.size else 0.0
    difficulty = float(np.clip(1.0 - signal_strength, 0.0, 1.0))

    variable = np.flatnonzero(np.std(selected, axis=0) > 1e-8)
    if variable.size >= 2:
        scores = _normal_scores(selected[:, variable])
        correlation = np.nan_to_num(np.corrcoef(scores, rowvar=False))
        higher_order = _higher_order_correlation_signature(correlation)
    else:
        higher_order = np.zeros(2, dtype=np.float64)

    sample = selected[: min(selected.shape[0], 256)]
    median = np.median(sample, axis=0)
    q25, q75 = np.quantile(sample, [0.25, 0.75], axis=0)
    scale = np.maximum((q75 - q25) / 1.349, np.std(sample, axis=0))
    scale = np.where(scale > 1e-8, scale, 1.0)
    normalized = np.round((sample - median) / scale, decimals=3)
    unique_fraction = float(
        np.unique(normalized, axis=0).shape[0] / max(normalized.shape[0], 1)
    )
    return {
        "feature_label_nmi": nmi,
        "feature_label_nmi_mean": float(nmi.mean()) if nmi.size else 0.0,
        "feature_label_nmi_max": float(nmi.max()) if nmi.size else 0.0,
        "class_entropy_normalized": normalized_entropy,
        "task_difficulty_proxy": difficulty,
        "higher_order_signature": higher_order,
        "sample_unique_fraction": unique_fraction,
    }


def _blueprint_reference_rows(
    blueprint: FrozenDatasetBlueprint,
    columns: np.ndarray,
    total_rows: int = 384,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic row-free reference reconstructed from frozen quantiles."""

    total_rows = max(int(total_rows), 16 * blueprint.n_classes)
    counts = _closest_proportional_counts(total_rows, blueprint.class_probs)
    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    offsets = np.mod(
        np.arange(columns.size, dtype=np.float64) * 0.6180339887498949,
        1.0,
    )
    for local_class, count in enumerate(counts):
        midpoint = (np.arange(int(count), dtype=np.float64) + 0.5) / int(count)
        uniforms = np.mod(midpoint[:, None] + offsets[None, :], 1.0)
        uniforms = np.clip(
            uniforms, blueprint.quantile_grid[0], blueprint.quantile_grid[-1]
        )
        x_parts.append(
            _inverse_frozen_quantiles(
                uniforms,
                blueprint.quantile_grid,
                blueprint.class_quantiles[local_class, columns],
                blueprint.column_types[columns],
                blueprint.class_zero_rates[local_class, columns],
                blueprint.class_zero_cdf_lower[local_class, columns],
            )
        )
        y_parts.append(
            np.full(int(count), local_class, dtype=np.int64)
        )
    return np.concatenate(x_parts), np.concatenate(y_parts)


def _blueprint_supervised_quality_metrics(
    generated_x: np.ndarray,
    generated_y: np.ndarray,
    blueprint: FrozenDatasetBlueprint,
) -> Dict[str, float]:
    """Label-aware and higher-order checks against a frozen row-free template."""

    width = min(generated_x.shape[1], blueprint.n_features)
    columns = _quality_columns(width)
    levels = blueprint.target_quantile_levels
    scales = blueprint.target_scales[columns]
    informative = scales > 1e-6
    class_errors: List[float] = []
    class_weights: List[float] = []
    for local_class in range(blueprint.n_classes):
        rows = generated_y.astype(np.int64) == local_class
        if not rows.any():
            return {
                "quality_class_conditional_quantile_scaled_mae": float("inf"),
                "quality_feature_label_nmi_mae": float("inf"),
                "quality_higher_order_dependence_error": float("inf"),
                "quality_task_difficulty_error": float("inf"),
            }
        candidate_q = np.quantile(
            generated_x[rows][:, columns], levels, axis=0
        )
        target_q = np.empty_like(candidate_q, dtype=np.float64)
        for local_col, source_col in enumerate(columns):
            target_q[:, local_col] = np.interp(
                levels,
                blueprint.quantile_grid,
                blueprint.class_quantiles[local_class, source_col],
            )
        if informative.any():
            class_error = float(
                np.mean(
                    np.abs(
                        candidate_q[:, informative] - target_q[:, informative]
                    )
                    / scales[informative]
                )
            )
        else:
            class_error = 0.0
        class_errors.append(class_error)
        class_weights.append(float(blueprint.class_probs[local_class]))
    class_conditional_error = float(
        np.average(class_errors, weights=np.asarray(class_weights))
    )

    common_edges = tuple(
        np.unique(blueprint.target_quantiles[:, source_col])
        for source_col in columns
    )
    candidate_structure = _task_structure_metrics(
        generated_x,
        generated_y,
        columns=columns,
        bin_edges=common_edges,
    )
    reference_x, reference_y = _blueprint_reference_rows(blueprint, columns)
    reference_structure = _task_structure_metrics(
        reference_x,
        reference_y,
        columns=np.arange(columns.size, dtype=np.int64),
        bin_edges=common_edges,
    )
    candidate_nmi = np.asarray(candidate_structure["feature_label_nmi"])
    reference_nmi = np.asarray(reference_structure["feature_label_nmi"])
    nmi_error = float(np.mean(np.abs(candidate_nmi - reference_nmi)))

    corr_columns = blueprint.correlation_columns
    corr_columns = corr_columns[corr_columns < width]
    if corr_columns.size > 24:
        corr_columns = corr_columns[
            np.linspace(0, corr_columns.size - 1, num=24, dtype=np.int64)
        ]
    variable = np.asarray(
        [
            column
            for column in corr_columns
            if np.std(generated_x[:, column]) > 1e-8
        ],
        dtype=np.int64,
    )
    if variable.size >= 2:
        lookup = {
            int(column): position
            for position, column in enumerate(blueprint.correlation_columns)
        }
        positions = np.asarray([lookup[int(column)] for column in variable])
        candidate_corr = np.nan_to_num(
            np.corrcoef(_normal_scores(generated_x[:, variable]), rowvar=False)
        )
        target_corr = blueprint.target_rank_correlation[
            np.ix_(positions, positions)
        ]
        higher_order_error = float(
            np.mean(
                np.abs(
                    _higher_order_correlation_signature(candidate_corr)
                    - _higher_order_correlation_signature(target_corr)
                )
            )
        )
    else:
        higher_order_error = 0.0
    difficulty_error = abs(
        float(candidate_structure["task_difficulty_proxy"])
        - float(reference_structure["task_difficulty_proxy"])
    )
    return {
        "quality_class_conditional_quantile_scaled_mae": class_conditional_error,
        "quality_feature_label_nmi_mae": nmi_error,
        "quality_feature_label_nmi": float(
            candidate_structure["feature_label_nmi_mean"]
        ),
        "quality_feature_label_nmi_target": float(
            reference_structure["feature_label_nmi_mean"]
        ),
        "quality_higher_order_dependence_error": higher_order_error,
        "quality_task_difficulty_proxy": float(
            candidate_structure["task_difficulty_proxy"]
        ),
        "quality_task_difficulty_target": float(
            reference_structure["task_difficulty_proxy"]
        ),
        "quality_task_difficulty_error": float(difficulty_error),
        "quality_sample_unique_fraction": float(
            candidate_structure["sample_unique_fraction"]
        ),
    }


def _task_fingerprint(
    values: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Fixed-width signature used to reject duplicate tasks within a batch."""

    columns = _quality_columns(values.shape[1])
    selected = np.asarray(values[:, columns], dtype=np.float64)
    structure = _task_structure_metrics(
        selected,
        labels,
        columns=np.arange(selected.shape[1], dtype=np.int64),
    )
    median = np.median(selected, axis=0)
    scale = np.std(selected, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    normalized = (selected - median) / scale
    weights = np.cos(np.arange(selected.shape[1], dtype=np.float64) + 0.5)
    weights /= max(float(np.linalg.norm(weights)), 1e-12)
    projection = normalized @ weights
    projection_q = np.quantile(projection, [0.1, 0.25, 0.5, 0.75, 0.9])
    higher_order = np.asarray(structure["higher_order_signature"])
    return np.asarray(
        [
            min(values.shape[1], HYBRID178_MAX_FEATURES)
            / HYBRID178_MAX_FEATURES,
            min(np.unique(labels).size, HYBRID178_MAX_CLASSES)
            / HYBRID178_MAX_CLASSES,
            float(structure["class_entropy_normalized"]),
            float(structure["feature_label_nmi_mean"]),
            float(structure["feature_label_nmi_max"]),
            float(structure["task_difficulty_proxy"]),
            float(higher_order[0]),
            float(higher_order[1]),
            float(structure["sample_unique_fraction"]),
            *projection_q.tolist(),
        ],
        dtype=np.float64,
    )


def _column_types(x: np.ndarray, kinds: np.ndarray) -> np.ndarray:
    types: List[str] = []
    for col in range(x.shape[1]):
        values = x[:, col]
        unique = np.unique(values)
        if kinds[col] == "categorical":
            types.append("binary" if unique.size <= 2 else "low_card")
            continue
        zero_ratio = float(np.mean(np.isclose(values, 0.0, atol=1e-8)))
        integer_ratio = float(np.mean(np.isclose(values, np.round(values), atol=1e-6)))
        if unique.size <= 2:
            types.append("binary")
        elif unique.size <= 16:
            types.append("low_card")
        elif zero_ratio >= 0.20:
            types.append("sparse")
        elif integer_ratio >= 0.98:
            types.append("integer")
        else:
            types.append("continuous")
    return np.asarray(types, dtype=object)


def _merge_class_groups(
    data: ProfileData, max_classes: int
) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
    """Map every real class into at most ``max_classes`` synthetic classes.

    Classes are assigned by deterministic greedy mass balancing.  Unlike
    choosing ten labels from a multiclass dataset, this retains information
    from every source class while keeping the model head bounded at ten.
    """

    class_ids = np.asarray(
        [idx for idx, rows in enumerate(data.class_indices) if rows.size],
        dtype=np.int64,
    )
    if class_ids.size < 2:
        raise ValueError("profile has fewer than two non-empty classes")
    target_k = min(int(max_classes), int(class_ids.size))
    if class_ids.size <= target_k:
        groups = tuple(np.asarray([idx], dtype=np.int64) for idx in class_ids)
    else:
        masses = data.class_probs[class_ids]
        order = np.argsort(-masses, kind="stable")
        bins: List[List[int]] = [[] for _ in range(target_k)]
        bin_mass = np.zeros(target_k, dtype=np.float64)
        for position in order:
            destination = int(np.argmin(bin_mass))
            source_class = int(class_ids[position])
            bins[destination].append(source_class)
            bin_mass[destination] += float(data.class_probs[source_class])
        groups = tuple(
            np.asarray(sorted(group), dtype=np.int64) for group in bins
        )
    probs = np.asarray(
        [float(data.class_probs[group].sum()) for group in groups],
        dtype=np.float64,
    )
    probs /= probs.sum()
    return groups, probs


def _normal_scores(values: np.ndarray) -> np.ndarray:
    """Column-wise rank Gaussianization using mid-ranks for discrete ties."""

    n_rows, width = values.shape
    scores = np.empty((n_rows, width), dtype=np.float64)
    for col in range(width):
        _, inverse, counts = np.unique(
            values[:, col], return_inverse=True, return_counts=True
        )
        cumulative = np.cumsum(counts, dtype=np.float64)
        mid_cdf = (cumulative - 0.5 * counts) / n_rows
        uniforms = mid_cdf[inverse]
        scores[:, col] = norm.ppf(np.clip(uniforms, 1e-4, 1.0 - 1e-4))
    return scores


def _sample_rank_gaussian(
    z_emp: np.ndarray,
    count: int,
    rng: np.random.Generator,
    *,
    full_covariance_max_features: int = 128,
    low_rank_max_factors: int = 32,
) -> Tuple[np.ndarray, str, int]:
    """Sample a shrinkage Gaussian copula, using low rank for wide tables."""

    n_fit, width = z_emp.shape
    if n_fit <= 1:
        return rng.standard_normal((count, width)), "independent", 0
    centered = z_emp - z_emp.mean(axis=0, keepdims=True)
    scales = centered.std(axis=0, ddof=1)
    standardized = np.divide(
        centered,
        scales,
        out=np.zeros_like(centered),
        where=scales > 1e-8,
    )

    if width <= int(full_covariance_max_features):
        covariance = np.atleast_2d(np.cov(standardized, rowvar=False))
        if covariance.shape != (width, width) or not np.isfinite(
            covariance
        ).all():
            covariance = np.eye(width, dtype=np.float64)
        diagonal = np.clip(np.diag(covariance), 1e-4, None)
        shrink = float(np.clip(width / max(n_fit + width, 1), 0.10, 0.90))
        covariance = (
            (1.0 - shrink) * covariance + shrink * np.diag(diagonal)
        )
        eigenvalues, eigenvectors = np.linalg.eigh(
            (covariance + covariance.T) * 0.5
        )
        factor = eigenvectors @ np.diag(
            np.sqrt(np.clip(eigenvalues, 1e-5, None))
        )
        return (
            rng.standard_normal((count, width)) @ factor.T,
            "full_shrinkage",
            width,
        )

    rank = min(int(low_rank_max_factors), n_fit - 1, width)
    if rank < 1:
        return rng.standard_normal((count, width)), "independent", 0
    oversampled_rank = min(width, rank + 8)
    matrix = standardized / np.sqrt(max(n_fit - 1, 1))
    projection = rng.standard_normal((width, oversampled_rank))
    projected = matrix @ projection
    q, _ = np.linalg.qr(projected, mode="reduced")
    compressed = q.T @ matrix
    _, singular_values, vt = np.linalg.svd(compressed, full_matrices=False)
    rank = min(rank, singular_values.size)
    loadings = vt[:rank].T * singular_values[:rank]
    explained = np.sum(loadings * loadings, axis=1)
    residual = np.clip(1.0 - explained, 1e-4, 1.0)
    factors = rng.standard_normal((count, rank)) @ loadings.T
    noise = rng.standard_normal((count, width)) * np.sqrt(residual)
    sampled = factors + noise
    return sampled, "low_rank", rank


def _fit_rank_gaussian_factor(
    z_emp: np.ndarray,
    rng: np.random.Generator,
    *,
    full_covariance_max_features: int = 128,
    low_rank_max_factors: int = 32,
) -> Tuple[str, np.ndarray, np.ndarray]:
    """Fit a reusable copula factor without storing any source row."""

    n_fit, width = z_emp.shape
    if n_fit <= 1:
        return (
            "independent",
            np.empty((width, 0), dtype=np.float64),
            np.ones(width, dtype=np.float64),
        )
    centered = z_emp - z_emp.mean(axis=0, keepdims=True)
    scales = centered.std(axis=0, ddof=1)
    standardized = np.divide(
        centered,
        scales,
        out=np.zeros_like(centered),
        where=scales > 1e-8,
    )
    if width <= int(full_covariance_max_features):
        covariance = np.atleast_2d(np.cov(standardized, rowvar=False))
        if covariance.shape != (width, width) or not np.isfinite(
            covariance
        ).all():
            covariance = np.eye(width, dtype=np.float64)
        diagonal = np.clip(np.diag(covariance), 1e-4, None)
        shrink = float(np.clip(width / max(n_fit + width, 1), 0.10, 0.90))
        covariance = (1.0 - shrink) * covariance + shrink * np.diag(diagonal)
        eigenvalues, eigenvectors = np.linalg.eigh(
            (covariance + covariance.T) * 0.5
        )
        factor = eigenvectors @ np.diag(
            np.sqrt(np.clip(eigenvalues, 1e-5, None))
        )
        return "full_shrinkage", factor, np.empty(0, dtype=np.float64)

    rank = min(int(low_rank_max_factors), n_fit - 1, width)
    if rank < 1:
        return (
            "independent",
            np.empty((width, 0), dtype=np.float64),
            np.ones(width, dtype=np.float64),
        )
    oversampled_rank = min(width, rank + 8)
    matrix = standardized / np.sqrt(max(n_fit - 1, 1))
    projection = rng.standard_normal((width, oversampled_rank))
    projected = matrix @ projection
    q, _ = np.linalg.qr(projected, mode="reduced")
    compressed = q.T @ matrix
    _, singular_values, vt = np.linalg.svd(compressed, full_matrices=False)
    rank = min(rank, singular_values.size)
    loadings = vt[:rank].T * singular_values[:rank]
    explained = np.sum(loadings * loadings, axis=1)
    residual = np.clip(1.0 - explained, 1e-4, 1.0)
    return "low_rank", loadings, residual


def _sample_frozen_copula(
    mode: str,
    factor: np.ndarray,
    residual: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    width = int(factor.shape[0])
    if mode == "full_shrinkage":
        return rng.standard_normal((count, factor.shape[1])) @ factor.T
    if mode == "low_rank":
        latent = rng.standard_normal((count, factor.shape[1])) @ factor.T
        noise = rng.standard_normal((count, width)) * np.sqrt(residual)
        return latent + noise
    if mode == "independent":
        return rng.standard_normal((count, width))
    raise ValueError(f"unknown frozen copula mode {mode!r}")


def _inverse_frozen_quantiles(
    uniforms: np.ndarray,
    quantile_grid: np.ndarray,
    quantiles: np.ndarray,
    column_types: np.ndarray,
    zero_rates: np.ndarray,
    zero_cdf_lower: np.ndarray,
) -> np.ndarray:
    """Apply row-free class-conditional inverse marginals."""

    uniforms = np.asarray(uniforms, dtype=np.float64)
    output = np.empty(uniforms.shape, dtype=np.float32)
    for col, kind in enumerate(column_types):
        u = np.clip(uniforms[:, col], quantile_grid[0], quantile_grid[-1])
        values = quantiles[col].astype(np.float64, copy=False)
        if kind in {"binary", "low_card", "integer"}:
            right = np.searchsorted(quantile_grid, u, side="left")
            right = np.clip(right, 0, quantile_grid.size - 1)
            left = np.maximum(right - 1, 0)
            choose_left = np.abs(u - quantile_grid[left]) <= np.abs(
                quantile_grid[right] - u
            )
            indices = np.where(choose_left, left, right)
            sampled = values[indices]
        else:
            sampled = np.interp(u, quantile_grid, values)
        if float(zero_rates[col]) > 0.0:
            sampled = sampled.copy()
            lower = float(zero_cdf_lower[col])
            upper = lower + float(zero_rates[col])
            zero_mask = (uniforms[:, col] >= lower) & (
                uniforms[:, col] < upper
            )
            sampled[zero_mask] = 0.0
        output[:, col] = sampled.astype(np.float32, copy=False)
    return output


def _transport_to_frozen_blueprint(
    source: np.ndarray,
    labels: np.ndarray,
    blueprint: FrozenDatasetBlueprint,
    rng: np.random.Generator,
    *,
    target_copula_blend_scale: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, float | bool]]:
    """Impose frozen marginals while adaptively correcting spurious rank dependence.

    The GraphSCM copula remains dominant for strongly dependent templates.  For
    weakly dependent, categorical or multiclass templates, however, preserving
    an unrelated GraphSCM copula creates correlations that are absent from the
    target profile.  Blend standardized GraphSCM normal scores with an
    independent row-free draw from the frozen class copula before applying the
    class-conditional inverse marginals.  The adaptive covariance weight is
    deliberately bounded below one, so this remains an official-profile branch
    rather than becoming a second pure-copula branch.
    """

    target_correlation = np.asarray(
        blueprint.target_rank_correlation, dtype=np.float64
    )
    if target_correlation.shape[0] >= 2:
        upper = np.triu_indices(target_correlation.shape[0], k=1)
        dependency_strength = float(
            np.mean(np.abs(target_correlation[upper]))
        )
    else:
        dependency_strength = 0.0
    discrete_fraction = float(
        np.mean(
            np.isin(
                blueprint.column_types.astype(str),
                ("binary", "low_card", "integer"),
            )
        )
    )
    class_complexity = float(
        np.clip((blueprint.n_classes - 2) / 8.0, 0.0, 1.0)
    )
    weak_dependency = float(
        np.clip((0.30 - dependency_strength) / 0.30, 0.0, 1.0)
    )
    blend_scale = float(target_copula_blend_scale)
    if not np.isfinite(blend_scale) or not 0.0 <= blend_scale <= 1.0:
        raise ValueError("target_copula_blend_scale must be in [0, 1]")
    unscaled_target_weight = float(
        np.clip(
            0.15
            + 0.70 * weak_dependency
            + 0.10 * discrete_fraction
            + 0.05 * class_complexity,
            0.15,
            0.90,
        )
    )
    # Retain the official GraphSCM row-level structure as the backbone.  A
    # scale below one only weakly steers its normal scores toward the frozen
    # target copula; class-conditional marginals and label proportions remain
    # fully profile matched by the inverse transport below.
    target_weight = float(unscaled_target_weight * blend_scale)

    output = np.empty_like(source, dtype=np.float32)
    for local_class in range(blueprint.n_classes):
        rows = np.flatnonzero(labels == local_class)
        if rows.size == 0:
            raise ValueError("frozen transport encountered an empty class")
        source_scores = _normal_scores(source[rows])
        source_scale = source_scores.std(axis=0, ddof=0)
        source_scores = np.divide(
            source_scores - source_scores.mean(axis=0, keepdims=True),
            source_scale,
            out=np.zeros_like(source_scores),
            where=source_scale > 1e-8,
        )
        target_scores = _sample_frozen_copula(
            blueprint.copula_modes[local_class],
            blueprint.copula_factors[local_class],
            blueprint.copula_residuals[local_class],
            rows.size,
            rng,
        )
        target_scale = target_scores.std(axis=0, ddof=0)
        target_scores = np.divide(
            target_scores - target_scores.mean(axis=0, keepdims=True),
            target_scale,
            out=np.zeros_like(target_scores),
            where=target_scale > 1e-8,
        )
        blended_scores = (
            np.sqrt(1.0 - target_weight) * source_scores
            + np.sqrt(target_weight) * target_scores
        )
        uniforms = _rank_stratified_uniforms(blended_scores)
        output[rows] = _inverse_frozen_quantiles(
            uniforms,
            blueprint.quantile_grid,
            blueprint.class_quantiles[local_class],
            blueprint.column_types,
            blueprint.class_zero_rates[local_class],
            blueprint.class_zero_cdf_lower[local_class],
        )
    return output, {
        "official_profile_rank_copula_shrinkage": True,
        "official_profile_target_copula_weight": target_weight,
        "official_profile_target_copula_weight_unscaled": unscaled_target_weight,
        "official_profile_copula_blend_scale": blend_scale,
        "official_profile_target_dependency_strength": dependency_strength,
        "official_profile_discrete_fraction": discrete_fraction,
        "official_profile_class_complexity": class_complexity,
    }


def _inverse_empirical_column(
    uniforms: np.ndarray,
    empirical: np.ndarray,
    kind: str,
    rng: np.random.Generator,
    *,
    continuous_noise: float,
) -> np.ndarray:
    """Smooth inverse empirical CDF without selecting a source row."""

    values = np.asarray(empirical, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.zeros(uniforms.size, dtype=np.float32)
    uniforms = np.clip(np.asarray(uniforms, dtype=np.float64), 1e-6, 1 - 1e-6)
    if kind == "sparse":
        zeros = np.isclose(values, 0.0, atol=1e-8)
        zero_probability = float(zeros.mean())
        out = np.zeros(uniforms.size, dtype=np.float32)
        active = uniforms >= zero_probability
        nonzero = values[~zeros]
        if active.any() and nonzero.size:
            active_uniforms = (
                uniforms[active] - zero_probability
            ) / max(1.0 - zero_probability, 1e-12)
            integer_ratio = float(
                np.mean(np.isclose(nonzero, np.round(nonzero), atol=1e-6))
            )
            active_kind = "integer" if integer_ratio >= 0.98 else "continuous"
            out[active] = _inverse_empirical_column(
                active_uniforms,
                nonzero,
                active_kind,
                rng,
                continuous_noise=continuous_noise,
            )
        return out
    if kind in {"binary", "low_card", "integer"}:
        unique, counts = np.unique(values, return_counts=True)
        cdf = np.cumsum(counts, dtype=np.float64) / counts.sum()
        indices = np.searchsorted(cdf, uniforms, side="right")
        return unique[np.clip(indices, 0, unique.size - 1)].astype(np.float32)

    sampled = np.quantile(values, uniforms, method="linear")
    if continuous_noise > 0.0 and np.unique(values).size > 2:
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median))) * 1.4826
        q25, q75 = np.quantile(values, [0.25, 0.75])
        scale = max(mad, float((q75 - q25) / 1.349), 1e-6)
        sampled += rng.normal(0.0, continuous_noise * scale, sampled.size)
        low, high = np.quantile(values, [0.001, 0.999])
        sampled = np.clip(sampled, low, high)
    return sampled.astype(np.float32)


def _class_conditional_transport(
    source: np.ndarray,
    labels: np.ndarray,
    template_x: np.ndarray,
    template_y: np.ndarray,
    class_groups: Tuple[np.ndarray, ...],
    kinds: np.ndarray,
    rng: np.random.Generator,
    *,
    continuous_noise: float = 0.01,
) -> np.ndarray:
    """Map a synthetic copula to real class-conditional feature marginals.

    Each output value is obtained from an inverse marginal CDF.  No source row
    index is ever sampled, and a small continuous jitter prevents the mapping
    from degenerating into a lookup table for numeric columns.
    """

    out = np.empty_like(source, dtype=np.float32)
    column_types = _column_types(template_x, kinds)
    for local_class, source_classes in enumerate(class_groups):
        target_rows = np.flatnonzero(labels == local_class)
        empirical_rows = template_x[np.isin(template_y, source_classes)]
        if target_rows.size == 0 or empirical_rows.shape[0] == 0:
            raise ValueError("empty class during class-conditional transport")
        for col, kind in enumerate(column_types):
            values = source[target_rows, col]
            order = np.argsort(values, kind="mergesort")
            uniforms = np.empty(target_rows.size, dtype=np.float64)
            uniforms[order] = (
                np.arange(target_rows.size, dtype=np.float64) + 0.5
            ) / target_rows.size
            out[target_rows, col] = _inverse_empirical_column(
                uniforms,
                empirical_rows[:, col],
                str(kind),
                rng,
                continuous_noise=continuous_noise,
            )
    return out


def _empirical_transport(source: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Column-wise monotone map that preserves the source rank copula."""

    out = np.empty_like(source, dtype=np.float32)
    n_rows = source.shape[0]
    for col in range(source.shape[1]):
        order = np.argsort(source[:, col], kind="mergesort")
        ranks = np.empty(n_rows, dtype=np.float64)
        ranks[order] = (np.arange(n_rows, dtype=np.float64) + 0.5) / n_rows
        out[:, col] = np.quantile(template[:, col], ranks, method="linear").astype(np.float32)
    return out


def _inject_gt_atoms(
    x: np.ndarray,
    y: np.ndarray,
    data: ProfileData,
    class_groups: Tuple[np.ndarray, ...],
    rng: np.random.Generator,
    *,
    ratio: float,
    min_rows: int,
    require_full_schema: bool,
    train_size: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Mix exact preprocessed GT-train rows into a generated task.

    A continuous generator assigns probability zero to every exact floating
    point GT row.  This helper adds a finite empirical component while keeping
    the generated labels and class balance unchanged.  Source rows are sampled
    within the output row's (possibly merged) class group.  Whenever possible,
    source rows are unique across support and query.
    """

    ratio = float(ratio)
    min_rows = int(min_rows)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("GT atom ratio must be in [0, 1]")
    if min_rows < 0:
        raise ValueError("GT atom minimum must be non-negative")
    if ratio == 0.0:
        return x, {
            "gt_atom_ratio_requested": 0.0,
            "gt_atom_count": 0,
            "gt_atom_fraction_actual": 0.0,
            "gt_atom_min_rows": min_rows,
            "gt_atom_finite_sample_hit_guaranteed": False,
        }
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.size:
        raise ValueError("GT atom injection expects x=(n,d) and y=(n,)")
    if require_full_schema and x.shape[1] != data.x.shape[1]:
        raise ValueError(
            "GT atom injection requires the complete GT feature schema: "
            f"generated={x.shape[1]}, GT={data.x.shape[1]}"
        )
    if x.shape[1] > data.x.shape[1]:
        raise ValueError("generated task is wider than the GT feature schema")

    count = min(y.size, max(min_rows, int(np.ceil(ratio * y.size))))
    if count <= 0:
        raise AssertionError("enabled GT atom mixture produced no atoms")
    atom_slots = rng.choice(y.size, size=count, replace=False).astype(np.int64)
    source_rows = np.full(count, -1, dtype=np.int64)
    out = x.astype(np.float32, copy=True)

    for local_class, source_classes in enumerate(class_groups):
        positions = np.flatnonzero(y[atom_slots] == local_class)
        if positions.size == 0:
            continue
        pool = np.concatenate(
            [data.class_indices[int(source_class)] for source_class in source_classes]
        )
        if pool.size == 0:
            raise ValueError(f"GT atom class group {local_class} has no source rows")
        chosen = rng.choice(
            pool,
            size=positions.size,
            replace=positions.size > pool.size,
        ).astype(np.int64)
        slots = atom_slots[positions]
        out[slots] = data.x[chosen, : x.shape[1]]
        source_rows[positions] = chosen

    if (source_rows < 0).any():
        missing = np.unique(y[atom_slots[source_rows < 0]]).tolist()
        raise ValueError(f"GT atom output labels have no class group: {missing}")
    if not np.array_equal(out[atom_slots], data.x[source_rows, : x.shape[1]]):
        raise AssertionError("GT atom injection failed exact preprocessed-row equality")

    support_sources = source_rows[atom_slots < int(train_size)]
    query_sources = source_rows[atom_slots >= int(train_size)]
    overlap = np.intersect1d(support_sources, query_sources)
    source_classes = data.y[source_rows]
    labels_exact = all(
        group.size == 1 and int(group[0]) == local_class
        for local_class, group in enumerate(class_groups)
    )
    return out, {
        "gt_atom_ratio_requested": ratio,
        "gt_atom_count": int(count),
        "gt_atom_fraction_actual": float(count / y.size),
        "gt_atom_min_rows": min_rows,
        "gt_atom_finite_sample_hit_guaranteed": True,
        "gt_atom_preprocessed_x_exact": True,
        "gt_atom_labels_exact": bool(labels_exact),
        "gt_atom_full_schema": x.shape[1] == data.x.shape[1],
        "gt_atom_slots": atom_slots.tolist(),
        "gt_atom_source_rows": source_rows.tolist(),
        "gt_atom_source_classes": source_classes.tolist(),
        "gt_atom_unique_source_rows": int(np.unique(source_rows).size),
        "gt_atom_support_query_source_disjoint": overlap.size == 0,
        "source_rows_sampled": True,
        "row_bootstrap": True,
        "empirical_atom_sampling": True,
        "synthetic_only": False,
        "real_row_replay": True,
    }


class Hybrid178Prior:
    """Official GraphSCM plus train-side profile-conditioned statistical branches."""

    def __init__(
        self,
        *,
        hybrid178_data_root: str,
        hybrid178_include_val: bool = False,
        hybrid178_profile_ratio: float = 0.20,
        hybrid178_copula_ratio: float = 0.05,
        hybrid178_profile_transport_ratio: float = 1.0,
        hybrid178_protected_profile_transport_ratio: Optional[float] = None,
        hybrid178_target_profile_transport_ratio: Optional[float] = None,
        hybrid178_profile_shape_jitter_strength: float = 0.0,
        hybrid178_protected_profile_shape_jitter_strength: Optional[float] = None,
        hybrid178_target_profile_shape_jitter_strength: Optional[float] = None,
        hybrid178_profile_supervised_candidate_count: int = 1,
        hybrid178_protected_profile_supervised_candidate_count: Optional[int] = None,
        hybrid178_target_profile_supervised_candidate_count: Optional[int] = None,
        hybrid178_profile_copula_blend_scale: float = 1.0,
        hybrid178_protected_profile_copula_blend_scale: Optional[float] = None,
        hybrid178_target_profile_copula_blend_scale: Optional[float] = None,
        hybrid178_smooth_ratio: float = 0.0,
        hybrid178_train_fraction: float = 0.80,
        hybrid178_seed: int = 178,
        hybrid178_schedule_start_batch: int = 0,
        hybrid178_profile_cache_size: int = 2,
        hybrid178_profile_max_rows: int = 200_000,
        hybrid178_exact_replay: bool = False,
        hybrid178_gt_atom_ratio: float = 0.0,
        hybrid178_gt_atom_min_rows: int = 1,
        hybrid178_gt_atom_require_full_schema: bool = True,
        hybrid178_runtime_isolated: bool = False,
        hybrid178_forbidden_data_root: Optional[str] = None,
        hybrid178_quality_gate: bool = True,
        hybrid178_protected_priority_extras: bool = False,
        hybrid178_protected_priority_hardness_weighted: bool = False,
        hybrid178_protected_priority_collapse_risk_weighted: bool = False,
        config: Optional[PriorConfig] = None,
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
        return_metadata: bool = False,
        **_unused,
    ):
        ratios = np.asarray(
            [hybrid178_profile_ratio, hybrid178_copula_ratio], dtype=np.float64
        )
        if not np.isfinite(ratios).all() or (ratios < 0).any() or ratios.sum() > 1.0 + 1e-12:
            raise ValueError(
                "hybrid178 profile/copula ratios must be non-negative and sum to at most 1"
            )
        profile_copula_blend_scale = float(
            hybrid178_profile_copula_blend_scale
        )
        profile_transport_ratio = float(hybrid178_profile_transport_ratio)
        if (
            not np.isfinite(profile_transport_ratio)
            or not 0.0 <= profile_transport_ratio <= 1.0
        ):
            raise ValueError(
                "hybrid178_profile_transport_ratio must be in [0, 1]"
            )
        protected_profile_transport_ratio = (
            None
            if hybrid178_protected_profile_transport_ratio is None
            else float(hybrid178_protected_profile_transport_ratio)
        )
        if protected_profile_transport_ratio is not None and (
            not np.isfinite(protected_profile_transport_ratio)
            or not 0.0 <= protected_profile_transport_ratio <= 1.0
        ):
            raise ValueError(
                "hybrid178_protected_profile_transport_ratio must be in [0, 1]"
            )
        target_profile_transport_ratio = (
            None
            if hybrid178_target_profile_transport_ratio is None
            else float(hybrid178_target_profile_transport_ratio)
        )
        if target_profile_transport_ratio is not None and (
            not np.isfinite(target_profile_transport_ratio)
            or not 0.0 <= target_profile_transport_ratio <= 1.0
        ):
            raise ValueError(
                "hybrid178_target_profile_transport_ratio must be in [0, 1]"
            )
        profile_shape_jitter_strength = float(
            hybrid178_profile_shape_jitter_strength
        )
        protected_profile_shape_jitter_strength = (
            None
            if hybrid178_protected_profile_shape_jitter_strength is None
            else float(hybrid178_protected_profile_shape_jitter_strength)
        )
        target_profile_shape_jitter_strength = (
            None
            if hybrid178_target_profile_shape_jitter_strength is None
            else float(hybrid178_target_profile_shape_jitter_strength)
        )
        for argument_name, strength in (
            ("hybrid178_profile_shape_jitter_strength", profile_shape_jitter_strength),
            (
                "hybrid178_protected_profile_shape_jitter_strength",
                protected_profile_shape_jitter_strength,
            ),
            (
                "hybrid178_target_profile_shape_jitter_strength",
                target_profile_shape_jitter_strength,
            ),
        ):
            if strength is not None and (
                not np.isfinite(strength) or not 0.0 <= strength <= 1.0
            ):
                raise ValueError(f"{argument_name} must be in [0, 1]")
        profile_supervised_candidate_count = int(
            hybrid178_profile_supervised_candidate_count
        )
        protected_profile_supervised_candidate_count = (
            None
            if hybrid178_protected_profile_supervised_candidate_count is None
            else int(hybrid178_protected_profile_supervised_candidate_count)
        )
        target_profile_supervised_candidate_count = (
            None
            if hybrid178_target_profile_supervised_candidate_count is None
            else int(hybrid178_target_profile_supervised_candidate_count)
        )
        for argument_name, candidate_count in (
            (
                "hybrid178_profile_supervised_candidate_count",
                profile_supervised_candidate_count,
            ),
            (
                "hybrid178_protected_profile_supervised_candidate_count",
                protected_profile_supervised_candidate_count,
            ),
            (
                "hybrid178_target_profile_supervised_candidate_count",
                target_profile_supervised_candidate_count,
            ),
        ):
            if candidate_count is not None and not 1 <= candidate_count <= 8:
                raise ValueError(f"{argument_name} must be in [1, 8]")
        effective_protected_candidate_count = (
            profile_supervised_candidate_count
            if protected_profile_supervised_candidate_count is None
            else protected_profile_supervised_candidate_count
        )
        effective_target_candidate_count = (
            profile_supervised_candidate_count
            if target_profile_supervised_candidate_count is None
            else target_profile_supervised_candidate_count
        )
        if max(
            profile_supervised_candidate_count,
            effective_protected_candidate_count,
            effective_target_candidate_count,
        ) > 1 and (
            not bool(hybrid178_runtime_isolated)
            or not bool(hybrid178_quality_gate)
        ):
            raise ValueError(
                "hybrid178 supervised candidate counts above one require "
                "runtime-isolated mode with the quality gate enabled"
            )
        effective_protected_jitter = (
            profile_shape_jitter_strength
            if protected_profile_shape_jitter_strength is None
            else protected_profile_shape_jitter_strength
        )
        effective_target_jitter = (
            profile_shape_jitter_strength
            if target_profile_shape_jitter_strength is None
            else target_profile_shape_jitter_strength
        )
        if any(
            candidate_count > 1 and jitter_strength != 0.0
            for candidate_count, jitter_strength in (
                (profile_supervised_candidate_count, profile_shape_jitter_strength),
                (
                    effective_protected_candidate_count,
                    effective_protected_jitter,
                ),
                (effective_target_candidate_count, effective_target_jitter),
            )
        ):
            raise ValueError(
                "supervised candidate selection and profile shape jitter are "
                "mutually exclusive"
            )
        if (
            not np.isfinite(profile_copula_blend_scale)
            or not 0.0 <= profile_copula_blend_scale <= 1.0
        ):
            raise ValueError(
                "hybrid178_profile_copula_blend_scale must be in [0, 1]"
            )
        protected_profile_copula_blend_scale = (
            None
            if hybrid178_protected_profile_copula_blend_scale is None
            else float(hybrid178_protected_profile_copula_blend_scale)
        )
        if protected_profile_copula_blend_scale is not None and (
            not np.isfinite(protected_profile_copula_blend_scale)
            or not 0.0 <= protected_profile_copula_blend_scale <= 1.0
        ):
            raise ValueError(
                "hybrid178_protected_profile_copula_blend_scale must be in [0, 1]"
            )
        target_profile_copula_blend_scale = (
            None
            if hybrid178_target_profile_copula_blend_scale is None
            else float(hybrid178_target_profile_copula_blend_scale)
        )
        if target_profile_copula_blend_scale is not None and (
            not np.isfinite(target_profile_copula_blend_scale)
            or not 0.0 <= target_profile_copula_blend_scale <= 1.0
        ):
            raise ValueError(
                "hybrid178_target_profile_copula_blend_scale must be in [0, 1]"
            )
        if float(hybrid178_smooth_ratio) != 0.0:
            raise ValueError(
                "hybrid178_smooth_ratio must be zero: real-row smoothed bootstrap "
                "is not a supported Hybrid-178 branch"
            )
        if bool(hybrid178_exact_replay):
            raise ValueError(
                "hybrid178_exact_replay is no longer supported: Hybrid-178 must "
                "use the bounded hybrid178_gt_atom_ratio instead of whole-task replay"
            )
        gt_atom_ratio = float(hybrid178_gt_atom_ratio)
        gt_atom_min_rows = int(hybrid178_gt_atom_min_rows)
        if not np.isfinite(gt_atom_ratio) or not 0.0 <= gt_atom_ratio <= 1.0:
            raise ValueError("hybrid178_gt_atom_ratio must be in [0, 1]")
        if gt_atom_min_rows < 0:
            raise ValueError("hybrid178_gt_atom_min_rows must be non-negative")
        if gt_atom_ratio > 0.0 and gt_atom_min_rows < 1:
            raise ValueError(
                "hybrid178_gt_atom_min_rows must be at least one when GT atoms are enabled"
            )
        if not 0.0 < float(hybrid178_train_fraction) < 1.0:
            raise ValueError("hybrid178_train_fraction must be between zero and one")
        if max_classes < 2 or max_features < 1:
            raise ValueError("hybrid178 requires max_classes >= 2 and max_features >= 1")
        if int(max_classes) > HYBRID178_MAX_CLASSES:
            raise ValueError(
                f"Hybrid-178 hard-caps generated classes at {HYBRID178_MAX_CLASSES}; "
                f"got max_classes={max_classes}"
            )
        if int(max_features) > HYBRID178_MAX_FEATURES:
            raise ValueError(
                f"Hybrid-178 hard-caps generated features at {HYBRID178_MAX_FEATURES}; "
                f"got max_features={max_features}"
            )
        if int(min_features) > int(max_features):
            raise ValueError("hybrid178 min_features may not exceed max_features")
        schedule_start_batch = int(hybrid178_schedule_start_batch)
        if schedule_start_batch < 0:
            raise ValueError("hybrid178_schedule_start_batch must be non-negative")

        self.runtime_isolated = bool(hybrid178_runtime_isolated)
        self.quality_gate = bool(hybrid178_quality_gate)
        self.protected_priority_extras = bool(
            hybrid178_protected_priority_extras
        )
        self.protected_priority_hardness_weighted = bool(
            hybrid178_protected_priority_hardness_weighted
        )
        self.protected_priority_collapse_risk_weighted = bool(
            hybrid178_protected_priority_collapse_risk_weighted
        )
        if (
            self.protected_priority_hardness_weighted
            and self.protected_priority_collapse_risk_weighted
        ):
            raise ValueError(
                "protected priority hardness and collapse-risk weighting are "
                "mutually exclusive"
            )
        if (
            (
                self.protected_priority_hardness_weighted
                or self.protected_priority_collapse_risk_weighted
            )
            and not self.protected_priority_extras
        ):
            raise ValueError(
                "weighted protected priority requires "
                "hybrid178_protected_priority_extras"
            )
        self.forbidden_data_root = (
            str(Path(hybrid178_forbidden_data_root).expanduser().resolve())
            if hybrid178_forbidden_data_root
            else None
        )
        self.compiled_manifest: Optional[Dict[str, object]] = None
        self.compiled_manifest_digest: Optional[str] = None
        self.blueprint_mode = False
        if self.runtime_isolated:
            if not self.forbidden_data_root:
                raise ValueError(
                    "hybrid178_forbidden_data_root is required in runtime-isolated mode"
                )
            if bool(hybrid178_include_val):
                raise ValueError("runtime-isolated Hybrid-178 permits train artifacts only")
            if gt_atom_ratio != 0.0:
                raise ValueError("runtime-isolated Hybrid-178 requires GT atom ratio zero")
            unresolved_root = Path(hybrid178_data_root).expanduser().absolute()
            _assert_non_symlink_path(unresolved_root, unresolved_root)
            resolved_root = unresolved_root.resolve()
            forbidden_root = Path(self.forbidden_data_root)
            if _path_is_within(resolved_root, forbidden_root) or _path_is_within(
                forbidden_root, resolved_root
            ):
                raise ValueError(
                    "runtime-isolated artifact root must be disjoint from the forbidden GT root"
                )
            manifest_path = resolved_root / ISOLATED_MANIFEST
            _assert_non_symlink_path(manifest_path, resolved_root)
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"runtime-isolated Hybrid-178 requires {manifest_path}"
                )
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
            artifact_type = manifest.get("artifact_type")
            if artifact_type == "hybrid178_frozen_statistical_blueprint_v3":
                required_values = {
                    "schema_version": 3,
                    "artifact_type": artifact_type,
                    "blueprint_filename": BLUEPRINT_V3_FILENAME,
                    "generation_max_classes": HYBRID178_MAX_CLASSES,
                    "generation_max_features": HYBRID178_MAX_FEATURES,
                    "deterministic_full_feature_coverage": True,
                    "contains_source_rows": False,
                    "source_rows_sampled": False,
                    "source_row_records_stored": False,
                    "runtime_real_data_access_required": False,
                    "test_data_used": False,
                }
                self.blueprint_mode = True
            elif artifact_type == "hybrid178_frozen_statistical_blueprint_v2":
                required_values = {
                    "schema_version": 2,
                    "artifact_type": artifact_type,
                    "contains_source_rows": False,
                    "source_rows_sampled": False,
                    "source_row_records_stored": False,
                    "runtime_real_data_access_required": False,
                    "test_data_used": False,
                }
                self.blueprint_mode = True
            elif artifact_type == "hybrid178_synthetic_surrogate_bank":
                required_values = {
                    "schema_version": 1,
                    "artifact_type": artifact_type,
                    "contains_source_rows": False,
                    "source_rows_sampled": False,
                    "runtime_real_data_access_required": False,
                    "test_data_used": False,
                }
            else:
                raise ValueError(
                    f"unsupported isolated Hybrid-178 artifact type: {artifact_type!r}"
                )
            for key, expected in required_values.items():
                if manifest.get(key) != expected:
                    raise ValueError(
                        f"invalid isolated manifest field {key!r}: "
                        f"expected {expected!r}, got {manifest.get(key)!r}"
                    )
            self.compiled_manifest = manifest
            self.compiled_manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()

        any_partial_transport = any(
            ratio is not None and ratio < 1.0
            for ratio in (
                profile_transport_ratio,
                protected_profile_transport_ratio,
                target_profile_transport_ratio,
            )
        )
        if any_partial_transport and not self.blueprint_mode:
            raise ValueError(
                "hybrid178_profile_transport_ratio below one requires a "
                "row-free frozen statistical blueprint"
            )

        self.data_root = str(Path(hybrid178_data_root).expanduser().resolve())
        self.include_val = bool(hybrid178_include_val)
        self.exact_replay = False
        self.gt_atom_ratio = gt_atom_ratio
        self.gt_atom_min_rows = gt_atom_min_rows
        self.gt_atom_require_full_schema = bool(
            hybrid178_gt_atom_require_full_schema
        )
        self.branch_probs = np.asarray(
            [1.0 - ratios.sum(), *ratios], dtype=np.float64
        )
        self.profile_transport_ratio = profile_transport_ratio
        transport_fraction = Fraction(str(profile_transport_ratio)).limit_denominator(
            1024
        )
        self.profile_transport_numerator = int(transport_fraction.numerator)
        self.profile_transport_denominator = int(transport_fraction.denominator)
        self.protected_profile_transport_ratio = protected_profile_transport_ratio
        protected_transport_fraction = (
            None
            if protected_profile_transport_ratio is None
            else Fraction(str(protected_profile_transport_ratio)).limit_denominator(
                1024
            )
        )
        self.protected_profile_transport_numerator = (
            None
            if protected_transport_fraction is None
            else int(protected_transport_fraction.numerator)
        )
        self.protected_profile_transport_denominator = (
            None
            if protected_transport_fraction is None
            else int(protected_transport_fraction.denominator)
        )
        self.target_profile_transport_ratio = target_profile_transport_ratio
        target_transport_fraction = (
            None
            if target_profile_transport_ratio is None
            else Fraction(str(target_profile_transport_ratio)).limit_denominator(1024)
        )
        self.target_profile_transport_numerator = (
            None
            if target_transport_fraction is None
            else int(target_transport_fraction.numerator)
        )
        self.target_profile_transport_denominator = (
            None
            if target_transport_fraction is None
            else int(target_transport_fraction.denominator)
        )
        self.profile_shape_jitter_strength = profile_shape_jitter_strength
        self.protected_profile_shape_jitter_strength = (
            protected_profile_shape_jitter_strength
        )
        self.target_profile_shape_jitter_strength = (
            target_profile_shape_jitter_strength
        )
        self.profile_supervised_candidate_count = (
            profile_supervised_candidate_count
        )
        self.protected_profile_supervised_candidate_count = (
            protected_profile_supervised_candidate_count
        )
        self.target_profile_supervised_candidate_count = (
            target_profile_supervised_candidate_count
        )
        self.profile_copula_blend_scale = profile_copula_blend_scale
        self.protected_profile_copula_blend_scale = (
            protected_profile_copula_blend_scale
        )
        self.target_profile_copula_blend_scale = (
            target_profile_copula_blend_scale
        )
        self.train_fraction = float(hybrid178_train_fraction)
        self.base_seed = int(hybrid178_seed)
        self.schedule_start_batch = schedule_start_batch
        self.profile_cache_size = int(hybrid178_profile_cache_size)
        self.profile_max_rows = int(hybrid178_profile_max_rows)
        self.batch_size = int(batch_size)
        self.batch_size_per_gp = max(1, int(batch_size_per_gp))
        self.min_features = int(min_features)
        self.max_features = int(max_features)
        self.max_classes = int(max_classes)
        self.min_seq_len = min_seq_len
        self.max_seq_len = int(max_seq_len)
        self.log_seq_len = bool(log_seq_len)
        self.seq_len_per_gp = bool(seq_len_per_gp)
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.replay_small = bool(replay_small)
        self.device = device
        self.return_metadata = bool(return_metadata)
        self.high_fidelity = self.runtime_isolated
        # The dominant branch must use the same GraphSCM configuration object
        # as the standalone official GraphPrior.  Keeping a private default
        # here silently ignored command-line GraphSCM settings.
        self.graph_config = config or PriorConfig()
        self._bank: Optional[ProfileBank | FrozenBlueprintBank] = None
        self._rng: Optional[np.random.Generator] = None
        self._scheduler: Optional[RollingProfileScheduler] = None
        self._priority_profile_ids: Tuple[int, ...] = ()
        self._priority_profile_weights: Tuple[float, ...] = ()
        self._identity: Optional[Tuple[int, int]] = None
        self._worker_batch_count = 0
        self._feature_view_counts: Dict[str, int] = {}
        self.last_metadata: List[Dict[str, object]] = []

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_bank"] = None
        state["_rng"] = None
        state["_scheduler"] = None
        state["_priority_profile_ids"] = ()
        state["_priority_profile_weights"] = ()
        state["_identity"] = None
        state["_worker_batch_count"] = 0
        state["_feature_view_counts"] = {}
        state["last_metadata"] = []
        return state

    def _worker_identity(self) -> Tuple[int, int]:
        info = get_worker_info()
        worker_id = int(info.id) if info is not None else 0
        rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
        return rank, worker_id

    def _ensure_state(
        self,
    ) -> Tuple[
        ProfileBank | FrozenBlueprintBank,
        np.random.Generator,
        RollingProfileScheduler,
    ]:
        identity = self._worker_identity()
        if self._rng is None or self._identity != identity:
            rank, worker_id = identity
            seed = (self.base_seed + 1_000_003 * rank + 9_973 * worker_id) % (2**32)
            self._rng = np.random.default_rng(seed)
            if self.blueprint_mode:
                self._bank = FrozenBlueprintBank(
                    self.data_root,
                    self.compiled_manifest or {},
                    forbidden_root=self.forbidden_data_root or "",
                    max_cached_datasets=self.profile_cache_size,
                )
            else:
                self._bank = ProfileBank(
                    self.data_root,
                    include_val=self.include_val,
                    max_cached_datasets=self.profile_cache_size,
                    max_rows_per_dataset=self.profile_max_rows,
                    seed=seed,
                    evaluation_compatible=False,
                    forbidden_root=self.forbidden_data_root,
                    reject_symlinks=self.runtime_isolated,
                )
            if self.runtime_isolated:
                manifest = self.compiled_manifest or {}
                expected_count = int(manifest.get("profile_count", -1))
                expected_names = tuple(manifest.get("dataset_names", ()))
                actual_names = tuple(profile.name for profile in self._bank.profiles)
                if expected_count != len(self._bank) or expected_names != actual_names:
                    raise ValueError(
                        "isolated artifact bank does not match its manifest: "
                        f"manifest_count={expected_count}, actual_count={len(self._bank)}"
                    )
            self._scheduler = RollingProfileScheduler(len(self._bank), self._rng)
            if self.protected_priority_extras:
                name_to_id = {
                    profile.name: index
                    for index, profile in enumerate(self._bank.profiles)
                }
                missing = tuple(
                    name
                    for name in HYBRID178_PROTECTED_PROFILE_NAMES
                    if name not in name_to_id
                )
                if missing:
                    raise ValueError(
                        "protected-priority Hybrid-178 requires every protected "
                        f"profile; missing={missing}"
                    )
                self._priority_profile_ids = tuple(
                    name_to_id[name]
                    for name in HYBRID178_PROTECTED_PROFILE_NAMES
                )
                if self.protected_priority_hardness_weighted:
                    missing_scores = tuple(
                        name
                        for name in HYBRID178_PROTECTED_PROFILE_NAMES
                        if name not in HYBRID178_PROTECTED_HARDNESS_SCORES
                    )
                    if missing_scores:
                        raise ValueError(
                            "hardness-weighted protected priority is missing "
                            f"row-free scores for {missing_scores}"
                        )
                    scores = np.asarray(
                        [
                            HYBRID178_PROTECTED_HARDNESS_SCORES[name]
                            for name in HYBRID178_PROTECTED_PROFILE_NAMES
                        ],
                        dtype=np.float64,
                    )
                    weights = np.exp(
                        HYBRID178_PROTECTED_HARDNESS_TEMPERATURE
                        * (scores - scores.mean())
                    )
                    self._priority_profile_weights = tuple(weights.tolist())
                elif self.protected_priority_collapse_risk_weighted:
                    missing_scores = tuple(
                        name
                        for name in HYBRID178_PROTECTED_PROFILE_NAMES
                        if name not in HYBRID178_PROTECTED_COLLAPSE_RISK_WEIGHTS
                    )
                    if missing_scores:
                        raise ValueError(
                            "collapse-risk-weighted protected priority is missing "
                            f"row-free weights for {missing_scores}"
                        )
                    self._priority_profile_weights = tuple(
                        float(HYBRID178_PROTECTED_COLLAPSE_RISK_WEIGHTS[name])
                        for name in HYBRID178_PROTECTED_PROFILE_NAMES
                    )
                else:
                    self._priority_profile_weights = ()
            else:
                self._priority_profile_ids = ()
                self._priority_profile_weights = ()
            self._identity = identity
            self._worker_batch_count = 0
            self._feature_view_counts = {}
        return self._bank, self._rng, self._scheduler

    def _take_schedule(
        self,
        batch_size: int,
        scheduler: RollingProfileScheduler,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, List[str], Optional[int]]:
        rank, worker_id = self._worker_identity()
        world_size = max(
            1, int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
        )
        if world_size == 1:
            return (
                scheduler.take(batch_size),
                self._branch_schedule(batch_size, rng),
                None,
            )

        info = get_worker_info()
        num_workers = int(info.num_workers) if info is not None else 1
        global_batch_index = (
            self.schedule_start_batch
            + worker_id
            + self._worker_batch_count * num_workers
        )
        self._worker_batch_count += 1
        profile_ids, branches = coordinated_global_schedule(
            n_profiles=scheduler.n_profiles,
            branch_probs=self.branch_probs,
            base_seed=self.base_seed,
            global_batch_index=global_batch_index,
            world_size=world_size,
            rank=rank,
            local_batch_size=batch_size,
            priority_profile_ids=self._priority_profile_ids,
            priority_profile_weights=(
                self._priority_profile_weights or None
            ),
        )
        return profile_ids, branches, global_batch_index

    @contextmanager
    def _task_seed(self, seed: int):
        np_state = np.random.get_state()
        py_state = random.getstate()
        torch_state = torch.random.get_rng_state()
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        try:
            yield
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)
            torch.random.set_rng_state(torch_state)

    def _sample_seq_len(self) -> int:
        return TabICLv2ClassificationPrior.sample_seq_len(
            self.min_seq_len,
            self.max_seq_len,
            self.log_seq_len,
            self.replay_small,
        )

    def _train_size(self, seq_len: int) -> int:
        desired = int(round(seq_len * self.train_fraction))
        return int(np.clip(desired, 1, seq_len - 1))

    def _official_train_size(
        self, seq_len: int, rng: np.random.Generator
    ) -> int:
        """Sample the support boundary exactly like standalone GraphPrior."""

        if isinstance(self.min_train_size, int) and isinstance(
            self.max_train_size, int
        ):
            if self.min_train_size == self.max_train_size:
                value = int(self.min_train_size)
            else:
                value = int(
                    rng.integers(self.min_train_size, self.max_train_size)
                )
        elif isinstance(self.min_train_size, float) and isinstance(
            self.max_train_size, float
        ):
            if self.min_train_size == self.max_train_size:
                ratio = float(self.min_train_size)
            else:
                ratio = float(
                    rng.uniform(self.min_train_size, self.max_train_size)
                )
            value = int(seq_len * ratio)
        else:
            raise ValueError("invalid official train-size range")
        return int(np.clip(value, 1, seq_len - 1))

    def _profile_copula_scale(self, profile_name: str) -> float:
        """Return the task-specific target-copula blend without changing schedules."""

        if (
            self.protected_profile_copula_blend_scale is not None
            and profile_name in HYBRID178_PROTECTED_PROFILE_NAMES
        ):
            return float(self.protected_profile_copula_blend_scale)
        if (
            self.target_profile_copula_blend_scale is not None
            and profile_name in HYBRID178_TARGET_PROFILE_NAMES
        ):
            return float(self.target_profile_copula_blend_scale)
        return float(self.profile_copula_blend_scale)

    def _profile_shape_jitter(self, profile_name: str) -> float:
        """Return the group-specific neighborhood width for shape-only tasks."""

        if (
            self.protected_profile_shape_jitter_strength is not None
            and profile_name in HYBRID178_PROTECTED_PROFILE_NAMES
        ):
            return float(self.protected_profile_shape_jitter_strength)
        if (
            self.target_profile_shape_jitter_strength is not None
            and profile_name in HYBRID178_TARGET_PROFILE_NAMES
        ):
            return float(self.target_profile_shape_jitter_strength)
        return float(self.profile_shape_jitter_strength)

    def _profile_supervised_candidates(self, profile_name: str) -> int:
        """Return the group-specific official GraphSCM candidate count."""

        if (
            self.protected_profile_supervised_candidate_count is not None
            and profile_name in HYBRID178_PROTECTED_PROFILE_NAMES
        ):
            return int(self.protected_profile_supervised_candidate_count)
        if (
            self.target_profile_supervised_candidate_count is not None
            and profile_name in HYBRID178_TARGET_PROFILE_NAMES
        ):
            return int(self.target_profile_supervised_candidate_count)
        return int(self.profile_supervised_candidate_count)

    def _profile_transport_enabled(
        self,
        profile_index: int,
        cycle_index: int,
        profile_name: Optional[str] = None,
    ) -> bool:
        """Rotate full marginal transport without adding count noise.

        For the intended ratio of 1/4, every global batch transports exactly
        44 or 45 of the 178 profile slots and every profile is transported once
        in each four-batch cycle.  Ratios are represented by a bounded rational
        period so the same invariant generalizes deterministically.
        """

        _, numerator, denominator = self._profile_transport_spec(profile_name)
        if numerator <= 0:
            return False
        if numerator >= denominator:
            return True
        residue = (int(profile_index) + int(cycle_index)) % denominator
        return residue < numerator

    def _profile_transport_spec(
        self, profile_name: Optional[str]
    ) -> Tuple[float, int, int]:
        protected_ratio = getattr(self, "protected_profile_transport_ratio", None)
        if (
            profile_name is not None
            and protected_ratio is not None
            and profile_name in HYBRID178_PROTECTED_PROFILE_NAMES
        ):
            return (
                float(protected_ratio),
                int(self.protected_profile_transport_numerator),
                int(self.protected_profile_transport_denominator),
            )
        target_ratio = getattr(self, "target_profile_transport_ratio", None)
        if (
            profile_name is not None
            and target_ratio is not None
            and profile_name in HYBRID178_TARGET_PROFILE_NAMES
        ):
            return (
                float(target_ratio),
                int(self.target_profile_transport_numerator),
                int(self.target_profile_transport_denominator),
            )
        return (
            float(self.profile_transport_ratio),
            int(self.profile_transport_numerator),
            int(self.profile_transport_denominator),
        )

    def _profile_targets(
        self,
        profile: DatasetProfile,
        seq_len: int,
        train_size: int,
        rng: np.random.Generator,
    ) -> Tuple[int, int, np.ndarray]:
        max_k = min(
            self.max_classes,
            profile.n_classes,
            train_size,
            seq_len - train_size,
            seq_len // 2,
        )
        if max_k < 2:
            raise ValueError("sequence split cannot contain two classes in support and query")
        probs = np.asarray(profile.class_probs, dtype=np.float64)
        if probs.size > max_k:
            chosen = rng.choice(
                probs.size, size=max_k, replace=False, p=probs / probs.sum()
            )
            probs = probs[chosen]
        probs = probs / probs.sum()
        feature_jitter = int(rng.choice([-1, 0, 0, 0, 1]))
        d = int(
            np.clip(
                profile.n_features + feature_jitter,
                self.min_features,
                self.max_features,
            )
        )
        return d, max_k, probs

    @staticmethod
    def _select_columns(
        data: ProfileData,
        count: int,
        rng: np.random.Generator,
        *,
        preserve_order: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        d = data.x.shape[1]
        if d >= count:
            cols = (
                np.arange(count, dtype=np.int64)
                if preserve_order
                else rng.choice(d, size=count, replace=False)
            )
        else:
            cols = np.concatenate(
                [np.arange(d), rng.choice(d, size=count - d, replace=True)]
            )
        return data.x[:, cols], data.kinds[cols]

    @staticmethod
    def _select_real_classes(
        data: ProfileData,
        max_classes: int,
        train_size: int,
        query_size: int,
        rng: np.random.Generator,
        require_two_rows: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        threshold = 2 if require_two_rows else 1
        available = np.asarray(
            [indices.size >= threshold for indices in data.class_indices], dtype=bool
        )
        ids = np.flatnonzero(available)
        k = min(ids.size, max_classes, train_size, query_size)
        if k < 2:
            raise ValueError("template lacks two usable classes")
        probs = data.class_probs[ids].astype(np.float64)
        if ids.size > k:
            selected = rng.choice(
                ids.size, size=k, replace=False, p=probs / probs.sum()
            )
            ids = ids[selected]
            probs = probs[selected]
        return ids.astype(np.int64), probs / probs.sum()

    def _official_candidate(
        self, seq_len: int, d: int, k: int, seed: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        with self._task_seed(seed):
            x, y = GraphSCM(
                regression=False,
                seq_len=seq_len,
                num_features=d,
                max_features=d,
                num_classes=k,
                config=self.graph_config,
                device=self.device,
            )()
        return (
            x.detach().cpu().numpy().astype(np.float32),
            y.detach().cpu().numpy().astype(np.int64),
        )

    def _resample_official(
        self,
        candidate_x: np.ndarray,
        candidate_y: np.ndarray,
        probs: np.ndarray,
        train_size: int,
        query_size: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        count_fn = _closest_proportional_counts if self.high_fidelity else None
        support_counts = (
            count_fn(train_size, probs)
            if count_fn is not None
            else _split_counts(train_size, probs, rng)
        )
        query_counts = (
            count_fn(query_size, probs)
            if count_fn is not None
            else _split_counts(query_size, probs, rng)
        )
        permutation = rng.permutation(candidate_y.size)
        split = candidate_y.size // 2
        source_parts = (permutation[:split], permutation[split:])
        output_x: List[np.ndarray] = []
        output_y: List[np.ndarray] = []
        for part_idx, counts in enumerate((support_counts, query_counts)):
            part = source_parts[part_idx]
            for cls in range(probs.size):
                pool = part[candidate_y[part] == cls]
                if pool.size == 0:
                    pool = np.flatnonzero(candidate_y == cls)
                if pool.size == 0:
                    raise ValueError("official candidate omitted a requested class")
                chosen = rng.choice(
                    pool,
                    size=int(counts[cls]),
                    replace=pool.size < counts[cls],
                )
                output_x.append(candidate_x[chosen])
                output_y.append(
                    np.full(int(counts[cls]), cls, dtype=np.int64)
                )
        x = np.concatenate(output_x, axis=0)
        y = np.concatenate(output_y)
        support_order = rng.permutation(train_size)
        query_order = train_size + rng.permutation(query_size)
        order = np.concatenate([support_order, query_order])
        return x[order], y[order]

    def _generate_frozen_official(
        self,
        profile: DatasetProfile,
        seq_len: int,
        train_size: int,
        rng: np.random.Generator,
        seed: int,
        transport: bool,
        bank: FrozenBlueprintBank,
        blueprint: Optional[FrozenDatasetBlueprint] = None,
        official_num_features: Optional[int] = None,
        profile_shape_only: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
        if transport and profile_shape_only:
            raise ValueError(
                "profile_shape_only and full marginal transport are mutually exclusive"
            )
        if blueprint is None:
            full_blueprint = bank.load(profile)
            blueprint, _, _, _, _ = _frozen_feature_view(
                full_blueprint, self.max_features, 0, self.base_seed
            )
        profile_conditioned = bool(transport or profile_shape_only)
        if profile_conditioned:
            source_d = int(blueprint.n_features)
            source_k = int(blueprint.n_classes)
            d = source_d
            k = source_k
            requested_probs: Optional[np.ndarray] = blueprint.class_probs
            shape_jitter_strength = (
                self._profile_shape_jitter(profile.name)
                if profile_shape_only
                else 0.0
            )
            if shape_jitter_strength > 0.0:
                shape_rng = np.random.default_rng(
                    (
                        int(seed)
                        ^ _stable_u32(blueprint.name)
                        ^ 0x1785A9E
                    )
                    % (2**32)
                )
                feature_multiplier = float(
                    np.exp(shape_rng.normal(0.0, 0.40 * shape_jitter_strength))
                )
                d = int(
                    np.clip(
                        round(source_d * feature_multiplier),
                        self.min_features,
                        self.max_features,
                    )
                )
                max_k = min(
                    self.max_classes,
                    train_size,
                    seq_len - train_size,
                )
                if (
                    max_k >= 2
                    and shape_rng.random() < 0.5 * shape_jitter_strength
                ):
                    class_delta = int(shape_rng.choice((-1, 1)))
                    k = int(np.clip(source_k + class_delta, 2, max_k))
                requested_probs = _resize_class_probabilities(
                    blueprint.class_probs, k
                )
                log_probs = np.log(np.maximum(requested_probs, 1e-12))
                log_probs += shape_rng.normal(
                    0.0, 0.35 * shape_jitter_strength, size=k
                )
                requested_probs = np.exp(log_probs - np.max(log_probs))
                requested_probs /= requested_probs.sum()
        else:
            # Match the standalone official prior: feature and class counts are
            # sampled independently of the scheduled 178 profile.  The profile
            # remains attached only so the DDP-global scheduler can prove
            # coverage of all templates in the conditioned minority branches.
            d = int(
                official_num_features
                if official_num_features is not None
                else round(rng.uniform(self.min_features, self.max_features))
            )
            d = int(np.clip(d, self.min_features, self.max_features))
            max_k = min(
                self.max_classes,
                train_size,
                seq_len - train_size,
            )
            if max_k < 2:
                raise ValueError(
                    "official-shape split cannot contain two classes"
                )
            k = int(rng.integers(2, max_k + 1))
            requested_probs = None
        query_size = seq_len - train_size
        # Conditioned official-profile tasks need a class-complete pool for
        # marginal transport.  The official backbone instead preserves the
        # standalone GraphPrior sample without class rebalancing or duplicate
        # row sampling.
        candidate_len = seq_len + 8 * k if profile_conditioned else seq_len
        last_error: Optional[Exception] = None
        for attempt in range(8):
            try:
                candidate_x, candidate_y = self._official_candidate(
                    candidate_len, d, k, seed + attempt
                )
                if not profile_conditioned:
                    if not (
                        np.isfinite(candidate_x).all()
                        and np.isfinite(candidate_y).all()
                    ):
                        raise ValueError("official candidate is non-finite")
                    # GraphPrior accepts the sampled task when support and query
                    # contain the same observed labels.  If the initial IID
                    # ordering misses that condition, it tries random row
                    # permutations before regenerating the graph.
                    accepted = False
                    for split_attempt in range(11):
                        support_labels = np.unique(candidate_y[:train_size])
                        query_labels = np.unique(candidate_y[train_size:])
                        if (
                            support_labels.size >= 2
                            and np.array_equal(support_labels, query_labels)
                        ):
                            accepted = True
                            break
                        if split_attempt < 10:
                            order = rng.permutation(candidate_y.size)
                            candidate_x = candidate_x[order]
                            candidate_y = candidate_y[order]
                    if not accepted:
                        raise ValueError(
                            "official candidate has an invalid support/query split"
                        )
                    observed_labels = np.unique(candidate_y)
                    return candidate_x, candidate_y, {
                        "official_graph_scm": True,
                        "official_graph_prior_exact_backbone": True,
                        "official_candidate_len": candidate_len,
                        "frozen_gt_marginal_transport": False,
                        "transport_scope": "none",
                        "source_rows_sampled": False,
                        "row_bootstrap": False,
                        "profile_conditioned": False,
                        "profile_coverage_only": True,
                        "all_template_features_preserved": False,
                        "source_class_count": blueprint.source_class_count,
                        "synthetic_class_count": int(observed_labels.size),
                        "official_shape_sampled_num_features": d,
                        "official_shape_requested_num_classes": k,
                        "official_shape_observed_num_classes": int(
                            observed_labels.size
                        ),
                        "row_free_blueprint": True,
                    }
                if requested_probs is None:
                    observed_labels, observed = np.unique(
                        candidate_y, return_counts=True
                    )
                    if observed_labels.size < 2:
                        raise ValueError(
                            "official candidate produced fewer than two classes"
                        )
                    candidate_y = np.searchsorted(
                        observed_labels, candidate_y
                    ).astype(np.int64)
                    synthetic_k = int(observed_labels.size)
                    observed = observed.astype(np.float64)
                    probs = observed / observed.sum()
                else:
                    synthetic_k = k
                    probs = requested_probs
                x, y = self._resample_official(
                    candidate_x,
                    candidate_y,
                    probs,
                    train_size,
                    query_size,
                    rng,
                )
                if transport:
                    # Keep correlation-alignment draws task-local.  Consuming the
                    # worker scheduler RNG here would change subsequent branch
                    # candidates and make an unrelated Copula task depend on
                    # whether an earlier profile needed stronger shrinkage.
                    transport_rng = np.random.default_rng(
                        (
                            int(seed)
                            ^ _stable_u32(blueprint.name)
                            ^ 0x178C0A1A
                        )
                        % (2**32)
                    )
                    x, copula_shrinkage_details = _transport_to_frozen_blueprint(
                        x,
                        y,
                        blueprint,
                        transport_rng,
                        target_copula_blend_scale=(
                            self._profile_copula_scale(profile.name)
                        ),
                    )
                else:
                    copula_shrinkage_details = {}
                return x, y, {
                    "official_graph_scm": True,
                    "official_candidate_len": candidate_len,
                    "frozen_gt_marginal_transport": bool(transport),
                    "transport_scope": (
                        "class_conditional_frozen_blueprint"
                        if transport
                        else "profile_shape_class_balance_only"
                        if profile_shape_only
                        else "none"
                    ),
                    "source_rows_sampled": False,
                    "row_bootstrap": False,
                    "profile_conditioned": profile_conditioned,
                    "profile_shape_conditioned": bool(profile_shape_only),
                    "profile_shape_jitter_strength": float(
                        shape_jitter_strength if profile_conditioned else 0.0
                    ),
                    "profile_shape_source_num_features": int(
                        source_d if profile_conditioned else d
                    ),
                    "profile_shape_source_num_classes": int(
                        source_k if profile_conditioned else k
                    ),
                    "profile_shape_feature_delta": int(
                        d - source_d if profile_conditioned else 0
                    ),
                    "profile_shape_class_delta": int(
                        synthetic_k - source_k if profile_conditioned else 0
                    ),
                    "profile_shape_requested_class_probs": (
                        np.asarray(probs, dtype=np.float64).tolist()
                        if profile_conditioned
                        else []
                    ),
                    "profile_full_marginal_transport": bool(transport),
                    "profile_coverage_only": not profile_conditioned,
                    "all_template_features_preserved": (
                        profile_conditioned and d == profile.n_features
                    ),
                    "source_class_count": blueprint.source_class_count,
                    "synthetic_class_count": synthetic_k,
                    "official_shape_sampled_num_features": d,
                    "official_shape_requested_num_classes": k,
                    "official_shape_observed_num_classes": synthetic_k,
                    "row_free_blueprint": True,
                    **copula_shrinkage_details,
                }
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                last_error = exc
        raise RuntimeError(f"frozen official GraphSCM failed after retries: {last_error}")

    def _generate_frozen_copula(
        self,
        profile: DatasetProfile,
        seq_len: int,
        train_size: int,
        rng: np.random.Generator,
        bank: FrozenBlueprintBank,
        blueprint: Optional[FrozenDatasetBlueprint] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
        if blueprint is None:
            full_blueprint = bank.load(profile)
            blueprint, _, _, _, _ = _frozen_feature_view(
                full_blueprint, self.max_features, 0, self.base_seed
            )
        support_counts = _closest_proportional_counts(
            train_size, blueprint.class_probs
        )
        query_counts = _closest_proportional_counts(
            seq_len - train_size, blueprint.class_probs
        )
        support_x: List[np.ndarray] = []
        query_x: List[np.ndarray] = []
        support_y: List[np.ndarray] = []
        query_y: List[np.ndarray] = []
        for local_class in range(blueprint.n_classes):
            support_count = int(support_counts[local_class])
            query_count = int(query_counts[local_class])
            total_count = support_count + query_count
            z = _sample_frozen_copula(
                blueprint.copula_modes[local_class],
                blueprint.copula_factors[local_class],
                blueprint.copula_residuals[local_class],
                total_count,
                rng,
            )
            uniforms = _rank_stratified_uniforms(z)
            sampled = _inverse_frozen_quantiles(
                uniforms,
                blueprint.quantile_grid,
                blueprint.class_quantiles[local_class],
                blueprint.column_types,
                blueprint.class_zero_rates[local_class],
                blueprint.class_zero_cdf_lower[local_class],
            )
            support_x.append(sampled[:support_count])
            query_x.append(sampled[support_count:])
            support_y.append(
                np.full(support_count, local_class, dtype=np.int64)
            )
            query_y.append(np.full(query_count, local_class, dtype=np.int64))
        sx, qx = np.concatenate(support_x), np.concatenate(query_x)
        sy, qy = np.concatenate(support_y), np.concatenate(query_y)
        support_order = rng.permutation(train_size)
        query_order = rng.permutation(seq_len - train_size)
        x = np.concatenate([sx[support_order], qx[query_order]])
        y = np.concatenate([sy[support_order], qy[query_order]])
        return x, y, {
            "class_conditional": True,
            "rank_gaussian_copula": True,
            "frozen_copula_factors": True,
            "copula_covariance_modes": list(blueprint.copula_modes),
            "low_rank_wide_copula": "low_rank" in blueprint.copula_modes,
            "max_copula_rank": max(
                (factor.shape[1] for factor in blueprint.copula_factors),
                default=0,
            ),
            "source_rows_sampled": False,
            "row_bootstrap": False,
            "all_template_features_preserved": (
                blueprint.n_features == profile.n_features
            ),
            "source_class_count": blueprint.source_class_count,
            "synthetic_class_count": blueprint.n_classes,
            "row_free_blueprint": True,
        }

    def _generate_official(
        self,
        profile: DatasetProfile,
        seq_len: int,
        train_size: int,
        rng: np.random.Generator,
        seed: int,
        transport: bool,
        bank: ProfileBank,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
        data = bank.load(profile)
        if transport:
            class_groups, probs = _merge_class_groups(data, self.max_classes)
            k = len(class_groups)
            d = int(
                np.clip(profile.n_features, self.min_features, self.max_features)
            )
            template_x, kinds = self._select_columns(
                data, d, rng, preserve_order=True
            )
        else:
            d = int(rng.integers(self.min_features, self.max_features + 1))
            max_k = min(
                self.max_classes,
                train_size,
                seq_len - train_size,
            )
            if max_k < 2:
                raise ValueError(
                    "official-shape split cannot contain two classes"
                )
            k = int(rng.integers(2, max_k + 1))
            class_groups = tuple(
                np.asarray([local_class], dtype=np.int64)
                for local_class in range(k)
            )
            probs = np.full(k, 1.0 / k, dtype=np.float64)
            template_x = np.empty((0, d), dtype=np.float32)
            kinds = np.full(d, "numeric", dtype=object)
        query_size = seq_len - train_size
        candidate_len = max(seq_len * 2, seq_len + 8 * k)
        last_error: Optional[Exception] = None
        for attempt in range(8):
            try:
                candidate_x, candidate_y = self._official_candidate(
                    candidate_len, d, k, seed + attempt
                )
                if not transport:
                    observed_labels, observed = np.unique(
                        candidate_y, return_counts=True
                    )
                    if observed_labels.size < 2:
                        raise ValueError(
                            "official candidate produced fewer than two classes"
                        )
                    candidate_y = np.searchsorted(
                        observed_labels, candidate_y
                    ).astype(np.int64)
                    synthetic_k = int(observed_labels.size)
                    observed = observed.astype(np.float64)
                    probs = observed / observed.sum()
                else:
                    synthetic_k = k
                x, y = self._resample_official(
                    candidate_x,
                    candidate_y,
                    probs,
                    train_size,
                    query_size,
                    rng,
                )
                if transport:
                    x = _class_conditional_transport(
                        x,
                        y,
                        template_x,
                        data.y,
                        class_groups,
                        kinds,
                        rng,
                        continuous_noise=(0.002 if self.high_fidelity else 0.01),
                    )
                return x, y, {
                    "official_graph_scm": True,
                    "empirical_marginal_transport": bool(transport),
                    "transport_scope": (
                        "class_conditional" if transport else "none"
                    ),
                    "source_rows_sampled": False,
                    "row_bootstrap": False,
                    "profile_conditioned": bool(transport),
                    "profile_coverage_only": not transport,
                    "all_template_features_preserved": (
                        bool(transport) and d == profile.n_features
                    ),
                    "source_class_count": len(data.class_indices),
                    "synthetic_class_count": synthetic_k,
                    "source_class_groups": [
                        group.tolist() for group in class_groups
                    ],
                    "official_shape_sampled_num_features": d,
                    "official_shape_requested_num_classes": k,
                    "official_shape_observed_num_classes": synthetic_k,
                }
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                last_error = exc
        raise RuntimeError(f"official GraphSCM failed after retries: {last_error}")

    def _generate_copula(
        self,
        profile: DatasetProfile,
        seq_len: int,
        train_size: int,
        rng: np.random.Generator,
        bank: ProfileBank,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
        data = bank.load(profile)
        class_groups, probs = _merge_class_groups(data, self.max_classes)
        d = int(np.clip(profile.n_features, self.min_features, self.max_features))
        template_x, kinds = self._select_columns(
            data, d, rng, preserve_order=True
        )
        column_types = _column_types(template_x, kinds)
        support_counts = (
            _closest_proportional_counts(train_size, probs)
            if self.high_fidelity
            else _split_counts(train_size, probs, rng)
        )
        query_counts = (
            _closest_proportional_counts(seq_len - train_size, probs)
            if self.high_fidelity
            else _split_counts(seq_len - train_size, probs, rng)
        )
        support_x: List[np.ndarray] = []
        query_x: List[np.ndarray] = []
        support_y: List[np.ndarray] = []
        query_y: List[np.ndarray] = []
        copula_modes: List[str] = []
        copula_ranks: List[int] = []
        for local_cls, source_classes in enumerate(class_groups):
            rows = template_x[np.isin(data.y, source_classes)]
            if rows.shape[0] > 4096:
                fit_indices = np.linspace(
                    0, rows.shape[0] - 1, num=4096, dtype=np.int64
                )
                fit_rows = rows[fit_indices]
            else:
                fit_rows = rows
            _, width = fit_rows.shape
            z_emp = _normal_scores(fit_rows)
            support_count = int(support_counts[local_cls])
            query_count = int(query_counts[local_cls])
            total_count = support_count + query_count
            z, copula_mode, copula_rank = _sample_rank_gaussian(
                z_emp, total_count, rng
            )
            copula_modes.append(copula_mode)
            copula_ranks.append(copula_rank)
            uniforms = (
                _rank_stratified_uniforms(z)
                if self.high_fidelity
                else norm.cdf(z)
            )
            sampled = np.empty((total_count, width), dtype=np.float32)
            for col, kind in enumerate(column_types):
                sampled[:, col] = _inverse_empirical_column(
                    uniforms[:, col],
                    rows[:, col],
                    str(kind),
                    rng,
                    continuous_noise=(0.002 if self.high_fidelity else 0.01),
                )
            support_x.append(sampled[:support_count])
            query_x.append(sampled[support_count:])
            support_y.append(
                np.full(support_count, local_cls, dtype=np.int64)
            )
            query_y.append(np.full(query_count, local_cls, dtype=np.int64))

        sx, qx = np.concatenate(support_x), np.concatenate(query_x)
        sy, qy = np.concatenate(support_y), np.concatenate(query_y)
        support_order = rng.permutation(train_size)
        query_order = rng.permutation(seq_len - train_size)
        x = np.concatenate([sx[support_order], qx[query_order]])
        y = np.concatenate([sy[support_order], qy[query_order]])
        return x, y, {
            "class_conditional": True,
            "rank_gaussian_copula": True,
            "covariance_shrinkage": True,
            "copula_covariance_modes": copula_modes,
            "low_rank_wide_copula": "low_rank" in copula_modes,
            "max_copula_rank": max(copula_ranks, default=0),
            "source_rows_sampled": False,
            "row_bootstrap": False,
            "all_template_features_preserved": d == profile.n_features,
            "source_class_count": len(data.class_indices),
            "synthetic_class_count": len(class_groups),
            "source_class_groups": [group.tolist() for group in class_groups],
        }

    def _smooth_values(
        self,
        x: np.ndarray,
        source_rows: np.ndarray,
        column_types: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        out = x.astype(np.float32, copy=True)
        for col, kind in enumerate(column_types):
            empirical = source_rows[:, col]
            if kind in {"binary", "low_card"}:
                probability = float(rng.uniform(0.02, 0.08))
                mask = rng.random(out.shape[0]) < probability
                if mask.any():
                    out[mask, col] = rng.choice(
                        empirical, size=int(mask.sum()), replace=True
                    )
            elif kind == "integer":
                q25, q75 = np.quantile(empirical, [0.25, 0.75])
                lam = max(
                    float((q75 - q25) * rng.uniform(0.01, 0.04)),
                    0.10,
                )
                delta = (
                    rng.poisson(lam, out.shape[0])
                    - rng.poisson(lam, out.shape[0])
                )
                out[:, col] = np.round(out[:, col] + delta)
                out[:, col] = np.clip(
                    out[:, col], np.min(empirical), np.max(empirical)
                )
            elif kind == "sparse":
                active = ~np.isclose(out[:, col], 0.0, atol=1e-8)
                deactivate = active & (
                    rng.random(out.shape[0]) < rng.uniform(0.01, 0.05)
                )
                activate = (~active) & (
                    rng.random(out.shape[0]) < rng.uniform(0.005, 0.03)
                )
                out[deactivate, col] = 0.0
                nonzero = empirical[
                    ~np.isclose(empirical, 0.0, atol=1e-8)
                ]
                if activate.any() and nonzero.size:
                    out[activate, col] = rng.choice(
                        nonzero, size=int(activate.sum()), replace=True
                    )
            else:
                median = float(np.median(empirical))
                mad = (
                    float(np.median(np.abs(empirical - median)))
                    * 1.4826
                )
                q25, q75 = np.quantile(empirical, [0.25, 0.75])
                scale = max(
                    mad,
                    float((q75 - q25) / 1.349),
                    float(np.std(empirical)),
                    1e-6,
                )
                noise = rng.normal(
                    0.0,
                    rng.uniform(0.02, 0.08) * scale,
                    out.shape[0],
                )
                low, high = np.quantile(empirical, [0.001, 0.999])
                out[:, col] = np.clip(out[:, col] + noise, low, high)
        return out

    def _generate_smooth(
        self,
        profile: DatasetProfile,
        seq_len: int,
        train_size: int,
        rng: np.random.Generator,
        bank: ProfileBank,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
        data = bank.load(profile)
        class_ids, probs = self._select_real_classes(
            data,
            self.max_classes,
            train_size,
            seq_len - train_size,
            rng,
            require_two_rows=True,
        )
        d = int(
            np.clip(
                profile.n_features + int(rng.choice([-1, 0, 0, 1])),
                self.min_features,
                self.max_features,
            )
        )
        template_x, kinds = self._select_columns(data, d, rng)
        types = _column_types(template_x, kinds)
        support_counts = _split_counts(train_size, probs, rng)
        query_counts = _split_counts(seq_len - train_size, probs, rng)
        support_x: List[np.ndarray] = []
        query_x: List[np.ndarray] = []
        support_y: List[np.ndarray] = []
        query_y: List[np.ndarray] = []
        support_sources: List[np.ndarray] = []
        query_sources: List[np.ndarray] = []
        for local_cls, source_cls in enumerate(class_ids):
            pool = np.flatnonzero(data.y == int(source_cls))
            pool = rng.permutation(pool)
            cut = int(
                np.clip(
                    round(pool.size * self.train_fraction),
                    1,
                    pool.size - 1,
                )
            )
            support_pool, query_pool = pool[:cut], pool[cut:]
            support_idx = rng.choice(
                support_pool,
                size=int(support_counts[local_cls]),
                replace=support_pool.size < support_counts[local_cls],
            )
            query_idx = rng.choice(
                query_pool,
                size=int(query_counts[local_cls]),
                replace=query_pool.size < query_counts[local_cls],
            )
            support_x.append(
                self._smooth_values(
                    template_x[support_idx], template_x[pool], types, rng
                )
            )
            query_x.append(
                self._smooth_values(
                    template_x[query_idx], template_x[pool], types, rng
                )
            )
            support_y.append(
                np.full(
                    int(support_counts[local_cls]),
                    local_cls,
                    dtype=np.int64,
                )
            )
            query_y.append(
                np.full(
                    int(query_counts[local_cls]),
                    local_cls,
                    dtype=np.int64,
                )
            )
            support_sources.append(support_idx)
            query_sources.append(query_idx)

        sx, qx = np.concatenate(support_x), np.concatenate(query_x)
        sy, qy = np.concatenate(support_y), np.concatenate(query_y)
        support_rows_ordered = np.concatenate(support_sources)
        query_rows_ordered = np.concatenate(query_sources)
        support_order = rng.permutation(train_size)
        query_order = rng.permutation(seq_len - train_size)
        sx, sy = sx[support_order], sy[support_order]
        qx, qy = qx[query_order], qy[query_order]
        support_rows_ordered = support_rows_ordered[support_order]
        query_rows_ordered = query_rows_ordered[query_order]
        support_unique = np.unique(np.concatenate(support_sources))
        query_unique = np.unique(np.concatenate(query_sources))
        if np.intersect1d(support_unique, query_unique).size:
            raise AssertionError(
                "smooth bootstrap support/query source pools overlap"
            )
        return np.concatenate([sx, qx]), np.concatenate([sy, qy]), {
            "class_conditional": True,
            "smoothed_bootstrap": True,
            "support_query_source_disjoint": True,
            "support_source_rows": support_unique.tolist(),
            "query_source_rows": query_unique.tolist(),
            "source_rows_ordered": np.concatenate(
                [support_rows_ordered, query_rows_ordered]
            ).tolist(),
        }

    def _generate_exact(
        self,
        profile: DatasetProfile,
        seq_len: int,
        train_size: int,
        rng: np.random.Generator,
        bank: ProfileBank,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
        """Replay real support-side joint rows without value perturbation."""

        data = bank.load(profile)
        d = int(data.x.shape[1])
        k = len(data.class_indices)
        if d != profile.n_features:
            raise ValueError(
                f"exact replay schema changed for {profile.name}: {profile.n_features} -> {d}"
            )
        if d > self.max_features:
            raise ValueError(
                f"exact replay needs max_features >= {d} for {profile.name}"
            )
        if k > self.max_classes:
            raise ValueError(
                f"exact replay needs max_classes >= {k} for {profile.name}"
            )
        support_counts = _closest_proportional_counts(train_size, data.class_probs)
        query_counts = _closest_proportional_counts(
            seq_len - train_size, data.class_probs
        )
        support_x: List[np.ndarray] = []
        query_x: List[np.ndarray] = []
        support_y: List[np.ndarray] = []
        query_y: List[np.ndarray] = []
        support_sources: List[np.ndarray] = []
        query_sources: List[np.ndarray] = []
        allowed_overlap: List[int] = []
        for cls, pool in enumerate(data.class_indices):
            shuffled = rng.permutation(pool)
            if shuffled.size == 1:
                support_pool = query_pool = shuffled
                allowed_overlap.append(int(shuffled[0]))
            else:
                cut = int(
                    np.clip(
                        round(shuffled.size * self.train_fraction),
                        1,
                        shuffled.size - 1,
                    )
                )
                support_pool, query_pool = shuffled[:cut], shuffled[cut:]
            support_idx = rng.choice(
                support_pool,
                size=int(support_counts[cls]),
                replace=support_pool.size < support_counts[cls],
            )
            query_idx = rng.choice(
                query_pool,
                size=int(query_counts[cls]),
                replace=query_pool.size < query_counts[cls],
            )
            support_x.append(data.x[support_idx])
            query_x.append(data.x[query_idx])
            support_y.append(
                np.full(int(support_counts[cls]), cls, dtype=np.int64)
            )
            query_y.append(
                np.full(int(query_counts[cls]), cls, dtype=np.int64)
            )
            support_sources.append(support_idx)
            query_sources.append(query_idx)

        sx, qx = np.concatenate(support_x), np.concatenate(query_x)
        sy, qy = np.concatenate(support_y), np.concatenate(query_y)
        support_rows_ordered = np.concatenate(support_sources)
        query_rows_ordered = np.concatenate(query_sources)
        support_order = rng.permutation(train_size)
        query_order = rng.permutation(seq_len - train_size)
        sx, sy = sx[support_order], sy[support_order]
        qx, qy = qx[query_order], qy[query_order]
        support_rows_ordered = support_rows_ordered[support_order]
        query_rows_ordered = query_rows_ordered[query_order]
        support_unique = np.unique(np.concatenate(support_sources))
        query_unique = np.unique(np.concatenate(query_sources))
        overlap = np.intersect1d(support_unique, query_unique)
        unexpected_overlap = np.setdiff1d(
            overlap, np.asarray(allowed_overlap, dtype=np.int64)
        )
        if unexpected_overlap.size:
            raise AssertionError("exact replay support/query source pools overlap")
        return np.concatenate([sx, qx]), np.concatenate([sy, qy]), {
            "exact_empirical_replay": True,
            "raw_values_perturbed": False,
            "all_template_features_preserved": True,
            "all_template_classes_preserved": True,
            "feature_order_preserved": True,
            "support_query_source_disjoint": overlap.size == 0,
            "rare_singleton_source_overlap": overlap.tolist(),
            "support_source_rows": support_unique.tolist(),
            "query_source_rows": query_unique.tolist(),
            "source_rows_ordered": np.concatenate(
                [support_rows_ordered, query_rows_ordered]
            ).tolist(),
        }

    def _finalize(
        self,
        x: np.ndarray,
        y: np.ndarray,
        train_size: int,
        rng: np.random.Generator,
        preserve_schema: bool = False,
        pad_to_max_features: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        x = np.nan_to_num(
            x, nan=0.0, posinf=1e9, neginf=-1e9
        ).astype(np.float32, copy=False)
        if not preserve_schema:
            keep = np.asarray(
                [np.unique(x[:, col]).size > 1 for col in range(x.shape[1])]
            )
            if not keep.any():
                raise ValueError("generated task has no non-constant features")
            x = x[:, keep]
        support = x[:train_size]
        mean = support.mean(axis=0, dtype=np.float64)
        std = support.std(axis=0, dtype=np.float64)
        std = np.where(std > 1e-6, std, 1.0)
        x = ((x - mean) / std).astype(np.float32)
        if not preserve_schema:
            x = np.clip(x, -100.0, 100.0)
        if not preserve_schema:
            x = x[:, rng.permutation(x.shape[1])]

        classes = np.unique(y)
        if classes.size < 2:
            raise ValueError("generated task has fewer than two classes")
        for part in (y[:train_size], y[train_size:]):
            if np.unique(part).size != classes.size:
                raise ValueError("not every class occurs in support and query")
        y = y.astype(np.int64)
        if not preserve_schema:
            y = rng.permutation(classes.size)[y]

        d = int(x.shape[1])
        if d < self.max_features and pad_to_max_features:
            x = np.pad(x, ((0, 0), (0, self.max_features - d)))
        elif d > self.max_features:
            x = x[:, : self.max_features]
            d = self.max_features
        return (
            torch.tensor(x, dtype=torch.float32, device=self.device),
            torch.tensor(y, dtype=torch.float32, device=self.device),
            torch.tensor(d, dtype=torch.long, device=self.device),
        )

    def _finalize_official_backbone(
        self,
        x: np.ndarray,
        y: np.ndarray,
        requested_features: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Apply only GraphPrior's constant-column packing to GraphSCM output."""

        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        requested_features = int(
            np.clip(requested_features, 1, min(x.shape[1], self.max_features))
        )
        active = x[:, :requested_features]
        keep = np.asarray(
            [np.unique(active[:, col]).size > 1 for col in range(active.shape[1])]
        )
        if not keep.any():
            raise ValueError("official GraphPrior task has no non-constant features")
        active = active[:, keep]
        d = int(active.shape[1])
        if d < self.max_features:
            active = np.pad(active, ((0, 0), (0, self.max_features - d)))
        return (
            torch.tensor(active, dtype=torch.float32, device=self.device),
            torch.tensor(y, dtype=torch.float32, device=self.device),
            torch.tensor(d, dtype=torch.long, device=self.device),
        )

    def _branch_schedule(
        self, batch_size: int, rng: np.random.Generator
    ) -> List[str]:
        expected = self.branch_probs * batch_size
        counts = np.floor(expected).astype(np.int64)
        remainder = batch_size - int(counts.sum())
        if remainder:
            order = np.argsort(-(expected - counts), kind="stable")
            counts[order[:remainder]] += 1
        branches = [
            name
            for name, count in zip(BRANCH_NAMES, counts)
            for _ in range(int(count))
        ]
        rng.shuffle(branches)
        return branches

    def _generate_one(
        self,
        branch: str,
        profile: DatasetProfile,
        seq_len: int,
        train_size: int,
        seed: int,
        bank: ProfileBank | FrozenBlueprintBank,
        rng: np.random.Generator,
        feature_view_index: int = 0,
        official_num_features: Optional[int] = None,
        profile_transport: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor, Dict[str, object]]:
        with self._task_seed(seed):
            if self.blueprint_mode:
                if not isinstance(bank, FrozenBlueprintBank):
                    raise TypeError("blueprint mode requires FrozenBlueprintBank")
                full_blueprint = bank.load(profile)
                (
                    blueprint,
                    feature_columns,
                    resolved_view,
                    feature_view_count,
                    precompiled_view_copula,
                ) = _frozen_feature_view(
                    full_blueprint,
                    self.max_features,
                    feature_view_index,
                    self.base_seed,
                )
                feature_details: Dict[str, object] = {
                    "source_feature_count": full_blueprint.n_features,
                    "synthetic_feature_count": blueprint.n_features,
                    "feature_view_index": resolved_view,
                    "feature_view_cycle_index": int(feature_view_index),
                    "feature_view_count": feature_view_count,
                    "feature_view_columns": feature_columns.tolist(),
                    "feature_view_union_covers_source_schema": True,
                    "feature_view_full_coverage_after_tasks": feature_view_count,
                    "feature_view_precompiled_class_copula": precompiled_view_copula,
                    "all_template_features_preserved": (
                        blueprint.n_features == full_blueprint.n_features
                    ),
                }
                atom_data = None
                atom_class_groups = None
                target_probs = blueprint.class_probs
            else:
                if not isinstance(bank, ProfileBank):
                    raise TypeError("profile mode requires ProfileBank")
                blueprint = None
                atom_data = bank.load(profile)
                atom_class_groups, target_probs = _merge_class_groups(
                    atom_data, self.max_classes
                )
                feature_details = {}

            def assess_quality(
                candidate_x: np.ndarray,
                candidate_y: np.ndarray,
                candidate_branch: str,
                attempt_count: int,
            ) -> Tuple[bool, Dict[str, object]]:
                structure = _task_structure_metrics(candidate_x, candidate_y)
                intrinsic_metrics: Dict[str, object] = {
                    "quality_feature_label_nmi": float(
                        structure["feature_label_nmi_mean"]
                    ),
                    "quality_feature_label_nmi_max": float(
                        structure["feature_label_nmi_max"]
                    ),
                    "quality_task_difficulty_proxy": float(
                        structure["task_difficulty_proxy"]
                    ),
                    "quality_higher_order_triangle": float(
                        np.asarray(structure["higher_order_signature"])[0]
                    ),
                    "quality_higher_order_spectral_concentration": float(
                        np.asarray(structure["higher_order_signature"])[1]
                    ),
                    "quality_sample_unique_fraction": float(
                        structure["sample_unique_fraction"]
                    ),
                }
                if candidate_branch == "official_shape":
                    # The dominant official branch is intentionally independent
                    # of the scheduled profile.  Only intrinsic validity and
                    # non-duplication are enforced here; comparing it with the
                    # profile would silently turn failed official tasks back
                    # into Copula tasks and recreate the old 90% bias.
                    #
                    # A fixed unique-row fraction is not valid for low-cardinality
                    # official tasks.  A valid binary/categorical SCM can have
                    # only two joint feature states even when the task has 1024
                    # rows.  The diversity metric itself is computed on at most
                    # 256 rows, so express the hard floor as two distinct states
                    # in that same sample.  Rich tasks still report their full
                    # unique fraction for monitoring; the hard gate only rejects
                    # a completely constant feature matrix.
                    quality_sample_size = min(candidate_x.shape[0], 256)
                    unique_min_rows = min(2, quality_sample_size)
                    unique_fraction_min = unique_min_rows / float(
                        max(quality_sample_size, 1)
                    )
                    unique_rows = int(
                        round(
                            intrinsic_metrics["quality_sample_unique_fraction"]
                            * quality_sample_size
                        )
                    )
                    passed = bool(
                        np.isfinite(candidate_x).all()
                        and np.isfinite(
                            list(intrinsic_metrics.values())
                        ).all()
                        and np.unique(candidate_y).size >= 2
                        and intrinsic_metrics["quality_sample_unique_fraction"]
                        >= unique_fraction_min
                    )
                    return passed, {
                        **intrinsic_metrics,
                        "quality_gate_passed": passed,
                        "quality_gate_attempts": attempt_count,
                        "quality_gate_reference": "official_graph_scm_intrinsic",
                        "quality_profile_matching_applied": False,
                        "quality_sample_unique_rows": unique_rows,
                        "quality_sample_unique_min_rows": unique_min_rows,
                        "quality_sample_unique_fraction_min": unique_fraction_min,
                    }
                if candidate_branch == "official_profile" and not profile_transport:
                    expected_features = int(
                        details["official_shape_sampled_num_features"]
                    )
                    expected_classes = int(details["synthetic_class_count"])
                    expected_probs = np.asarray(
                        details["profile_shape_requested_class_probs"],
                        dtype=np.float64,
                    )
                    expected_probs = np.clip(expected_probs, 0.0, None)
                    expected_probs /= max(float(expected_probs.sum()), 1e-12)
                    observed_counts = np.bincount(
                        np.asarray(candidate_y, dtype=np.int64),
                        minlength=expected_classes,
                    )[:expected_classes].astype(np.float64)
                    observed_probs = observed_counts / max(
                        float(observed_counts.sum()), 1.0
                    )
                    class_tv = 0.5 * float(
                        np.abs(observed_probs - expected_probs).sum()
                    )
                    quality_sample_size = min(candidate_x.shape[0], 256)
                    unique_fraction_min = min(2, quality_sample_size) / float(
                        max(quality_sample_size, 1)
                    )
                    class_tv_limit = max(
                        0.03, 1.25 * expected_probs.size / float(seq_len)
                    )
                    supervised_metrics: Dict[str, float] = {}
                    if (
                        expected_features == blueprint.n_features
                        and expected_classes == blueprint.n_classes
                    ):
                        supervised_metrics = _blueprint_supervised_quality_metrics(
                            candidate_x, candidate_y, blueprint
                        )
                    passed = bool(
                        np.isfinite(candidate_x).all()
                        and np.isfinite(candidate_y).all()
                        and candidate_x.shape[1] == expected_features
                        and np.unique(candidate_y).size == expected_classes
                        and class_tv <= class_tv_limit
                        and intrinsic_metrics["quality_sample_unique_fraction"]
                        >= unique_fraction_min
                    )
                    return passed, {
                        **intrinsic_metrics,
                        **supervised_metrics,
                        "quality_class_tv": class_tv,
                        "quality_gate_passed": passed,
                        "quality_gate_attempts": attempt_count,
                        "quality_gate_reference": (
                            "requested_profile_neighborhood_shape_class_balance_"
                            "with_official_features"
                        ),
                        "quality_profile_matching_applied": True,
                        "quality_profile_matching_scope": (
                            "feature_count_class_count_and_class_balance"
                        ),
                        "quality_full_marginal_transport_applied": False,
                        "quality_class_tv_limit": class_tv_limit,
                        "quality_sample_unique_fraction_min": unique_fraction_min,
                    }
                metrics = (
                    _blueprint_quality_metrics(candidate_x, candidate_y, blueprint)
                    if blueprint is not None
                    else _task_quality_metrics(
                        candidate_x, candidate_y, atom_data, target_probs
                    )
                )
                if blueprint is not None:
                    metrics.update(
                        _blueprint_supervised_quality_metrics(
                            candidate_x, candidate_y, blueprint
                        )
                    )
                else:
                    metrics.update(intrinsic_metrics)
                # Requiring every class in both support and query imposes a
                # finite-sample TV floor, especially for 10 classes at short
                # audit lengths. At the production length (1024) the fixed
                # 0.03 threshold remains the active constraint.
                class_tv_limit = max(
                    0.03, 1.25 * target_probs.size / float(seq_len)
                )
                correlation_limit = (
                    0.32
                    if self.blueprint_mode and candidate_branch == "copula"
                    else 0.40 if candidate_branch == "copula" else 0.80
                )
                marginal_limit = 0.18 if self.blueprint_mode else 0.25
                zero_rate_limit = 0.04 if self.blueprint_mode else 0.06
                class_conditional_limit = 0.35
                feature_label_nmi_limit = 0.20
                # The Copula branch is compared with a finite generated sample,
                # so the two nonlinear correlation summaries have sampling
                # error of order 1/sqrt(n).  A fixed 0.12 cutoff incorrectly
                # rejected otherwise excellent tasks at n=1024 (and is even
                # less defensible for shorter official sequence draws).  Keep
                # the strict 0.12 transport cutoff, but calibrate Copula's hard
                # gate to sample size and cap it so high-order mismatch remains
                # an actual rejection criterion rather than logging only.
                if self.blueprint_mode and candidate_branch == "copula":
                    finite_sample_higher_order_limit = min(
                        0.30, 0.12 + 1.5 / np.sqrt(max(seq_len, 1))
                    )
                    # Triangle closure is cubic in the correlation matrix;
                    # spectral concentration is quadratic.  Their absolute
                    # difference therefore cannot sensibly be constrained by
                    # a smaller, correlation-independent constant when the
                    # pairwise rank matrix itself is allowed finite error.
                    # Require the nonlinear summaries to agree within the
                    # cubic perturbation scale, with an absolute 0.50 ceiling.
                    higher_order_limit = float(
                        min(
                            0.50,
                            max(
                                finite_sample_higher_order_limit,
                                3.0 * metrics["quality_rank_correlation_mae"],
                            ),
                        )
                    )
                else:
                    higher_order_limit = 0.12
                task_difficulty_limit = 0.25
                # A softened official-profile transport deliberately preserves
                # more official GraphSCM dependence than the frozen target.
                # Keep class balance, marginals, sparsity, pairwise rank error,
                # quantile coverage and uniqueness strict, while calibrating
                # the three dependence-derived gates to the configured blend.
                # Otherwise the quality fallback silently turns every softened
                # task back into a pure Copula task and defeats the setting.
                if self.blueprint_mode and candidate_branch == "official_profile":
                    softness = 1.0 - self._profile_copula_scale(profile.name)
                    higher_order_limit = float(0.12 + 0.38 * softness)
                    feature_label_nmi_limit = float(0.20 + 0.60 * softness)
                    task_difficulty_limit = float(0.25 + 0.50 * softness)
                unique_fraction_min = 0.02
                passed = bool(
                    metrics["quality_class_tv"] <= class_tv_limit
                    and metrics["quality_marginal_quantile_scaled_mae"]
                    <= marginal_limit
                    and metrics["quality_zero_rate_mae"] <= zero_rate_limit
                    and metrics["quality_rank_correlation_mae"]
                    <= correlation_limit
                    and metrics.get("quality_quantile_bin_coverage", 1.0) >= 0.75
                    and metrics.get(
                        "quality_class_conditional_quantile_scaled_mae", 0.0
                    )
                    <= class_conditional_limit
                    and metrics.get("quality_feature_label_nmi_mae", 0.0)
                    <= feature_label_nmi_limit
                    and metrics.get(
                        "quality_higher_order_dependence_error", 0.0
                    )
                    <= higher_order_limit
                    and metrics.get("quality_task_difficulty_error", 0.0)
                    <= task_difficulty_limit
                    and metrics.get("quality_sample_unique_fraction", 1.0)
                    >= unique_fraction_min
                )
                return passed, {
                    **metrics,
                    "quality_gate_passed": passed,
                    "quality_gate_attempts": attempt_count,
                    "quality_gate_reference": (
                        "frozen_gt_statistical_blueprint"
                        if self.blueprint_mode
                        else "frozen_synthetic_surrogate"
                    ),
                    "quality_class_tv_limit": class_tv_limit,
                    "quality_marginal_quantile_scaled_mae_limit": marginal_limit,
                    "quality_zero_rate_mae_limit": zero_rate_limit,
                    "quality_rank_correlation_mae_limit": correlation_limit,
                    "quality_quantile_bin_coverage_min": 0.75,
                    "quality_class_conditional_quantile_scaled_mae_limit": (
                        class_conditional_limit
                    ),
                    "quality_feature_label_nmi_mae_limit": feature_label_nmi_limit,
                    "quality_higher_order_dependence_error_limit": (
                        higher_order_limit
                    ),
                    "quality_task_difficulty_error_limit": task_difficulty_limit,
                    "quality_sample_unique_fraction_min": unique_fraction_min,
                    "quality_profile_matching_applied": True,
                }

            supervised_selection_count = (
                self._profile_supervised_candidates(profile.name)
                if (
                    self.blueprint_mode
                    and branch == "official_profile"
                    and not profile_transport
                    and self._profile_shape_jitter(profile.name) == 0.0
                )
                else 1
            )
            max_attempts = (
                max(3, supervised_selection_count)
                if self.runtime_isolated and self.quality_gate
                else supervised_selection_count
            )
            quality_details: Dict[str, object] = {}
            best_supervised_candidate = None
            supervised_candidate_scores: List[float] = []
            for quality_attempt in range(max_attempts):
                attempt_seed = seed + 1009 * quality_attempt
                if self.blueprint_mode:
                    if branch == "official_shape":
                        x, y, details = self._generate_frozen_official(
                            profile,
                            seq_len,
                            train_size,
                            rng,
                            attempt_seed,
                            False,
                            bank,
                            blueprint,
                            official_num_features,
                        )
                    elif branch == "official_profile":
                        x, y, details = self._generate_frozen_official(
                            profile,
                            seq_len,
                            train_size,
                            rng,
                            attempt_seed,
                            profile_transport,
                            bank,
                            blueprint,
                            profile_shape_only=not profile_transport,
                        )
                    elif branch == "copula":
                        x, y, details = self._generate_frozen_copula(
                            profile, seq_len, train_size, rng, bank, blueprint
                        )
                    else:
                        raise ValueError(f"unknown hybrid178 branch {branch!r}")
                else:
                    if branch == "official_shape":
                        x, y, details = self._generate_official(
                            profile,
                            seq_len,
                            train_size,
                            rng,
                            attempt_seed,
                            False,
                            bank,
                        )
                    elif branch == "official_profile":
                        x, y, details = self._generate_official(
                            profile,
                            seq_len,
                            train_size,
                            rng,
                            attempt_seed,
                            True,
                            bank,
                        )
                    elif branch == "copula":
                        x, y, details = self._generate_copula(
                            profile, seq_len, train_size, rng, bank
                        )
                    else:
                        raise ValueError(f"unknown hybrid178 branch {branch!r}")

                if not (self.runtime_isolated and self.quality_gate):
                    break
                passed, quality_details = assess_quality(
                    x, y, branch, quality_attempt + 1
                )
                if supervised_selection_count > 1:
                    if passed:
                        nmi_error = float(
                            quality_details.get(
                                "quality_feature_label_nmi_mae", np.inf
                            )
                        )
                        higher_order_error = float(
                            quality_details.get(
                                "quality_higher_order_dependence_error", np.inf
                            )
                        )
                        difficulty_error = float(
                            quality_details.get(
                                "quality_task_difficulty_error", np.inf
                            )
                        )
                        supervised_score = float(
                            (
                                nmi_error / 0.20
                                + higher_order_error / 0.12
                                + difficulty_error / 0.25
                            )
                            / 3.0
                        )
                        supervised_candidate_scores.append(supervised_score)
                        if np.isfinite(supervised_score) and (
                            best_supervised_candidate is None
                            or supervised_score < best_supervised_candidate[0]
                        ):
                            best_supervised_candidate = (
                                supervised_score,
                                x,
                                y,
                                details,
                                quality_details,
                                quality_attempt,
                            )
                    if quality_attempt + 1 < supervised_selection_count:
                        continue
                    if best_supervised_candidate is not None:
                        (
                            _,
                            x,
                            y,
                            details,
                            quality_details,
                            selected_attempt,
                        ) = best_supervised_candidate
                        quality_details = {
                            **quality_details,
                            "quality_supervised_candidate_count": (
                                supervised_selection_count
                            ),
                            "quality_supervised_candidate_selected_index": int(
                                selected_attempt
                            ),
                            "quality_supervised_candidate_score": float(
                                best_supervised_candidate[0]
                            ),
                            "quality_supervised_candidate_scores": (
                                supervised_candidate_scores
                            ),
                            "quality_supervised_candidate_selection_applied": True,
                        }
                        passed = True
                if passed:
                    break
            else:
                # A scheduled conditioned slot is part of the DDP-global
                # 178-profile coverage contract.  If transported GraphSCM
                # repeatedly misses the supervised gate, keep that slot
                # conditioned by falling back to the frozen Copula generator;
                # silently replacing it with official_shape would make the
                # scheduled coverage claim false for this profile.
                if (
                    self.blueprint_mode
                    and branch == "official_profile"
                    and profile_transport
                ):
                    primary_quality_details = quality_details
                    for fallback_attempt in range(3):
                        # Keep fallback sampling task-local so a retry does not
                        # perturb the shared scheduler RNG or later tasks.
                        fallback_rng = np.random.default_rng(
                            seed + 5003 + fallback_attempt
                        )
                        x, y, details = self._generate_frozen_copula(
                            profile,
                            seq_len,
                            train_size,
                            fallback_rng,
                            bank,
                            blueprint,
                        )
                        passed, quality_details = assess_quality(
                            x,
                            y,
                            "copula",
                            max_attempts + fallback_attempt + 1,
                        )
                        if passed:
                            details.update(
                                {
                                    "quality_gate_fallback_used": True,
                                    "quality_gate_fallback_from": branch,
                                    "quality_gate_fallback_branch": "copula",
                                    "quality_gate_primary_attempts": max_attempts,
                                    "quality_gate_primary_rank_correlation_mae": primary_quality_details.get(
                                        "quality_rank_correlation_mae"
                                    ),
                                    "quality_gate_primary_feature_label_nmi_mae": primary_quality_details.get(
                                        "quality_feature_label_nmi_mae"
                                    ),
                                    "quality_gate_primary_task_difficulty_error": primary_quality_details.get(
                                        "quality_task_difficulty_error"
                                    ),
                                }
                            )
                            break
                    else:
                        raise ValueError(
                            "generated task and conditioned copula fallback failed "
                            "isolated "
                            f"statistical quality gate: {quality_details}"
                        )
                else:
                    raise ValueError(
                        "generated task failed isolated statistical quality gate: "
                        f"{quality_details}"
                    )
            effective_branch = str(
                details.get("quality_gate_fallback_branch", branch)
            )
            details.update(quality_details)
            details.update(feature_details)
            if effective_branch == "official_shape":
                details.update(
                    {
                        "synthetic_feature_count": int(x.shape[1]),
                        "feature_view_used_for_generation": False,
                        "all_template_features_preserved": False,
                    }
                )
            else:
                details["feature_view_used_for_generation"] = True
            if x.shape[1] > HYBRID178_MAX_FEATURES:
                raise AssertionError(
                    f"generated Hybrid-178 task exceeded {HYBRID178_MAX_FEATURES} features"
                )
            if np.unique(y).size > HYBRID178_MAX_CLASSES:
                raise AssertionError(
                    f"generated Hybrid-178 task exceeded {HYBRID178_MAX_CLASSES} classes"
                )
            if self.blueprint_mode:
                atom_details = {
                    "gt_atom_ratio_requested": 0.0,
                    "gt_atom_count": 0,
                    "gt_atom_fraction_actual": 0.0,
                    "gt_atom_min_rows": self.gt_atom_min_rows,
                    "gt_atom_finite_sample_hit_guaranteed": False,
                    "source_rows_sampled": False,
                    "row_bootstrap": False,
                    "synthetic_only": True,
                    "real_row_replay": False,
                }
            else:
                x, atom_details = _inject_gt_atoms(
                    x,
                    y,
                    atom_data,
                    atom_class_groups,
                    rng,
                    ratio=self.gt_atom_ratio,
                    min_rows=self.gt_atom_min_rows,
                    require_full_schema=self.gt_atom_require_full_schema,
                    train_size=train_size,
                )
            details.update(atom_details)
            if bool(details.get("official_graph_prior_exact_backbone", False)):
                x_t, y_t, d_t = self._finalize_official_backbone(
                    x,
                    y,
                    int(details["official_shape_sampled_num_features"]),
                )
            else:
                x_t, y_t, d_t = self._finalize(
                    x,
                    y,
                    train_size,
                    rng,
                    preserve_schema=(
                        effective_branch in {"official_profile", "copula"}
                        or self.gt_atom_ratio > 0.0
                    ),
                    pad_to_max_features=True,
                )
            if int(details.get("gt_atom_count", 0)) > 0 and atom_data is not None:
                atom_slots = np.asarray(details["gt_atom_slots"], dtype=np.int64)
                atom_sources = np.asarray(
                    details["gt_atom_source_rows"], dtype=np.int64
                )
                raw = np.nan_to_num(
                    x, nan=0.0, posinf=1e9, neginf=-1e9
                ).astype(np.float32, copy=False)
                support = raw[:train_size]
                mean = support.mean(axis=0, dtype=np.float64)
                std = support.std(axis=0, dtype=np.float64)
                std = np.where(std > 1e-6, std, 1.0)
                expected_atoms = (
                    (atom_data.x[atom_sources, : raw.shape[1]] - mean) / std
                ).astype(np.float32)
                actual_atoms = (
                    x_t[torch.as_tensor(atom_slots, device=x_t.device), : raw.shape[1]]
                    .detach()
                    .cpu()
                    .numpy()
                )
                if not np.array_equal(actual_atoms, expected_atoms):
                    raise AssertionError(
                        "GT atom was not preserved by task normalization"
                    )
                details["gt_atom_model_input_exact"] = True

        metadata: Dict[str, object] = {
            "prior_kind": "hybrid178",
            "branch": branch,
            "effective_branch": effective_branch,
            "profile": profile.name,
            "profile_splits": list(bank.splits),
            "profile_coverage_scheduled": True,
            "official_profile_transport_enabled": bool(
                branch == "official_profile" and profile_transport
            ),
            "official_profile_transport_effective": bool(
                details.get("profile_full_marginal_transport", False)
            ),
            "official_profile_transport_ratio": self._profile_transport_spec(
                profile.name
            )[0],
            "profile_supervised_candidate_count_configured": (
                self._profile_supervised_candidates(profile.name)
            ),
            "benchmark_conditioned": bool(
                details.get(
                    "profile_conditioned",
                    effective_branch != "official_shape",
                )
            ),
            "runtime_isolated": self.runtime_isolated,
            "runtime_real_data_access": False if self.runtime_isolated else None,
            "runtime_profile_source": (
                "frozen_gt_statistical_blueprint"
                if self.blueprint_mode
                else "compiled_synthetic_surrogate"
                if self.runtime_isolated
                else "configured_profile_root"
            ),
            "compiled_manifest_sha256": self.compiled_manifest_digest,
            "synthetic_only": True,
            "real_row_replay": False,
            "test_data_used": False,
            "seq_len": seq_len,
            "train_size": train_size,
            "final_num_features": int(d_t.item()),
            "final_num_classes": int(torch.unique(y_t).numel()),
            **details,
        }
        return x_t, y_t, d_t, metadata

    @torch.no_grad()
    def get_batch(self, batch_size: Optional[int] = None):
        bank, rng, scheduler = self._ensure_state()
        batch_size = int(batch_size or self.batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        profile_ids, branches, global_batch_index = self._take_schedule(
            batch_size, scheduler, rng
        )
        x_list: List[Tensor] = []
        y_list: List[Tensor] = []
        d_list: List[Tensor] = []
        seq_lens: List[int] = []
        train_sizes: List[int] = []
        metadata: List[Dict[str, object]] = []

        global_seq_len = (
            self._sample_seq_len() if not self.seq_len_per_gp else None
        )
        size_per_gp = min(self.batch_size_per_gp, batch_size)
        gp_seq_len: Optional[int] = None
        gp_official_train_size: Optional[int] = None
        gp_official_num_features: Optional[int] = None
        global_official_train_size = (
            self._official_train_size(int(global_seq_len), rng)
            if global_seq_len is not None
            else None
        )
        for idx, (profile_idx, branch) in enumerate(
            zip(profile_ids.tolist(), branches)
        ):
            if idx % size_per_gp == 0:
                gp_official_num_features = int(
                    round(rng.uniform(self.min_features, self.max_features))
                )
                gp_official_num_features = int(
                    np.clip(
                        gp_official_num_features,
                        self.min_features,
                        self.max_features,
                    )
                )
                if self.seq_len_per_gp:
                    gp_seq_len = self._sample_seq_len()
                    gp_official_train_size = self._official_train_size(
                        int(gp_seq_len), rng
                    )
            if self.seq_len_per_gp:
                seq_len = int(gp_seq_len)
                official_train_size = int(gp_official_train_size)
            else:
                seq_len = int(global_seq_len)
                official_train_size = int(global_official_train_size)
            # Trainer micro-batches require one shared support boundary.  Use
            # the official GraphPrior draw for every branch in this group: the
            # conditioned minority supplies shape/distribution information,
            # while support-size sampling remains part of the official
            # backbone rather than another benchmark-specific constraint.
            train_size = official_train_size
            profile = bank.profiles[int(profile_idx)]
            seed = int(rng.integers(0, 2**31 - 1))
            if global_batch_index is not None:
                rank, _ = self._worker_identity()
                world_size = max(
                    1,
                    int(
                        os.environ.get(
                            "WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")
                        )
                    ),
                )
                global_slot = rank * batch_size + idx
                global_task_index = (
                    global_batch_index * world_size * batch_size + global_slot
                )
                feature_view_index = global_task_index // len(bank)
            else:
                feature_view_index = self._feature_view_counts.get(profile.name, 0)
                self._feature_view_counts[profile.name] = feature_view_index + 1
            profile_transport = True
            if branch == "official_profile":
                transport_cycle = (
                    global_batch_index
                    if global_batch_index is not None
                    else feature_view_index
                )
                profile_transport = self._profile_transport_enabled(
                    int(profile_idx), int(transport_cycle), profile.name
                )
            try:
                x, y, d, task_metadata = self._generate_one(
                    branch,
                    profile,
                    seq_len,
                    train_size,
                    seed,
                    bank,
                    rng,
                    feature_view_index,
                    gp_official_num_features,
                    profile_transport,
                )
            except (
                RuntimeError,
                ValueError,
                np.linalg.LinAlgError,
            ) as exc:
                if branch == "official_shape":
                    raise
                x, y, d, task_metadata = self._generate_one(
                    "official_profile",
                    profile,
                    seq_len,
                    train_size,
                    seed + 17,
                    bank,
                    rng,
                    feature_view_index,
                    profile_transport=(
                        profile_transport if branch == "official_profile" else True
                    ),
                )
                task_metadata["requested_branch"] = branch
                task_metadata["fallback_reason"] = (
                    f"{type(exc).__name__}: {exc}"
                )
            if global_batch_index is not None:
                rank, _ = self._worker_identity()
                world_size = max(
                    1,
                    int(
                        os.environ.get(
                            "WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")
                        )
                    ),
                )
                task_metadata["ddp_global_batch_index"] = global_batch_index
                task_metadata["ddp_global_slot"] = rank * batch_size + idx
                coverage_capacity = bool(
                    world_size * batch_size * self.branch_probs[1:].sum()
                    >= len(bank)
                )
                scheduled_conditioned = branch != "official_shape"
                effective_conditioned = (
                    task_metadata.get("effective_branch") != "official_shape"
                )
                if scheduled_conditioned and not effective_conditioned:
                    raise AssertionError(
                        "conditioned Hybrid-178 schedule slot lost its profile "
                        "conditioning during generation"
                    )
                task_metadata[
                    "ddp_global_conditioned_profile_coverage_guaranteed"
                ] = coverage_capacity
                task_metadata[
                    "ddp_global_conditioned_fallback_policy"
                ] = "conditioned_only"
                task_metadata[
                    "ddp_global_conditioned_slot_preserved"
                ] = bool(not scheduled_conditioned or effective_conditioned)
            task_metadata["protected_priority_extras_enabled"] = bool(
                self.protected_priority_extras
            )
            task_metadata["protected_priority_hardness_weighted"] = bool(
                self.protected_priority_hardness_weighted
            )
            task_metadata["protected_priority_collapse_risk_weighted"] = bool(
                self.protected_priority_collapse_risk_weighted
            )
            (
                task_transport_ratio,
                task_transport_numerator,
                task_transport_denominator,
            ) = self._profile_transport_spec(profile.name)
            task_metadata["profile_transport_ratio_effective"] = float(
                task_transport_ratio
            )
            task_metadata["profile_transport_rotation_numerator"] = int(
                task_transport_numerator
            )
            task_metadata["profile_transport_rotation_denominator"] = int(
                task_transport_denominator
            )
            if (
                self.protected_priority_hardness_weighted
                and profile.name in HYBRID178_PROTECTED_HARDNESS_SCORES
            ):
                priority_position = self._priority_profile_ids.index(
                    int(profile_idx)
                )
                task_metadata["protected_priority_row_free_score"] = float(
                    HYBRID178_PROTECTED_HARDNESS_SCORES[profile.name]
                )
                task_metadata["protected_priority_sampling_weight"] = float(
                    self._priority_profile_weights[priority_position]
                )
            elif (
                self.protected_priority_collapse_risk_weighted
                and profile.name in HYBRID178_PROTECTED_COLLAPSE_RISK_WEIGHTS
            ):
                priority_position = self._priority_profile_ids.index(
                    int(profile_idx)
                )
                task_metadata["protected_priority_row_free_collapse_risk_weight"] = float(
                    HYBRID178_PROTECTED_COLLAPSE_RISK_WEIGHTS[profile.name]
                )
                task_metadata["protected_priority_sampling_weight"] = float(
                    self._priority_profile_weights[priority_position]
                )
            x_list.append(x)
            y_list.append(y)
            d_list.append(d)
            seq_lens.append(seq_len)
            train_sizes.append(train_size)
            metadata.append(task_metadata)

        # Cross-generated-task diversity is checked after final normalization,
        # within every rank-local batch.  The global scheduler separately
        # guarantees profile coverage; this check prevents accidental duplicate
        # task generation without coupling all DDP data workers.
        diversity_min_l2 = 1e-5
        if len(x_list) > 1:
            fingerprints = np.stack(
                [
                    _task_fingerprint(
                        x_task[:, : int(d_task.item())]
                        .detach()
                        .cpu()
                        .numpy(),
                        y_task.detach().cpu().numpy(),
                    )
                    for x_task, y_task, d_task in zip(x_list, y_list, d_list)
                ]
            )
            distances = np.linalg.norm(
                fingerprints[:, None, :] - fingerprints[None, :, :],
                axis=-1,
            )
            np.fill_diagonal(distances, np.inf)
            nearest = distances.min(axis=1)
        else:
            nearest = np.asarray([1.0], dtype=np.float64)
        diversity_passed = nearest >= diversity_min_l2
        for item, distance, passed in zip(
            metadata, nearest.tolist(), diversity_passed.tolist()
        ):
            item["quality_cross_task_diversity_scope"] = "rank_local_batch"
            item["quality_cross_task_fingerprint_nearest_l2"] = float(distance)
            item["quality_cross_task_fingerprint_min_l2"] = diversity_min_l2
            item["quality_cross_task_diversity_passed"] = bool(passed)
        if self.quality_gate and not bool(np.all(diversity_passed)):
            raise ValueError(
                "Hybrid-178 generated near-duplicate tasks within one local batch: "
                f"nearest_l2={float(np.min(nearest)):.6g}"
            )

        if self.seq_len_per_gp:
            x_batch = nested_tensor(x_list, device=self.device)
            y_batch = nested_tensor(y_list, device=self.device)
        else:
            x_batch = torch.stack(x_list).to(self.device)
            y_batch = torch.stack(y_list).to(self.device)
        d_batch = torch.stack(d_list).to(self.device)
        seq_lens_t = torch.tensor(
            seq_lens, dtype=torch.long, device=self.device
        )
        train_sizes_t = torch.tensor(
            train_sizes, dtype=torch.long, device=self.device
        )
        self.last_metadata = metadata
        output = (
            x_batch,
            y_batch,
            d_batch,
            seq_lens_t,
            train_sizes_t,
        )
        return (*output, metadata) if self.return_metadata else output

    def __repr__(self) -> str:
        ratios = dict(zip(BRANCH_NAMES, self.branch_probs.tolist()))
        loaded = len(self._bank) if self._bank is not None else "lazy"
        splits = "train+val" if self.include_val else "train"
        return (
            "Hybrid178Prior("
            f"profiles={loaded}, splits={splits}, branch_ratios={ratios}, "
            f"gt_atom_ratio={self.gt_atom_ratio}, "
            f"gt_atom_min_rows={self.gt_atom_min_rows}, "
            f"runtime_isolated={self.runtime_isolated}, "
            f"profile_transport_ratio={self.profile_transport_ratio}, "
            "protected_profile_transport_ratio="
            f"{self.protected_profile_transport_ratio}, "
            "target_profile_transport_ratio="
            f"{self.target_profile_transport_ratio}, "
            "profile_shape_jitter_strength="
            f"{self.profile_shape_jitter_strength}, "
            "protected_profile_shape_jitter_strength="
            f"{self.protected_profile_shape_jitter_strength}, "
            "target_profile_shape_jitter_strength="
            f"{self.target_profile_shape_jitter_strength}, "
            "profile_supervised_candidate_count="
            f"{self.profile_supervised_candidate_count}, "
            "protected_profile_supervised_candidate_count="
            f"{self.protected_profile_supervised_candidate_count}, "
            "target_profile_supervised_candidate_count="
            f"{self.target_profile_supervised_candidate_count}, "
            "profile_copula_blend_scale="
            f"{self.profile_copula_blend_scale}, "
            "protected_profile_copula_blend_scale="
            f"{self.protected_profile_copula_blend_scale}, "
            "target_profile_copula_blend_scale="
            f"{self.target_profile_copula_blend_scale}, "
            f"protected_priority_extras={self.protected_priority_extras}, "
            "protected_priority_hardness_weighted="
            f"{self.protected_priority_hardness_weighted}, "
            "protected_priority_collapse_risk_weighted="
            f"{self.protected_priority_collapse_risk_weighted}, "
            f"train_fraction={self.train_fraction}, seed={self.base_seed}, "
            f"schedule_start_batch={self.schedule_start_batch})"
        )
