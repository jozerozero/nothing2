from __future__ import annotations

import math
import os
from fractions import Fraction
import re
import json
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from train_only_knn_mixup_auxiliary import TrainOnlyKNNMixup
from train_only_paired_ulp_warp_auxiliary import TrainOnlyPairedULPWarp
from train_only_topology_preserving_ulp_warp_auxiliary import (
    TrainOnlyTopologyPreservingULPWarp,
)
from train_only_affine_symmetry_auxiliary import TrainOnlyAffineSymmetry
from train_only_routed_power2_affine_auxiliary import (
    TrainOnlyRoutedPower2Affine,
)
from train_only_expanding_power2_affine_auxiliary import (
    TrainOnlyExpandingPower2Affine,
)

from ._tabiclv2_classification import (
    TabICLv2ClassificationGenerator,
    TabICLv2ClassificationPrior,
    _ordinal_encode,
    _pad_features,
    _remove_outliers,
    _standardize,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_PROTECTED_CACHE_DIRS = (
    PROJECT_ROOT / "evaluation_results/data178_health_dataset_cache",
    PROJECT_ROOT / "evaluation_results/data178_collapse_dataset_cache",
    PROJECT_ROOT / "evaluation_results/data178_support_attention_cache",
)

PROTECTED_DATASET_NAMES = (
    "internet_firewall",
    "ada",
    "ada_agnostic",
    "MIC",
    "thyroid-dis",
    "mfeat-zernike",
    "waveform_database_generator",
    "autoUniv-au4-2500",
    "pc1",
    "pc4",
    "phoneme",
    "UJI_Pen_Characters",
    "splice",
    "naticusdroid+android+permissions+dataset",
    "mfeat-pixel",
    "vehicle",
    "allbp",
    "baseball",
    "National_Health_and_Nutrition_Health_Survey",
    "Customer_Personality_Analysis",
)

TRAIN_ONLY_PROTECTED_DATASET_NAMES = tuple(
    name for name in PROTECTED_DATASET_NAMES
    if name != "waveform_database_generator"
)

TRAIN_ONLY_GT_SOURCES = {"train_gt", "clean_train_gt", "protected_train_gt"}
TRAIN_ONLY_K2_SOURCES = {"train_knn_k2", "knn_k2", "protected_train_knn_k2"}
TRAIN_ONLY_L1_SOURCES = {
    "train_paired_ulp_warp",
    "paired_ulp_warp",
    "protected_train_paired_ulp_warp",
}
TRAIN_ONLY_L2_SOURCES = {
    "train_topology_ulp_warp",
    "topology_ulp_warp",
    "protected_train_topology_ulp_warp",
}
TRAIN_ONLY_L3_SOURCES = {
    "train_affine_symmetry",
    "affine_symmetry",
    "protected_train_affine_symmetry",
}
TRAIN_ONLY_L5_SOURCES = {
    "train_routed_power2_affine",
    "routed_power2_affine",
    "protected_train_routed_power2_affine",
}
TRAIN_ONLY_L6_SOURCES = {
    "train_expanding_power2_affine",
    "expanding_power2_affine",
    "protected_train_expanding_power2_affine",
}

OBSERVED_LABEL_SCOPE = {
    "scope": "observed_label_training_effect_emulation_only",
    "dataset": "waveform_database_generator",
    "allowed_inputs": ["X_train", "y_train"],
    "external_label_join_allowed": False,
    "class_ids_are_task_local_and_exchangeable": True,
    "physical_label_semantics_claimed": False,
    "physical_or_unique_dgp_claimed": False,
}


def _exact_global_mix_count(
    *,
    batch_size: int,
    ratio: float,
    logical_batch: int,
    rank: int,
    world_size: int,
) -> int:
    """Return a deterministic per-rank count with an exact global period mean."""

    if batch_size <= 0 or world_size <= 0:
        raise ValueError("batch_size and world_size must be positive")
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("ratio must be in [0, 1]")
    fraction = Fraction(str(ratio)).limit_denominator(10_000)
    if not math.isclose(float(fraction), ratio, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("ratio is not representable by the bounded exact scheduler")
    period = fraction.denominator
    base = (batch_size * fraction.numerator) // period
    extras_total = (
        batch_size * world_size * fraction.numerator
        - base * world_size * period
    )
    if not 0 <= extras_total <= world_size * period:
        raise RuntimeError("invalid exact-mixture residual")
    extras_per_phase, remainder_phases = divmod(extras_total, period)
    phase = logical_batch % period
    extras_this_phase = extras_per_phase + int(phase < remainder_phases)
    offset = (
        extras_per_phase * phase + min(phase, remainder_phases)
    ) % world_size
    relative_rank = (rank % world_size - offset) % world_size
    return base + int(relative_rank < extras_this_phase)


def _repair_model_space_feature_collisions(
    x: Tensor, reference_x: Tensor
) -> tuple[Tensor, int, int, int]:
    """Remove byte-exact postprocessed feature matches with minimal ULP moves."""

    device, dtype = x.device, x.dtype
    values = np.ascontiguousarray(x.detach().cpu().numpy()).copy()
    reference = np.ascontiguousarray(reference_x.detach().cpu().numpy())
    reference_keys = {
        np.ascontiguousarray(row).tobytes() for row in reference
    }
    collision_indices = [
        index for index, row in enumerate(values)
        if np.ascontiguousarray(row).tobytes() in reference_keys
    ]
    repairs = 0
    for row_index in collision_indices:
        resolved = False
        for attempt in range(max(8, 4 * values.shape[1])):
            column = (row_index + attempt) % values.shape[1]
            direction = np.inf if (attempt // values.shape[1]) % 2 == 0 else -np.inf
            values[row_index, column] = np.nextafter(
                values[row_index, column], direction, dtype=values.dtype
            )
            repairs += 1
            if np.ascontiguousarray(values[row_index]).tobytes() not in reference_keys:
                resolved = True
                break
        if not resolved:
            raise RuntimeError("unable to remove a model-space exact train collision")
    remaining = sum(
        np.ascontiguousarray(row).tobytes() in reference_keys for row in values
    )
    if remaining:
        raise RuntimeError("model-space exact train collision remains after repair")
    return torch.as_tensor(values, dtype=dtype, device=device), len(collision_indices), repairs, remaining


def _parse_list(value: Optional[str | Sequence[str]], default: Sequence[str]) -> tuple[str, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return tuple(default)
        return tuple(part for part in re.split(r"[,;]", text) if part)
    return tuple(str(part) for part in value if str(part))


def _parse_cache_dirs(value: Optional[str | Sequence[str]]) -> tuple[Path, ...]:
    if value is None:
        return tuple(DEFAULT_PROTECTED_CACHE_DIRS)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return tuple(DEFAULT_PROTECTED_CACHE_DIRS)
        parts = [part for part in re.split(r"[,;:]", text) if part]
    else:
        parts = [str(part) for part in value if str(part)]
    return tuple(Path(part) for part in parts)


@dataclass
class _ProtectedDataset:
    name: str
    x: np.ndarray
    y: np.ndarray
    class_values: np.ndarray
    class_probs: np.ndarray
    class_indices: tuple[np.ndarray, ...]
    near_zero_ratio: float
    integer_like_ratio: float
    abs_q95: float
    abs_q99: float

    @property
    def n_features(self) -> int:
        return int(self.x.shape[1])

    @property
    def n_classes(self) -> int:
        return int(len(self.class_values))


def _encode_labels(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_arr = np.asarray(y)
    if y_arr.dtype.kind in {"O", "U", "S"}:
        values, inv = np.unique(y_arr.astype(str), return_inverse=True)
    else:
        values, inv = np.unique(y_arr, return_inverse=True)
    return inv.astype(np.int64, copy=False), values


def _load_protected_dataset(
    path: Path,
    name: str,
    *,
    train_only: bool = False,
) -> _ProtectedDataset:
    data = np.load(path, allow_pickle=True)
    if train_only:
        if set(data.files) != {"X_train", "y_train"}:
            raise ValueError(
                f"train-only protected cache has forbidden keys: {sorted(data.files)}"
            )
        x = np.asarray(data["X_train"]).astype(np.float32, copy=False)
        y_raw = np.asarray(data["y_train"])
    else:
        x = np.concatenate([data["X_train"], data["X_test"]], axis=0).astype(np.float32, copy=False)
        y_raw = np.concatenate([data["y_train"], data["y_test"]], axis=0)
    y, values = _encode_labels(y_raw)
    x = np.nan_to_num(x, nan=0.0, posinf=1e9, neginf=-1e9).astype(np.float32, copy=False)

    counts = np.bincount(y, minlength=len(values)).astype(np.float64)
    class_probs = counts / max(float(counts.sum()), 1.0)
    class_indices = tuple(np.flatnonzero(y == cls).astype(np.int64, copy=False) for cls in range(len(values)))
    near_zero = float(np.mean(np.abs(x) < 1e-8)) if x.size else 0.0
    integer_like = float(np.mean(np.isclose(x, np.round(x), atol=1e-6))) if x.size else 0.0
    abs_x = np.abs(x[np.isfinite(x)])
    if abs_x.size:
        abs_q95 = float(np.quantile(abs_x, 0.95))
        abs_q99 = float(np.quantile(abs_x, 0.99))
    else:
        abs_q95 = 1.0
        abs_q99 = 1.0
    return _ProtectedDataset(
        name=name,
        x=x,
        y=y,
        class_values=values,
        class_probs=class_probs,
        class_indices=class_indices,
        near_zero_ratio=near_zero,
        integer_like_ratio=integer_like,
        abs_q95=max(abs_q95, 1e-6),
        abs_q99=max(abs_q99, 1e-6),
    )


class ProtectedBatchAuxPrior:
    """Auxiliary protected-style prior used inside a larger TabICLv2 batch."""

    def __init__(
        self,
        *,
        source: str,
        batch_size: int = 256,
        min_features: int = 2,
        max_features: int = 100,
        max_classes: int = 10,
        min_seq_len: Optional[int] = None,
        max_seq_len: int = 1024,
        log_seq_len: bool = False,
        min_train_size: int | float = 0.1,
        max_train_size: int | float = 0.9,
        replay_small: bool = False,
        protected_cache_dirs: Optional[str | Sequence[str]] = None,
        protected_dataset_names: Optional[str | Sequence[str]] = None,
        device: str = "cpu",
    ):
        self.source = (source or "gt").strip().lower()
        self.train_only_source = (
            self.source in TRAIN_ONLY_GT_SOURCES
            or self.source in TRAIN_ONLY_K2_SOURCES
            or self.source in TRAIN_ONLY_L1_SOURCES
            or self.source in TRAIN_ONLY_L2_SOURCES
            or self.source in TRAIN_ONLY_L3_SOURCES
            or self.source in TRAIN_ONLY_L5_SOURCES
            or self.source in TRAIN_ONLY_L6_SOURCES
        )
        self.batch_size = int(batch_size)
        self.min_features = int(min_features)
        self.max_features = int(max_features)
        self.max_classes = int(max_classes)
        self.min_seq_len = min_seq_len
        self.max_seq_len = int(max_seq_len)
        self.log_seq_len = bool(log_seq_len)
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.replay_small = bool(replay_small)
        self.protected_cache_dirs = _parse_cache_dirs(
            protected_cache_dirs or os.environ.get("PROTECTED_BATCH_MIX_CACHE_DIRS")
        )
        default_names = (
            TRAIN_ONLY_PROTECTED_DATASET_NAMES
            if self.train_only_source
            else PROTECTED_DATASET_NAMES
        )
        if protected_dataset_names is None:
            self._protected_dataset_names_explicit = False
        elif isinstance(protected_dataset_names, str):
            # The training CLI exports an unset optional list as an empty
            # string.  `_parse_list` already treats that spelling as the
            # default selector, so it must not activate the explicit-list
            # equality firewall.
            self._protected_dataset_names_explicit = bool(
                protected_dataset_names.strip()
            )
        else:
            self._protected_dataset_names_explicit = any(
                str(part).strip() for part in protected_dataset_names
            )
        self.protected_dataset_names = _parse_list(protected_dataset_names, default_names)
        self.device = device
        self._datasets: Optional[tuple[_ProtectedDataset, ...]] = None
        self._rng: Optional[np.random.Generator] = None
        self._runtime4_prior = None
        self._train_only_manifest = None
        self._all178_fold = None
        self._train_only_source_family_by_name: dict[str, str] = {}
        self._train_only_family_members: dict[str, tuple[str, ...]] = {}
        self._train_only_generators: dict[str, object] = {}
        self._train_only_audit = {
            "tasks": 0,
            "synthetic_tasks": 0,
            "nearcopy_tasks": 0,
            "nearcopy_rows": 0,
            "nearcopy_max_ulp_steps": 0,
            "nearcopy_normalized_delta_rmse_max": 0.0,
            "affine_tasks": 0,
            "affine_rows": 0,
            "affine_translation_abs_max": 0.0,
            "affine_normalized_delta_rmse_min": None,
            "affine_normalized_delta_rmse_max": 0.0,
            "routed_tasks": 0,
            "routed_rows": 0,
            "routed_factor2_tasks": 0,
            "routed_affine_tasks": 0,
            "routed_max_ulp_steps": 0,
            "routed_normalized_delta_rmse_min": None,
            "routed_normalized_delta_rmse_max": 0.0,
            "exact_train_collisions": 0,
            "novelty_repairs": 0,
            "continuous_jitter_repairs": 0,
            "model_space_collisions_before_repair": 0,
            "model_space_repairs": 0,
            "source_counts": {},
            "family_counts": {},
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_datasets"] = None
        state["_rng"] = None
        state["_runtime4_prior"] = None
        state["_train_only_manifest"] = None
        state["_all178_fold"] = None
        state["_train_only_source_family_by_name"] = {}
        state["_train_only_family_members"] = {}
        state["_train_only_generators"] = {}
        state["_train_only_audit"] = {
            "tasks": 0,
            "synthetic_tasks": 0,
            "nearcopy_tasks": 0,
            "nearcopy_rows": 0,
            "nearcopy_max_ulp_steps": 0,
            "nearcopy_normalized_delta_rmse_max": 0.0,
            "affine_tasks": 0,
            "affine_rows": 0,
            "affine_translation_abs_max": 0.0,
            "affine_normalized_delta_rmse_min": None,
            "affine_normalized_delta_rmse_max": 0.0,
            "routed_tasks": 0,
            "routed_rows": 0,
            "routed_factor2_tasks": 0,
            "routed_affine_tasks": 0,
            "routed_max_ulp_steps": 0,
            "routed_normalized_delta_rmse_min": None,
            "routed_normalized_delta_rmse_max": 0.0,
            "exact_train_collisions": 0,
            "novelty_repairs": 0,
            "continuous_jitter_repairs": 0,
            "model_space_collisions_before_repair": 0,
            "model_space_repairs": 0,
            "source_counts": {},
            "family_counts": {},
        }
        return state

    def train_only_audit_snapshot(self) -> dict:
        return {
            "tasks": int(self._train_only_audit["tasks"]),
            "synthetic_tasks": int(self._train_only_audit["synthetic_tasks"]),
            "nearcopy_tasks": int(self._train_only_audit["nearcopy_tasks"]),
            "nearcopy_rows": int(self._train_only_audit["nearcopy_rows"]),
            "nearcopy_max_ulp_steps": int(
                self._train_only_audit["nearcopy_max_ulp_steps"]
            ),
            "nearcopy_normalized_delta_rmse_max": float(
                self._train_only_audit["nearcopy_normalized_delta_rmse_max"]
            ),
            "affine_tasks": int(self._train_only_audit["affine_tasks"]),
            "affine_rows": int(self._train_only_audit["affine_rows"]),
            "affine_translation_abs_max": float(
                self._train_only_audit["affine_translation_abs_max"]
            ),
            "affine_normalized_delta_rmse_min": (
                None if self._train_only_audit["affine_normalized_delta_rmse_min"] is None
                else float(self._train_only_audit["affine_normalized_delta_rmse_min"])
            ),
            "affine_normalized_delta_rmse_max": float(
                self._train_only_audit["affine_normalized_delta_rmse_max"]
            ),
            "routed_tasks": int(self._train_only_audit["routed_tasks"]),
            "routed_rows": int(self._train_only_audit["routed_rows"]),
            "routed_factor2_tasks": int(
                self._train_only_audit["routed_factor2_tasks"]
            ),
            "routed_affine_tasks": int(
                self._train_only_audit["routed_affine_tasks"]
            ),
            "routed_max_ulp_steps": int(
                self._train_only_audit["routed_max_ulp_steps"]
            ),
            "routed_normalized_delta_rmse_min": (
                None if self._train_only_audit["routed_normalized_delta_rmse_min"] is None
                else float(self._train_only_audit["routed_normalized_delta_rmse_min"])
            ),
            "routed_normalized_delta_rmse_max": float(
                self._train_only_audit["routed_normalized_delta_rmse_max"]
            ),
            "exact_train_collisions": int(
                self._train_only_audit["exact_train_collisions"]
            ),
            "novelty_repairs": int(self._train_only_audit["novelty_repairs"]),
            "continuous_jitter_repairs": int(
                self._train_only_audit["continuous_jitter_repairs"]
            ),
            "model_space_collisions_before_repair": int(
                self._train_only_audit["model_space_collisions_before_repair"]
            ),
            "model_space_repairs": int(
                self._train_only_audit["model_space_repairs"]
            ),
            "source_counts": dict(sorted(
                self._train_only_audit["source_counts"].items()
            )),
            "family_counts": dict(sorted(
                self._train_only_audit["family_counts"].items()
            )),
            "all178_fold": self._all178_fold,
        }

    def _record_train_only_task(
        self,
        source_name: str,
        *,
        synthetic: bool,
        exact_collisions: int = 0,
        novelty_repairs: int = 0,
        continuous_jitter_repairs: int = 0,
        model_space_collisions_before_repair: int = 0,
        model_space_repairs: int = 0,
        nearcopy: bool = False,
        nearcopy_rows: int = 0,
        nearcopy_max_ulp_steps: int = 0,
        nearcopy_normalized_delta_rmse: float = 0.0,
        affine: bool = False,
        affine_rows: int = 0,
        affine_translation: float = 0.0,
        affine_normalized_delta_rmse: float = 0.0,
        routed: bool = False,
        routed_rows: int = 0,
        routed_route: str = "",
        routed_max_ulp_steps: int = 0,
        routed_normalized_delta_rmse: float = 0.0,
    ) -> None:
        audit = self._train_only_audit
        audit["tasks"] += 1
        audit["synthetic_tasks"] += int(synthetic)
        audit["nearcopy_tasks"] += int(nearcopy)
        audit["nearcopy_rows"] += int(nearcopy_rows)
        audit["nearcopy_max_ulp_steps"] = max(
            int(audit["nearcopy_max_ulp_steps"]), int(nearcopy_max_ulp_steps)
        )
        audit["nearcopy_normalized_delta_rmse_max"] = max(
            float(audit["nearcopy_normalized_delta_rmse_max"]),
            float(nearcopy_normalized_delta_rmse),
        )
        audit["affine_tasks"] += int(affine)
        audit["affine_rows"] += int(affine_rows)
        audit["affine_translation_abs_max"] = max(
            float(audit["affine_translation_abs_max"]),
            abs(float(affine_translation)),
        )
        if affine:
            current_min = audit["affine_normalized_delta_rmse_min"]
            audit["affine_normalized_delta_rmse_min"] = (
                float(affine_normalized_delta_rmse)
                if current_min is None
                else min(float(current_min), float(affine_normalized_delta_rmse))
            )
            audit["affine_normalized_delta_rmse_max"] = max(
                float(audit["affine_normalized_delta_rmse_max"]),
                float(affine_normalized_delta_rmse),
            )
        audit["routed_tasks"] += int(routed)
        audit["routed_rows"] += int(routed_rows)
        audit["routed_factor2_tasks"] += int(
            routed and routed_route in {"factor2_ulp", "expanding_power2_exact"}
        )
        audit["routed_affine_tasks"] += int(routed and routed_route == "l3_affine")
        audit["routed_max_ulp_steps"] = max(
            int(audit["routed_max_ulp_steps"]), int(routed_max_ulp_steps)
        )
        if routed:
            current_min = audit["routed_normalized_delta_rmse_min"]
            audit["routed_normalized_delta_rmse_min"] = (
                float(routed_normalized_delta_rmse)
                if current_min is None
                else min(float(current_min), float(routed_normalized_delta_rmse))
            )
            audit["routed_normalized_delta_rmse_max"] = max(
                float(audit["routed_normalized_delta_rmse_max"]),
                float(routed_normalized_delta_rmse),
            )
        audit["exact_train_collisions"] += int(exact_collisions)
        audit["novelty_repairs"] += int(novelty_repairs)
        audit["continuous_jitter_repairs"] += int(continuous_jitter_repairs)
        audit["model_space_collisions_before_repair"] += int(
            model_space_collisions_before_repair
        )
        audit["model_space_repairs"] += int(model_space_repairs)
        counts = audit["source_counts"]
        counts[source_name] = int(counts.get(source_name, 0)) + 1
        family = self._train_only_source_family_by_name.get(
            source_name, f"legacy::{source_name}"
        )
        family_counts = audit["family_counts"]
        family_counts[family] = int(family_counts.get(family, 0)) + 1
        every = int(os.environ.get("PROTECT_TRAIN_ONLY_AUDIT_EVERY", "100"))
        if every > 0 and audit["tasks"] % every == 0:
            print(
                "[protect-train-only-audit] "
                + json.dumps(self.train_only_audit_snapshot(), sort_keys=True),
                file=sys.stderr,
                flush=True,
            )

    def _ensure_train_only_manifest(self) -> dict:
        if self._train_only_manifest is not None:
            return self._train_only_manifest
        path = Path(os.environ.get("PROTECT_TRAIN_ONLY_CACHE_MANIFEST", ""))
        expected = os.environ.get("PROTECT_TRAIN_ONLY_CACHE_MANIFEST_SHA256", "").lower()
        if not path.is_file() or not expected:
            raise FileNotFoundError(
                "train-only sources require PROTECT_TRAIN_ONLY_CACHE_MANIFEST "
                "and PROTECT_TRAIN_ONLY_CACHE_MANIFEST_SHA256"
            )
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(
                f"train-only cache manifest SHA256 mismatch: expected={expected}, actual={actual}"
            )
        manifest = json.loads(path.read_text())

        plan_text = os.environ.get("ALL178_CROSSFIT_PLAN", "")
        plan_expected = os.environ.get("ALL178_CROSSFIT_PLAN_SHA256", "").lower()
        if bool(plan_text) != bool(plan_expected):
            raise ValueError("All178 plan path and SHA256 must be provided together")
        if plan_text:
            plan_path = Path(plan_text)
            if not plan_path.is_file():
                raise FileNotFoundError(f"missing All178 cross-fit plan: {plan_path}")
            plan_actual = hashlib.sha256(plan_path.read_bytes()).hexdigest()
            if plan_actual != plan_expected:
                raise ValueError(
                    f"All178 cross-fit plan SHA256 mismatch: expected={plan_expected}, "
                    f"actual={plan_actual}"
                )
            plan = json.loads(plan_path.read_text())
            plan_type = plan.get("artifact_type")
            if plan_type not in {
                "all178_source_family_crossfit_plan",
                "hard_dataset_source_family_crossfit_plan",
            }:
                raise ValueError("unexpected All178/hard cross-fit plan artifact type")
            hard_selected_names = None
            hard = None
            if plan_type == "hard_dataset_source_family_crossfit_plan":
                hard_path = Path(os.environ.get("HARD_DATASET_WHITELIST", ""))
                hard_expected = os.environ.get(
                    "HARD_DATASET_WHITELIST_SHA256", ""
                ).lower()
                if not hard_path.is_file() or not hard_expected:
                    raise FileNotFoundError(
                        "hard cross-fit requires HARD_DATASET_WHITELIST and its SHA256"
                    )
                hard_actual = hashlib.sha256(hard_path.read_bytes()).hexdigest()
                if hard_actual != hard_expected or plan.get("hard_whitelist_sha256") != hard_actual:
                    raise ValueError("hard whitelist hash/plan binding mismatch")
                hard = json.loads(hard_path.read_text())
                expected_criterion = {
                    "column": "accuracy", "operator": "<",
                    "strict_boundary": True, "threshold": 0.9,
                }
                if hard.get("criterion") != expected_criterion:
                    raise ValueError("hard whitelist is not strict accuracy < 0.9")
                if plan.get("selection_evaluation_sha256") != hard.get(
                    "model_snapshot", {}
                ).get("evaluation_sha256"):
                    raise ValueError("hard plan/evaluation snapshot binding mismatch")
                hard_selected_names = set(hard.get("selected_dataset_names", ()))
            if int(plan.get("fold_count", -1)) != 4:
                raise ValueError("All178 cross-fit plan must bind exactly four folds")
            if plan.get("train_only_manifest_sha256") != actual:
                raise ValueError("All178 plan does not bind the active train-only manifest")
            if manifest.get("artifact_type") != "all178_train_only_cache_manifest":
                raise ValueError("unexpected All178 train-only manifest artifact type")
            if int(manifest.get("dataset_count", -1)) != 178:
                raise ValueError("All178 manifest must bind exactly 178 panel datasets")
            records = manifest.get("records", [])
            names = tuple(record.get("name") for record in records)
            if len(records) != 178 or len(set(names)) != 178:
                raise ValueError("All178 manifest record count/name uniqueness failure")
            if tuple(manifest.get("dataset_names", ())) != names:
                raise ValueError("All178 manifest dataset order differs from records")
            by_name = {record["name"]: record for record in records}
            for record in records:
                if record.get("source_keys_read") != ["X_train", "y_train"]:
                    raise ValueError(
                        f"All178 source-key firewall failed for {record.get('name')}"
                    )
                if not record.get("source_family"):
                    raise ValueError(f"missing source family for {record.get('name')}")
            final_all96 = os.environ.get(
                "HARD96_FINAL_ALL96", ""
            ).strip().lower() == "true"
            if final_all96:
                if hard_selected_names is None or not isinstance(hard, dict):
                    raise ValueError(
                        "HARD96_FINAL_ALL96 requires the hard-dataset cross-fit plan"
                    )
                selected_names = tuple(hard.get("selected_dataset_names", ()))
                if (
                    len(selected_names) != 96
                    or len(set(selected_names)) != 96
                    or set(selected_names) != hard_selected_names
                    or int(hard.get("selected_dataset_count", -1)) != 96
                    or int(hard.get("selected_train_eligible_count", -1)) != 96
                ):
                    raise ValueError("final Hard96 selection must bind 96 unique sources")
                if any(
                    name not in by_name or not by_name[name].get("train_eligible")
                    for name in selected_names
                ):
                    raise ValueError(
                        "final Hard96 selection contains an unknown or ineligible source"
                    )
                selected_families = {
                    by_name[name]["source_family"] for name in selected_names
                }
                if (
                    len(selected_families) != 73
                    or int(hard.get("selected_source_family_count", -1)) != 73
                ):
                    raise ValueError(
                        "final Hard96 selection must bind 73 source families"
                    )
                selection_scope: int | str = "all96"
            else:
                try:
                    fold_index = int(os.environ["ALL178_CROSSFIT_FOLD"])
                except (KeyError, ValueError) as error:
                    raise ValueError(
                        "ALL178_CROSSFIT_FOLD must be an integer in [0, 3]"
                    ) from error
                folds = {
                    int(fold["fold"]): fold for fold in plan.get("folds", [])
                }
                if set(folds) != {0, 1, 2, 3} or fold_index not in folds:
                    raise ValueError("All178 plan fold identifiers are invalid")
                fold = folds[fold_index]
                selected_names = tuple(fold.get("train_dataset_names", ()))
                if (
                    hard_selected_names is not None
                    and not set(selected_names) <= hard_selected_names
                ):
                    raise ValueError(
                        "hard fold selects a dataset outside the < 0.9 whitelist"
                    )
                selected_hash = hashlib.sha256(
                    ("\n".join(selected_names) + "\n").encode()
                ).hexdigest()
                if selected_hash != fold.get("train_names_sha256"):
                    raise ValueError("All178 fold train-name hash mismatch")
                if len(selected_names) != int(
                    fold.get("train_dataset_count", -1)
                ):
                    raise ValueError("All178 fold train dataset count mismatch")
                if any(
                    name not in by_name or not by_name[name].get("train_eligible")
                    for name in selected_names
                ):
                    raise ValueError(
                        "All178 fold selects an unknown or ineligible source"
                    )
                selected_families = {
                    by_name[name]["source_family"] for name in selected_names
                }
                held_families = set(fold.get("holdout_source_families", ()))
                if selected_families & held_families:
                    raise ValueError("All178 fold source-family firewall failed")
                if len(selected_families) != int(
                    fold.get("train_source_family_count", -1)
                ):
                    raise ValueError(
                        "All178 fold train source-family count mismatch"
                    )
                selection_scope = fold_index
            if "waveform_database_generator" in selected_names:
                allow_waveform = os.environ.get(
                    "HARD96_ALLOW_OBSERVED_LABEL_WAVEFORM", ""
                ).strip().lower() == "true"
                manifest_scope = manifest.get("observed_label_scope_amendment", {})
                hard_scope = (
                    hard.get("observed_label_scope_amendment", {})
                    if isinstance(hard, dict) else {}
                )
                required_scope = dict(OBSERVED_LABEL_SCOPE)
                if (
                    not allow_waveform
                    or any(manifest_scope.get(key) != value for key, value in required_scope.items())
                    or any(hard_scope.get(key) != value for key, value in required_scope.items())
                ):
                    raise ValueError(
                        "Waveform requires the explicit observed-label performance-"
                        "emulation scope in both manifest and whitelist"
                    )
            if self._protected_dataset_names_explicit:
                if tuple(self.protected_dataset_names) != selected_names:
                    raise ValueError("explicit dataset names differ from the selected training scope")
            else:
                self.protected_dataset_names = selected_names
            family_members: dict[str, list[str]] = {}
            for name in selected_names:
                family = by_name[name]["source_family"]
                family_members.setdefault(family, []).append(name)
            self._train_only_source_family_by_name = {
                name: by_name[name]["source_family"] for name in selected_names
            }
            self._train_only_family_members = {
                family: tuple(sorted(members))
                for family, members in sorted(family_members.items())
            }
            self._all178_fold = selection_scope
        else:
            if manifest.get("dataset_count") != 19:
                raise ValueError("train-only cache manifest must bind exactly 19 datasets")
            names = tuple(manifest.get("dataset_names", ()))
            if names != tuple(TRAIN_ONLY_PROTECTED_DATASET_NAMES):
                raise ValueError("train-only cache manifest dataset order/name mismatch")
            if "waveform_database_generator" in names:
                raise ValueError("Waveform is forbidden from clean train-only arms")
            self._train_only_source_family_by_name = {
                name: f"legacy::{name}" for name in names
            }
            self._train_only_family_members = {
                f"legacy::{name}": (name,) for name in names
            }
        self._train_only_manifest = manifest
        return manifest

    def _ensure_runtime4_prior(self):
        if self._runtime4_prior is not None:
            return self._runtime4_prior

        import hashlib

        from posterior_prior_v2_runtime import IndexedPosteriorPrior
        from posterior_prior_v3_mechanisms import (
            build_v3_mechanisms,
            make_tabiclv2_base_sampler,
        )

        manifest_path = Path(os.environ.get("PROTECT_RUNTIME4_MANIFEST", ""))
        if not manifest_path.is_file():
            raise FileNotFoundError(
                "PROTECT_RUNTIME4_MANIFEST must name the frozen row-free runtime manifest"
            )
        raw = manifest_path.read_bytes()
        actual_sha = hashlib.sha256(raw).hexdigest()
        expected_sha = os.environ.get("PROTECT_RUNTIME4_MANIFEST_SHA256", "").lower()
        if not expected_sha or actual_sha != expected_sha:
            raise ValueError(
                f"runtime4 manifest SHA256 mismatch: expected={expected_sha!r}, "
                f"actual={actual_sha}"
            )
        manifest = json.loads(raw)
        episode = manifest["episode"]
        if int(episode["max_features"]) != self.max_features:
            raise ValueError("runtime4 manifest max_features differs from trainer")
        if int(episode["max_classes"]) != self.max_classes:
            raise ValueError("runtime4 manifest max_classes differs from trainer")
        mechanisms = build_v3_mechanisms(
            manifest,
            base_task_sampler=make_tabiclv2_base_sampler(),
        )
        self._runtime4_prior = IndexedPosteriorPrior(
            manifest,
            mechanisms,
            experiment_seed=int(os.environ.get("PROTECT_RUNTIME4_EXPERIMENT_SEED", "20260720")),
            max_features=self.max_features,
            max_classes=self.max_classes,
            min_train_fraction=float(episode["minimum_train_fraction"]),
            max_train_fraction=float(episode["maximum_train_fraction"]),
            max_attempts=512,
        )
        return self._runtime4_prior

    @property
    def rng(self) -> np.random.Generator:
        if self._rng is None:
            self._rng = np.random.default_rng()
        return self._rng

    def _ensure_datasets(self) -> tuple[_ProtectedDataset, ...]:
        if self._datasets is not None:
            return self._datasets

        manifest_by_name = {}
        if self.train_only_source:
            manifest = self._ensure_train_only_manifest()
            manifest_by_name = {record["name"]: record for record in manifest["records"]}
        datasets: list[_ProtectedDataset] = []
        missing: list[str] = []
        for name in self.protected_dataset_names:
            safe_name = name.replace("/", "_")
            match: Optional[Path] = None
            for cache_dir in self.protected_cache_dirs:
                candidates = sorted(cache_dir.glob(f"{safe_name}__*.npz"))
                if candidates:
                    match = candidates[0]
                    break
            if match is None:
                missing.append(name)
                continue
            if self.train_only_source:
                record = manifest_by_name[name]
                expected_file_hash = record.get(
                    "sanitized_file_sha256", record.get("destination_file_sha256")
                )
                if not expected_file_hash:
                    raise ValueError(f"missing train-only cache hash for {name}")
                digest = hashlib.sha256(match.read_bytes()).hexdigest()
                if digest != expected_file_hash:
                    raise ValueError(f"train-only cache file hash mismatch for {name}")
            datasets.append(_load_protected_dataset(
                match, name, train_only=self.train_only_source
            ))

        if missing and self.train_only_source:
            raise FileNotFoundError(f"missing train-only protected caches: {missing}")
        if not datasets:
            cache_desc = ", ".join(str(path) for path in self.protected_cache_dirs)
            raise FileNotFoundError(f"No protected .npz caches found in: {cache_desc}")
        self._datasets = tuple(datasets)
        return self._datasets

    def _train_only_recipe(
        self,
        seq_len: int,
        train_size: int,
        *,
        logical_batch: int,
        task_slot: int,
    ):
        from deterministic_task_seeds import derive_seed

        manifest_hash = os.environ["PROTECT_TRAIN_ONLY_CACHE_MANIFEST_SHA256"]
        experiment_seed = int(os.environ.get(
            "PROTECT_RUNTIME4_EXPERIMENT_SEED", "20260720"
        ))
        rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
        common = dict(
            experiment_seed=experiment_seed,
            logical_batch=int(logical_batch),
            task_slot=int(task_slot),
            rank=rank,
            worker=0,
            manifest_hash=manifest_hash,
        )
        recipe_rng = np.random.default_rng(derive_seed(
            **common, stream="train_only_common_recipe"
        ))
        datasets = self._ensure_datasets()
        if self._train_only_family_members:
            family_names = tuple(sorted(self._train_only_family_members))
            family = family_names[int(recipe_rng.integers(0, len(family_names)))]
            members = self._train_only_family_members[family]
            source_name = members[int(recipe_rng.integers(0, len(members)))]
            dataset_by_name = {dataset.name: dataset for dataset in datasets}
            ds = dataset_by_name[source_name]
        else:
            ds = datasets[int(recipe_rng.integers(0, len(datasets)))]
        class_ids = np.asarray([
            index for index, rows in enumerate(ds.class_indices) if len(rows) > 0
        ], dtype=np.int64)
        if len(class_ids) < 2:
            raise ValueError(f"train-only dataset {ds.name} has fewer than two classes")
        if len(class_ids) > self.max_classes:
            class_probs = ds.class_probs[class_ids]
            class_probs /= class_probs.sum()
            class_ids = np.asarray(recipe_rng.choice(
                class_ids,
                size=self.max_classes,
                replace=False,
                p=class_probs,
            ), dtype=np.int64)
        num_classes = len(class_ids)
        train_size = self._adjust_train_size(train_size, seq_len, num_classes)
        probabilities = ds.class_probs[class_ids].astype(np.float64, copy=True)
        probabilities = 0.90 * probabilities / probabilities.sum() + 0.10 / num_classes
        minimum = 2
        remaining = max(seq_len - minimum * num_classes, 0)
        counts = recipe_rng.multinomial(remaining, probabilities) + minimum
        counts[0] += seq_len - int(counts.sum())
        if ds.n_features > self.max_features:
            columns = np.sort(recipe_rng.choice(
                ds.n_features, size=self.max_features, replace=False
            )).astype(np.int64)
        else:
            columns = np.arange(ds.n_features, dtype=np.int64)
        return ds, class_ids, counts.astype(np.int64), columns, train_size, common

    def _sample_train_only_one(
        self,
        seq_len: int,
        train_size: int,
        *,
        logical_batch: int,
        task_slot: int,
        synthetic: bool,
        nearcopy: bool = False,
        topology: bool = False,
        affine: bool = False,
        routed: bool = False,
        expanding: bool = False,
    ):
        from deterministic_task_seeds import derive_seed
        from indexed_global_rng import isolated_global_rng

        ds, class_ids, counts, columns, train_size, common = self._train_only_recipe(
            seq_len,
            train_size,
            logical_batch=logical_batch,
            task_slot=task_slot,
        )
        if sum(map(int, (nearcopy, topology, affine, routed, expanding))) > 1:
            raise ValueError("L1, L2, L3, L5 and L6 endpoints are mutually exclusive")
        if synthetic:
            generator_key = f"{ds.name}|{','.join(map(str, columns.tolist()))}"
            generator = self._train_only_generators.get(generator_key)
            if generator is None:
                generator = (
                    TrainOnlyExpandingPower2Affine(ds.x[:, columns], ds.y)
                    if expanding
                    else TrainOnlyRoutedPower2Affine(ds.x[:, columns], ds.y)
                    if routed
                    else TrainOnlyAffineSymmetry(ds.x[:, columns], ds.y)
                    if affine
                    else TrainOnlyTopologyPreservingULPWarp(ds.x[:, columns], ds.y)
                    if topology
                    else TrainOnlyPairedULPWarp(ds.x[:, columns], ds.y)
                    if nearcopy
                    else TrainOnlyKNNMixup(ds.x[:, columns], ds.y)
                )
                self._train_only_generators[generator_key] = generator
            if nearcopy or topology or affine or routed or expanding:
                sample = generator.sample(
                    seq_len,
                    seed=derive_seed(**common, stream="train_only_gt_rows"),
                    source_class_ids=class_ids,
                    class_counts=counts,
                )
                endpoint = "L6" if expanding else "L5" if routed else "L3" if affine else "L2" if topology else "L1"
                if sample.raw_source_collisions != 0:
                    raise RuntimeError(f"{endpoint} task contains a raw train collision")
                if sample.float32_source_collisions != 0:
                    raise RuntimeError(
                        f"{endpoint} task contains a float32 train collision"
                    )
                if topology or affine or routed or expanding:
                    if not sample.float32_equality_partition_preserved:
                        raise RuntimeError(f"{endpoint} task changed float32 equality topology")
                    if sample.rank_order_violations != 0:
                        raise RuntimeError(f"{endpoint} task changed selected-column rank order")
                if affine and not sample.source_interval_disjoint:
                    raise RuntimeError("L3 task did not leave the source feature interval")
            else:
                sample = generator.sample(
                    seq_len,
                    seed=derive_seed(**common, stream="train_only_k2_rows"),
                    max_classes=self.max_classes,
                    source_class_ids=class_ids,
                    class_counts=counts,
                    strict_no_exact_collision=True,
                )
                if sample.exact_train_collisions != 0:
                    raise RuntimeError("K2 strict task contains an exact train collision")
                if sample.float32_train_collisions != 0:
                    raise RuntimeError("K2 strict task contains a float32 train collision")
            x_np, y_np = sample.x, sample.y
            selected = np.isin(ds.y, class_ids)
            reference_x_np = ds.x[selected][:, columns]
            local_class = {
                int(source_class): index
                for index, source_class in enumerate(class_ids.tolist())
            }
            reference_y_np = np.asarray(
                [local_class[int(value)] for value in ds.y[selected]],
                dtype=np.int64,
            )
        else:
            row_rng = np.random.default_rng(derive_seed(
                **common, stream="train_only_gt_rows"
            ))
            row_indices = []
            labels = []
            for local_class, (source_class, count) in enumerate(zip(class_ids, counts)):
                pool = ds.class_indices[int(source_class)]
                chosen = row_rng.choice(pool, size=int(count), replace=len(pool) < int(count))
                row_indices.append(chosen)
                labels.append(np.full(int(count), local_class, dtype=np.int64))
            rows = np.concatenate(row_indices)
            y_np = np.concatenate(labels)
            permutation = row_rng.permutation(len(rows))
            x_np = ds.x[rows[permutation]][:, columns]
            y_np = y_np[permutation]
            self._record_train_only_task(ds.name, synthetic=False)
        x = torch.as_tensor(x_np, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(y_np, dtype=torch.float32, device=self.device)
        post_seed = derive_seed(**common, stream="train_only_common_postprocess")
        with isolated_global_rng(post_seed):
            if synthetic:
                result, model_audit = self._postprocess_train_only_preserve_schema(
                    x,
                    y,
                    seq_len,
                    train_size,
                    reference_x=torch.as_tensor(
                        reference_x_np, dtype=torch.float32, device=self.device
                    ),
                    reference_y=torch.as_tensor(
                        reference_y_np, dtype=torch.long, device=self.device
                    ),
                )
                self._record_train_only_task(
                    ds.name,
                    synthetic=True,
                    exact_collisions=model_audit["collisions_after_repair"],
                    novelty_repairs=(
                        sample.rows_translated
                        if affine
                        else sample.rows_transformed
                        if (routed or expanding)
                        else sample.rows_warped
                        if (nearcopy or topology)
                        else sample.novelty_repairs
                    ),
                    continuous_jitter_repairs=(
                        0
                        if (nearcopy or topology or affine or routed or expanding)
                        else sample.continuous_jitter_repairs
                    ),
                    model_space_collisions_before_repair=model_audit[
                        "collisions_before_repair"
                    ],
                    model_space_repairs=model_audit["ulp_repairs"],
                    nearcopy=(nearcopy or topology),
                    nearcopy_rows=(
                        sample.rows_warped if (nearcopy or topology) else 0
                    ),
                    nearcopy_max_ulp_steps=(
                        sample.ulp_steps
                        if topology
                        else sample.maximum_ulp_steps
                        if nearcopy
                        else 0
                    ),
                    nearcopy_normalized_delta_rmse=(
                        sample.normalized_delta_rmse
                        if (nearcopy or topology)
                        else 0.0
                    ),
                    affine=affine,
                    affine_rows=sample.rows_translated if affine else 0,
                    affine_translation=sample.translation if affine else 0.0,
                    affine_normalized_delta_rmse=(
                        sample.normalized_delta_rmse if affine else 0.0
                    ),
                    routed=(routed or expanding),
                    routed_rows=sample.rows_transformed if (routed or expanding) else 0,
                    routed_route=sample.route if (routed or expanding) else "",
                    routed_max_ulp_steps=sample.ulp_steps if (routed or expanding) else 0,
                    routed_normalized_delta_rmse=(
                        sample.normalized_delta_rmse if (routed or expanding) else 0.0
                    ),
                )
                return result
            return self._postprocess_train_only_preserve_schema(
                x, y, seq_len, train_size
            )

    def _sample_train_size(self, seq_len: int, mode: Optional[str] = None) -> int:
        if mode == "long_support":
            ratio = float(self.rng.uniform(0.76, 0.94))
            return int(np.clip(round(seq_len * ratio), 2, seq_len - 2))
        return TabICLv2ClassificationPrior.sample_train_size(
            self.min_train_size, self.max_train_size, seq_len
        )

    def _adjust_train_size(self, train_size: int, seq_len: int, num_classes: int) -> int:
        if seq_len < 2 * num_classes:
            return max(1, min(train_size, seq_len - 1))
        return int(np.clip(train_size, num_classes, seq_len - num_classes))

    def _class_counts(self, probs: np.ndarray, seq_len: int, num_classes: int) -> np.ndarray:
        min_per_class = 2
        if seq_len < min_per_class * num_classes:
            num_classes = max(2, seq_len // min_per_class)
            probs = probs[:num_classes] / max(float(probs[:num_classes].sum()), 1e-12)
        remaining = max(seq_len - min_per_class * num_classes, 0)
        counts = self.rng.multinomial(remaining, probs[:num_classes]) + min_per_class
        counts[0] += seq_len - int(counts.sum())
        return counts.astype(np.int64, copy=False)

    def _postprocess(self, x: Tensor, y: Tensor, seq_len: int, train_size: int):
        x = torch.nan_to_num(x, nan=0.0, posinf=1e9, neginf=-1e9).clamp(min=-1e6, max=1e9)
        post = TabICLv2ClassificationGenerator(
            seq_len=seq_len,
            train_size=train_size,
            num_features=int(x.shape[1]),
            max_features=self.max_features,
            max_classes=self.max_classes,
            device=self.device,
            extra_trees_filter=False,
            return_metadata=False,
        )
        x, y = post._postprocess(x, y)
        if not post._fix_split_coverage(x, y):
            raise ValueError("protected batch aux prior failed train/test class coverage")
        d = torch.tensor(x.shape[1], dtype=torch.long, device=self.device)
        return _pad_features(x, self.max_features), y.float(), d, int(train_size)

    def _postprocess_train_only_preserve_schema(
        self,
        x: Tensor,
        y: Tensor,
        seq_len: int,
        train_size: int,
        reference_x: Optional[Tensor] = None,
        reference_y: Optional[Tensor] = None,
    ):
        """Pair H/K without sample-dependent constant-column deletion."""

        x = torch.nan_to_num(x, nan=0.0, posinf=1e9, neginf=-1e9).clamp(
            min=-1e6, max=1e9
        )
        if (reference_x is None) != (reference_y is None):
            raise ValueError("reference_x and reference_y must be supplied together")
        if reference_x is not None:
            reference_x = torch.nan_to_num(
                reference_x, nan=0.0, posinf=1e9, neginf=-1e9
            ).clamp(min=-1e6, max=1e9)
        y = _ordinal_encode(y)
        permutation = torch.randperm(x.shape[0], device=x.device)
        x, y = x[permutation], y[permutation]
        # Constant recipe columns safely standardize to zero. Keeping them is
        # necessary for H and K to expose the exact same declared schema.
        clip_mean = x.mean(dim=0, keepdim=True)
        clip_std = torch.std(x, dim=0, keepdim=True, unbiased=False)
        clip_std = torch.nan_to_num(
            clip_std, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp_min(1e-6)
        lower, upper = clip_mean - 4.0 * clip_std, clip_mean + 4.0 * clip_std
        x = _remove_outliers(x)
        if reference_x is not None:
            reference_x = reference_x.clamp(lower, upper)
            standard_mean = x.mean(dim=0, keepdim=True)
            standard_std = torch.std(x, dim=0, keepdim=True, unbiased=False)
            standard_std = torch.nan_to_num(
                standard_std, nan=0.0, posinf=0.0, neginf=0.0
            ).clamp_min(1e-6)
            reference_x = (reference_x - standard_mean) / standard_std
        x = _standardize(x)
        column_permutation = torch.randperm(x.shape[1], device=x.device)
        x = x[:, column_permutation]
        if reference_x is not None:
            reference_x = reference_x[:, column_permutation]
        class_permutation = torch.randperm(int(y.max().item()) + 1, device=x.device)
        y = class_permutation[y.long()].float()
        model_audit = None
        if reference_x is not None:
            reference_y = class_permutation[reference_y.long()].float()
            x, before, repairs, after = _repair_model_space_feature_collisions(
                x, reference_x
            )
            model_audit = {
                "collisions_before_repair": before,
                "ulp_repairs": repairs,
                "collisions_after_repair": after,
            }
        post = TabICLv2ClassificationGenerator(
            seq_len=seq_len,
            train_size=train_size,
            num_features=int(x.shape[1]),
            max_features=self.max_features,
            max_classes=self.max_classes,
            device=self.device,
            extra_trees_filter=False,
            return_metadata=False,
        )
        if not post._fix_split_coverage(x, y):
            raise ValueError("train-only H/K failed train/test class coverage")
        if not torch.isfinite(x).all():
            raise ValueError("train-only H/K preprocessing produced nonfinite features")
        d = torch.tensor(x.shape[1], dtype=torch.long, device=self.device)
        result = _pad_features(x.float(), self.max_features), y, d, int(train_size)
        return (result, model_audit) if model_audit is not None else result

    def _sample_real_gt_one(self, seq_len: int, train_size: int):
        datasets = self._ensure_datasets()
        ds = datasets[int(self.rng.integers(0, len(datasets)))]
        available = np.asarray([idx.size > 0 for idx in ds.class_indices], dtype=bool)
        class_ids = np.flatnonzero(available)
        if class_ids.size < 2:
            raise ValueError(f"protected dataset {ds.name} has fewer than two classes")

        if class_ids.size > self.max_classes:
            probs = ds.class_probs[class_ids]
            probs = probs / max(float(probs.sum()), 1e-12)
            class_ids = self.rng.choice(class_ids, size=self.max_classes, replace=False, p=probs)
            class_ids = np.asarray(class_ids, dtype=np.int64)
        num_classes = int(class_ids.size)
        train_size = self._adjust_train_size(train_size, seq_len, num_classes)

        probs = ds.class_probs[class_ids].astype(np.float64, copy=True)
        probs = 0.90 * probs / max(float(probs.sum()), 1e-12) + 0.10 / num_classes
        counts = self._class_counts(probs, seq_len, num_classes)
        row_indices: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        for local_cls, (orig_cls, count) in enumerate(zip(class_ids.tolist(), counts.tolist())):
            pool = ds.class_indices[int(orig_cls)]
            chosen = self.rng.choice(pool, size=int(count), replace=pool.size < int(count))
            row_indices.append(chosen.astype(np.int64, copy=False))
            labels.append(np.full(int(count), local_cls, dtype=np.int64))
        rows = np.concatenate(row_indices)
        y_np = np.concatenate(labels)
        perm = self.rng.permutation(rows.shape[0])
        rows, y_np = rows[perm], y_np[perm]

        x_np = ds.x[rows]
        if x_np.shape[1] > self.max_features:
            cols = self.rng.choice(x_np.shape[1], size=self.max_features, replace=False)
            x_np = x_np[:, cols]
        x = torch.tensor(x_np, dtype=torch.float32, device=self.device)
        y = torch.tensor(y_np, dtype=torch.float32, device=self.device)
        return self._postprocess(x, y, seq_len, train_size)

    def _sample_profile_aug_one(self, seq_len: int, train_size: int):
        datasets = self._ensure_datasets()
        ds = datasets[int(self.rng.integers(0, len(datasets)))]
        num_classes = int(np.clip(ds.n_classes, 2, self.max_classes))
        train_size = self._adjust_train_size(train_size, seq_len, num_classes)
        probs = ds.class_probs[:num_classes].astype(np.float64, copy=True)
        probs = probs / max(float(probs.sum()), 1e-12)
        probs = 0.82 * probs + 0.18 / num_classes
        counts = self._class_counts(probs, seq_len, num_classes)
        y_np = np.concatenate([np.full(int(count), cls, dtype=np.int64) for cls, count in enumerate(counts)])
        self.rng.shuffle(y_np)
        y = torch.tensor(y_np, dtype=torch.float32, device=self.device)
        y_long = y.long()

        feature_jitter = float(self.rng.uniform(0.70, 1.25))
        num_features = int(np.clip(round(ds.n_features * feature_jitter), self.min_features, self.max_features))
        num_features = max(2, num_features)
        zero_ratio = float(np.clip(ds.near_zero_ratio + self.rng.normal(0.0, 0.08), 0.0, 0.92))
        integer_like = float(np.clip(ds.integer_like_ratio + self.rng.normal(0.0, 0.08), 0.0, 1.0))
        tail_scale = float(np.clip(ds.abs_q99 / max(ds.abs_q95, 1e-6), 1.0, 1_000.0))

        if zero_ratio > 0.30 or integer_like > 0.65:
            density = float(np.clip(1.0 - zero_ratio, 0.015, 0.55))
            x = torch.zeros(seq_len, num_features, dtype=torch.float32, device=self.device)
            class_bias = torch.randn(num_classes, num_features, dtype=torch.float32, device=self.device) * 0.7
            for col in range(num_features):
                col_density = float(np.clip(self.rng.lognormal(math.log(density), 0.55), 0.003, 0.80))
                active = torch.rand(seq_len, device=self.device) < col_density
                if integer_like > 0.55:
                    if self.rng.random() < 0.55:
                        rate = math.exp(float(self.rng.uniform(math.log(0.6), math.log(80.0))))
                        values = torch.poisson(torch.full((seq_len,), rate, dtype=torch.float32, device=self.device))
                    else:
                        levels = int(self.rng.integers(2, 32))
                        values = torch.randint(0, levels, (seq_len,), dtype=torch.int64, device=self.device).float()
                else:
                    values = torch.randn(seq_len, dtype=torch.float32, device=self.device)
                values = values + class_bias[y_long, col]
                x[:, col] = torch.where(active, values, torch.zeros_like(values))

            signal_cols = torch.randperm(num_features, device=self.device)[: max(2, min(num_features, num_classes + 4))]
            for cls in range(num_classes):
                rows = (y_long == cls).nonzero(as_tuple=False).flatten()
                if rows.numel() == 0:
                    continue
                cls_cols = signal_cols[cls % signal_cols.numel() :: max(1, num_classes)]
                if cls_cols.numel() == 0:
                    cls_cols = signal_cols[:1]
                lift = float(math.exp(self.rng.uniform(math.log(1.5), math.log(16.0))))
                x[rows[:, None], cls_cols[None, :]] = x[rows[:, None], cls_cols[None, :]] + lift
        else:
            latent_dim = int(self.rng.integers(2, min(10, num_features) + 1))
            centers = torch.randn(num_classes, latent_dim, dtype=torch.float32, device=self.device) * float(
                self.rng.uniform(0.7, 1.9)
            )
            latent = centers[y_long] + torch.randn(seq_len, latent_dim, dtype=torch.float32, device=self.device) * float(
                self.rng.uniform(0.45, 1.35)
            )
            proj = torch.randn(latent_dim, num_features, dtype=torch.float32, device=self.device)
            x = latent @ proj
            x = x + torch.randn(seq_len, num_features, dtype=torch.float32, device=self.device) * float(
                self.rng.uniform(0.18, 0.90)
            )
            if integer_like > 0.20:
                disc_cols = torch.randperm(num_features, device=self.device)[: max(1, int(num_features * integer_like * 0.35))]
                levels = float(self.rng.integers(3, 18))
                vals = x[:, disc_cols]
                vals = (vals - vals.mean(dim=0, keepdim=True)) / vals.std(dim=0, keepdim=True).clamp_min(1e-3)
                x[:, disc_cols] = torch.round(vals * 1.5 + levels / 2.0).clamp(0.0, levels - 1.0)
            if zero_ratio > 0.03:
                zero_mask = torch.rand(seq_len, num_features, device=self.device) < zero_ratio
                x = torch.where(zero_mask, torch.zeros_like(x), x)

        outlier_prob = float(np.clip((tail_scale - 1.0) / 400.0, 0.0005, 0.02))
        outlier_mask = torch.rand(seq_len, num_features, device=self.device) < outlier_prob
        outlier_count = int(outlier_mask.sum().item())
        if outlier_count > 0:
            x[outlier_mask] = x[outlier_mask] + torch.randn(outlier_count, device=self.device) * float(
                self.rng.uniform(8.0, 80.0)
            )
        return self._postprocess(x, y, seq_len, train_size)

    def _sample_p05_hard_one(self, seq_len: int, train_size: Optional[int] = None):
        mode = str(
            self.rng.choice(
                ["sparse_binary", "near_zero", "extreme_imbalance", "long_support"],
                p=[0.30, 0.25, 0.25, 0.20],
            )
        )
        train_size = int(train_size) if train_size is not None else self._sample_train_size(seq_len, mode=mode)
        max_classes = max(2, int(self.max_classes))
        if mode == "extreme_imbalance":
            num_classes = int(self.rng.integers(2, min(max_classes, 5) + 1))
            majority = float(self.rng.uniform(0.88, 0.985))
            rest = self.rng.dirichlet(np.full(num_classes - 1, 0.35)) * (1.0 - majority)
            probs = np.concatenate([[majority], rest])
        else:
            num_classes = 2 if mode in {"sparse_binary", "near_zero"} else int(self.rng.integers(2, min(max_classes, 6) + 1))
            alpha = 0.45 if mode == "long_support" else 1.5
            probs = self.rng.dirichlet(np.full(num_classes, alpha, dtype=np.float64))
            if mode == "sparse_binary":
                majority = float(self.rng.uniform(0.55, 0.90))
                probs = np.array([majority, 1.0 - majority], dtype=np.float64)
        train_size = self._adjust_train_size(train_size, seq_len, num_classes)
        counts = self._class_counts(probs / probs.sum(), seq_len, num_classes)
        y_np = np.concatenate([np.full(int(count), cls, dtype=np.int64) for cls, count in enumerate(counts)])
        self.rng.shuffle(y_np)
        y = torch.tensor(y_np, dtype=torch.float32, device=self.device)
        y_long = y.long()

        if mode == "near_zero":
            num_features = int(self.rng.integers(max(20, self.min_features), min(self.max_features, 100) + 1))
            x = torch.randn(seq_len, num_features, dtype=torch.float32, device=self.device) * float(self.rng.uniform(1e-5, 2e-3))
            active_cols = torch.randperm(num_features, device=self.device)[: max(2, int(num_features * self.rng.uniform(0.04, 0.14)))]
            for cls in range(num_classes):
                rows = (y_long == cls).nonzero(as_tuple=False).flatten()
                if rows.numel() == 0:
                    continue
                shift = float(self.rng.choice([-1.0, 1.0]) * self.rng.uniform(0.01, 0.12) * (cls + 1))
                x[rows[:, None], active_cols[None, :]] = x[rows[:, None], active_cols[None, :]] + shift
            zero_mask = torch.rand(seq_len, num_features, device=self.device) < float(self.rng.uniform(0.65, 0.96))
            x = torch.where(zero_mask, torch.zeros_like(x), x)
        elif mode == "sparse_binary":
            num_features = int(self.rng.integers(max(24, self.min_features), min(self.max_features, 100) + 1))
            base_density = float(self.rng.uniform(0.005, 0.08))
            x = (torch.rand(seq_len, num_features, device=self.device) < base_density).float()
            signal_cols = torch.randperm(num_features, device=self.device)[: max(3, int(num_features * self.rng.uniform(0.05, 0.18)))]
            for cls in range(num_classes):
                rows = (y_long == cls).nonzero(as_tuple=False).flatten()
                if rows.numel() == 0:
                    continue
                p = float(self.rng.uniform(0.04, 0.22) if cls == 0 else self.rng.uniform(0.16, 0.48))
                x[rows[:, None], signal_cols[None, :]] = (
                    torch.rand(rows.numel(), signal_cols.numel(), device=self.device) < p
                ).float()
        elif mode == "extreme_imbalance":
            num_features = int(self.rng.integers(max(6, self.min_features), min(self.max_features, 48) + 1))
            x = torch.zeros(seq_len, num_features, dtype=torch.float32, device=self.device)
            for col in range(num_features):
                if self.rng.random() < 0.65:
                    rate = math.exp(float(self.rng.uniform(math.log(0.3), math.log(200.0))))
                    values = torch.poisson(torch.full((seq_len,), rate, dtype=torch.float32, device=self.device))
                else:
                    values = torch.randn(seq_len, dtype=torch.float32, device=self.device)
                zero_mask = torch.rand(seq_len, device=self.device) < float(self.rng.uniform(0.10, 0.70))
                x[:, col] = torch.where(zero_mask, torch.zeros_like(values), values)
            for cls in range(1, num_classes):
                rows = (y_long == cls).nonzero(as_tuple=False).flatten()
                if rows.numel() == 0:
                    continue
                cols = torch.randperm(num_features, device=self.device)[: max(1, min(4, num_features))]
                lift = float(math.exp(self.rng.uniform(math.log(4.0), math.log(256.0))))
                x[rows[:, None], cols[None, :]] = x[rows[:, None], cols[None, :]] + lift
        else:
            num_features = int(self.rng.integers(max(16, self.min_features), min(self.max_features, 96) + 1))
            latent_dim = int(self.rng.integers(2, min(8, num_features) + 1))
            centers = torch.randn(num_classes, latent_dim, dtype=torch.float32, device=self.device) * float(
                self.rng.uniform(0.45, 1.20)
            )
            latent = centers[y_long] + torch.randn(seq_len, latent_dim, dtype=torch.float32, device=self.device) * float(
                self.rng.uniform(0.65, 1.45)
            )
            proj = torch.randn(latent_dim, num_features, dtype=torch.float32, device=self.device)
            x = latent @ proj + torch.randn(seq_len, num_features, dtype=torch.float32, device=self.device) * float(
                self.rng.uniform(0.30, 1.10)
            )
            zero_cols = torch.randperm(num_features, device=self.device)[: max(1, int(num_features * self.rng.uniform(0.08, 0.25)))]
            zero_mask = torch.rand(seq_len, zero_cols.numel(), device=self.device) < float(self.rng.uniform(0.20, 0.65))
            x[:, zero_cols] = torch.where(zero_mask, torch.zeros_like(x[:, zero_cols]), x[:, zero_cols])

        return self._postprocess(x, y, seq_len, train_size)

    def _sample_runtime4_one(
        self,
        seq_len: int,
        train_size: int,
        *,
        logical_batch: int,
        task_slot: int,
    ):
        from deterministic_task_seeds import derive_seed
        from indexed_global_rng import isolated_global_rng

        prior = self._ensure_runtime4_prior()
        rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
        isolation_seed = derive_seed(
            experiment_seed=prior.experiment_seed,
            logical_batch=int(logical_batch),
            task_slot=int(task_slot),
            stream="runtime4_candidate_isolation",
            rank=rank,
            worker=0,
            manifest_hash=prior.manifest_hash,
        )
        with isolated_global_rng(isolation_seed):
            task = prior.sample_task(
                logical_batch=int(logical_batch),
                task_slot=int(task_slot),
                rows=int(seq_len),
                train_size=int(train_size),
                rank=rank,
                worker=0,
            )
        x = torch.as_tensor(task.x, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(task.y, dtype=torch.float32, device=self.device)
        keep = [column for column in range(x.shape[1]) if torch.unique(x[:, column]).numel() > 1]
        if not keep:
            raise ValueError("runtime4 task has no nonconstant features")
        x = _standardize(_remove_outliers(x[:, keep]))
        if not torch.isfinite(x).all():
            raise ValueError("runtime4 task preprocessing produced nonfinite values")
        d = torch.tensor(x.shape[1], dtype=torch.long, device=self.device)
        return _pad_features(x, self.max_features), y, d, int(train_size)

    def _sample_one(
        self,
        seq_len: int,
        train_size: Optional[int] = None,
        *,
        logical_batch: int = 0,
        task_slot: int = 0,
    ):
        last_value_error: Optional[ValueError] = None
        for _attempt in range(64):
            sample_train_size = int(train_size) if train_size is not None else self._sample_train_size(seq_len)
            try:
                if self.source in {"gt", "protected_gt", "real_gt", "protected_real_gt"}:
                    return self._sample_real_gt_one(seq_len, sample_train_size)
                if self.source in TRAIN_ONLY_GT_SOURCES:
                    return self._sample_train_only_one(
                        seq_len,
                        sample_train_size,
                        logical_batch=logical_batch,
                        task_slot=task_slot,
                        synthetic=False,
                    )
                if self.source in TRAIN_ONLY_K2_SOURCES:
                    return self._sample_train_only_one(
                        seq_len,
                        sample_train_size,
                        logical_batch=logical_batch,
                        task_slot=task_slot,
                        synthetic=True,
                    )
                if self.source in TRAIN_ONLY_L1_SOURCES:
                    return self._sample_train_only_one(
                        seq_len,
                        sample_train_size,
                        logical_batch=logical_batch,
                        task_slot=task_slot,
                        synthetic=True,
                        nearcopy=True,
                    )
                if self.source in TRAIN_ONLY_L2_SOURCES:
                    return self._sample_train_only_one(
                        seq_len,
                        sample_train_size,
                        logical_batch=logical_batch,
                        task_slot=task_slot,
                        synthetic=True,
                        topology=True,
                    )
                if self.source in TRAIN_ONLY_L3_SOURCES:
                    return self._sample_train_only_one(
                        seq_len,
                        sample_train_size,
                        logical_batch=logical_batch,
                        task_slot=task_slot,
                        synthetic=True,
                        affine=True,
                    )
                if self.source in TRAIN_ONLY_L5_SOURCES:
                    return self._sample_train_only_one(
                        seq_len,
                        sample_train_size,
                        logical_batch=logical_batch,
                        task_slot=task_slot,
                        synthetic=True,
                        routed=True,
                    )
                if self.source in TRAIN_ONLY_L6_SOURCES:
                    return self._sample_train_only_one(
                        seq_len,
                        sample_train_size,
                        logical_batch=logical_batch,
                        task_slot=task_slot,
                        synthetic=True,
                        expanding=True,
                    )
                if self.source in {"aug", "protected_aug", "profile_aug", "protected_like"}:
                    return self._sample_profile_aug_one(seq_len, sample_train_size)
                if self.source in {"p05_hard", "hard", "hard_profile", "protected_hard"}:
                    return self._sample_p05_hard_one(seq_len, sample_train_size)
                if self.source in {"runtime4", "protect_runtime4", "rowfree_runtime4"}:
                    return self._sample_runtime4_one(
                        seq_len,
                        sample_train_size,
                        logical_batch=logical_batch,
                        task_slot=task_slot,
                    )
                raise ValueError(
                    "Unknown protected_batch_mix_source "
                    f"{self.source!r}; expected gt, train_gt, train_knn_k2, "
                    "train_paired_ulp_warp, train_topology_ulp_warp, "
                    "train_affine_symmetry, train_routed_power2_affine, "
                    "train_expanding_power2_affine, "
                    "aug, p05_hard, or runtime4."
                )
            except ValueError as error:
                last_value_error = error
                continue
        detail = (
            f"last_value_error={type(last_value_error).__name__}: {last_value_error}"
            if last_value_error is not None
            else "last_value_error=<none>"
        )
        raise RuntimeError(
            "Could not sample a valid protected batch aux task "
            f"from source={self.source!r} logical_batch={logical_batch} "
            f"task_slot={task_slot} seq_len={seq_len} train_size={train_size}; "
            f"{detail}"
        ) from last_value_error

    def get_batch(
        self,
        batch_size: Optional[int] = None,
        *,
        seq_len: Optional[int] = None,
        train_size: Optional[int] = None,
        logical_batch: int = 0,
    ):
        batch_size = int(batch_size or self.batch_size)
        seq_len = int(seq_len or self.max_seq_len)
        x_list: list[Tensor] = []
        y_list: list[Tensor] = []
        d_list: list[Tensor] = []
        train_sizes: list[int] = []
        for task_slot in range(batch_size):
            x, y, d, sample_train_size = self._sample_one(
                seq_len,
                train_size=train_size,
                logical_batch=int(logical_batch),
                task_slot=int(task_slot),
            )
            x_list.append(x)
            y_list.append(y)
            d_list.append(d)
            train_sizes.append(int(sample_train_size))

        x_batch = torch.stack(x_list).to(self.device)
        y_batch = torch.stack(y_list).to(self.device)
        d_batch = torch.stack(d_list).to(self.device)
        seq_lens_t = torch.full((batch_size,), seq_len, dtype=torch.long, device=self.device)
        train_sizes_t = torch.tensor(train_sizes, dtype=torch.long, device=self.device)
        return x_batch, y_batch, d_batch, seq_lens_t, train_sizes_t

    def __repr__(self) -> str:
        loaded = len(self._datasets) if self._datasets is not None else "lazy"
        return (
            "ProtectedBatchAuxPrior("
            f"source={self.source!r}, datasets={loaded}, max_features={self.max_features}, "
            f"max_classes={self.max_classes})"
        )


class TabICLv2ProtectedBatchMixPrior:
    """Batch-internal protected-style mixture over the default TabICLv2 classifier prior."""

    def __init__(
        self,
        *,
        protected_batch_mix_ratio: float,
        protected_batch_mix_source: str = "gt",
        protected_batch_mix_cache_dirs: Optional[str | Sequence[str]] = None,
        protected_batch_mix_dataset_names: Optional[str | Sequence[str]] = None,
        **prior_kwargs,
    ):
        self.protected_batch_mix_ratio = float(np.clip(protected_batch_mix_ratio, 0.0, 1.0))
        self.protected_batch_mix_source = protected_batch_mix_source
        self.experiment_seed = int(
            os.environ.get("PROTECT_RUNTIME4_EXPERIMENT_SEED", "20260720")
        )
        self.replacement_enabled = os.environ.get(
            "PROTECT_RUNTIME4_REPLACEMENT_ENABLED", "true"
        ).lower() in {"1", "true", "yes"}
        self._logical_batch = int(os.environ.get("PROTECT_RUNTIME4_LOGICAL_BATCH_START", "0"))
        self.exact_global_mix_ratio = os.environ.get(
            "PROTECTED_BATCH_MIX_EXACT_GLOBAL_RATIO", ""
        ).strip().lower() == "true"
        self._mix_tasks = 0
        self._mix_total_tasks = 0

        dense_prior_kwargs = dict(prior_kwargs)
        dense_prior_kwargs["seq_len_per_gp"] = False
        self.base_prior = TabICLv2ClassificationPrior(**dense_prior_kwargs)
        self.aux_prior = ProtectedBatchAuxPrior(
            source=protected_batch_mix_source,
            protected_cache_dirs=protected_batch_mix_cache_dirs,
            protected_dataset_names=protected_batch_mix_dataset_names,
            **{
                key: value
                for key, value in dense_prior_kwargs.items()
                if key
                in {
                    "batch_size",
                    "min_features",
                    "max_features",
                    "max_classes",
                    "min_seq_len",
                    "max_seq_len",
                    "log_seq_len",
                    "min_train_size",
                    "max_train_size",
                    "replay_small",
                    "device",
                }
            },
        )

    def get_batch(self, batch_size: Optional[int] = None):
        from deterministic_task_seeds import TaskCoordinates

        logical_batch = self._logical_batch
        self._logical_batch += 1
        x, y, d, seq_lens, train_sizes = self.base_prior.get_batch(batch_size=batch_size)
        batch_size = int(batch_size or x.shape[0])
        if self.protected_batch_mix_ratio <= 0.0 or batch_size <= 0:
            return x, y, d, seq_lens, train_sizes

        rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
        if self.exact_global_mix_ratio:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            mix_count = _exact_global_mix_count(
                batch_size=batch_size,
                ratio=self.protected_batch_mix_ratio,
                logical_batch=logical_batch,
                rank=rank,
                world_size=world_size,
            )
        else:
            mix_count = int(round(batch_size * self.protected_batch_mix_ratio))
            mix_count = int(np.clip(mix_count, 1, batch_size))
        self._mix_tasks += mix_count
        self._mix_total_tasks += batch_size
        audit_every = int(os.environ.get("PROTECTED_BATCH_MIX_AUDIT_EVERY", "0"))
        if audit_every > 0 and (logical_batch + 1) % audit_every == 0:
            print(
                "[protected-batch-mix-audit] "
                + json.dumps(
                    {
                        "exact_global_ratio": self.exact_global_mix_ratio,
                        "logical_batches": logical_batch + 1,
                        "protected_tasks": self._mix_tasks,
                        "total_tasks": self._mix_total_tasks,
                        "realized_ratio": self._mix_tasks / self._mix_total_tasks,
                        "rank": rank,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
        mask_rng = TaskCoordinates(
            self.experiment_seed,
            logical_batch,
            0,
            rank=rank,
            worker=0,
        ).rng("replacement_mask", manifest_hash="shared")
        idx_np = np.sort(mask_rng.choice(batch_size, size=mix_count, replace=False))
        idx = torch.as_tensor(idx_np, dtype=torch.long, device=x.device)
        if not self.replacement_enabled:
            return x, y, d, seq_lens, train_sizes

        seq_len = int(x.shape[1])
        train_size = int(train_sizes[0].item())
        aux_x, aux_y, aux_d, aux_seq_lens, aux_train_sizes = self.aux_prior.get_batch(
            batch_size=mix_count,
            seq_len=seq_len,
            train_size=train_size,
            logical_batch=logical_batch,
        )

        x[idx] = aux_x.to(device=x.device, dtype=x.dtype)
        y[idx] = aux_y.to(device=y.device, dtype=y.dtype)
        d[idx] = aux_d.to(device=d.device, dtype=d.dtype)
        seq_lens[idx] = aux_seq_lens.to(device=seq_lens.device, dtype=seq_lens.dtype)
        train_sizes[idx] = aux_train_sizes.to(device=train_sizes.device, dtype=train_sizes.dtype)
        return x, y, d, seq_lens, train_sizes

    def __repr__(self) -> str:
        return (
            "TabICLv2ProtectedBatchMixPrior("
            f"protected_batch_mix_ratio={self.protected_batch_mix_ratio}, "
            f"protected_batch_mix_source={self.protected_batch_mix_source!r}, "
            f"replacement_enabled={self.replacement_enabled}, "
            f"exact_global_mix_ratio={self.exact_global_mix_ratio}, "
            f"experiment_seed={self.experiment_seed}, "
            f"base_prior={self.base_prior!r}, aux_prior={self.aux_prior!r})"
        )
