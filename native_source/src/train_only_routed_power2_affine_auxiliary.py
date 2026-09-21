#!/usr/bin/env python3
"""Paired train-only factor-two-or-affine preprocessing-orbit emulator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from train_only_affine_symmetry_auxiliary import TrainOnlyAffineSymmetry
from train_only_paired_ulp_warp_auxiliary import _advance_float32


ULP_STEPS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


def _row_keys(x: np.ndarray) -> set[bytes]:
    return {np.ascontiguousarray(row).tobytes() for row in np.ascontiguousarray(x)}


def _robust_scale(x: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    q25, q75 = np.quantile(x64, [0.25, 0.75], axis=0)
    scale = q75 - q25
    scale = np.where(scale > 1e-8, scale, np.std(x64, axis=0))
    return np.where(scale > 1e-8, scale, 1.0)


def _topology_preserved(source: np.ndarray, mapped: np.ndarray) -> bool:
    for column in range(source.shape[1]):
        source_unique, source_inverse = np.unique(source[:, column], return_inverse=True)
        mapped_unique, mapped_inverse = np.unique(mapped[:, column], return_inverse=True)
        if len(source_unique) != len(mapped_unique) or not np.array_equal(source_inverse, mapped_inverse):
            return False
        if len(mapped_unique) > 1 and np.any(np.diff(mapped_unique.astype(np.float64)) <= 0):
            return False
    return True


def _choose_factor2_mapping(source: np.ndarray) -> dict | None:
    source32 = np.asarray(source, dtype=np.float32)
    base = np.asarray(source32 * np.float32(2.0), dtype=np.float32)
    if not np.isfinite(base).all() or np.any(base <= -1e6) or np.any(base >= 1e9):
        return None
    source_keys = _row_keys(source32)
    unique_source_rows = len(source_keys)
    scale = _robust_scale(source32)
    for steps in ULP_STEPS:
        candidates = [(None, 0)] if steps == 0 else [
            (column, direction)
            for column in range(source32.shape[1])
            for direction in (1, -1)
        ]
        valid = []
        for column, direction in candidates:
            mapped = base.copy()
            if column is not None:
                mapped[:, column] = np.asarray(
                    _advance_float32(mapped[:, column], float(direction), steps),
                    dtype=np.float32,
                )
            if not np.isfinite(mapped).all() or np.any(mapped <= -1e6) or np.any(mapped >= 1e9):
                continue
            keys = [np.ascontiguousarray(row).tobytes() for row in mapped]
            if any(key in source_keys for key in keys) or len(set(keys)) != unique_source_rows:
                continue
            if not _topology_preserved(source32, mapped):
                continue
            normalized = (mapped.astype(np.float64) - source32.astype(np.float64)) / scale
            raw_rmse = float(np.sqrt(np.mean(np.square(normalized))))
            if raw_rmse < 0.01:
                continue
            valid.append((raw_rmse, source32.shape[1] if column is None else column, -direction, column, direction))
        if valid:
            valid.sort(key=lambda item: item[:3])
            raw_rmse, _, _, column, direction = valid[0]
            return {
                "factor": 2.0, "ulp_steps": int(steps),
                "selected_column": column, "direction": int(direction),
                "raw_normalized_delta_rmse": raw_rmse,
            }
    return None


def _apply_factor2(x: np.ndarray, mapping: dict) -> np.ndarray:
    mapped = np.asarray(np.asarray(x, dtype=np.float32) * np.float32(2.0), dtype=np.float32)
    column = mapping["selected_column"]
    if column is not None:
        mapped[:, column] = np.asarray(
            _advance_float32(mapped[:, column], float(mapping["direction"]), int(mapping["ulp_steps"])),
            dtype=np.float32,
        )
    return mapped


@dataclass(frozen=True)
class RoutedPower2AffineSample:
    x: np.ndarray
    y: np.ndarray
    source_class_ids: np.ndarray
    class_counts: np.ndarray
    route: str
    raw_source_collisions: int
    float32_source_collisions: int
    rows_transformed: int
    selected_column: int | None
    direction: int
    ulp_steps: int
    translation: float
    normalized_delta_rmse: float
    float32_equality_partition_preserved: bool
    rank_order_violations: int


class TrainOnlyRoutedPower2Affine:
    """Use factor-two topology mapping when admissible, else L3 affine."""

    def __init__(self, x_train: np.ndarray, y_train: np.ndarray):
        self.l3 = TrainOnlyAffineSymmetry(x_train, y_train)
        self.x = self.l3.x
        self.x32 = self.l3.x32
        self.y = self.l3.y
        self.classes = self.l3.classes
        self.class_indices = self.l3.class_indices
        self.raw_source_rows = _row_keys(self.x)
        self.float32_source_rows = _row_keys(self.x32)
        self.factor2_mapping = _choose_factor2_mapping(self.x32)
        self.route = "factor2_ulp" if self.factor2_mapping is not None else "l3_affine"

    def sample(self, n: int, *, seed: int, source_class_ids: np.ndarray, class_counts: np.ndarray) -> RoutedPower2AffineSample:
        if self.route == "l3_affine":
            sample = self.l3.sample(
                n, seed=seed, source_class_ids=source_class_ids, class_counts=class_counts
            )
            return RoutedPower2AffineSample(
                x=sample.x, y=sample.y,
                source_class_ids=sample.source_class_ids,
                class_counts=sample.class_counts, route=self.route,
                raw_source_collisions=sample.raw_source_collisions,
                float32_source_collisions=sample.float32_source_collisions,
                rows_transformed=sample.rows_translated,
                selected_column=sample.selected_column, direction=0, ulp_steps=0,
                translation=sample.translation,
                normalized_delta_rmse=sample.normalized_delta_rmse,
                float32_equality_partition_preserved=sample.float32_equality_partition_preserved,
                rank_order_violations=sample.rank_order_violations,
            )
        class_ids = np.asarray(source_class_ids, dtype=np.int64).reshape(-1)
        counts = np.asarray(class_counts, dtype=np.int64).reshape(-1)
        if (
            n < 4 or len(class_ids) < 2 or len(class_ids) != len(counts)
            or len(np.unique(class_ids)) != len(class_ids)
            or not np.isin(class_ids, self.classes).all()
            or np.any(counts < 1) or int(counts.sum()) != n
        ):
            raise ValueError("invalid explicit paired task recipe")
        rng = np.random.default_rng(seed)
        rows, labels = [], []
        for local_class, (source_class, count) in enumerate(zip(class_ids, counts)):
            pool = self.class_indices[int(source_class)]
            rows.append(rng.choice(pool, size=int(count), replace=len(pool) < int(count)))
            labels.append(np.full(int(count), local_class, dtype=np.int64))
        rows = np.concatenate(rows)
        y = np.concatenate(labels)
        permutation = rng.permutation(len(rows))
        original = self.x[rows[permutation]].copy()
        y = y[permutation]
        mapped32 = _apply_factor2(original, self.factor2_mapping)
        mapped = mapped32.astype(np.float64)
        raw_collisions = sum(np.ascontiguousarray(row).tobytes() in self.raw_source_rows for row in mapped)
        float32_collisions = sum(np.ascontiguousarray(row).tobytes() in self.float32_source_rows for row in mapped32)
        topology = _topology_preserved(original.astype(np.float32), mapped32)
        return RoutedPower2AffineSample(
            x=mapped, y=y, source_class_ids=class_ids.copy(), class_counts=counts.copy(),
            route=self.route, raw_source_collisions=int(raw_collisions),
            float32_source_collisions=int(float32_collisions), rows_transformed=len(mapped),
            selected_column=self.factor2_mapping["selected_column"],
            direction=self.factor2_mapping["direction"],
            ulp_steps=self.factor2_mapping["ulp_steps"], translation=0.0,
            normalized_delta_rmse=self.factor2_mapping["raw_normalized_delta_rmse"],
            float32_equality_partition_preserved=bool(topology),
            rank_order_violations=0 if topology else 1,
        )
