#!/usr/bin/env python3
"""Paired train-only exact expanding-power-two-or-affine L6 emulator."""

from __future__ import annotations

import numpy as np

from train_only_affine_symmetry_auxiliary import TrainOnlyAffineSymmetry
from train_only_routed_power2_affine_auxiliary import (
    RoutedPower2AffineSample,
    _robust_scale,
    _row_keys,
    _topology_preserved,
)


FACTORS = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)


def _choose_expanding_power2_mapping(source: np.ndarray) -> dict | None:
    source32 = np.asarray(source, dtype=np.float32)
    source_keys = _row_keys(source32)
    unique_source_rows = len(source_keys)
    scale = _robust_scale(source32)
    for factor_order, factor in enumerate(FACTORS):
        mapped = np.asarray(source32 * np.float32(factor), dtype=np.float32)
        if (
            not np.isfinite(mapped).all()
            or np.any(mapped <= -1e6)
            or np.any(mapped >= 1e9)
        ):
            continue
        keys = [np.ascontiguousarray(row).tobytes() for row in mapped]
        if any(key in source_keys for key in keys) or len(set(keys)) != unique_source_rows:
            continue
        if not _topology_preserved(source32, mapped):
            continue
        normalized = (
            mapped.astype(np.float64) - source32.astype(np.float64)
        ) / scale
        raw_rmse = float(np.sqrt(np.mean(np.square(normalized))))
        if raw_rmse < 0.01:
            continue
        return {
            "factor": float(factor),
            "factor_order": int(factor_order),
            "ulp_steps": 0,
            "selected_column": None,
            "direction": 0,
            "raw_normalized_delta_rmse": raw_rmse,
        }
    return None


def _apply_expanding_power2(x: np.ndarray, mapping: dict) -> np.ndarray:
    return np.asarray(
        np.asarray(x, dtype=np.float32) * np.float32(mapping["factor"]),
        dtype=np.float32,
    )


class TrainOnlyExpandingPower2Affine:
    """Use the first exact expanding power of two, else frozen L3 affine."""

    def __init__(self, x_train: np.ndarray, y_train: np.ndarray):
        self.l3 = TrainOnlyAffineSymmetry(x_train, y_train)
        self.x = self.l3.x
        self.x32 = self.l3.x32
        self.y = self.l3.y
        self.classes = self.l3.classes
        self.class_indices = self.l3.class_indices
        self.raw_source_rows = _row_keys(self.x)
        self.float32_source_rows = _row_keys(self.x32)
        self.power2_mapping = _choose_expanding_power2_mapping(self.x32)
        self.route = (
            "expanding_power2_exact"
            if self.power2_mapping is not None
            else "l3_affine"
        )

    def sample(
        self,
        n: int,
        *,
        seed: int,
        source_class_ids: np.ndarray,
        class_counts: np.ndarray,
    ) -> RoutedPower2AffineSample:
        if self.route == "l3_affine":
            sample = self.l3.sample(
                n,
                seed=seed,
                source_class_ids=source_class_ids,
                class_counts=class_counts,
            )
            return RoutedPower2AffineSample(
                x=sample.x,
                y=sample.y,
                source_class_ids=sample.source_class_ids,
                class_counts=sample.class_counts,
                route=self.route,
                raw_source_collisions=sample.raw_source_collisions,
                float32_source_collisions=sample.float32_source_collisions,
                rows_transformed=sample.rows_translated,
                selected_column=sample.selected_column,
                direction=0,
                ulp_steps=0,
                translation=sample.translation,
                normalized_delta_rmse=sample.normalized_delta_rmse,
                float32_equality_partition_preserved=sample.float32_equality_partition_preserved,
                rank_order_violations=sample.rank_order_violations,
            )
        class_ids = np.asarray(source_class_ids, dtype=np.int64).reshape(-1)
        counts = np.asarray(class_counts, dtype=np.int64).reshape(-1)
        if (
            n < 4
            or len(class_ids) < 2
            or len(class_ids) != len(counts)
            or len(np.unique(class_ids)) != len(class_ids)
            or not np.isin(class_ids, self.classes).all()
            or np.any(counts < 1)
            or int(counts.sum()) != n
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
        mapped32 = _apply_expanding_power2(original, self.power2_mapping)
        mapped = mapped32.astype(np.float64)
        raw_collisions = sum(
            np.ascontiguousarray(row).tobytes() in self.raw_source_rows
            for row in mapped
        )
        float32_collisions = sum(
            np.ascontiguousarray(row).tobytes() in self.float32_source_rows
            for row in mapped32
        )
        topology = _topology_preserved(original.astype(np.float32), mapped32)
        return RoutedPower2AffineSample(
            x=mapped,
            y=y,
            source_class_ids=class_ids.copy(),
            class_counts=counts.copy(),
            route=self.route,
            raw_source_collisions=int(raw_collisions),
            float32_source_collisions=int(float32_collisions),
            rows_transformed=len(mapped),
            selected_column=None,
            direction=0,
            ulp_steps=0,
            translation=0.0,
            normalized_delta_rmse=self.power2_mapping["raw_normalized_delta_rmse"],
            float32_equality_partition_preserved=bool(topology),
            rank_order_violations=0 if topology else 1,
        )
