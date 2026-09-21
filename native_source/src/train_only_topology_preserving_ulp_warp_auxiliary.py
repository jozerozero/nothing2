#!/usr/bin/env python3
"""Topology-preserving paired train-only ULP-warp performance emulator.

This is an intentionally near-copy positive control, not an independent DGP
or privacy mechanism.  Unlike the earlier per-row warp, it applies one global
monotone float32 translation to one column.  Consequently, duplicate/equality
groups and the rank order of that column are preserved while every emitted
row is absent from the source table in raw and float32 space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from train_only_paired_ulp_warp_auxiliary import _advance_float32, _row_bytes


ULP_STEPS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


@dataclass(frozen=True)
class TopologyPreservingULPWarpSample:
    x: np.ndarray
    y: np.ndarray
    source_class_ids: np.ndarray
    class_counts: np.ndarray
    raw_source_collisions: int
    float32_source_collisions: int
    rows_warped: int
    selected_column: int
    direction: int
    ulp_steps: int
    normalized_delta_rmse: float
    float32_equality_partition_preserved: bool
    rank_order_violations: int
    source_range_exceedances: int


def _row_group_ids(x: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(x)
    groups: dict[bytes, int] = {}
    inverse = np.empty(len(contiguous), dtype=np.int64)
    for index, row in enumerate(contiguous):
        key = np.ascontiguousarray(row).tobytes()
        if key not in groups:
            groups[key] = len(groups)
        inverse[index] = groups[key]
    return inverse


class TrainOnlyTopologyPreservingULPWarp:
    """Draw a GT-paired task and apply a frozen global monotone ULP map."""

    def __init__(self, x_train: np.ndarray, y_train: np.ndarray):
        x = np.asarray(x_train, dtype=np.float64)
        y_raw = np.asarray(y_train).reshape(-1)
        if x.ndim != 2 or len(x) != len(y_raw) or len(x) < 4:
            raise ValueError("invalid train-only table")
        if not np.isfinite(x).all():
            raise ValueError("train-only table contains nonfinite predictors")
        _, y = np.unique(y_raw, return_inverse=True)
        if len(np.unique(y)) < 2:
            raise ValueError("train-only table has fewer than two classes")

        self.x = x.copy()
        self.x32 = self.x.astype(np.float32)
        self.y = y.astype(np.int64, copy=False)
        self.classes = np.unique(self.y)
        self.class_indices = {
            int(class_id): np.flatnonzero(self.y == class_id)
            for class_id in self.classes
        }
        self.lower32 = np.min(self.x32, axis=0)
        self.upper32 = np.max(self.x32, axis=0)
        self.mutable_columns = np.flatnonzero(self.lower32 < self.upper32)
        if not self.mutable_columns.size:
            raise RuntimeError("source table has no mutable float32 feature")

        q25, q75 = np.quantile(self.x, [0.25, 0.75], axis=0)
        scale = q75 - q25
        scale = np.where(scale > 1e-8, scale, np.std(self.x, axis=0))
        self.scale = np.where(scale > 1e-8, scale, 1.0)
        self.raw_source_rows = _row_bytes(self.x)
        self.float32_source_rows = _row_bytes(self.x32)
        self.unique_source_float32_rows = len(self.float32_source_rows)

        (
            self.selected_column,
            self.direction,
            self.ulp_steps,
            self.table_normalized_delta_rmse,
            self.table_source_range_exceedances,
        ) = self._select_mapping()

    def _candidate(self, column: int, direction: int, steps: int) -> np.ndarray | None:
        displaced = np.asarray(
            _advance_float32(self.x32[:, column], float(direction), steps),
            dtype=np.float32,
        )
        if not np.isfinite(displaced).all() or np.any(displaced == self.x32[:, column]):
            return None
        source_values = np.unique(self.x32[:, column])
        mapped_values = np.asarray(
            _advance_float32(source_values, float(direction), steps),
            dtype=np.float32,
        )
        if (
            len(np.unique(mapped_values)) != len(source_values)
            or np.any(np.diff(mapped_values.astype(np.float64)) <= 0)
        ):
            return None
        return displaced

    def _valid_zero_membership_injective_mapping(
        self, column: int, displaced: np.ndarray
    ) -> bool:
        candidate = self.x32.copy()
        candidate[:, column] = displaced
        candidate_rows = [
            np.ascontiguousarray(row).tobytes() for row in candidate
        ]
        if any(row in self.float32_source_rows for row in candidate_rows):
            return False
        return len(set(candidate_rows)) == self.unique_source_float32_rows

    def _select_mapping(self) -> tuple[int, int, int, float, int]:
        for steps in ULP_STEPS:
            candidates: list[tuple[float, int, int, np.ndarray, int]] = []
            for column in self.mutable_columns:
                column = int(column)
                for direction_order, direction in enumerate((1, -1)):
                    displaced = self._candidate(column, direction, steps)
                    if displaced is None:
                        continue
                    delta = (
                        displaced.astype(np.float64) - self.x[:, column]
                    ) / self.scale[column]
                    normalized_rmse = float(
                        np.sqrt(np.mean(np.square(delta)) / self.x.shape[1])
                    )
                    range_exceedances = int(
                        np.count_nonzero(displaced < self.lower32[column])
                        + np.count_nonzero(displaced > self.upper32[column])
                    )
                    candidates.append(
                        (
                            normalized_rmse,
                            column,
                            direction_order,
                            displaced,
                            range_exceedances,
                        )
                    )
            candidates.sort(key=lambda item: item[:3])
            for normalized_rmse, column, direction_order, displaced, exceedances in candidates:
                if self._valid_zero_membership_injective_mapping(column, displaced):
                    direction = 1 if direction_order == 0 else -1
                    return column, direction, int(steps), normalized_rmse, exceedances
        raise RuntimeError(
            "cannot construct a zero-membership injective global ULP mapping"
        )

    def sample(
        self,
        n: int,
        *,
        seed: int,
        source_class_ids: np.ndarray,
        class_counts: np.ndarray,
    ) -> TopologyPreservingULPWarpSample:
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
        row_indices = []
        labels = []
        for local_class, (source_class, count) in enumerate(zip(class_ids, counts)):
            pool = self.class_indices[int(source_class)]
            chosen = rng.choice(pool, size=int(count), replace=len(pool) < int(count))
            row_indices.append(chosen)
            labels.append(np.full(int(count), local_class, dtype=np.int64))
        rows = np.concatenate(row_indices)
        y = np.concatenate(labels)
        permutation = rng.permutation(len(rows))
        original = self.x[rows[permutation]].copy()
        y = y[permutation]

        warped = original.copy()
        displaced = np.asarray(
            _advance_float32(
                original[:, self.selected_column].astype(np.float32),
                float(self.direction),
                self.ulp_steps,
            ),
            dtype=np.float32,
        )
        warped[:, self.selected_column] = displaced.astype(np.float64)

        raw_collisions = sum(
            np.ascontiguousarray(row).tobytes() in self.raw_source_rows for row in warped
        )
        warped32 = warped.astype(np.float32)
        float32_collisions = sum(
            np.ascontiguousarray(row).tobytes() in self.float32_source_rows
            for row in warped32
        )
        equality_preserved = np.array_equal(
            _row_group_ids(original.astype(np.float32)),
            _row_group_ids(warped32),
        )
        source_values = np.unique(original[:, self.selected_column].astype(np.float32))
        mapped_values = np.unique(warped32[:, self.selected_column])
        rank_violations = int(
            len(mapped_values) != len(source_values)
            or np.any(np.diff(mapped_values.astype(np.float64)) <= 0)
        )
        normalized_delta = (warped - original) / self.scale
        exceedances = int(
            np.count_nonzero(displaced < self.lower32[self.selected_column])
            + np.count_nonzero(displaced > self.upper32[self.selected_column])
        )
        return TopologyPreservingULPWarpSample(
            x=warped,
            y=y,
            source_class_ids=class_ids.copy(),
            class_counts=counts.copy(),
            raw_source_collisions=int(raw_collisions),
            float32_source_collisions=int(float32_collisions),
            rows_warped=len(warped),
            selected_column=self.selected_column,
            direction=self.direction,
            ulp_steps=self.ulp_steps,
            normalized_delta_rmse=float(
                np.sqrt(np.mean(np.square(normalized_delta)))
            ),
            float32_equality_partition_preserved=bool(equality_preserved),
            rank_order_violations=rank_violations,
            source_range_exceedances=exceedances,
        )
