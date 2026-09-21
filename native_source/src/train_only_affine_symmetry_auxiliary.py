#!/usr/bin/env python3
"""Paired train-only positive-affine preprocessing-orbit emulator.

This generator is deliberately GT-derived.  It samples the exact paired H
rows and translates one full-table feature interval outside its original
support.  The map is positive, injective and monotone in float32, so the
production per-task clipping and standardization should nearly quotient it
out.  It is a training-effect control, not an independently identified DGP or
privacy mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _row_bytes(x: np.ndarray) -> set[bytes]:
    contiguous = np.ascontiguousarray(x)
    return {
        np.ascontiguousarray(row).tobytes()
        for row in contiguous
    }


@dataclass(frozen=True)
class AffineSymmetrySample:
    x: np.ndarray
    y: np.ndarray
    source_class_ids: np.ndarray
    class_counts: np.ndarray
    raw_source_collisions: int
    float32_source_collisions: int
    rows_translated: int
    selected_column: int
    translation: float
    normalized_delta_rmse: float
    float32_equality_partition_preserved: bool
    rank_order_violations: int
    source_interval_disjoint: bool


def _equality_inverse(values: np.ndarray) -> np.ndarray:
    return np.unique(np.asarray(values), return_inverse=True)[1]


class TrainOnlyAffineSymmetry:
    """Draw a paired GT task and apply one deterministic affine symmetry."""

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
        self.raw_source_rows = _row_bytes(self.x)
        self.float32_source_rows = _row_bytes(self.x32)
        (
            self.selected_column,
            self.translation,
            self.table_normalized_delta_rmse,
        ) = self._select_mapping()

    @staticmethod
    def _robust_scale(values: np.ndarray) -> float:
        values64 = np.asarray(values, dtype=np.float64)
        q25, q75 = np.quantile(values64, [0.25, 0.75])
        scale = float(q75 - q25)
        if scale <= 1e-8:
            scale = float(np.std(values64))
        if scale <= 1e-8:
            scale = max(float(np.max(np.abs(values64))), 1.0)
        return scale

    def _select_mapping(self) -> tuple[int, np.float32, float]:
        candidates: list[tuple[float, int, int, np.float32]] = []
        for column in range(self.x32.shape[1]):
            values = self.x32[:, column]
            if not np.isfinite(values).all():
                continue
            source_unique, source_inverse = np.unique(values, return_inverse=True)
            low = float(source_unique.min())
            high = float(source_unique.max())
            span = max(high - low, 0.0)
            scale = self._robust_scale(values)
            spacing = float(np.spacing(np.float32(max(abs(low), abs(high), 1.0))))
            margin = max(0.25 * scale, 16.0 * abs(spacing), 1e-6)
            for direction_order, shift64 in enumerate((span + margin, -(span + margin))):
                shift = np.float32(shift64)
                mapped = np.asarray(values + shift, dtype=np.float32)
                if (
                    not np.isfinite(mapped).all()
                    or np.any(mapped <= -1e6)
                    or np.any(mapped >= 1e9)
                    or np.any(mapped == values)
                ):
                    continue
                mapped_unique, mapped_inverse = np.unique(mapped, return_inverse=True)
                if (
                    len(mapped_unique) != len(source_unique)
                    or not np.array_equal(mapped_inverse, source_inverse)
                    or (
                        len(mapped_unique) > 1
                        and np.any(np.diff(mapped_unique.astype(np.float64)) <= 0)
                    )
                ):
                    continue
                if not (
                    float(mapped_unique.min()) > high
                    or float(mapped_unique.max()) < low
                ):
                    continue
                normalized_rmse = abs(float(shift)) / scale / np.sqrt(self.x32.shape[1])
                candidates.append(
                    (normalized_rmse, int(column), direction_order, shift)
                )
        if not candidates:
            raise RuntimeError(
                "no finite clamp-safe disjoint injective affine translation"
            )
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        normalized_rmse, column, _, shift = candidates[0]
        return column, shift, float(normalized_rmse)

    def sample(
        self,
        n: int,
        *,
        seed: int,
        source_class_ids: np.ndarray,
        class_counts: np.ndarray,
    ) -> AffineSymmetrySample:
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
        row_indices: list[np.ndarray] = []
        labels: list[np.ndarray] = []
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

        translated32 = original.astype(np.float32)
        translated32[:, self.selected_column] = np.asarray(
            translated32[:, self.selected_column] + self.translation,
            dtype=np.float32,
        )
        translated = translated32.astype(np.float64)
        raw_collisions = sum(
            np.ascontiguousarray(row).tobytes() in self.raw_source_rows
            for row in translated
        )
        float32_collisions = sum(
            np.ascontiguousarray(row).tobytes() in self.float32_source_rows
            for row in translated32
        )
        original_values = original[:, self.selected_column].astype(np.float32)
        mapped_values = translated32[:, self.selected_column]
        equality_preserved = np.array_equal(
            _equality_inverse(original_values), _equality_inverse(mapped_values)
        )
        source_unique = np.unique(original_values)
        mapped_unique = np.unique(mapped_values)
        rank_violations = int(
            len(mapped_unique) != len(source_unique)
            or (
                len(mapped_unique) > 1
                and np.any(np.diff(mapped_unique.astype(np.float64)) <= 0)
            )
        )
        full_low = float(self.x32[:, self.selected_column].min())
        full_high = float(self.x32[:, self.selected_column].max())
        interval_disjoint = bool(
            float(mapped_values.min()) > full_high
            or float(mapped_values.max()) < full_low
        )
        scale = self._robust_scale(self.x32[:, self.selected_column])
        normalized_rmse = abs(float(self.translation)) / scale / np.sqrt(self.x32.shape[1])
        return AffineSymmetrySample(
            x=translated,
            y=y,
            source_class_ids=class_ids.copy(),
            class_counts=counts.copy(),
            raw_source_collisions=int(raw_collisions),
            float32_source_collisions=int(float32_collisions),
            rows_translated=len(translated),
            selected_column=self.selected_column,
            translation=float(self.translation),
            normalized_delta_rmse=float(normalized_rmse),
            float32_equality_partition_preserved=bool(equality_preserved),
            rank_order_violations=rank_violations,
            source_interval_disjoint=interval_disjoint,
        )
