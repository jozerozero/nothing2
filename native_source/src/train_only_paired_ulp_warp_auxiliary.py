#!/usr/bin/env python3
"""Paired train-only ULP-warp performance emulator.

This intentionally near-copy generator is a positive-control performance
emulator, not a physical DGP or privacy mechanism.  It draws the same rows as
the GT task stream and deterministically moves one feature of every row far
enough to survive a float32 round trip while remaining inside the source
column range and avoiding every source row exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ULP_STEPS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


@dataclass(frozen=True)
class PairedULPWarpSample:
    x: np.ndarray
    y: np.ndarray
    source_class_ids: np.ndarray
    class_counts: np.ndarray
    raw_source_collisions: int
    float32_source_collisions: int
    rows_warped: int
    maximum_ulp_steps: int
    normalized_delta_rmse: float


def _row_bytes(x: np.ndarray) -> set[bytes]:
    return {np.ascontiguousarray(row).tobytes() for row in x}


def _advance_float32(value: np.ndarray | np.float32, direction: float, steps: int):
    target = np.float32(np.inf if direction > 0 else -np.inf)
    result = np.asarray(value, dtype=np.float32).copy()
    for _ in range(int(steps)):
        result = np.nextafter(result, target, dtype=np.float32)
    return np.float32(result) if result.ndim == 0 else result


class TrainOnlyPairedULPWarp:
    """Draw a GT-paired task and remove exact membership by minimal ULP warps."""

    def __init__(self, x_train: np.ndarray, y_train: np.ndarray):
        x = np.asarray(x_train, dtype=np.float64)
        y_raw = np.asarray(y_train).reshape(-1)
        if x.ndim != 2 or len(x) != len(y_raw) or len(x) < 4:
            raise ValueError("invalid train-only table")
        if not np.isfinite(x).all():
            raise ValueError("train-only table contains nonfinite predictors")
        classes, y = np.unique(y_raw, return_inverse=True)
        if len(classes) < 2:
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
        q25, q75 = np.quantile(self.x, [0.25, 0.75], axis=0)
        scale = q75 - q25
        scale = np.where(scale > 1e-8, scale, np.std(self.x, axis=0))
        self.scale = np.where(scale > 1e-8, scale, 1.0)
        self.raw_source_rows = _row_bytes(self.x)
        self.float32_source_rows = _row_bytes(self.x32)

    def _is_novel(self, row: np.ndarray) -> bool:
        raw = np.ascontiguousarray(row).tobytes()
        as_float32 = np.ascontiguousarray(row.astype(np.float32)).tobytes()
        return raw not in self.raw_source_rows and as_float32 not in self.float32_source_rows

    def _warp_row(
        self,
        row: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int]:
        if not self.mutable_columns.size:
            raise RuntimeError("selected source columns have no mutable feature")
        columns = rng.permutation(self.mutable_columns)
        preferred_sign = -1.0 if rng.random() < 0.5 else 1.0
        for steps in ULP_STEPS:
            ranked_candidates = []
            row32 = row.astype(np.float32)
            for direction_order, direction in enumerate((preferred_sign, -preferred_sign)):
                displaced = _advance_float32(row32, direction, steps)
                for column_order, column in enumerate(columns):
                    column = int(column)
                    candidate32 = displaced[column]
                    if not (
                        np.isfinite(candidate32)
                        and self.lower32[column] <= candidate32 <= self.upper32[column]
                    ):
                        continue
                    normalized_delta = abs(float(candidate32) - float(row[column])) / self.scale[column]
                    ranked_candidates.append(
                        (
                            normalized_delta,
                            direction_order,
                            column_order,
                            column,
                            candidate32,
                        )
                    )
            # Minimize the source-scale-normalized displacement.  The seeded
            # column permutation and preferred sign provide deterministic ties.
            ranked_candidates.sort(key=lambda candidate: candidate[:3])
            for _, _, _, column, candidate32 in ranked_candidates:
                candidate = row.copy()
                candidate[column] = float(candidate32)
                if self._is_novel(candidate):
                    return candidate, int(steps)
        raise RuntimeError("cannot remove source-row collision by frozen ULP search")

    def sample(
        self,
        n: int,
        *,
        seed: int,
        source_class_ids: np.ndarray,
        class_counts: np.ndarray,
    ) -> PairedULPWarpSample:
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

        warped = np.empty_like(original)
        maximum_steps = 0
        for row_index, row in enumerate(original):
            warped[row_index], used_steps = self._warp_row(row, rng)
            maximum_steps = max(maximum_steps, used_steps)

        raw_collisions = sum(
            np.ascontiguousarray(row).tobytes() in self.raw_source_rows for row in warped
        )
        float32_collisions = sum(
            np.ascontiguousarray(row.astype(np.float32)).tobytes()
            in self.float32_source_rows
            for row in warped
        )
        normalized_delta = (warped - original) / self.scale
        return PairedULPWarpSample(
            x=warped,
            y=y,
            source_class_ids=class_ids.copy(),
            class_counts=counts.copy(),
            raw_source_collisions=int(raw_collisions),
            float32_source_collisions=int(float32_collisions),
            rows_warped=len(warped),
            maximum_ulp_steps=maximum_steps,
            normalized_delta_rmse=float(np.sqrt(np.mean(np.square(normalized_delta)))),
        )
