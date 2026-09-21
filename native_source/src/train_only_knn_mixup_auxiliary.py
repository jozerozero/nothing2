#!/usr/bin/env python3
"""Train-only source-conditioned KNN-mixup performance emulator.

This is deliberately a separate engineering lane from the row-free Runtime4
task law. It stores and uses source training rows as interpolation anchors, is
not a recovered physical DGP, and is not claimed to be private.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def infer_column_kinds(x: np.ndarray) -> np.ndarray:
    kinds = []
    for column in range(x.shape[1]):
        values = x[:, column]
        unique = np.unique(values)
        integer_like = np.all(np.isclose(values, np.round(values), atol=1e-6))
        if len(unique) <= 2:
            kinds.append("binary")
        elif integer_like and len(unique) <= 64:
            kinds.append("low_card")
        elif integer_like:
            kinds.append("integer")
        else:
            kinds.append("continuous")
    return np.asarray(kinds, dtype=object)


def proportional_counts(n: int, probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities /= probabilities.sum()
    counts = np.floor(n * probabilities).astype(int)
    if n >= len(counts):
        counts = np.maximum(counts, 1)
    while counts.sum() > n:
        choices = np.flatnonzero(counts > 1)
        counts[choices[np.argmax(counts[choices] - n * probabilities[choices])]] -= 1
    remainder = n - int(counts.sum())
    if remainder:
        order = np.argsort(-(n * probabilities - counts))
        counts[order[:remainder]] += 1
    return counts


def collision_mask(generated: np.ndarray, source: np.ndarray) -> np.ndarray:
    source_rows = {np.ascontiguousarray(row).tobytes() for row in source}
    return np.asarray([
        np.ascontiguousarray(row).tobytes() in source_rows for row in generated
    ])


@dataclass(frozen=True)
class MixupSample:
    x: np.ndarray
    y: np.ndarray
    source_class_ids: np.ndarray
    source_class_probabilities: np.ndarray
    exact_train_collision_rate: float
    exact_train_collisions: int
    float32_train_collisions: int
    novelty_repairs: int
    novelty_repair_rate: float
    continuous_jitter_repairs: int
    strict_collision_control: bool


class TrainOnlyKNNMixup:
    """Local same-class manifold interpolation over explicitly passed rows."""

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
        self.y = y.astype(np.int64, copy=False)
        self.original_class_values = classes.copy()
        self.classes, counts = np.unique(self.y, return_counts=True)
        self.class_probabilities = counts / counts.sum()
        self.class_rows = {
            int(cls): self.x[self.y == cls] for cls in self.classes
        }
        self.kinds = infer_column_kinds(self.x)
        q25, q75 = np.quantile(self.x, [0.25, 0.75], axis=0)
        scale = q75 - q25
        scale = np.where(scale > 1e-8, scale, np.std(self.x, axis=0))
        self.scale = np.where(scale > 1e-8, scale, 1.0)
        self._source_row_bytes = {
            np.ascontiguousarray(row).tobytes() for row in self.x
        }
        self._source_row_float32_bytes = {
            np.ascontiguousarray(row.astype(np.float32)).tobytes()
            for row in self.x
        }
        self.lower = np.min(self.x, axis=0)
        self.upper = np.max(self.x, axis=0)

    def _collision_mask(self, generated: np.ndarray) -> np.ndarray:
        return np.asarray([
            np.ascontiguousarray(row).tobytes() in self._source_row_bytes
            for row in generated
        ])

    def _float32_collision_mask(self, generated: np.ndarray) -> np.ndarray:
        return np.asarray([
            np.ascontiguousarray(row.astype(np.float32)).tobytes()
            in self._source_row_float32_bytes
            for row in generated
        ])

    def _strict_collision_mask(self, generated: np.ndarray) -> np.ndarray:
        return self._collision_mask(generated) | self._float32_collision_mask(generated)

    def _repair_collisions_with_empirical_support(
        self,
        generated: np.ndarray,
        source: np.ndarray,
        mask: np.ndarray,
        rng: np.random.Generator,
        *,
        max_attempts_per_row: int = 256,
    ) -> int:
        """Make colliding rows novel using only class-observed column values.

        This controls exact copies but is not a differential-privacy mechanism:
        a repaired row may remain at Hamming or numerical distance one from a
        source member.  If no empirical-support recombination is novel, the
        caller retains fail-closed behavior.
        """

        value_options = []
        mutable_columns = []
        for column in range(source.shape[1]):
            values = np.unique(source[:, column])
            value_options.append(values)
            if len(values) > 1:
                mutable_columns.append(column)
        if not mutable_columns:
            return 0
        repaired = 0
        max_width = min(4, len(mutable_columns))
        for row_index in np.flatnonzero(mask):
            original = generated[row_index].copy()
            accepted = False
            for attempt in range(max_attempts_per_row):
                candidate = original.copy()
                # Prefer the least disruptive single-column change, then allow
                # small empirical-support recombinations if it remains seen.
                width = 1 if attempt < 64 else int(rng.integers(2, max_width + 1))
                columns = rng.choice(mutable_columns, size=width, replace=False)
                for column in np.atleast_1d(columns):
                    options = value_options[int(column)]
                    current = candidate[int(column)]
                    alternate = options[options != current]
                    if alternate.size:
                        candidate[int(column)] = rng.choice(alternate)
                if np.ascontiguousarray(candidate).tobytes() not in self._source_row_bytes:
                    generated[row_index] = candidate
                    repaired += 1
                    accepted = True
                    break
            if not accepted:
                generated[row_index] = original
        return repaired

    def _repair_collisions_with_continuous_jitter(
        self,
        generated: np.ndarray,
        mask: np.ndarray,
        rng: np.random.Generator,
        *,
        max_attempts_per_row: int = 256,
    ) -> int:
        """Repair empirically impossible copies with minimal numeric jitter.

        A class can contain several rows but only one unique predictor vector.
        In that case no same-class empirical-value recombination can be novel.
        We perturb only columns that are continuous in the complete source
        table, remain inside the observed global range, and require novelty to
        survive a float32 round trip.  This is an explicit performance-
        emulation fallback, not a privacy mechanism or a recovered physical
        data-generating law.
        """

        continuous_columns = np.flatnonzero(self.kinds == "continuous")
        if not continuous_columns.size:
            return 0
        repaired = 0
        for row_index in np.flatnonzero(mask):
            original = generated[row_index].copy()
            accepted = False
            for _ in range(max_attempts_per_row):
                column = int(rng.choice(continuous_columns))
                center = float(original[column])
                center32 = np.float32(center)
                next32 = np.nextafter(
                    center32, np.float32(np.inf), dtype=np.float32
                )
                float32_step = abs(float(next32) - float(center32))
                minimum = max(
                    float(self.scale[column]) * 5e-4,
                    float32_step * 8.0,
                    1e-7,
                )
                magnitude = minimum * float(rng.uniform(1.0, 4.0))
                direction = -1.0 if rng.random() < 0.5 else 1.0
                proposed = center + direction * magnitude
                if not (self.lower[column] <= proposed <= self.upper[column]):
                    proposed = center - direction * magnitude
                if not (self.lower[column] <= proposed <= self.upper[column]):
                    continue
                candidate = original.copy()
                candidate[column] = proposed
                raw_bytes = np.ascontiguousarray(candidate).tobytes()
                float32_bytes = np.ascontiguousarray(
                    candidate.astype(np.float32)
                ).tobytes()
                if (
                    raw_bytes not in self._source_row_bytes
                    and float32_bytes not in self._source_row_float32_bytes
                ):
                    generated[row_index] = candidate
                    repaired += 1
                    accepted = True
                    break
            if not accepted:
                generated[row_index] = original
        return repaired

    def _draw_class(
        self,
        source: np.ndarray,
        count: int,
        rng: np.random.Generator,
        donor_pool_size: int,
    ) -> np.ndarray:
        if len(source) == 1:
            return np.repeat(source, count, axis=0)
        base_index = rng.integers(0, len(source), size=count)
        base = source[base_index]
        pool_size = min(int(donor_pool_size), len(source) - 1)
        candidate_index = rng.integers(0, len(source), size=(count, pool_size))
        candidate_index = np.where(
            candidate_index == base_index[:, None],
            (candidate_index + 1) % len(source),
            candidate_index,
        )
        candidate = source[candidate_index]
        delta = (candidate - base[:, None, :]) / self.scale[None, None, :]
        distance = np.mean(np.minimum(np.square(delta), 100.0), axis=2)
        positive = distance > 1e-12
        chosen = np.argmin(np.where(positive, distance, np.inf), axis=1)
        no_distinct = ~positive.any(axis=1)
        if no_distinct.any():
            chosen[no_distinct] = np.argmin(distance[no_distinct], axis=1)
        donor = candidate[np.arange(count), chosen]

        mixed = base.copy()
        lam = rng.uniform(0.08, 0.92, size=(count, 1))
        numeric = np.flatnonzero(np.isin(self.kinds, ("continuous", "integer")))
        categorical = np.flatnonzero(np.isin(self.kinds, ("binary", "low_card")))
        if numeric.size:
            mixed[:, numeric] = (
                (1.0 - lam) * base[:, numeric] + lam * donor[:, numeric]
            )
            integer = np.flatnonzero(self.kinds == "integer")
            if integer.size:
                mixed[:, integer] = np.round(mixed[:, integer])
        if categorical.size:
            use_donor = rng.random((count, len(categorical))) < lam
            mixed[:, categorical] = np.where(
                use_donor, donor[:, categorical], base[:, categorical]
            )
            unchanged = np.all(mixed == base, axis=1)
            for row in np.flatnonzero(unchanged):
                differing = np.flatnonzero(donor[row] != base[row])
                if differing.size:
                    column = int(rng.choice(differing))
                    mixed[row, column] = donor[row, column]
        return mixed

    def sample(
        self,
        n: int,
        *,
        seed: int,
        max_classes: int = 20,
        donor_pool_size: int = 24,
        strict_no_exact_collision: bool = False,
        collision_retries: int = 8,
        source_class_ids: np.ndarray | None = None,
        class_counts: np.ndarray | None = None,
    ) -> MixupSample:
        if n < 4 or max_classes < 2 or donor_pool_size < 1:
            raise ValueError("invalid sample request")
        rng = np.random.default_rng(seed)
        if (source_class_ids is None) != (class_counts is None):
            raise ValueError("source_class_ids and class_counts must be supplied together")
        if source_class_ids is None:
            class_ids = self.classes.copy()
            probabilities = self.class_probabilities.copy()
            if len(class_ids) > max_classes:
                class_ids = rng.choice(
                    class_ids, size=max_classes, replace=False, p=probabilities
                )
                probabilities = self.class_probabilities[class_ids]
        else:
            class_ids = np.asarray(source_class_ids, dtype=np.int64).reshape(-1)
            requested_counts = np.asarray(class_counts, dtype=np.int64).reshape(-1)
            if (
                len(class_ids) < 2
                or len(class_ids) > max_classes
                or len(class_ids) != len(requested_counts)
                or len(np.unique(class_ids)) != len(class_ids)
                or not np.isin(class_ids, self.classes).all()
                or np.any(requested_counts < 1)
                or int(requested_counts.sum()) != n
            ):
                raise ValueError("invalid explicit class recipe")
            probabilities = self.class_probabilities[class_ids]
        probabilities = probabilities / probabilities.sum()
        counts = (
            proportional_counts(n, probabilities)
            if class_counts is None
            else requested_counts
        )
        rows = []
        labels = []
        novelty_repairs = 0
        continuous_jitter_repairs = 0
        for local_class, (source_class, count) in enumerate(zip(class_ids, counts)):
            source = self.class_rows[int(source_class)]
            generated = self._draw_class(
                source, int(count), rng, donor_pool_size
            )
            if strict_no_exact_collision:
                mask = self._strict_collision_mask(generated)
                for _ in range(int(collision_retries)):
                    if not mask.any():
                        break
                    generated[mask] = self._draw_class(
                        source, int(mask.sum()), rng, donor_pool_size
                    )
                    mask = self._strict_collision_mask(generated)
                if mask.any():
                    novelty_repairs += self._repair_collisions_with_empirical_support(
                        generated, source, mask, rng
                    )
                    mask = self._strict_collision_mask(generated)
                if mask.any():
                    repaired = self._repair_collisions_with_continuous_jitter(
                        generated, mask, rng
                    )
                    continuous_jitter_repairs += repaired
                    novelty_repairs += repaired
                    mask = self._strict_collision_mask(generated)
                if mask.any():
                    raise RuntimeError(
                        "cannot remove exact train collisions for this table; "
                        "strict zero-copy generation is infeasible"
                    )
            rows.append(generated)
            labels.append(np.full(int(count), local_class, dtype=np.int64))
        x = np.concatenate(rows)
        y = np.concatenate(labels)
        permutation = rng.permutation(len(y))
        x, y = x[permutation], y[permutation]
        collisions = int(self._collision_mask(x).sum())
        float32_collisions = int(self._float32_collision_mask(x).sum())
        return MixupSample(
            x=x,
            y=y,
            source_class_ids=class_ids.copy(),
            source_class_probabilities=probabilities.copy(),
            exact_train_collision_rate=collisions / len(x),
            exact_train_collisions=collisions,
            float32_train_collisions=float32_collisions,
            novelty_repairs=novelty_repairs,
            novelty_repair_rate=novelty_repairs / len(x),
            continuous_jitter_repairs=continuous_jitter_repairs,
            strict_collision_control=bool(strict_no_exact_collision),
        )
