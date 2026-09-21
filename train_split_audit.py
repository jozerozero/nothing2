#!/usr/bin/env python3
"""Audit GT regression targets and build a leakage-safe anonymous RW profile.

Every template is derived from an official *training* split. Real target
preprocessing mirrors the evaluation contract: fit an invertible identity or
asinh transform on training/support targets, apply it, then use population
standardization (the convention used by sklearn's StandardScaler). Dataset
names and raw targets are retained only in the human-readable audit and never
copied into the training profile.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


SUITES = ("talent", "BCCO", "CTR23", "OpenMLCC18", "PFN", "TabArena", "TabZilla")
OPENML_ARFF_SUITES = {"CTR23", "OpenMLCC18", "PFN", "TabArena", "TabZilla"}
QUANTILE_LEVELS = np.linspace(0.0, 1.0, 257, dtype=np.float64)


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def _suite_regression_dir(root: Path, suite: str) -> Path:
    candidates = (
        root / suite / "regression",
        root / suite.lower() / "regression",
        root / suite / "Regression",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _dataset_dirs(regression_dir: Path) -> list[Path]:
    if not regression_dir.exists():
        return []
    # Path.is_dir follows suite symlinks, which is required for these benchmark
    # layouts. Sorting makes profile construction deterministic.
    return sorted(p for p in regression_dir.iterdir() if p.is_dir() and not p.name.startswith("."))


def _normalized_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.casefold())


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _numeric_target(values: Any, source: str) -> np.ndarray:
    y = np.asarray(values)
    if y.ndim > 2 or (y.ndim == 2 and 1 not in y.shape):
        raise ValueError(f"target must be scalar-valued, got shape={y.shape} from {source}")
    try:
        y = y.reshape(-1).astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"target is not numeric in {source}: {exc}") from exc
    if y.size < 2:
        raise ValueError(f"target has fewer than two training rows in {source}")
    if not np.isfinite(y).all():
        raise ValueError(f"target contains NaN or infinity in {source}")
    if float(np.max(y)) == float(np.min(y)):
        raise ValueError(f"target is constant in {source}")
    return y


def _load_talent_train(dataset_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    target = dataset_dir / "y_train.npy"
    if not target.is_file():
        raise FileNotFoundError("missing TALENT y_train.npy")
    try:
        values = np.load(target, allow_pickle=False)
        object_coercion = False
    except ValueError as exc:
        if "Object arrays cannot be loaded" not in str(exc):
            raise
        # Explicitly trusted benchmark source; two TALENT targets are numeric
        # values stored in object arrays.
        values = np.load(target, allow_pickle=True)
        object_coercion = True
    info = _load_json(dataset_dir / "info.json") if (dataset_dir / "info.json").is_file() else {}
    return _numeric_target(values, str(target)), {
        "format": "talent_npy",
        "split": "train",
        "openml_dataset_id": str(info["openml_id"]) if info.get("openml_id") is not None else None,
        "target_feature": None,
        "object_to_float_coercion": object_coercion,
    }


def _load_bcco_train(dataset_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    files = sorted(dataset_dir.glob("*_train.csv"))
    if len(files) != 1:
        raise FileNotFoundError(f"expected exactly one BCCO *_train.csv, found {len(files)}")
    with files[0].open(newline="", errors="replace") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        candidates = [field for field in fields if field.casefold() == "target"]
        if len(candidates) != 1:
            raise ValueError(f"expected one target column, found {candidates}")
        target = candidates[0]
        values = [row[target] for row in reader]
    return _numeric_target(values, str(files[0])), {
        "format": "bcco_csv",
        "split": "train",
        "openml_dataset_id": None,
        "target_feature": target,
        "object_to_float_coercion": False,
    }


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)


def _arff_attribute_name(line: str) -> str:
    rest = re.sub(r"^\s*@attribute\s+", "", line, flags=re.IGNORECASE).lstrip()
    if not rest:
        raise ValueError("empty ARFF attribute declaration")
    if rest[0] not in {"'", '"'}:
        return rest.split(None, 1)[0]
    quote = rest[0]
    escaped = False
    chars: list[str] = []
    for char in rest[1:]:
        if escaped:
            chars.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == quote:
            return "".join(chars)
        else:
            chars.append(char)
    raise ValueError(f"unterminated quoted ARFF attribute: {line.rstrip()}")


def _split_arff_row(line: str) -> list[str]:
    fields: list[str] = []
    chars: list[str] = []
    quote: str | None = None
    escaped = False
    for char in line.rstrip("\r\n"):
        if escaped:
            chars.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif quote is not None:
            if char == quote:
                quote = None
            else:
                chars.append(char)
        elif char in {"'", '"'}:
            quote = char
        elif char == ",":
            fields.append("".join(chars).strip())
            chars = []
        else:
            chars.append(char)
    if quote is not None:
        raise ValueError("unterminated quote in ARFF data row")
    fields.append("".join(chars).strip())
    return fields


def _load_arff_target_column(path: Path, target: str, selected_row_ids: np.ndarray) -> np.ndarray:
    """Read only the numeric target column, tolerating unrelated string data.

    scipy's legacy ARFF loader rejects STRING attributes and some valid quoted
    nominal values before exposing any column.  The target-profile audit does
    not need feature columns, so this narrow parser deliberately reads only the
    schema and the requested scalar target while still validating every row's
    width.
    """

    attributes: list[str] = []
    target_index: int | None = None
    selected = {int(row_id) for row_id in selected_row_ids}
    values: dict[int, str] = {}
    in_data = False
    row_index = 0
    with path.open(errors="replace") as handle:
        for raw in handle:
            stripped = raw.strip()
            if not stripped or stripped.startswith("%"):
                continue
            if not in_data:
                if stripped.casefold().startswith("@attribute"):
                    name = _arff_attribute_name(raw)
                    if name.casefold() == target.casefold():
                        if target_index is not None:
                            raise ValueError(f"target_feature={target!r} resolves more than once")
                        target_index = len(attributes)
                    attributes.append(name)
                elif stripped.casefold() == "@data":
                    in_data = True
                continue
            if stripped.startswith("{"):
                raise ValueError("sparse ARFF rows are unsupported by the target-only parser")
            if row_index not in selected:
                row_index += 1
                continue
            row = _split_arff_row(raw)
            if len(row) != len(attributes):
                raise ValueError(f"ARFF row width {len(row)} != schema width {len(attributes)}")
            if target_index is None:
                raise ValueError(f"target_feature={target!r} did not resolve in data.arff")
            values[row_index] = row[target_index]
            row_index += 1
    if not in_data:
        raise ValueError("data.arff has no @DATA section")
    if target_index is None:
        raise ValueError(f"target_feature={target!r} did not resolve in data.arff")
    missing = selected - values.keys()
    if missing:
        raise ValueError(f"official split row ids are outside data.arff: {sorted(missing)[:5]}")
    return _numeric_target([values[int(row_id)] for row_id in selected_row_ids], str(path))


def _load_openml_split_columns(path: Path) -> dict[str, list[str]]:
    """Read the four OpenML split columns without a scipy dependency.

    The benchmark split files use dense ARFF rows.  Keeping this parser narrow
    makes the train-only leakage audit runnable in a minimal NumPy environment
    while retaining the same quoted-name/quoted-value handling as the target
    parser above.
    """

    attributes: list[str] = []
    columns: dict[str, list[str]] = {}
    in_data = False
    with path.open(errors="replace") as handle:
        for raw in handle:
            stripped = raw.strip()
            if not stripped or stripped.startswith("%"):
                continue
            if not in_data:
                if stripped.casefold().startswith("@attribute"):
                    name = _arff_attribute_name(raw)
                    attributes.append(name)
                    columns[name.casefold()] = []
                elif stripped.casefold() == "@data":
                    in_data = True
                continue
            if stripped.startswith("{"):
                raise ValueError("sparse split ARFF rows are unsupported")
            row = _split_arff_row(raw)
            if len(row) != len(attributes):
                raise ValueError(f"split ARFF row width {len(row)} != schema width {len(attributes)}")
            for name, value in zip(attributes, row, strict=True):
                columns[name.casefold()].append(value)
    if not in_data:
        raise ValueError("splits.arff has no @DATA section")
    return columns


def _integer_split_column(values: list[str], name: str) -> np.ndarray:
    parsed: list[int] = []
    for value in values:
        number = float(value)
        if not math.isfinite(number) or not number.is_integer():
            raise ValueError(f"splits.arff {name} contains non-integer value {value!r}")
        parsed.append(int(number))
    return np.asarray(parsed, dtype=np.int64)


def _load_openml_fold0_train(dataset_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    membership = _load_json(dataset_dir / "membership.json")
    target = str(membership.get("target_feature") or "")
    if not target:
        raise ValueError("membership.json has no target_feature")
    split_columns = _load_openml_split_columns(dataset_dir / "splits.arff")
    required = {"type", "rowid", "repeat", "fold"}
    if not required <= split_columns.keys():
        raise ValueError(f"splits.arff missing fields {sorted(required - split_columns.keys())}")
    lengths = {len(split_columns[name]) for name in required}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("splits.arff required columns have inconsistent or empty lengths")
    repeat = _integer_split_column(split_columns["repeat"], "repeat")
    fold = _integer_split_column(split_columns["fold"], "fold")
    rowid = _integer_split_column(split_columns["rowid"], "rowid")
    split_type = np.asarray([_decode(value).upper() for value in split_columns["type"]])
    selected_repeat = int(np.min(repeat))
    selected_fold = int(np.min(fold[repeat == selected_repeat]))
    mask = (repeat == selected_repeat) & (fold == selected_fold) & (split_type == "TRAIN")
    row_ids = rowid[mask]
    if row_ids.size < 2 or np.unique(row_ids).size != row_ids.size:
        raise ValueError("official fold-0 TRAIN row ids are missing or duplicated")
    if int(row_ids.min()) < 0:
        raise ValueError("official split row id is negative")
    values = _load_arff_target_column(dataset_dir / "data.arff", target, row_ids)
    return _numeric_target(values, str(dataset_dir / "data.arff")), {
        "format": "openml_arff",
        "split": f"repeat={selected_repeat},fold={selected_fold},type=TRAIN",
        "openml_dataset_id": str(membership["openml_dataset_id"]) if membership.get("openml_dataset_id") is not None else None,
        "openml_task_id": str(membership["openml_task_id"]) if membership.get("openml_task_id") is not None else None,
        "target_feature": target,
        "object_to_float_coercion": False,
    }


def _load_train_target(suite: str, dataset_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    if suite == "talent":
        return _load_talent_train(dataset_dir)
    if suite == "BCCO":
        return _load_bcco_train(dataset_dir)
    if suite in OPENML_ARFF_SUITES:
        return _load_openml_fold0_train(dataset_dir)
    raise ValueError(f"unsupported suite: {suite}")


def _moments(values: np.ndarray) -> tuple[float, float]:
    centered = values - np.mean(values)
    std = float(np.std(values))
    if std <= 0:
        return 0.0, 0.0
    z = centered / std
    return float(np.mean(z**3)), float(np.mean(z**4) - 3.0)


def _fit_gt_target_transform(values: np.ndarray) -> tuple[str, float, float]:
    q01, q25, q50, q75, q99 = np.quantile(values, (0.01, 0.25, 0.5, 0.75, 0.99))
    iqr = max(float(q75 - q25), 1e-12)
    tail_ratio = float((q99 - q01) / iqr)
    std = max(float(np.std(values, ddof=0)), 1e-12)
    skew = float(np.mean(((values - np.mean(values)) / std) ** 3))
    if abs(skew) <= 2.0 and tail_ratio <= 25.0:
        return "identity", 0.0, 1.0
    return "asinh", float(q50), max(iqr / 1.349, 0.05 * std, 1e-12)


def _apply_gt_target_transform(values: np.ndarray, kind: str, center: float, scale: float) -> np.ndarray:
    if kind == "identity":
        return values.copy()
    if kind == "asinh":
        return np.arcsinh((values - center) / scale)
    raise ValueError(f"unknown target transform: {kind}")


def _audit_values(y: np.ndarray, metadata: dict[str, Any]) -> tuple[dict[str, Any], list[float]]:
    kind, center, scale = _fit_gt_target_transform(y)
    transformed = _apply_gt_target_transform(y, kind, center, scale)
    model_mean = float(np.mean(transformed))
    model_std = float(np.std(transformed, ddof=0))
    if not math.isfinite(model_std) or model_std <= 0:
        raise ValueError("target becomes constant/non-finite after GT transform")
    # Exact model-input convention: sklearn StandardScaler uses population std.
    normalized = (transformed - model_mean) / model_std
    if not np.isfinite(normalized).all():
        raise ValueError("model-space target is non-finite")
    skew, excess_kurtosis = _moments(y)
    q01, q25, _, q75, q99 = np.quantile(y, (0.01, 0.25, 0.5, 0.75, 0.99))
    stats = {
        **metadata,
        "recommended_target_transform": kind,
        "target_transform_center": center,
        "target_transform_scale": scale,
        "model_standardizer_mean": model_mean,
        "model_standardizer_std_ddof0": model_std,
        "n_train": int(y.size),
        "raw_min": float(np.min(y)),
        "raw_max": float(np.max(y)),
        "raw_range": float(np.ptp(y)),
        "raw_mean": float(np.mean(y)),
        "raw_std_ddof0": float(np.std(y, ddof=0)),
        "raw_skew": skew,
        "raw_excess_kurtosis": excess_kurtosis,
        "raw_tail_ratio": float((q99 - q01) / max(float(q75 - q25), 1e-12)),
        "raw_quantiles": {str(q): float(np.quantile(y, q)) for q in (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)},
        "model_space_min": float(np.min(normalized)),
        "model_space_max": float(np.max(normalized)),
        "model_space_mean": float(np.mean(normalized)),
        "model_space_std_ddof0": float(np.std(normalized, ddof=0)),
    }
    template = np.quantile(normalized, QUANTILE_LEVELS).astype(np.float32).tolist()
    return stats, template


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    args = parser.parse_args()

    benchmark_root = args.benchmark_root.resolve()
    suite_summary: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    templates: list[list[float]] = []
    transform_kinds: list[str] = []
    template_source_suites: list[str] = []
    rejected: list[dict[str, str]] = []
    duplicates: list[dict[str, str]] = []
    seen_names: dict[str, str] = {}
    seen_openml: dict[str, str] = {}

    for suite in SUITES:
        reg_dir = _suite_regression_dir(benchmark_root, suite)
        dirs = _dataset_dirs(reg_dir)
        accepted = 0
        suite_rejected = 0
        suite_duplicates = 0
        for dataset_dir in dirs:
            try:
                y, metadata = _load_train_target(suite, dataset_dir)
            except Exception as exc:
                rejected.append({"suite": suite, "dataset": dataset_dir.name, "reason": str(exc)})
                suite_rejected += 1
                continue

            name_key = _normalized_name(dataset_dir.name)
            openml_key = metadata.get("openml_dataset_id")
            duplicate_of = seen_names.get(name_key) or (seen_openml.get(str(openml_key)) if openml_key else None)
            if duplicate_of is not None:
                duplicates.append({"suite": suite, "dataset": dataset_dir.name, "duplicate_of": duplicate_of})
                suite_duplicates += 1
                continue
            identity = f"{suite}/{dataset_dir.name}"
            seen_names[name_key] = identity
            if openml_key:
                seen_openml[str(openml_key)] = identity

            try:
                stats, template = _audit_values(y, metadata)
            except Exception as exc:
                rejected.append({"suite": suite, "dataset": dataset_dir.name, "reason": str(exc)})
                suite_rejected += 1
                continue
            records.append({"suite": suite, "dataset": dataset_dir.name, **stats})
            templates.append(template)
            transform_kinds.append(str(stats["recommended_target_transform"]))
            template_source_suites.append(suite)
            accepted += 1
        suite_summary[suite] = {
            "regression_dir": str(reg_dir),
            "dataset_count_following_symlinks": len(dirs),
            "accepted_unique_train_targets": accepted,
            "duplicate_count": suite_duplicates,
            "rejected_count": suite_rejected,
        }

    if not templates:
        raise RuntimeError("no valid GT regression training targets were found")

    suite_template_counts = {suite: template_source_suites.count(suite) for suite in SUITES}
    audit = {
        "schema_version": 2,
        "benchmark_root": str(benchmark_root),
        "scope": "GT regression target preprocessing/postprocessing and RW profile",
        "suite_summary": suite_summary,
        "valid_unique_count": len(records),
        "duplicate_count": len(duplicates),
        "rejected_count": len(rejected),
        "duplicates": duplicates,
        "rejected": rejected,
        "target_statistics": records,
        "leakage_policy": {
            "training_profile_reads": [
                "TALENT y_train.npy",
                "BCCO *_train.csv target column",
                "OpenML suite repeat=0/fold=0/TRAIN rows",
            ],
            "training_profile_forbids": ["TALENT y_val.npy", "TALENT y_test.npy", "BCCO *_test.csv", "OpenML TEST rows"],
            "deduplication": "OpenML dataset id when available, plus normalized dataset name",
            "audit_names_are_not_copied_to_training_profile": True,
        },
    }
    profile = {
        "schema_version": 2,
        "profile_type": "anonymous_train_only_regression_quantile_templates",
        "source_scope": "deduplicated_GT_regression_suites",
        "source_suites": list(SUITES),
        "source_split": "official_train_only",
        "source_suite_template_counts": suite_template_counts,
        "template_count": len(templates),
        "quantile_levels": QUANTILE_LEVELS.astype(np.float32).tolist(),
        "templates": templates,
        "template_transform_kinds": transform_kinds,
        "template_source_suites": template_source_suites,
        "gt_target_transform_policy": {
            "profile_fit_split": "official_train_only",
            "evaluation_fit_split": "support_only",
            "candidates": ["identity", "asinh"],
            "asinh_selection": "abs(skew)>2 or (q99-q01)/IQR>25",
            "inverse_required_before_metrics": True,
            "metrics_units": "original_target_units",
        },
        "model_target_preprocessing": {
            "order": ["GT identity/asinh transform", "StandardScaler"],
            "standard_scaling_ddof": 0,
            "fit_split": "support_only",
            "query_targets_never_fit_statistics": True,
        },
        "model_target_postprocessing": {
            "order": ["inverse StandardScaler", "inverse GT identity/asinh transform"],
            "before_metrics": True,
            "prediction_units": "original_target_units",
        },
        "privacy_and_leakage": {
            "contains_dataset_names": False,
            "contains_raw_targets": False,
            "reads_validation_targets": False,
            "reads_test_targets": False,
            "contains_only_empirical_quantile_templates": True,
        },
    }
    _json_dump(args.output_audit, audit)
    _json_dump(args.output_profile, profile)
    print(json.dumps({
        "audit": str(args.output_audit),
        "profile": str(args.output_profile),
        "valid_unique": len(templates),
        "duplicates": len(duplicates),
        "rejected": len(rejected),
        "suite_template_counts": suite_template_counts,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
