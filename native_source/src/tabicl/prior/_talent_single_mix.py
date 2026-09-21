from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np

from ._tabiclv2_classification import TabICLv2ClassificationPrior


DEFAULT_TALENT_DATASET_PATH = (
    Path(__file__).resolve().parents[3]
    / "dataset_improvement/improve protect/talent_single_mix_pkg_extracted/talent_single_mix_pkg/dataset.py"
)

PROTECTED_RESTORE_IDS = {7, 8}
PROTECTED_WEIGHT_IDS = {1, 7, 8, 34, 56}
SHAPE_WEIGHT_IDS = {1, 8, 56}
PROTECTED_GENERATED_IDS = {101, 102, 103, 104, 105}
PROTECTED_GENERATED_SHAPEBOOST_REPEATS = {
    101: 1,  # imbalance
    102: 4,  # continuous shape
    103: 2,  # sparse categorical
    104: 1,  # mixed survey
    105: 2,  # high-class shape
}


def _load_talent_module(dataset_path: Optional[str] = None):
    path = Path(
        dataset_path
        or os.environ.get("TALENT_SINGLE_MIX_DATASET_PATH", "")
        or DEFAULT_TALENT_DATASET_PATH
    )
    if not path.exists():
        raise FileNotFoundError(
            f"TALENT single-mix dataset.py not found at {path}. "
            "Set TALENT_SINGLE_MIX_DATASET_PATH or extract talent_single_mix_pkg.zip."
        )
    spec = importlib.util.spec_from_file_location("tabicl_talent_single_mix_dataset", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load TALENT single-mix module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _set_generators_for_source(talent_batcher, module, source: str) -> None:
    source = (source or "default51").strip().lower()
    if source == "default51":
        return

    default_excluded = set(getattr(module, "MIX_EXCLUDED_DATASET_IDS", frozenset()))
    include_ids = {
        spec.dataset_id
        for spec in module.DATASET_SPECS.values()
        if spec.dataset_id not in default_excluded
    }

    weight_ids: set[int] = set()
    repeat_overrides: dict[int, int] = {}
    if source in {"protected_generated", "protected_gen", "protgen"}:
        include_ids = set(PROTECTED_GENERATED_IDS)
    elif source in {"protected_generated_shapeboost", "protgen_shapeboost", "protected_gen_shapeboost"}:
        include_ids = set(PROTECTED_GENERATED_IDS)
        repeat_overrides = PROTECTED_GENERATED_SHAPEBOOST_REPEATS
    elif source in {"protected_restore", "protected_weighted"}:
        include_ids |= PROTECTED_RESTORE_IDS
        weight_ids = PROTECTED_WEIGHT_IDS if source == "protected_weighted" else set()
    elif source == "shape_weighted":
        include_ids |= {8}
        weight_ids = SHAPE_WEIGHT_IDS
    else:
        raise ValueError(
            "Unknown talent_mix_source "
            f"{source!r}; expected default51, protected_generated, protected_generated_shapeboost, "
            "protected_restore, protected_weighted, or shape_weighted."
        )

    generators = []
    for spec in module.DATASET_SPECS.values():
        if spec.dataset_id not in include_ids:
            continue
        repeats = repeat_overrides.get(spec.dataset_id, 3 if spec.dataset_id in weight_ids else 1)
        generators.extend(module.TalentSingleGenerator(spec) for _ in range(repeats))
    talent_batcher.generators = tuple(generators)


class TalentSinglePriorAdapter:
    """Adapter from the packaged 6-tuple TALENT prior to TabICL's 5-tuple prior API."""

    def __init__(
        self,
        batch_size: int = 256,
        batch_size_per_gp: int = 4,
        batch_size_per_subgp: Optional[int] = None,
        min_features: int = 2,
        max_features: int = 100,
        max_classes: int = 10,
        min_seq_len: Optional[int] = None,
        max_seq_len: int = 1024,
        log_seq_len: bool = False,
        seq_len_per_gp: bool = False,
        min_train_size: Union[int, float] = 0.1,
        max_train_size: Union[int, float] = 0.9,
        replay_small: bool = False,
        talent_mix_source: str = "default51",
        talent_mix_dataset_path: Optional[str] = None,
        device: str = "cpu",
    ):
        del batch_size_per_subgp
        self.talent_mix_source = talent_mix_source
        self.talent_mix_dataset_path = talent_mix_dataset_path
        self._dataset_kwargs = dict(
            batch_size=batch_size,
            batch_size_per_gp=batch_size_per_gp,
            min_features=min_features,
            max_features=max_features,
            max_classes=max_classes,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            log_seq_len=log_seq_len,
            # The packaged TALENT prior treats seq_len_per_gp as per-sample
            # variation, while this training loop requires every micro-batch to
            # share one seq_len/train_size. Keep TALENT batches shape-safe; the
            # synthetic base prior still receives the original seq_len_per_gp.
            seq_len_per_gp=False,
            min_train_size=min_train_size,
            max_train_size=max_train_size,
            replay_small=replay_small,
            prior_type="talent_single_mix",
            device=device,
        )
        self._module = None
        self._dataset = None

    def __getstate__(self):
        state = self.__dict__.copy()
        # DataLoader forkserver pickles the parent dataset before worker start.
        # Dynamic modules and generator instances from the packaged file are not
        # picklable, so each worker rebuilds them lazily on first batch.
        state["_module"] = None
        state["_dataset"] = None
        return state

    def _ensure_dataset(self):
        if self._dataset is None:
            module = _load_talent_module(self.talent_mix_dataset_path)
            dataset = module.PriorDataset(**self._dataset_kwargs)
            _set_generators_for_source(dataset.prior, module, self.talent_mix_source)
            self._module = module
            self._dataset = dataset
        return self._dataset

    def get_batch(self, batch_size: Optional[int] = None):
        dataset = self._ensure_dataset()
        X, y, d, seq_lens, train_sizes, _graphs = dataset.get_batch(batch_size=batch_size)
        return X, y, d, seq_lens, train_sizes

    def __repr__(self) -> str:
        dataset = self._dataset
        generator_count = len(getattr(dataset.prior, "generators", ())) if dataset is not None else "lazy"
        return (
            "TalentSinglePriorAdapter("
            f"source={self.talent_mix_source!r}, generators={generator_count}, "
            f"dataset={dataset!r})"
        )


class TabICLv2TalentMixPrior:
    """Batch-level interleaving of the default TabICLv2 classifier prior and TALENT-sim."""

    def __init__(
        self,
        *,
        talent_mix_ratio: float,
        talent_mix_source: str = "default51",
        talent_mix_dataset_path: Optional[str] = None,
        **prior_kwargs,
    ):
        self.talent_mix_ratio = float(np.clip(talent_mix_ratio, 0.0, 1.0))
        self.talent_mix_source = talent_mix_source
        self.rng = np.random.default_rng()

        self.base_prior = TabICLv2ClassificationPrior(**prior_kwargs)
        self.talent_prior = TalentSinglePriorAdapter(
            talent_mix_source=talent_mix_source,
            talent_mix_dataset_path=talent_mix_dataset_path,
            **{
                key: value
                for key, value in prior_kwargs.items()
                if key
                in {
                    "batch_size",
                    "batch_size_per_gp",
                    "min_features",
                    "max_features",
                    "max_classes",
                    "min_seq_len",
                    "max_seq_len",
                    "log_seq_len",
                    "seq_len_per_gp",
                    "min_train_size",
                    "max_train_size",
                    "replay_small",
                    "device",
                }
            },
        )

    def get_batch(self, batch_size: Optional[int] = None):
        if self.talent_mix_ratio <= 0.0:
            return self.base_prior.get_batch(batch_size=batch_size)
        if self.talent_mix_ratio >= 1.0 or self.rng.random() < self.talent_mix_ratio:
            return self.talent_prior.get_batch(batch_size=batch_size)
        return self.base_prior.get_batch(batch_size=batch_size)

    def __repr__(self) -> str:
        return (
            "TabICLv2TalentMixPrior("
            f"talent_mix_ratio={self.talent_mix_ratio}, "
            f"talent_mix_source={self.talent_mix_source!r}, "
            f"base_prior={self.base_prior!r}, talent_prior={self.talent_prior!r})"
        )
