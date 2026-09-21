"""Synthetic prior data generation for TabICL pre-training.

Only :class:`PriorDataset` is public. All other symbols in this subpackage
are internal pre-training utilities and may change without notice. A CLI
entry point is provided via ``python -m tabicl.prior``.
"""

from ._dataset import PriorDataset

__all__ = ["PriorDataset"]

# Frozen T25 task, installed identically in every generator process.
from ._t25_safe_tail import install as _install_t25_safe_tail
_install_t25_safe_tail()
