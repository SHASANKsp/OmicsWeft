"""Per-omic normalization preprocessors.

Each is a small, composable Preprocessor. Different omics want different
transforms (log-CPM for counts, standardization for continuous intensities,
arcsinh for heavy-tailed data), so normalization is applied per modality.
"""

from __future__ import annotations

import numpy as np

from ..core.base import Preprocessor
from ..core.omicsdata import OmicsData, _to_dense
from ..core.registry import PREPROCESSORS


def _set_X(data: OmicsData, modality: str, X: np.ndarray) -> None:
    data.get_omic(modality).X = X


@PREPROCESSORS.register("log1p")
class Log1p(Preprocessor):
    """Natural log of (1 + x). Safe for non-negative data."""

    def __init__(self, modalities: list[str] | None = None) -> None:
        self.modalities = modalities

    def apply(self, data: OmicsData) -> OmicsData:
        for m in self._targets(data):
            X = _to_dense(data.get_omic(m).X)
            _set_X(data, m, np.log1p(np.clip(X, a_min=0, a_max=None)))
        return data


@PREPROCESSORS.register("total_count")
class TotalCount(Preprocessor):
    """Library-size normalization to a target sum (CPM-style), then optional log."""

    def __init__(
        self, target: float = 1e6, log: bool = True, modalities: list[str] | None = None
    ) -> None:
        self.target = target
        self.log = log
        self.modalities = modalities

    def apply(self, data: OmicsData) -> OmicsData:
        for m in self._targets(data):
            X = _to_dense(data.get_omic(m).X)
            sizes = X.sum(axis=1, keepdims=True)
            sizes[sizes == 0] = 1.0
            Xn = X / sizes * self.target
            if self.log:
                Xn = np.log1p(Xn)
            _set_X(data, m, Xn)
        return data


@PREPROCESSORS.register("standardize")
class Standardize(Preprocessor):
    """Zero-mean, unit-variance per feature (z-score)."""

    def __init__(self, modalities: list[str] | None = None) -> None:
        self.modalities = modalities

    def apply(self, data: OmicsData) -> OmicsData:
        for m in self._targets(data):
            X = _to_dense(data.get_omic(m).X)
            mu = np.nanmean(X, axis=0, keepdims=True)
            sd = np.nanstd(X, axis=0, keepdims=True)
            sd[sd == 0] = 1.0
            _set_X(data, m, (X - mu) / sd)
        return data


@PREPROCESSORS.register("arcsinh")
class Arcsinh(Preprocessor):
    """Inverse hyperbolic sine with a cofactor — a gentle, zero-safe log-like
    transform for heavy-tailed intensities (e.g. proteomics)."""

    def __init__(self, cofactor: float = 5.0, modalities: list[str] | None = None) -> None:
        self.cofactor = cofactor
        self.modalities = modalities

    def apply(self, data: OmicsData) -> OmicsData:
        for m in self._targets(data):
            X = _to_dense(data.get_omic(m).X)
            _set_X(data, m, np.arcsinh(X / self.cofactor))
        return data
