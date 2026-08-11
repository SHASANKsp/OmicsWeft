"""Missing-data handling: filtering and imputation.

Some downstream integrators handle missingness natively; others need complete
matrices. These preprocessors cover the common cases with scikit-learn.
"""

from __future__ import annotations

import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer

from ..core.base import Preprocessor
from ..core.omicsdata import OmicsData, _to_dense
from ..core.registry import PREPROCESSORS


@PREPROCESSORS.register("drop_missing_features")
class DropMissingFeatures(Preprocessor):
    """Remove features whose fraction of missing values exceeds ``max_frac``."""

    def __init__(self, max_frac: float = 0.2, modalities: list[str] | None = None) -> None:
        self.max_frac = max_frac
        self.modalities = modalities

    def apply(self, data: OmicsData) -> OmicsData:
        for m in self._targets(data):
            ad = data.get_omic(m)
            X = _to_dense(ad.X)
            frac = np.isnan(X).mean(axis=0)
            keep = frac <= self.max_frac
            data.mdata.mod[m] = ad[:, keep].copy()
        data.mdata.update()
        return data


@PREPROCESSORS.register("impute_simple")
class ImputeSimple(Preprocessor):
    """Constant/mean/median imputation."""

    def __init__(self, strategy: str = "median", modalities: list[str] | None = None) -> None:
        self.strategy = strategy
        self.modalities = modalities

    def apply(self, data: OmicsData) -> OmicsData:
        for m in self._targets(data):
            ad = data.get_omic(m)
            X = _to_dense(ad.X)
            if np.isnan(X).any():
                ad.X = SimpleImputer(strategy=self.strategy).fit_transform(X)
        return data


@PREPROCESSORS.register("impute_knn")
class ImputeKNN(Preprocessor):
    """K-nearest-neighbour imputation (per modality)."""

    def __init__(self, n_neighbors: int = 5, modalities: list[str] | None = None) -> None:
        self.n_neighbors = n_neighbors
        self.modalities = modalities

    def apply(self, data: OmicsData) -> OmicsData:
        for m in self._targets(data):
            ad = data.get_omic(m)
            X = _to_dense(ad.X)
            if np.isnan(X).any():
                k = min(self.n_neighbors, max(1, X.shape[0] - 1))
                ad.X = KNNImputer(n_neighbors=k).fit_transform(X)
        return data
