"""Feature selection.

Unsupervised selectors (variance / MAD) are safe to run before splitting.
The supervised selector uses labels and MUST be applied inside CV folds only —
it is provided for completeness and flagged accordingly.
"""

from __future__ import annotations

import numpy as np

from ..core.base import Preprocessor
from ..core.omicsdata import OmicsData, _to_dense
from ..core.registry import PREPROCESSORS


def _top_k(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, scores.shape[0])
    idx = np.argsort(scores)[::-1][:k]
    keep = np.zeros(scores.shape[0], dtype=bool)
    keep[idx] = True
    return keep


@PREPROCESSORS.register("select_variance")
class SelectVariance(Preprocessor):
    """Keep the top-k highest-variance features per modality."""

    def __init__(self, k: int = 2000, modalities: list[str] | None = None) -> None:
        self.k = k
        self.modalities = modalities

    def apply(self, data: OmicsData) -> OmicsData:
        for m in self._targets(data):
            ad = data.get_omic(m)
            X = _to_dense(ad.X)
            keep = _top_k(np.nanvar(X, axis=0), self.k)
            data.mdata.mod[m] = ad[:, keep].copy()
        data.mdata.update()
        return data


@PREPROCESSORS.register("select_mad")
class SelectMAD(Preprocessor):
    """Keep the top-k features by median absolute deviation (robust)."""

    def __init__(self, k: int = 2000, modalities: list[str] | None = None) -> None:
        self.k = k
        self.modalities = modalities

    def apply(self, data: OmicsData) -> OmicsData:
        for m in self._targets(data):
            ad = data.get_omic(m)
            X = _to_dense(ad.X)
            med = np.nanmedian(X, axis=0, keepdims=True)
            mad = np.nanmedian(np.abs(X - med), axis=0)
            keep = _top_k(mad, self.k)
            data.mdata.mod[m] = ad[:, keep].copy()
        data.mdata.update()
        return data
