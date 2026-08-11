"""Batch-effect correction.

The default is a dependency-free linear residualization (the same idea as
limma's ``removeBatchEffect``): regress each feature on the batch indicator and
keep the residuals plus the grand mean. An optional ComBat path is available if
``pycombat`` is installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.base import Preprocessor
from ..core.omicsdata import OmicsData, _to_dense
from ..core.registry import PREPROCESSORS


def _residualize(X: np.ndarray, batch: np.ndarray) -> np.ndarray:
    """Remove additive per-batch means from each feature."""
    out = X.astype(float).copy()
    grand = np.nanmean(out, axis=0, keepdims=True)
    for b in np.unique(batch):
        mask = batch == b
        if mask.sum() == 0:
            continue
        bmean = np.nanmean(out[mask], axis=0, keepdims=True)
        out[mask] = out[mask] - bmean + grand
    return out


@PREPROCESSORS.register("batch_linear")
class BatchLinear(Preprocessor):
    """Linear (mean-shift) batch correction using a sample-sheet column."""

    def __init__(self, batch_key: str, modalities: list[str] | None = None) -> None:
        self.batch_key = batch_key
        self.modalities = modalities

    def apply(self, data: OmicsData) -> OmicsData:
        if not data.has_labels(self.batch_key):
            raise KeyError(
                f"batch_key {self.batch_key!r} not in sample sheet; have "
                f"{list(data.obs.columns)}"
            )
        for m in self._targets(data):
            ad = data.get_omic(m)
            batch = (
                data.labels(self.batch_key)
                .reindex(ad.obs_names)
                .astype("object")
                .to_numpy()
            )
            ad.X = _residualize(_to_dense(ad.X), batch)
        return data


@PREPROCESSORS.register("combat")
class ComBat(Preprocessor):
    """ComBat batch correction (requires the optional ``pycombat`` extra)."""

    def __init__(self, batch_key: str, modalities: list[str] | None = None) -> None:
        self.batch_key = batch_key
        self.modalities = modalities

    def apply(self, data: OmicsData) -> OmicsData:
        try:
            from combat.pycombat import pycombat
        except Exception as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "ComBat needs the optional 'batch' extra: pip install omicsweft[batch]"
            ) from exc
        for m in self._targets(data):
            ad = data.get_omic(m)
            batch = data.labels(self.batch_key).reindex(ad.obs_names)
            # pycombat expects features x samples
            df = pd.DataFrame(_to_dense(ad.X).T, columns=ad.obs_names)
            corrected = pycombat(df, batch.to_list())
            ad.X = corrected.to_numpy().T
        return data
