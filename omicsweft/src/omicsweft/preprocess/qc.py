"""Quality-control diagnostics.

QC does not alter data by default; it computes per-modality statistics and a
sample-level outlier flag, attaching a report to ``data.mdata.uns['qc']`` and
optionally dropping flagged outlier samples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.base import Preprocessor
from ..core.omicsdata import OmicsData, _to_dense
from ..core.registry import PREPROCESSORS
from ..core.utils import get_logger

_LOG = get_logger("omicsweft.qc")


def compute_qc(data: OmicsData) -> pd.DataFrame:
    """Return a per-modality QC table."""
    rows = []
    for m in data.modalities:
        X = _to_dense(data.get_omic(m).X)
        rows.append(
            {
                "modality": m,
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "pct_missing": float(np.isnan(X).mean() * 100),
                "mean": float(np.nanmean(X)),
                "std": float(np.nanstd(X)),
                "min": float(np.nanmin(X)),
                "max": float(np.nanmax(X)),
            }
        )
    return pd.DataFrame(rows)


@PREPROCESSORS.register("qc")
class QC(Preprocessor):
    """Compute QC stats; optionally drop per-modality outlier samples.

    Outliers are samples whose per-sample mean is more than ``z_thresh`` robust
    z-scores from the modality median (median / MAD based).
    """

    def __init__(
        self,
        drop_outliers: bool = False,
        z_thresh: float = 5.0,
        modalities: list[str] | None = None,
    ) -> None:
        self.drop_outliers = drop_outliers
        self.z_thresh = z_thresh
        self.modalities = modalities

    def apply(self, data: OmicsData) -> OmicsData:
        report = compute_qc(data)
        data.mdata.uns["qc"] = report
        _LOG.info("QC report:\n%s", report.to_string(index=False))

        if not self.drop_outliers:
            return data

        for m in self._targets(data):
            ad = data.get_omic(m)
            X = _to_dense(ad.X)
            sample_mean = np.nanmean(X, axis=1)
            med = np.median(sample_mean)
            mad = np.median(np.abs(sample_mean - med)) or 1.0
            z = 0.6745 * (sample_mean - med) / mad
            keep = np.abs(z) <= self.z_thresh
            if (~keep).any():
                _LOG.warning(
                    "%s: dropping %d outlier sample(s)", m, int((~keep).sum())
                )
                data.mdata.mod[m] = ad[keep].copy()
        data.mdata.update()
        return data
