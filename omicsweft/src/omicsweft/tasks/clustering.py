"""Clustering task + metrics (domain-free)."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from ..core.base import Embedding, Task
from ..core.omicsdata import OmicsData
from ..core.registry import TASKS


@TASKS.register("clustering")
class ClusteringTask(Task):
    """KMeans on the embedding; internal + (optional) external metrics.

    Parameters
    ----------
    n_clusters
        Number of clusters. If ``None`` and ``label_key`` is set, uses the number
        of distinct labels.
    label_key
        Optional sample-sheet column with ground-truth groups (enables ARI/NMI).
    """

    kind = "clustering"

    def __init__(
        self,
        n_clusters: int | None = None,
        label_key: str | None = None,
        random_state: int = 0,
    ) -> None:
        self.n_clusters = n_clusters
        self.label_key = label_key
        self.random_state = random_state

    def evaluate(self, embedding: Embedding, data: OmicsData) -> dict:
        X = embedding.X
        y_true = None
        if self.label_key and data.has_labels(self.label_key):
            y_true = (
                data.labels(self.label_key).reindex(embedding.samples).to_numpy()
            )

        k = self.n_clusters
        if k is None:
            if y_true is None:
                raise ValueError("clustering needs n_clusters or a label_key")
            k = len(np.unique(y_true[~_isnan(y_true)]))

        km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
        pred = km.fit_predict(X)

        out: dict[str, float] = {"n_clusters": float(k)}
        if X.shape[0] > k >= 2:
            out["silhouette"] = float(silhouette_score(X, pred))
            out["davies_bouldin"] = float(davies_bouldin_score(X, pred))
        if y_true is not None:
            mask = ~_isnan(y_true)
            out["ari"] = float(adjusted_rand_score(y_true[mask], pred[mask]))
            out["nmi"] = float(
                normalized_mutual_info_score(y_true[mask], pred[mask])
            )
        return out


def _isnan(a: np.ndarray) -> np.ndarray:
    """NaN mask that works for object/string arrays too."""
    try:
        return np.isnan(a.astype(float))
    except (TypeError, ValueError):
        return np.array([x is None or (isinstance(x, float) and np.isnan(x)) for x in a])
