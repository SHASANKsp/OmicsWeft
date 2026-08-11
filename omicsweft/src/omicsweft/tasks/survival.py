"""OPTIONAL survival task — the one place Kaplan-Meier / log-rank lives.

This is deliberately a plugin, not core. It is only usable when a dataset carries
time-to-event labels, and it requires the optional ``survival`` extra
(``pip install omicsweft[survival]``) which pulls in lifelines. A dataset without
time/event columns (e.g. an IBD cohort) never touches this module — that is the
disease-agnostic guarantee, enforced by construction.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from ..core.base import Embedding, Task
from ..core.omicsdata import OmicsData
from ..core.registry import TASKS


@TASKS.register("survival")
class SurvivalTask(Task):
    """Stratify samples by clustering the embedding, then test separation.

    Metrics: multivariate log-rank p-value across clusters, and a concordance
    index using cluster assignment as a risk proxy.
    """

    kind = "survival"

    def __init__(
        self,
        time_key: str,
        event_key: str,
        n_groups: int = 2,
        random_state: int = 0,
    ) -> None:
        self.time_key = time_key
        self.event_key = event_key
        self.n_groups = n_groups
        self.random_state = random_state

    def evaluate(self, embedding: Embedding, data: OmicsData) -> dict:
        try:
            from lifelines.statistics import multivariate_logrank_test
            from lifelines.utils import concordance_index
        except Exception as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "the survival task needs the optional 'survival' extra: "
                "pip install omicsweft[survival]"
            ) from exc

        t = data.labels(self.time_key).reindex(embedding.samples).to_numpy(dtype=float)
        e = data.labels(self.event_key).reindex(embedding.samples).to_numpy(dtype=float)
        mask = ~(np.isnan(t) | np.isnan(e))
        X, t, e = embedding.X[mask], t[mask], e[mask]
        if len(t) < self.n_groups + 1:
            raise ValueError("not enough samples with survival labels")

        groups = KMeans(
            n_clusters=self.n_groups, n_init=10, random_state=self.random_state
        ).fit_predict(X)

        lr = multivariate_logrank_test(t, groups, e)
        # C-index: use group index as an ordinal risk score
        c_index = concordance_index(t, -groups.astype(float), e)
        return {
            "n_groups": float(self.n_groups),
            "logrank_p": float(lr.p_value),
            "logrank_stat": float(lr.test_statistic),
            "c_index": float(c_index),
        }
