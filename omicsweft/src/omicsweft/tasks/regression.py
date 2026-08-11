"""Supervised regression task on a joint embedding (domain-free).

Useful for continuous phenotypes (e.g. a metabolic marker), which is exactly the
kind of non-cancer target that shows the core is not survival-centric.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from ..core.base import Embedding, Task
from ..core.omicsdata import OmicsData
from ..core.registry import TASKS


def _make_model(name: str, random_state: int):
    if name == "ridge":
        return Ridge()
    if name == "rf":
        return RandomForestRegressor(n_estimators=300, random_state=random_state)
    raise ValueError(f"unknown model {name!r}; use 'ridge' or 'rf'")


@TASKS.register("regression")
class RegressionTask(Task):
    kind = "regression"

    def __init__(
        self,
        target_key: str,
        model: str = "ridge",
        n_splits: int = 5,
        random_state: int = 0,
    ) -> None:
        self.target_key = target_key
        self.model = model
        self.n_splits = n_splits
        self.random_state = random_state

    def evaluate(self, embedding: Embedding, data: OmicsData) -> dict:
        y_raw = data.labels(self.target_key).reindex(embedding.samples)
        y = y_raw.to_numpy(dtype=float)
        mask = ~np.isnan(y)
        X, y = embedding.X[mask], y[mask]
        if len(y) < self.n_splits:
            raise ValueError("not enough labelled samples for regression CV")

        kf = KFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        pred = np.empty_like(y)
        for tr, te in kf.split(X):
            reg = _make_model(self.model, self.random_state)
            reg.fit(X[tr], y[tr])
            pred[te] = reg.predict(X[te])

        return {
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": float(mean_absolute_error(y, pred)),
            "r2": float(r2_score(y, pred)),
        }
