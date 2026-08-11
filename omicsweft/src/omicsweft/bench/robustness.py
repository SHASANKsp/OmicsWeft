"""Robustness benchmarking.

Perturbs a dataset (Gaussian feature noise or random feature dropout) at several
levels, re-runs an integrator + task, and reports how a chosen metric degrades.
A method that holds its metric under perturbation is more trustworthy than one
that only wins on pristine data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.omicsdata import OmicsData, _to_dense
from ..core.registry import INTEGRATORS, TASKS
from ..core.utils import get_logger, set_seed

_LOG = get_logger("omicsweft.robust")


def _perturb(data: OmicsData, kind: str, level: float, rng) -> OmicsData:
    out = data.copy()
    for m in out.modalities:
        ad = out.get_omic(m)
        X = _to_dense(ad.X).copy()
        if kind == "noise":
            sd = np.nanstd(X, axis=0, keepdims=True)
            X = X + rng.normal(0, level, X.shape) * sd
        elif kind == "dropout":
            drop = rng.random(X.shape) < level
            X[drop] = 0.0
        else:
            raise ValueError("kind must be 'noise' or 'dropout'")
        ad.X = X
    return out


def robustness_curve(
    data: OmicsData,
    integrator: str,
    task: str,
    metric: str,
    levels=(0.0, 0.25, 0.5, 1.0),
    kind: str = "noise",
    n_repeats: int = 3,
    integrator_params: dict | None = None,
    task_params: dict | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Return a tidy table of ``metric`` vs perturbation ``level`` (mean +/- std)."""
    integrator_params = integrator_params or {}
    task_params = task_params or {}
    rng = np.random.default_rng(seed)

    rows = []
    for level in levels:
        vals = []
        for rep in range(n_repeats):
            set_seed(seed + rep)
            pert = data if level == 0 else _perturb(data, kind, level, rng)
            emb = INTEGRATORS.create(integrator, **integrator_params).fit_transform(pert)
            res = TASKS.create(task, **task_params).evaluate(emb, pert)
            if metric in res:
                vals.append(res[metric])
        if vals:
            rows.append(
                {"level": level, "kind": kind, f"{metric}_mean": float(np.mean(vals)),
                 f"{metric}_std": float(np.std(vals)), "n": len(vals)}
            )
    df = pd.DataFrame(rows)
    _LOG.info("robustness (%s, %s vs %s):\n%s", integrator, metric, kind,
              df.to_string(index=False))
    return df
