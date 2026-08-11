"""Joint / integrative Non-negative Matrix Factorization (from scratch).

Each non-negative omic block ``X_m (samples x features)`` is factorized as
``W @ H_m`` with a **shared** sample-coefficient matrix ``W (samples x k)`` and a
per-omic basis ``H_m (k x features)``. The shared ``W`` is the joint embedding;
the ``H_m`` are interpretable per-omic loadings. This is the multiplicative-update
intNMF idea, implemented in numpy so it has no external dependency.

Inputs are shifted to be non-negative per block if needed (NMF requires it).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.base import Embedding, Integrator
from ..core.omicsdata import OmicsData
from ..core.registry import INTEGRATORS

_EPS = 1e-10


@INTEGRATORS.register("jnmf")
class JointNMF(Integrator):
    def __init__(
        self,
        n_components: int = 10,
        max_iter: int = 200,
        tol: float = 1e-4,
        modalities: list[str] | None = None,
        random_state: int = 0,
    ) -> None:
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.modalities = modalities
        self.random_state = random_state

    def fit_transform(self, data: OmicsData) -> Embedding:
        mods = self.modalities or data.modalities
        samples = data.common_samples(mods)
        if len(samples) < self.n_components:
            raise ValueError("jnmf needs at least n_components shared samples")

        # per-block non-negative matrices over the shared sample set
        blocks = []
        for m in mods:
            X = np.asarray(data.get_omic(m)[samples].X, dtype=float)
            X = np.nan_to_num(X, nan=0.0)
            mn = X.min()
            if mn < 0:
                X = X - mn  # shift to non-negative
            blocks.append(X)

        rng = np.random.default_rng(self.random_state)
        n = len(samples)
        k = self.n_components
        W = rng.random((n, k)) + _EPS
        Hs = [rng.random((k, B.shape[1])) + _EPS for B in blocks]

        prev_err = np.inf
        for _ in range(self.max_iter):
            # update each H_m: H <- H * (W^T X) / (W^T W H)
            for i, B in enumerate(blocks):
                num = W.T @ B
                den = (W.T @ W @ Hs[i]) + _EPS
                Hs[i] *= num / den
            # update shared W using all blocks jointly
            num = sum(B @ Hs[i].T for i, B in enumerate(blocks))
            den = W @ sum(Hs[i] @ Hs[i].T for i in range(len(blocks))) + _EPS
            W *= num / den

            err = sum(
                float(np.linalg.norm(B - W @ Hs[i])) for i, B in enumerate(blocks)
            )
            if abs(prev_err - err) / (prev_err + _EPS) < self.tol:
                break
            prev_err = err

        loadings = {
            m: pd.DataFrame(
                Hs[i].T,
                index=list(data.get_omic(m).var_names),
                columns=[f"factor_{j}" for j in range(k)],
            )
            for i, m in enumerate(mods)
        }
        return Embedding(
            X=W,
            samples=pd.Index(samples),
            method="jnmf",
            meta={"modalities": list(mods), "loadings": loadings},
        )
