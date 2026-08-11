"""Similarity Network Fusion (Wang et al., 2014), implemented from scratch.

For each modality we build a sample-similarity network, then iteratively fuse
them via cross-diffusion into a single network. A spectral embedding of the
fused network is returned so the result plugs into the standard Embedding /
Task interface (KMeans on this embedding approximates spectral clustering).

Kept dependency-free (numpy/scipy) so it runs in the core with no external
package API to track. ``snfpy`` can be swapped in later behind the same class.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from ..core.base import Embedding, Integrator
from ..core.omicsdata import OmicsData
from ..core.registry import INTEGRATORS


def _affinity(X: np.ndarray, k: int, mu: float) -> np.ndarray:
    """Scaled-exponential affinity with a local-scaling bandwidth."""
    d = cdist(X, X, metric="euclidean")
    n = d.shape[0]
    k = min(k, n - 1)
    # mean distance to k nearest neighbours (excluding self) per sample
    order = np.sort(d, axis=1)
    knn_mean = order[:, 1 : k + 1].mean(axis=1)
    eps = 1e-10
    sigma = (knn_mean[:, None] + knn_mean[None, :] + d) / 3.0 * mu + eps
    W = np.exp(-(d**2) / (2.0 * sigma**2))
    return W


def _normalized(W: np.ndarray) -> np.ndarray:
    W = W.copy()
    np.fill_diagonal(W, 0.0)
    row = W.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    P = W / (2.0 * row)
    np.fill_diagonal(P, 0.5)
    return P


def _knn_kernel(P: np.ndarray, k: int) -> np.ndarray:
    n = P.shape[0]
    k = min(k, n - 1)
    S = np.zeros_like(P)
    for i in range(n):
        idx = np.argsort(P[i])[::-1][:k]
        S[i, idx] = P[i, idx]
    rs = S.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return S / rs


@INTEGRATORS.register("snf")
class SNF(Integrator):
    def __init__(
        self,
        n_components: int = 10,
        k: int = 20,
        mu: float = 0.5,
        n_iters: int = 20,
        modalities: list[str] | None = None,
        random_state: int = 0,
    ) -> None:
        self.n_components = n_components
        self.k = k
        self.mu = mu
        self.n_iters = n_iters
        self.modalities = modalities
        self.random_state = random_state

    def fit_transform(self, data: OmicsData) -> Embedding:
        mods = self.modalities or data.modalities
        samples = data.common_samples(mods)
        if len(samples) < 3:
            raise ValueError("SNF needs at least 3 shared samples")

        # per-omic transition + knn kernels
        P, S = [], []
        for m in mods:
            ad = data.get_omic(m)[samples]
            X = np.asarray(ad.X, dtype=float)
            W = _affinity(X, self.k, self.mu)
            Pm = _normalized(W)
            P.append(Pm)
            S.append(_knn_kernel(Pm, self.k))

        # cross-diffusion fusion
        n_views = len(P)
        for _ in range(self.n_iters):
            P_next = []
            for v in range(n_views):
                others = sum(P[j] for j in range(n_views) if j != v) / max(
                    n_views - 1, 1
                )
                Pv = S[v] @ others @ S[v].T
                P_next.append(_normalized(Pv))
            P = P_next
        fused = sum(P) / n_views
        fused = (fused + fused.T) / 2.0

        # spectral embedding of the fused network (normalized Laplacian)
        deg = fused.sum(axis=1)
        deg[deg == 0] = 1e-10
        d_inv_sqrt = 1.0 / np.sqrt(deg)
        L = np.eye(fused.shape[0]) - (d_inv_sqrt[:, None] * fused * d_inv_sqrt[None, :])
        _, vecs = np.linalg.eigh(L)
        k = min(self.n_components, fused.shape[0] - 1)
        emb = vecs[:, 1 : k + 1]  # skip the trivial first eigenvector

        return Embedding(
            X=emb,
            samples=samples,
            method="snf",
            meta={"modalities": list(mods), "fused_affinity": fused},
        )
