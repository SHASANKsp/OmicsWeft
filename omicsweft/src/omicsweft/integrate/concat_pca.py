"""Concatenation + PCA — the simplest sensible integration baseline.

Standardizes each modality, concatenates features over the shared sample set,
and reduces to ``k`` components. It is deliberately included as a baseline every
other method should be measured against.
"""

from __future__ import annotations

from sklearn.decomposition import PCA

from ..core.base import Embedding, Integrator
from ..core.omicsdata import OmicsData
from ..core.registry import INTEGRATORS


@INTEGRATORS.register("concat_pca")
class ConcatPCA(Integrator):
    def __init__(
        self,
        n_components: int = 10,
        modalities: list[str] | None = None,
        standardize: bool = True,
        random_state: int = 0,
    ) -> None:
        self.n_components = n_components
        self.modalities = modalities
        self.standardize = standardize
        self.random_state = random_state

    def fit_transform(self, data: OmicsData) -> Embedding:
        mods = self.modalities or data.modalities
        X, samples, slices = data.concat_matrix(mods, samples="common")

        if self.standardize:
            # standardize within each modality block to prevent one omic (or a
            # larger block) from dominating the shared components
            X = X.copy()
            for sl in slices.values():
                block = X[:, sl]
                mu = block.mean(axis=0, keepdims=True)
                sd = block.std(axis=0, keepdims=True)
                sd[sd == 0] = 1.0
                X[:, sl] = (block - mu) / sd

        k = min(self.n_components, min(X.shape) - 1)
        k = max(k, 1)
        emb = PCA(n_components=k, random_state=self.random_state).fit_transform(X)
        return Embedding(
            X=emb,
            samples=samples,
            method="concat_pca",
            meta={"modalities": list(mods), "block_slices": slices},
        )
