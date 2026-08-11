"""MOFA / MOFA+ integrator (via ``mofapy2``).

Multi-Omics Factor Analysis learns a small set of latent factors shared across
omics, each with per-omic feature weights (loadings). It handles partially
overlapping samples and missing values natively, and its factors + loadings are
directly interpretable — which is why it is the first "real" method wired in
after the baselines.

The learned factors become the joint :class:`Embedding`; the per-omic loadings
are stashed in ``embedding.meta['loadings']`` for the interpretation layer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.base import Embedding, Integrator
from ..core.omicsdata import OmicsData, _to_dense
from ..core.registry import INTEGRATORS


@INTEGRATORS.register("mofa")
class MOFA(Integrator):
    """Wrapper around mofapy2's entry point.

    Parameters
    ----------
    n_factors
        Number of latent factors to learn.
    samples
        ``"common"`` (default) restricts to samples shared by all modalities, or
        ``"union"`` to use every sample and let MOFA handle the missing blocks.
    convergence_mode
        mofapy2 convergence mode: ``"fast"``, ``"medium"``, or ``"slow"``.
    """

    def __init__(
        self,
        n_factors: int = 10,
        samples: str = "common",
        scale_views: bool = True,
        convergence_mode: str = "fast",
        max_iter: int = 1000,
        modalities: list[str] | None = None,
        seed: int = 0,
        verbose: bool = False,
    ) -> None:
        self.n_factors = n_factors
        self.samples = samples
        self.scale_views = scale_views
        self.convergence_mode = convergence_mode
        self.max_iter = max_iter
        self.modalities = modalities
        self.seed = seed
        self.verbose = verbose

    def fit_transform(self, data: OmicsData) -> Embedding:
        try:
            from mofapy2.run.entry_point import entry_point
        except Exception as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "MOFA needs mofapy2: pip install mofapy2"
            ) from exc

        mods = self.modalities or data.modalities
        if self.samples == "union":
            sample_idx = data.sample_names
        else:
            sample_idx = data.common_samples(mods)
        if len(sample_idx) < 3:
            raise ValueError("MOFA needs at least 3 samples")

        # mofapy2 wants a list (views) of [group] matrices, samples x features,
        # with NaN for missing entries.
        data_matrix = []
        feature_names = []
        for m in mods:
            ad = data.get_omic(m)
            block = pd.DataFrame(
                _to_dense(ad.X), index=ad.obs_names, columns=ad.var_names
            ).reindex(sample_idx)  # missing samples -> NaN rows
            data_matrix.append([block.to_numpy()])
            feature_names.append(list(ad.var_names))

        ep = entry_point()
        ep.set_data_options(scale_views=self.scale_views)
        ep.set_data_matrix(
            data_matrix,
            views_names=list(mods),
            groups_names=["group0"],
            samples_names=[list(sample_idx)],
            features_names=feature_names,
        )
        ep.set_model_options(factors=self.n_factors)
        ep.set_train_options(
            iter=self.max_iter,
            convergence_mode=self.convergence_mode,
            seed=self.seed,
            verbose=self.verbose,
            quiet=not self.verbose,
        )
        ep.build()
        ep.run()

        # factors: (n_samples x n_factors) for our single group
        expectations = ep.model.getExpectations()
        Z = np.asarray(expectations["Z"]["E"])

        # loadings per view: getExpectations()["W"] is per-view (D_m x n_factors)
        w_node = expectations["W"]
        w_list = w_node["E"] if isinstance(w_node, dict) and "E" in w_node else w_node

        def _as_w(obj):
            if isinstance(obj, dict) and "E" in obj:
                return np.asarray(obj["E"])
            return np.asarray(obj)

        loadings = {}
        for i, m in enumerate(mods):
            w = _as_w(w_list[i])
            loadings[m] = pd.DataFrame(
                w,
                index=feature_names[i],
                columns=[f"factor_{k}" for k in range(w.shape[1])],
            )

        return Embedding(
            X=Z,
            samples=pd.Index(sample_idx),
            method="mofa",
            meta={"modalities": list(mods), "loadings": loadings},
        )
