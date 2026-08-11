"""Interpretation layer.

Turns an :class:`Embedding` (and the data it came from) into ranked features and
biological summaries. Interpreters register under the INTERPRETERS registry.

  - FactorLoadings      top features per latent factor, for methods that expose
                        loadings (MOFA, JointNMF) via ``embedding.meta['loadings']``
  - EmbeddingCorrelation model-agnostic: correlate raw features with embedding
                        dimensions to rank features for *any* integrator
  - PermutationImportance task-driven feature importance on the embedding
  - Enrichment          optional GO/pathway over-representation (needs gseapy)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.base import Embedding, Interpreter
from ..core.omicsdata import OmicsData, _to_dense
from ..core.registry import INTERPRETERS


@INTERPRETERS.register("factor_loadings")
class FactorLoadings(Interpreter):
    """Top-``n`` features per factor from a method's loading matrices."""

    def __init__(self, top_n: int = 20) -> None:
        self.top_n = top_n

    def interpret(self, embedding: Embedding, data: OmicsData) -> dict:
        loadings = embedding.meta.get("loadings")
        if not loadings:
            raise ValueError(
                f"integrator {embedding.method!r} exposes no loadings; use the "
                "embedding_correlation interpreter instead"
            )
        out: dict[str, dict[str, pd.Series]] = {}
        for modality, L in loadings.items():
            per_factor = {}
            for factor in L.columns:
                s = L[factor].abs().sort_values(ascending=False).head(self.top_n)
                per_factor[factor] = L[factor].loc[s.index]
            out[modality] = per_factor
        return {"top_features": out}


@INTERPRETERS.register("embedding_correlation")
class EmbeddingCorrelation(Interpreter):
    """Rank features by absolute correlation with embedding dimensions.

    Works for any integrator (including deep ones with no explicit loadings).
    """

    def __init__(self, top_n: int = 20, modalities: list[str] | None = None) -> None:
        self.top_n = top_n
        self.modalities = modalities

    def interpret(self, embedding: Embedding, data: OmicsData) -> dict:
        mods = self.modalities or data.modalities
        Z = embedding.X
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
        out: dict[str, pd.DataFrame] = {}
        for m in mods:
            ad = data.get_omic(m)
            common = ad.obs_names.intersection(embedding.samples)
            if len(common) < 3:
                continue
            X = _to_dense(ad[common].X)
            X = np.nan_to_num(X, nan=0.0)
            Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
            Zc = Z[[embedding.samples.get_loc(s) for s in common]]
            corr = (Xs.T @ Zc) / len(common)  # (features x k)
            strength = np.abs(corr).max(axis=1)
            top = pd.DataFrame(
                {"feature": ad.var_names, "max_abs_corr": strength}
            ).sort_values("max_abs_corr", ascending=False).head(self.top_n)
            out[m] = top.reset_index(drop=True)
        return {"top_features": out}


@INTERPRETERS.register("permutation_importance")
class PermutationImportance(Interpreter):
    """Permutation importance of embedding dimensions for a labelled task."""

    def __init__(self, label_key: str, n_repeats: int = 10, random_state: int = 0) -> None:
        self.label_key = label_key
        self.n_repeats = n_repeats
        self.random_state = random_state

    def interpret(self, embedding: Embedding, data: OmicsData) -> dict:
        from sklearn.inspection import permutation_importance
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder

        y_raw = data.labels(self.label_key).reindex(embedding.samples)
        mask = y_raw.notna().to_numpy()
        X = embedding.X[mask]
        y = LabelEncoder().fit_transform(y_raw[mask].astype(str))
        clf = LogisticRegression(max_iter=2000).fit(X, y)
        r = permutation_importance(
            clf, X, y, n_repeats=self.n_repeats, random_state=self.random_state
        )
        imp = pd.DataFrame(
            {"dim": [f"dim_{i}" for i in range(X.shape[1])],
             "importance": r.importances_mean,
             "std": r.importances_std}
        ).sort_values("importance", ascending=False).reset_index(drop=True)
        return {"dimension_importance": imp}


@INTERPRETERS.register("enrichment")
class Enrichment(Interpreter):
    """GO / pathway over-representation of top features (needs the 'enrich' extra)."""

    def __init__(
        self,
        gene_sets: str = "GO_Biological_Process_2021",
        top_n: int = 100,
        modality: str | None = None,
    ) -> None:
        self.gene_sets = gene_sets
        self.top_n = top_n
        self.modality = modality

    def interpret(self, embedding: Embedding, data: OmicsData) -> dict:
        try:
            import gseapy as gp
        except Exception as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "enrichment needs the 'enrich' extra: pip install omicsweft[enrich]"
            ) from exc

        base = EmbeddingCorrelation(top_n=self.top_n, modalities=(
            [self.modality] if self.modality else None
        )).interpret(embedding, data)["top_features"]
        modality = self.modality or next(iter(base))
        genes = list(base[modality]["feature"].astype(str))
        res = gp.enrichr(gene_list=genes, gene_sets=self.gene_sets, outdir=None)
        return {"enrichment": res.results}
