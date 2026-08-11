"""The one in-memory container everything else operates on.

``OmicsData`` wraps a :class:`mudata.MuData` (one AnnData per omic over a shared
sample axis) and adds the conveniences the rest of the package needs: a reliable
sample sheet, matrix access, sample subsetting, and a coverage summary that makes
partial overlap (a sample missing some omics) a first-class, non-error state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from anndata import AnnData
from mudata import MuData

try:  # adopt mudata >=0.4 semantics early; we track the sample sheet ourselves
    import mudata as _mudata

    _mudata.set_options(pull_on_update=False)
except Exception:  # pragma: no cover - older/newer mudata without the option
    pass


def _to_dense(x) -> np.ndarray:
    """Return a dense float array from a possibly-sparse matrix."""
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=float)


@dataclass
class CoverageSummary:
    n_samples: int
    n_modalities: int
    per_modality: dict[str, int]
    n_complete: int  # samples present in every modality
    combo_counts: dict[str, int]  # e.g. {"rna+protein": 12}

    def as_frame(self) -> pd.DataFrame:
        rows = [{"modality": m, "n_samples": n} for m, n in self.per_modality.items()]
        return pd.DataFrame(rows)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        lines = [
            (
                f"OmicsData coverage: {self.n_samples} samples across "
                f"{self.n_modalities} modalities"
            ),
            f"  complete (all modalities): {self.n_complete}/{self.n_samples}",
        ]
        for m, n in self.per_modality.items():
            lines.append(f"  {m}: {n} samples")
        return "\n".join(lines)


class OmicsData:
    """Container for a multi-omics dataset.

    Parameters
    ----------
    mdata
        A MuData with one modality per omic. Sample identifiers are the shared
        ``obs_names`` axis; modalities need not cover the same samples.
    sample_sheet
        Optional per-sample table (labels, batch, covariates), indexed by
        sample id. If omitted, ``mdata.obs`` is used.
    name
        Optional dataset name (used in run manifests and benchmark tables).
    """

    def __init__(
        self,
        mdata: MuData,
        sample_sheet: pd.DataFrame | None = None,
        name: str = "dataset",
    ) -> None:
        self.mdata = mdata
        self.name = name
        if sample_sheet is None:
            sample_sheet = mdata.obs.copy()
        # Align the sheet to the union sample axis; keep every sample even if it
        # has no metadata (values become NaN).
        self._sheet = sample_sheet.reindex(mdata.obs_names)

    # ------------------------------------------------------------------ #
    # construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_matrices(
        cls,
        matrices: dict[str, pd.DataFrame],
        sample_sheet: pd.DataFrame | None = None,
        name: str = "dataset",
    ) -> OmicsData:
        """Build from ``{omic_name: DataFrame(samples x features)}``.

        Each DataFrame is indexed by sample id; blocks may cover different
        (overlapping) sample sets. This is the in-memory equivalent of the
        on-disk generic format used by the loader.
        """
        if not matrices:
            raise ValueError("at least one omic matrix is required")

        mods: dict[str, AnnData] = {}
        for omic, df in matrices.items():
            if df.index.has_duplicates:
                raise ValueError(f"omic {omic!r} has duplicate sample ids")
            if df.columns.has_duplicates:
                raise ValueError(f"omic {omic!r} has duplicate feature ids")
            ad = AnnData(
                X=df.to_numpy(dtype=float),
                obs=pd.DataFrame(index=df.index.astype(str)),
                var=pd.DataFrame(index=df.columns.astype(str)),
            )
            mods[omic] = ad

        mdata = MuData(mods)
        if sample_sheet is not None:
            sample_sheet = sample_sheet.copy()
            sample_sheet.index = sample_sheet.index.astype(str)
        return cls(mdata, sample_sheet=sample_sheet, name=name)

    # ------------------------------------------------------------------ #
    # basic accessors
    # ------------------------------------------------------------------ #
    @property
    def modalities(self) -> list[str]:
        return list(self.mdata.mod)

    @property
    def sample_names(self) -> pd.Index:
        return self.mdata.obs_names

    @property
    def n_samples(self) -> int:
        return self.mdata.n_obs

    @property
    def obs(self) -> pd.DataFrame:
        """The sample sheet (labels / covariates), aligned to the sample axis."""
        return self._sheet

    def get_omic(self, modality: str) -> AnnData:
        if modality not in self.mdata.mod:
            raise KeyError(f"no modality {modality!r}; have {self.modalities}")
        return self.mdata.mod[modality]

    def matrix(self, modality: str) -> tuple[np.ndarray, pd.Index]:
        """Return ``(X, sample_index)`` for one modality (dense)."""
        ad = self.get_omic(modality)
        return _to_dense(ad.X), ad.obs_names

    def labels(self, key: str) -> pd.Series:
        """Return one column of the sample sheet as a Series (by sample id)."""
        if key not in self._sheet.columns:
            raise KeyError(
                f"no column {key!r} in sample sheet; have "
                f"{list(self._sheet.columns)}"
            )
        return self._sheet[key]

    def has_labels(self, key: str) -> bool:
        return key in self._sheet.columns

    # ------------------------------------------------------------------ #
    # sample-overlap logic
    # ------------------------------------------------------------------ #
    def common_samples(self, modalities: list[str] | None = None) -> pd.Index:
        """Sample ids present in *all* of the requested modalities."""
        mods = modalities or self.modalities
        idx: pd.Index | None = None
        for m in mods:
            names = self.get_omic(m).obs_names
            idx = names if idx is None else idx.intersection(names)
        return idx if idx is not None else pd.Index([])

    def concat_matrix(
        self, modalities: list[str] | None = None, samples: str | pd.Index = "common"
    ) -> tuple[np.ndarray, pd.Index, dict[str, slice]]:
        """Concatenate modalities feature-wise over a shared sample set.

        Returns the stacked matrix, the sample index, and a map from modality to
        its column slice (useful for per-omic interpretation downstream).
        """
        mods = modalities or self.modalities
        if isinstance(samples, str):
            if samples != "common":
                raise ValueError("samples must be 'common' or an Index")
            sample_idx = self.common_samples(mods)
        else:
            sample_idx = pd.Index(samples)
        if len(sample_idx) == 0:
            raise ValueError(
                "no samples are shared across the requested modalities "
                f"({mods}); integration needs an overlapping set"
            )

        blocks, slices, start = [], {}, 0
        for m in mods:
            ad = self.get_omic(m)
            sub = ad[sample_idx]
            block = _to_dense(sub.X)
            blocks.append(block)
            slices[m] = slice(start, start + block.shape[1])
            start += block.shape[1]
        return np.hstack(blocks), sample_idx, slices

    def copy(self) -> OmicsData:
        """Return a deep copy (uses AnnData.copy, which deepcopy cannot do safely)."""
        mods = {m: self.get_omic(m).copy() for m in self.modalities}
        md = MuData(mods)
        return OmicsData(md, sample_sheet=self._sheet.copy(), name=self.name)

    def subset_samples(self, samples: pd.Index) -> OmicsData:
        """Return a new OmicsData restricted to the given sample ids."""
        samples = pd.Index(samples).astype(str)
        mods = {}
        for m in self.modalities:
            ad = self.get_omic(m)
            keep = ad.obs_names.intersection(samples)
            mods[m] = ad[keep].copy()
        md = MuData(mods)
        sheet = self._sheet.reindex(samples.intersection(self._sheet.index))
        return OmicsData(md, sample_sheet=sheet, name=self.name)

    # ------------------------------------------------------------------ #
    # diagnostics
    # ------------------------------------------------------------------ #
    def coverage(self) -> CoverageSummary:
        per_mod = {m: int(self.get_omic(m).n_obs) for m in self.modalities}
        # membership per sample
        membership: dict[str, list[str]] = {}
        for m in self.modalities:
            for s in self.get_omic(m).obs_names:
                membership.setdefault(s, []).append(m)
        n_complete = sum(
            1 for mods in membership.values() if len(mods) == len(self.modalities)
        )
        combo_counts: dict[str, int] = {}
        for mods in membership.values():
            key = "+".join(sorted(mods))
            combo_counts[key] = combo_counts.get(key, 0) + 1
        return CoverageSummary(
            n_samples=self.n_samples,
            n_modalities=len(self.modalities),
            per_modality=per_mod,
            n_complete=n_complete,
            combo_counts=dict(sorted(combo_counts.items(), key=lambda kv: -kv[1])),
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        shapes = ", ".join(
            f"{m}:{self.get_omic(m).shape}" for m in self.modalities
        )
        return f"OmicsData(name={self.name!r}, n_samples={self.n_samples}, {shapes})"
