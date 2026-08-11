"""Passthrough loaders for datasets already stored as AnnData / MuData."""

from __future__ import annotations

from pathlib import Path

from anndata import read_h5ad
from mudata import MuData, read_h5mu

from ..core.omicsdata import OmicsData
from ..core.registry import DATASETS


@DATASETS.register("h5mu")
def load_h5mu(path: str | Path, name: str | None = None) -> OmicsData:
    """Load a MuData ``.h5mu`` file directly."""
    mdata = read_h5mu(path)
    return OmicsData(mdata, name=name or Path(path).stem)


@DATASETS.register("h5ad")
def load_h5ad(
    path: str | Path, modality: str | None = None, name: str | None = None
) -> OmicsData:
    """Load a single ``.h5ad`` as a one-modality OmicsData."""
    ad = read_h5ad(path)
    mod = modality or "omic"
    mdata = MuData({mod: ad})
    return OmicsData(mdata, sample_sheet=ad.obs.copy(), name=name or Path(path).stem)
