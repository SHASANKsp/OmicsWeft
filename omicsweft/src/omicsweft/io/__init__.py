"""Dataset loaders. The generic matrix-per-omic + sample-sheet loader is the
default entry point; AnnData/MuData passthroughs are provided for convenience."""

from .anndata_io import load_h5ad, load_h5mu
from .generic import load_generic

__all__ = ["load_generic", "load_h5ad", "load_h5mu"]
