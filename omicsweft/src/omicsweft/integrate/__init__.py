"""Integration methods that produce a joint sample :class:`Embedding`.

v0.1 baselines (dependency-free): concat+PCA, SNF.
v0.2 additions:
  - MOFA            (mofapy2)   — interpretable factors, native missing data
  - JointNMF        (from scratch) — shared non-negative sample coefficients
  - MultiViewVAE    (torch)     — Product-of-Experts VAE, missing-modality aware
  - MOGONET         (torch)     — supervised per-omic GCN + VCDN fusion

torch- and mofapy2-backed methods import their backend lazily, so the core
package works without them installed.
"""

from ..core.registry import INTEGRATORS
from .concat_pca import ConcatPCA
from .jnmf import JointNMF
from .mofa import MOFA
from .mogonet import MOGONET
from .snf import SNF
from .vae import MultiViewVAE

__all__ = [
    "INTEGRATORS",
    "MOFA",
    "MOGONET",
    "SNF",
    "ConcatPCA",
    "JointNMF",
    "MultiViewVAE",
]
