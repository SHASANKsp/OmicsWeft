"""Preprocessing plugins and a small chaining helper.

All preprocessors are domain-free and operate per modality. Import order below
matters only in that it triggers registration.
"""

from __future__ import annotations

from ..core.base import Preprocessor
from ..core.omicsdata import OmicsData
from ..core.registry import PREPROCESSORS
from .batch import BatchLinear, ComBat
from .impute import DropMissingFeatures, ImputeKNN, ImputeSimple
from .normalize import Arcsinh, Log1p, Standardize, TotalCount
from .qc import QC, compute_qc
from .select import SelectMAD, SelectVariance


def run_pipeline(data: OmicsData, steps: list[Preprocessor]) -> OmicsData:
    """Apply an ordered list of preprocessors, returning the transformed data."""
    for step in steps:
        data = step.apply(data)
    return data


__all__ = [
    "PREPROCESSORS",
    "QC",
    "Arcsinh",
    "BatchLinear",
    "ComBat",
    "DropMissingFeatures",
    "ImputeKNN",
    "ImputeSimple",
    "Log1p",
    "SelectMAD",
    "SelectVariance",
    "Standardize",
    "TotalCount",
    "compute_qc",
    "run_pipeline",
]
