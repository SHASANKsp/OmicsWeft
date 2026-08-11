"""The generic loader — the package's default, disease-agnostic entry point.

Expected on-disk layout (a single directory)::

    my_study/
        manifest.yaml        # optional; inferred if absent
        sample_sheet.csv     # one row per sample, a sample-id column
        rna.csv              # samples x features (or transposed; see manifest)
        methylation.csv
        protein.csv

manifest.yaml (optional but recommended)::

    sample_sheet: sample_sheet.csv
    sample_id_column: sample_id
    omics:
      rna:          { path: rna.csv,     orientation: samples_x_features }
      methylation:  { path: methyl.csv,  orientation: samples_x_features }
      protein:      { path: protein.csv, orientation: features_x_samples }

If no manifest is present, every tabular file except the sample sheet is treated
as an omic block named after its filename, assumed samples-x-features.

The steps: read manifest (or infer) -> load sample sheet -> load + orient each
omic -> build one AnnData per omic (partial overlap allowed) -> assemble MuData
-> validate + coverage summary -> return OmicsData.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..core.omicsdata import OmicsData
from ..core.registry import DATASETS
from ..core.utils import get_logger

_LOG = get_logger("omicsweft.io")

_TABLE_SUFFIXES = {".csv", ".tsv", ".txt", ".parquet"}
_SHEET_STEMS = {"sample_sheet", "samples", "metadata", "clinical", "sample_meta"}


def _read_table(path: Path, index_col: int | str | None = 0) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
        if index_col is not None:
            first = df.columns[0] if isinstance(index_col, int) else index_col
            df = df.set_index(first)
        return df
    sep = "\t" if suffix in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep, index_col=index_col)


def _orient(df: pd.DataFrame, orientation: str, omic: str) -> pd.DataFrame:
    """Return a samples-x-features frame regardless of stored orientation."""
    if orientation in ("samples_x_features", "sxf"):
        out = df
    elif orientation in ("features_x_samples", "fxs"):
        out = df.T
    else:
        raise ValueError(
            f"omic {omic!r}: unknown orientation {orientation!r}; use "
            "'samples_x_features' or 'features_x_samples'"
        )
    out.index = out.index.astype(str)
    out.columns = out.columns.astype(str)
    return out


def _infer_manifest(directory: Path) -> dict[str, Any]:
    """Build a manifest by inspecting the directory contents."""
    tables = [
        p for p in sorted(directory.iterdir())
        if p.suffix.lower() in _TABLE_SUFFIXES
    ]
    if not tables:
        raise FileNotFoundError(f"no tabular files found in {directory}")

    sheet_path = next(
        (p for p in tables if p.stem.lower() in _SHEET_STEMS), None
    )
    omic_paths = [p for p in tables if p is not sheet_path]
    if not omic_paths:
        raise ValueError(
            f"{directory} has no omic matrices (only a sample sheet was found)"
        )

    manifest: dict[str, Any] = {
        "sample_id_column": 0,
        "omics": {
            p.stem: {"path": p.name, "orientation": "samples_x_features"}
            for p in omic_paths
        },
    }
    if sheet_path is not None:
        manifest["sample_sheet"] = sheet_path.name
    _LOG.info(
        "inferred manifest: omics=%s, sample_sheet=%s",
        list(manifest["omics"]),
        manifest.get("sample_sheet", "<none>"),
    )
    return manifest


def _validate(data: OmicsData) -> None:
    cov = data.coverage()
    for m, n in cov.per_modality.items():
        if n == 0:
            _LOG.warning("modality %r has zero samples", m)
    if cov.n_complete == 0 and cov.n_modalities > 1:
        _LOG.warning(
            "no sample is present in every modality — integrators that need a "
            "shared sample set will operate on pairwise/subset overlaps only"
        )
    _LOG.info("%s", cov)


@DATASETS.register("generic")
def load_generic(
    directory: str | Path,
    manifest: str | Path | dict | None = None,
    name: str | None = None,
) -> OmicsData:
    """Load a dataset in the generic matrix-per-omic + sample-sheet format.

    Parameters
    ----------
    directory
        Folder containing the omic matrices (+ optional manifest / sample sheet).
    manifest
        Explicit manifest as a path or dict. If ``None``, ``manifest.yaml`` in
        ``directory`` is used when present, otherwise one is inferred.
    name
        Dataset name; defaults to the directory name.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a directory")

    # 1. resolve manifest
    if manifest is None:
        mpath = directory / "manifest.yaml"
        if mpath.exists():
            with open(mpath) as fh:
                manifest = yaml.safe_load(fh)
        else:
            manifest = _infer_manifest(directory)
    elif not isinstance(manifest, dict):
        with open(manifest) as fh:
            manifest = yaml.safe_load(fh)

    id_col = manifest.get("sample_id_column", 0)

    # 2. sample sheet (optional)
    sheet: pd.DataFrame | None = None
    if manifest.get("sample_sheet"):
        sheet = _read_table(directory / manifest["sample_sheet"], index_col=id_col)
        sheet.index = sheet.index.astype(str)

    # 3-4. load + orient each omic into samples x features
    matrices: dict[str, pd.DataFrame] = {}
    for omic, spec in manifest["omics"].items():
        raw = _read_table(directory / spec["path"], index_col=0)
        matrices[omic] = _orient(
            raw, spec.get("orientation", "samples_x_features"), omic
        )

    # 5-6. assemble container (MuData handles the sample union / partial overlap)
    data = OmicsData.from_matrices(
        matrices, sample_sheet=sheet, name=name or directory.name
    )

    # 7. validate + coverage
    _validate(data)
    return data
