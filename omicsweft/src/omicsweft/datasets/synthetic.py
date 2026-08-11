"""Synthetic multi-omics generator.

Produces a small dataset with known latent cluster structure shared across omics,
plus a sample sheet carrying: a categorical subtype label, a continuous marker,
a batch label, and (optionally) time-to-event columns. Used as the CI fixture and
the quickstart example, and to exercise the on-disk generic loader.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..core.omicsdata import OmicsData


def make_synthetic(
    n_samples: int = 120,
    n_clusters: int = 3,
    omics: dict[str, int] | None = None,
    partial_overlap: float = 0.0,
    with_survival: bool = True,
    seed: int = 0,
) -> OmicsData:
    """Create a synthetic :class:`OmicsData`.

    Parameters
    ----------
    omics
        ``{name: n_features}``. Defaults to three omics of differing widths.
    partial_overlap
        Fraction of samples to randomly drop from each omic (to simulate the
        partial-overlap case). 0 keeps all samples in all omics.
    """
    rng = np.random.default_rng(seed)
    omics = omics or {"rna": 200, "methylation": 150, "protein": 80}

    sample_ids = np.array([f"S{i:04d}" for i in range(n_samples)])
    cluster = rng.integers(0, n_clusters, size=n_samples)

    matrices: dict[str, pd.DataFrame] = {}
    for omic, n_feat in omics.items():
        # cluster centroids in feature space + gaussian noise
        centroids = rng.normal(0, 3, size=(n_clusters, n_feat))
        X = centroids[cluster] + rng.normal(0, 1, size=(n_samples, n_feat))
        # a few informative features per cluster get a stronger signal
        X += rng.normal(0, 0.5, size=(n_samples, n_feat))
        feat_ids = [f"{omic.upper()}_{j}" for j in range(n_feat)]
        df = pd.DataFrame(X, index=sample_ids, columns=feat_ids)

        if partial_overlap > 0:
            n_drop = round(partial_overlap * n_samples)
            drop = rng.choice(n_samples, size=n_drop, replace=False)
            df = df.drop(index=sample_ids[drop])
        matrices[omic] = df

    # sample sheet
    subtype = np.array([f"subtype_{c}" for c in cluster])
    batch = rng.choice(["batch_A", "batch_B"], size=n_samples)
    marker = cluster * 2.0 + rng.normal(0, 0.5, size=n_samples)  # continuous target
    sheet = pd.DataFrame(
        {"subtype": subtype, "batch": batch, "marker": marker},
        index=sample_ids,
    )
    if with_survival:
        base = rng.exponential(scale=500.0, size=n_samples)
        time = base * (1.0 + cluster)  # cluster affects survival time
        event = rng.binomial(1, 0.7, size=n_samples)
        sheet["os_time"] = time
        sheet["os_event"] = event

    return OmicsData.from_matrices(matrices, sample_sheet=sheet, name="synthetic")


def write_generic(data: OmicsData, directory: str | Path) -> Path:
    """Write an OmicsData to disk in the generic loader format."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    omics_spec = {}
    for m in data.modalities:
        ad = data.get_omic(m)
        df = pd.DataFrame(
            np.asarray(ad.X), index=ad.obs_names, columns=ad.var_names
        )
        df.index.name = "sample_id"
        df.to_csv(directory / f"{m}.csv")
        omics_spec[m] = {"path": f"{m}.csv", "orientation": "samples_x_features"}

    sheet = data.obs.copy()
    sheet.index.name = "sample_id"
    sheet.to_csv(directory / "sample_sheet.csv")

    import yaml

    manifest = {
        "sample_sheet": "sample_sheet.csv",
        "sample_id_column": "sample_id",
        "omics": omics_spec,
    }
    with open(directory / "manifest.yaml", "w") as fh:
        yaml.safe_dump(manifest, fh, sort_keys=False)
    return directory
