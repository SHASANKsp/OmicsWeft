"""Case study — Step 1: get your data into the generic format.

Standalone script (not part of the omicsweft package). It uses the installed
`omicsweft` library but lives alongside your analysis, not inside it.

Two functions:

  make_practice_cohort(out)   writes a realistic practice dataset (breast-cancer-
                              style: 3 omics, subtypes, survival, batch) to disk in
                              the generic format. Use this to learn the flow.

  conform_raw_files(...)      shapes *real* per-omic matrices + a clinical table
                              (the shape TCGA / cBioPortal / LinkedOmics / MLOmics
                              exports come in) into the same generic format. Adapt
                              the paths/columns to your files.

Run:  python prepare_data.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from omicsweft.datasets import make_synthetic, write_generic


def make_practice_cohort(out: str = "data/brca_practice") -> Path:
    """Write a practice cohort to ``out`` in the generic format."""
    data = make_synthetic(
        n_samples=200,
        n_clusters=4,           # 4 "subtypes"
        omics={"rna": 800, "methylation": 600, "protein": 200},
        partial_overlap=0.08,   # a few samples miss an omic, like real cohorts
        with_survival=True,
        seed=42,
    )
    path = write_generic(data, out)
    print(f"wrote practice cohort -> {path}")
    print("  files:", sorted(p.name for p in path.iterdir()))
    return path


def conform_raw_files(
    rna_path: str,
    methyl_path: str,
    protein_path: str,
    clinical_path: str,
    out: str = "data/my_study",
    matrices_are_features_by_samples: bool = True,
) -> Path:
    """Conform real raw files into the generic format.

    Typical public exports store each omic as a matrix with **features in rows and
    samples in columns** (genes x samples), tab-separated, plus a clinical table
    keyed by sample id. This reads them, transposes to samples x features, aligns
    on sample id, and writes the generic directory.

    Adapt the ``sep``, the id columns, and the clinical column names to your files.
    """
    from omicsweft import OmicsData

    def _load_matrix(path: str) -> pd.DataFrame:
        df = pd.read_csv(path, sep="\t", index_col=0)
        if matrices_are_features_by_samples:
            df = df.T                      # -> samples x features
        df.index = df.index.astype(str)
        return df

    matrices = {
        "rna": _load_matrix(rna_path),
        "methylation": _load_matrix(methyl_path),
        "protein": _load_matrix(protein_path),
    }
    # clinical / sample sheet: one row per sample id; keep the columns you need
    clinical = pd.read_csv(clinical_path, sep="\t", index_col=0)
    clinical.index = clinical.index.astype(str)
    # rename to the label/covariate names your tasks will reference, e.g.:
    # clinical = clinical.rename(columns={"PAM50": "subtype",
    #                                     "OS_MONTHS": "os_time",
    #                                     "OS_STATUS": "os_event"})

    data = OmicsData.from_matrices(matrices, sample_sheet=clinical, name="my_study")
    path = write_generic(data, out)
    print(f"conformed real data -> {path}")
    return path


if __name__ == "__main__":
    make_practice_cohort()
