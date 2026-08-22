# OmicsWeft — Case Study (standalone)

These are **standalone** case-study materials. They are *not* part of the
`omicsweft` package — they use the installed library but live in their own folder,
so the package stays clean.

## Contents
- `CASE_STUDY.md`   — the detailed, end-to-end walkthrough (read this first)
- `prepare_data.py` — Step 1: put data into the generic format (practice cohort +
                      a real-data conformer for TCGA/cBioPortal/MLOmics exports)
- `case_study.py`   — Step 2: the full analysis (load → preprocess → integrate →
                      evaluate → interpret → robustness), reproducible via config

## Prerequisites
Install the frozen package first (from the omicsweft folder):

    conda activate omicsweft
    pip install -e ".[dev,survival]"
    pip install torch          # for the mvae / mogonet integrators

## Run
From this folder:

    python prepare_data.py                          # writes data/brca_practice/
    python case_study.py data/brca_practice results/brca

Outputs land in `results/brca/` (results.csv, config.yaml, manifest.json).
