"""End-to-end quickstart.

Generates a synthetic multi-omics dataset, writes it to disk in the generic
format, then loads it back and benchmarks two integrators across four tasks.

Run:  python examples/quickstart.py
"""

from __future__ import annotations

import tempfile

import pandas as pd

from omicsweft.bench import run
from omicsweft.core.config import Config
from omicsweft.datasets import make_synthetic, write_generic


def main() -> None:
    pd.set_option("display.width", 120)

    # 1. make a dataset and write it in the generic format (as your real data would be)
    tmp = tempfile.mkdtemp()
    data = make_synthetic(n_samples=150, n_clusters=4, partial_overlap=0.1, seed=7)
    study_dir = write_generic(data, tmp)
    print("wrote generic dataset to", study_dir, "\n")

    # 2. describe the whole run as a config (reproducible + serialisable)
    cfg = Config(
        name="quickstart",
        seed=0,
        dataset={"name": "generic", "params": {"directory": str(study_dir)}},
        preprocess=[
            {"name": "standardize"},
            {"name": "select_variance", "params": {"k": 150}},
        ],
        integrators=[
            {"name": "concat_pca", "params": {"n_components": 10}},
            {"name": "snf", "params": {"n_components": 8, "n_iters": 15}},
        ],
        tasks=[
            {"name": "clustering", "params": {"label_key": "subtype"}},
            {"name": "classification", "params": {"label_key": "subtype"}},
            {"name": "regression", "params": {"target_key": "marker"}},
            # survival is optional; comment out if lifelines isn't installed
            {"name": "survival", "params": {"time_key": "os_time", "event_key": "os_event"}},
        ],
    )

    # 3. run and print the benchmark table
    results = run(cfg)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
