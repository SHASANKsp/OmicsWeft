"""Case study — Step 2: the end-to-end analysis.

Standalone script (not part of the omicsweft package). Reads a dataset in the
generic format (produced by prepare_data.py or your own conformed data) and runs
the whole pipeline: load -> inspect -> preprocess -> integrate (catalogue) ->
evaluate -> interpret -> robustness, saving a reproducible results table +
manifest.

Run:  python case_study.py [data_dir] [out_dir]
"""

from __future__ import annotations

import sys
import warnings

import pandas as pd

from omicsweft.bench import robustness_curve, run_and_save
from omicsweft.core.config import Config
from omicsweft.core.registry import INTERPRETERS
from omicsweft.integrate import MOFA
from omicsweft.io import load_generic
from omicsweft.preprocess import compute_qc

warnings.filterwarnings("ignore")


def main(data_dir: str = "data/brca_practice", out_dir: str = "results/brca") -> None:
    pd.set_option("display.width", 140)

    # 1. LOAD + INSPECT ----------------------------------------------------
    data = load_generic(data_dir)
    print("=== dataset ===")
    print(data)
    print("\n=== coverage ===")
    print(data.coverage())
    print("\n=== QC (per omic) ===")
    print(compute_qc(data).to_string(index=False))

    # 2. BENCHMARK THE CATALOGUE ------------------------------------------
    # The dataset is declared in the config, so the run is fully reproducible from
    # the saved manifest (and works with the frozen package's run_and_save).
    # Note: 'mogonet' is supervised and would use the label it is scored on, so it
    # is left out of an unsupervised subtype-discovery benchmark (leakage).
    cfg = Config(
        name="brca_case_study",
        seed=0,
        dataset={"name": "generic", "params": {"directory": data_dir}},
        preprocess=[
            {"name": "standardize"},
            {"name": "select_variance", "params": {"k": 400}},
        ],
        integrators=[
            {"name": "concat_pca", "params": {"n_components": 10}},
            {"name": "snf", "params": {"n_components": 8, "n_iters": 20}},
            {"name": "jnmf", "params": {"n_components": 10}},
            {"name": "mofa", "params": {"n_factors": 10}},
            {"name": "mvae", "params": {"n_components": 16, "epochs": 100}},
        ],
        tasks=[
            {"name": "clustering", "params": {"label_key": "subtype"}},
            {"name": "classification", "params": {"label_key": "subtype"}},
            {"name": "survival", "params": {"time_key": "os_time", "event_key": "os_event"}},
        ],
    )
    results = run_and_save(cfg, out_dir)

    print("\n=== benchmark (wide) ===")
    wide = results.pivot_table(index="integrator", columns=["task", "metric"], values="value")
    print(wide.to_string())

    # 3. INTERPRET the factor model ---------------------------------------
    emb = MOFA(n_factors=10).fit_transform(data)
    loads = INTERPRETERS.create("factor_loadings", top_n=8).interpret(emb, data)
    print("\n=== MOFA: top RNA features on factor_0 ===")
    print(list(loads["top_features"]["rna"]["factor_0"].index))

    # 4. ROBUSTNESS of the chosen method ----------------------------------
    print("\n=== robustness: concat_pca clustering vs feature dropout ===")
    rc = robustness_curve(
        data, "concat_pca", "clustering", "ari",
        levels=(0.0, 0.25, 0.5, 0.75), kind="dropout", n_repeats=3,
        integrator_params={"n_components": 10},
        task_params={"label_key": "subtype"},
    )
    print(rc.to_string(index=False))
    print(f"\nresults + manifest written to: {out_dir}/")


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "data/brca_practice"
    o = sys.argv[2] if len(sys.argv) > 2 else "results/brca"
    main(d, o)
