"""v0.2 showcase — full Python-native catalogue, interpretation, robustness.

Runs all unsupervised integrators through the benchmark, interprets the MOFA
factors, and draws a robustness curve.

Run:  python examples/catalogue_v2.py
"""

from __future__ import annotations

import warnings

import pandas as pd

from omicsweft.bench import robustness_curve, run
from omicsweft.core.config import Config
from omicsweft.core.registry import INTERPRETERS
from omicsweft.datasets import make_synthetic
from omicsweft.integrate import MOFA

warnings.filterwarnings("ignore")


def main() -> None:
    pd.set_option("display.width", 120)
    data = make_synthetic(n_samples=150, n_clusters=4, partial_overlap=0.1, seed=7)

    # 1. benchmark the unsupervised catalogue on clustering + classification
    cfg = Config(
        name="catalogue_v2",
        seed=0,
        preprocess=[{"name": "standardize"}, {"name": "select_variance", "params": {"k": 150}}],
        integrators=[
            {"name": "concat_pca", "params": {"n_components": 10}},
            {"name": "snf", "params": {"n_components": 8, "n_iters": 15}},
            {"name": "jnmf", "params": {"n_components": 10}},
            {"name": "mofa", "params": {"n_factors": 10}},
            {"name": "mvae", "params": {"n_components": 16, "epochs": 80}},
        ],
        tasks=[
            {"name": "clustering", "params": {"label_key": "subtype"}},
            {"name": "classification", "params": {"label_key": "subtype"}},
        ],
    )
    results = run(cfg, data=data)
    print("=== catalogue benchmark ===")
    pivot = results[results["metric"].isin(["ari", "macro_f1"])].pivot_table(
        index="integrator", columns="metric", values="value"
    )
    print(pivot.to_string(), "\n")

    # 2. interpret MOFA factors -> top features
    emb = MOFA(n_factors=10).fit_transform(data)
    top = INTERPRETERS.create("factor_loadings", top_n=5).interpret(emb, data)
    print("=== MOFA factor_0 top RNA features ===")
    print(list(top["top_features"]["rna"]["factor_0"].index), "\n")

    # 3. robustness of concat_pca clustering to feature dropout
    print("=== robustness (dropout) ===")
    rc = robustness_curve(
        data, "concat_pca", "clustering", "ari",
        levels=(0.0, 0.25, 0.5, 0.75), kind="dropout", n_repeats=3,
        integrator_params={"n_components": 10},
        task_params={"label_key": "subtype"},
    )
    print(rc.to_string(index=False))


if __name__ == "__main__":
    main()
