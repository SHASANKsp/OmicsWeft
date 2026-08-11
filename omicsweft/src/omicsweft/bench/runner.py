"""The benchmark runner — the reason to build package-first.

Given a :class:`Config` (dataset + preprocessing + integrators + tasks), it runs
every integrator x task combination through the *public* package API and returns
a tidy results table. Because it is config-driven and seeded, every number is
reproducible from the accompanying :class:`RunManifest`.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from ..core.config import Config, RunManifest
from ..core.omicsdata import OmicsData
from ..core.registry import DATASETS, INTEGRATORS, PREPROCESSORS, TASKS
from ..core.utils import get_logger, set_seed
from ..preprocess import run_pipeline

_LOG = get_logger("omicsweft.bench")


def _build_preprocessors(specs):
    return [PREPROCESSORS.create(s.name, **s.params) for s in specs]


def run(config: Config, data: OmicsData | None = None) -> pd.DataFrame:
    """Execute a benchmark run and return a results DataFrame.

    Parameters
    ----------
    config
        The run specification.
    data
        An already-loaded OmicsData. If ``None``, the dataset is loaded from
        ``config.dataset`` via the DATASETS registry.
    """
    set_seed(config.seed)

    if data is None:
        if config.dataset is None:
            raise ValueError("no dataset provided and config.dataset is empty")
        data = DATASETS.create(config.dataset.name, **config.dataset.params)

    # preprocessing is applied once, up front (shared across integrators)
    if config.preprocess:
        data = run_pipeline(data, _build_preprocessors(config.preprocess))

    rows = []
    for isp in config.integrators:
        integrator = INTEGRATORS.create(isp.name, **isp.params)
        t0 = time.perf_counter()
        try:
            emb = integrator.fit_transform(data)
        except Exception as exc:  # keep the sweep going; record the failure
            _LOG.warning("integrator %s failed: %s", isp.name, exc)
            rows.append(
                {"integrator": isp.name, "task": "-", "metric": "error",
                 "value": str(exc), "seconds": None}
            )
            continue
        fit_seconds = time.perf_counter() - t0
        _LOG.info("integrator %s -> embedding %s in %.2fs",
                  isp.name, emb.X.shape, fit_seconds)

        for tsp in config.tasks:
            task = TASKS.create(tsp.name, **tsp.params)
            try:
                metrics = task.evaluate(emb, data)
            except Exception as exc:
                _LOG.warning("task %s failed on %s: %s", tsp.name, isp.name, exc)
                rows.append(
                    {"integrator": isp.name, "task": tsp.name, "metric": "error",
                     "value": str(exc), "seconds": fit_seconds}
                )
                continue
            for metric, value in metrics.items():
                rows.append(
                    {"integrator": isp.name, "task": tsp.name, "metric": metric,
                     "value": value, "seconds": fit_seconds}
                )

    return pd.DataFrame(rows)


def run_and_save(config: Config, outdir: str | Path) -> pd.DataFrame:
    """Run a benchmark and persist results + manifest for reproducibility."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results = run(config)
    results.to_csv(outdir / "results.csv", index=False)
    RunManifest.capture(config).save(outdir / "manifest.json")
    config.to_yaml(outdir / "config.yaml")
    _LOG.info("wrote results + manifest to %s", outdir)
    return results


def _cli() -> None:  # pragma: no cover - thin entry point
    import argparse

    ap = argparse.ArgumentParser(prog="omicsweft", description="run an OmicsWeft benchmark")
    ap.add_argument("config", help="path to a config.yaml")
    ap.add_argument("-o", "--outdir", default="weft_run", help="output directory")
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)
    res = run_and_save(cfg, args.outdir)
    print(res.to_string(index=False))
