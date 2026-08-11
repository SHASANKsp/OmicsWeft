# OmicsWeft

**An end-to-end, disease-agnostic multi-omics integration toolkit.**

`OmicsWeft` weaves multiple omics layers over a shared sample axis into one joint
representation, then evaluates it — through a single, uniform API. The core knows
nothing about any disease: it deals in omics blocks, samples, and abstract tasks.
Cancer-specific analysis such as survival / Kaplan–Meier is an *optional task
plugin*, loaded only when a dataset actually carries time-to-event labels.

> Status: **v0.2 — full Python-native catalogue + interpretation + robustness.**
> Six integrators across four families, an interpretation layer, and a
> noise/dropout robustness harness, all behind the v0.1 contracts. R-bridge
> methods and the knowledge-graph layer remain deferred to later phases.

## Install

```bash
# recommended: conda / Miniforge, Python 3.12
conda env create -f environment.yml
conda activate omicsweft
pip install -e .

# optional extras
pip install -e ".[survival]"   # Kaplan–Meier / log-rank via lifelines
pip install -e ".[dev]"        # pytest, ruff
```

## Quickstart

```python
import omicsweft
from omicsweft.datasets import make_synthetic
from omicsweft.integrate import ConcatPCA
from omicsweft.tasks import ClusteringTask

data = make_synthetic(n_samples=150, n_clusters=4)   # or load your own
emb  = ConcatPCA(n_components=10).fit_transform(data)
print(ClusteringTask(label_key="subtype").evaluate(emb, data))
```

Config-driven benchmark across methods and tasks:

```python
from omicsweft.bench import run
from omicsweft.core.config import Config

cfg = Config(
    dataset={"name": "generic", "params": {"directory": "my_study/"}},
    preprocess=[{"name": "standardize"}, {"name": "select_variance", "params": {"k": 2000}}],
    integrators=[{"name": "concat_pca"}, {"name": "snf"}],
    tasks=[{"name": "clustering", "params": {"label_key": "subtype"}}],
)
print(run(cfg))
```

## The generic data format (default entry point)

Point the loader at a directory:

```
my_study/
├── manifest.yaml        # optional; inferred from the folder if absent
├── sample_sheet.csv     # one row per sample; a sample-id column + labels/covariates
├── rna.csv              # samples × features (orientation configurable)
├── methylation.csv
└── protein.csv
```

Every dataset — including public benchmarks — is conformed to this one contract,
so there is a single code path for all data. Modalities need not cover the same
samples; partial overlap is tracked, not treated as an error.

## Architecture

A domain-free **core** (the `OmicsData` container, plugin registries, config,
seeding, run manifests) with everything else as **plugins** discovered by name:

| Stage | Plugin type | Ships in v0.1 |
|-------|-------------|---------------|
| Ingest | `Dataset` | generic loader, h5ad/h5mu passthrough |
| Preprocess | `Preprocessor` | normalize, impute, batch, feature-select, QC |
| Integrate | `Integrator` | concat+PCA, SNF, joint-NMF, MOFA, multi-view VAE, MOGONET |
| Task & evaluate | `Task` | clustering, classification, regression, *survival (optional)* |
| Interpret | `Interpreter` | factor loadings, embedding correlation, permutation importance, *enrichment (optional)* |
| Benchmark | runner | any integrator × dataset × task → tables + manifest; **robustness curves** |

Because everything runs through the registries and one config, the benchmark
runner exercises the real package, and every result regenerates from its manifest.

## Roadmap

- **v0.1:** skeleton, generic loader, preprocessing, two baseline integrators,
  task/evaluator system, benchmark harness, tests.
- **v0.2 (this):** MOFA (`mofapy2`), joint-NMF, multi-view VAE, MOGONET
  (all Python-native), interpretation layer, robustness benchmarking.
- **v0.3+ (deferred):** R bridge (DIABLO, NEMO, MCIA, iCluster, RGCCA…), more
  GNNs (MoGCN, DeepMoIC, SUPREME, MOGAT), single-cell module (`scvi-tools`).
- **Later phase:** knowledge-graph layer.

## Deep-learning integrators

`mvae` and `mogonet` use PyTorch and auto-select the GPU when available
(`device="cuda"`); they fall back to CPU otherwise. `mogonet` is **supervised**
(needs a `label_key`) — evaluating it by predicting the *same* label is circular,
so use it to embed for a different target or hold labels out.

## License

MIT — see `LICENSE`.
