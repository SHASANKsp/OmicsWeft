# OmicsWeft — End-to-End Case Study

A complete, verified walkthrough: from a fresh machine to a reproducible
multi-omics benchmark with interpretation and robustness. The example is framed
as a **breast-cancer subtype-discovery** study (3 omics, 4 subtypes, survival),
but nothing in the package is cancer-specific — swap in any disease by changing
the dataset and the labels your tasks reference.

Every command below was run to produce the outputs shown. Two helper scripts accompany this tutorial:

- `prepare_data.py` — puts data into the generic format
- `case_study.py`  — runs the whole analysis on that data

---

## Step 0 — Prerequisites & install (Windows + conda)

You need Miniforge/Miniconda. Then, from the unzipped package folder:

```bash
conda env create -f environment.yml     # creates the "omicsweft" env (Python 3.12)
conda activate omicsweft
pip install -e ".[dev,survival]"         # editable install + tests + survival extra
```

`survival` pulls in `lifelines` (needed for the Kaplan-Meier / log-rank task). For
the deep methods and enrichment, add what you need:

```bash
pip install torch                        # mvae + mogonet (uses your GPU automatically)
pip install -e ".[enrich]"               # gseapy, for pathway enrichment
```

Confirm the install:

```bash
python -c "import omicsweft; print(omicsweft.__version__); print(omicsweft.INTEGRATORS.list())"
# 0.2.0
# ['concat_pca', 'jnmf', 'mofa', 'mogonet', 'mvae', 'snf']
pytest -q                                # 34 passed
```

---

## Step 1 — Get your data into the generic format

OmicsWeft reads **one directory** containing a matrix per omic plus a sample
sheet. This is the only "shape" you ever prepare; every dataset is conformed to it
once, and everything downstream is identical for practice data and real data.

### 1a. To learn the flow: generate a practice cohort

```bash
mkdir case_study && cd case_study
python prepare_data.py
```

```
wrote practice cohort -> data/brca_practice
  files: ['manifest.yaml', 'methylation.csv', 'protein.csv', 'rna.csv', 'sample_sheet.csv']
```

The directory now looks like:

```
data/brca_practice/
├── manifest.yaml        # names each omic file + its orientation + the sample sheet
├── sample_sheet.csv     # sample_id, subtype, batch, marker, os_time, os_event
├── rna.csv              # 200 samples × 800 features (samples in rows)
├── methylation.csv      # 200 × 600
└── protein.csv          # 200 × 200
```

`sample_sheet.csv` carries the labels/covariates your tasks will reference by name
(`subtype` for clustering/classification, `os_time`/`os_event` for survival). The
manifest is optional — if absent, the loader infers it from the folder.

### 1b. For real data: conform raw files once

Public exports (TCGA / cBioPortal / LinkedOmics / MLOmics) usually give you each
omic as a **features × samples** matrix plus a clinical table. `prepare_data.py`
includes `conform_raw_files(...)` showing the pattern — read each matrix, transpose
to samples × features, align on sample id, attach the clinical sheet, and write the
generic directory:

```python
from omicsweft import OmicsData
from omicsweft.datasets import write_generic
import pandas as pd

rna  = pd.read_csv("RNA.txt",    sep="\t", index_col=0).T      # -> samples × genes
meth = pd.read_csv("METH.txt",   sep="\t", index_col=0).T
prot = pd.read_csv("RPPA.txt",   sep="\t", index_col=0).T
clin = pd.read_csv("clinical.txt", sep="\t", index_col=0)
clin = clin.rename(columns={"PAM50": "subtype",
                            "OS_MONTHS": "os_time",
                            "OS_STATUS": "os_event"})     # -> names your tasks use

data = OmicsData.from_matrices(
    {"rna": rna, "methylation": meth, "protein": prot},
    sample_sheet=clin, name="brca_tcga",
)
write_generic(data, "data/brca_tcga")        # same format as the practice cohort
```

From here on, the two are indistinguishable — point the analysis at either folder.

---

## Step 2 — Load and inspect

```python
from omicsweft.io import load_generic
from omicsweft.preprocess import compute_qc

data = load_generic("data/brca_practice")
print(data)
print(data.coverage())
print(compute_qc(data).to_string(index=False))
```

```
OmicsData(name='brca_practice', n_samples=200, rna:(184, 800), methylation:(184, 600), protein:(184, 200))

OmicsData coverage: 200 samples across 3 modalities
  complete (all modalities): 158/200
  rna: 184 samples
  methylation: 184 samples
  protein: 184 samples

   modality  n_samples  n_features  pct_missing      mean      std        min       max
        rna        184         800          0.0 -0.063633 3.223676 -12.670801 12.487149
methylation        184         600          0.0 -0.027836 3.217205 -12.609653 12.051699
    protein        184         200          0.0  0.069362 3.242058 -12.068849 11.477170
```

Read this before modelling: 200 samples total, **158 complete across all three
omics** (the rest miss an omic — handled, not an error). Integrators that need a
shared sample set will use those 158; the coverage line tells you so up front.

---

## Step 3 — Preprocess (declared once, shared by every method)

You rarely call preprocessors by hand — you declare them in the run config so the
same pipeline feeds every integrator and is captured in the manifest. For a manual
run it looks like:

```python
from omicsweft.preprocess import run_pipeline, Standardize, SelectVariance
data = run_pipeline(data, [Standardize(), SelectVariance(k=400)])
```

`standardize` puts every feature on a comparable scale (so a wide omic block can't
dominate); `select_variance` keeps the 400 most variable features per omic. Both
are unsupervised and safe to run before splitting.

---

## Step 4 — Integrate + evaluate as a benchmark

The point of the package is to run **many methods through one config** and compare.
`case_study.py` does exactly this. Run it:

```bash
python case_study.py data/brca_practice results/brca
```

The config it uses (worth reading — this *is* the experiment):

```python
Config(
    name="brca_case_study", seed=0,
    dataset={"name": "generic", "params": {"directory": "data/brca_practice"}},
    preprocess=[{"name": "standardize"},
                {"name": "select_variance", "params": {"k": 400}}],
    integrators=[{"name": "concat_pca", "params": {"n_components": 10}},
                 {"name": "snf",        "params": {"n_components": 8, "n_iters": 20}},
                 {"name": "jnmf",       "params": {"n_components": 10}},
                 {"name": "mofa",       "params": {"n_factors": 10}},
                 {"name": "mvae",       "params": {"n_components": 16, "epochs": 100}}],
    tasks=[{"name": "clustering",     "params": {"label_key": "subtype"}},
           {"name": "classification", "params": {"label_key": "subtype"}},
           {"name": "survival",       "params": {"time_key": "os_time",
                                                 "event_key": "os_event"}}],
)
```

Result (the real benchmark table):

```
task       classification                      clustering                              survival
metric              auprc auroc macro_f1        ari davies_bouldin  nmi silhouette   c_index logrank_p logrank_stat
integrator
concat_pca            1.0   1.0      1.0        1.0       0.221       1.0     0.848     0.525   0.16769        1.903
jnmf                  1.0   1.0      1.0        1.0       0.368       1.0     0.745     0.525   0.16769        1.903
mofa                  1.0   1.0      1.0        1.0       0.804       1.0     0.508     0.569   0.00006       16.002
mvae                  1.0   1.0      1.0        1.0       0.077       1.0     0.945     0.529   0.16769        1.903
snf                   1.0   1.0      1.0        1.0       1.535       1.0     0.390     0.475   0.16769        1.903
```

**How to read this — the key lesson.** Every method recovers the four subtypes
perfectly (ARI/NMI = 1.0, macro-F1 = 1.0) because the subtype signal is strong. But
they **diverge on survival**: MOFA's factors separate the survival groups sharply
(log-rank p ≈ 0.00006) while the others don't (p ≈ 0.17). Same data, same subtypes
— different embedding geometry, different downstream value. This is precisely why
the package offers many methods and a fair benchmark rather than one "best" method:
**the winner depends on the task**. On real data you'd also weigh internal metrics
(silhouette, Davies–Bouldin) and runtime.

Notes:
- **`mogonet` is deliberately absent here.** It is supervised; scoring it on the
  same `subtype` it trained on would be circular. Use it only to embed for a
  *different* target, or with held-out labels.
- Every cell is reproducible: `results/brca/` now holds `results.csv`,
  `config.yaml`, and `manifest.json` (with a config hash + environment).

### Same run from the command line

You don't need Python for the benchmark — save the config and use the CLI:

```bash
python -c "from omicsweft.core.config import Config; import yaml; \
Config(name='brca', dataset={'name':'generic','params':{'directory':'data/brca_practice'}}, \
preprocess=[{'name':'standardize'},{'name':'select_variance','params':{'k':400}}], \
integrators=[{'name':'concat_pca'},{'name':'mofa','params':{'n_factors':10}}], \
tasks=[{'name':'clustering','params':{'label_key':'subtype'}}]).to_yaml('config.yaml')"

omicsweft config.yaml -o results/cli_run
```

---

## Step 5 — Interpret the model

For a factor method (MOFA, joint-NMF) you can read which features drive each factor:

```python
from omicsweft.integrate import MOFA
from omicsweft.core.registry import INTERPRETERS

emb = MOFA(n_factors=10).fit_transform(data)
loads = INTERPRETERS.create("factor_loadings", top_n=8).interpret(emb, data)
print(loads["top_features"]["rna"]["factor_0"].index.tolist())
# ['RNA_799', 'RNA_728', 'RNA_51', 'RNA_316', 'RNA_5', 'RNA_623', 'RNA_548', 'RNA_604']
```

For **any** integrator (including the deep ones with no explicit loadings), use the
model-agnostic interpreter:

```python
INTERPRETERS.create("embedding_correlation", top_n=20).interpret(emb, data)
INTERPRETERS.create("permutation_importance", label_key="subtype").interpret(emb, data)
```

With the `enrich` extra installed, turn the top features into pathways:

```python
INTERPRETERS.create("enrichment", gene_sets="GO_Biological_Process_2021").interpret(emb, data)
```

On real data, swap the synthetic `RNA_*` ids for real gene symbols during Step 1
(the feature-ID harmonization you set up in preprocessing) so enrichment is meaningful.

---

## Step 6 — Robustness (does the result survive perturbation?)

A method that only wins on pristine data isn't trustworthy. Perturb and re-measure:

```python
from omicsweft.bench import robustness_curve
robustness_curve(
    data, "concat_pca", "clustering", "ari",
    levels=(0.0, 0.25, 0.5, 0.75), kind="dropout", n_repeats=3,
    integrator_params={"n_components": 10},
    task_params={"label_key": "subtype"},
)
```

```
 level    kind  ari_mean  ari_std  n
  0.00 dropout       1.0      0.0  3
  0.25 dropout       1.0      0.0  3
  0.50 dropout       1.0      0.0  3
  0.75 dropout       1.0      0.0  3
```

(Here the subtype signal is redundant enough to survive heavy dropout; on real data
you'll see curves bend, and you compare methods by how gracefully they degrade. Use
`kind="noise"` for additive Gaussian noise instead.)

---

## Step 7 — Reproduce or share

Everything needed to reproduce a run is in the output folder:

```
results/brca/
├── results.csv     # tidy: integrator, task, metric, value, seconds
├── config.yaml     # the exact experiment
└── manifest.json   # config hash + seed + Python/OS + package version
```

Re-run identically with `omicsweft results/brca/config.yaml -o results/brca_repeat`,
or load `results.csv` into pandas for your own plots.

---

## Adapting this to your real study — a checklist

1. **Conform your data once** (Step 1b): per-omic matrices → samples × features,
   one sample sheet with your label columns. Map feature ids to real gene/protein
   symbols here.
2. **Rename the label keys** in the tasks to your columns (e.g. `subtype` →
   `histology`, or drop survival entirely for a non-cancer study — the core never
   needs it).
3. **Pick tasks that fit your question**: subtype discovery → clustering; known
   labels → classification; a continuous phenotype (e.g. a metabolic marker) →
   regression; time-to-event → survival.
4. **Start with the fast methods** (`concat_pca`, `snf`, `jnmf`, `mofa`) to get a
   baseline, then add `mvae`; use `mogonet` only for supervised embedding of a
   held-out target.
5. **Read the benchmark by task, not overall** — expect different winners for
   clustering vs survival, as above.
6. **Check robustness** before trusting a winner.
7. **Keep the manifest** with any figure you produce.

## Common gotchas

- **Partial overlap:** integrators use the shared sample set; the coverage summary
  tells you how many that is. If it's tiny, reconsider which omics to combine.
- **`mogonet` leakage:** never score a supervised integrator on the label it trained
  on.
- **MOFA prints a banner:** that's mofapy2's normal startup output, not an error.
- **R methods (DIABLO, NEMO, MCIA, iCluster, RGCCA):** deferred to a later version;
  not available yet.
- **Feature ids:** enrichment is only meaningful once ids are real gene symbols.
