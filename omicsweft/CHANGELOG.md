# Changelog

## 0.2.0

### Added
- **Integrators (Python-native):**
  - `mofa` — MOFA/MOFA+ via mofapy2; interpretable factors + loadings, native
    missing-data handling.
  - `jnmf` — joint/integrative NMF from scratch; shared non-negative sample
    coefficients + per-omic loadings.
  - `mvae` — multi-view Product-of-Experts VAE (PyTorch); missing-modality aware,
    GPU-capable.
  - `mogonet` — MOGONET-style supervised per-omic dense GCN + VCDN fusion
    (PyTorch, no torch-geometric needed).
- **Interpretation layer** (`Interpreter` plugins): `factor_loadings`,
  `embedding_correlation`, `permutation_importance`, and optional `enrichment`
  (gseapy).
- **Robustness benchmarking**: `bench.robustness_curve` perturbs data with
  Gaussian noise or feature dropout and reports metric-vs-level curves.
- `OmicsData.copy()` for safe dataset duplication.

### Notes
- R-bridge classical methods (DIABLO, NEMO, MCIA, iCluster, RGCCA) and additional
  GNNs remain deferred, consistent with the Python-first plan.

## 0.1.0
- Package skeleton, `OmicsData` (MuData) container, plugin registries, config +
  run manifests, seeding/logging.
- Generic matrix-per-omic + sample-sheet loader (default entry point) + h5ad/h5mu.
- Preprocessing: normalize, impute, batch, feature-select, QC.
- Integrators: concat+PCA, SNF (from scratch).
- Tasks: clustering, classification, regression (core); survival (optional).
- Benchmark runner + CLI; synthetic dataset; full test suite.
