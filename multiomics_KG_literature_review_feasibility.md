# Multi-Omics Integration + Biological Knowledge Graphs
## Literature Review, Feasibility Analysis, and Package Requirements

*Prepared as a planning document for building an open multi-omics analysis package with a knowledge-graph–augmented integration approach as the proposed novel contribution.*

---

## 0. Executive summary

Multi-omics integration has moved through four broad waves: (1) classical statistical / matrix-factorization methods (iCluster, MOFA, JIVE), (2) multivariate correlation methods (CCA/PLS families, DIABLO), (3) similarity- and network-based fusion (SNF, NEMO), and (4) deep learning (variational autoencoders, and especially graph neural networks such as MOGONET and successors). A fifth, currently active wave layers **prior biological knowledge** into models — pathway-constrained "visible"/biologically-informed neural networks (P-NET, DeepOmix) and, increasingly, **knowledge-graph–guided GNNs** (GNNRAI, AMOGEL, MPK-GNN).

**The single most important finding for your plan:** "combine multi-omics with a biological knowledge graph" is *not* an empty niche. It is an emerging but already-populated research area. This does not kill your idea — it sharpens it. Your novelty cannot be "we added a KG"; it must be a *specific* mechanism, scope, or engineering contribution that existing KG-guided methods do not offer. Sections 4 and 8 lay out concrete ways to differentiate.

The package idea itself (a unified toolkit wrapping many integration methods with solid pre/post-processing) is **highly feasible and genuinely useful** — the field is fragmented across R and Python, and practitioners routinely complain that published methods are not "off-the-shelf." A well-engineered, well-documented package with consistent data structures and benchmarking would have value even before the KG contribution lands.

---

## Part 1 — How multi-omics evolved

### 1.1 Origins and drivers

Multi-omics grew out of systems biology's premise that a single molecular layer cannot explain complex phenotypes. The enabling driver was the falling cost and rising throughput of assays: genomics (WGS/WES, SNP arrays), transcriptomics (microarrays → RNA-seq), epigenomics (methylation arrays, ATAC-seq), proteomics and phosphoproteomics (mass spec, RPPA), and metabolomics. Once the *same* samples could be profiled on multiple layers, "integrate them" became both possible and necessary.

Large consortia created the raw material and the incentive:
- **TCGA** (The Cancer Genome Atlas): >11,000 patients across 33 cancer types with matched mRNA, miRNA, methylation, copy number, mutation, and (for many) RPPA protein data. This became the de facto substrate for method development.
- **CPTAC** (Clinical Proteomic Tumor Analysis Consortium): added deep proteomics/phosphoproteomics to many TCGA-style cohorts ("proteogenomics").
- Non-cancer consortia: **GTEx** (tissue expression + genotype), **Human Microbiome Project / HMP2 (iHMP)** (microbiome multi-omics), **ENCODE / Roadmap Epigenomics** (regulatory annotation).

### 1.2 The five waves (rough timeline)

| Wave | Period (approx.) | Representative approaches | Core idea |
|------|------------------|---------------------------|-----------|
| 1. Statistical / factorization | ~2009–2016 | iCluster / iClusterPlus / iClusterBayes, JIVE, MOFA | Find shared latent factors / joint clusters across layers |
| 2. Multivariate correlation | ~2011–2018 | CCA, RGCCA, sPLS, MCIA, moCluster, **DIABLO (mixOmics)** | Maximize covariance/correlation between blocks; supervised variants |
| 3. Similarity / network fusion | ~2014–2019 | **SNF**, NEMO, PSDF, LRAcluster, BCC | Build per-omics sample-similarity networks, then fuse |
| 4. Deep learning | ~2018–present | VAEs (MOFA+ moved here conceptually), **MOGONET**, MoGCN, DeepMoIC, SUPREME, MOGAT; single-cell: totalVI, MultiVI | Learn nonlinear joint representations; GNNs on patient/feature graphs |
| 5. Knowledge-infused | ~2020–present | P-NET, DeepOmix, NeST-VNN (visible/BINN); GNNRAI, AMOGEL, MPK-GNN (KG-guided GNN); LLM/foundation-model-assisted | Constrain or guide models with pathways / regulatory networks / knowledge graphs |

The waves are additive, not replacements: a 2025 pipeline may use SNF-style graph construction feeding a GNN whose architecture is constrained by pathway knowledge. Benchmarks consistently show that (a) integration usually beats best-single-omic, but (b) **no method dominates across datasets** — performance is dataset- and task-dependent (Rappoport & Shamir 2018 cancer benchmark; Duan et al. 2021 subtyping benchmark; Tini et al. 2019). This "no free lunch" result is the strongest justification for a *package that offers many methods* rather than betting on one.

---

## Part 2 — Taxonomy of integration methods (the "all methods" catalogue)

A useful organizing axis is **when** integration happens:

- **Early / concatenation-based:** stack all features into one matrix, then model. Simple; suffers from dimensionality imbalance across omics and loss of modality structure.
- **Intermediate / joint transformation:** learn a shared latent space or fused graph that jointly represents all layers (most modern methods). Best trade-off of the three.
- **Late / model-level:** model each omic separately, then combine predictions/similarities (ensemble, or fuse similarity networks).

Below is a method catalogue grouped by family. This is intended as the "wrap-list" for the package.

### 2.1 Matrix factorization / latent variable
- **iCluster / iClusterPlus / iClusterBayes** — joint latent model for clustering across genomic, epigenomic, transcriptomic layers; Bayesian variant improves feature selection. R.
- **MOFA / MOFA+ (MEFISTO for temporal/spatial)** — unsupervised Bayesian factor analysis; decomposes each omic into shared factors + per-omic weights; handles partially overlapping samples and missing data; strong interpretability via factor–weight inspection. Python/R (`mofapy2`, `MOFA2`).
- **JIVE** — separates Joint vs Individual Variation Explained across blocks.
- **LRAcluster** — low-rank probabilistic model for fast integrative clustering.

### 2.2 Multivariate correlation / covariance
- **CCA / RGCCA / SGCCA** — (regularized/sparse) generalized canonical correlation across many blocks.
- **sPLS / MCIA / moCluster** — sparse PLS and multiple co-inertia; MCIA and moCluster identify joint patterns across blocks. moCluster reported strong simulated-data performance.
- **DIABLO (mixOmics)** — *supervised* multi-block sPLS-DA: finds correlated, discriminative signatures across omics tied to an outcome. Widely used; R (`mixOmics`). Caveats: assumes aligned samples; not causal; leakage risk if feature selection/tuning done on full data.

### 2.3 Similarity / network fusion
- **SNF (Similarity Network Fusion)** — per-omic patient-similarity networks fused via message-passing into one network; frequently a top performer in cancer subtyping benchmarks. R/Python.
- **NEMO** — neighborhood-based; faster than SNF/iClusterBayes and supports *partial* datasets (samples missing some omics). Reported ~400× faster than iClusterBayes.
- **PSDF, BCC (Bayesian Consensus Clustering), Patient-Specific Data Fusion** — consensus/fusion variants.

### 2.4 Deep learning — non-graph
- **Autoencoders / Variational Autoencoders (VAEs)** — learn nonlinear joint latent space; conditional/multi-view VAEs handle missing modalities; product-of-experts / mixture-of-experts formulations for combining modality-specific encoders.
- **Single-cell generative models:** **totalVI** (RNA + surface protein / CITE-seq), **MultiVI** (RNA + ATAC / 10x Multiome), **PeakVI**, **DestVI** (spatial deconvolution). Python (`scvi-tools`). These are the single-cell analogue of bulk multi-omics integration and are very mature.
- **MOMA** — multi-task attention learning for interpretation + classification.

### 2.5 Deep learning — graph neural networks (currently dominant for supervised tasks)
- **MOGONET** (2021, Nat Commun) — the reference GNN approach: per-omic GCNs on cosine-similarity patient graphs + a View Correlation Discovery Network (VCDN) for label fusion. Enables classification + biomarker identification. Limitations noted in follow-ups: expensive ablation-based feature importance; graph built via a manually tuned similarity threshold.
- **MoGCN** — autoencoder dimensionality reduction + SNF graph + GCN; shallow, degrades on larger TCGA cohorts.
- **DeepMoIC** — autoencoders + SNF + *deep* GCN with residual/identity mappings; scales better than shallow GCNs.
- **SUPREME** — GCN framework integrating clinical + multi-omics; strong on survival/subtype tasks.
- **MODILM** — MOGONET-like but uses GAT (attention) instead of GCN.
- **MOGAT** — graph-attention survival analysis; empirically found ~3 omics often optimal (more omics ≠ better).
- **MO-GCAN, MOLUNGN, MOTGNN, TF-DWGNet, CMGL** — 2024–2025 refinements adding attention, gating, tensor fusion, confidence weighting, directed/weighted edges, or interpretability.
- **CLCLSA** — handles *incomplete* multi-omics via cross-omics autoencoders + contrastive learning + self-attention.

### 2.6 Knowledge-infused / biologically-informed (directly relevant to your KG idea)
This is the family your proposal lands in — read carefully.
- **Visible / Biologically-Informed Neural Networks (VNN/BINN):** network *connectivity* is constrained by gene→pathway→process hierarchies (Gene Ontology, Reactome, KEGG) so weights are interpretable.
  - **P-NET** (Elmarakeby et al. 2021, Nature) — Reactome-structured sparse net; classified primary vs metastatic prostate cancer from mutation + CNV; became the canonical example.
  - **DeepOmix** — signaling-pathway-informed, survival prediction from multi-omics.
  - **NeST-VNN** — uses multiprotein assemblies as the hierarchy.
  - A 2025 systematic review found **86** BINN/VNN papers — the sub-field is already substantial.
- **Knowledge-graph–guided GNNs (the closest prior art to your idea):**
  - **GNNRAI** — per-omic GNN feature extractors run over *prior-knowledge graphs*, aligned + integrated; handles incomplete data; identifies biomarkers *and* cross-domain interactions (demonstrated on Alzheimer's).
  - **AMOGEL** — association-rule-mined multi-omics graph + prior-knowledge edges as auxiliary; attention-based gene ranking.
  - **MPK-GNN** (Multiple Prior Knowledge GNN) — DNN sample module + GNN feature module fusing prior-knowledge graph with omics.
  - **Prior-knowledge multilevel GNN** — hierarchical gene→regulatory-network→pathway feature extraction for tumor risk.

**Takeaway:** pathway/network priors and even explicit KGs are already being fused with multi-omics GNNs. What is *comparatively* underexplored (your opportunity space) is discussed in Part 4.

---

## Part 3 — Preprocessing & postprocessing utilities the package will need

A large fraction of real-world effort is not the integration algorithm — it is the plumbing. A credible package must cover:

### 3.1 Preprocessing / harmonization
- **Ingestion & alignment:** map each omic to a common sample index; enforce "sample X means the same entity in every block"; handle partially overlapping samples.
- **Feature ID harmonization:** the single biggest silent source of bugs. Map probes/transcripts/proteins/metabolites to stable identifiers (Ensembl gene IDs, UniProt, HGNC, ChEBI/HMDB, CpG→gene). Needed anyway to link omics features to KG nodes.
- **Missing data:** per-feature filtering; imputation (mean, KNN, matrix-completion, modality-transfer via VAEs); and *support for methods that natively handle missingness* (NEMO, MOFA, CLCLSA).
- **Normalization & transformation:** per-omic (e.g., log-CPM/TPM for RNA, M-values for methylation, VSN/log for proteomics, quantile/standardization); scaling to prevent high-dimensional omics from dominating concatenation.
- **Batch-effect correction:** ComBat / limma / Harmony / scVI-style; critical because integration can otherwise learn batch instead of biology.
- **Feature selection / dimensionality reduction:** variance/MAD filtering, univariate association, or model-embedded sparsity; document leakage risk (do selection inside CV folds).
- **Quality control:** per-omic QC reports, sample outlier detection, distribution diagnostics.

### 3.2 Core integration layer
- Unified API wrapping the method families in Part 2, ideally on a common in-memory structure (see MuData/MultiAssayExperiment in Part 5).
- Consistent handling of supervised vs unsupervised, paired vs partial, bulk vs single-cell.

### 3.3 Postprocessing / interpretation
- **Clustering evaluation:** silhouette, ARI/NMI (vs known subtypes), and **survival separation** (log-rank / Kaplan–Meier) — the standard cancer-subtyping yardsticks.
- **Classification evaluation:** macro precision/recall/F1 (class imbalance is common), AUROC/AUPRC, calibrated CV.
- **Biomarker / feature attribution:** integrated gradients, attention weights, SHAP; cross-omics interaction extraction.
- **Biological interpretation:** pathway/GO enrichment, network visualization — the natural hand-off point to the KG.
- **Reproducibility scaffolding:** seed control, config capture, run manifests, benchmark harness on standard datasets.

---

## Part 4 — The KG contribution: feasibility & where the real novelty is

### 4.1 What knowledge graphs bring
Biomedical KGs integrate heterogeneous relations (genes, proteins, pathways, diseases, drugs, phenotypes, anatomy, side effects) into one queryable structure. Leading resources:

| KG | Scale / focus | Notes |
|----|---------------|-------|
| **PrimeKG** | ~129K nodes, ~4M edges, 10 biological scales, 17K diseases | Precision-medicine oriented; rich drug–disease (indication/contraindication/off-label) edges; clinical-text descriptions. Most cited for AI use. |
| **Hetionet** | compounds, diseases, genes, pathways, symptoms, anatomy | Drug-repurposing focus (Himmelstein 2017). |
| **CKG (Clinical Knowledge Graph)** | 16M+ nodes, 220M+ rels | Proteomics-workflow oriented, open-source platform. |
| **SPOKE** | large heterogeneous biomedical KG | Connects omics to clinical/real-world data. |
| **Monarch, DisGeNET, HPO, Reactome, KEGG, STRING/BioGRID, OmniPath** | phenotype/gene-disease/pathway/PPI | The building blocks many KGs and pathway-informed models draw on. |
| **PrimeKG++** | augmented PrimeKG | Adds richer gene/protein context + LM embeddings. |

The value of a KG comes from *integration across sources*, letting analyses surface indirect links (e.g., a drug's off-label mechanism overlapping an underexplored disease pathway). Important caveat to state honestly in any paper: **a KG is a queryable hypothesis space, not ground truth** — construction choices (which edges, confidence scoring, contradiction handling) shape everything downstream.

### 4.2 Honest novelty assessment
Because pathway-informed and KG-guided multi-omics GNNs already exist (Part 2.6), "we combine multi-omics with a KG to improve results" will read as incremental to reviewers. To be genuinely novel, differentiate along one or more of these axes:

1. **Mechanism of KG use.** Most prior work uses the KG as *static architectural priors* (fix which gene→pathway edges exist). Less explored: using KG **embeddings** (e.g., TransE/RotatE/knowledge-graph node2vec, or an LLM over KG text) as *initialization or auxiliary features*; **message passing across a heterogeneous KG jointly with omics**; or **KG-derived attention priors** that are *learned/updated* rather than fixed.
2. **Feature-graph vs patient-graph.** Many GNN methods build *patient* similarity graphs. Using the KG as the *feature* graph (genes/proteins as nodes, KG relations as edges, omics values as node features) and doing joint patient×feature reasoning is a cleaner, less crowded framing.
3. **Bidirectional benefit (your stated goal).** You said the KG should make multi-omics better *and* multi-omics should make the KG better. The second direction is the more novel half: use multi-omics evidence to **weight, prune, or propose edges** in the KG (data-driven KG refinement / link prediction conditioned on omics). Very little published work closes this loop well.
4. **Missing-modality / cross-omic imputation via KG.** Use KG connectivity to impute or regularize a missing omic layer — a practically valuable, underexplored niche.
5. **Interpretability + provenance.** A KG lets every prediction be traced to named biological entities and cited edges. A package that outputs *auditable, provenance-tagged explanations* is a real engineering/scientific contribution even if the model core is conventional.
6. **Generalization / transfer.** Show KG priors improve performance specifically in **low-sample** regimes or **cross-cohort transfer** — the situation where priors help most (small-effect functional features, as GNNRAI showed).

A defensible framing: *"An open, modular multi-omics package whose novel contribution is a knowledge-graph layer that (a) supplies a heterogeneous feature graph and KG-embedding priors for integration, and (b) refines the KG using multi-omics evidence, with fully provenance-tagged interpretation."* That is specific, bidirectional, and not obviously covered by GNNRAI/AMOGEL/MPK-GNN.

### 4.3 Feasibility verdict
- **Package (methods + pre/post):** feasible, high value, moderate effort. Main cost is engineering breadth and cross-language wrapping.
- **KG-augmented integration (direction 1–2, KG→omics):** feasible; competitive field, so novelty must be crisp.
- **KG refinement from omics (direction 3):** feasible and more novel, but harder to *validate* (needs a gold standard of "correct" edges or a downstream task where refined-KG demonstrably helps).
- **Biggest risks:** (i) identifier harmonization and KG–omics entity mapping is laborious and error-prone; (ii) coverage gaps (metabolites/lipids map poorly to gene-centric KGs); (iii) benchmark leakage and over-optimistic results if KG-derived features leak label information; (iv) proving the KG *causally* adds value beyond a strong non-KG baseline — always include ablations (same model, KG removed).

---

## Part 5 — Package architecture & requirements

### 5.1 Language & data-structure decision
The ecosystem is split:
- **Python:** `scanpy`/`anndata`, `muon`/**`mudata`** (multimodal container), `scvi-tools` (deep single-cell integration), PyTorch + **PyTorch Geometric / DGL** (GNNs), scikit-learn. Best for the deep-learning and KG-GNN core.
- **R/Bioconductor:** `mixOmics` (DIABLO), `MOFA2`, `SNFtool`, `iClusterPlus`, **`MultiAssayExperiment`** (multimodal container), `SingleCellMultiModal` (ready-made datasets). Best for classical methods.

Recommendation: **Python-first** (for GNN/KG and long-term ML tooling), adopt **MuData/AnnData** as the native structure, and reach into R methods either by reimplementation or via `rpy2`/subprocess wrappers where a method has no good Python equivalent (e.g., some iCluster/DIABLO variants).

### 5.2 Suggested module layout
- `io/` — loaders for TCGA/CPTAC/LinkedOmics/MLOmics/AnnData/MuData; sample alignment.
- `harmonize/` — feature-ID mapping to Ensembl/UniProt/HGNC/ChEBI; KG-node linking.
- `preprocess/` — normalization, batch correction, imputation, feature selection, QC reports.
- `integrate/` — wrappers: `mofa`, `snf`, `nemo`, `diablo`, `mogonet`, `vae`, plus your `kg_gnn`.
- `kg/` — KG loaders (PrimeKG, Hetionet, Reactome, STRING), embedding utilities, feature-graph construction, KG-refinement/link-prediction, provenance.
- `evaluate/` — clustering/classification/survival metrics, benchmark harness.
- `interpret/` — attribution, enrichment, provenance-tagged explanations, visualization.
- `benchmarks/` — reproducible pipelines on standard datasets (Rappoport–Shamir, MLOmics, NeurIPS-2021).

### 5.3 Engineering requirements
- Consistent, typed API; config-driven runs; deterministic seeding.
- Test suite + CI; documented tutorials; example notebooks per method.
- Compute: bulk-omics methods run on CPU/modest GPU; GNNs and single-cell VAEs want a GPU; KG embedding of PrimeKG-scale graphs (~4M edges) is tractable on one modern GPU but plan for memory.
- Licensing: check each wrapped method's license; check each dataset's access tier (open vs controlled) and each KG's redistribution terms before bundling.

---

## Part 6 — Freely accessible datasets & databases for a multi-omics pilot

Grouped by how "ready-to-run" they are. Prioritize the ML-ready benchmarks first for a fast pilot.

### 6.1 ML-ready benchmarks (start here)
- **MLOmics** (GitHub: chenzRG/Cancer-Multi-Omics-Benchmark) — 8,314 samples, all 32 TCGA cancer types, four omics, stratified features, provided baselines, subtype labels, imputation tasks. Purpose-built to be off-the-shelf for ML. **Best first pilot dataset.**
- **Rappoport & Shamir cancer benchmark** — 10 TCGA cancer types with mRNA, methylation, miRNA; the classic clustering benchmark. Processed data + scripts public (acgt.cs.tau.ac.il/multi_omic_benchmark; GitHub Shamir-Lab/Multi-Omics-Cancer-Benchmark). Standard for comparing clustering methods.
- **Duan et al. 2021 subtyping benchmark** — 10 methods across 9 TCGA cancers with noise-robustness protocol; useful as an evaluation template.

### 6.2 Bulk multi-omics portals (download & assemble)
- **TCGA via GDC** (portal.gdc.cancer.gov) and **cBioPortal** (cbioportal.org) — matched mRNA, miRNA, methylation, CNV, mutation, RPPA + clinical/survival. Open tier needs no certification; controlled tier (individual genotypes) needs dbGaP approval.
- **CPTAC** (proteomic.datacommons.cancer.gov / PDC) — proteomics + phosphoproteomics matched to genomics ("proteogenomics").
- **LinkedOmics** (linkedomics.org) — 32 TCGA + 10 CPTAC cohorts with a query/association interface; convenient for pulling matched layers without raw-file wrangling.
- **cBioPortal** also aggregates many non-TCGA public multi-omic studies.

### 6.3 Non-cancer / population multi-omics
- **GTEx** (gtexportal.org) — expression + genotype across tissues; eQTL-style multi-omics.
- **Human Microbiome Project / iHMP (HMP2)** — microbiome multi-omics (metagenomics, metatranscriptomics, metabolomics, host).
- **UK Biobank** (application required) and **MetaboLights / Metabolomics Workbench** (open) — metabolomics repositories.
- **GEO** and **ArrayExpress/BioStudies** — vast raw repositories where many matched multi-omic studies live (e.g., GSE accessions).
- **OmicsDI** (omicsdi.org) — a discovery index across omics repositories; good for finding matched datasets.

### 6.4 Single-cell multi-omics (if you go single-cell)
- **NeurIPS 2021 Multimodal Single-Cell Integration** — bone-marrow mononuclear cells, paired **10x Multiome (RNA+ATAC)** and **CITE-seq (RNA+protein)**; GEO **GSE194122**; also on Kaggle and openproblems.bio. Largest realistic single-cell multimodal benchmark with defined tasks/metrics. **Best single-cell pilot.**
- **SingleCellMultiModal** (Bioconductor) — one-command access to landmark CITE-seq, scNMT, 10x Multiome, seqFISH, G&T, SCoPE2, ECCITE-seq datasets, pre-packaged as MultiAssayExperiment.
- **10x Genomics public datasets** — PBMC Multiome, mouse brain Multiome, NSCLC/kidney CITE-seq, etc.
- **Human Cell Atlas / CELLxGENE** — large curated single-cell corpora.

### 6.5 Knowledge graphs & biological priors (for the KG layer)
- **PrimeKG** (open; harvard/Marinka Zitnik lab) — primary candidate; rich, AI-tuned, well-documented.
- **Hetionet** (open) — good for drug-repurposing framing.
- **CKG** (open-source) — proteomics-centric, very large.
- **Reactome, KEGG, WikiPathways** — pathway priors for BINN/VNN-style constraints.
- **STRING, BioGRID, OmniPath, HumanBase** — protein–protein / regulatory networks for feature-graph edges.
- **DisGeNET, HPO, Monarch, DrugBank/ChEMBL** — gene–disease, phenotype, drug relations to enrich the KG.

*Access note:* TCGA/CPTAC open tiers, LinkedOmics, MLOmics, Rappoport–Shamir, NeurIPS-2021, Bioconductor datasets, PrimeKG, Hetionet, Reactome, STRING, DisGeNET are all freely downloadable. Individual-level controlled genotype data (dbGaP) and UK Biobank need applications. Always re-check the license before redistributing anything inside your package.

---

## Part 7 — Recommended phased plan

**Phase 0 — Scaffolding (fast).** Adopt MuData/AnnData; build `io` + `harmonize` + `preprocess` + `evaluate`. Reproduce a known result (e.g., SNF or MOFA on the Rappoport–Shamir benchmark, or a baseline on MLOmics). This alone validates the plumbing and gives you baselines.

**Phase 1 — Method breadth.** Wrap 4–6 representative methods across families (MOFA, SNF/NEMO, DIABLO, a VAE, MOGONET). Ship a benchmark harness. At this point the package is already useful and citable as a resource.

**Phase 2 — KG layer.** Integrate PrimeKG (+ Reactome/STRING). Build KG-node linking and a KG-derived feature graph. Implement a KG-guided GNN with a **strict ablation** (identical model minus KG) to isolate the KG's contribution.

**Phase 3 — Novel contribution.** Pursue one sharp differentiator from Section 4.2 — most promising: KG-as-feature-graph + KG-embedding priors *and* the bidirectional loop (omics-driven KG edge weighting/refinement) with provenance-tagged interpretation. Validate on ≥2 cohorts and report low-sample / cross-cohort gains.

**Phase 4 — Release.** Docs, tutorials, tests, a benchmarking leaderboard, and a methods paper positioning the package as both a resource and the KG method's home.

### Key risks to manage throughout
1. Identifier/entity harmonization is the hidden time sink — budget for it.
2. Always ablate the KG; reviewers will demand proof it beats a strong non-KG baseline.
3. Guard against leakage (feature selection and KG-derived features inside CV).
4. Metabolomics/lipidomics map poorly onto gene-centric KGs — scope which omics your KG layer supports.
5. Validating omics-driven KG refinement needs a concrete downstream task, not just "the graph looks better."

---

## Selected references (by name/author for follow-up)

Reviews & benchmarks: technical review of integration methods (Briefings in Bioinformatics 2025 / arXiv 2501.17729); Picard et al. 2021 integration strategies; Rappoport & Shamir 2018 (NAR) cancer benchmark; Duan et al. 2021 (PLoS Comput Biol) subtyping benchmark; Tini et al. 2019; "Harnessing AI in Multi-Modal Omics" (Annu Rev Biomed Data Sci 2024); BioDesign Research 2024 multiomics principles.
Classical methods: iCluster/iClusterPlus/iClusterBayes; MOFA/MOFA+; JIVE; DIABLO (mixOmics); SNF; NEMO; MCIA; moCluster.
Deep/GNN: MOGONET (Wang et al. 2021, Nat Commun); MoGCN; DeepMoIC (2024); SUPREME; MOGAT (2024); MODILM; CLCLSA (2024); MO-GCAN/MOLUNGN/MOTGNN/TF-DWGNet/CMGL (2024–2025); totalVI/MultiVI (scvi-tools).
Knowledge-infused: P-NET (Elmarakeby et al. 2021, Nature); DeepOmix; NeST-VNN; VNN/BINN systematic review (Frontiers 2025); GNNRAI (npj Syst Biol Appl 2025); AMOGEL; MPK-GNN; prior-knowledge multilevel GNN (Brief Bioinform 2024).
Knowledge graphs: PrimeKG (Chandak et al. 2023, Sci Data); PrimeKG++; Hetionet (Himmelstein et al. 2017); CKG; SPOKE; Monarch; DisGeNET.
Datasets: TCGA/GDC; CPTAC/PDC; LinkedOmics; MLOmics; NeurIPS-2021 multimodal single-cell (GSE194122); SingleCellMultiModal (Bioconductor).
