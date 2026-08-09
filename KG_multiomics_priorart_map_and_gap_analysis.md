# Knowledge-Guided & KG-Augmented Multi-Omics: Prior-Art Map + Gap Analysis

*Companion to the earlier literature review. Purpose: locate a defensible novelty gap for a knowledge-graph–augmented multi-omics package by mapping existing methods according to the **exact mechanism** by which they use biological knowledge.*

---

## 1. How to read this map

The useful question is not "who else combined a KG with multi-omics" (many have) but **"how, exactly, does each method inject knowledge, and at what level does it operate?"** Two axes organize everything:

**Axis 1 — Mechanism of knowledge use:**
- **(A) Architecture constraint** — knowledge fixes *which weights exist* (sparse "visible"/biologically-informed nets). Knowledge is baked into topology; not updatable.
- **(B) Knowledge-primed decoder/latent** — an autoencoder's latent nodes/decoder edges are tied to ontology terms/gene sets.
- **(C) Prior graph as feature graph** — a biological network (PPI/pathway/GRN) is the adjacency over which a GNN passes messages; omics are node features.
- **(D) KG embedding / heterogeneous multi-relational KG** — the KG *is* the object; entities/relations are embedded for link prediction.
- **(E) Network propagation** — omics scores are diffused over a fixed network.
- **(F) Omics → knowledge refinement** — data changes the graph (edge reweighting/pruning/proposal). The "reverse" direction.

**Axis 2 — Level of operation:**
- **Patient/sample-level** (predict a patient's subtype/survival/response), vs
- **Entity-level** (predict drug–disease, gene–disease, DDI edges), vs
- **Gene/network-level** (driver genes, functional modules).

Your stated vision — a full biological KG making patient multi-omics analysis better *and* multi-omics refining the KG — sits at the intersection of **(C/D) + (F)** operating at **patient-level**. That specific intersection is the thinnest part of the literature (Section 4).

---

## 2. Prior-art matrix

| Method (year) | Knowledge source | Mechanism | Level | Data / task | Interpretability | Key limitation vs your goal |
|---|---|---|---|---|---|---|
| **DCell** (2018, Ideker) | Gene Ontology hierarchy | A (6 nodes/term) | gene→phenotype | yeast genotype → growth/phenotype | subsystem activations | not human multi-omics; fixed topology |
| **DrugCell** (Kuenzi 2020) | GO hierarchy + drug structure | A | cell-line-level | 509K cell-line×drug pairs (GDSC+CTRP) → drug response | GO subsystems | genotype only; static hierarchy |
| **MOViDA / Ferraro** (2023) | GO + pathways | A | cell-line | multi-omics drug response | pathway nodes | narrow prior; no KG feedback |
| **KPNN** (Fortelny & Bock 2020) | literal regulatory network | A (node=gene, edge=regulation) | single-cell | TCR stimulation | node = gene/protein | single modality; fixed net |
| **P-NET** (Elmarakeby 2021, *Nature*) | Reactome (5 pathway layers) | A (~71k weights, sparse) | patient | prostate: mutation+CNV → primary vs metastatic | full ("visible"): genes→pathways | 2 omics only; Reactome fixed; no reverse loop |
| **DeepOmix** (2021) | signaling pathways | A | patient | multi-omics → survival | pathway layer | fixed prior |
| **NeST-VNN** (Park 2024) | NeST multiprotein assemblies | A | patient | tumor tasks | assembly nodes | single hierarchy |
| **Pathway Activity Autoencoders** (2026) | pathways | A+B | patient | breast: multi-omics → survival/subtype | pathway-constrained latent | fixed prior |
| **VEGA** (Seninge 2021) | Reactome / TF gene sets | B (sparse decoder) | single-cell | scRNA representation | latent = gene set | decoder-side only; single modality |
| **expiMap** (Lotfollahi 2023, *Nat Cell Biol*) | pathway gene sets | B + *soft membership learning* | single-cell | reference mapping + perturbation | latent = pathway; can add **de novo** nodes | closest to "updating knowledge," but decoder-side, single-omic, not a multi-relational KG |
| **OntoVAE** (Doncevic & Herrmann 2023) | full GO / HPO ontology | B (latent+decoder = ontology) | bulk + single-cell | genetic/drug perturbation | every node = ontology term | transcriptome-centric; ontology fixed |
| **GNNRAI** (2025, *npj Syst Biol Appl*) | prior-knowledge graphs (per omic) | C | patient | Alzheimer's multi-omics → biomarkers + interactions; handles incomplete | integrated gradients + integrated Hessians | uses pathway/PPI-style priors, not a full multi-relational KG; no reverse loop |
| **AMOGEL** (2025) | assoc-rule multi-omics graph + prior-knowledge edges | C (hybrid data+KG edges) | patient | BRCA/KIPAN subtype | attention gene ranking | KG edges are *auxiliary*; no KG refinement output |
| **MPK-GNN** (2023, *IEEE TNNLS*) | **multiple** gene-gene interaction networks | C (multi-graph fusion) | patient | multi-omics classification | feature-module attention | priors are PPI-type; graph fixed; one-directional |
| **Prior-knowledge multilevel GNN** (2024, *Brief Bioinform*) | GRN + pathways (staged) | C (gene→GRN→pathway) | patient | tumor risk | Grad-CAM pathways | hierarchical but fixed; one-directional |
| **DGP-AMIO** (2025, *Bioinformatics*) | integrated gene-interaction DBs | C + edge features | gene-level | disease-gene prediction (omics = node feat, DB = edge feat) | attention over edge types | entity/gene-level, not patient multi-omics |
| **Graphene** (2022, *Cell Systems*) | multiple molecular networks | C/D self-supervised | gene-level | gene–disease, comorbidity; **can update with new interactomes/omics**; refine GWAS | embedding analysis | gene-level; "update" ≠ per-patient omics-driven KG refinement |
| **R-GCN biomedical KG** (2026) | multi-relational gene-drug-disease KG | D (KGE / R-GCN) | entity | drug–disease repurposing + side-effect burden | relation-aware embeddings | omics only as features; no patient integration |
| **BioMedKG / PrimeKG++** (2025) | PrimeKG++ + LM text + sequences | D (contrastive + KGE) | entity | link prediction (incl. unseen nodes) | multimodal embeddings | KG-centric; not patient multi-omics analysis |
| **MOFGCN** | multi-omics cell-line similarity + drug het net | C/D | cell-line | drug response | — | similarity-derived; entity-level |
| **HyperNetWalk** (2026) | layered signaling-regulatory net, **PPI reweighted by tumor co-expression** | E + partial F | gene-level | personalized + cohort driver-gene ID | reverse inference | omics→edge reweight is a *heuristic*, gene-level, not learned/validated KG refinement |
| **Network-diffusion methods** | any prior network | E | gene/module | omics scores propagated | consensus features | topology fixed; omics don't change edges |
| *Contrast class:* **MOGONET, MoGCN, DeepMoIC, SUPREME, MOGAT** | *none (data-derived patient-similarity graph)* | — | patient | subtype/survival | VCDN / ablation / attention | graph is built from data, **not** knowledge — these are the non-KG baselines you must beat |

---

## 3. What each mechanism family has already locked down

**(A) Architecture-constraint VNN/BINN — saturated.** From DCell (2018) through P-NET, DeepOmix, NeST-VNN, a 2025 systematic review counted ~86 BINN/VNN papers. Using a pathway/ontology hierarchy to sparsify a net for an interpretable predictive task is now a *standard technique*, not a novelty. Notably, even *within* this crowded family there are open sub-questions the community admits: almost no one has systematically compared the impact of Reactome vs KEGG vs GO as the knowledge source, and no broad architecture search over the pathway layers has been published. A rigorous knowledge-source study is a small but real contribution.

**(B) Knowledge-primed AE/VAE — mature for single-cell.** VEGA, expiMap, OntoVAE tie latent/decoder to gene sets or ontologies. expiMap is the important one for you: it can learn *soft memberships* and add *de novo* latent nodes — i.e., it nudges the knowledge. But it does so on the decoder of a single-omic (transcriptome) VAE, with a flat pathway prior, not a multi-relational KG, and the "new relationships" are latent memberships, not audited KG edges.

**(C) Prior graph as feature graph — active and competitive.** GNNRAI, AMOGEL, MPK-GNN, multilevel-GNN all run GNNs over biological networks with omics as node features. This is exactly the "obvious" version of your idea and it is being actively published in 2024–2025. Crucially, though: their priors are **narrow** (PPI, single pathway DB, GRN), the graph is **fixed and global** (an OgBench analysis notes most omics-GNNs share one global topology across all samples), and the flow is **one-directional** (KG → prediction, never prediction → KG).

**(D) KG embedding — a separate world, entity-level.** R-GCN/KGE work (BioMedKG, PrimeKG++, polypharmacy/DDI models) treats the KG as the primary object and predicts *edges between entities* (drug–disease, drug–drug). Multi-omics, when present, is reduced to node features or cell-line similarity. These do not ingest a *patient's full multi-omics profile* — so they are prior art for "KG link prediction," not for "KG-augmented patient multi-omics integration."

**(E/F) Reverse direction — barely touched.** Omics-informed edge weighting exists but as **heuristics**: HyperNetWalk reweights PPI edges by tumor co-expression; DGP-AMIO uses database provenance as edge features. Graphene can be "updated with new interactomes or other omics." Network diffusion superimposes omics on nodes. None of these is a **learned, evaluated, first-class KG-refinement output** where per-patient multi-omics evidence *proposes, prunes, or re-scores KG edges* and the refined KG is then shown to help. This is the genuinely open territory.

---

## 4. The gap, stated precisely

Crossing mechanism × level against the existing work, the **empty cells** are:

1. **Full multi-relational KG (PrimeKG-scale: genes↔pathways↔diseases↔drugs↔phenotypes↔anatomy) as the integration substrate for patient-level multi-omics.** Existing (C) methods use *narrow* priors (one PPI or one pathway DB). Almost nobody feeds a patient's multi-omics into a *rich, multi-scale, multi-relational* KG for a patient-level task.
2. **Patient/sample-specific graph topology** derived from that patient's omics, instead of one fixed global graph shared across all samples (the limitation OgBench flags).
3. **A learned, validated bidirectional loop:** KG → integration, *and* multi-omics evidence → KG edge confidence / de novo edges, where the refined KG is evaluated both by (i) held-out edge recovery and (ii) downstream-task lift over the raw KG. expiMap (decoder soft membership), Graphene (network update), HyperNetWalk (co-expression reweighting) each do a *fragment* of this; none closes the full loop on a multi-relational KG with per-patient multi-omics.
4. **Provenance-tagged, auditable interpretation** where every prediction traces to named KG entities and specific cited edges — an engineering + scientific contribution that is currently ad hoc.
5. **Knowledge-source ablation as first-class science** (Reactome vs KEGG vs GO vs full KG), which the VNN community explicitly calls under-studied.
6. **Metabolite/lipid inclusion** into gene-centric KGs (weak coverage everywhere today).

---

## 5. Defensible novelty positions (ranked), each with closest prior art and how to beat it

### Position A — *headline novelty:* bidirectional omics ⇄ KG
**Claim:** A framework where (1) a full multi-relational biomedical KG (PrimeKG + Reactome + STRING) guides patient multi-omics integration, and (2) the integrated multi-omics evidence *refines the KG* — learned edge-confidence re-scoring plus de novo edge proposal — with the refined KG validated by held-out edge recovery and by improved downstream prediction versus the raw KG.
**Closest prior art:** expiMap (soft membership, decoder-side, single-omic, flat pathway prior); Graphene (network representation update, gene-level, no per-patient omics); HyperNetWalk (co-expression edge reweighting, heuristic, gene-level).
**How you beat it:** none of them (i) operate on a multi-relational KG, (ii) ingest per-patient multi-omics, and (iii) *evaluate* the refined KG as an output. Doing all three is new.
**Validation:** mask a fraction of KG edges → recovery AUROC conditioned on omics; then show downstream subtype/survival lift with refined-vs-raw KG; ablate the reverse loop entirely.
**Risk (high):** validating "the KG got better" needs a gold standard or a strong downstream proxy; scope this carefully or reviewers will call it unfalsifiable.

### Position B — *reliable backbone:* full-KG feature graph + KG-embedding priors, patient-level, with provenance
**Claim:** Patient multi-omics integrated over a *full multi-relational KG* (not just PPI/pathway), using KG-embedding (RotatE/R-GCN) node initialization + patient-specific subgraph extraction, emitting provenance-tagged explanations.
**Closest prior art:** GNNRAI, MPK-GNN, AMOGEL (all use narrow, fixed, one-directional priors).
**How you beat it:** richer multi-relational substrate + KGE priors + patient-specific topology + auditable interpretation; ablate to prove the *full KG* beats a single-PPI prior and beats non-KG baselines (MOGONET/DeepMoIC).
**Risk (moderate):** must show the richer KG actually helps, not just adds parameters.

### Position C — *resource contribution:* the package itself
**Claim:** An open, benchmarked, MuData-native package unifying the method families (MOFA/SNF/DIABLO/VAE/GNN) *plus* a first-class KG module, with a reproducible benchmark harness.
**Closest prior art:** fragmented single-method repos; platforms like OmicsAnalyst/OmicsNet (web, not a KG-native library).
**Risk (low):** citable as a resource regardless of whether A lands.

**Recommended combination:** ship **C** as the vehicle, build **B** as the dependable methodological core (publishable on its own), and pursue **A** as the differentiating headline — with the reverse loop scoped to one well-validated task first (e.g., omics-conditioned re-scoring of gene–disease edges evaluated on held-out associations).

---

## 6. Baselines and ablations you will be required to run

To make *any* KG claim credible, pre-register these comparisons:
- **Non-KG deep baselines:** MOGONET, DeepMoIC, SUPREME (data-derived patient-similarity graphs).
- **Non-KG classical baselines:** MOFA+, SNF/NEMO, DIABLO.
- **Narrow-prior KG baselines:** a single-PPI or single-pathway version of your own model (this isolates "full multi-relational KG" as the contribution).
- **KG-removed ablation:** identical architecture, edges shuffled/randomized (the P-NET reusability study showed random sparsification is the right control — knowledge must beat random structure).
- **Reverse-loop ablation:** model with vs without the omics→KG refinement.
- **Leakage audit:** ensure KG-derived features can't encode the label (e.g., disease→gene edges leaking the diagnosis).

---

## 7. One-paragraph positioning statement (draft)

*"Prior knowledge-guided multi-omics methods either (i) hard-code a single pathway or ontology as a fixed network architecture (P-NET, DeepOmix, OntoVAE) or (ii) run GNNs over a narrow, static gene-interaction prior (GNNRAI, MPK-GNN, AMOGEL); separately, biomedical KG-embedding methods predict entity-level links (drug–disease, DDI) without ingesting patient multi-omics. None integrate a patient's full multi-omics profile over a rich multi-relational knowledge graph while also using that multi-omics evidence to refine the graph. We introduce [NAME], which (1) integrates multi-omics over a PrimeKG-scale multi-relational KG with KG-embedding priors and patient-specific subgraphs, (2) refines KG edge confidence and proposes de novo edges from multi-omics evidence, validated by held-out edge recovery and downstream lift, and (3) emits provenance-tagged, entity-traceable explanations — released as an open, benchmarked package."*

---

## 8. Sources consulted for this map (by name/author)

VNN/BINN lineage: DCell (Ma et al. 2018); DrugCell (Kuenzi et al. 2020); MOViDA (Ferraro et al. 2023); KPNN (Fortelny & Bock 2020); P-NET (Elmarakeby et al. 2021, *Nature*) + reusability report (Pedersen et al. 2023); DeepOmix (2021); NeST-VNN (Park et al. 2024); Pathway Activity Autoencoders (2026); "Visible neural networks for multi-omics integration: a critical review" (Frontiers 2025, ~86 papers).
Knowledge-primed AE/VAE: VEGA (Seninge et al. 2021); expiMap (Lotfollahi et al. 2023, *Nat Cell Biol*); OntoVAE (Doncevic & Herrmann 2023, *Bioinformatics*).
Prior-graph GNNs: GNNRAI (2025, *npj Syst Biol Appl*); AMOGEL (2025); MPK-GNN (2023, *IEEE TNNLS*); prior-knowledge multilevel GNN (2024, *Brief Bioinform*); DGP-AMIO (2025, *Bioinformatics*); Schulte-Sasse et al. (GCN cancer-gene). Contrast: MOGONET (Wang et al. 2021), MoGCN, DeepMoIC, SUPREME, MOGAT.
KG embedding / heterogeneous KG: R-GCN drug-repurposing + side-effect KG (2026); BioMedKG / PrimeKG++ (2025); polypharmacy/DDI KGE; MOFGCN.
Reverse-direction / network: HyperNetWalk (2026); network-diffusion review (network-based integrative multi-omics, PMC9703081); Graphene (2022, *Cell Systems*); Ohmnet (Zitnik & Leskovec 2017); OgBench (fixed-global-topology observation). Reviews: "Graph machine learning for integrated multi-omics" (Valous et al. 2024, *Br J Cancer*); "A Review on Knowledge Graphs for Healthcare" (*J Biomed Inform* 2025).
