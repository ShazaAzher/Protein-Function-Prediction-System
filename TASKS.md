# Build plan — protein-fn-pred

Revised from the original candidate-generation/rerank/graph-agent plan. Changes and why:

- **No Neo4j.** True-path propagation is a reverse-topological max-pass over ~40–50k GO terms — milliseconds in-memory with `networkx`/`scipy.sparse`. A live graph DB adds a server dependency Kaggle/Colab don't have by default, for a problem that isn't a database problem.
- **No LLM text synthesis or LLM conflict agent as core path.** At CAFA test-superset scale (order 10^5 sequences), per-protein LLM calls are a real cost/latency problem, not a hypothetical one, and neither played a role in top CAFA5 solutions.
- **Classifier + homology ensemble instead of retrieval-and-rerank**, mirroring what actually placed on CAFA5 (bounded top-K label space per ontology, PLM embeddings, gradient-boosted/MLP classifier, homology transfer as a parallel signal, propagation not agent-adjudication).
- **The paper's contribution is the Partial-Knowledge (PK) setting**, introduced in the CAFA5 organizers' April 2026 analysis paper: proteins with existing same-aspect annotations, predicting *additional* new terms in that same aspect. PK now accounts for ~70% of accumulated annotations and current methods measurably underperform on it (47–60% F-max drop vs. no-knowledge), because — per the organizers — no CAFA5 submission was built to specifically exploit the known-annotation signal. That's the gap this project targets.

---

## Phase 0 — Data foundation & PK infrastructure

Nothing downstream is trustworthy without correct parsing, joins, and splits — and the PK split in particular *is* the experimental design for the paper, not a side task.

**0.1 OBO parser** (`data/parsing.py`)
Parse `go-basic.obo` into a graph: term ID, `name`, `namespace` (→ MFO/BPO/CCO), `def` text, `is_a`/`part_of` edges.

**0.2 FASTA + TSV parsers** (`data/parsing.py`)
`train_sequences.fasta`, `testsuperset.fasta` (no curated text — confirmed), `train_terms.tsv`, `train_taxonomy.tsv`, `testsuperset-taxon-list.tsv`, `IA.tsv`. Validate join integrity (every `EntryID` in `train_terms.tsv` resolves in `train_sequences.fasta`).

**0.3 Taxonomy tiering** (`data/taxonomy.py`)
`taxon_id -> kingdom` for observed species; tag well-annotated vs. sparse tier from frequency in `train_taxonomy.tsv`. Drives candidate-generation routing and, later, whether the PK-conditioning gain holds for sparse species.

**0.4 NK / LK / PK flagging + known-term extraction** (`data/dataset.py`)
Per protein, per aspect: classify NK / LK / PK exactly per the CAFA5 definitions, and extract the known-term set T0 for PK proteins. This is now central plumbing — every downstream model branches on it.

**0.5 Synthetic PK/NK validation split** (`eval/pk_split.py`)
No timestamp column in `train_terms.tsv`, so real temporal accretion isn't directly constructible. Simulate it: for proteins with ≥2 terms in an aspect, randomly mask a fraction as "newly added" (held-out target, T1\T0), keep the rest as T0. Proteins with zero known terms in that aspect form the NK comparison group. This is a proxy for real accretion — say so plainly in the writeup.

**0.6 In-memory GO ancestor matrix + true-path propagation** (`go_graph/ancestors.py`)
Precomputed once from the OBO graph. Replaces the originally-planned Neo4j load entirely.

---

## Phase 1 — Metrics and baselines (validate the loop before anything fancy)

**1.1 IA-weighted F-max, PK-aware** (`eval/metrics.py`)
Per the CAFA5 derivation, conditional information content for PK collapses to ordinary IA weights summed over the newly-added terms only — no novel metric needed, just correct filtering: exclude T0 from both predictions and ground truth when scoring PK-setting proteins.

**1.2 Naive frequency baseline** (`eval/baseline.py`)
Run first, on both the NK and PK groups separately, to validate the metric implementation and get a sanity floor. The NK/PK gap on this trivial baseline is itself worth reporting.

**1.3 Homology-transfer baseline** (`candidates/sequence_similarity.py`)
Diamond/MMseqs2 search against `train_sequences.fasta`, kingdom-tier routing. Doubles as an ensemble partner later (Phase 3).

**1.4 Cold-start classifier** (`model/encoder.py`, `model/classifier.py`)
ESM2 embeddings → multi-label classifier over the top-K terms per ontology (config: `label_space` in `pipeline.yaml`). Trained and evaluated normally — no knowledge of T0. This is the "no conditioning" arm of the ablation.

---

## Phase 2 — Known-term conditioning (the paper's core contribution)

**2.1 GO term embeddings** (`go_graph/term_embeddings.py`)
Co-occurrence-based embedding from `train_terms.tsv` to start; node2vec/DeepWalk over the DAG as a stretch goal if the simple version underperforms.

**2.2 Known-term-set encoder** (`condition/known_terms.py`)
Pool a protein's T0 term embeddings into a fixed-size conditioning vector.

**2.3 PK-conditioned classifier** (`model/classifier.py`)
Identical backbone to 1.4, with `[ESM2 embedding ; known-term conditioning vector]` as input. Trained on the PK portion of the synthetic split (Task 0.5). The *only* difference from 1.4 is whether T0 is fed in — that's the whole experiment.

**2.4 Co-occurrence conditional prior** (`condition/cooccurrence_prior.py`)
Cheap, interpretable secondary baseline: P(new term | known terms) from direct cross-tabulation of `train_terms.tsv`. Not the main method, but a necessary comparison point — if this alone captures most of the PK gain, that's an important finding to report honestly rather than bury.

**2.5 Ablation harness**
Run 1.4 (cold-start) vs. 2.3 (PK-conditioned) vs. 2.4 (co-occurrence prior) on the identical synthetic PK split, broken down by ontology and taxon tier (0.3). This produces the paper's central result table.

---

## Phase 3 — Ensemble and full pipeline

**3.1 Score blending** (`model/ensemble.py`)
Blend classifier scores (routed cold-start or PK-conditioned per protein, per the Phase 0.4 flag) with homology-transfer scores. Configurable weights in `pipeline.yaml`.

**3.2 Threshold tuning** (extends `eval/metrics.py`)
Per-ontology threshold search against the validation split, optimizing weighted F-max.

**3.3 Pipeline orchestration** (`pipeline/build_candidates.py`, `pipeline/train_classifier.py`, `scripts/run_pipeline.py`)
Candidate generation (expensive, cacheable) separated from classifier training + ensembling (cheap, iterable).

**3.4 Submission formatting** (`pipeline/make_submission.py`)
Match `sample_submission.tsv` schema for the CAFA6 test superset.

---

## Phase 4 — Analysis for the paper

**4.1** PK-conditioning gain broken down by taxon tier — does it hold for sparse species where T0 might be less informative?
**4.2** Gain broken down by known-term-set size — is there a point of diminishing or negative returns?
**4.3** Qualitative error analysis on a handful of cases.
**4.4** Figures matching the CAFA5 paper's presentation (bootstrapped F-max with confidence intervals) for direct comparability.

---

## Phase 5 — Tests

Minimum coverage before trusting any tuned output:
- `test_parsing.py` — OBO/FASTA/TSV parsers against fixtures.
- `test_taxonomy.py` — kingdom mapping for all observed taxa.
- `test_pk_split.py` — no leakage of held-out terms into T0, correct NK/PK assignment.
- `test_metrics.py` — F-max against hand-computed toy examples, PK filtering correctness.
- `test_ancestors.py` — propagation monotonicity on a synthetic DAG.
- `test_known_terms.py` — NK/LK/PK flagging matches the CAFA5 definitions exactly.

---

## Suggested order

1. Phase 0 in full — especially 0.5, since it's the experimental design.
2. Phase 1.1 + 1.2 — validated scoring loop, early.
3. Phase 1.3 + 1.4 — real baselines, both submittable on their own.
4. Phase 2 in full — this is the actual contribution; don't rush it.
5. Phase 3 — wrap into a repeatable, submittable pipeline.
6. Phase 4 — once 2.5's ablation is stable, not before.

**Open item before locking this in:** a targeted search of bioRxiv/arXiv for post-April-2026 work explicitly targeting the PK setting — the organizers' paper is recent enough that nothing turned up yet, but that's weak evidence, not proof.
