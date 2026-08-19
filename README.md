# Protein Function Prediction System
## Architecture Documentation

**Task:** Predict Gene Ontology (GO) term annotations for protein sequences across three subontologies — Molecular Function (MFO), Biological Process (BPO), Cellular Component (CCO) — evaluated by information-accretion-weighted F-max (CAFA protocol).

**Approach:** Retrieval-and-rerank over GO term semantics, grounded in a knowledge graph, rather than per-term multi-label classification.

---

## 1. Problem Framing

Standard multi-label classification treats each of the ~40,000–50,000 GO terms as an independent target class. This breaks down for two structural reasons specific to this dataset:

1. **Extreme label sparsity.** Most GO terms have a handful of experimentally-validated positive examples in `train_terms.tsv`. A classifier head trained per term (or jointly, with per-term output units) has almost no signal for the long tail, which is exactly where the weighted F-max metric rewards correct predictions most (via `IA.tsv`).
2. **The labels are not opaque categories — they are natural-language concepts.** Every GO term in `go-basic.obo` carries a human-written definition. Treating "GO:0003824 — catalytic activity" as class index #4021 throws away the one signal that best distinguishes rare terms from each other: what they *mean*.

This motivates reframing the task as **semantic retrieval**: given a protein's available evidence (sequence, homologs, literature/description text), retrieve and rank the GO terms whose definitions best match that evidence, rather than fitting a classifier per label.

---

## 2. System Overview

```
                    ┌──────────────────────────────┐
                    │      Test Protein Input      │
                    │  (sequence, taxon ID)        │
                    └───────────────┬──────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
            ▼                       ▼                       ▼
┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│  Sequence-Similarity│   │  Embedding k-NN     │   │  Text/Description   │
│  Candidate Generator│   │  Candidate Generator│   │  Synthesis (LLM)    │
│  (Diamond/MMseqs2)  │   │  (ESM2/ESM3 + ANN)  │   │  for novel proteins │
└──────────┬──────────┘   └──────────┬──────────┘   └──────────┬──────────┘
           │                          │                          │
           └──────────────┬───────────┴──────────────────────────┘
                           ▼
              ┌───────────────────────────────┐
              │   Candidate GO Term Shortlist │
              │        (~50–200 terms)        │
              └───────────────┬───────────────┘
                              ▼
              ┌────────────────────────────────┐
              │  Semantic Reranker             │
              │  (protein text ↔ GO definition)│
              │  cross-encoder / LLM scorer    │
              └───────────────┬────────────────┘
                              ▼
              ┌────────────────────────────────┐
              │   Graph-RAG Consistency Layer  │
              │   (Neo4j: GO DAG structure)    │
              │   - true-path propagation      │
              │   - sibling/ancestor conflict  │
              │     resolution via LLM agent   │
              └───────────────┬────────────────┘
                              ▼
              ┌─────────────────────────────────┐
              │   Score Ensemble + Thresholding │
              │   (tuned against IA-weighted    │
              │    F-max on temporal val split) │
              └───────────────┬─────────────────┘
                              ▼
                   Final GO Term Predictions
```

---

## 3. Component Details

### 3.1 Candidate Generation (recall stage)

Purpose: cheaply narrow ~40,000 possible terms down to a shortlist worth scoring carefully, per protein.

| Generator | Method | Data used |
|---|---|---|
| Sequence similarity | Diamond or MMseqs2 alignment of test sequences against `train_sequences.fasta` | Top-k homologs → union of their annotated terms in `train_terms.tsv`, weighted by identity/bitscore |
| Embedding k-NN | Protein language model embeddings (ESM2-650M/3B, or ESM3 for joint sequence-structure-function representations), approximate nearest-neighbor search (FAISS) against cached train embeddings | Terms from nearest neighbors in embedding space |

Both generators run independently and their candidate sets are unioned. This stage is intentionally recall-oriented (cast a wide net, cheaply) — precision is enforced downstream.

### 3.2 Text Synthesis for Description-Poor Proteins

GORetriever-style methods rely on each protein having descriptive text (UniProt free-text description, associated literature). Most `train_sequences.fasta` entries have this from Swiss-Prot curation. For test proteins that lack rich text, an LLM generates a short functional description conditioned on:
- the protein's top sequence homologs and their known annotations,
- conserved domain/motif information if available,
- taxonomic context.

This description is only a bridge to enable text-based reranking — it is not used as a ground-truth label, and downstream scoring still requires corroboration from the candidate generators.

### 3.3 Semantic Reranker

For every (protein, candidate GO term) pair:
- Retrieve the term's definition text from `go-basic.obo` (`def:` field).
- Score relevance between the protein's text (curated or LLM-synthesized) and the term definition using either:
  - a sentence-embedding cross-encoder fine-tuned on GO definition ↔ protein description pairs, or
  - an LLM prompted to output a 0–1 relevance score with brief justification.
- This is the step that recovers signal for rare/long-tail terms, since it doesn't require the term to have appeared often in training — only that its *definition* is semantically close to the protein's evidence.

### 3.4 Graph-RAG Consistency Layer

The GO DAG (`go-basic.obo`) is loaded into Neo4j, mirroring the knowledge-graph pattern already used for DrugSense and threat_graph_new: nodes are GO terms, edges are `is_a` / `part_of` relations.

Two operations run here:
1. **True-path propagation.** For every predicted term, its score is propagated to all ancestors (`score(ancestor) = max(score(ancestor), score(descendant))`), guaranteeing DAG-consistent output, which the CAFA evaluation protocol requires implicitly.
2. **Conflict resolution via graph-local reasoning.** Where the reranker assigns similar scores to a term and a more specific child/sibling, an LLM agent inspects the local subgraph (the term, its parents, its children, and their definitions) and decides whether to promote the more specific term — this matters most in BPO, which has deep, highly branched hierarchies where naive score propagation tends to over-flatten predictions toward generic ancestor terms.

### 3.5 Ensemble and Thresholding

Final score per (protein, term) is a weighted blend of:
- sequence-similarity score,
- embedding k-NN score,
- semantic-reranker score (post graph-consistency adjustment).

Blend weights and the final decision threshold are tuned on a **temporal validation split** — holding out the most recently added annotations from `train_terms.tsv` to simulate the actual CAFA test mechanism — optimizing directly for the IA-weighted F-max defined by `IA.tsv`, per subontology.

---

## 4. Why This Should Work

**It matches the empirically best-performing published approach for this exact problem.** GORetriever, the retrieval-and-rerank system behind GOCurator (first place, CAFA5, 1,600+ teams), substantially outperformed embedding-classifier baselines on held-out CAFA5 proteins — e.g. an F1 of 0.667 versus 0.076–0.357 for methods like LR-ESM and ATGO on a representative BPO test case. This isn't a speculative "let's try LLMs" idea; it's an architecture with a track record on this specific benchmark family.

**It targets the actual failure mode of this dataset.** The core difficulty in CAFA-style tasks is the long tail: thousands of GO terms with few positive training examples. Per-term classifiers have nothing to learn from for those terms. Semantic retrieval sidesteps this because a term's *definition text* carries information regardless of how many times it was observed as a label — a rare term with a clear, specific definition can still be retrieved correctly on the strength of that definition matching the protein's evidence, not on how many training proteins happened to carry that label.

**It decomposes the problem into stages with well-matched, individually strong methods**, rather than asking one model to do everything:
- Sequence similarity handles the "this protein is basically a known one" case cheaply and reliably (the oldest, most robust signal in the field).
- Embedding k-NN handles cases with weaker but real sequence homology, where alignment-based methods lose sensitivity but PLM embeddings retain it.
- Semantic reranking handles cases where the correct term is *conceptually* right but wasn't the literal top-scoring neighbor — recovering precision on specific, rare terms.
- The graph layer enforces a structural constraint (true-path consistency) that no single upstream model is required to learn on its own, and adds a place to inject reasoning for genuinely ambiguous cases.

**It's efficient because each stage narrows the problem for the next one.** Rather than scoring 40,000 terms per protein end-to-end, the pipeline reduces to a shortlist of ~50–200 candidates before the (relatively) expensive semantic scoring step runs — keeping compute proportional to the candidates worth considering, not the full label space.

**It reuses infrastructure already proven to work in this codebase.** The Graph-RAG pattern (Neo4j-backed ontology graph + retrieval-augmented LLM reasoning) is the same architecture validated in DrugSense and threat_graph_new — the main new engineering surface is the candidate generation and reranking stages, not the graph layer itself.

---

## 5. Known Limitations and Mitigations

| Limitation | Mitigation |
|---|---|
| Test proteins may lack any text evidence, undermining semantic reranking | LLM-synthesized description from homolog annotations as fallback; ensemble weighting downweights the reranker score when text confidence is low |
| LLM reranking/description calls add latency and cost at scale | Only run on the shortlisted candidates (~50–200 terms), not the full label space; batch calls; cache by protein where sequences are near-identical |
| Graph-agent conflict resolution could introduce inconsistent judgments across similar cases | Constrain the agent to a narrow decision (promote/demote among already-candidate terms), log rationale, and validate against the temporal holdout set before trusting it in the final pipeline |
| No ground truth exists yet for true test-set proteins (annotations accumulate post-deadline) | All threshold and blend-weight tuning must use the temporal split of *existing* training data, mirroring CAFA's own evaluation timeline, not a random train/test split |

---

## 6. Summary

This architecture replaces a per-term classification approach with a retrieval-and-rerank pipeline over GO term semantics, backed by a knowledge graph for structural consistency. It is motivated by the label-sparsity problem intrinsic to GO annotation, validated by the empirically best-performing published system on this exact benchmark family (GOCurator/GORetriever, CAFA5 winner), and built to reuse Graph-RAG infrastructure already proven in prior work.
