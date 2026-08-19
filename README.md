# protein-fn-pred

CAFA6 protein function prediction (GO term prediction from sequence), built
around a specific research question: does explicitly conditioning a
classifier on a protein's already-known same-aspect GO annotations improve
prediction of newly-added terms in that aspect — the **Partial-Knowledge
(PK)** setting introduced in the CAFA5 organizers' analysis paper — beyond
what a cold-start model achieves on identical data?

See `TASKS.md` for the full build plan and task breakdown, and the project
memory notes for the surrounding architecture decisions (why no Neo4j, why
classifier + homology ensemble instead of retrieval-and-rerank, why PK over
the other candidate research angles).

## Setup

```bash
pip install -e ".[dev]"
```

Place competition files (`train_sequences.fasta`, `train_terms.tsv`,
`train_taxonomy.tsv`, `go-basic.obo`, `testsuperset.fasta`,
`testsuperset-taxon-list.tsv`, `IA.tsv`, `sample_submission.tsv`) in
`data/raw/` — not committed, see `.gitignore`.

## Status

Scaffold only — every module in `src/goprot/` is a stub with a docstring
pointing at its task ID in `TASKS.md`. Nothing is implemented yet.
