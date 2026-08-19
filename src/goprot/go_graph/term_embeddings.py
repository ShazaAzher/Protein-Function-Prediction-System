"""Task 2.1 — GO term embeddings for known-term-set conditioning.

Used to encode a protein's T0 (already-known terms) into a fixed-size dense
vector for the PK-conditioned classifier. Start simple: co-occurrence-based
embedding (terms that appear together across train_terms.tsv end up close),
node2vec/DeepWalk over the DAG structure as a stretch goal if the simple
version underperforms.

- fit_term_embeddings(train_terms, go_graph, dim) -> dict[go_id, np.ndarray]
- pool_known_terms(term_ids: set[str], term_embeddings, method="mean") -> np.ndarray
"""


def fit_term_embeddings(train_terms, go_graph, dim: int = 64):
    raise NotImplementedError


def pool_known_terms(term_ids: set, term_embeddings: dict, method: str = "mean"):
    raise NotImplementedError
