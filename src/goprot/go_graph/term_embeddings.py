"""Task 2.1 — GO term embeddings for known-term-set conditioning.

Used to encode a protein's T0 (already-known terms) into a fixed-size dense
vector for the PK-conditioned classifier. Start simple: co-occurrence-based
embedding (terms that appear together across train_terms.tsv end up close),
node2vec/DeepWalk over the DAG structure as a stretch goal if the simple
version underperforms.

- fit_term_embeddings(train_terms, go_graph, dim) -> dict[go_id, np.ndarray]
- pool_known_terms(term_ids: set[str], term_embeddings, method="mean") -> np.ndarray
"""
import pandas as pd
import numpy as np 
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

def fit_term_embeddings(train_terms: pd.DataFrame, dim: int = 64, min_count: int = 1) -> dict[str, np.ndarray]:
    groups = train_terms.groupby(["accession", "aspect"])["go_id"].apply(set)
    vocab = sorted(train_terms["go_id"].unique())
    term_to_idx = {term: i for i, term in enumerate(vocab)}
    n = len(vocab)

    rows, cols = [], []
    for term_set in groups:
        if len(term_set) < 2:
            continue
        indices = [term_to_idx[t] for t in term_set]
        for i in indices:
            for j in indices:
                if i != j:
                    rows.append(i); cols.append(j)

    cooc = sp.csr_matrix(([1.0] * len(rows), (rows, cols)), shape=(n, n)) if rows else sp.csr_matrix((n, n))
    counts = np.asarray(cooc.sum(axis=1)).flatten()
    keep = np.where(counts >= min_count)[0]
    if len(keep) == 0:
        return {}
    if len(keep) < 2:
        return {vocab[keep[0]]: np.zeros(dim)}

    cooc_kept = cooc[keep][:, keep].astype(np.float64)
    effective_dim = min(dim, len(keep) - 1)
    reduced = TruncatedSVD(n_components=effective_dim, random_state=42).fit_transform(cooc_kept)

    embeddings = {}
    for local_i, global_i in enumerate(keep):
        vec = reduced[local_i]
        if effective_dim < dim:
            vec = np.pad(vec, (0, dim - effective_dim))
        embeddings[vocab[global_i]] = vec
    return embeddings


def pool_known_terms(term_ids: set[str], term_embeddings: dict[str, np.ndarray], method: str = "mean", dim: int | None = None) -> np.ndarray:
    vectors = [term_embeddings[t] for t in term_ids if t in term_embeddings]
    if not vectors:
        if dim is None:
            if not term_embeddings:
                raise ValueError("dim must be given explicitly when term_embeddings is empty")
            dim = next(iter(term_embeddings.values())).shape[0]
        return np.zeros(dim)
    stacked = np.stack(vectors)
    if method == "mean":
        return stacked.mean(axis=0)
    if method == "max":
        return stacked.max(axis=0)
    raise ValueError(f"unknown pooling method: {method!r} (expected 'mean' or 'max')")