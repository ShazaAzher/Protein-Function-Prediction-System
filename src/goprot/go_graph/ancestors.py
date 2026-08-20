"""Task 0.6 — in-memory GO DAG ancestor matrix + true-path propagation.

Deliberately NOT Neo4j — see project notes. A sparse ancestor matrix built once
with networkx from the OBO graph (Task 0.1) is a few milliseconds per pass and
has no infra dependency, which matters for Kaggle/Colab parity.

- build_ancestor_matrix(go_graph) -> scipy.sparse matrix, term_index -> term_index
- propagate(scores: dict[go_id, float], ancestor_matrix, term_index) -> dict[go_id, float]
  score[ancestor] = max(score[ancestor], score[descendant]) for every scored term.
"""
from __future__ import annotations   # stdlib — lets type hints like `str | None` work on Python 3.10 syntax
from dataclasses import dataclass    # stdlib — for the AncestorIndex bundle
import networkx as nx                # third-party — DAG traversal (nx.descendants), same lib parse_obo uses
import scipy.sparse as sp            # third-party — the ancestor matrix itself (CSR sparse boolean matrix)

@dataclass(frozen=True)
class AncestorIndex:
    matrix: sp.csr_matrix          # matrix[i, j] == True iff term j is a strict ancestor of term i
    term_index: dict[str, int]     # go_id -> row/col index
    index_to_term: list[str]       # index -> go_id


def build_ancestor_matrix(go_graph, relations: set[str] | None = None) -> AncestorIndex:
    if relations is not None:
        filtered = nx.DiGraph()
        filtered.add_nodes_from(go_graph.nodes)
        for u, v, data in go_graph.edges(data=True):
            if data.get("relation") in relations:
                filtered.add_edge(u, v)
        graph = filtered
    else:
        graph = go_graph  # default: ALL edge types, matching the CAFA5 paper's propagation rule

    terms = sorted(graph.nodes)
    term_index = {term: i for i, term in enumerate(terms)}
    rows, cols = [], []
    for term in terms:
        i = term_index[term]
        for ancestor in nx.descendants(graph, term):
            rows.append(i)
            cols.append(term_index[ancestor])
    matrix = sp.csr_matrix(([True] * len(rows), (rows, cols)), shape=(len(terms),) * 2, dtype=bool)
    return AncestorIndex(matrix=matrix, term_index=term_index, index_to_term=terms)


def propagate(scores: dict[str, float], ancestor_index: AncestorIndex) -> dict[str, float]:
    result = dict(scores)
    for term, score in scores.items():
        i = ancestor_index.term_index.get(term)
        if i is None:
            continue
        row = ancestor_index.matrix.getrow(i)
        for j in row.indices:
            ancestor_term = ancestor_index.index_to_term[j]
            if score > result.get(ancestor_term, float("-inf")):
                result[ancestor_term] = score
    return result