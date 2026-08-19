"""Task 0.6 — in-memory GO DAG ancestor matrix + true-path propagation.

Deliberately NOT Neo4j — see project notes. A sparse ancestor matrix built once
with networkx from the OBO graph (Task 0.1) is a few milliseconds per pass and
has no infra dependency, which matters for Kaggle/Colab parity.

- build_ancestor_matrix(go_graph) -> scipy.sparse matrix, term_index -> term_index
- propagate(scores: dict[go_id, float], ancestor_matrix, term_index) -> dict[go_id, float]
  score[ancestor] = max(score[ancestor], score[descendant]) for every scored term.
"""


def build_ancestor_matrix(go_graph):
    raise NotImplementedError


def propagate(scores: dict, ancestor_matrix, term_index: dict) -> dict:
    raise NotImplementedError
