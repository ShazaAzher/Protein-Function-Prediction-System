"""Task 2.2 — known-term-set encoder for the PK-conditioned classifier.

Wraps go_graph/term_embeddings.py: given a protein's T0 for an aspect,
produce the conditioning vector that gets concatenated to the ESM2 embedding
before the classifier head (model/classifier.py).
"""


def encode_known_terms(protein_id: str, aspect: str, known_terms: set, term_embeddings) -> "np.ndarray":
    raise NotImplementedError
