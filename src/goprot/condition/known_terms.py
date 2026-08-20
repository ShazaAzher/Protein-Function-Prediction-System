"""Task 2.2 — known-term-set encoder for the PK-conditioned classifier.

Wraps go_graph/term_embeddings.py: given a protein's T0 for an aspect,
produce the conditioning vector that gets concatenated to the ESM2 embedding
before the classifier head (model/classifier.py).
"""
import numpy as np 
from goprot.go_graph.term_embeddings import pool_known_terms

def encode_known_terms(known_terms: set[str], term_embeddings: dict[str, np.ndarray], method: str = "mean", dim: int | None = None) -> np.ndarray:
    return pool_known_terms(known_terms, term_embeddings, method=method, dim=dim)