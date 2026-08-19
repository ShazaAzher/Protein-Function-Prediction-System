"""Task 1.4 (cold-start) and Task 2.3 (PK-conditioned) — the paper's two
comparison arms. Same backbone, same top-K label space per ontology
(configs/pipeline.yaml); the ONLY difference is whether the known-term
conditioning vector (condition/known_terms.py) is concatenated to the
ESM2 embedding before the classifier head. That's the whole experiment.
"""


class ColdStartClassifier:
    """Embedding -> sigmoid scores over top-K terms. No knowledge of T0."""

    def fit(self, embeddings, labels):
        raise NotImplementedError

    def predict(self, embeddings) -> dict:
        raise NotImplementedError


class PKConditionedClassifier:
    """[Embedding ; known-term conditioning vector] -> sigmoid scores over top-K terms."""

    def fit(self, embeddings, known_term_vectors, labels):
        raise NotImplementedError

    def predict(self, embeddings, known_term_vectors) -> dict:
        raise NotImplementedError
