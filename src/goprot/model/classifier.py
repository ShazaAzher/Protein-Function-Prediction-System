"""Task 1.4 (cold-start) and Task 2.3 (PK-conditioned) — the paper's two
comparison arms. Same backbone, same top-K label space per ontology
(configs/pipeline.yaml); the ONLY difference is whether the known-term
conditioning vector (condition/known_terms.py) is concatenated to the
ESM2 embedding before the classifier head. That's the whole experiment.
"""

from __future__ import annotations

import numpy as np
import torch


class _MultiLabelMLP:
    """Shared fit/predict machinery. Not meant to be instantiated directly
    -- ColdStartClassifier and PKConditionedClassifier only differ in how
    they build the per-protein input feature vector.
    """

    def __init__(self, label_space: list[str], hidden_dim: int = 256, lr: float = 1e-3, device: str | None = None):
        self.label_space = list(label_space)
        self.term_to_idx = {term: i for i, term in enumerate(self.label_space)}
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: torch.nn.Module | None = None  # built lazily once input dim is known

    def _build_model(self, input_dim: int) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, len(self.label_space)),
        )  # raw logits -- BCEWithLogitsLoss during fit, sigmoid applied only at predict time

    def _labels_to_tensor(self, accessions: list[str], labels: dict[str, set[str]]) -> torch.Tensor:
        Y = torch.zeros(len(accessions), len(self.label_space))
        for i, accession in enumerate(accessions):
            for term in labels.get(accession, set()):
                idx = self.term_to_idx.get(term)
                if idx is not None:  # terms outside label_space silently ignored
                    Y[i, idx] = 1.0
        return Y

    def _fit_on_features(self, X: torch.Tensor, Y: torch.Tensor, epochs: int, batch_size: int, seed: int) -> None:
        torch.manual_seed(seed)
        self._model = self._build_model(X.shape[1]).to(self.device)
        X, Y = X.to(self.device), Y.to(self.device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        self._model.train()
        n = X.shape[0]
        for _epoch in range(epochs):
            perm = torch.randperm(n)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                optimizer.zero_grad()
                loss = loss_fn(self._model(X[idx]), Y[idx])
                loss.backward()
                optimizer.step()

    def _predict_on_features(self, accessions: list[str], X: torch.Tensor) -> dict[str, dict[str, float]]:
        if self._model is None:
            raise RuntimeError("call fit() before predict()")
        X = X.to(self.device)
        self._model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(self._model(X)).cpu().numpy()
        return {
            accession: {term: float(probs[i, j]) for j, term in enumerate(self.label_space)}
            for i, accession in enumerate(accessions)
        }


class ColdStartClassifier(_MultiLabelMLP):
    """Embedding -> sigmoid scores over a fixed top-K label space. Never
    sees known-term conditioning -- the "no conditioning" arm of the
    Phase 2 ablation (Task 2.5).
    """

    def fit(
        self,
        embeddings: dict[str, np.ndarray],
        labels: dict[str, set[str]],
        epochs: int = 20,
        batch_size: int = 32,
        seed: int = 42,
    ) -> "ColdStartClassifier":
        accessions = sorted(embeddings)
        if not accessions:
            raise ValueError("fit() called with no embeddings")
        X = torch.tensor(np.stack([embeddings[a] for a in accessions]), dtype=torch.float32)
        Y = self._labels_to_tensor(accessions, labels)
        self._fit_on_features(X, Y, epochs, batch_size, seed)
        return self

    def predict(self, embeddings: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
        accessions = sorted(embeddings)
        X = torch.tensor(np.stack([embeddings[a] for a in accessions]), dtype=torch.float32)
        return self._predict_on_features(accessions, X)


class PKConditionedClassifier(_MultiLabelMLP):
    """[Embedding ; known-term conditioning vector] -> sigmoid scores over
    top-K terms. Identical backbone to ColdStartClassifier -- the only
    difference is the conditioning vector concatenated to the input before
    it hits the same Linear -> ReLU -> Linear head. Deliberate: isolating
    the effect of conditioning means not also changing the architecture.
    """

    @staticmethod
    def _require_all_present(accessions: list[str], known_term_vectors: dict[str, np.ndarray]) -> None:
        missing = [a for a in accessions if a not in known_term_vectors]
        if missing:
            raise ValueError(
                f"known_term_vectors missing entries for {len(missing)} accession(s), e.g. {missing[:5]}"
            )

    def fit(
        self,
        embeddings: dict[str, np.ndarray],
        known_term_vectors: dict[str, np.ndarray],
        labels: dict[str, set[str]],
        epochs: int = 20,
        batch_size: int = 32,
        seed: int = 42,
    ) -> "PKConditionedClassifier":
        accessions = sorted(embeddings)
        if not accessions:
            raise ValueError("fit() called with no embeddings")
        self._require_all_present(accessions, known_term_vectors)

        X = torch.tensor(
            np.stack([np.concatenate([embeddings[a], known_term_vectors[a]]) for a in accessions]),
            dtype=torch.float32,
        )
        Y = self._labels_to_tensor(accessions, labels)
        self._fit_on_features(X, Y, epochs, batch_size, seed)
        return self

    def predict(
        self, embeddings: dict[str, np.ndarray], known_term_vectors: dict[str, np.ndarray]
    ) -> dict[str, dict[str, float]]:
        accessions = sorted(embeddings)
        self._require_all_present(accessions, known_term_vectors)
        X = torch.tensor(
            np.stack([np.concatenate([embeddings[a], known_term_vectors[a]]) for a in accessions]),
            dtype=torch.float32,
        )
        return self._predict_on_features(accessions, X)