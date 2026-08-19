"""Task 2.4 — co-occurrence conditional prior, P(new term | known terms).

Cheap, interpretable secondary baseline computed directly from train_terms.tsv
cross-tabulation. Not the main method, but a necessary comparison point in the
ablation (Task 2.5) — if this simple prior captures most of the PK gain, that's
an important, honest finding in itself.
"""


def fit_cooccurrence_prior(train_terms) -> dict:
    """-> {known_go_id: {candidate_go_id: conditional_prob}}"""
    raise NotImplementedError


def score(known_terms: set, prior: dict) -> dict:
    raise NotImplementedError
