"""Task 1.1 — IA-weighted precision/recall/F-max, PK-aware.

Per the CAFA5 derivation, conditional information content for PK collapses to
the ordinary IA weights summed over the newly-added terms only (T1 \ T0) — see
project notes / Appendix A of the CAFA5 paper. So this does NOT need a novel
metric, just correct filtering: for PK-setting proteins, exclude T0 from both
the predicted set and the ground-truth set before scoring.
"""


def weighted_precision_recall(predictions: dict, ground_truth: dict, ia_weights: dict, known_terms: dict | None = None):
    """known_terms, if given, is {protein_id: set[go_id]} to exclude (PK filtering)."""
    raise NotImplementedError


def f_max(predictions: dict, ground_truth: dict, ia_weights: dict, known_terms: dict | None = None) -> float:
    raise NotImplementedError
