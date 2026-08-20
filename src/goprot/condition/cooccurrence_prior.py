"""Task 2.4 — co-occurrence conditional prior, P(new term | known terms).

Cheap, interpretable secondary baseline computed directly from train_terms.tsv
cross-tabulation. Not the main method, but a necessary comparison point in the
ablation (Task 2.5) — if this simple prior captures most of the PK gain, that's
an important, honest finding in itself.
"""


"""Task 2.4 — co-occurrence conditional prior, P(candidate | known term).

Cheap, interpretable secondary baseline computed directly from
train_terms.tsv cross-tabulation. Not the main method (PKConditionedClassifier,
Task 2.3, is) -- a necessary comparison point in the Task 2.5 ablation. If
this simple prior alone captures most of the PK gain, that's an important,
honest finding to report, not something to bury under a fancier model's
headline number.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd


def fit_cooccurrence_prior(train_terms: pd.DataFrame) -> dict[str, dict[str, float]]:
    """-> {known_go_id: {candidate_go_id: P(candidate | known)}}

    P(b | a) = (number of (accession, aspect) groups containing BOTH a and
    b) / (number of groups containing a). Computed pairwise per (a, b) --
    NOT a full joint P(candidate | entire known set), which would need a
    separate count for every observed COMBINATION of known terms and isn't
    remotely tractable at this data's scale. score() below combines the
    pairwise priors for a specific known-term set via noisy-OR.
    """
    groups = train_terms.groupby(["accession", "aspect"])["go_id"].apply(set)

    term_group_count: dict[str, int] = defaultdict(int)
    pair_count: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for term_set in groups:
        for a in term_set:
            term_group_count[a] += 1
        for a in term_set:
            for b in term_set:
                if a != b:
                    pair_count[a][b] += 1

    prior: dict[str, dict[str, float]] = {}
    for a, counts in pair_count.items():
        denom = term_group_count[a]
        prior[a] = {b: c / denom for b, c in counts.items()}
    return prior


def score(known_terms: set[str], prior: dict[str, dict[str, float]]) -> dict[str, float]:
    """Combine per-known-term conditional priors into one score per
    candidate via noisy-OR:

        P(candidate | T0) = 1 - prod over a in T0 of (1 - P(candidate | a))

    Treats each known term as independent evidence for a candidate --
    standard way to combine several pairwise conditional-probability
    signals into one estimate without needing a fitted joint distribution
    over the full known-term set (intractable at this data's scale — see
    fit_cooccurrence_prior's docstring).

    Known terms with no recorded co-occurrence data contribute nothing --
    silently skipped. -> {candidate_go_id: score}, scores in [0, 1].
    """
    complements: dict[str, float] = {}
    for known_term in known_terms:
        for candidate, p in prior.get(known_term, {}).items():
            complements[candidate] = complements.get(candidate, 1.0) * (1.0 - p)
    return {candidate: 1.0 - complement for candidate, complement in complements.items()}