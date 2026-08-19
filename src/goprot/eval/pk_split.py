"""Task 0.5 — synthetic NK/PK validation split.

No timestamp column in train_terms.tsv, so real temporal accretion isn't
directly constructible. This mirrors the CAFA5 PK protocol locally instead:
for training proteins with >= min_known_terms_for_pk terms in an aspect,
randomly hold out mask_fraction of them as "newly added" (the held-out target,
T1 \ T0) and keep the rest as "already known" (T0, the conditioning input).
Proteins with zero known terms in an aspect form the NK comparison group.

This is a proxy for real temporal accretion, not equivalent to it —
say so plainly in the writeup, don't oversell it.
"""


def make_synthetic_split(train_terms, mask_fraction: float = 0.3, min_known_terms_for_pk: int = 2, seed: int = 42):
    """-> DataFrame[protein_id, aspect, known_terms(T0), held_out_terms(T1\T0), setting("NK"|"PK")]"""
    raise NotImplementedError
