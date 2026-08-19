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
import hashlib
import random
import pandas as pd
from goprot.data.dataset import known_terms_by_protein, ASPECTS

def _stable_seed(*parts: object) -> int:
    """Reproducible per-(seed, accession, aspect) integer seed. Deliberately
    NOT Python's built-in hash() -- string hashing is randomized per-process
    by default, which would silently break the "seed=42 always gives the
    same split" guarantee across separate runs.
    """
    digest = hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(digest, 16) % (2**32)


def _sample_holdout(rng: random.Random, terms: list, n_holdout: int):
    """Return (t0_frozenset, held_out_frozenset) by sampling n_holdout items."""
    held_out = frozenset(rng.sample(terms, n_holdout))
    t0 = frozenset(t for t in terms if t not in held_out)
    return t0, held_out


def make_synthetic_split(train_terms, mask_fraction=0.3, min_known_terms_for_pk=2, seed=42) -> pd.DataFrame:
    if not 0.0 < mask_fraction < 1.0:
        raise ValueError(f"mask_fraction must be in (0, 1), got {mask_fraction}")
    if min_known_terms_for_pk < 2:
        raise ValueError("min_known_terms_for_pk must be >= 2 ...")

    known = known_terms_by_protein(train_terms)
    rows = []
    for accession, by_aspect in known.items():
        for aspect in ASPECTS:
            terms = sorted(by_aspect[aspect])
            n = len(terms)
            if n < min_known_terms_for_pk:
                continue
            rng = random.Random(_stable_seed(seed, accession, aspect))
            n_holdout = max(1, round(n * mask_fraction))
            n_holdout = min(n_holdout, n - 1)
            t0, held_out = _sample_holdout(rng, terms, n_holdout)

            rows.append({"accession": accession, "aspect": aspect,
                         "known_terms": t0, "held_out_terms": held_out, "setting": "PK"})
            rows.append({"accession": accession, "aspect": aspect,
                         "known_terms": frozenset(), "held_out_terms": frozenset(terms), "setting": "NK"})
    return pd.DataFrame(rows)