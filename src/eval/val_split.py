"""
Task 0.5 -- Validation split strategy.

CAVEAT (read before using): train_terms.tsv as provided has no
timestamp/evidence-date column, so a true temporal holdout -- mimicking
CAFA's actual annotation-accumulation window between submission
deadline and evaluation -- is NOT constructible from this file alone.

This module implements the documented fallback: a stratified random
holdout at the PROTEIN level (not the term level, since CAFA scores
per-protein predictions), stratified by how many GO terms each protein
has, with an explicit protection rule so rare terms are never entirely
removed from the training side.

If a versioned UniProt/GOA history becomes available later, prefer
reconstructing real annotation dates and replacing this with an actual
temporal split -- this fallback is knowingly optimistic (validation
performance will likely overstate real test-set performance, since a
random split doesn't reproduce the difficulty of predicting genuinely
NEW annotations that didn't exist yet at training time).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_MIN_PROTEINS_TO_HOLD_OUT = 5


def _term_frequency(train_terms: pd.DataFrame) -> pd.Series:
    """Number of distinct proteins annotated with each term."""
    return train_terms.groupby("term")["EntryID"].nunique()


def identify_protected_proteins(
    train_terms: pd.DataFrame,
    min_proteins_per_term: int = DEFAULT_MIN_PROTEINS_TO_HOLD_OUT,
) -> set[str]:
    """
    Identify proteins that must stay in the training split because
    they carry a term that is already too rare to risk losing from
    training entirely.

    A protein is "protected" if it is annotated with ANY term that has
    fewer than `min_proteins_per_term` total annotated proteins in the
    full training set. Holding out such a protein could zero out that
    term's training signal completely.
    """
    freq = _term_frequency(train_terms)
    rare_terms = set(freq[freq < min_proteins_per_term].index)

    protected = set(
        train_terms.loc[train_terms["term"].isin(rare_terms), "EntryID"]
    )
    return protected


def _protein_term_count_bucket(count: int) -> str:
    """Coarse buckets for stratification -- exact term counts would
    create too many singleton strata for sklearn's stratified split."""
    if count <= 1:
        return "1"
    if count <= 3:
        return "2-3"
    if count <= 10:
        return "4-10"
    return "11+"


def build_validation_split(
    train_sequences: pd.DataFrame,
    train_terms: pd.DataFrame,
    holdout_frac: float = 0.2,
    min_proteins_per_term: int = DEFAULT_MIN_PROTEINS_TO_HOLD_OUT,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Build a stratified random protein-level train/val split.

    Returns:
        DataFrame with columns: accession, split ("train" or "val")
        covering every accession in train_sequences.
    """
    all_accessions = train_sequences["accession"].tolist()

    protected = identify_protected_proteins(train_terms, min_proteins_per_term)
    holdout_eligible = [a for a in all_accessions if a not in protected]

    if len(holdout_eligible) < 2:
        # Not enough eligible proteins to split at all (e.g. tiny
        # fixture/demo data) -- everything stays in train, val is empty.
        return pd.DataFrame(
            {"accession": all_accessions, "split": ["train"] * len(all_accessions)}
        )

    terms_per_protein = train_terms.groupby("EntryID")["term"].nunique()
    strata = [
        _protein_term_count_bucket(int(terms_per_protein.get(a, 0)))
        for a in holdout_eligible
    ]

    # StratifiedShuffleSplit / train_test_split need >=2 members per
    # stratum; fall back to an unstratified split if the eligible pool
    # is too small/uneven for that (common in fixtures, rare in the
    # real dataset with thousands of proteins).
    strata_counts = pd.Series(strata).value_counts()
    can_stratify = (strata_counts >= 2).all() and len(holdout_eligible) >= 5

    try:
        train_ids, val_ids = train_test_split(
            holdout_eligible,
            test_size=holdout_frac,
            random_state=random_state,
            stratify=strata if can_stratify else None,
        )
    except ValueError:
        # Final fallback: plain random split.
        train_ids, val_ids = train_test_split(
            holdout_eligible,
            test_size=holdout_frac,
            random_state=random_state,
        )

    split_map = {a: "train" for a in protected}
    split_map.update({a: "train" for a in train_ids})
    split_map.update({a: "val" for a in val_ids})

    return pd.DataFrame(
        {
            "accession": all_accessions,
            "split": [split_map[a] for a in all_accessions],
        }
    )


def validate_split_coverage(
    split_df: pd.DataFrame,
    train_terms: pd.DataFrame,
) -> dict:
    """
    Sanity check: confirm no term's training-side support dropped to
    zero after the split. Returns a summary dict; raises if the
    protection rule somehow failed.
    """
    train_accessions = set(split_df.loc[split_df["split"] == "train", "accession"])

    terms_in_train_split = set(
        train_terms.loc[train_terms["EntryID"].isin(train_accessions), "term"]
    )
    all_terms = set(train_terms["term"])

    zeroed_out = sorted(all_terms - terms_in_train_split)

    if zeroed_out:
        raise ValueError(
            f"{len(zeroed_out)} term(s) have zero training-side support "
            f"after the split -- protection rule failed: {zeroed_out}"
        )

    n_train = int((split_df["split"] == "train").sum())
    n_val = int((split_df["split"] == "val").sum())

    return {
        "n_train": n_train,
        "n_val": n_val,
        "n_terms_total": len(all_terms),
        "n_terms_covered_in_train_split": len(terms_in_train_split),
    }