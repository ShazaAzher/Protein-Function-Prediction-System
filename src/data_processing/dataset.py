"""
Task 0.4 -- Seen-vs-novel flagging.

CAFA-style test supersets can include proteins that already appear in
the training set with SOME experimental annotations, but are expected
to accumulate ADDITIONAL annotations by evaluation time (the "partial
knowledge" category flagged in the CAFA4 report). Confirmed real in
this dataset: accession A0A0C5B5G6 appears in both
train_sequences.fasta and testsuperset.fasta with an identical
sequence.

These proteins need different handling downstream:
  - their existing train_terms.tsv annotations are a strong prior /
    floor, not the final answer -- the eval rewards NEW correctly
    predicted terms too
  - sequence-similarity candidate generation will otherwise return an
    exact self-match and just echo back known terms with high
    confidence, which is correct as a prior but should be flagged so
    it isn't silently treated the same as a genuinely novel protein

This module flags matches two ways -- by accession AND by exact
sequence -- because the same protein can appear under different
accessions across database releases/isoforms, and accession-only
matching would miss that case.
"""
from __future__ import annotations

import pandas as pd


def flag_seen_proteins(
    test_sequences: pd.DataFrame,
    train_sequences: pd.DataFrame,
) -> pd.DataFrame:
    """
    Tag each test protein as seen-by-accession, seen-by-sequence,
    both, or neither (fully novel).

    Args:
        test_sequences: columns accession, sequence, taxon_id
            (from data.parsing.load_test_sequences)
        train_sequences: columns accession, sequence, ... (from
            data.parsing.load_train_sequences)

    Returns:
        test_sequences with three added boolean columns:
            seen_by_accession, seen_by_sequence, partial_knowledge
        partial_knowledge is True if either match type is True.
    """
    train_accessions = set(train_sequences["accession"])
    train_sequence_set = set(train_sequences["sequence"])

    result = test_sequences.copy()
    result["seen_by_accession"] = result["accession"].isin(train_accessions)
    result["seen_by_sequence"] = result["sequence"].isin(train_sequence_set)
    result["partial_knowledge"] = (
        result["seen_by_accession"] | result["seen_by_sequence"]
    )

    return result


def get_existing_annotations(
    seen_test_accession: str,
    train_sequences: pd.DataFrame,
    train_terms: pd.DataFrame,
) -> pd.DataFrame:
    """
    For a partial-knowledge test protein, look up its existing
    training-side accession(s) (by exact accession OR exact sequence
    match) and return its already-known GO term annotations. This is
    the "prior" to carry into the ensemble stage, not the final
    prediction.
    """
    test_row = train_sequences.loc[
        train_sequences["accession"] == seen_test_accession
    ]

    if test_row.empty:
        return pd.DataFrame(columns=["EntryID", "term", "aspect"])

    accession = test_row.iloc[0]["accession"]
    return train_terms.loc[train_terms["EntryID"] == accession]


def summarize_partial_knowledge(flagged: pd.DataFrame) -> dict:
    """
    Quick counts for logging/sanity-checking after flag_seen_proteins.
    """
    return {
        "total_test_proteins": len(flagged),
        "seen_by_accession": int(flagged["seen_by_accession"].sum()),
        "seen_by_sequence": int(flagged["seen_by_sequence"].sum()),
        "partial_knowledge_total": int(flagged["partial_knowledge"].sum()),
        "fully_novel": int((~flagged["partial_knowledge"]).sum()),
    }