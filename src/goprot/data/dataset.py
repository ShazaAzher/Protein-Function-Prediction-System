"""Task 0.4 — NK / LK / PK flagging and per-aspect known-term extraction.

This is now core plumbing, not an edge case: every downstream model needs to know,
per protein per aspect, which GO terms are already known (T0) before deciding whether
to run the cold-start path or the PK-conditioned path.

- flag_knowledge_setting(protein_id, terms_by_aspect) -> {"BPO": "NK"|"LK"|"PK", ...}
  Mirrors the CAFA5 definitions exactly:
    NK: no experimental annotation in any aspect
    LK: annotation in >=1 aspect, but not in this one
    PK: existing annotation in this aspect (predicting *additional* terms in it)
- known_terms_for_aspect(protein_id, aspect, terms_by_aspect) -> set[go_id]
  The T0 set fed into the PK-conditioning module (condition/known_terms.py).
"""
import pandas as pd
from goprot.data.parsing import NAMESPACE_TO_ASPECT

ASPECTS = tuple(NAMESPACE_TO_ASPECT.values())  # ("P", "F", "C")


def known_terms_by_protein(train_terms: pd.DataFrame) -> dict[str, dict[str, set[str]]]:
    """-> {accession: {"P": {go_id, ...}, "F": {...}, "C": {...}}}"""
    result: dict[str, dict[str, set[str]]] = {}
    for accession, go_id, aspect in zip(train_terms["accession"], train_terms["go_id"], train_terms["aspect"]):
        result.setdefault(accession, {a: set() for a in ASPECTS})
        result[accession][aspect].add(go_id)
    return result


def known_terms_for_aspect(protein_id, aspect, known_terms_map) -> set[str]:
    return known_terms_map.get(protein_id, {}).get(aspect, set())


def flag_knowledge_setting(known_terms_by_aspect: dict[str, set[str]]) -> dict[str, str]:
    """NK: zero terms in all three aspects -> every aspect is NK.
    Otherwise, per aspect: PK if this aspect already has >=1 known term,
    LK if this aspect is empty but the protein has terms elsewhere.
    """
    has_any_terms = any(known_terms_by_aspect.get(a) for a in ASPECTS)
    if not has_any_terms:
        return {a: "NK" for a in ASPECTS}
    return {a: ("PK" if known_terms_by_aspect.get(a) else "LK") for a in ASPECTS}


def flag_all_training_proteins(train_terms: pd.DataFrame) -> pd.DataFrame:
    """-> DataFrame[accession, aspect, setting, known_terms]. Every row here
    is "LK" or "PK", never "NK" — train_terms.tsv only lists proteins with
    >=1 annotation already. This IS the candidate pool Task 0.5's synthetic
    PK split draws from: any "PK" row has a known-term set that can be
    partially masked to simulate T0/T1.
    """
    known = known_terms_by_protein(train_terms)
    rows = []
    for accession, by_aspect in known.items():
        settings = flag_knowledge_setting(by_aspect)
        for aspect in ASPECTS:
            rows.append({
                "accession": accession, "aspect": aspect,
                "setting": settings[aspect], "known_terms": frozenset(by_aspect[aspect]),
            })
    return pd.DataFrame(rows)


def sequence_hash_overlap(test_df: pd.DataFrame, train_df: pd.DataFrame) -> dict[str, str]:
    """-> {test_accession: train_accession} for exact sequence matches —
    catches both the same-accession case (A0A0C5B5G6, confirmed real in
    your samples) and accession renames the ID alone would miss.
    """
    train_by_seq = {}
    for accession, sequence in zip(train_df["accession"], train_df["sequence"]):
        train_by_seq.setdefault(sequence, accession)
    matches = {}
    for accession, sequence in zip(test_df["accession"], test_df["sequence"]):
        if sequence in train_by_seq:
            matches[accession] = train_by_seq[sequence]
    return matches


def flag_test_proteins(test_accessions, known_terms_map, accession_remap=None) -> pd.DataFrame:
    """The real seen-vs-novel join at test time. accession_remap (from
    sequence_hash_overlap) redirects a renamed test accession to its
    training-side known terms. No resolvable history -> genuinely novel -> NK.
    """
    accession_remap = accession_remap or {}
    empty_known = {a: set() for a in ASPECTS}
    rows = []
    for test_accession in test_accessions:
        lookup_accession = accession_remap.get(test_accession, test_accession)
        by_aspect = known_terms_map.get(lookup_accession, empty_known)
        settings = flag_knowledge_setting(by_aspect)
        for aspect in ASPECTS:
            rows.append({
                "accession": test_accession, "aspect": aspect,
                "setting": settings[aspect], "known_terms": frozenset(by_aspect.get(aspect, set())),
            })
    return pd.DataFrame(rows)