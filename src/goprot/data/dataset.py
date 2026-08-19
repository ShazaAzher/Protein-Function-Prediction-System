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


def flag_knowledge_setting(protein_id: str, terms_by_aspect: dict) -> dict:
    raise NotImplementedError


def known_terms_for_aspect(protein_id: str, aspect: str, terms_by_aspect: dict) -> set:
    raise NotImplementedError
