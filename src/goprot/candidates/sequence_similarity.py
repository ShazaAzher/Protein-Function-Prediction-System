"""Task 1.3 — Diamond/MMseqs2 homology-transfer baseline.

Doubles as: (a) the standalone homology baseline for the NK/LK/PK ablation table,
and (b) an ensemble partner for the final blended pipeline (Task 3.1).
Kingdom-tier routing: search same-kingdom training subset first, fall back to
the full training set only if same-kingdom hits are weak (sparse-species case).
"""


def build_diamond_db(train_fasta_path: str, db_path: str) -> None:
    raise NotImplementedError


def search(query_fasta_path: str, db_path: str):
    raise NotImplementedError


def transfer_scores(hits, train_terms) -> dict:
    """hits -> {protein_id: {go_id: score}} weighted by identity/bitscore."""
    raise NotImplementedError
