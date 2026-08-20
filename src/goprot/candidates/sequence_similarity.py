"""Task 1.3 — Diamond/MMseqs2 homology-transfer baseline.

Doubles as: (a) the standalone homology baseline for the NK/LK/PK ablation table,
and (b) an ensemble partner for the final blended pipeline (Task 3.1).
Kingdom-tier routing: search same-kingdom training subset first, fall back to
the full training set only if same-kingdom hits are weak (sparse-species case).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from goprot.data.dataset import known_terms_by_protein

_OUTPUT_FIELDS = ["qseqid", "sseqid", "pident", "length", "evalue", "bitscore"]


def _require_diamond() -> None:
    if shutil.which("diamond") is None:
        raise RuntimeError(
            "diamond binary not found on PATH. Install it from "
            "https://github.com/bbuchfink/diamond/releases (diamond-linux64.tar.gz, "
            "extract and put on PATH) before calling build_diamond_db/search."
        )


def _extract_accession(sseqid: str) -> str:
    """diamond's sseqid is the training FASTA header up to first whitespace
    -- e.g. 'sp|A0A0C5B5G6|MOTSC_HUMAN'. Extract the 2nd pipe field (the
    accession), matching parse_train_fasta's own extraction, so hits join
    cleanly against train_terms.tsv. Falls back to the raw string if it
    doesn't look pipe-delimited (a custom/non-UniProt-format training FASTA).
    """
    parts = sseqid.split("|")
    return parts[1] if len(parts) >= 2 else sseqid


def build_diamond_db(train_fasta_path: str, db_path: str) -> None:
    """`diamond makedb --in train_fasta_path -d db_path`."""
    _require_diamond()
    subprocess.run(
        ["diamond", "makedb", "--in", str(train_fasta_path), "-d", str(db_path)],
        check=True, capture_output=True, text=True,
    )


def search(query_fasta_path: str, db_path: str, sensitivity: str = "--very-sensitive") -> pd.DataFrame:
    """`diamond blastp` against db_path, tabular output.

    -> DataFrame[query_accession, train_accession, pident, length, evalue, bitscore]

    query_accession is diamond's qseqid as-is: testsuperset.fasta headers
    are already bare accessions ('>accession taxon_id', per Task 0.2), and
    diamond's qseqid is the header up to first whitespace, so no extra
    parsing is needed on the query side -- only sseqid (the training side)
    needs _extract_accession.
    """
    _require_diamond()
    fd, output_path = tempfile.mkstemp(suffix=".tsv")
    os.close(fd)
    try:
        subprocess.run(
            [
                "diamond", "blastp",
                "-q", str(query_fasta_path), "-d", str(db_path),
                "-o", output_path, "-f", "6", *_OUTPUT_FIELDS,
                sensitivity,
            ],
            check=True, capture_output=True, text=True,
        )
        hits = pd.read_csv(output_path, sep="\t", header=None, names=_OUTPUT_FIELDS)
    finally:
        os.remove(output_path)

    hits = hits.rename(columns={"qseqid": "query_accession", "sseqid": "train_accession_raw"})
    hits["train_accession"] = hits["train_accession_raw"].apply(_extract_accession)
    return hits[["query_accession", "train_accession", "pident", "length", "evalue", "bitscore"]]


def transfer_scores(
    hits: pd.DataFrame,
    train_terms: pd.DataFrame,
    aspect: str,
    top_k: int = 10,
    score_col: str = "bitscore",
) -> dict[str, dict[str, float]]:
    """Per query protein: its top_k homolog hits (by score_col, descending),
    unioning the GO terms those homologs carry IN THE GIVEN ASPECT ONLY --
    matching naive_baseline's per-aspect signature, so this is called once
    per aspect like everything else in eval/, ready for direct per-aspect
    F-max scoring.

    Each candidate term is scored by the MAX normalized homolog score among
    homologs carrying it (normalized against the query's own best-hit
    score, so the top homolog's terms always score 1.0). -> {query_accession:
    {go_id: score}}, scores in (0, 1].
    """
    known = known_terms_by_protein(train_terms)  # {accession: {aspect: set(go_id)}}

    scores: dict[str, dict[str, float]] = {}
    for query_accession, group in hits.groupby("query_accession"):
        top_hits = group.sort_values(score_col, ascending=False).head(top_k)
        if top_hits.empty or top_hits[score_col].max() <= 0:
            continue
        max_score = top_hits[score_col].max()

        term_scores: dict[str, float] = {}
        for _, hit in top_hits.iterrows():
            homolog_terms = known.get(hit["train_accession"], {}).get(aspect, set())
            normalized = hit[score_col] / max_score
            for term in homolog_terms:
                if normalized > term_scores.get(term, 0.0):
                    term_scores[term] = normalized

        if term_scores:
            scores[query_accession] = term_scores

    return scores