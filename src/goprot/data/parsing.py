"""Task 0.1 / 0.2 — OBO, FASTA, and TSV parsers.

parse_obo is fully implemented below (custom line-based parser, no obonet
dependency — full control over def-text cleaning and namespace/aspect
mapping, and one less thing that can silently disagree with the rest of
the pipeline about edge direction).

The FASTA/TSV parsers (0.2) are stubbed next — see their docstrings for
the exact schema each must produce; validate join integrity (every
accession in train_terms.tsv resolves in train_sequences.fasta) once
those land.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterator
import pandas as pd
import networkx as nx

# GO obo `namespace:` values -> single-letter aspect codes, matching the
# `aspect` column in train_terms.tsv (P/F/C) -- NOT the long BPO/MFO/CCO
# codes used elsewhere in the competition docs. Both are provided here so
# downstream code never has to re-derive the mapping.
NAMESPACE_TO_ASPECT = {
    "biological_process": "P",
    "molecular_function": "F",
    "cellular_component": "C",
}
ASPECT_TO_LONG = {"P": "BPO", "F": "MFO", "C": "CCO"}

SUBONTOLOGY_ROOTS = {
    "BPO": "GO:0008150",
    "CCO": "GO:0005575",
    "MFO": "GO:0003674",
}

_DEF_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')
_UP_TO_BANG_RE = re.compile(r"^([^!]+)")


def _unescape(text: str) -> str:
    return text.replace('\\"', '"').replace("\\\\", "\\")


def _parse_term_stanza(lines: list[str]) -> dict | None:
    """Parse one [Term] stanza's raw lines into a flat attrs dict.

    Returns None if there's no id line (malformed stanza -- skip it rather
    than crash on one bad stanza in a 40k-term file).
    """
    term_id = None
    name = None
    namespace = None
    definition = ""
    is_obsolete = False
    is_a: list[str] = []
    relationship: list[tuple[str, str]] = []

    for line in lines:
        if ":" not in line:
            continue
        key, _, rest = line.partition(":")
        key = key.strip()
        rest = rest.strip()

        if key == "id":
            term_id = rest
        elif key == "name":
            name = rest
        elif key == "namespace":
            namespace = rest
        elif key == "def":
            m = _DEF_RE.search(rest)
            definition = _unescape(m.group(1)) if m else ""
        elif key == "is_obsolete":
            is_obsolete = rest.lower() == "true"
        elif key == "is_a":
            m = _UP_TO_BANG_RE.match(rest)
            if m:
                is_a.append(m.group(1).strip())
        elif key == "relationship":
            m = _UP_TO_BANG_RE.match(rest)
            if m:
                parts = m.group(1).strip().split(None, 1)
                if len(parts) == 2:
                    relationship.append((parts[0], parts[1]))

    if term_id is None:
        return None

    return {
        "id": term_id,
        "name": name,
        "namespace": namespace,
        "aspect": NAMESPACE_TO_ASPECT.get(namespace),
        "def": definition,
        "is_obsolete": is_obsolete,
        "is_a": is_a,
        "relationship": relationship,
    }


def parse_obo(path: str | Path) -> nx.DiGraph:
    """Parse go-basic.obo into a directed graph.

    Nodes: GO term id -> attrs {name, namespace, aspect (P/F/C), def, is_obsolete}.
    Edges: child -> parent (the direction the file literally declares them in),
    with attr {relation: "is_a" | "part_of" | "regulates" | "positively_regulates"
    | "negatively_regulates" | ...}.

    graph.successors(term_id)            -> immediate parents (one hop)
    networkx.descendants(graph, term_id) -> ALL ancestors, transitively
        (this is what true-path propagation in go_graph/ancestors.py needs --
        note the networkx-vs-biology naming clash: "descendants" of a node in
        this child->parent digraph are its biological *ancestors*.)

    Obsolete terms are parsed but dropped from the returned graph entirely --
    they're not valid prediction targets and shouldn't silently show up as
    dangling nodes with no data via an edge from a non-obsolete child.
    """
    path = Path(path)
    graph: nx.DiGraph = nx.DiGraph()
    header: dict[str, str] = {}

    seen_any_stanza = False
    in_term_stanza = False
    stanza_lines: list[str] = []
    obsolete_ids: set[str] = set()

    def flush_stanza() -> None:
        nonlocal stanza_lines
        if stanza_lines:
            attrs = _parse_term_stanza(stanza_lines)
            if attrs is not None:
                if attrs["is_obsolete"]:
                    obsolete_ids.add(attrs["id"])
                else:
                    graph.add_node(
                        attrs["id"],
                        name=attrs["name"],
                        namespace=attrs["namespace"],
                        aspect=attrs["aspect"],
                        def_=attrs["def"],
                    )
                    for parent in attrs["is_a"]:
                        graph.add_edge(attrs["id"], parent, relation="is_a")
                    for rel_type, parent in attrs["relationship"]:
                        graph.add_edge(attrs["id"], parent, relation=rel_type)
        stanza_lines = []

    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if line.startswith("[") and line.endswith("]"):
                flush_stanza()
                seen_any_stanza = True
                in_term_stanza = line == "[Term]"
                continue
            if in_term_stanza:
                if line.strip():
                    stanza_lines.append(line)
            elif not seen_any_stanza and ":" in line:
                key, _, value = line.partition(":")
                header[key.strip()] = value.strip()
        flush_stanza()

    # Drop any edges that point at an obsolete or missing parent (shouldn't
    # happen in a well-formed go-basic.obo, but don't let one bad reference
    # silently break propagation).
    for node in list(graph.nodes):
        for _, parent in list(graph.out_edges(node)):
            if parent in obsolete_ids or parent not in graph:
                graph.remove_edge(node, parent)

    graph.graph["header"] = header
    graph.graph["obsolete_ids"] = obsolete_ids
    return graph

def _iter_fasta_records(path: str | Path) -> Iterator[tuple[str, str]]:
    """Yield (header_without_gt, sequence) for each record. Shared by both FASTA parsers."""
    header = None
    seq_chunks: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_chunks)
                header = line[1:]
                seq_chunks = []
            else:
                seq_chunks.append(line.strip())
        if header is not None:
            yield header, "".join(seq_chunks)


# UniProt-style header: db|accession|EntryName ProteinName OS=Organism OX=taxid
# [GN=GeneName] [PE=n] [SV=n]. GN/PE/SV are optional; protein_name and organism
# can both contain spaces, so OS=/OX= are the reliable anchors.
_TRAIN_HEADER_RE = re.compile(
    r"^(?P<db>[^|]+)\|(?P<accession>[^|]+)\|(?P<entry_name>\S+)\s+"
    r"(?P<protein_name>.*?)\s+OS=(?P<organism>.*?)\s+OX=(?P<taxon_id>\d+)"
    r"(?:\s+GN=(?P<gene_name>.*?))?"   # was \S+ — broke on multi-word gene names
    r"(?:\s+PE=\d+)?"
    r"(?:\s+SV=\d+)?\s*$"
)


def parse_train_fasta(path: str | Path) -> pd.DataFrame:
    """-> DataFrame[accession, sequence, source_db, entry_name, protein_name,
    gene_name, organism, taxon_id]. gene_name is None when a record has no
    GN= field. Raises ValueError on any header that doesn't match the
    expected UniProt format, naming the offending header, rather than
    silently emitting a row of Nones.
    """
    rows = []
    for header, sequence in _iter_fasta_records(path):
        m = _TRAIN_HEADER_RE.match(header)
        if not m:
            raise ValueError(f"train_sequences.fasta header didn't match expected format: {header!r}")
        rows.append({
            "accession": m.group("accession"),
            "sequence": sequence,
            "source_db": m.group("db"),
            "entry_name": m.group("entry_name"),
            "protein_name": m.group("protein_name"),
            "gene_name": m.group("gene_name"),
            "organism": m.group("organism"),
            "taxon_id": int(m.group("taxon_id")),
        })
    return pd.DataFrame(rows)


def parse_test_fasta(path: str | Path) -> pd.DataFrame:
    """-> DataFrame[accession, sequence, taxon_id]. Header: '>accession taxon_id'."""
    rows = []
    for header, sequence in _iter_fasta_records(path):
        parts = header.split()
        if len(parts) != 2:
            raise ValueError(f"testsuperset.fasta header didn't match 'accession taxon_id': {header!r}")
        accession, taxon_id = parts
        rows.append({"accession": accession, "sequence": sequence, "taxon_id": int(taxon_id)})
    return pd.DataFrame(rows)


def _has_header(path: str | Path, n_bytes: int = 4096) -> bool:
    """Sniff whether a TSV's first line is a header. Your samples disagree on
    this across files — train_terms.tsv shows a header row, train_taxonomy.tsv
    and IA.tsv don't — so this is sniffed per-file rather than assumed.
    """
    with open(path, encoding="utf-8") as f:
        sample = f.read(n_bytes)
    try:
        return csv.Sniffer().has_header(sample)
    except csv.Error:
        return True


def _read_positional_tsv(path: str | Path, colnames: list[str]) -> pd.DataFrame:
    """Assign canonical column names *positionally* regardless of whether a
    header is present or what it's called — don't depend on exact header
    text matching across files.
    """
    has_header = _has_header(path)
    df = pd.read_csv(path, sep="\t", header=0 if has_header else None)
    if len(df.columns) != len(colnames):
        raise ValueError(f"{path}: expected {len(colnames)} columns {colnames}, found {len(df.columns)}: {list(df.columns)}")
    df.columns = colnames
    return df


def parse_terms_tsv(path: str | Path) -> pd.DataFrame:
    """-> DataFrame[accession, go_id, aspect]. aspect must be one of P/F/C
    (matches NAMESPACE_TO_ASPECT from parse_obo) — raises if not.
    """
    df = _read_positional_tsv(path, ["accession", "go_id", "aspect"])
    unexpected = set(df["aspect"].unique()) - set(NAMESPACE_TO_ASPECT.values())
    if unexpected:
        raise ValueError(f"train_terms.tsv has aspect codes not in {set(NAMESPACE_TO_ASPECT.values())}: {unexpected}")
    return df


def parse_taxonomy_tsv(path: str | Path) -> pd.DataFrame:
    df = _read_positional_tsv(path, ["accession", "taxon_id"])
    df["taxon_id"] = df["taxon_id"].astype(int)
    return df


def parse_ia_tsv(path: str | Path) -> pd.DataFrame:
    df = _read_positional_tsv(path, ["go_id", "ia_weight"])
    df["ia_weight"] = df["ia_weight"].astype(float)
    return df


def validate_join_integrity(terms_df: pd.DataFrame, sequences_df: pd.DataFrame) -> None:
    """Raise ValueError listing any train_terms.tsv accessions with no
    matching train_sequences.fasta row — catches a mismatched file pair or
    parsing bug early, before it silently drops labeled examples downstream.
    """
    missing = set(terms_df["accession"]) - set(sequences_df["accession"])
    if missing:
        sample = sorted(missing)[:10]
        raise ValueError(f"{len(missing)} accession(s) in train_terms.tsv have no matching sequence, e.g. {sample}")