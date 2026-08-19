from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import obonet
import pandas as pd

NAMESPACE_MAP = {
    "molecular_function": "MFO",
    "biological_process": "BPO",
    "cellular_component": "CCO",
}


# ---------------------------------------------------------------------
# GO graph
# ---------------------------------------------------------------------

@dataclass
class GOTerm:
    go_id: str
    name: str
    namespace: str
    definition: str


class GOGraph:
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph

    @classmethod
    def from_obo(cls, obo_path: str) -> "GOGraph":
        g = obonet.read_obo(obo_path)
        return cls(g)

    def get_term(self, go_id: str) -> Optional[GOTerm]:
        node = self.graph.nodes.get(go_id)
        if not node:
            return None

        return GOTerm(
            go_id=go_id,
            name=node.get("name", ""),
            namespace=NAMESPACE_MAP.get(node.get("namespace", ""), node.get("namespace", "")),
            definition=node.get("def", ""),
        )

    def parents(self, go_id: str) -> Dict[str, List[str]]:
        """
        Return this term's direct parents, split by relation type.

        obonet stores the relation type as the EDGE KEY on the
        MultiDiGraph (not as a "relation" field inside the edge data
        dict) -- a term can have multiple is_a parents, so the key is
        what disambiguates parallel edges between the same node pair.
        Unlabeled "is_a: GO:xxxxxxx ! name" lines are keyed "is_a";
        typed relations like "relationship: part_of GO:xxxxxxx ! name"
        are keyed "part_of".
        """
        rels: Dict[str, List[str]] = {"is_a": [], "part_of": []}

        for _, parent, key in self.graph.out_edges(go_id, keys=True):
            if key in rels:
                rels[key].append(parent)

        return rels

    def all_go_ids(self) -> set[str]:
        return set(self.graph.nodes)


# ---------------------------------------------------------------------
# FASTA utilities
# ---------------------------------------------------------------------

def read_fasta(path: str | Path) -> List[Tuple[str, str]]:
    """
    Read a FASTA file.

    Returns:
        List of (header, sequence) tuples.
    """
    records = []
    header = None
    sequence_parts: List[str] = []

    with open(path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(sequence_parts)))

                header = line[1:].strip()
                sequence_parts = []
            else:
                if header is None:
                    raise ValueError(
                        f"Invalid FASTA: sequence encountered before header "
                        f"in {path}"
                    )

                sequence_parts.append(line)

    if header is not None:
        records.append((header, "".join(sequence_parts)))

    return records


# ---------------------------------------------------------------------
# Training FASTA
# ---------------------------------------------------------------------

UNIPROT_HEADER_RE = re.compile(
    r"^"
    r"(?:(?P<db>[^|]+)\|)?"
    r"(?P<accession>[^| ]+)"
    r"(?:\|(?P<entry_name>[^ ]+))?"
    r"\s*"
    r"(?P<description>.*?)"
    r"(?:\s+OS=(?P<organism>.*?)\s+"
    r"OX=(?P<taxon_id>\d+))?"
    r"(?:\s+GN=(?P<gene_name>\S+))?"
    r"(?:\s+PE=(?P<pe>\d+))?"
    r"(?:\s+SV=(?P<sv>\d+))?"
    r"$"
)


def parse_train_header(header: str) -> Dict[str, Optional[str]]:
    """
    Parse a UniProt-style training FASTA header.

    Example:
        sp|A0A0C5B5G6|MOTSC_HUMAN Mitochondrial-derived peptide MOTS-c
        OS=Homo sapiens OX=9606 GN=MT-RNR1 PE=1 SV=1
    """
    match = UNIPROT_HEADER_RE.match(header)

    if not match:
        first_token = header.split(maxsplit=1)[0]

        if "|" in first_token:
            parts = first_token.split("|")
            db = parts[0]
            accession = parts[1]
            entry_name = parts[2] if len(parts) > 2 else None
        else:
            db = None
            accession = first_token
            entry_name = None

        description = header[len(first_token):].strip()

        return {
            "accession": accession,
            "source_db": db,
            "entry_name": entry_name,
            "protein_name": _extract_protein_name(description),
            "gene_name": _extract_field(description, r"GN=(\S+)"),
            "organism": _extract_organism(description),
            "taxon_id": _extract_field(description, r"OX=(\d+)"),
        }

    groups = match.groupdict()

    return {
        "accession": groups["accession"],
        "source_db": groups["db"],
        "entry_name": groups["entry_name"],
        "protein_name": groups["description"].strip(),
        "gene_name": groups["gene_name"],
        "organism": groups["organism"],
        "taxon_id": groups["taxon_id"],
    }


def _extract_field(text: str, pattern: str) -> Optional[str]:
    match = re.search(pattern, text)
    return match.group(1) if match else None


def _extract_organism(text: str) -> Optional[str]:
    match = re.search(
        r"\bOS=(.*?)(?=\s+(?:OX|GN|PE|SV|CC|DR|KW|FT)=|$)",
        text,
    )
    return match.group(1).strip() if match else None


def _extract_protein_name(text: str) -> Optional[str]:
    if " OS=" in text:
        return text.split(" OS=", 1)[0].strip()
    return text.strip() or None


def load_train_sequences(path: str | Path) -> pd.DataFrame:
    """
    Load train_sequences.fasta.

    Columns:
        accession, sequence, source_db, protein_name, gene_name,
        organism, taxon_id, entry_name
    """
    rows = []

    for header, sequence in read_fasta(path):
        parsed = parse_train_header(header)
        rows.append(
            {
                "accession": parsed["accession"],
                "sequence": sequence,
                "source_db": parsed["source_db"],
                "protein_name": parsed["protein_name"],
                "gene_name": parsed["gene_name"],
                "organism": parsed["organism"],
                "taxon_id": parsed["taxon_id"],
                "entry_name": parsed["entry_name"],
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(f"No FASTA records found in {path}")

    if df["accession"].duplicated().any():
        duplicates = df.loc[df["accession"].duplicated(keep=False), "accession"].tolist()
        raise ValueError(f"Duplicate accessions in {path}: {duplicates[:10]}")

    return df


# ---------------------------------------------------------------------
# Test FASTA
# ---------------------------------------------------------------------

def parse_test_header(header: str) -> Dict[str, str]:
    """
    Parse test FASTA headers.

    Expected:
        >A0A0C5B5G6 9606
    """
    parts = header.split()

    if len(parts) < 2:
        raise ValueError(
            f"Invalid test FASTA header. Expected 'accession taxon_id', "
            f"got: {header!r}"
        )

    accession = parts[0]
    taxon_id = parts[1]

    if not taxon_id.isdigit():
        raise ValueError(f"Invalid taxon ID in test FASTA header: {header!r}")

    return {"accession": accession, "taxon_id": taxon_id}


def load_test_sequences(path: str | Path) -> pd.DataFrame:
    """
    Load testsuperset.fasta.

    Columns:
        accession, sequence, taxon_id

    No curated text is present in test headers -- description synthesis
    for test proteins happens downstream (text/synthesis.py), not here.
    """
    rows = []

    for header, sequence in read_fasta(path):
        parsed = parse_test_header(header)
        rows.append(
            {
                "accession": parsed["accession"],
                "sequence": sequence,
                "taxon_id": parsed["taxon_id"],
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(f"No FASTA records found in {path}")

    if df["accession"].duplicated().any():
        duplicates = df.loc[df["accession"].duplicated(keep=False), "accession"].tolist()
        raise ValueError(f"Duplicate accessions in {path}: {duplicates[:10]}")

    return df


# ---------------------------------------------------------------------
# TSV loaders
# ---------------------------------------------------------------------

def load_train_terms(path: str | Path) -> pd.DataFrame:
    """
    Load train_terms.tsv. Expected columns: EntryID, term, aspect.
    """
    df = pd.read_csv(path, sep="\t", dtype=str)

    required = {"EntryID", "term", "aspect"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    return df


def load_train_taxonomy(path: str | Path) -> pd.DataFrame:
    """
    Load train_taxonomy.tsv. Expected columns: EntryID, taxon_id (no header row).
    """
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["EntryID", "taxon_id"],
        dtype=str,
    )
    return df


def load_test_taxon_list(path: str | Path) -> pd.DataFrame:
    """
    Load testsuperset-taxon-list.tsv.

    Expected: tab-separated with a header row (ID, Species). Species
    names may contain spaces and parentheses (e.g. "Escherichia coli
    (strain K12)"), so this must NOT be parsed with a whitespace
    separator -- only tabs delimit fields here.
    """
    df = pd.read_csv(path, sep="\t", dtype=str)

    if set(df.columns) != {"ID", "Species"}:
        raise ValueError(
            f"Unexpected columns in {path}: {df.columns.tolist()} "
            f"(expected ID, Species)"
        )

    return df


def load_ia(path: str | Path) -> pd.DataFrame:
    """
    Load IA.tsv. Columns: term, information_accretion (no header row).
    """
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["term", "information_accretion"],
        dtype={"term": str},
    )

    df["information_accretion"] = pd.to_numeric(
        df["information_accretion"], errors="raise"
    )

    return df


# ---------------------------------------------------------------------
# Join validation
# ---------------------------------------------------------------------

class JoinValidationError(ValueError):
    pass


def validate_train_joins(
    train_sequences: pd.DataFrame,
    train_terms: pd.DataFrame,
    train_taxonomy: pd.DataFrame,
) -> Dict[str, List[str]]:
    """
    Validate all training-side accession joins. Raises
    JoinValidationError on any integrity problem; returns a dict of
    missing references otherwise (empty lists = clean).
    """
    sequence_ids = set(train_sequences["accession"].astype(str))
    term_ids = set(train_terms["EntryID"].astype(str))
    taxonomy_ids = set(train_taxonomy["EntryID"].astype(str))

    missing_terms = sorted(term_ids - sequence_ids)
    missing_taxonomy = sorted(taxonomy_ids - sequence_ids)

    result = {
        "terms_missing_from_sequences": missing_terms,
        "taxonomy_missing_from_sequences": missing_taxonomy,
    }

    problems = []
    if missing_terms:
        problems.append(
            "train_terms.tsv references accessions missing from "
            f"train_sequences.fasta: {missing_terms}"
        )
    if missing_taxonomy:
        problems.append(
            "train_taxonomy.tsv references accessions missing from "
            f"train_sequences.fasta: {missing_taxonomy}"
        )

    if problems:
        raise JoinValidationError("\n".join(problems))

    return result


def validate_test_taxa(
    test_sequences: pd.DataFrame,
    test_taxon_list: pd.DataFrame,
) -> None:
    """
    Ensure every taxon ID in testsuperset.fasta resolves in
    testsuperset-taxon-list.tsv.
    """
    sequence_taxa = set(test_sequences["taxon_id"].astype(str))
    known_taxa = set(test_taxon_list["ID"].astype(str))

    missing = sorted(sequence_taxa - known_taxa)

    if missing:
        raise JoinValidationError(
            "testsuperset.fasta contains taxon IDs absent from "
            f"testsuperset-taxon-list.tsv: {missing}"
        )


def validate_ia_terms(
    ia: pd.DataFrame,
    go_ids: Optional[set[str]] = None,
) -> None:
    """
    Validate IA terms. If go_ids is supplied, every IA term must exist
    in the GO graph.
    """
    if ia["term"].duplicated().any():
        duplicates = ia.loc[ia["term"].duplicated(keep=False), "term"].tolist()
        raise JoinValidationError(f"Duplicate GO terms in IA.tsv: {duplicates[:10]}")

    if go_ids is not None:
        missing = sorted(set(ia["term"]) - go_ids)
        if missing:
            raise JoinValidationError(
                f"IA.tsv contains GO terms absent from the GO graph: {missing}"
            )


# ---------------------------------------------------------------------
# One-shot dataset loader
# ---------------------------------------------------------------------

def load_dataset(
    train_sequences_path: str | Path,
    train_terms_path: str | Path,
    train_taxonomy_path: str | Path,
    test_sequences_path: str | Path,
    test_taxon_list_path: str | Path,
    ia_path: str | Path,
    go_obo_path: Optional[str | Path] = None,
) -> Dict[str, object]:
    """
    Load and validate all supplied files. If go_obo_path is given, also
    loads the GO graph and cross-checks IA.tsv terms against it.
    """
    train_sequences = load_train_sequences(train_sequences_path)
    train_terms = load_train_terms(train_terms_path)
    train_taxonomy = load_train_taxonomy(train_taxonomy_path)

    test_sequences = load_test_sequences(test_sequences_path)
    test_taxon_list = load_test_taxon_list(test_taxon_list_path)

    ia = load_ia(ia_path)

    validate_train_joins(train_sequences, train_terms, train_taxonomy)
    validate_test_taxa(test_sequences, test_taxon_list)

    go_graph = None
    if go_obo_path is not None:
        go_graph = GOGraph.from_obo(str(go_obo_path))
        validate_ia_terms(ia, go_ids=go_graph.all_go_ids())
    else:
        validate_ia_terms(ia)

    return {
        "train_sequences": train_sequences,
        "train_terms": train_terms,
        "train_taxonomy": train_taxonomy,
        "test_sequences": test_sequences,
        "test_taxon_list": test_taxon_list,
        "ia": ia,
        "go_graph": go_graph,
    }