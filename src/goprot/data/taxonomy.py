"""Task 0.3 — taxonomy tiering.

Three things live here:

1. taxon_tier() -- purely computational, needs nothing but train_taxonomy.tsv
   counts. Drives candidate-generation routing (Task 1.1) and, later, whether
   the PK-conditioning gain holds for sparse species (Task 4.1).

2. classify_kingdom() / parse_ncbi_taxdump() -- the real kingdom classifier.
   train_taxonomy.tsv has 1,381 unique taxon IDs (confirmed) -- a hand-curated
   dict was only ever going to cover a couple dozen well-known model
   organisms, nowhere near enough. This walks the actual NCBI taxonomy tree
   (nodes.dmp, from the public taxdump -- a static reference file, downloaded
   once, not a live API call) up from a taxon to its superkingdom, or to
   Metazoa/Fungi/Viridiplantae if it's a eukaryote, to get a real answer
   instead of a guess.

3. TAXON_TO_KINGDOM -- kept as a small fast-path for the handful of very
   common organisms, so trivial cases don't need the taxdump file loaded.
   Not the source of truth anymore; classify_kingdom() is.

To get nodes.dmp: https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz
(names.dmp is optional -- only needed if you want scientific names attached,
kingdom classification only needs nodes.dmp).
"""

from __future__ import annotations

from typing import Iterable

import networkx as nx
import pandas as pd

ALLOWED_KINGDOMS = {"Animalia", "Fungi", "Plantae", "Protista", "Bacteria", "Archaea"}

SUPERKINGDOM_TAXIDS = {"2": "Bacteria", "2157": "Archaea", "2759": "Eukaryota", "10239": "Viruses"}
EUKARYOTE_CLADE_TAXIDS = {
    "33208": "Animalia",   # Metazoa
    "4751": "Fungi",
    "33090": "Plantae",    # Viridiplantae
}

# Small fast-path for very common organisms -- avoids needing the taxdump
# file loaded just to classify human/mouse/yeast/E. coli. NOT a substitute
# for classify_kingdom() at train_taxonomy.tsv's actual scale (1,381 taxa).
TAXON_TO_KINGDOM: dict[int, str] = {
    9606: "Animalia", 10090: "Animalia", 10116: "Animalia",
    7227: "Animalia", 6239: "Animalia",
    4932: "Fungi",
    3702: "Plantae",
    83333: "Bacteria",
}


def compute_taxon_counts(train_taxonomy: pd.DataFrame) -> dict[int, int]:
    return train_taxonomy["taxon_id"].value_counts().to_dict()


def taxon_tier(taxon_id: int, taxon_counts: dict[int, int], well_annotated_threshold: int = 500) -> str:
    return "well_annotated" if taxon_counts.get(taxon_id, 0) >= well_annotated_threshold else "sparse"


def tier_all_taxa(taxon_counts: dict[int, int], well_annotated_threshold: int = 500) -> dict[int, str]:
    return {taxon_id: taxon_tier(taxon_id, taxon_counts, well_annotated_threshold) for taxon_id in taxon_counts}


def parse_ncbi_taxdump(nodes_path: str, names_path: str | None = None) -> nx.DiGraph:
    """Parse NCBI taxdump nodes.dmp (+ optional names.dmp) into a child->parent
    graph: node attrs {rank, name (if names.dmp given)}.
    """
    graph: nx.DiGraph = nx.DiGraph()

    with open(nodes_path, encoding="utf-8") as f:
        for line in f:
            fields = [x.strip() for x in line.split("|")]
            if len(fields) < 3 or not fields[0]:
                continue
            tax_id, parent_id, rank = fields[0], fields[1], fields[2]
            graph.add_node(tax_id, rank=rank)
            if parent_id and parent_id != tax_id:
                graph.add_edge(tax_id, parent_id)

    if names_path:
        with open(names_path, encoding="utf-8") as f:
            for line in f:
                fields = [x.strip() for x in line.split("|")]
                if len(fields) < 4:
                    continue
                tax_id, name_txt, _unique, name_class = fields[0], fields[1], fields[2], fields[3]
                if name_class == "scientific name" and tax_id in graph:
                    graph.nodes[tax_id]["name"] = name_txt

    return graph

def parse_merged_dmp(path: str) -> dict[str, str]:
    """Parse merged.dmp -> {old_tax_id: new_tax_id}."""
    merged: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            fields = [x.strip() for x in line.split("|")]
            if len(fields) < 2 or not fields[0]:
                continue
            merged[fields[0]] = fields[1]
    return merged


def classify_kingdom(taxon_id, taxdump_graph, merged_map=None, max_depth=50):
    node = str(taxon_id)
    if node not in taxdump_graph:
        if merged_map and node in merged_map:
            node = merged_map[node]
        if node not in taxdump_graph:
            return None
    for _ in range(max_depth):
        if node in EUKARYOTE_CLADE_TAXIDS:
            return EUKARYOTE_CLADE_TAXIDS[node]
        if node in SUPERKINGDOM_TAXIDS:
            kingdom = SUPERKINGDOM_TAXIDS[node]
            return "Protista" if kingdom == "Eukaryota" else kingdom
        parents = list(taxdump_graph.successors(node))
        if not parents:
            return None
        node = parents[0]
    return None


def resolve_kingdoms(
    taxon_ids: Iterable[int],
    taxdump_graph: nx.DiGraph | None = None,
    overrides: dict[int, str] | None = None,
) -> tuple[dict[int, str], list[int]]:
    """Resolution order: overrides > TAXON_TO_KINGDOM fast-path > taxdump
    classification. Without taxdump_graph, only the fast-path organisms
    resolve — everything else is honestly "unresolved", not guessed.
    """
    table = {**TAXON_TO_KINGDOM, **(overrides or {})}
    resolved: dict[int, str] = {}
    unresolved: list[int] = []
    for taxon_id in taxon_ids:
        if taxon_id in table:
            resolved[taxon_id] = table[taxon_id]
        elif taxdump_graph is not None:
            kingdom = classify_kingdom(taxon_id, taxdump_graph)
            if kingdom is not None:
                resolved[taxon_id] = kingdom
            else:
                unresolved.append(taxon_id)
        else:
            unresolved.append(taxon_id)
    return resolved, sorted(unresolved)


from goprot.data.taxonomy import parse_ncbi_taxdump, parse_merged_dmp, resolve_kingdoms

taxdump = parse_ncbi_taxdump("nodes.dmp", "names.dmp")
merged = parse_merged_dmp("merged.dmp")
resolved, unresolved = resolve_kingdoms(real_taxa, taxdump_graph=taxdump, merged_map=merged)