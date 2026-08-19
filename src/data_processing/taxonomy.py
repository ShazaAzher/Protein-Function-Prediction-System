"""
Task 0.3 -- Taxonomy tiering.

Maps each taxon ID observed in the dataset to:
  - kingdom      : fixed biological fact, hand-curated (small, fixed
                   species list -- not worth an NCBI taxonomy API call)
  - tier         : computed from how many training proteins exist for
                   that taxon in train_taxonomy.tsv (dense model
                   organism vs. sparse/single-genome-project species)

Both feed candidate generation (same-kingdom-first, same-tier-aware
homolog/embedding search) and the taxon-plausibility filter in the
reranker stage.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

# ---------------------------------------------------------------------
# Kingdom mapping
# ---------------------------------------------------------------------
# Hand-curated for the species observed in testsuperset-taxon-list.tsv.
# Extend this dict if new taxa appear in future data drops -- there is
# no fallback lookup, so an unmapped taxon_id should fail loudly rather
# than silently defaulting to "Unknown" (see get_kingdom below).

TAXON_KINGDOM: Dict[str, str] = {
    "9606": "Animalia",     # Homo sapiens
    "10116": "Animalia",    # Rattus norvegicus
    "39947": "Plantae",     # Oryza sativa subsp. japonica
    "7955": "Animalia",     # Danio rerio
    "7227": "Animalia",     # Drosophila melanogaster
    "6239": "Animalia",     # Caenorhabditis elegans
    "3702": "Plantae",      # Arabidopsis thaliana
    "10090": "Animalia",    # Mus musculus
    "9913": "Animalia",     # Bos taurus
    "44689": "Protista",    # Dictyostelium discoideum
    "224308": "Bacteria",   # Bacillus subtilis (strain 168)
    "559292": "Fungi",      # Saccharomyces cerevisiae (strain ATCC 204508 / S288c)
    "83333": "Bacteria",    # Escherichia coli (strain K12)
    "208964": "Bacteria",   # Pseudomonas aeruginosa (strain ATCC 15692 / ...)
    "4839": "Fungi",        # Rhizomucor miehei
    "1344414": "Fungi",     # Hypocrea jecorina (strain ATCC 56765 / ...)
    "4102": "Plantae",      # Petunia hybrida
    "3880": "Plantae",      # Medicago truncatula
    "27334": "Fungi",       # Penicillium expansum
    "83332": "Bacteria",    # Mycobacterium tuberculosis (strain ATCC 25618 / H37Rv)
    "8705": "Animalia",     # Vipera ammodytes ammodytes
    "745531": "Fungi",      # Phlebiopsis gigantea (strain 11061_1 CR5-6)
    "5270": "Fungi",        # Mycosarcoma maydis
    "571544": "Animalia",   # Sicarius terrosus
    "3750": "Plantae",      # Malus domestica
    "6279": "Animalia",     # Brugia malayi
    "30457": "Animalia",    # Pygoscelis papua
    "3871": "Plantae",      # Lupinus angustifolius
    "69780": "Fungi",       # Penicillium ochrochloron
    "36329": "Protista",    # Plasmodium falciparum (isolate 3D7)
    "242507": "Fungi",      # Pyricularia oryzae (strain 70-15 / ...)
    "4577": "Plantae",      # Zea mays
}


class UnknownTaxonError(KeyError):
    pass


def get_kingdom(taxon_id: str | int) -> str:
    """
    Look up the kingdom for a taxon ID. Raises UnknownTaxonError rather
    than silently returning a placeholder, so new/unmapped taxa are
    caught immediately instead of quietly degrading downstream
    kingdom-plausibility filtering.
    """
    key = str(taxon_id)
    if key not in TAXON_KINGDOM:
        raise UnknownTaxonError(
            f"No kingdom mapping for taxon_id={key!r}. "
            f"Add it to TAXON_KINGDOM in taxonomy.py."
        )
    return TAXON_KINGDOM[key]


def build_kingdom_table(taxon_ids: list[str] | pd.Series) -> pd.DataFrame:
    """
    Build a taxon_id -> kingdom lookup table for a given set of taxon
    IDs, e.g. taken from testsuperset-taxon-list.tsv.
    """
    unique_ids = sorted(set(str(t) for t in taxon_ids))
    return pd.DataFrame(
        {
            "taxon_id": unique_ids,
            "kingdom": [get_kingdom(t) for t in unique_ids],
        }
    )


# ---------------------------------------------------------------------
# Tier computation
# ---------------------------------------------------------------------
# Tier is computed, not hand-curated: it reflects how much training
# signal actually exists for a taxon in THIS dataset, not a general
# fact about the species. A species could be well-studied in general
# but sparsely represented in this particular train_taxonomy.tsv.

def compute_taxon_tiers(
    train_taxonomy: pd.DataFrame,
    dense_threshold: int = 50,
) -> pd.DataFrame:
    """
    Compute a tier (1 = dense/well-represented, 2 = sparse) per taxon
    ID, based on protein counts in train_taxonomy.tsv.

    Args:
        train_taxonomy: DataFrame with columns EntryID, taxon_id
            (as loaded by data.parsing.load_train_taxonomy).
        dense_threshold: minimum protein count for tier 1. Defaults to
            50 -- tune once the real train_taxonomy.tsv is available;
            model organisms in CAFA-style datasets typically have
            thousands of proteins, while single-genome-project species
            often have a few dozen or fewer.

    Returns:
        DataFrame with columns: taxon_id, train_protein_count, tier
    """
    counts = (
        train_taxonomy.groupby("taxon_id")["EntryID"]
        .nunique()
        .rename("train_protein_count")
        .reset_index()
    )

    counts["tier"] = counts["train_protein_count"].apply(
        lambda n: 1 if n >= dense_threshold else 2
    )

    return counts.sort_values("train_protein_count", ascending=False).reset_index(
        drop=True
    )


def build_taxonomy_reference(
    test_taxon_list: pd.DataFrame,
    train_taxonomy: pd.DataFrame,
    dense_threshold: int = 50,
) -> pd.DataFrame:
    """
    Build the full per-taxon reference table used by candidate
    generation and the reranker's taxon-plausibility filter: every
    taxon in the test superset, joined with kingdom and tier.

    Taxa with zero training proteins get tier=2 and
    train_protein_count=0 (rather than being dropped), since they
    still need to be routed through candidate generation -- they will
    just rely more heavily on cross-kingdom/embedding-based search
    than on same-taxon homology.
    """
    kingdoms = build_kingdom_table(test_taxon_list["ID"])
    tiers = compute_taxon_tiers(train_taxonomy, dense_threshold=dense_threshold)

    ref = kingdoms.merge(tiers, on="taxon_id", how="left")
    ref["train_protein_count"] = ref["train_protein_count"].fillna(0).astype(int)
    ref["tier"] = ref["tier"].fillna(2).astype(int)

    return ref.merge(
        test_taxon_list.rename(columns={"ID": "taxon_id", "Species": "species"}),
        on="taxon_id",
        how="left",
    )