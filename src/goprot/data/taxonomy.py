"""Task 0.3 — taxonomy tiering.

- taxon_to_kingdom: dict[taxon_id -> {Animalia, Fungi, Plantae, Bacteria, Protista}]
  Hand-curated for the ~32-90 species observed in testsuperset-taxon-list.tsv.
- taxon_tier(taxon_id, train_taxonomy) -> "well_annotated" | "sparse"
  Based on frequency in train_taxonomy.tsv. Drives candidate-generation routing
  and, downstream, how much weight the PK-conditioning signal gets for sparse species.
"""

TAXON_TO_KINGDOM: dict[int, str] = {}


def taxon_tier(taxon_id: int, train_taxonomy_counts: dict[int, int]) -> str:
    raise NotImplementedError
