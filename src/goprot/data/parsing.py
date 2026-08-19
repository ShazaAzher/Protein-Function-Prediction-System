"""Task 0.1 / 0.2 — OBO, FASTA, and TSV parsers.

- parse_obo(path) -> networkx.DiGraph over GO terms (id, name, namespace, def, is_a/part_of edges)
- parse_train_fasta(path) -> DataFrame[accession, sequence, source_db, protein_name, gene_name, organism]
- parse_test_fasta(path) -> DataFrame[accession, sequence, taxon_id]   (no curated text — confirmed)
- parse_terms_tsv(path) -> DataFrame[accession, go_id, aspect]
- parse_taxonomy_tsv(path) -> DataFrame[accession, taxon_id]
- parse_ia_tsv(path) -> DataFrame[go_id, ia_weight]

Validate join integrity: every accession in train_terms.tsv must resolve in train_sequences.fasta.
"""


def parse_obo(path: str):
    raise NotImplementedError


def parse_train_fasta(path: str):
    raise NotImplementedError


def parse_test_fasta(path: str):
    raise NotImplementedError


def parse_terms_tsv(path: str):
    raise NotImplementedError


def parse_taxonomy_tsv(path: str):
    raise NotImplementedError


def parse_ia_tsv(path: str):
    raise NotImplementedError
