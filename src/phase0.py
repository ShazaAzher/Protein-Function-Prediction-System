from goprot.data.parsing import parse_obo, parse_train_fasta, parse_terms_tsv, parse_taxonomy_tsv, validate_join_integrity
from goprot.data.taxonomy import parse_ncbi_taxdump, resolve_kingdoms
from goprot.data.dataset import flag_all_training_proteins
from goprot.eval.pk_split import make_synthetic_split
from goprot.go_graph.ancestors import build_ancestor_matrix, propagate

go_graph = parse_obo("data/raw/Train/go-basic.obo")
sequences = parse_train_fasta("data/raw/Train/train_sequences.fasta")
terms = parse_terms_tsv("data/raw/Train/train_terms.tsv")
taxonomy = parse_taxonomy_tsv("data/raw/Train/train_taxonomy.tsv")
validate_join_integrity(terms, sequences)  # raises if anything's mismatched

taxdump = parse_ncbi_taxdump("data/raw/nodes.dmp", "data/raw/names.dmp")
resolved, unresolved = resolve_kingdoms(taxonomy["taxon_id"].unique(), taxdump_graph=taxdump)
print(f"kingdoms resolved: {len(resolved)}, unresolved: {len(unresolved)}")

flagged = flag_all_training_proteins(terms)
print(flagged["setting"].value_counts())  # how much PK vs LK exists in real data

split = make_synthetic_split(terms)
print(f"synthetic PK/NK validation rows: {len(split)}")

ancestor_index = build_ancestor_matrix(go_graph)
print(propagate({terms.iloc[0]["go_id"]: 0.9}, ancestor_index))