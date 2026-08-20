from goprot.data.parsing import parse_obo, parse_train_fasta, parse_terms_tsv, parse_taxonomy_tsv, validate_join_integrity
from goprot.data.taxonomy import parse_ncbi_taxdump, resolve_kingdoms, parse_merged_dmp
from goprot.data.dataset import flag_all_training_proteins
from goprot.eval.pk_split import make_synthetic_split
from goprot.go_graph.ancestors import build_ancestor_matrix, propagate
import pandas as pd 

go_graph = parse_obo("data/raw/Train/go-basic.obo")
sequences = parse_train_fasta("data/raw/Train/train_sequences.fasta")
terms = parse_terms_tsv("data/raw/Train/train_terms.tsv")
taxonomy = parse_taxonomy_tsv("data/raw/Train/train_taxonomy.tsv")
validate_join_integrity(terms, sequences)  # raises if anything's mismatched

taxdump = parse_ncbi_taxdump("data/raw/taxdump/nodes.dmp", "data/raw/taxdump/names.dmp")
merged = parse_merged_dmp("data/raw/taxdump/merged.dmp")
resolved, unresolved = resolve_kingdoms(taxonomy["taxon_id"].unique(), taxdump_graph=taxdump, merged_map=merged)
print(f"kingdoms resolved: {len(resolved)}, unresolved: {len(unresolved)}")
print(unresolved)

flagged = flag_all_training_proteins(terms)
print(flagged["setting"].value_counts())  # how much PK vs LK exists in real data
pk_rows = flagged[flagged["setting"] == "PK"].copy()
pk_rows["n_known"] = pk_rows["known_terms"].apply(len)

bins = [1, 2, 3, 6, 11, 21, 51, 101, float("inf")]
labels = ["1", "2", "3-5", "6-10", "11-20", "21-50", "51-100", "101+"]
pk_rows["bucket"] = pd.cut(pk_rows["n_known"], bins=bins, labels=labels, right=False)

counts = pk_rows["bucket"].value_counts().reindex(labels).fillna(0).astype(int)
print(counts)
print(f"\nTotal PK aspects: {len(pk_rows)}")


split = make_synthetic_split(terms)
print(f"synthetic PK/NK validation rows: {len(split)}")

ancestor_index = build_ancestor_matrix(go_graph)
print(propagate({terms.iloc[0]["go_id"]: 0.9}, ancestor_index))

