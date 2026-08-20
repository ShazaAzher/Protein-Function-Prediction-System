from __future__ import annotations

"""Task 3.3b — trains both comparison arms on the synthetic split.

Deliberately does NOT blend in homology scores -- that would confound the
comparison this script exists to make (does conditioning on T0 help?) with
a different question (does adding homology help?). Homology blending
belongs in the final submission pipeline, not this ablation comparison.
"""

import sys
from pathlib import Path
import yaml

from goprot.condition.known_terms import encode_known_terms
from goprot.data.dataset import known_terms_by_protein
from goprot.data.parsing import ASPECT_TO_LONG, parse_ia_tsv, parse_terms_tsv
from goprot.eval.metrics import f_max
from goprot.eval.pk_split import make_synthetic_split
from goprot.go_graph.term_embeddings import fit_term_embeddings
from goprot.model.classifier import ColdStartClassifier, PKConditionedClassifier
from goprot.model.encoder import load_cached_embeddings


def main(config_path: str = "configs/pipeline.yaml", aspect: str = "F") -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    raw_train_dir = Path(config["paths"]["raw_train_dir"])
    raw_test_dir = Path(config["paths"]["raw_test_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])

    train_terms = parse_terms_tsv(raw_train_dir / "train_terms.tsv")
    ia_weights = parse_ia_tsv(raw_train_dir / "IA.tsv").set_index("go_id")["ia_weight"].to_dict()
    train_embeddings = load_cached_embeddings(str(processed_dir / "train_embeddings.parquet"))

    top_k_key = f"top_k_{ASPECT_TO_LONG[aspect].lower()}"
    top_k = config["label_space"][top_k_key]
    aspect_terms = train_terms[train_terms["aspect"] == aspect]
    term_freq = aspect_terms.groupby("go_id")["accession"].nunique().sort_values(ascending=False)
    label_space = list(term_freq.head(top_k).index)

    split_config = config["validation_split"]
    split = make_synthetic_split(
        train_terms, mask_fraction=split_config["mask_fraction"],
        min_known_terms_for_pk=split_config["min_known_terms_for_pk"], seed=split_config["seed"],
    )
    pk_rows = split[(split["aspect"] == aspect) & (split["setting"] == "PK")]
    if pk_rows.empty:
        raise ValueError(f"No PK-eligible instances for aspect {aspect!r} -- check min_known_terms_for_pk.")

    term_dim = config["pk_conditioning"]["term_embedding_dim"]
    term_embeddings = fit_term_embeddings(train_terms, dim=term_dim)
    pool_method = config["pk_conditioning"]["known_term_pooling"]

    pk_embeddings, pk_known_vectors, pk_labels = {}, {}, {}
    for _, row in pk_rows.iterrows():
        accession = row["accession"]
        if accession not in train_embeddings:
            continue
        pk_embeddings[accession] = train_embeddings[accession]
        pk_known_vectors[accession] = encode_known_terms(row["known_terms"], term_embeddings, method=pool_method, dim=term_dim)
        pk_labels[accession] = row["held_out_terms"]

    if not pk_embeddings:
        raise ValueError(f"None of the PK-eligible accessions for aspect {aspect!r} have a cached embedding.")

    known = known_terms_by_protein(train_terms)
    cold_start_embeddings = {a: e for a, e in train_embeddings.items() if a in known}
    cold_start_labels = {a: known[a].get(aspect, set()) for a in cold_start_embeddings}

    cold_start_clf = ColdStartClassifier(label_space=label_space, hidden_dim=256)
    cold_start_clf.fit(cold_start_embeddings, cold_start_labels)

    pk_clf = PKConditionedClassifier(label_space=label_space, hidden_dim=256)
    pk_clf.fit(pk_embeddings, pk_known_vectors, pk_labels)

    cold_start_predictions = cold_start_clf.predict(pk_embeddings)
    pk_predictions = pk_clf.predict(pk_embeddings, pk_known_vectors)

    ground_truth = {a: pk_labels[a] for a in pk_embeddings}
    known_terms_map = {row["accession"]: row["known_terms"] for _, row in pk_rows.iterrows() if row["accession"] in pk_embeddings}

    cold_start_fmax = f_max(cold_start_predictions, ground_truth, ia_weights, known_terms=known_terms_map)
    pk_fmax = f_max(pk_predictions, ground_truth, ia_weights, known_terms=known_terms_map)

    return {"aspect": aspect, "n_pk_instances": len(pk_embeddings), "cold_start_fmax": cold_start_fmax, "pk_fmax": pk_fmax}


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/pipeline.yaml", sys.argv[2] if len(sys.argv) > 2 else "F")