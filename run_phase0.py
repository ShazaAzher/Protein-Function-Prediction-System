"""
Run all of Phase 0 end to end:
    0.1  OBO parsing              (data/parsing.py -> GOGraph)
    0.2  FASTA + TSV parsing      (data/parsing.py -> load_dataset)
    0.3  Taxonomy tiering         (data/taxonomy.py)
    0.4  Seen-vs-novel flagging   (data/dataset.py)
    0.5  Validation split         (eval/val_split.py)

Writes all processed artifacts to data/processed/ and prints a summary
so the whole phase can be sanity-checked in one run before anything
downstream (candidate generation, reranking, etc.) depends on it.

Usage:
    python scripts/run_phase0.py \
        --raw-dir data/raw \
        --processed-dir data/processed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.data_processing.parsing import load_dataset  # noqa: E402
from src.data_processing.taxonomy import build_taxonomy_reference  # noqa: E402
from src.data_processing.dataset import (  # noqa: E402
    flag_seen_proteins,
    summarize_partial_knowledge,
)
from src.eval.val_split import (  # noqa: E402
    build_validation_split,
    validate_split_coverage,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--go-obo", type=Path, default=Path("data/raw/Train/go-basic.obo"))
    parser.add_argument("--dense-threshold", type=int, default=50)
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--min-proteins-per-term", type=int, default=5)
    args = parser.parse_args()

    raw = args.raw_dir
    out = args.processed_dir
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PHASE 0.1 / 0.2 -- Parsing + join validation")
    print("=" * 60)
    dataset = load_dataset(
        train_sequences_path=raw / "Train" / "train_sequences.fasta",
        train_terms_path=raw / "Train" / "train_terms.tsv",
        train_taxonomy_path=raw / "Train" / "train_taxonomy.tsv",
        test_sequences_path=raw / "Test" / "testsuperset.fasta",
        test_taxon_list_path=raw / "Test" / "testsuperset-taxon-list.tsv",
        ia_path=raw / "IA.tsv",
        go_obo_path=args.go_obo,
    )

    train_sequences = dataset["train_sequences"]
    train_terms = dataset["train_terms"]
    train_taxonomy = dataset["train_taxonomy"]
    test_sequences = dataset["test_sequences"]
    test_taxon_list = dataset["test_taxon_list"]
    ia = dataset["ia"]
    go_graph = dataset["go_graph"]

    train_sequences.to_parquet(out / "train_sequences.parquet", index=False)
    train_terms.to_parquet(out / "train_terms.parquet", index=False)
    train_taxonomy.to_parquet(out / "train_taxonomy.parquet", index=False)
    test_sequences.to_parquet(out / "test_sequences.parquet", index=False)
    test_taxon_list.to_parquet(out / "test_taxon_list.parquet", index=False)
    ia.to_parquet(out / "ia_weights.parquet", index=False)

    print(f"  train_sequences : {len(train_sequences)} rows")
    print(f"  train_terms     : {len(train_terms)} rows")
    print(f"  train_taxonomy  : {len(train_taxonomy)} rows")
    print(f"  test_sequences  : {len(test_sequences)} rows")
    print(f"  test_taxon_list : {len(test_taxon_list)} rows")
    print(f"  ia_weights      : {len(ia)} rows")
    if go_graph is not None:
        print(f"  go_graph        : {len(go_graph.all_go_ids())} terms")
    print("  All join-integrity checks passed.\n")

    print("=" * 60)
    print("PHASE 0.3 -- Taxonomy tiering")
    print("=" * 60)
    taxonomy_reference = build_taxonomy_reference(
        test_taxon_list, train_taxonomy, dense_threshold=args.dense_threshold
    )
    taxonomy_reference.to_parquet(out / "taxonomy_reference.parquet", index=False)

    tier_counts = taxonomy_reference["tier"].value_counts().to_dict()
    kingdom_counts = taxonomy_reference["kingdom"].value_counts().to_dict()
    print(f"  {len(taxonomy_reference)} taxa mapped, 0 unmapped (would have raised)")
    print(f"  Tier counts    : {tier_counts}")
    print(f"  Kingdom counts : {kingdom_counts}\n")

    print("=" * 60)
    print("PHASE 0.4 -- Seen-vs-novel flagging")
    print("=" * 60)
    flagged_test = flag_seen_proteins(test_sequences, train_sequences)
    flagged_test.to_parquet(out / "test_seen_flags.parquet", index=False)

    summary = summarize_partial_knowledge(flagged_test)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print()

    print("=" * 60)
    print("PHASE 0.5 -- Validation split")
    print("=" * 60)
    print(
        "  CAVEAT: no timestamp column in train_terms.tsv -- this is a "
        "stratified random holdout, NOT a true temporal split.\n"
    )
    split_df = build_validation_split(
        train_sequences,
        train_terms,
        holdout_frac=args.holdout_frac,
        min_proteins_per_term=args.min_proteins_per_term,
    )
    split_df.to_parquet(out / "train_val_split.parquet", index=False)

    coverage = validate_split_coverage(split_df, train_terms)
    for k, v in coverage.items():
        print(f"  {k}: {v}")

    print(f"\nAll Phase 0 artifacts written to {out}/")


if __name__ == "__main__":
    main()