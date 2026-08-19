"""
Parse raw data/ files and write validated, typed artifacts to
data/processed/ as parquet.

Run once (or whenever raw data changes) rather than re-parsing FASTA/
OBO/TSV inline in every downstream script.

Usage:
    python scripts/build_processed_data.py \
        --raw-dir data/raw \
        --processed-dir data/processed
"""
from __future__ import annotations

import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from goprot.data.parsing import (  # noqa: E402
    load_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--go-obo",
        type=Path,
        default=None,
        help="Optional path to go-basic.obo, to cross-validate IA.tsv terms.",
    )
    args = parser.parse_args()

    raw = args.raw_dir
    out = args.processed_dir
    out.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        train_sequences_path=raw / "Train" / "train_sequences.fasta",
        train_terms_path=raw / "Train" / "train_terms.tsv",
        train_taxonomy_path=raw / "Train" / "train_taxonomy.tsv",
        test_sequences_path=raw / "Test" / "testsuperset.fasta",
        test_taxon_list_path=raw / "Test" / "testsuperset-taxon-list.tsv",
        ia_path=raw / "IA.tsv",
        go_obo_path=args.go_obo,
    )

    dataset["train_sequences"].to_parquet(out / "train_sequences.parquet", index=False)
    dataset["train_terms"].to_parquet(out / "train_terms.parquet", index=False)
    dataset["train_taxonomy"].to_parquet(out / "train_taxonomy.parquet", index=False)
    dataset["test_sequences"].to_parquet(out / "test_sequences.parquet", index=False)
    dataset["test_taxon_list"].to_parquet(out / "test_taxon_list.parquet", index=False)
    dataset["ia"].to_parquet(out / "ia_weights.parquet", index=False)

    print(f"Wrote processed artifacts to {out}/")
    for name in [
        "train_sequences",
        "train_terms",
        "train_taxonomy",
        "test_sequences",
        "test_taxon_list",
        "ia",
    ]:
        n = len(dataset[name])
        print(f"  {name}.parquet  ({n} rows)")


if __name__ == "__main__":
    main()