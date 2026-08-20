from __future__ import annotations
"""Task 3.3a — candidate-generation stage (expensive, cacheable).
Runs embedding generation + homology search once; downstream classifier
training and ensembling (Task 3.3b) can iterate against cached outputs.
"""

"""Task 3.3a — candidate-generation stage (expensive, cacheable)."""

import sys
from pathlib import Path
from typing import Callable
import yaml

from goprot.candidates.sequence_similarity import build_diamond_db, search
from goprot.data.parsing import parse_test_fasta, parse_train_fasta
from goprot.model.encoder import embed_sequences, save_embeddings_cache


def main(config_path: str = "configs/pipeline.yaml", model_loader: Callable[[], tuple] | None = None) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    raw_train_dir = Path(config["paths"]["raw_train_dir"])
    raw_test_dir = Path(config["paths"]["raw_test_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_fasta_path = raw_train_dir / "train_sequences.fasta"
    test_fasta_path = raw_test_dir / "testsuperset.fasta"

    train_df = parse_train_fasta(train_fasta_path)
    test_df = parse_test_fasta(test_fasta_path)

    db_path = str(processed_dir / "train_diamond")
    build_diamond_db(str(train_fasta_path), db_path)
    hits = search(str(test_fasta_path), db_path)
    hits.to_parquet(processed_dir / "diamond_hits.parquet")

    model_name = config["embedding"]["model"]
    batch_size = config["embedding"]["batch_size"]
    train_sequences = list(zip(train_df["accession"], train_df["sequence"]))
    test_sequences = list(zip(test_df["accession"], test_df["sequence"]))
    train_embeddings = embed_sequences(train_sequences, model_name=model_name, batch_size=batch_size, model_loader=model_loader)
    test_embeddings = embed_sequences(test_sequences, model_name=model_name, batch_size=batch_size, model_loader=model_loader)

    save_embeddings_cache(train_embeddings, str(processed_dir / "train_embeddings.parquet"))
    save_embeddings_cache(test_embeddings, str(processed_dir / "test_embeddings.parquet"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/pipeline.yaml")