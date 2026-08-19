"""ESM2 embedding wrapper (Task 1.x infra, shared by cold-start and PK paths).

Batch-embed train and test sequences, fp16, cache to disk (parquet/hdf5) —
one-time cost per sequence set.
"""


def embed_sequences(fasta_records, model_name: str = "esm2_t33_650M_UR50D", batch_size: int = 32):
    raise NotImplementedError


def load_cached_embeddings(cache_path: str):
    raise NotImplementedError
