from __future__ import annotations


"""ESM2 embedding wrapper -- shared infra for the cold-start (Task 1.4) and
PK-conditioned (Task 2.3) classifiers.

HONESTY NOTE: I could not download real ESM2 weights in this sandbox --
dl.fbaipublicfiles.com returns 403 through the network proxy here (same
kind of allowlist constraint that blocked the NCBI taxdump earlier, not a
real access-denial from Meta's public bucket). embed_sequences below is
written against fair-esm's documented API (confirmed `import esm` works;
the model download itself is what's blocked) and is unit-tested with a
stub model injected via model_loader, NOT run against a real ESM2 forward
pass. Do a quick smoke test on your end -- a handful of sequences through
esm2_t6_8M_UR50D, the smallest checkpoint (~30MB) -- before trusting this
at scale.
"""



from typing import Callable

import numpy as np
import pandas as pd
import torch


def embed_sequences(
    sequences: list[tuple[str, str]],
    model_name: str = "esm2_t33_650M_UR50D",
    batch_size: int = 32,
    device: str | None = None,
    model_loader: Callable[[], tuple] | None = None,
) -> dict[str, np.ndarray]:
    """Mean-pool per-residue ESM2 embeddings into one vector per protein.

    sequences: [(accession, sequence), ...]
    model_loader: optional callable() -> (model, alphabet), for injecting a
        stub model in tests without downloading real weights. Defaults to
        `esm.pretrained.load_model_and_alphabet(model_name)` from fair-esm.

    -> {accession: np.ndarray of shape (embed_dim,)}

    Excludes the BOS/EOS tokens from the mean-pool -- fair-esm's batch
    converter prepends BOS and appends EOS, so real residues for a
    sequence of length L sit at token positions [1, L], not [0, L-1].
    """
    if model_loader is None:
        import esm  # local import -- keeps `import esm` optional for callers who only use the stub path in tests
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    else:
        model, alphabet = model_loader()

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    repr_layer = model.num_layers

    embeddings: dict[str, np.ndarray] = {}
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        _labels, _strs, tokens = batch_converter(batch)
        tokens = tokens.to(device)
        with torch.no_grad():
            out = model(tokens, repr_layers=[repr_layer], return_contacts=False)
        reps = out["representations"][repr_layer]
        for j, (accession, seq) in enumerate(batch):
            embeddings[accession] = reps[j, 1 : len(seq) + 1].mean(0).cpu().numpy()

    return embeddings


def save_embeddings_cache(embeddings: dict[str, np.ndarray], cache_path: str) -> None:
    """Parquet cache -- avoids re-running the ESM2 forward pass on
    sequences already embedded in a prior run.
    """
    df = pd.DataFrame({
        "accession": list(embeddings.keys()),
        "embedding": [emb.tolist() for emb in embeddings.values()],
    })
    df.to_parquet(cache_path)


def load_cached_embeddings(cache_path: str) -> dict[str, np.ndarray]:
    df = pd.read_parquet(cache_path)
    return {row["accession"]: np.array(row["embedding"]) for _, row in df.iterrows()}
