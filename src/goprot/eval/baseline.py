"""Task 1.2 — naive frequency baseline. Run first, before anything else,
to validate the metrics implementation. Run separately on the NK and PK
splits — the PK/NK gap on this trivial baseline is itself informative.
"""
import pandas as pd 

def naive_baseline(train_terms: pd.DataFrame, aspect: str) -> dict[str, float]:
    aspect_terms = train_terms[train_terms["aspect"] == aspect]
    if aspect_terms.empty:
        return {}
    n_proteins = aspect_terms["accession"].nunique()
    counts = aspect_terms.groupby("go_id")["accession"].nunique()
    return (counts / n_proteins).to_dict()