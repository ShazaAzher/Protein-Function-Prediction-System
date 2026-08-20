"""Task 3.1 — blend classifier scores (cold-start or PK-conditioned,
routed per protein by the NK/LK/PK flag from data/dataset.py) with
homology-transfer scores. Configurable weights in pipeline.yaml.
"""

def blend(
    classifier_scores: dict[str, dict[str, float]],
    homology_scores: dict[str, dict[str, float]],
    classifier_weight: float,
    homology_weight: float,
) -> dict[str, dict[str, float]]:
    if classifier_weight < 0 or homology_weight < 0:
        raise ValueError(
            f"weights must be non-negative, got classifier_weight={classifier_weight}, "
            f"homology_weight={homology_weight}"
        )

    proteins = set(classifier_scores) | set(homology_scores)
    blended: dict[str, dict[str, float]] = {}
    for protein_id in proteins:
        clf = classifier_scores.get(protein_id, {})
        hom = homology_scores.get(protein_id, {})
        terms = set(clf) | set(hom)
        if not terms:
            continue
        blended[protein_id] = {
            term: classifier_weight * clf.get(term, 0.0) + homology_weight * hom.get(term, 0.0)
            for term in terms
        }
    return blended