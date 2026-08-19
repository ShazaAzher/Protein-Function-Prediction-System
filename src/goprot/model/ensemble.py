"""Task 3.1 — blend classifier scores (cold-start or PK-conditioned,
routed per protein by the NK/LK/PK flag from data/dataset.py) with
homology-transfer scores. Configurable weights in pipeline.yaml.
"""


def blend(classifier_scores: dict, homology_scores: dict, classifier_weight: float, homology_weight: float) -> dict:
    raise NotImplementedError
