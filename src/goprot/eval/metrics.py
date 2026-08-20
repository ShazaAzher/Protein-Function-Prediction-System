"""Task 1.1 — IA-weighted precision/recall/F-max, PK-aware.

Per the CAFA5 derivation, conditional information content for PK collapses to
the ordinary IA weights summed over the newly-added terms only (T1 \ T0) — see
project notes / Appendix A of the CAFA5 paper. So this does NOT need a novel
metric, just correct filtering: for PK-setting proteins, exclude T0 from both
the predicted set and the ground-truth set before scoring.
"""


def weighted_precision_recall(predictions, ground_truth, ia_weights, known_terms=None, threshold=0.0):
    precisions, recalls = [], []
    for protein_id, true_terms in ground_truth.items():
        known = known_terms.get(protein_id, set()) if known_terms else set()
        true_eval = true_terms - known
        true_weight = sum(ia_weights.get(t, 0.0) for t in true_eval)
        if true_weight == 0.0:
            continue  # nothing left to evaluate -- recall undefined, not zero

        scores = predictions.get(protein_id, {})
        predicted = {term for term, score in scores.items() if score >= threshold} - known

        overlap_weight = sum(ia_weights.get(t, 0.0) for t in predicted & true_eval)
        recalls.append(overlap_weight / true_weight)

        if predicted:
            predicted_weight = sum(ia_weights.get(t, 0.0) for t in predicted)
            if predicted_weight > 0.0:
                precisions.append(overlap_weight / predicted_weight)

    wpr = sum(precisions) / len(precisions) if precisions else 0.0
    wrc = sum(recalls) / len(recalls) if recalls else 0.0
    return wpr, wrc


def f_max(predictions, ground_truth, ia_weights, known_terms=None, thresholds=None):
    if thresholds is None:
        thresholds = [i / 100 for i in range(101)]
    best = 0.0
    for tau in thresholds:
        wpr, wrc = weighted_precision_recall(predictions, ground_truth, ia_weights, known_terms, threshold=tau)
        if wpr + wrc == 0.0:
            continue
        best = max(best, 2 * wpr * wrc / (wpr + wrc))
    return best