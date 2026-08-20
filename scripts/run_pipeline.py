"""Task 3.3c — single command, full pipeline end-to-end for a given config."""
from __future__ import annotations
import sys
from typing import Callable
from goprot.pipeline import build_candidates, train_classifier


def main(config_path: str = "configs/pipeline.yaml", model_loader: Callable[[], tuple] | None = None) -> list[dict]:
    build_candidates.main(config_path, model_loader=model_loader)
    results = []
    for aspect in ("P", "F", "C"):
        try:
            results.append(train_classifier.main(config_path, aspect=aspect))
        except ValueError as e:
            print(f"[{aspect}] skipped: {e}")
    for r in results:
        print(f"  {r['aspect']}: n={r['n_pk_instances']:>6}  cold-start={r['cold_start_fmax']:.4f}  pk-conditioned={r['pk_fmax']:.4f}")
    return results


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/pipeline.yaml")