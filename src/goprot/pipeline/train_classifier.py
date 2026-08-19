"""Task 3.3b — trains both comparison arms (cold-start, PK-conditioned) on
the synthetic split, runs the ablation (Task 2.5), and blends with homology
scores for the final submission-ready pipeline.
"""


def main(config_path: str) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/pipeline.yaml")
