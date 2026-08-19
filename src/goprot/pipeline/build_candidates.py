"""Task 3.3a — candidate-generation stage (expensive, cacheable).
Runs embedding generation + homology search once; downstream classifier
training and ensembling (Task 3.3b) can iterate against cached outputs.
"""


def main(config_path: str) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/pipeline.yaml")
