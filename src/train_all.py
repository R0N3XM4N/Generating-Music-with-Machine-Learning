"""Stage 2: train all four models from the same prepared corpus and split."""

from .train import main


if __name__ == "__main__":
    main(forced_model="all")
