"""
ml/fit_score_normalizer.py

The one remaining step before backend/model_runtime.py can run for
real: fits AGL's ScoreNormalizer on real validation-set scores and
saves the result to ml/saved_models/norm_bounds.json.

Run once (and again any time a model is retrained), from the project
root, after export_validation_scores.py has produced
validation_raw_scores.npz:

    python ml/export_validation_scores.py   # if not already done
    python ml/fit_score_normalizer.py

This lives in ml/ next to export_validation_scores.py since it
consumes that script's output directly, but it needs ensemble/ too --
see the sys.path line below for why that's not just a bare import.
"""

import sys
from pathlib import Path

# ml/ scripts normally do bare `from config import ...`, which works
# because running `python ml/some_script.py` puts ml/ itself on
# sys.path automatically. That's not enough here -- this script also
# needs ensemble/, a sibling top-level package, which only resolves if
# the repo ROOT is on sys.path too. Adding it explicitly makes this
# script correct regardless of the working directory it's run from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from config import SAVED_MODELS_DIR
from ensemble.score_normalization import ScoreNormalizer

SCORES_PATH = SAVED_MODELS_DIR / "validation_raw_scores.npz"
OUTPUT_PATH = SAVED_MODELS_DIR / "norm_bounds.json"


def main():
    if not SCORES_PATH.exists():
        raise FileNotFoundError(
            f"{SCORES_PATH} doesn't exist yet -- run "
            f"ml/export_validation_scores.py first."
        )

    data = np.load(SCORES_PATH)
    val_scores = {
        "isolation_forest": data["isolation_forest"],
        "autoencoder": data["autoencoder"],
        "gmm": data["gmm"],
    }
    # y_val is in this file too (for evaluate.py's future use) but
    # deliberately unused here -- fitting the normalizer's bounds only
    # ever touches the scores, never the labels.

    normalizer = ScoreNormalizer().fit(val_scores)
    normalizer.save(OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH}")
    for model_name, (lo, hi) in normalizer._bounds.items():
        print(f"  {model_name:16s} lo={lo:.4f}  hi={hi:.4f}")


if __name__ == "__main__":
    main()