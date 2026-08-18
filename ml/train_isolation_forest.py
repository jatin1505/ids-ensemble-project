"""
ml/train_isolation_forest.py

Trains the first of the three ensemble models. Isolation Forest works
by building many random trees, each isolating points by repeatedly
picking a random feature and a random split value. Points that are
"different" from the bulk of the data get isolated in fewer splits --
a shorter average path length across the forest. Points needing more
splits to isolate are considered typical/normal.

Trained benign-only, like every model in this ensemble -- see
docs/DECISIONS.md #1 for why that's required, not optional.

Run from the project root:
    python ml/train_isolation_forest.py

Input:  data/processed/X_train_benign.npy  (from preprocessing.py)
Output: ml/saved_models/isolation_forest.joblib
"""

import time

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from config import PROCESSED_DIR, SAVED_MODELS_DIR, SEED

# contamination='auto' lets sklearn set its own internal threshold --
# we don't use IsolationForest's built-in predict()/threshold at all,
# since AGL's risk_engine.py does its own thresholding on the fused
# ensemble score later. We only ever call decision_function() from here on.
N_ESTIMATORS = 200
CONTAMINATION = "auto"


def main():
    X_train = np.load(PROCESSED_DIR / "X_train_benign.npy")
    print(f"Loaded benign training matrix: {X_train.shape}")

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=SEED,
        n_jobs=-1,  # use all available CPU cores -- this is the one
                    # model in the ensemble where that's trivial to do
    )

    print(f"Training IsolationForest (n_estimators={N_ESTIMATORS})...")
    t0 = time.time()
    model.fit(X_train)
    print(f"Done in {time.time() - t0:.1f}s")

    # Quick sanity check: score the training data itself and look at
    # the distribution. decision_function() -> higher = more normal.
    # This does NOT need to match anything precisely -- it's just a
    # smoke test that the model isn't producing garbage (e.g. every
    # score identical, which would mean something's wrong upstream).
    train_scores = model.decision_function(X_train)
    print(f"\nSanity check -- decision_function() on the training data "
          f"it was fit on (should skew toward 'normal', i.e. positive):")
    print(f"    min={train_scores.min():.4f}  "
          f"max={train_scores.max():.4f}  "
          f"mean={train_scores.mean():.4f}  "
          f"std={train_scores.std():.4f}")

    out_path = SAVED_MODELS_DIR / "isolation_forest.joblib"
    joblib.dump(model, out_path)
    print(f"\nSaved model to {out_path}")
    print(
        "\nReminder for ensemble/score_normalization.py: this model's "
        "decision_function() runs 'higher = normal' -- it needs a sign "
        "flip (anomaly_score = -decision_function(x)) before fusion. "
        "See docs/ARCHITECTURE.md section 3, step 1."
    )


if __name__ == "__main__":
    main()