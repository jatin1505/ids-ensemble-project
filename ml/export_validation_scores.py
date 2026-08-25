"""
ml/export_validation_scores.py

The trained models (ml/saved_models/*.joblib, *.keras) and the processed
data (data/processed/*.npy) are gitignored on purpose -- they're large and
don't belong in git. That also means I can't see them just by cloning the
repo, and score_normalization.py can't be fit on real numbers until someone
runs this and shares ONLY the output, which is a few MB of numbers, not the
models or the traffic itself.

Run this locally once isolation_forest.joblib, autoencoder.keras, gmm.joblib
and data/processed/X_val.npy / y_val.npy all exist:

    python ml/export_validation_scores.py

Output: ml/saved_models/validation_raw_scores.npz
    Upload just this one file back into the chat.

What's inside it: raw (un-flipped, un-normalized) scores from all three
models on the validation set, plus the ground-truth labels. Sign-flipping
and [0,1] scaling stay ensemble/score_normalization.py's job, not this
script's -- this script's only job is "run the frozen models, save what
comes out."
"""
import time

import joblib
import numpy as np
from tensorflow import keras

from config import PROCESSED_DIR, SAVED_MODELS_DIR


def main():
    print("Loading validation set...")
    X_val = np.load(PROCESSED_DIR / "X_val.npy")
    y_val = np.load(PROCESSED_DIR / "y_val.npy", allow_pickle=True)
    print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")

    print("\nLoading trained models...")
    if_model = joblib.load(SAVED_MODELS_DIR / "isolation_forest.joblib")
    ae_model = keras.models.load_model(SAVED_MODELS_DIR / "autoencoder.keras")
    gmm_model = joblib.load(SAVED_MODELS_DIR / "gmm.joblib")

    print("\nScoring the validation set with each model "
          "(raw scores -- no sign flip, no normalization yet)...")

    t0 = time.time()
    if_scores = if_model.decision_function(X_val)
    print(f"  isolation_forest: {time.time() - t0:.1f}s")

    t0 = time.time()
    recon = ae_model.predict(X_val, verbose=0)
    ae_scores = np.mean(np.square(X_val - recon), axis=1)
    print(f"  autoencoder:      {time.time() - t0:.1f}s")

    t0 = time.time()
    gmm_scores = gmm_model.score_samples(X_val)
    print(f"  gmm:              {time.time() - t0:.1f}s")

    out_path = SAVED_MODELS_DIR / "validation_raw_scores.npz"
    np.savez_compressed(
        out_path,
        isolation_forest=if_scores,
        autoencoder=ae_scores,
        gmm=gmm_scores,
        y_val=y_val,
    )
    print(f"\nSaved: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    print("Upload just this file -- it's scores + labels, not the models "
          "or the traffic itself.")


if __name__ == "__main__":
    main()