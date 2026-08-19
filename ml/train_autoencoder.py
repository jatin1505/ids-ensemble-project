"""
ml/train_autoencoder.py

Trains the second ensemble model. An autoencoder is a neural net
trained to compress each row down to a small "bottleneck" (8 numbers,
here) and then reconstruct the original 46 features from just that.
Trained only on benign traffic, it gets good at compressing/rebuilding
NORMAL patterns and stays bad at rebuilding anything it's never seen --
so reconstruction error (mean squared error between input and output)
becomes the anomaly signal: low error = looks normal, high error =
doesn't fit the patterns it learned.

Trained benign-only, like every model in this ensemble -- see
docs/DECISIONS.md #1 for why that's required, not optional.

--- Design notes worth understanding ---

1. Output layer is linear, not sigmoid. Features are StandardScaler'd
   (mean 0, values go negative) -- a sigmoid output caps at [0,1] and
   would make correct reconstruction impossible by construction.

2. The validation split used for early stopping here is carved out of
   the BENIGN TRAINING rows themselves (10%), NOT data/processed/X_val.npy.
   That file is reserved for AGL's threshold calibration and the final
   ensemble evaluation -- touching it here to tune this model's own
   training would blur that boundary. This internal split exists only
   to know when to stop training, nothing else.

3. EarlyStopping + restore_best_weights: trains for up to MAX_EPOCHS,
   but stops once validation loss stops improving and rolls back to
   the best checkpoint -- protects against quietly overfitting the
   ~117K benign rows available for training.

Run from ml/:
    python train_autoencoder.py

Input:  data/processed/X_train_benign.npy  (from preprocessing.py)
Output: ml/saved_models/autoencoder.keras
"""

import time

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

from config import PROCESSED_DIR, SAVED_MODELS_DIR, SEED

INTERNAL_VAL_FRACTION = 0.10  # carved out of benign TRAIN rows only
BOTTLENECK_DIM = 8
BATCH_SIZE = 256
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
LEARNING_RATE = 1e-3


def build_autoencoder(n_features: int) -> keras.Model:
    inputs = keras.Input(shape=(n_features,), name="flow_features")

    # Encoder: 46 -> 32 -> 16 -> 8
    x = layers.Dense(32, activation="relu")(inputs)
    x = layers.Dense(16, activation="relu")(x)
    bottleneck = layers.Dense(BOTTLENECK_DIM, activation="relu", name="bottleneck")(x)

    # Decoder: 8 -> 16 -> 32 -> 46
    x = layers.Dense(16, activation="relu")(bottleneck)
    x = layers.Dense(32, activation="relu")(x)
    # Linear output -- NOT sigmoid. Scaled features can be negative.
    outputs = layers.Dense(n_features, activation="linear", name="reconstruction")(x)

    model = keras.Model(inputs, outputs, name="autoencoder")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
    )
    return model


def main():
    tf.random.set_seed(SEED)

    X = np.load(PROCESSED_DIR / "X_train_benign.npy")
    print(f"Loaded benign training matrix: {X.shape}")

    # Internal train/val split -- carved from benign TRAIN rows only,
    # NOT the official validation.csv split. See docstring note #2.
    X_fit, X_internal_val = train_test_split(
        X, test_size=INTERNAL_VAL_FRACTION, random_state=SEED
    )
    print(f"Internal split for early stopping: {X_fit.shape[0]:,} train, "
          f"{X_internal_val.shape[0]:,} held out (not the official validation set)")

    model = build_autoencoder(n_features=X.shape[1])
    model.summary()

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    print(f"\nTraining (max {MAX_EPOCHS} epochs, "
          f"early stopping patience={EARLY_STOPPING_PATIENCE})...")
    t0 = time.time()
    history = model.fit(
        X_fit, X_fit,  # autoencoder: input IS the target
        validation_data=(X_internal_val, X_internal_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=2,
    )
    elapsed = time.time() - t0
    epochs_run = len(history.history["loss"])
    print(f"\nStopped after {epochs_run} epochs ({elapsed:.1f}s total). "
          f"Best val_loss={min(history.history['val_loss']):.6f}")

    # Sanity check: reconstruction error on the internal held-out
    # benign rows (data the model didn't train on, but is still
    # benign) -- should be low and tight if training went well.
    recon = model.predict(X_internal_val, verbose=0)
    mse_per_row = np.mean(np.square(X_internal_val - recon), axis=1)
    print(f"\nSanity check -- reconstruction MSE on held-out benign rows:")
    print(f"    min={mse_per_row.min():.4f}  "
          f"max={mse_per_row.max():.4f}  "
          f"mean={mse_per_row.mean():.4f}  "
          f"std={mse_per_row.std():.4f}")

    out_path = SAVED_MODELS_DIR / "autoencoder.keras"
    model.save(out_path)
    print(f"\nSaved model to {out_path}")
    print(
        "\nReminder for ensemble/score_normalization.py: this model's "
        "reconstruction MSE runs 'higher = anomalous' already -- it "
        "does NOT need a sign flip, unlike Isolation Forest and GMM. "
        "See docs/ARCHITECTURE.md section 3, step 1."
    )


if __name__ == "__main__":
    main()