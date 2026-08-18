"""
ml/preprocessing.py

Turns the three raw CICIoT2023 CSVs into what every training script and
the ensemble actually needs: a benign-only, scaled feature matrix for
training, and cleaned+scaled test/validation matrices (with labels) for
evaluation.

Run once, from the project root:
    python ml/preprocessing.py

What this does, in order:
  1. Streams train.csv in chunks (it's 1.6GB / 5.49M rows -- too big to
     casually pd.read_csv() in one call on a student laptop) and:
       a. hashes every row (to catch cross-split duplicates later)
       b. keeps only the benign rows' feature values
  2. Fits a StandardScaler on the benign-only training features ONLY.
     This is the one thing every other script depends on getting right:
     if the scaler ever sees attack rows or test/val rows during
     fitting, that's data leakage. See docs/DECISIONS.md #5 and the
     ARCHITECTURE.md note on "never refit at inference."
  3. Streams test.csv and validation.csv, drops any row whose hash
     matches a row in train (the ~4.9% leakage found during EDA -- see
     docs/DECISIONS.md #5), then scales what's left with the SAME
     fitted scaler.
  4. Saves everything to data/processed/ as .npy arrays, plus the
     fitted scaler and the exact feature-column order to
     ml/saved_models/ -- every other script (train_isolation_forest.py,
     train_autoencoder.py, train_gmm.py, and eventually
     ensemble/score_normalization.py at inference time) loads these
     instead of re-deriving them, so everyone agrees on column order
     and scale.

Output files:
    ml/saved_models/scaler.joblib
    ml/saved_models/feature_columns.json
    data/processed/X_train_benign.npy
    data/processed/X_test.npy       data/processed/y_test.npy
    data/processed/X_val.npy        data/processed/y_val.npy
"""

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import (
    BENIGN_LABEL,
    LABEL_COL,
    PROCESSED_DIR,
    SAVED_MODELS_DIR,
    TEST_CSV,
    TRAIN_CSV,
    VAL_CSV,
)

CHUNKSIZE = 500_000


def _get_feature_columns(csv_path) -> list[str]:
    """Read just the header row to get feature column names/order."""
    header = pd.read_csv(csv_path, nrows=0)
    return [c for c in header.columns if c != LABEL_COL]


def build_train_hashes_and_benign_matrix(feature_cols: list[str]):
    """
    Single pass over train.csv.

    Returns:
        train_hashes: set of row hashes, used to detect leaked
                       duplicates in test/validation.
        benign_df:    DataFrame of ONLY the benign rows' feature
                       columns, still unscaled (scaler gets fit on this
                       right after this function returns).
    """
    train_hashes: set[int] = set()
    benign_chunks: list[pd.DataFrame] = []

    total_rows = 0
    total_benign = 0

    print(f"[1/4] Streaming {TRAIN_CSV.name} in chunks of {CHUNKSIZE:,} rows...")
    t0 = time.time()

    for i, chunk in enumerate(pd.read_csv(TRAIN_CSV, chunksize=CHUNKSIZE)):
        total_rows += len(chunk)

        # Hash every row (features + label together) for dup detection.
        row_hashes = pd.util.hash_pandas_object(chunk, index=False)
        train_hashes.update(row_hashes.tolist())

        # Keep only benign rows' feature columns for training.
        benign_rows = chunk.loc[chunk[LABEL_COL] == BENIGN_LABEL, feature_cols]
        if len(benign_rows):
            benign_chunks.append(benign_rows)
            total_benign += len(benign_rows)

        print(f"    chunk {i + 1}: {total_rows:,} rows read so far, "
              f"{total_benign:,} benign so far", end="\r")

    print()  # newline after the \r progress line
    print(f"    done in {time.time() - t0:.1f}s -- "
          f"{total_rows:,} total rows, {total_benign:,} benign "
          f"({100 * total_benign / total_rows:.2f}%)")

    benign_df = pd.concat(benign_chunks, ignore_index=True)
    return train_hashes, benign_df


def clean_and_scale_eval_split(csv_path, feature_cols, scaler, train_hashes, split_name):
    """
    Streams a test/validation CSV, drops rows that are exact duplicates
    of a train row (data leakage), then scales the remaining features
    with the scaler already fit on benign train data.

    Returns:
        X: scaled feature matrix (np.ndarray)
        y: label array (np.ndarray of strings) -- kept for evaluation,
           NOT used for training anything.
    """
    print(f"\n[+] Streaming {csv_path.name}...")
    t0 = time.time()

    kept_feature_chunks = []
    kept_label_chunks = []
    total_rows = 0
    dropped_rows = 0

    for chunk in pd.read_csv(csv_path, chunksize=CHUNKSIZE):
        total_rows += len(chunk)
        row_hashes = pd.util.hash_pandas_object(chunk, index=False)
        is_leaked = row_hashes.isin(train_hashes)
        dropped_rows += int(is_leaked.sum())

        clean_chunk = chunk.loc[~is_leaked]
        if len(clean_chunk):
            kept_feature_chunks.append(clean_chunk[feature_cols])
            kept_label_chunks.append(clean_chunk[LABEL_COL])

    X_df = pd.concat(kept_feature_chunks, ignore_index=True)
    y = pd.concat(kept_label_chunks, ignore_index=True).to_numpy()

    X = scaler.transform(X_df.to_numpy(dtype="float64"))

    print(f"    {split_name}: {total_rows:,} rows read, "
          f"{dropped_rows:,} dropped as train-duplicates "
          f"({100 * dropped_rows / total_rows:.2f}%), "
          f"{len(y):,} rows kept -- done in {time.time() - t0:.1f}s")

    return X.astype("float32"), y


def main():
    feature_cols = _get_feature_columns(TRAIN_CSV)
    print(f"Confirmed {len(feature_cols)} feature columns (+ '{LABEL_COL}').")

    # ---- Pass 1: train.csv -> hashes + benign-only feature rows ----
    train_hashes, benign_df = build_train_hashes_and_benign_matrix(feature_cols)

    # ---- Fit scaler on benign-only train features ONLY ----
    print("\n[2/4] Fitting StandardScaler on benign-only train features...")
    scaler = StandardScaler()
    X_train_benign = scaler.fit_transform(benign_df.to_numpy(dtype="float64"))
    X_train_benign = X_train_benign.astype("float32")
    print(f"    scaler fit on {X_train_benign.shape[0]:,} benign rows, "
          f"{X_train_benign.shape[1]} features.")

    # ---- Clean + scale test/validation splits ----
    print("\n[3/4] Cleaning and scaling test/validation splits (dropping "
          "train-duplicates)...")
    X_test, y_test = clean_and_scale_eval_split(
        TEST_CSV, feature_cols, scaler, train_hashes, "test"
    )
    X_val, y_val = clean_and_scale_eval_split(
        VAL_CSV, feature_cols, scaler, train_hashes, "validation"
    )

    # ---- Save everything ----
    print("\n[4/4] Saving processed arrays and artifacts...")
    np.save(PROCESSED_DIR / "X_train_benign.npy", X_train_benign)
    np.save(PROCESSED_DIR / "X_test.npy", X_test)
    np.save(PROCESSED_DIR / "y_test.npy", y_test)
    np.save(PROCESSED_DIR / "X_val.npy", X_val)
    np.save(PROCESSED_DIR / "y_val.npy", y_val)

    joblib.dump(scaler, SAVED_MODELS_DIR / "scaler.joblib")
    with open(SAVED_MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    print(f"""
Done. Saved:
  {PROCESSED_DIR / 'X_train_benign.npy'}  shape={X_train_benign.shape}
  {PROCESSED_DIR / 'X_test.npy'}          shape={X_test.shape}
  {PROCESSED_DIR / 'y_test.npy'}          shape={y_test.shape}
  {PROCESSED_DIR / 'X_val.npy'}           shape={X_val.shape}
  {PROCESSED_DIR / 'y_val.npy'}           shape={y_val.shape}
  {SAVED_MODELS_DIR / 'scaler.joblib'}
  {SAVED_MODELS_DIR / 'feature_columns.json'}

Every training script (train_isolation_forest.py, train_autoencoder.py,
train_gmm.py) should load X_train_benign.npy directly -- do not re-read
the raw CSVs or refit a scaler anywhere else.
""")


if __name__ == "__main__":
    main()