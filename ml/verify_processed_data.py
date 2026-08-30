"""
ml/verify_processed_data.py

Run this immediately after ml/preprocessing.py finishes, before running
any training or export script. Catches an interrupted/truncated write
(the exact failure signature we hit: a valid .npy header with zero
actual data payload) right away, instead of discovering it several
steps later in train_isolation_forest.py or export_validation_scores.py.

Checks each expected file for:
  - existence
  - correct shape (rows are dataset-dependent so only column count /
    1D-ness is enforced strictly; row counts are printed for a manual
    sanity check against the numbers in docs/ARCHITECTURE.md)
  - no NaN / Inf values
  - (for the two label arrays) at least one BenignTraffic row and more
    than one distinct label -- a truncated/all-zero file would fail
    this trivially

Run from ml/:
    python verify_processed_data.py
"""

import sys

import numpy as np
import joblib

from config import BENIGN_LABEL, PROCESSED_DIR, SAVED_MODELS_DIR

N_FEATURES = 46

# path -> (expected_ndim, expected_last_dim or None)
ARRAY_CHECKS = {
    "X_train_benign.npy": (2, N_FEATURES),
    "X_test.npy": (2, N_FEATURES),
    "y_test.npy": (1, None),
    "X_val.npy": (2, N_FEATURES),
    "y_val.npy": (1, None),
}


def check_array(name: str, expected_ndim: int, expected_last_dim) -> bool:
    path = PROCESSED_DIR / name
    if not path.exists():
        print(f"  [FAIL] {name}: file does not exist")
        return False

    try:
        arr = np.load(path, allow_pickle=True)
    except Exception as e:
        print(f"  [FAIL] {name}: could not load -- {e}")
        return False

    if arr.size == 0:
        print(f"  [FAIL] {name}: array is empty (size=0) -- this is the "
              f"exact signature of an interrupted/truncated write")
        return False

    if arr.ndim != expected_ndim:
        print(f"  [FAIL] {name}: expected {expected_ndim}D, got {arr.ndim}D")
        return False

    if expected_last_dim is not None and arr.shape[-1] != expected_last_dim:
        print(f"  [FAIL] {name}: expected {expected_last_dim} feature columns, "
              f"got {arr.shape[-1]}")
        return False

    if arr.dtype.kind == "f":
        if np.isnan(arr).any():
            print(f"  [FAIL] {name}: contains NaN values")
            return False
        if np.isinf(arr).any():
            print(f"  [FAIL] {name}: contains Inf values")
            return False

    if arr.dtype.kind == "O":  # label arrays
        unique_labels = np.unique(arr)
        if len(unique_labels) < 2:
            print(f"  [FAIL] {name}: only {len(unique_labels)} distinct label(s) "
                  f"found -- expected benign + multiple attack types")
            return False
        if BENIGN_LABEL not in unique_labels:
            print(f"  [FAIL] {name}: '{BENIGN_LABEL}' not found in labels")
            return False
        benign_count = int((arr == BENIGN_LABEL).sum())
        print(f"  [OK]   {name}: shape={arr.shape}, "
              f"{len(unique_labels)} distinct labels, "
              f"{benign_count:,} benign rows")
        return True

    print(f"  [OK]   {name}: shape={arr.shape}, dtype={arr.dtype}")
    return True


def main():
    print(f"Checking {PROCESSED_DIR} ...\n")
    results = [check_array(name, ndim, last_dim) for name, (ndim, last_dim) in ARRAY_CHECKS.items()]

    print()
    scaler_path = SAVED_MODELS_DIR / "scaler.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        if getattr(scaler, "n_features_in_", None) == N_FEATURES:
            print(f"  [OK]   scaler.joblib: n_features_in_={N_FEATURES}")
        else:
            print(f"  [FAIL] scaler.joblib: unexpected n_features_in_="
                  f"{getattr(scaler, 'n_features_in_', None)}")
            results.append(False)
    else:
        print(f"  [FAIL] scaler.joblib: does not exist")
        results.append(False)

    print()
    if all(results):
        print("ALL CHECKS PASSED -- safe to proceed to "
              "export_validation_scores.py / fit_score_normalizer.py.")
        sys.exit(0)
    else:
        print("ONE OR MORE CHECKS FAILED -- re-run preprocessing.py before "
              "doing anything else. Don't let the machine sleep or close the "
              "terminal mid-run, and confirm you have several GB of free disk "
              "space first.")
        sys.exit(1)


if __name__ == "__main__":
    main()