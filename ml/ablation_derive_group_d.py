"""
ml/ablation_derive_group_d.py

Feasibility spike, step 6. The relationship check suggests two of the four
Group D features are (nearly) functions of features we can already compute:

    Magnitue ~ sqrt(2 * AVG)        Radius ~ sqrt(2) * Std

while Covariance and Variance could NOT be predicted from other features. This
script tests, with the FROZEN saved models (no retraining, nothing saved), how
much detection is lost if a live extractor
  (a) derives Magnitue and Radius from AVG and Std, and
  (b) cannot supply Covariance and/or Variance (neutralized to benign median).

Every scenario also neutralizes the 12 unreproducible features, exactly as the
step-3 'wide' scenario did, so the first row below reproduces 'wide'.

Caveat: neutralizing on frozen models is pessimistic (the autoencoder sees an
inconsistent row). The fair test of whether Covariance/Variance matter is
retraining without them (ml/ablation_retrain_subset.py).

Run from project root:   python ml/ablation_derive_group_d.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from tensorflow import keras

from config import BENIGN_LABEL, PROCESSED_DIR, SAVED_MODELS_DIR, SEED
from ensemble.risk_engine import fuse
from ensemble.score_normalization import ScoreNormalizer

UNREPRODUCIBLE = ["IAT", "Weight", "Header_Length", "urg_count", "rst_count",
                  "ack_count", "syn_count", "fin_count", "flow_duration",
                  "Number", "Rate", "Srate"]

feature_cols = json.loads((SAVED_MODELS_DIR / "feature_columns.json").read_text())
ix = {c: i for i, c in enumerate(feature_cols)}
scaler = joblib.load(SAVED_MODELS_DIR / "scaler.joblib")

X_train = np.load(PROCESSED_DIR / "X_train_benign.npy")
X_val = np.load(PROCESSED_DIR / "X_val.npy")
y_val = np.load(PROCESSED_DIR / "y_val.npy", allow_pickle=True)
is_attack = (y_val != BENIGN_LABEL).astype(int)

# Same evaluation rows as steps 3 and 4 (same seed, same procedure). Keep this
# block identical to those scripts so the numbers are comparable.
rng = np.random.default_rng(SEED)
benign_idx = np.where(is_attack == 0)[0]
attack_idx = rng.choice(np.where(is_attack == 1)[0], size=100_000, replace=False)
keep = np.concatenate([benign_idx, attack_idx])
X_eval, y = X_val[keep], is_attack[keep]

# Work in RAW units so formulas like sqrt(2 * AVG) mean what they say, then
# scale back with the frozen scaler before the models see anything.
raw_eval = scaler.inverse_transform(X_eval.astype("float64"))
benign_median = np.median(scaler.inverse_transform(X_train.astype("float64")), axis=0)


def col(D, name):
    return D[:, ix[name]]


def derived(name):
    # Un-scaling float noise can leave AVG/Std a hair below 0, so clip first.
    if name == "Magnitue":
        return np.sqrt(2.0 * np.clip(col(raw_eval, "AVG"), 0, None))
    return np.sqrt(2.0) * np.clip(col(raw_eval, "Std"), 0, None)  # Radius


# --- Diagnostic: how close are the derived values to the real ones? -------
print("Derived vs real Group D values on the evaluation rows:")
std0 = col(raw_eval, "Std") <= 1e-6
print(f"  Rows with Std == 0: {std0.sum():,} ({std0.mean():.1%}); real Radius is "
      f"~0 on {np.mean(col(raw_eval, 'Radius')[std0] <= 1e-6):.1%} of them")
for name in ("Magnitue", "Radius"):
    real = col(raw_eval, name)
    for subset, mask in [("benign", y == 0), ("attack", y == 1)]:
        big = mask & (real > 1e-3)
        err = np.abs(derived(name)[big] - real[big]) / real[big]
        print(f"  {name:<9}{subset:<8} median rel. error {np.median(err):.3f}  "
              f"p95 {np.percentile(err, 95):.3f}  (rows: {big.sum():,})")


def make_scenario(derive, neutralize):
    D = raw_eval.copy()
    for name in UNREPRODUCIBLE + neutralize:
        D[:, ix[name]] = benign_median[ix[name]]
    if derive:
        D[:, ix["Magnitue"]] = derived("Magnitue")
        D[:, ix["Radius"]] = derived("Radius")
    return scaler.transform(D).astype("float32")


if_model = joblib.load(SAVED_MODELS_DIR / "isolation_forest.joblib")
ae_model = keras.models.load_model(SAVED_MODELS_DIR / "autoencoder.keras")
gmm_model = joblib.load(SAVED_MODELS_DIR / "gmm.joblib")
normalizer = ScoreNormalizer.load(SAVED_MODELS_DIR / "norm_bounds.json")


def score_all(X_in):
    recon = ae_model.predict(X_in, verbose=0)
    raw = {
        "isolation_forest": if_model.decision_function(X_in),
        "autoencoder": np.mean(np.square(X_in - recon), axis=1),
        "gmm": gmm_model.score_samples(X_in),
    }
    return normalizer.transform(raw)


SCENARIOS = [
    ("wide (reference, Group D real)", False, []),
    ("derive M+R; Cov+Var real", True, []),
    ("derive M+R; Cov neutralized", True, ["Covariance"]),
    ("derive M+R; Var neutralized", True, ["Variance"]),
    ("derive M+R; Cov+Var neutralized", True, ["Covariance", "Variance"]),
]

print(f"\nEvaluating on {len(y):,} rows ({int((y == 0).sum()):,} benign, "
      f"{int((y == 1).sum()):,} attack). Frozen models, ROC-AUC.\n")
print(f"{'scenario':<34}{'IF':>7}{'AE':>7}{'GMM':>7}{'all3':>8}{'AE+GMM':>9}")
for label, derive, neutralize in SCENARIOS:
    norm = score_all(make_scenario(derive, neutralize))
    auc = {m: roc_auc_score(y, s) for m, s in norm.items()}
    all3 = roc_auc_score(y, fuse(norm))
    ae_gmm = roc_auc_score(y, fuse({m: norm[m] for m in ("autoencoder", "gmm")}))
    print(f"{label:<34}{auc['isolation_forest']:>7.3f}{auc['autoencoder']:>7.3f}"
          f"{auc['gmm']:>7.3f}{all3:>8.3f}{ae_gmm:>9.3f}")