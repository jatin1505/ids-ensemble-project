"""
ml/ablation_neutralize_features.py

Feasibility spike, step 3. Re-scores the validation set three ways:
  1. baseline    - all 46 features untouched
  2. core        - the clearly-unreproducible features replaced by their
                   benign-typical value
  3. wide        - core + features whose definitions are uncertain
If detection barely changes, live capture can omit those features safely.
If it collapses, the models lean on artifact features live traffic can't supply.

Run from project root:   python ml/ablation_neutralize_features.py
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
from ensemble.risk_engine import LOW_MEDIUM_THRESHOLD, fuse
from ensemble.score_normalization import ScoreNormalizer

CORE = ["IAT", "Weight", "Header_Length", "urg_count", "rst_count"]
WIDE = CORE + ["ack_count", "syn_count", "fin_count", "flow_duration",
               "Number", "Rate", "Srate"]

feature_cols = json.loads((SAVED_MODELS_DIR / "feature_columns.json").read_text())
X_train = np.load(PROCESSED_DIR / "X_train_benign.npy")
X_val = np.load(PROCESSED_DIR / "X_val.npy")
y_val = np.load(PROCESSED_DIR / "y_val.npy", allow_pickle=True)
is_attack = (y_val != BENIGN_LABEL).astype(int)

# ROC-AUC, recall and false-positive rate don't depend on the benign:attack
# ratio, so subsample attacks (fixed seed) to keep this fast.
rng = np.random.default_rng(SEED)
benign_idx = np.where(is_attack == 0)[0]
attack_idx = rng.choice(np.where(is_attack == 1)[0], size=100_000, replace=False)
keep = np.concatenate([benign_idx, attack_idx])
X, y = X_val[keep], is_attack[keep]

# "Benign-typical" value for each feature, in SCALED units
fill = np.median(X_train, axis=0)

if_model = joblib.load(SAVED_MODELS_DIR / "isolation_forest.joblib")
ae_model = keras.models.load_model(SAVED_MODELS_DIR / "autoencoder.keras")
gmm_model = joblib.load(SAVED_MODELS_DIR / "gmm.joblib")
normalizer = ScoreNormalizer.load(SAVED_MODELS_DIR / "norm_bounds.json")


def neutralize(X_in, names):
    X_out = X_in.copy()
    idx = [feature_cols.index(n) for n in names]
    X_out[:, idx] = fill[idx]
    return X_out


def score_all(X_in):
    recon = ae_model.predict(X_in, verbose=0)
    raw = {
        "isolation_forest": if_model.decision_function(X_in),
        "autoencoder": np.mean(np.square(X_in - recon), axis=1),
        "gmm": gmm_model.score_samples(X_in),
    }
    norm = normalizer.transform(raw)
    norm["fused"] = fuse(norm)
    return norm


GROUP_D = ["Magnitue", "Radius", "Covariance", "Variance"]
SCENARIOS = [("baseline", []), ("core", CORE), ("wide", WIDE),
             ("minimal", WIDE + GROUP_D)]
SUBSETS = {                      # fusion variants, same equal-weight rule
    "all3": ["isolation_forest", "autoencoder", "gmm"],
    "AE+GMM": ["autoencoder", "gmm"],
    "AE+IF": ["autoencoder", "isolation_forest"],
}

print(f"Evaluating on {len(y):,} rows ({int((y == 0).sum()):,} benign, "
      f"{int((y == 1).sum()):,} attack). Threshold = {LOW_MEDIUM_THRESHOLD}\n")
print(f"{'scenario':<10}{'IF':>7}{'AE':>7}{'GMM':>7}"
      f"{'all3':>8}{'AE+GMM':>9}{'AE+IF':>8}{'recall':>9}{'benignFPR':>11}")

for label, names in SCENARIOS:
    scores = score_all(neutralize(X, names) if names else X)
    single = {m: roc_auc_score(y, scores[m])
              for m in ("isolation_forest", "autoencoder", "gmm")}
    fused_auc = {k: roc_auc_score(y, fuse({m: scores[m] for m in ms}))
                 for k, ms in SUBSETS.items()}
    flagged = scores["fused"] >= LOW_MEDIUM_THRESHOLD   # all-three fusion
    print(f"{label:<10}{single['isolation_forest']:>7.3f}{single['autoencoder']:>7.3f}"
          f"{single['gmm']:>7.3f}{fused_auc['all3']:>8.3f}{fused_auc['AE+GMM']:>9.3f}"
          f"{fused_auc['AE+IF']:>8.3f}{flagged[y == 1].mean():>9.3f}"
          f"{flagged[y == 0].mean():>11.3f}")