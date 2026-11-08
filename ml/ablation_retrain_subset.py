"""
ml/ablation_retrain_subset.py

Feasibility spike, step 4. Step 3 neutralized features at scoring time, which is
pessimistic for models trained WITH those features. The fair test is to retrain
on only the features a live extractor can produce, and compare against the
step-3 baseline. Nothing is saved; ml/saved_models is untouched.

Subsetting scaled columns is valid without refitting: StandardScaler works per
column, so each kept column's scaling is unchanged.

Run from project root:   python ml/ablation_retrain_subset.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import joblib
import numpy as np
import tensorflow as tf
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from tensorflow import keras

from config import BENIGN_LABEL, PROCESSED_DIR, SAVED_MODELS_DIR, SEED
from ensemble.risk_engine import fuse
from ensemble.score_normalization import ScoreNormalizer
# Reuse the real training settings so the only thing that differs is the features
from train_autoencoder import (BATCH_SIZE, EARLY_STOPPING_PATIENCE,
                               INTERNAL_VAL_FRACTION, MAX_EPOCHS, build_autoencoder)
from train_gmm import COVARIANCE_TYPE, N_INIT, REG_COVAR

CONSTANTS = ["Drate", "ece_flag_number", "cwr_flag_number",
             "Telnet", "SMTP", "SSH", "IRC", "DHCP"]
UNREPRODUCIBLE = ["IAT", "Weight", "Header_Length", "urg_count", "rst_count",
                  "ack_count", "syn_count", "fin_count", "flow_duration",
                  "Number", "Rate", "Srate"]
GROUP_D = ["Magnitue", "Radius", "Covariance", "Variance"]

SCENARIOS = [
    ("25 feat (no Covariance)", CONSTANTS + UNREPRODUCIBLE + ["Covariance"]),
    ("25 feat (no Variance)", CONSTANTS + UNREPRODUCIBLE + ["Variance"]),
]

feature_cols = json.loads((SAVED_MODELS_DIR / "feature_columns.json").read_text())
X_train = np.load(PROCESSED_DIR / "X_train_benign.npy")
X_val = np.load(PROCESSED_DIR / "X_val.npy")
y_val = np.load(PROCESSED_DIR / "y_val.npy", allow_pickle=True)
is_attack = (y_val != BENIGN_LABEL).astype(int)

# Same evaluation rows as step 3 (same seed, same procedure)
rng = np.random.default_rng(SEED)
benign_idx = np.where(is_attack == 0)[0]
attack_idx = rng.choice(np.where(is_attack == 1)[0], size=100_000, replace=False)
keep = np.concatenate([benign_idx, attack_idx])
X_eval, y = X_val[keep], is_attack[keep]

# Reuse the component count the real GMM selected (BIC elbow) instead of re-running it
K = joblib.load(SAVED_MODELS_DIR / "gmm.joblib").n_components


def run_scenario(name, drop):
    cols = [i for i, c in enumerate(feature_cols) if c not in drop]
    Xtr, Xev = X_train[:, cols], X_eval[:, cols]
    print(f"\n[{name}] training on {Xtr.shape[1]} features...")

    iso = IsolationForest(n_estimators=200, contamination="auto",
                          random_state=SEED, n_jobs=-1).fit(Xtr)

    tf.random.set_seed(SEED)
    fit, hold = train_test_split(Xtr, test_size=INTERNAL_VAL_FRACTION, random_state=SEED)
    ae = build_autoencoder(Xtr.shape[1])
    ae.fit(fit, fit, validation_data=(hold, hold), epochs=MAX_EPOCHS,
           batch_size=BATCH_SIZE, verbose=0,
           callbacks=[keras.callbacks.EarlyStopping(
               monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
               restore_best_weights=True)])

    gmm = GaussianMixture(n_components=K, covariance_type=COVARIANCE_TYPE,
                          reg_covar=REG_COVAR, random_state=SEED,
                          max_iter=200, n_init=N_INIT).fit(Xtr)

    raw = {"isolation_forest": iso.decision_function(Xev),
           "autoencoder": np.mean(np.square(Xev - ae.predict(Xev, verbose=0)), axis=1),
           "gmm": gmm.score_samples(Xev)}
    norm = ScoreNormalizer().fit(raw).transform(raw)   # sign-flips + scales, as in production

    auc = {m: roc_auc_score(y, s) for m, s in norm.items()}
    auc["all3"] = roc_auc_score(y, fuse(norm))
    auc["AE+GMM"] = roc_auc_score(y, fuse({m: norm[m] for m in ("autoencoder", "gmm")}))
    return auc


print(f"Eval rows: {len(y):,}   GMM components reused from saved model: K={K}")
print("Step-3 reference (saved models, neutralized 'wide'): "
      "IF .921  AE .990  GMM .987  all3 .950  AE+GMM .987")
results = [(name, run_scenario(name, drop)) for name, drop in SCENARIOS]

print(f"\n{'scenario':<26}{'IF':>7}{'AE':>7}{'GMM':>7}{'all3':>8}{'AE+GMM':>9}")
for name, a in results:
    print(f"{name:<26}{a['isolation_forest']:>7.3f}{a['autoencoder']:>7.3f}"
          f"{a['gmm']:>7.3f}{a['all3']:>8.3f}{a['AE+GMM']:>9.3f}")