"""
ml/train_gmm.py

Trains the third and final ensemble model. Gaussian Mixture Model
fits K multivariate Gaussian "blobs" to the benign training data --
together they approximate the shape of what normal traffic looks like
in this 46-dimensional feature space. At inference, score_samples()
gives the log-likelihood of a new point under that learned
distribution: high likelihood = looks like normal traffic, low
likelihood = doesn't fit any of the learned blobs = anomalous.

Trained benign-only, like every model in this ensemble -- see
docs/DECISIONS.md #1 for why that's required, not optional.

--- Two design decisions worth understanding, not just running ---

1. covariance_type='diag', not 'full'.
   'full' lets each component model correlations BETWEEN every pair of
   features (46x47/2 = 1,081 parameters PER component). Tried it first
   -- it's expensive (70s+ at just 20 components) and, worse, directly
   fights the reason GMM was chosen over One-Class SVM in the first
   place: fast per-row inference for the live WebSocket loop. 'diag'
   only models each feature's own variance (46 parameters/component)
   -- far cheaper to fit and score, and a standard simplification for
   GMM anomaly detection at this dimensionality.

2. n_components picked by BIC "elbow," not the raw minimum.
   Checked: BIC keeps decreasing all the way out to 30 components,
   never flattening. That's expected for network-traffic features
   (packet rates, IATs, byte counts) which are heavy-tailed, not
   actually shaped like a mixture of Gaussians -- so BIC just keeps
   rewarding more components for chasing that non-Gaussian shape.
   Blindly taking the minimum of whatever range you searched isn't a
   real answer, it's just "whatever the largest K I tried was."
   Instead: fit a spread of component counts, find where the
   per-component BIC improvement drops off sharply (the "elbow"), and
   use that -- balances fit quality against overfitting risk (only
   ~130K training rows) and inference speed.

Run from ml/:
    python train_gmm.py

Input:  data/processed/X_train_benign.npy  (from preprocessing.py)
Output: ml/saved_models/gmm.joblib
"""

import time

import joblib
import numpy as np
from sklearn.mixture import GaussianMixture

from config import PROCESSED_DIR, SAVED_MODELS_DIR, SEED

N_COMPONENTS_CANDIDATES = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
COVARIANCE_TYPE = "diag"

# Default reg_covar (1e-6) can be numerically unstable with correlated
# features (Tot sum, Min, Max, AVG, Std are all related). Bumping it
# adds a small floor to every variance estimate -- standard stability
# fix, costs a little sharpness in the density estimate.
REG_COVAR = 1e-3

# sklearn's EM optimization only runs ONCE per fit by default (n_init=1),
# starting from a single random initialization -- it can converge to a
# different local optimum depending on tiny floating-point differences
# between machines/platforms (observed in practice: BIC was
# non-monotonic at n_components=8 on Windows vs. Linux in testing).
# n_init=5 reruns EM from 5 different starting points per component
# count and keeps whichever converged best -- costs more compute time,
# buys real reproducibility across the team's different machines.
N_INIT = 5

# How aggressively to stop adding components: a candidate's
# per-component BIC improvement must be at least this fraction of the
# single biggest improvement seen (which is always the 1->2 jump) to
# still count as "worth it." Lower = keeps more components.
ELBOW_THRESHOLD_FRACTION = 0.15


def select_best_n_components(X_train):
    print(f"Fitting candidates={N_COMPONENTS_CANDIDATES} (covariance_type='diag')...")
    fitted = []
    for k in N_COMPONENTS_CANDIDATES:
        t0 = time.time()
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=COVARIANCE_TYPE,
            reg_covar=REG_COVAR,
            random_state=SEED,
            max_iter=200,
            n_init=N_INIT,
        )
        gmm.fit(X_train)
        bic = gmm.bic(X_train)
        fitted.append((k, bic, gmm))
        print(f"    n_components={k:>2}  BIC={bic:>14,.1f}  "
              f"converged={gmm.converged_}  ({time.time() - t0:.1f}s)")

    # Per-component marginal improvement between consecutive candidates.
    deltas = []
    for (k0, bic0, _), (k1, bic1, _) in zip(fitted, fitted[1:]):
        per_component_gain = (bic0 - bic1) / (k1 - k0)  # positive = BIC improved
        deltas.append(per_component_gain)

    max_gain = max(deltas)
    threshold = ELBOW_THRESHOLD_FRACTION * max_gain

    print(f"\nLargest single per-component BIC gain: {max_gain:,.1f} "
          f"(elbow threshold = {threshold:,.1f}, {ELBOW_THRESHOLD_FRACTION:.0%} of that)")

    # Walk forward; the elbow is the LAST candidate before gains drop
    # below threshold.
    elbow_idx = 0
    for i, gain in enumerate(deltas):
        if gain >= threshold:
            elbow_idx = i + 1  # this candidate still justified itself
        else:
            break

    best_k, best_bic, best_gmm = fitted[elbow_idx]
    print(f"Elbow selected: n_components={best_k} (BIC={best_bic:,.1f}) -- "
          f"components beyond this gave < {ELBOW_THRESHOLD_FRACTION:.0%} of the "
          f"biggest per-component improvement seen.")
    return best_gmm, best_k


def main():
    X_train = np.load(PROCESSED_DIR / "X_train_benign.npy")
    print(f"Loaded benign training matrix: {X_train.shape}")

    model, best_k = select_best_n_components(X_train)

    train_scores = model.score_samples(X_train)
    print(f"\nSanity check -- score_samples() (log-likelihood) on the "
          f"training data it was fit on:")
    print(f"    min={train_scores.min():.2f}  "
          f"max={train_scores.max():.2f}  "
          f"mean={train_scores.mean():.2f}  "
          f"std={train_scores.std():.2f}")

    out_path = SAVED_MODELS_DIR / "gmm.joblib"
    joblib.dump(model, out_path)
    print(f"\nSaved model to {out_path}")
    print(
        "\nReminder for ensemble/score_normalization.py: this model's "
        "score_samples() runs 'higher = normal' -- same direction as "
        "Isolation Forest, NOT the same as the Autoencoder. It needs a "
        "sign flip (anomaly_score = -score_samples(x)) before fusion. "
        "See docs/ARCHITECTURE.md section 3, step 1 / docs/DECISIONS.md #9."
    )


if __name__ == "__main__":
    main()