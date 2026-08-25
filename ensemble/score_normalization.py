"""
ensemble/score_normalization.py

Purpose
-------
The three models produce anomaly scores on different scales, and TWO of
the three run in the OPPOSITE direction from the third:

    isolation_forest  - scikit-learn decision_function(): LOWER = more anomalous
    gmm                 - log-likelihood under the fitted "normal traffic"
                          mixture: LOWER = more anomalous (a point the
                          mixture finds unlikely gets a very negative score
                          -- same direction as isolation_forest)
    autoencoder        - reconstruction error:             HIGHER = more anomalous

Before risk_engine.py can combine these three numbers into one fused score,
they need to be on the same [0, 1] scale, pointing the same direction. That
is the ONLY job of this file. It does not decide risk levels -- that is
risk_engine.py's job.

This is the same problem addressed in the outlier-ensemble literature --
see Kriegel, Kröger, Schubert & Zimek, "Interpreting and Unifying Outlier
Scores" (SDM 2011) -- worth citing in the report if asked why normalization
is a real, documented step and not something we invented ad hoc.

Note (team decision): swapped from LSTM to GMM after EDA found CICIoT2023's
CSV export has no IP/timestamp/session ID to build sequence windows from --
see docs/DECISIONS.md. Nothing else in this file needed to change for the
swap, which is exactly why SCORE_DIRECTION below is a per-model config dict
instead of hardcoded logic.

Design: fit / transform, mirroring ml/preprocessing.py's scaler
-----------------------------------------------------------------
fit() learns per-model bounds from the offline validation set. transform()
applies those FROZEN bounds to new scores. This matters for the same reason
it mattered for the feature scaler: if we recomputed bounds from whatever
scores happen to show up during the live demo, the same flow could get a
different normalized score -- and therefore a different risk level --
depending purely on what else was replayed around it. Freezing the bounds
after fitting is what makes scoring reproducible between evaluation and the
live demo.

We fit on the [1st, 99th] percentile rather than the true min/max.
Reconstruction and prediction errors are typically right-skewed with
occasional extreme values; scaling off the true min/max means a single
freak validation example can compress every other score into a sliver near
0. Clipping the fitting range is a light, easy-to-defend fix for that (a
form of Winsorizing). It's a starting default, not a fixed rule -- revisit
once we see Member A's real score distributions.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Literal
import numpy as np

from shared.schemas import ModelName

# The single place that encodes the Isolation Forest sign convention.
# "lower_is_anomalous" models get negated before scaling; everything else
# is used as-is. A 4th model later just means adding one line here.
SCORE_DIRECTION: dict[ModelName, Literal["higher_is_anomalous", "lower_is_anomalous"]] = {
    "isolation_forest": "lower_is_anomalous",
    "gmm": "lower_is_anomalous",
    "autoencoder": "higher_is_anomalous",
}

CLIP_LOWER_PERCENTILE = 1.0
CLIP_UPPER_PERCENTILE = 99.0


class ScoreNormalizer:
    """Learns per-model (lo, hi) bounds on a validation set, then maps any
    future raw score for that model into [0, 1], where 1 always means
    'more anomalous', regardless of which model produced it."""

    def __init__(self) -> None:
        self._bounds: dict[str, tuple[float, float]] = {}
        self._fitted = False

    def fit(self, validation_scores: dict[str, np.ndarray]) -> "ScoreNormalizer":
        """
        validation_scores: {"isolation_forest": array([...]), "autoencoder": array([...]), "lstm": array([...])}
        Raw scores from the mixed benign+attack validation set -- never the
        training set, and never the held-out test set.
        """
        for model_name, scores in validation_scores.items():
            oriented = self._orient(model_name, np.asarray(scores, dtype=float))
            lo = float(np.percentile(oriented, CLIP_LOWER_PERCENTILE))
            hi = float(np.percentile(oriented, CLIP_UPPER_PERCENTILE))
            if hi <= lo:  # degenerate/near-constant scores -- avoid divide-by-zero later
                hi = lo + 1e-9
            self._bounds[model_name] = (lo, hi)
        self._fitted = True
        return self

    def transform(self, raw_scores: dict[str, float]) -> dict[str, float]:
        """Works for a single flow's scores (plain floats) or a whole batch
        (numpy arrays) -- the arithmetic is identical either way."""
        if not self._fitted:
            raise RuntimeError(
                "ScoreNormalizer has no bounds yet. Call fit() during offline "
                "evaluation, or load() a saved file during live inference."
            )
        normalized: dict[str, float] = {}
        for model_name, raw in raw_scores.items():
            lo, hi = self._bounds[model_name]
            oriented = self._orient(model_name, raw)
            scaled = (oriented - lo) / (hi - lo)
            clipped = np.clip(scaled, 0.0, 1.0)
            normalized[model_name] = clipped if isinstance(raw, np.ndarray) else float(clipped)
        return normalized

    def _orient(self, model_name: str, value):
        """Flip sign for models where LOWER raw score = MORE anomalous, so
        that after this step, higher always means more anomalous."""
        return -value if SCORE_DIRECTION[model_name] == "lower_is_anomalous" else value

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self._bounds))

    @classmethod
    def load(cls, path: str | Path) -> "ScoreNormalizer":
        """Used by the live pipeline: load bounds fit offline, never re-fit."""
        obj = cls()
        obj._bounds = {k: tuple(v) for k, v in json.loads(Path(path).read_text()).items()}
        obj._fitted = True
        return obj