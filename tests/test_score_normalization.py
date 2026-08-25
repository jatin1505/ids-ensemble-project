"""
tests/test_score_normalization.py

Standalone checks for ensemble/score_normalization.py using SYNTHETIC scores
shaped like what we expect from the three real models. This lets us verify
the normalization MATH before Member A's trained models exist -- we don't
need a real Isolation Forest / Autoencoder / GMM to check that a sign flip
and a percentile clip behave the way we designed them to.

Run from the project root:
    PYTHONPATH=. python tests/test_score_normalization.py
    PYTHONPATH=. pytest tests/
"""
import tempfile
from pathlib import Path

import numpy as np

from ensemble.score_normalization import ScoreNormalizer, SCORE_DIRECTION


def make_fake_validation_scores(n: int = 1000, seed: int = 42) -> dict[str, np.ndarray]:
    """
    Mimics the SHAPE we expect from real scores, not real values:
      - isolation_forest: roughly symmetric around 0 (scikit-learn's
        decision_function), LOWER = more anomalous
      - gmm: log-likelihood under the fitted mixture -- mostly moderately
        negative with a long LEFT tail (a few points the model finds wildly
        unlikely), LOWER = more anomalous
      - autoencoder: non-negative, right-skewed reconstruction error, with
        a handful of extreme outliers -- this is exactly what makes the
        percentile clipping matter
    """
    rng = np.random.default_rng(seed)
    isolation_forest = rng.normal(loc=0.0, scale=0.15, size=n)
    gmm = -rng.gamma(shape=2.0, scale=1.5, size=n)
    gmm[:5] -= 30  # a few points the mixture considers wildly unlikely
    autoencoder = rng.gamma(shape=2.0, scale=0.5, size=n)
    autoencoder[:5] *= 20  # a few freak outliers, like a real error distribution
    return {"isolation_forest": isolation_forest, "gmm": gmm, "autoencoder": autoencoder}


def test_sign_flip_is_correct_for_all_lower_is_anomalous_models():
    """For EVERY model marked 'lower_is_anomalous' in SCORE_DIRECTION (today:
    isolation_forest and gmm), its MOST NEGATIVE raw score must end up with a
    HIGHER normalized score than its most positive one. Parametrized over
    SCORE_DIRECTION instead of hardcoding isolation_forest, so this already
    covers gmm after the LSTM swap, and covers any future model added the
    same way -- without editing this test again."""
    val_scores = make_fake_validation_scores()
    normalizer = ScoreNormalizer().fit(val_scores)

    lower_is_anomalous_models = [
        name for name, direction in SCORE_DIRECTION.items()
        if direction == "lower_is_anomalous"
    ]
    assert lower_is_anomalous_models, "expected at least one lower_is_anomalous model"

    for model_name in lower_is_anomalous_models:
        most_anomalous_raw = val_scores[model_name].min()
        most_normal_raw = val_scores[model_name].max()

        anomalous_norm = normalizer.transform({model_name: most_anomalous_raw})
        normal_norm = normalizer.transform({model_name: most_normal_raw})

        assert anomalous_norm[model_name] > normal_norm[model_name], (
            f"{model_name}: sign flip failed or missing"
        )


def test_output_always_in_zero_one():
    """Live traffic WILL sometimes be more extreme than anything seen during
    validation. Those cases must clip to exactly 0 or 1, not error out or
    escape the range -- risk_engine.py's weighted sum assumes every input
    is in [0, 1]."""
    val_scores = make_fake_validation_scores()
    normalizer = ScoreNormalizer().fit(val_scores)

    extreme = {"isolation_forest": -999.0, "gmm": -999.0, "autoencoder": 999.0}
    out = normalizer.transform(extreme)
    for model_name, score in out.items():
        assert 0.0 <= score <= 1.0, f"{model_name} produced {score}, outside [0,1]"


def test_save_load_roundtrip():
    """This is what makes 'fit offline once, load frozen at inference time'
    actually work. If save/load don't round-trip exactly, live scores would
    silently drift from what evaluate.py measured."""
    val_scores = make_fake_validation_scores()
    normalizer = ScoreNormalizer().fit(val_scores)

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "norm_bounds.json"
        normalizer.save(path)
        reloaded = ScoreNormalizer.load(path)

    sample = {"isolation_forest": 0.1, "gmm": -3.0, "autoencoder": 0.8}
    assert normalizer.transform(sample) == reloaded.transform(sample)


if __name__ == "__main__":
    test_sign_flip_is_correct_for_all_lower_is_anomalous_models()
    test_output_always_in_zero_one()
    test_save_load_roundtrip()
    print("All score_normalization checks passed.")