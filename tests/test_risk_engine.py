"""
tests/test_risk_engine.py

Covers ensemble/risk_engine.py -- the fusion + threshold logic. Doesn't
touch score_normalization.py's own correctness (test_score_normalization.py
already covers that); these tests assume normalized [0,1] scores are
already correct and check what happens after that point.
"""

import pytest

from ensemble.risk_engine import (
    LOW_MEDIUM_THRESHOLD,
    MEDIUM_HIGH_THRESHOLD,
    MODEL_WEIGHTS,
    build_risk_event,
    classify,
    fuse,
)


def test_equal_weights_average_correctly():
    scores = {"isolation_forest": 0.6, "autoencoder": 0.6, "gmm": 0.6}
    assert fuse(scores) == pytest.approx(0.6)


def test_all_high_scores_fuse_high():
    scores = {"isolation_forest": 0.95, "autoencoder": 0.9, "gmm": 0.99}
    assert fuse(scores) > 0.9


def test_all_low_scores_fuse_low():
    scores = {"isolation_forest": 0.05, "autoencoder": 0.1, "gmm": 0.02}
    assert fuse(scores) < 0.1


def test_missing_model_renormalizes_by_available_weight():
    # Two models both say 0.9, one is missing entirely (not "0", just
    # absent from the dict -- e.g. a model failed to load). The fused
    # score should still reflect ~0.9, not be diluted toward ~0.6 as if
    # the missing model had silently scored 0.
    partial = {"isolation_forest": 0.9, "autoencoder": 0.9}
    assert fuse(partial) == pytest.approx(0.9, abs=1e-6)


def test_unrecognized_model_names_raise():
    with pytest.raises(ValueError):
        fuse({"some_future_model": 0.5})


# UPDATED 2026-08-30: LOW_MEDIUM_THRESHOLD was recalibrated from 0.4 to
# 0.05 against real validation-set data -- see docs/DECISIONS.md #12.
# Per this test file's own original docstring: "if AGL retunes
# LOW_MEDIUM_THRESHOLD or MEDIUM_HIGH_THRESHOLD, this test needs
# updating to match, and that's expected, not a sign the test caught a
# regression." This locks in boundary BEHAVIOR (which side of >= a
# boundary value lands on) against the current real values, not the
# old placeholders.
@pytest.mark.parametrize(
    "score,expected",
    [
        (0.0, "Low"),
        (0.049, "Low"),
        (0.05, "Medium"),   # exactly at the (new) low/medium boundary
        (0.69, "Medium"),
        (0.7, "High"),      # medium/high boundary -- still unvalidated, see risk_engine.py TODO
        (1.0, "High"),
    ],
)
def test_classify_boundaries(score, expected):
    assert classify(score) == expected


def test_thresholds_match_documented_calibration():
    # Guards against LOW_MEDIUM_THRESHOLD silently drifting back to an
    # old placeholder value in a future edit without docs/DECISIONS.md
    # and this test both being updated together.
    assert LOW_MEDIUM_THRESHOLD == pytest.approx(0.05)
    assert MEDIUM_HIGH_THRESHOLD == pytest.approx(0.7)


def test_build_risk_event_end_to_end():
    event = build_risk_event(
        flow_id="flow-1",
        timestamp=1000.0,
        normalized_scores={"isolation_forest": 0.9, "autoencoder": 0.9, "gmm": 0.9},
        protocol="TCP",
        attack_category="DDoS-SYN_Flood",
    )
    assert event.risk_level == "High"
    assert event.flow_id == "flow-1"
    assert event.protocol == "TCP"
    assert event.attack_category == "DDoS-SYN_Flood"
    assert event.src_ip is None  # not populated -- CICIoT2023 has no real IPs


def test_weights_sum_to_one():
    # Not a hard requirement (fuse() re-normalizes regardless), but if
    # this ever drifts, it's worth knowing on purpose, not by accident.
    assert sum(MODEL_WEIGHTS.values()) == pytest.approx(1.0)