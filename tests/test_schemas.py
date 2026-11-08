"""
tests/test_schemas.py

Covers the `source` field added to RiskEvent for the live-capture extension
(docs/LIVE_CAPTURE_HANDOFF.md, decision D5). The point of the field is that a
replayed event can never be mistaken for a live one, so these tests pin down:
the safe default, the JSON the dashboard receives, and rejection of anything
that is not "replay" or "live".

Run from the project root:
    PYTHONPATH=. pytest tests/test_schemas.py
"""
import json

import pytest
from pydantic import ValidationError

from shared.schemas import RiskEvent


def _event(**overrides):
    fields = dict(
        flow_id="flow-1",
        timestamp=1000.0,
        risk_level="Low",
        final_score=0.1,
        model_breakdown={"isolation_forest": 0.1, "autoencoder": 0.1, "gmm": 0.1},
    )
    fields.update(overrides)
    return RiskEvent(**fields)


def test_source_defaults_to_replay():
    # Every existing producer (replay_engine.py, risk_engine.py) builds events
    # without a source and must keep working, labeled as replay.
    assert _event().source == "replay"


def test_live_source_survives_json_round_trip():
    event = _event(source="live")
    restored = RiskEvent.model_validate_json(event.model_dump_json())
    assert restored.source == "live"


def test_source_is_present_in_the_json_sent_to_the_dashboard():
    payload = json.loads(_event(source="live").model_dump_json())
    assert payload["source"] == "live"
    assert json.loads(_event().model_dump_json())["source"] == "replay"


def test_unknown_source_is_rejected():
    with pytest.raises(ValidationError):
        _event(source="simulated")