"""
Replay engine.

Reads a sample of real CICIoT2023 flows from disk and streams them
over the WebSocket at a fixed interval, standing in for a live feed.
This is offline data being replayed on a timer -- it is NOT live
packet capture, and the report/demo should always describe it that
way.

Why there's no src_ip/dst_ip here: CICIoT2023's CSV format has no IP,
port, timestamp, or session ID at all -- confirmed both by inspecting
the actual file and by checking the original paper (the CSVs are
packet stats summarized over fixed-size windows during export, which
is what drops that info). That's also why there's no per-source
buffering anywhere in this file: with no way to group flows by device,
there's nothing to buffer by. It's the same constraint that forced the
LSTM -> GMM swap -- see shared/schemas.py and docs/DECISIONS.md.

UPDATED: risk_level, final_score, and model_breakdown are no longer
random -- every flow now goes through backend/model_runtime.py's real
Isolation Forest + Autoencoder + GMM + risk engine. attack_category is
now populated too, straight from the CSV's ground-truth label column
(including "BenignTraffic") -- this is real data we already have
access to because we control the replay, not something the models
predicted. See shared/schemas.py's field comment: it must never be
read as a model output.
"""

import asyncio
import time
from pathlib import Path

import pandas as pd

from backend import model_runtime
from backend.websocket_manager import manager

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "CICIOT23" / "validation" / "validation.csv"
SAMPLE_SIZE = 500              # rows kept in memory and cycled for the demo
REPLAY_INTERVAL_SECONDS = 2
RANDOM_SEED = 42                # reproducible sample -- same rows every run

# One-hot-style indicator columns in the raw CSV (0.0/1.0 per row), not
# a single clean "protocol" field -- CICIoT2023 doesn't include IP
# addresses or ports at all, so this is the most specific "what kind of
# traffic was this" info actually available to show on the dashboard.
# Confirmed reliable against the real data (every row in a 500-row test
# sample resolved to a value), so this is populated on every event
# despite the field being Optional in the schema.
PROTOCOL_COLUMNS = ["TCP", "UDP", "ICMP", "ARP"]

LABEL_COL = "label"  # matches ml/config.py's LABEL_COL


def _derive_protocol(row: pd.Series) -> str:
    for col in PROTOCOL_COLUMNS:
        if row.get(col, 0.0) == 1.0:
            return col
    return "Other"


def _load_replay_sample() -> pd.DataFrame:
    # Loading the full file once at startup and sampling in memory,
    # rather than streaming row-by-row from disk, is deliberately the
    # simple option here -- 500 rows is plenty of variety for a live
    # demo, and a fixed random_state means every teammate who runs this
    # gets the same sample.
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Replay dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if len(df) < SAMPLE_SIZE:
        raise ValueError(f"Replay dataset has {len(df)} rows; expected at least {SAMPLE_SIZE}.")
    return df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)


async def run_replay():
    model_runtime.load_runtime()  # once, before the loop starts
    sample = _load_replay_sample()
    print(f"[replay] loaded {len(sample)} rows from {DATA_PATH}")

    i = 0
    while True:
        row = sample.iloc[i % len(sample)]

        event = model_runtime.score_flow(
            row,
            flow_id=f"flow-{i}",
            timestamp=time.time(),
            protocol=_derive_protocol(row),
            attack_category=row[LABEL_COL],
        )

        print(f"[replay] flow-{i} true_label={row[LABEL_COL]!r} -> "
              f"{event.risk_level} (score={event.final_score})")
        await manager.broadcast(event)
        i += 1
        await asyncio.sleep(REPLAY_INTERVAL_SECONDS)
