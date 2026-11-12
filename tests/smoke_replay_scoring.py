"""
tests/smoke_replay_scoring.py

Smoke test for the REPLAY scoring path on a new machine (for example after
moving the project to Linux). It loads the real frozen models through
backend/model_runtime.py and scores a handful of real validation rows, then
checks that what comes out is sane.

Why it exists: `python -m backend.model_runtime` only imports the module. It
prints nothing because nothing in it runs at import time (the models are
loaded later, by model_runtime.load_runtime(), when the replay engine
starts). So a silent import proves the imports resolve, but NOT that the
models, scaler and normalizer files are present and load. This script does.

It is deliberately NOT named test_*.py, so pytest does not collect it: it
loads TensorFlow and takes several seconds, which does not belong in the
fast unit-test run.

Run from the project root:
    python tests/smoke_replay_scoring.py

What "good" looks like:
  - the models load without a FileNotFoundError,
  - every row gets a final_score between 0 and 1 and a Low/Medium/High level,
  - model_breakdown has exactly isolation_forest, autoencoder and gmm,
  - a per-row scoring time of well under a second (the live loop needs this).
Some benign rows being flagged Medium or High is expected: at the current
threshold about 40% of benign validation rows are (docs/LIVE_CAPTURE_HANDOFF.md,
section 2.8). This script reports it but does not treat it as a failure.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import time

import pandas as pd

DATA_PATH = ROOT / "data" / "CICIOT23" / "validation" / "validation.csv"
LABEL_COL = "label"          # matches ml/config.py
BENIGN_LABEL = "BenignTraffic"
ROWS_READ = 100_000          # the file is shuffled, so the first rows hold many labels
N_BENIGN = 5
N_ATTACK_LABELS = 6

EXPECTED_MODELS = {"isolation_forest", "autoencoder", "gmm"}
VALID_LEVELS = {"Low", "Medium", "High"}


def pick_rows() -> pd.DataFrame:
    if not DATA_PATH.exists():
        sys.exit(f"Not found: {DATA_PATH}\nCopy validation.csv from your Windows copy "
                 f"(docs/LINUX_DEV_SETUP.md, section 3).")
    df = pd.read_csv(DATA_PATH, nrows=ROWS_READ)
    benign = df[df[LABEL_COL] == BENIGN_LABEL].head(N_BENIGN)
    attacks = df[df[LABEL_COL] != BENIGN_LABEL]
    # One row from each of the most common attack labels, for variety.
    top_labels = attacks[LABEL_COL].value_counts().index[:N_ATTACK_LABELS]
    attack_rows = pd.concat([attacks[attacks[LABEL_COL] == lab].head(1) for lab in top_labels])
    return pd.concat([benign, attack_rows]).reset_index(drop=True)


def main() -> None:
    sample = pick_rows()  # fails first, with a clear message, if the data file is missing

    from backend import model_runtime  # imported late: it pulls in TensorFlow, which is slow
    print(f"Scoring {len(sample)} rows from {DATA_PATH.name} "
          f"({N_BENIGN} benign + one row from each of {N_ATTACK_LABELS} attack labels)\n")

    print("Loading the models (model_runtime.load_runtime()) ...")
    t0 = time.time()
    model_runtime.load_runtime()
    print(f"  loaded in {time.time() - t0:.1f}s\n")

    problems = []
    flagged = {"benign": 0, "attack": 0}
    total = {"benign": 0, "attack": 0}
    times_ms = []

    print(f"{'true label':<26}{'level':<8}{'score':>7}   {'IF':>5}{'AE':>6}{'GMM':>6}   ms")
    for i, row in sample.iterrows():
        t = time.time()
        event = model_runtime.score_flow(
            row,
            flow_id=f"smoke-{i}",
            timestamp=time.time(),
            protocol="TCP",
            attack_category=row[LABEL_COL],
        )
        ms = (time.time() - t) * 1000
        times_ms.append(ms)

        breakdown = event.model_breakdown
        print(f"{str(row[LABEL_COL]):<26}{event.risk_level:<8}{event.final_score:>7.3f}   "
              f"{breakdown.get('isolation_forest', float('nan')):>5.2f}"
              f"{breakdown.get('autoencoder', float('nan')):>6.2f}"
              f"{breakdown.get('gmm', float('nan')):>6.2f}   {ms:.0f}")

        if not 0.0 <= event.final_score <= 1.0:
            problems.append(f"row {i}: final_score {event.final_score} is outside [0, 1]")
        if event.risk_level not in VALID_LEVELS:
            problems.append(f"row {i}: unexpected risk_level {event.risk_level!r}")
        if set(breakdown) != EXPECTED_MODELS:
            problems.append(f"row {i}: model_breakdown has {sorted(breakdown)}, "
                            f"expected {sorted(EXPECTED_MODELS)}")

        kind = "benign" if row[LABEL_COL] == BENIGN_LABEL else "attack"
        total[kind] += 1
        flagged[kind] += event.risk_level != "Low"

    print(f"\nFlagged Medium or High: benign {flagged['benign']}/{total['benign']}, "
          f"attack {flagged['attack']}/{total['attack']}")
    print(f"Scoring time per row: mean {sum(times_ms) / len(times_ms):.0f} ms, "
          f"max {max(times_ms):.0f} ms (the first row is usually the slowest)")

    print()
    if problems:
        print("PROBLEMS FOUND:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("OK: the replay scoring path loads and produces sane results.")


if __name__ == "__main__":
    main()
