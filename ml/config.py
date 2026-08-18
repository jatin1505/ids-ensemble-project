"""
Shared config for every script in ml/ and ensemble/.

Import from here instead of hardcoding paths/seeds in each script --
when the data location changes (e.g. deploying to a different machine),
this is the one place to fix it.
"""

from pathlib import Path

# ---- Reproducibility -------------------------------------------------
SEED = 42

# ---- Paths -------------------------------------------------------------
# Adjust ROOT if your folder layout differs -- everything else is relative to it.
ROOT = Path(__file__).resolve().parent.parent  # ids-ensemble-project/

DATA_DIR = ROOT / "data" / "CICIOT23"
TRAIN_CSV = DATA_DIR / "train" / "train.csv"
TEST_CSV = DATA_DIR / "test" / "test.csv"
VAL_CSV = DATA_DIR / "validation" / "validation.csv"

PROCESSED_DIR = ROOT / "data" / "processed"
SAVED_MODELS_DIR = ROOT / "ml" / "saved_models"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---- Labels --------------------------------------------------------------
LABEL_COL = "label"
BENIGN_LABEL = "BenignTraffic"

# ---- Feature list ----------------------------------------------------
# All 46 numeric columns -- everything in the CSV except the label column.
# Populated dynamically in preprocessing.py from the actual CSV header
# (kept here as a single source of truth once confirmed, so training
# scripts don't each re-derive it).
FEATURE_COLUMNS: list[str] = []  # filled in by preprocessing.py on first run