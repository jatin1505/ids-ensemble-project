"""
ml/check_live_ranges.py

Task A5 of the live-capture extension (docs/LIVE_CAPTURE_HANDOFF.md): the live
range check, GO/NO-GO #2.

Question it answers: when the extractor turns our lab's NORMAL traffic into
feature rows, do the 25 computed features land inside the range the models saw
as benign during training? A feature far outside that range gets a huge z-score
after scaling, and the models will call the row anomalous whatever it is.

Input:   the windows CSV written by `python -m live.extract_pcap`
         (columns: ts_start, ts_end, n_packets, phase, then the 46 features)
         data/processed/X_train_benign.npy and ml/saved_models/scaler.joblib
         (the benign training data and the scaler fitted on it)

Run from the project root:

    python ml/check_live_ranges.py data/live/session1/windows.csv
    python ml/check_live_ranges.py data/live/session1/windows.csv --by-phase

What it prints, for each of the 25 computed features, using only the windows of
the two NORMAL phases (normal_light and normal_busy):
    benign p1 / median / p99   the range in the training data, in raw units
    live  p1 / median / p99    the same for our lab's normal traffic
    % out                      share of live windows outside the benign [p1, p99]
    z(median)                  the live median in the units the models see
                               (standard deviations from the benign mean)
    flag                       OUT if at least 50% of live windows are outside,
                               check if at least 10%, otherwise ok

HOW TO READ AN "OUT":
    It means one of two different things, and this script cannot tell which:
      (a) our lab traffic really is different from the IoT traffic the dataset
          was recorded from (domain shift, expected for some features: a web
          server's frame sizes and TTLs are not an IoT device's), or
      (b) the extractor does not reproduce the dataset's feature faithfully.
    Telling them apart needs the extractor to be checked on a dataset PCAP
    (docs/EXTRACTION_SPEC.md, section 6). Until then, an OUT is a finding to
    report, not yet a verdict.

With --by-phase it also prints the live median of every feature in every phase
(floods, scan, normal), which shows how different the attack phases look.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd

from config import PROCESSED_DIR, SAVED_MODELS_DIR
from live.feature_extractor import COMPUTED, FEATURE_COLUMNS, load_feature_columns

NORMAL_PHASES = ["normal_light", "normal_busy"]
FLAG_OUT = 0.5      # at least this share of live windows outside the benign range: OUT
FLAG_CHECK = 0.1    # at least this share: check
TOLERANCE = 1e-6    # float noise from un-scaling (values like -3.5e-13 are really 0)


def benign_ranges():
    """Raw-unit p1, median, p99 of every feature in the benign training data, plus the scaler."""
    load_feature_columns(SAVED_MODELS_DIR / "feature_columns.json")   # raises if the order differs
    scaler = joblib.load(SAVED_MODELS_DIR / "scaler.joblib")
    X_scaled = np.load(PROCESSED_DIR / "X_train_benign.npy")
    raw = scaler.inverse_transform(X_scaled.astype("float64"))
    p1, p50, p99 = np.percentile(raw, [1, 50, 99], axis=0)
    return scaler, p1, p50, p99


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("windows_csv", type=Path)
    parser.add_argument("--by-phase", action="store_true",
                        help="also print the live median of every feature in every phase")
    args = parser.parse_args()

    windows = pd.read_csv(args.windows_csv)
    windows = windows[windows["phase"] != "unlabeled"]
    normal = windows[windows["phase"].isin(NORMAL_PHASES)]
    if normal.empty:
        sys.exit(f"No windows from {NORMAL_PHASES} in {args.windows_csv}. "
                 f"Did you pass --phases to live.extract_pcap?")

    scaler, p1, p50, p99 = benign_ranges()
    print(f"Live windows from the normal phases: {len(normal):,} "
          f"({', '.join(f'{p}: {(normal.phase == p).sum():,}' for p in NORMAL_PHASES)})")
    print(f"Benign training rows: {np.load(PROCESSED_DIR / 'X_train_benign.npy', mmap_mode='r').shape[0]:,}\n")

    print(f"{'feature':<18}{'benign p1':>10}{'median':>10}{'p99':>10}   "
          f"{'live p1':>10}{'median':>10}{'p99':>10}   {'% out':>6}{'z(med)':>9}  flag")
    out_features, check_features = [], []
    for name in COMPUTED:
        i = FEATURE_COLUMNS.index(name)
        live = normal[name].to_numpy(dtype="float64")
        share_out = float(np.mean((live < p1[i] - TOLERANCE) | (live > p99[i] + TOLERANCE)))
        l1, l50, l99 = np.percentile(live, [1, 50, 99])
        z = (l50 - scaler.mean_[i]) / scaler.scale_[i]
        flag = "OUT" if share_out >= FLAG_OUT else "check" if share_out >= FLAG_CHECK else "ok"
        (out_features if flag == "OUT" else check_features if flag == "check" else []).append(name)
        print(f"{name:<18}{p1[i]:>10.4g}{p50[i]:>10.4g}{p99[i]:>10.4g}   "
              f"{l1:>10.4g}{l50:>10.4g}{l99:>10.4g}   {share_out:>6.0%}{z:>9.1f}  {flag}")

    print(f"\nOUT ({len(out_features)}): {', '.join(out_features) or 'none'}")
    print(f"check ({len(check_features)}): {', '.join(check_features) or 'none'}")
    print(f"ok ({len(COMPUTED) - len(out_features) - len(check_features)} of {len(COMPUTED)})")
    print("\nAn OUT is either real domain shift (our web-server traffic is not IoT traffic) or an extractor\n"
          "difference. Only a dataset PCAP check can separate the two; see the docstring at the top of this file.")

    if args.by_phase:
        order = [p for p in ["normal_light", "normal_busy", "http_flood", "syn_flood",
                             "udp_flood", "icmp_flood", "port_scan"] if p in set(windows["phase"])]
        order += sorted(set(windows["phase"]) - set(order))
        medians = windows.groupby("phase")[COMPUTED].median().loc[order].T
        print("\nLive median of every computed feature, by phase:")
        print(medians.to_string(float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
