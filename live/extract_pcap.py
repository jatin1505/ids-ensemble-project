"""
live/extract_pcap.py

Command-line tool: reads a lab capture and writes one CSV row per 10-packet
window, with the 46 feature columns (NaN written as an empty cell), the
window's start and end time, and, if you give it the phase log, the lab phase
the window belongs to. This is the input for the live range check and the live
evaluation.

Usage, from the repo root:

    python -m live.extract_pcap data/live/session1/session1_*.pcap data/live/session1/windows.csv \
        --phases data/live/session1/phases_*.csv

Options:
    --phases FILE   the phase log written by lab/run_phases.sh. Windows that lie
                    fully inside a phase get its name; every other window
                    (idle gaps, phase boundaries) is labeled "unlabeled".
    --margin SEC    a window must be at least this far from both edges of its
                    phase to be labeled (default 1.0 s; the lab runs on one
                    clock, so a small margin is enough).
    --window N      packets per window (default 10).
    --block N       packets per block of running statistics (default 20).
"""
from __future__ import annotations

import argparse
import bisect
import csv
import sys
from collections import Counter
from pathlib import Path

from live.feature_extractor import (
    BLOCK_SIZE, FEATURE_COLUMNS, WINDOW_SIZE, WindowExtractor, read_pcap,
)

HEADER = ["ts_start", "ts_end", "n_packets", "phase"] + FEATURE_COLUMNS


def load_phases(path: Path) -> list[tuple[float, float, str]]:
    """(start, end, name) for every phase, sorted by start time."""
    with open(path, newline="") as f:
        rows = [(float(r["start_epoch"]), float(r["end_epoch"]), r["phase"])
                for r in csv.DictReader(f)]
    return sorted(rows)


def label_window(ts_start: float, ts_end: float, phases, starts, margin: float) -> str:
    """The phase that contains the whole window, with a margin, else 'unlabeled'."""
    i = bisect.bisect_right(starts, ts_start) - 1
    if i >= 0:
        start, end, name = phases[i]
        if start + margin <= ts_start and ts_end <= end - margin:
            return name
    return "unlabeled"


def format_value(value: float) -> str:
    return "" if value != value else f"{value:.10g}"   # NaN -> empty cell


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("pcap", type=Path)
    parser.add_argument("out_csv", type=Path)
    parser.add_argument("--phases", type=Path)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--window", type=int, default=WINDOW_SIZE)
    parser.add_argument("--block", type=int, default=BLOCK_SIZE)
    args = parser.parse_args(argv)

    phases = load_phases(args.phases) if args.phases else []
    starts = [p[0] for p in phases]

    extractor = WindowExtractor(args.window, args.block)
    packets = 0
    per_phase: Counter = Counter()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        for ts, buf in read_pcap(args.pcap):
            packets += 1
            window = extractor.add(ts, buf)
            if window is None:
                continue
            phase = (label_window(window.ts_start, window.ts_end, phases, starts, args.margin)
                     if phases else "unlabeled")
            per_phase[phase] += 1
            writer.writerow(
                [f"{window.ts_start:.6f}", f"{window.ts_end:.6f}", window.n_packets, phase]
                + [format_value(v) for v in window.features.to_numpy()]
            )

    print(f"Read {packets:,} packets; skipped {extractor.skipped:,} that were not IPv4 or ARP.")
    print(f"Wrote {sum(per_phase.values()):,} windows of {args.window} packets to {args.out_csv}")
    if phases:
        print("\nWindows per phase:")
        for name in [p[2] for p in phases] + ["unlabeled"]:
            if name in per_phase:
                print(f"  {name:<14}{per_phase[name]:>9,}")
                per_phase.pop(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())