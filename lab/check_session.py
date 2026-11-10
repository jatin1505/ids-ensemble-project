#!/usr/bin/env python3
"""
lab/check_session.py

Checks one lab capture session (docs/LAB_SETUP.md): did every phase run, did
it put packets in the capture, and do the packet sizes look like a normal
Ethernet network? It needs only tcpdump and the Python standard library, and
it does not need sudo (the capture files belong to you).

Usage:
    python3 lab/check_session.py <capture.pcap> <phases.csv>

Example, after a dry run:
    python3 lab/check_session.py ~/lab-captures/dryrun_*.pcap ~/lab-captures/phases_*.csv

If you have several sessions in ~/lab-captures, give the exact two files of
the session you want instead of using *.

It reports, for every phase in the phase log:
  - the exit code (0 = normal finish, 124 = stopped by the time limit, which
    is how the flood phases normally end; anything else = the phase failed),
  - how many packets the capture holds between the phase's start and end time,
  - the mean and largest Ethernet frame size.
"""
import bisect
import csv
import re
import subprocess
import sys
from pathlib import Path

ACCEPTED_EXIT_CODES = {0, 124}
MIN_PACKETS = 100      # fewer packets than this in a phase: it did not really run
MAX_NORMAL_FRAME = 1514  # MTU 1500 + 14-byte Ethernet header

# With "-tt -e -q" every line starts with an epoch timestamp and contains
# "length N", where N is the full Ethernet frame size.
LINE = re.compile(r"^(\d+\.\d+)\s.*?\blength (\d+)")


def read_capture(pcap):
    try:
        proc = subprocess.Popen(
            ["tcpdump", "-nr", str(pcap), "-tt", "-e", "-q"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
    except FileNotFoundError:
        sys.exit("tcpdump is not installed. Run: bash lab/lab_setup.sh")
    times, sizes = [], []
    for line in proc.stdout:
        match = LINE.match(line)
        if match:
            times.append(float(match.group(1)))
            sizes.append(int(match.group(2)))
    proc.wait()
    return times, sizes


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    pcap, phases_csv = Path(sys.argv[1]), Path(sys.argv[2])
    for path in (pcap, phases_csv):
        if not path.is_file():
            sys.exit(f"File not found: {path}")

    print(f"Reading {pcap.name} ({pcap.stat().st_size / 1e6:.1f} MB) ...")
    times, sizes = read_capture(pcap)
    if not times:
        sys.exit("No packets could be read from the capture file.")

    with open(phases_csv, newline="") as f:
        phases = list(csv.DictReader(f))
    if not phases:
        sys.exit("The phase log has no phases.")

    problems = []
    print(f"\nCapture: {len(times):,} packets over {times[-1] - times[0]:.0f} s\n")
    print(f"{'phase':<14}{'seconds':>8}{'exit':>6}{'packets':>11}{'pkts/s':>9}{'mean frame':>12}{'max frame':>11}")

    in_phases = 0
    for row in phases:
        start, end = float(row["start_epoch"]), float(row["end_epoch"])
        code = int(row["exit_code"])
        lo = bisect.bisect_left(times, start)
        hi = bisect.bisect_right(times, end)
        count = hi - lo
        in_phases += count
        window = sizes[lo:hi]
        mean_frame = sum(window) / count if count else 0.0
        max_frame = max(window) if window else 0
        seconds = end - start
        print(f"{row['phase']:<14}{seconds:>8.1f}{code:>6}{count:>11,}"
              f"{count / max(seconds, 0.001):>9,.0f}{mean_frame:>12.0f}{max_frame:>11}")

        if code not in ACCEPTED_EXIT_CODES:
            problems.append(f"{row['phase']}: exit code {code}, the phase failed")
        if count < MIN_PACKETS:
            problems.append(f"{row['phase']}: only {count} packets captured, "
                            f"so the phase did not really run (or the capture was not running)")

    print(f"\nPackets outside any phase (idle gaps, setup): {len(times) - in_phases:,}")

    oversized = sum(1 for s in sizes if s > MAX_NORMAL_FRAME)
    if oversized:
        print(f"Frames larger than {MAX_NORMAL_FRAME} bytes: {oversized:,}")
        print("  Network offloading was probably on. This is not an error by itself, but it changes the")
        print("  packet-length features. Check the .notes.txt file and record which setting you used.")
    else:
        print(f"No frame is larger than {MAX_NORMAL_FRAME} bytes (offloading off, as intended).")

    print()
    if problems:
        print("PROBLEMS FOUND. Do not use this session:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("OK: every phase ran and produced traffic.")


if __name__ == "__main__":
    main()