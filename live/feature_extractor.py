"""
live/feature_extractor.py

Turns captured packets into CICIoT2023-style feature rows, so that the frozen
models can score traffic captured in the lab. Full specification and the
evidence behind every rule: docs/EXTRACTION_SPEC.md.

How a CICIoT2023 row is made (reconstructed from the authors' pcap2csv code,
its commented-out original logic, their data-summary notebook, and our own
experiments on the dataset):

  1. Every IPv4 or ARP frame, in capture order, becomes one per-packet row.
     Other frame types (IPv6 and so on) are skipped.
  2. Some per-packet values describe the packet itself: IP protocol number,
     TTL, TCP flags, protocol indicators, frame length ("Tot size").
  3. Some describe the packets seen so far in the current BLOCK of up to 20
     consecutive packets: sum, min, max, mean and (population) standard
     deviation of the frame lengths. The block restarts after every 20th packet.
     This is not per flow: flows play no part in these features.
  4. Magnitue, Radius and Variance come from the same block. The original code
     fed the SAME list of frame lengths in as both "incoming" and "outgoing",
     so Magnitue = sqrt(2*mean), Radius = sqrt(2*variance), and Variance is 1
     when the block's lengths are not all equal and 0 otherwise.
  5. Every WINDOW of 10 consecutive per-packet rows is averaged into one
     output row (a plain mean of every column).

What this module returns for each window: a Series indexed by the 46 feature
names of ml/saved_models/feature_columns.json, in that order:
  - 25 features computed as above (COMPUTED),
  - 8 features forced to 0, because they are constant in the benign training
    data (CONSTANT_ZERO),
  - 13 features set to NaN, because they cannot be reproduced and are filled
    later, in backend/model_runtime.py, from live/live_fill_values.json
    (NEUTRALIZED). This module never invents values.

Honest limits: the extraction code the authors published is a later, simplified
version, and the original per-packet code survives only as comments, so block
size 20 and the exact reset rule are reconstructed, not copied. They are
parameters (BLOCK_SIZE) so they can be checked against a real dataset PCAP.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import dpkt
import pandas as pd

# Reconstructed from the authors' code (see module docstring). Parameters, not
# facts: they are meant to be validated against a real dataset PCAP.
BLOCK_SIZE = 20    # packets per block of running length statistics
WINDOW_SIZE = 10   # packets averaged into one row (paper section 4: benign and 14 attack types)

# The 46 features, in the exact order of ml/saved_models/feature_columns.json.
FEATURE_COLUMNS = [
    "flow_duration", "Header_Length", "Protocol Type", "Duration", "Rate", "Srate", "Drate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number",
    "ack_flag_number", "ece_flag_number", "cwr_flag_number",
    "ack_count", "syn_count", "fin_count", "urg_count", "rst_count",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP",
    "IPv", "LLC", "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT", "Number",
    "Magnitue", "Radius", "Covariance", "Variance", "Weight",
]

# 25 features computed from packets.
COMPUTED = [
    "Protocol Type", "Duration",
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number", "ack_flag_number",
    "HTTP", "HTTPS", "DNS", "TCP", "UDP", "ARP", "ICMP", "IPv", "LLC",
    "Tot sum", "Min", "Max", "AVG", "Std", "Tot size",
    "Magnitue", "Radius", "Variance",
]

# 8 features constant in the benign training data: forced to 0.
CONSTANT_ZERO = ["Drate", "ece_flag_number", "cwr_flag_number",
                 "Telnet", "SMTP", "SSH", "IRC", "DHCP"]

# 13 features that cannot be reproduced: left as NaN here, filled later.
NEUTRALIZED = [c for c in FEATURE_COLUMNS if c not in COMPUTED and c not in CONSTANT_ZERO]

assert len(FEATURE_COLUMNS) == 46 and len(set(FEATURE_COLUMNS)) == 46
assert len(COMPUTED) == 25 and len(CONSTANT_ZERO) == 8 and len(NEUTRALIZED) == 13
assert set(COMPUTED).isdisjoint(CONSTANT_ZERO)

_ETH_IP = dpkt.ethernet.ETH_TYPE_IP
_ETH_ARP = dpkt.ethernet.ETH_TYPE_ARP

# Per-packet values that do not depend on other packets.
_PACKET_KEYS = [
    "Protocol Type", "Duration",
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number", "ack_flag_number",
    "HTTP", "HTTPS", "DNS", "TCP", "UDP", "ARP", "ICMP", "IPv", "LLC",
]


@dataclass
class Window:
    """One output row: the mean of WINDOW_SIZE consecutive per-packet rows."""
    features: pd.Series   # 46 values in FEATURE_COLUMNS order (NaN = to be filled later)
    ts_start: float       # timestamp of the first packet in the window
    ts_end: float         # timestamp of the last packet in the window
    n_packets: int        # normally WINDOW_SIZE; fewer only for a flushed partial window


def parse_packet(buf: bytes) -> dict | None:
    """Per-packet values for one Ethernet frame, or None if it is not IPv4 or ARP.

    Follows the authors' rules: protocol indicators come from the IP protocol
    number and ports; HTTP and HTTPS are TCP port 80 and 443, DNS is UDP port 53;
    LLC is 1 for every IP packet (a quirk of the original code: its LLC test is
    always true, which is why LLC and IPv have identical statistics).
    """
    try:
        eth = dpkt.ethernet.Ethernet(buf)
    except Exception:  # unreadable frame: the original code discards it too
        return None

    values = dict.fromkeys(_PACKET_KEYS, 0.0)
    values["length"] = float(len(buf))   # full Ethernet frame length

    if eth.type == _ETH_ARP:
        values["ARP"] = 1.0              # protocol number and TTL stay 0 for ARP
        return values

    if eth.type != _ETH_IP:
        return None
    ip = eth.data
    if not isinstance(ip, dpkt.ip.IP):
        return None

    values["IPv"] = 1.0
    values["LLC"] = 1.0
    values["Protocol Type"] = float(ip.p)
    values["Duration"] = float(ip.ttl)   # "Duration" is really the IP time-to-live

    transport = ip.data
    if isinstance(transport, dpkt.tcp.TCP):
        values["TCP"] = 1.0
        flags = transport.flags
        values["fin_flag_number"] = float(bool(flags & dpkt.tcp.TH_FIN))
        values["syn_flag_number"] = float(bool(flags & dpkt.tcp.TH_SYN))
        values["rst_flag_number"] = float(bool(flags & dpkt.tcp.TH_RST))
        values["psh_flag_number"] = float(bool(flags & dpkt.tcp.TH_PUSH))
        values["ack_flag_number"] = float(bool(flags & dpkt.tcp.TH_ACK))
        values["HTTP"] = float(transport.sport == 80 or transport.dport == 80)
        values["HTTPS"] = float(transport.sport == 443 or transport.dport == 443)
    elif isinstance(transport, dpkt.udp.UDP):
        values["UDP"] = 1.0
        values["DNS"] = float(transport.sport == 53 or transport.dport == 53)
    if ip.p == dpkt.ip.IP_PROTO_ICMP:
        values["ICMP"] = 1.0
    return values


def _length_stats(lengths: list[float]) -> tuple[float, float, float, float, float]:
    """sum, min, max, mean and POPULATION variance of the block's frame lengths."""
    n = len(lengths)
    total = sum(lengths)
    mean = total / n
    variance = sum((x - mean) ** 2 for x in lengths) / n
    return total, min(lengths), max(lengths), mean, variance


class WindowExtractor:
    """Feed packets one at a time with add(); a finished Window comes back every
    window_size usable packets. Keeps all the state of the original pipeline:
    the current block of frame lengths and the rows of the current window."""

    def __init__(self, window_size: int = WINDOW_SIZE, block_size: int = BLOCK_SIZE):
        if window_size < 1 or block_size < 1:
            raise ValueError("window_size and block_size must be at least 1")
        self.window_size = window_size
        self.block_size = block_size
        self.skipped = 0       # frames that were not IPv4 or ARP, or could not be read
        self._lengths: list[float] = []
        self._rows: list[dict] = []
        self._timestamps: list[float] = []

    def add(self, ts: float, buf: bytes) -> Window | None:
        packet = parse_packet(buf)
        if packet is None:
            self.skipped += 1
            return None

        # Running statistics over the current block, INCLUDING this packet.
        self._lengths.append(packet["length"])
        total, minimum, maximum, mean, variance = _length_stats(self._lengths)
        row = {key: packet[key] for key in _PACKET_KEYS}
        row.update({
            "Tot sum": total,
            "Min": minimum,
            "Max": maximum,
            "AVG": mean,
            "Std": math.sqrt(variance),
            "Tot size": packet["length"],
            # Incoming and outgoing were the same list in the original code:
            "Magnitue": math.sqrt(2.0 * mean),
            "Radius": math.sqrt(2.0 * variance),
            "Variance": 1.0 if variance != 0 else 0.0,
        })
        if len(self._lengths) == self.block_size:
            self._lengths = []           # the block restarts after its last packet

        self._rows.append(row)
        self._timestamps.append(ts)
        if len(self._rows) == self.window_size:
            return self._emit()
        return None

    def flush(self) -> Window | None:
        """The partial window left at the end of a capture, or None. The original
        pipeline keeps it, but averaging fewer than window_size packets makes a
        slightly different kind of row, so callers should use this deliberately."""
        if not self._rows:
            return None
        return self._emit()

    def _emit(self) -> Window:
        n = len(self._rows)
        values = dict.fromkeys(FEATURE_COLUMNS, float("nan"))
        for name in CONSTANT_ZERO:
            values[name] = 0.0
        for name in COMPUTED:
            values[name] = sum(r[name] for r in self._rows) / n
        window = Window(
            features=pd.Series(values, index=FEATURE_COLUMNS, dtype="float64"),
            ts_start=self._timestamps[0],
            ts_end=self._timestamps[-1],
            n_packets=n,
        )
        self._rows = []
        self._timestamps = []
        return window


def extract_windows(
    packets: Iterable[tuple[float, bytes]],
    window_size: int = WINDOW_SIZE,
    block_size: int = BLOCK_SIZE,
    emit_partial: bool = False,
) -> Iterator[Window]:
    """All windows from an iterable of (timestamp, Ethernet frame bytes)."""
    extractor = WindowExtractor(window_size, block_size)
    for ts, buf in packets:
        window = extractor.add(ts, buf)
        if window is not None:
            yield window
    if emit_partial:
        window = extractor.flush()
        if window is not None:
            yield window


def read_pcap(path: str | Path) -> Iterator[tuple[float, bytes]]:
    """(timestamp, frame) for every packet of a classic .pcap file.

    A capture stopped with Ctrl+C can end in a half-written packet; reading
    stops quietly there instead of raising."""
    with open(path, "rb") as f:
        reader = dpkt.pcap.Reader(f)
        if reader.datalink() != dpkt.pcap.DLT_EN10MB:
            raise ValueError(
                f"{path}: expected Ethernet frames (link type {dpkt.pcap.DLT_EN10MB}), "
                f"found link type {reader.datalink()}. Capture on the lab interface, not 'any'."
            )
        try:
            for ts, buf in reader:
                yield ts, buf
        except (dpkt.NeedData, struct.error):
            return


def extract_pcap(path: str | Path, window_size: int = WINDOW_SIZE,
                 block_size: int = BLOCK_SIZE, emit_partial: bool = False) -> Iterator[Window]:
    """All windows of a pcap file."""
    return extract_windows(read_pcap(path), window_size, block_size, emit_partial)


def load_feature_columns(path: str | Path) -> list[str]:
    """Read ml/saved_models/feature_columns.json and check it matches FEATURE_COLUMNS.

    The saved file is the truth the models were trained with; if it ever differs
    from the list in this module, scoring would silently use the wrong columns,
    so this raises instead."""
    import json
    saved = json.loads(Path(path).read_text())
    if saved != FEATURE_COLUMNS:
        raise ValueError(f"{path} does not match FEATURE_COLUMNS in live/feature_extractor.py")
    return saved