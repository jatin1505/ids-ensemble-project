#!/usr/bin/env bash
#
# lab/capture.sh
#
# Captures the traffic the lab web server sees (see docs/LAB_SETUP.md).
# Start it BEFORE run_phases.sh and stop it (Ctrl+C) AFTER run_phases.sh ends.
#
# Usage:     sudo bash lab/capture.sh [session-name]
# Example:   sudo bash lab/capture.sh session1
#
# Writes, in ~/lab-captures/ of the user who ran sudo:
#     <session-name>_<UTC timestamp>.pcap         the capture
#     <session-name>_<UTC timestamp>.notes.txt    interface and offload settings
#
# Filter: IPv4 and ARP only, excluding SSH (port 22). A "ports 80 and 443 only"
# filter would be wrong: it would drop the UDP and ICMP flood phases.
#
# Environment variables (all optional):
#     NS       namespace to capture in. Default lab-s. Set NS= (empty) to capture
#              in the normal namespace instead, for example inside a real VM.
#     IFACE    interface to capture on. Default veth-s.
#     OUTDIR   where to write files. Default ~/lab-captures.

set -euo pipefail

if [ "$EUID" -ne 0 ]; then
    echo "Run with sudo:  sudo bash $0 [session-name]"
    exit 1
fi

NAME="${1:-session}"
NS="${NS-lab-s}"
IFACE="${IFACE:-veth-s}"

# Run a command inside the lab namespace, or directly when NS is empty.
in_ns() {
    if [ -n "$NS" ]; then
        ip netns exec "$NS" "$@"
    else
        "$@"
    fi
}

# Check that the namespace exists AND works. A leftover from before a reboot or
# crash can still be listed by name while being broken.
if [ -n "$NS" ] && ! ip netns exec "$NS" true 2> /dev/null; then
    echo "Namespace ${NS} is missing or broken. Rebuild the lab:"
    echo "  sudo bash lab/lab_down.sh && sudo bash lab/lab_up.sh"
    exit 1
fi

REAL_USER="${SUDO_USER:-root}"
REAL_HOME="$(getent passwd "$REAL_USER" | cut -d: -f6)"
OUTDIR="${OUTDIR:-$REAL_HOME/lab-captures}"
mkdir -p "$OUTDIR"
chown "$REAL_USER": "$OUTDIR"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
FILE="$OUTDIR/${NAME}_${STAMP}.pcap"
NOTES="$OUTDIR/${NAME}_${STAMP}.notes.txt"

# Record how the interface was configured. Offload settings (GRO/GSO/TSO) can
# make a capture show packets larger than the wire MTU, which changes the
# packet-length features (Max, Tot sum, AVG, Std). Keeping a record means a
# surprising range check later can be traced back to this.
{
    echo "capture start (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "namespace: ${NS:-<none>}"
    echo "interface: $IFACE"
    in_ns ip -br addr show "$IFACE" || true
    in_ns ethtool -k "$IFACE" 2> /dev/null \
        | grep -E 'generic-receive-offload|generic-segmentation-offload|tcp-segmentation-offload' \
        || echo "ethtool offload info unavailable"
} > "$NOTES"
chown "$REAL_USER": "$NOTES"

echo "Capturing in namespace '${NS:-<none>}' on ${IFACE} -> ${FILE}"
echo "Filter: (ip or arp) and not port 22"
echo "Press Ctrl+C to stop."

# Handle Ctrl+C in this shell so the summary below still runs. tcpdump itself
# still gets a normal Ctrl+C (handled signals are reset in child processes).
trap ':' INT
in_ns tcpdump -i "$IFACE" -nn -s 0 -U -Z "$REAL_USER" -w "$FILE" '(ip or arp) and not port 22' || true

echo
echo "Stopped. Files:"
ls -l "$FILE" "$NOTES"