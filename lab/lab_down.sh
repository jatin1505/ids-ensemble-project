#!/usr/bin/env bash
#
# lab/lab_down.sh
#
# Removes the lab created by lab_up.sh: stops the lab's nginx and deletes both
# network namespaces (which also removes the virtual cable between them).
# Safe to run more than once, and safe if the lab is only half created.
#
#     sudo bash lab/lab_down.sh
#
# Your captures in ~/lab-captures are not touched.

set -uo pipefail

if [ "$EUID" -ne 0 ]; then
    echo "Run with sudo:  sudo bash $0"
    exit 1
fi

NGINX_PID=/run/ids-lab-nginx.pid

if [ -f "$NGINX_PID" ]; then
    kill "$(cat "$NGINX_PID")" 2> /dev/null || true
    sleep 1
    rm -f "$NGINX_PID"
fi

for ns in lab-s lab-t; do
    if ip netns list | grep -qE "^${ns}( |\$)"; then
        # Stop anything still running inside the namespace, then delete it.
        ip netns pids "$ns" 2> /dev/null | xargs -r kill 2> /dev/null || true
        sleep 0.5
        if ip netns del "$ns" 2> /dev/null; then
            echo "Removed namespace ${ns}"
        else
            # A leftover from before a reboot or crash: the name is still listed
            # but nothing is behind it, so only the empty marker file remains.
            rm -f "/run/netns/${ns}" && echo "Removed leftover namespace file for ${ns}"
        fi
    fi
done

echo "Lab is down."