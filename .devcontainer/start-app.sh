#!/usr/bin/env bash
# Run each time the codespace starts: open the desktop app, with the Lab 3 demo, on the browser desktop.
# The app keeps running in the background; its messages go to /tmp/pymetrix-gui.log.
set -uo pipefail

export DISPLAY="${DISPLAY:-:1}"
for _ in $(seq 1 60); do  # the desktop starts alongside this script: wait for it
    if xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

if pgrep -f "pymetrix-gui" >/dev/null 2>&1; then
    exit 0  # already open
fi
nohup pymetrix-gui --demo "${PYMETRIX_DEMO:-forecast_round}" --maximized >/tmp/pymetrix-gui.log 2>&1 &
echo "The PyMetrix app is starting on the browser desktop (port 6080)."
