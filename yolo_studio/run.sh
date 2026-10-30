#!/bin/sh
# run.sh -- start Lichee YOLO Studio from this directory.
#
#   ./run.sh
#   QT_QPA_PLATFORM=offscreen ./run.sh --smoke-test
#
# Uses .venv/bin/python when present, otherwise python3. On Linux app.py checks
# the system libraries Qt needs and prints an actionable message instead of
# Qt's "Could not load the Qt platform plugin xcb" abort.
set -eu

SELF=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

PYTHON=${PYTHON:-python3}
if [ -x "$SELF/.venv/bin/python" ]; then
    PYTHON="$SELF/.venv/bin/python"
fi
exec "$PYTHON" "$SELF/app.py" "$@"
