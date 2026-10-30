#!/bin/sh
# run.sh -- start Lichee YOLO Studio from this directory.
#
#   ./run.sh
#   QT_QPA_PLATFORM=offscreen ./run.sh --smoke-test
#
# Uses .venv/bin/python when present, otherwise python3. On Linux the script
# checks the Qt/X11 system dependency and prints an actionable message instead
# of Qt's "Could not load the Qt platform plugin xcb" abort.
set -eu

SELF=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

if [ "$(uname -s)" = "Linux" ] && \
        [ "${YOLO_STUDIO_SKIP_QT_CHECK:-0}" != "1" ]; then
    platform=${QT_QPA_PLATFORM:-}
    case "$platform" in
        offscreen*|minimal*|wayland*) ;;
        *)
            if command -v ldconfig >/dev/null 2>&1 && \
                    ! ldconfig -p 2>/dev/null | grep -q 'libxcb-cursor\.so\.0'; then
                cat >&2 <<'EOF'
Lichee YOLO Studio needs libxcb-cursor0 when Qt uses the X11 (xcb) backend.
Install it and run this script again:

  Debian/Ubuntu: sudo apt install libxcb-cursor0
  Fedora:        sudo dnf install xcb-util-cursor
  Arch:          sudo pacman -S xcb-util-cursor

On Wayland:  QT_QPA_PLATFORM=wayland ./run.sh
Skip check:  YOLO_STUDIO_SKIP_QT_CHECK=1 ./run.sh
EOF
                exit 1
            fi
            ;;
    esac
fi

PYTHON=${PYTHON:-python3}
if [ -x "$SELF/.venv/bin/python" ]; then
    PYTHON="$SELF/.venv/bin/python"
fi
exec "$PYTHON" "$SELF/app.py" "$@"
