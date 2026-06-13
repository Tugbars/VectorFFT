#!/bin/bash
# Build the raylib prelaunch GUI (section 40b). Fetches raylib 5.0
# (pinned release) if no system raylib is found, statically links it,
# and installs the binary at generator/build_tuned/preflight_gui.
# Runtime needs only X11/GL (WSLg provides both on Windows 11).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
GENROOT=$(cd "$HERE/.." && pwd)
RL_VER=5.0
RL_DIR="$GENROOT/_raylib/raylib-${RL_VER}_linux_amd64"
if [ ! -f "$RL_DIR/lib/libraylib.a" ]; then
  mkdir -p "$GENROOT/_raylib"
  echo "[build_preflight_gui] fetching raylib $RL_VER release"
  curl -sL -o "$GENROOT/_raylib/raylib.tar.gz" \
    "https://github.com/raysan5/raylib/releases/download/${RL_VER}/raylib-${RL_VER}_linux_amd64.tar.gz"
  tar xzf "$GENROOT/_raylib/raylib.tar.gz" -C "$GENROOT/_raylib"
fi
mkdir -p "$GENROOT/build_tuned"
CC="${CC:-gcc}"
$CC -O2 -o "$GENROOT/build_tuned/preflight_gui" "$HERE/preflight_gui.c" \
    -I "$RL_DIR/include" "$RL_DIR/lib/libraylib.a" \
    -lGL -lm -lpthread -ldl -lrt -lX11
echo "[build_preflight_gui] built $GENROOT/build_tuned/preflight_gui"
