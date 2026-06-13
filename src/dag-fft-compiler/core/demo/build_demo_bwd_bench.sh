#!/bin/bash
set -e
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
PROTO_CORE=$ROOT/src/prototype-core
GENERATED=$ROOT/src/prototype/generated
CC=${CC:-icx}
CACHE_TAG=$(basename "$CC" | tr -d '.-')
OBJ_DIR=$ROOT/src/prototype/build_tuned/obj/avx2_${CACHE_TAG}
CFLAGS="-O2 -mavx2 -mfma -xHost -Wno-incompatible-pointer-types"

EXTRA_LIB_DIRS=(
  "C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\2025.3\\lib"
  "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.42.34433\\lib\\x64"
  "C:\\Program Files (x86)\\Windows Kits\\10\\Lib\\10.0.22621.0\\ucrt\\x64"
  "C:\\Program Files (x86)\\Windows Kits\\10\\Lib\\10.0.22621.0\\um\\x64"
)
for win_p in "${EXTRA_LIB_DIRS[@]}"; do
  unix_p=$(cygpath -u "$win_p" 2>/dev/null || echo "")
  if [ -n "$unix_p" ] && [ -d "$unix_p" ]; then
    if [ -z "$LIB" ]; then LIB="$win_p"; else LIB="$LIB;$win_p"; fi
  fi
done
export LIB

OUT_DIR=$ROOT/src/prototype/build_tuned
OUT=$OUT_DIR/demo_bwd_bench
RSP=$OBJ_DIR/link_dp_one.rsp
if [ ! -f "$RSP" ]; then
  echo "[build] expected $RSP — run build_demo_dp_one_cell.sh first"
  exit 1
fi
$CC $CFLAGS -I $PROTO_CORE -I $GENERATED \
    "$PROTO_CORE/demo/demo_bwd_bench.c" \
    @$RSP -o $OUT 2>&1 | grep -v -E "warning|note|deprecated|expanded|\^|^[ ]*\\\\|->" | tail -5
echo "[build] built $OUT"
