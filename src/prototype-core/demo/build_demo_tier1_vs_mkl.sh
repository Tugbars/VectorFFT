#!/bin/bash
# build_demo_tier1_vs_mkl.sh — bench prototype-core vs MKL on one cell.
# Links MKL via the single-DLL interface (mkl_rt.lib). Reuses the
# per-toolchain .o cache from the other demos.
set -e

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
PROTO_CORE=$ROOT/src/prototype-core
GENERATED=$ROOT/src/prototype/generated

CC=${CC:-icx}
CACHE_TAG=$(basename "$CC" | tr -d '.-')
OBJ_DIR=$ROOT/src/prototype/build_tuned/obj/avx2_${CACHE_TAG}

MKL_ROOT="C:\\Program Files (x86)\\Intel\\oneAPI\\mkl\\2025.3"
MKL_INC=$(cygpath -u "$MKL_ROOT")/include

CFLAGS="-O2 -mavx2 -mfma -xHost -Wno-incompatible-pointer-types"

# Link paths: oneAPI compiler runtime + MKL + MSVC + Windows SDK.
EXTRA_LIB_DIRS=(
  "C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\2025.3\\lib"
  "C:\\Program Files (x86)\\Intel\\oneAPI\\mkl\\2025.3\\lib"
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
OUT=$OUT_DIR/demo_tier1_vs_mkl

# Use the existing link.rsp from the DP demo (same set of codelet .o files).
RSP=$OBJ_DIR/link_dp_one.rsp
if [ ! -f "$RSP" ]; then
  echo "[build] expected $RSP from prior build — please run build_demo_dp_one_cell.sh first"
  exit 1
fi

$CC $CFLAGS \
    -I $PROTO_CORE -I $GENERATED \
    -I "$MKL_INC" \
    "$PROTO_CORE/demo/demo_tier1_vs_mkl.c" \
    @$RSP \
    mkl_rt.lib \
    -o $OUT 2>&1 | grep -v -E "warning|note|deprecated|expanded|\^|^[ ]*\\\\|->" | tail -5

echo "[build] built $OUT"
