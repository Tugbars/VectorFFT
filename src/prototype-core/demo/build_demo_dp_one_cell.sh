#!/bin/bash
# build_demo_dp_one_cell.sh — DP planner on a single cell.
set -e

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
PROTO_CORE=$ROOT/src/prototype-core
CODELETS_ROOT=$ROOT/src/prototype/codelets/avx2
GENERATED=$ROOT/src/prototype/generated

CC=${CC:-gcc-15}
NJOBS=${NJOBS:-$(nproc 2>/dev/null || echo 4)}

CACHE_TAG=$(basename "$CC" | tr -d '.-')
OBJ_DIR=$ROOT/src/prototype/build_tuned/obj/avx2_${CACHE_TAG}

CFLAGS="-O2 -mavx2 -mfma -march=native -Wno-incompatible-pointer-types"
case "$CC" in
  *icx*|*icc*) CFLAGS="-O2 -mavx2 -mfma -xHost -Wno-incompatible-pointer-types" ;;
esac

case "$CC" in
  *icx*|*icc*)
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
    ;;
esac

OUT_DIR=$ROOT/src/prototype/build_tuned
mkdir -p $OUT_DIR $OBJ_DIR
OUT=$OUT_DIR/demo_dp_one_cell

CODELETS=()
for fam in primes small_pow2 mid_pow2 large_pow2 composites; do
  for f in $CODELETS_ROOT/$fam/r*.c; do
    [ -f "$f" ] && CODELETS+=("$f")
  done
done

STUBS_C=$OBJ_DIR/r1024_stubs.c
if [ ! -f "$STUBS_C" ]; then
  cat > $STUBS_C <<EOF
#include <stddef.h>
__attribute__((target("avx2,fma"))) void radix1024_n1_fwd_avx2(double *a,double *b,const double *c,const double *d,size_t e,size_t f){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;}
__attribute__((target("avx2,fma"))) void radix1024_n1_bwd_avx2(double *a,double *b,const double *c,const double *d,size_t e,size_t f){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;}
__attribute__((target("avx2,fma"))) void radix1024_t1_dit_fwd_avx2(double *a,double *b,const double *c,const double *d,size_t e,size_t f){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;}
__attribute__((target("avx2,fma"))) void radix1024_t1_dit_log3_fwd_avx2(double *a,double *b,const double *c,const double *d,size_t e,size_t f){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;}
EOF
fi

ALL_SOURCES=("${CODELETS[@]}" "$STUBS_C")

compile_one() {
  local src=$1
  local obj=$OBJ_DIR/$(basename "$src" .c).o
  if [ ! -f "$obj" ] || [ "$src" -nt "$obj" ]; then
    $CC $CFLAGS -c "$src" -o "$obj"
  fi
}
export -f compile_one
export CC CFLAGS OBJ_DIR

printf '%s\n' "${ALL_SOURCES[@]}" | xargs -P $NJOBS -I {} bash -c 'compile_one "$@"' _ {} >/dev/null

RSP=$OBJ_DIR/link_dp_one.rsp
> $RSP
for src in "${ALL_SOURCES[@]}"; do
  echo "$(cygpath -m "$OBJ_DIR/$(basename "$src" .c).o")" >> $RSP
done

$CC $CFLAGS \
    -I $PROTO_CORE \
    -I $GENERATED \
    "$PROTO_CORE/demo/demo_dp_one_cell.c" \
    @$RSP \
    -o $OUT -lm 2>&1 | grep -v warning | tail -3

echo "[build_demo_dp_one_cell] built $OUT"
