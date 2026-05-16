#!/bin/bash
# build_demo_planner.sh — Phase 3 validation demo.
#
# Same parallel + cached + R=1024-stubbed pattern as
# cost_model/build_measure_cpe.sh. Links the full registry-required
# codelet set so vfft_proto_registry_init_avx2 has every symbol it
# references defined.
#
# CC: gcc-15 default. Override with CC=icx for Intel compiler.
set -e

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
PROTO_CORE=$ROOT/src/prototype-core
CODELETS_ROOT=$ROOT/src/prototype/codelets/avx2
GENERATED=$ROOT/src/prototype/generated

CC=${CC:-gcc-15}
NJOBS=${NJOBS:-$(nproc 2>/dev/null || echo 4)}

# Reuse measure_cpe's .o cache when available so we don't rebuild
# 378 codelets just for this demo.
OBJ_DIR=$ROOT/src/prototype/build_tuned/obj/avx2

CFLAGS="-O2 -mavx2 -mfma -march=native -Wno-incompatible-pointer-types"
case "$CC" in
  *icx*|*icc*) CFLAGS="-O2 -mavx2 -mfma -xHost" ;;
esac

OUT_DIR=$ROOT/src/prototype/build_tuned
mkdir -p $OUT_DIR $OBJ_DIR
OUT=$OUT_DIR/demo_planner

# Codelet inventory (skip xl_pow2 / R=1024 — we'll stub those).
CODELETS=()
for fam in primes small_pow2 mid_pow2 large_pow2 composites; do
  for f in $CODELETS_ROOT/$fam/r*.c; do
    [ -f "$f" ] && CODELETS+=("$f")
  done
done
n=${#CODELETS[@]}

# R=1024 stubs (registry references them; we don't bench R=1024 here).
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

echo "[build_demo_planner] CC=$CC, NJOBS=$NJOBS, codelets=$n (+ R=1024 stubs)"

# Parallel compile w/ .o caching (shared cache with measure_cpe).
compile_one() {
  local src=$1
  local obj=$OBJ_DIR/$(basename "$src" .c).o
  if [ ! -f "$obj" ] || [ "$src" -nt "$obj" ]; then
    $CC $CFLAGS -c "$src" -o "$obj"
  fi
}
export -f compile_one
export CC CFLAGS OBJ_DIR

ALL_SOURCES=("${CODELETS[@]}" "$STUBS_C")
NEED=0
for src in "${ALL_SOURCES[@]}"; do
  obj=$OBJ_DIR/$(basename "$src" .c).o
  if [ ! -f "$obj" ] || [ "$src" -nt "$obj" ]; then NEED=$((NEED+1)); fi
done
echo "[build_demo_planner] compiling $NEED / ${#ALL_SOURCES[@]} sources ($((${#ALL_SOURCES[@]}-NEED)) cached)"

T0=$(date +%s)
printf '%s\n' "${ALL_SOURCES[@]}" | xargs -P $NJOBS -I {} bash -c 'compile_one "$@"' _ {}
T1=$(date +%s)
echo "[build_demo_planner] compile phase: $((T1-T0))s"

# Link.
ALL_OBJS=()
for src in "${ALL_SOURCES[@]}"; do
  ALL_OBJS+=("$OBJ_DIR/$(basename "$src" .c).o")
done

T0=$(date +%s)
$CC $CFLAGS \
    -I $PROTO_CORE \
    -I $GENERATED \
    "$PROTO_CORE/demo/demo_planner.c" \
    "${ALL_OBJS[@]}" \
    -o $OUT -lm
T1=$(date +%s)
echo "[build_demo_planner] link phase: $((T1-T0))s"
echo "[build_demo_planner] built $OUT"
ls -la $OUT
