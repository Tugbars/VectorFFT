#!/bin/bash
# build_measure_memboundness.sh — Build the memory-boundness measurement tool.
#
# Sibling to build_measure_cpe.sh; same caching/parallelism design, just
# targets a different source file. Re-uses the same codelet .o cache so
# repeat builds are instant.
#
# Usage:
#   bash cost_model/build_measure_memboundness.sh
#       Default: AVX-2, gcc-15. Output: build_tuned/measure_memboundness.
#
# Environment overrides:
#   ISA=avx2|avx512    Target ISA (default avx2)
#   CC=<compiler>      Compiler (default gcc-15; gcc-11, clang, icx work too)
#   NJOBS=<int>        Parallel compile jobs (default = nproc)
#   CLEAN=1            Discard cached .o files first
set -e

ROOT=$(cd "$(dirname "$0")/.." && pwd)
ISA=${ISA:-avx2}
CC=${CC:-gcc-15}
CLEAN=${CLEAN:-0}
NJOBS=${NJOBS:-$(nproc 2>/dev/null || sysctl -n hw.physicalcpu 2>/dev/null || echo 4)}

OUT_DIR=$ROOT/build_tuned
OBJ_DIR=$OUT_DIR/obj/$ISA

if [ "$CLEAN" = "1" ] || [ "$CLEAN" = "true" ]; then
  echo "[build_measure_memboundness] CLEAN=1: discarding cached .o files in $OBJ_DIR"
  rm -rf "$OBJ_DIR"
fi

mkdir -p $OBJ_DIR

# icx/icc on Windows needs LIB pointing at oneAPI runtime + MSVC + Win SDK.
# Same hack the demo build scripts use.
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

if [ "$ISA" = "avx2" ]; then
  CFLAGS="-O3 -mavx2 -mfma -march=native -Wno-incompatible-pointer-types"
  OUT=$OUT_DIR/measure_memboundness
  DEFS=""
elif [ "$ISA" = "avx512" ]; then
  CFLAGS="-O3 -mavx512f -mavx512dq -mfma -march=native -Wno-incompatible-pointer-types"
  OUT=$OUT_DIR/measure_memboundness_avx512
  DEFS="-DVFFT_ISA_AVX512=1"
else
  echo "ERROR: unknown ISA '$ISA' (use avx2 or avx512)"
  exit 1
fi

CODELETS=()
for fam in primes small_pow2 mid_pow2 large_pow2 composites; do
  for f in $ROOT/codelets/$ISA/$fam/r*.c; do
    [ -f "$f" ] && CODELETS+=("$f")
  done
done
n=${#CODELETS[@]}

echo "[build_measure_memboundness] ISA=$ISA, CC=$CC, NJOBS=$NJOBS"
echo "[build_measure_memboundness] codelet count=$n  (xl_pow2 skipped, stubbed)"

# R=1024 stubs (registry references them; we don't measure them here).
STUBS_C=$OBJ_DIR/r1024_stubs.c
if [ ! -f "$STUBS_C" ]; then
  cat > $STUBS_C <<EOF
#include <stddef.h>
__attribute__((target("$ISA")))
void radix1024_n1_fwd_$ISA(double *a, double *b, const double *c,
                            const double *d, size_t e, size_t f)
    { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; }
__attribute__((target("$ISA")))
void radix1024_n1_bwd_$ISA(double *a, double *b, const double *c,
                            const double *d, size_t e, size_t f)
    { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; }
__attribute__((target("$ISA")))
void radix1024_t1_dit_fwd_$ISA(double *a, double *b, const double *c,
                                const double *d, size_t e, size_t f)
    { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; }
__attribute__((target("$ISA")))
void radix1024_t1_dit_log3_fwd_$ISA(double *a, double *b, const double *c,
                                     const double *d, size_t e, size_t f)
    { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; }
EOF
fi

compile_one() {
  local src=$1
  local obj=$OBJ_DIR/$(basename "$src" .c).o
  if [ ! -f "$obj" ] || [ "$src" -nt "$obj" ]; then
    $CC $CFLAGS $DEFS -I $ROOT/cost_model -c "$src" -o "$obj"
  fi
}
export -f compile_one
export CC CFLAGS DEFS OBJ_DIR ROOT

ALL_SOURCES=("${CODELETS[@]}" "$STUBS_C" "$ROOT/cost_model/measure_memboundness.c")
TOTAL=${#ALL_SOURCES[@]}
NEED=0
for src in "${ALL_SOURCES[@]}"; do
  obj=$OBJ_DIR/$(basename "$src" .c).o
  if [ ! -f "$obj" ] || [ "$src" -nt "$obj" ]; then
    NEED=$((NEED+1))
  fi
done
echo "[build_measure_memboundness] compiling $NEED / $TOTAL sources ($((TOTAL-NEED)) cached)"

T0=$(date +%s)
printf '%s\n' "${ALL_SOURCES[@]}" | xargs -P $NJOBS -I {} bash -c 'compile_one "$@"' _ {}
T1=$(date +%s)
echo "[build_measure_memboundness] compile phase: $((T1-T0))s"

echo "[build_measure_memboundness] linking..."
T0=$(date +%s)
RSP=$OBJ_DIR/link_memboundness.rsp
> $RSP
for src in "${ALL_SOURCES[@]}"; do
  obj=$OBJ_DIR/$(basename "$src" .c).o
  # On Windows/MSYS, response files want forward-slash or escaped paths.
  if command -v cygpath >/dev/null 2>&1; then
    echo "$(cygpath -m "$obj")" >> $RSP
  else
    echo "$obj" >> $RSP
  fi
done
$CC $CFLAGS @$RSP -o $OUT -lm
T1=$(date +%s)
echo "[build_measure_memboundness] link phase: $((T1-T0))s"

echo "[build_measure_memboundness] built $OUT"
ls -la $OUT
