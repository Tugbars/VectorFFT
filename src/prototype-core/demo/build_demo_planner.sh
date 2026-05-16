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
# 378 codelets just for this demo. Per-toolchain cache (gcc vs icx
# produce incompatible .o formats on Windows — separate subdirs).
CACHE_TAG=$(basename "$CC" | tr -d '.-')
OBJ_DIR=$ROOT/src/prototype/build_tuned/obj/avx2_${CACHE_TAG}

CFLAGS="-O2 -mavx2 -mfma -march=native -Wno-incompatible-pointer-types"
case "$CC" in
  *icx*|*icc*) CFLAGS="-O2 -mavx2 -mfma -xHost -Wno-incompatible-pointer-types" ;;
esac

# CLEAN=1 forces full recompile.
if [ "${CLEAN:-0}" = "1" ]; then
  echo "[build_demo_planner] CLEAN=1 — flushing $OBJ_DIR"
  rm -rf "$OBJ_DIR"
fi

# ICX needs the oneAPI + MSVC + Windows-SDK lib paths to find libircmt /
# libcmt / ucrt / kernel32. When setvars.bat hasn't been sourced, fill in
# the standard candidate paths ourselves. Append (don't replace) so a
# pre-sourced setvars.bat wins.
case "$CC" in
  *icx*|*icc*)
    EXTRA_LIB_DIRS=(
      "C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\2025.3\\lib"
      "C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\lib"
      "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.42.34433\\lib\\x64"
      "C:\\Program Files (x86)\\Windows Kits\\10\\Lib\\10.0.22621.0\\ucrt\\x64"
      "C:\\Program Files (x86)\\Windows Kits\\10\\Lib\\10.0.22621.0\\um\\x64"
    )
    for win_p in "${EXTRA_LIB_DIRS[@]}"; do
      unix_p=$(cygpath -u "$win_p" 2>/dev/null || echo "")
      if [ -n "$unix_p" ] && [ -d "$unix_p" ]; then
        if [ -z "$LIB" ]; then
          LIB="$win_p"
        else
          LIB="$LIB;$win_p"
        fi
      fi
    done
    export LIB
    ;;
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

# Link — use a response file so we don't blow past Windows' command-line
# length limit when passing 378 codelet .o paths.
# ICX/clang doesn't translate POSIX /c/... paths when reading from
# response files; convert to mixed Windows form (C:/...) via cygpath.
RSP=$OBJ_DIR/link.rsp
> $RSP
HAVE_CYGPATH=0
command -v cygpath >/dev/null 2>&1 && HAVE_CYGPATH=1
for src in "${ALL_SOURCES[@]}"; do
  OBJ_PATH="$OBJ_DIR/$(basename "$src" .c).o"
  if [ $HAVE_CYGPATH -eq 1 ]; then
    OBJ_PATH=$(cygpath -m "$OBJ_PATH")
  fi
  echo "$OBJ_PATH" >> $RSP
done

T0=$(date +%s)
$CC $CFLAGS \
    -I $PROTO_CORE \
    -I $GENERATED \
    "$PROTO_CORE/demo/demo_planner.c" \
    @$RSP \
    -o $OUT -lm
T1=$(date +%s)
echo "[build_demo_planner] link phase: $((T1-T0))s"
echo "[build_demo_planner] built $OUT"
ls -la $OUT
