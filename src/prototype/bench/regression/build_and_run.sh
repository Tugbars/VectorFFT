#!/bin/bash
# Regression bench: Hand (Python gen_radix*.py) vs OCaml (vfft_v2) for
# R=16, 25, 32, 64 across BOTH AVX-512 AND AVX2.
#
# Runs separate benches for each ISA so you can see whether either path
# regressed after generator/emitter changes.
#
# Prerequisites:
#   - vfft_v2 built (dune build)
#   - Python generators at $PYGEN (where your gen_radix{16,25,32,64}.py live)
#   - bench/references/{radix16,radix32}_handcoded.h (AVX-512 hand)
#   - bench/references/radix64_t1_inplace_handcoded.h
#   - gcc-11 (or override CC=gcc-13)
#
# Usage:
#   PYGEN=/path/to/python ./build_and_run.sh         # both ISAs
#   ISA=avx512 PYGEN=... ./build_and_run.sh           # AVX-512 only
#   ISA=avx2   PYGEN=... ./build_and_run.sh           # AVX2 only
#   CC=gcc-13 EXTRA_CFLAGS='' ./build_and_run.sh      # baseline compare
#
# Target -march defaults to icelake-server (ICX). Override with MARCH:
#   MARCH=skylake-avx512 ./build_and_run.sh           # SKX/CLX
#   MARCH=native ./build_and_run.sh                   # auto-detect host

set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
GEN_OCAML=${GEN_OCAML:-$ROOT/_build/default/bin/gen_radix.exe}
# PYGEN: flat dir containing gen_radix{16,25,32,64}.py (legacy).
# PYGEN_BASE: dir containing per-radix subdirs r{16,25,32,64}/gen_radix{R}.py
# (matches highSpeedFFT repo layout). One of the two must be set.
PYGEN=${PYGEN:-}
PYGEN_BASE=${PYGEN_BASE:-$ROOT/../vectorfft_tune/radixes}
CC=${CC:-gcc}
EXTRA_CFLAGS=${EXTRA_CFLAGS:--flive-range-shrinkage -Wno-incompatible-pointer-types}
MARCH=${MARCH:-icelake-server}
ISA=${ISA:-both}

# Resolve the Python generator file path for a given R.
pygen_path() {
  local R=$1 file="gen_radix${R}.py"
  if [ -n "$PYGEN" ] && [ -f "$PYGEN/$file" ]; then
    echo "$PYGEN/$file"
  elif [ -f "$PYGEN_BASE/r${R}/$file" ]; then
    echo "$PYGEN_BASE/r${R}/$file"
  else
    return 1
  fi
}
WORK=$(mktemp -d)
trap "rm -rf $WORK" EXIT

if [ ! -x "$GEN_OCAML" ]; then
  echo "ERROR: OCaml generator not built. Run 'dune build' first."
  exit 1
fi

echo "Generator: $GEN_OCAML"
echo "Hand gen:  $PYGEN"
echo "Compiler:  $CC -march=$MARCH $EXTRA_CFLAGS"
echo "ISA:       $ISA"
echo ""

# Generate hand AVX2 reference for a given R via the matching Python gen
gen_hand_avx2() {
  local R=$1
  local pypath
  pypath=$(pygen_path "$R") || {
    echo "ERROR: Python generator for R=$R not found."
    echo "Set PYGEN to a flat dir, or PYGEN_BASE to dir with r{R}/gen_radix{R}.py."
    return 1
  }
  python3 "$pypath" --isa avx2 --variant ct_t1_dit 2>/dev/null \
    | sed '/^=== /d' > "$WORK/r${R}_hand_avx2.h"
}

# ──────────────────────────────────────────────────────────────────────
# AVX-512 bench
# ──────────────────────────────────────────────────────────────────────
run_avx512() {
  echo "════════════════════════════════════════════════════════════════"
  echo "  AVX-512 regression bench"
  echo "════════════════════════════════════════════════════════════════"

  echo "Generating OCaml AVX-512 codelets..."
  for R in 16 25 32 64; do
    $GEN_OCAML $R --twiddled --in-place --isa avx512 --emit-c \
      > "$WORK/r${R}_ocaml_avx512.c"
  done

  echo "Generating hand R=25 AVX-512 from Python..."
  R25_PY=$(pygen_path 25) || { echo "ERROR: gen_radix25.py not found"; exit 1; }
  python3 "$R25_PY" --isa avx512 --variant ct_t1_dit 2>/dev/null \
    | sed '/^=== /d' > "$WORK/r25_hand.h"

  cp "$ROOT/bench/references/radix16_handcoded.h"            "$WORK/r16_hand.h"
  cp "$ROOT/bench/references/radix32_handcoded.h"            "$WORK/r32_hand.h"
  cp "$ROOT/bench/references/radix64_t1_inplace_handcoded.h" "$WORK/r64_hand.h"
  cp "$(dirname "$0")/regression_bench.c" "$WORK/"

  echo "Building AVX-512 bench..."
  $CC -O3 -mavx512f -mavx512dq -mfma -march=$MARCH $EXTRA_CFLAGS \
    -I"$WORK" \
    "$WORK"/r{16,25,32,64}_ocaml_avx512.c \
    "$WORK/regression_bench.c" \
    -o "$WORK/regression_bench_avx512" -lm

  echo ""
  "$WORK/regression_bench_avx512"

  if command -v python3 >/dev/null 2>&1; then
    echo ""
    echo "--- AVX-512 5-run stability summary ---"
    cp "$(dirname "$0")/summarize.py" "$WORK/summarize_avx512.py"
    sed -i "s|./regression_bench|$WORK/regression_bench_avx512|" \
      "$WORK/summarize_avx512.py"
    python3 "$WORK/summarize_avx512.py"
  fi
}

# ──────────────────────────────────────────────────────────────────────
# AVX2 bench
# ──────────────────────────────────────────────────────────────────────
run_avx2() {
  echo ""
  echo "════════════════════════════════════════════════════════════════"
  echo "  AVX2 regression bench"
  echo "════════════════════════════════════════════════════════════════"

  echo "Generating OCaml AVX2 codelets..."
  for R in 16 25 32 64; do
    $GEN_OCAML $R --twiddled --in-place --isa avx2 --emit-c \
      > "$WORK/r${R}_ocaml_avx2.c"
  done

  echo "Generating hand AVX2 codelets from Python..."
  for R in 16 25 32 64; do
    gen_hand_avx2 $R
  done

  cp "$(dirname "$0")/regression_bench_avx2.c" "$WORK/"

  echo "Building AVX2 bench..."
  # Codelets use per-function target("avx2,fma") attributes so the bench
  # infrastructure can also use AVX2 (timer, copies). -march=$MARCH still
  # applies — ICX supports both ISAs and the codelet target attribute
  # gates the SIMD width per-function regardless.
  $CC -O3 -mavx2 -mfma -march=$MARCH $EXTRA_CFLAGS \
    -I"$WORK" \
    "$WORK"/r{16,25,32,64}_ocaml_avx2.c \
    "$WORK/regression_bench_avx2.c" \
    -o "$WORK/regression_bench_avx2" -lm

  echo ""
  "$WORK/regression_bench_avx2"

  if command -v python3 >/dev/null 2>&1; then
    echo ""
    echo "--- AVX2 5-run stability summary ---"
    cp "$(dirname "$0")/summarize.py" "$WORK/summarize_avx2.py"
    sed -i "s|./regression_bench|$WORK/regression_bench_avx2|" \
      "$WORK/summarize_avx2.py"
    python3 "$WORK/summarize_avx2.py"
  fi
}

case $ISA in
  avx512) run_avx512 ;;
  avx2)   run_avx2 ;;
  both)
    run_avx512
    run_avx2
    ;;
  *)
    echo "ERROR: unknown ISA '$ISA' (use avx512 | avx2 | both)"
    exit 1
    ;;
esac