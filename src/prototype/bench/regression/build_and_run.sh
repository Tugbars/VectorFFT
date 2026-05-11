#!/bin/bash
# Regression bench: Hand (Python gen_radix*.py) vs OCaml (vfft_v2) for
# R=16, 25, 32, 64. Run this after any change to the emitter or
# scheduler to detect performance regressions.
#
# Coverage: 4 radixes × 5 K values × 5 runs = 100 measurements;
# reports median ratio per (R, K) cell with min/max spread.
#
# Prerequisites:
#   - vfft_v2 built (dune build)
#   - Python generators at $PYGEN (default: /mnt/user-data/uploads,
#     or wherever your gen_radix25.py / gen_radix32.py / etc. live)
#   - bench/references/{radix16,radix32}_handcoded.h
#   - bench/references/radix64_t1_inplace_handcoded.h
#   - gcc-11 (or override CC=gcc-13)
#
# Usage:
#   PYGEN=/path/to/python ./build_and_run.sh
#   CC=gcc-13 EXTRA_CFLAGS='' ./build_and_run.sh    # baseline comparison

set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
GEN_OCAML=${GEN_OCAML:-$ROOT/_build/default/bin/gen_radix.exe}
PYGEN=${PYGEN:-/mnt/user-data/uploads}
CC=${CC:-gcc-11}
EXTRA_CFLAGS=${EXTRA_CFLAGS:--flive-range-shrinkage}
WORK=$(mktemp -d)
trap "rm -rf $WORK" EXIT

if [ ! -x "$GEN_OCAML" ]; then
  echo "ERROR: OCaml generator not built. Run 'dune build' first."
  exit 1
fi

echo "Generator: $GEN_OCAML"
echo "Hand gen:  $PYGEN"
echo "Compiler:  $CC $EXTRA_CFLAGS"
echo ""

# Generate OCaml codelets with default production config
echo "Generating OCaml codelets (default = recipe + SU + spill auto-fires)..."
for R in 16 25 32 64; do
  $GEN_OCAML $R --twiddled --in-place --emit-c > "$WORK/r${R}_ocaml.c"
done

# Generate hand R=25 (the only one without a pre-existing handcoded.h)
echo "Generating hand R=25 from Python..."
python3 "$PYGEN/gen_radix25.py" --isa avx512 --variant ct_t1_dit > "$WORK/r25_hand.h" 2>/dev/null
sed -i '/^=== /d' "$WORK/r25_hand.h"

# Copy pre-existing handcoded headers for R=16, 32, 64
cp "$ROOT/bench/references/radix16_handcoded.h"           "$WORK/r16_hand.h"
cp "$ROOT/bench/references/radix32_handcoded.h"           "$WORK/r32_hand.h"
cp "$ROOT/bench/references/radix64_t1_inplace_handcoded.h" "$WORK/r64_hand.h"

# Copy the bench harness
cp "$(dirname "$0")/regression_bench.c" "$WORK/"

echo "Building bench..."
$CC -O3 -mavx512f -mavx512dq -mfma -march=skylake-avx512 $EXTRA_CFLAGS \
  -I"$WORK" \
  "$WORK"/r{16,25,32,64}_ocaml.c \
  "$WORK/regression_bench.c" \
  -o "$WORK/regression_bench" -lm

echo ""
echo "Single run:"
"$WORK/regression_bench"

if command -v python3 >/dev/null 2>&1; then
  echo ""
  echo "Running 5-run stability summary..."
  cp "$(dirname "$0")/summarize.py" "$WORK/"
  cd "$WORK" && python3 summarize.py
fi
