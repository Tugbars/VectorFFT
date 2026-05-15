#!/bin/bash
# Build score_and_time_plans against the AVX-2 prototype codelets.
# Mirrors build_measure_cpe.sh — same codelet inventory, just a different
# main TU.
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CC=${CC:-gcc-15}
OUT_DIR=$ROOT/build_tuned
mkdir -p $OUT_DIR
OUT=$OUT_DIR/score_and_time_plans

CFLAGS="-O3 -mavx2 -mfma -march=native -Wno-incompatible-pointer-types"

CODELETS=""
for fam in primes small_pow2 mid_pow2 large_pow2 composites; do
  for f in $ROOT/codelets/avx2/$fam/r*_n1_fwd.c \
           $ROOT/codelets/avx2/$fam/r*_t1_dit_fwd.c; do
    [ -f "$f" ] && CODELETS="$CODELETS $f"
  done
done

n=$(echo $CODELETS | wc -w)
echo "[build_score_and_time] codelet count=$n"
echo "[build_score_and_time] compiling..."

$CC $CFLAGS \
  -I $ROOT/cost_model \
  $ROOT/cost_model/score_and_time_plans.c $CODELETS \
  -o $OUT -lm

echo "[build_score_and_time] built $OUT"
ls -la $OUT
