#!/bin/bash
# Runtime correctness test for all 8 R={2,5,7,11,13,17,19} codelet variants
# vs brute-force scalar DFT. Exercises the full DIT/DIF × Flat/Log3 ×
# t1/t1s combinatorial coverage, ensuring the planner's DIF path is
# complete (DIF requires t1_dif, t1_dif_log3, t1s_dif, t1s_dif_log3 —
# all of which were absent from the original prime_bench harness).
#
# Coverage spans 7 prime radixes × 8 variants = 56 codelets.
#
# Compiler choice (doc 38): gcc-11 + -flive-range-shrinkage cuts stack ops
# by 12-36% on AVX-512 vs gcc-13 default, and preserves correctness. Set
# CC env var to override, e.g. CC=gcc-13 ./build_and_run.sh.
#
# Usage: ./build_and_run.sh

set -e
GEN=${GEN:-../../../_build/default/bin/gen_radix.exe}
CC=${CC:-gcc-11}
EXTRA_CFLAGS=${EXTRA_CFLAGS:--flive-range-shrinkage}
WORK=$(mktemp -d)
trap "rm -rf $WORK" EXIT

# Generate all 8 variants × 7 radixes = 56 codelet .c files
for R in 2 5 7 11 13 17 19; do
  $GEN $R --twiddled --in-place                        --emit-c > "$WORK/gen_r${R}_t1_dit.c"
  $GEN $R --twiddled --in-place --dif                  --emit-c > "$WORK/gen_r${R}_t1_dif.c"
  $GEN $R --twiddled --in-place        --log3          --emit-c > "$WORK/gen_r${R}_t1_dit_log3.c"
  $GEN $R --twiddled --in-place --dif  --log3          --emit-c > "$WORK/gen_r${R}_t1_dif_log3.c"
  $GEN $R --twiddled --in-place --t1s                  --emit-c > "$WORK/gen_r${R}_t1s_dit.c"
  $GEN $R --twiddled --in-place --t1s --dif            --emit-c > "$WORK/gen_r${R}_t1s_dif.c"
  $GEN $R --twiddled --in-place --t1s        --log3    --emit-c > "$WORK/gen_r${R}_t1s_dit_log3.c"
  $GEN $R --twiddled --in-place --t1s --dif  --log3    --emit-c > "$WORK/gen_r${R}_t1s_dif_log3.c"
done

$CC -O3 $EXTRA_CFLAGS -mavx512f -mavx512dq -mfma -march=native test_all8_runtime.c \
    "$WORK"/gen_r*.c -o "$WORK/test_all8" -lm

"$WORK/test_all8"
