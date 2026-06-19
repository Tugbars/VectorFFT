#!/bin/bash
# build_demo_impulse.sh — Phase 2 correctness demo.
#
# Builds the impulse test (N=16 K=4) that validates twiddle.h's
# Method C twiddle compute produces FFT-correct output.
set -e

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
PROTO_CORE=$ROOT/src/prototype-core
CODELETS=$ROOT/src/prototype/codelets/avx2
GENERATED=$ROOT/src/prototype/generated

CC=${CC:-gcc-15}
CFLAGS="-O2 -mavx2 -mfma -march=native -Wno-incompatible-pointer-types"

OUT_DIR=$ROOT/src/prototype/build_tuned
mkdir -p $OUT_DIR
OUT=$OUT_DIR/demo_impulse

# Need: R=4 n1 + t1s codelets (the impulse plan uses radix-4 throughout).
# Plus the codelets the OTHER hardcoded plan_executors entries reference
# (R=4 t1, R=8 *, and R=4/R=8 log3) so plan_executors.h links cleanly.
SOURCES=(
  $PROTO_CORE/demo/demo_impulse.c
  $CODELETS/small_pow2/r4_n1_fwd.c
  $CODELETS/small_pow2/r4_t1s_dit_fwd.c
  $CODELETS/small_pow2/r4_t1_dit_fwd.c
  $CODELETS/small_pow2/r4_t1_dit_fwd_log3.c
  $CODELETS/small_pow2/r8_n1_fwd.c
  $CODELETS/small_pow2/r8_t1s_dit_fwd.c
  $CODELETS/small_pow2/r8_t1_dit_fwd.c
  $CODELETS/small_pow2/r8_t1_dit_fwd_log3.c
)

echo "[build_demo_impulse] linking $(echo "${SOURCES[@]}" | wc -w) sources"
$CC $CFLAGS \
    -I $PROTO_CORE \
    -I $GENERATED \
    "${SOURCES[@]}" \
    -o $OUT -lm

echo "[build_demo_impulse] built $OUT"
ls -la $OUT
