#!/bin/bash
# build_demo.sh — Phase 1 prototype-core smoke test.
#
# Builds the N=1024 K=128 demo program that exercises:
#   - plan.h (hand-constructed stride_plan_t)
#   - executor.h (two-tier dispatch)
#   - executor_generic.h (cold-cell fallback, not used for this cell)
#   - plan_executors.h (specialized fast path — matches this plan)
#
# Codelet set: R=4, R=8 only (the cells the demo plan + the other
# hardcoded plan_executors entries reference). Far smaller link than
# measure_cpe's full 378-codelet sweep.
set -e

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
PROTO_CORE=$ROOT/src/prototype-core
CODELETS=$ROOT/src/prototype/codelets/avx2
GENERATED=$ROOT/src/prototype/generated

CC=${CC:-gcc-15}
CFLAGS="-O2 -mavx2 -mfma -march=native -Wno-incompatible-pointer-types"

OUT_DIR=$ROOT/src/prototype/build_tuned
mkdir -p $OUT_DIR
OUT=$OUT_DIR/demo_n1024_k128

# Sources: demo program + the codelets it (and the hardcoded
# plan_executors entries) reference. R=4 + R=8 covers everything.
SOURCES=(
  $PROTO_CORE/demo/demo_n1024_k128.c
  $CODELETS/small_pow2/r4_n1_fwd.c
  $CODELETS/small_pow2/r4_t1s_dit_fwd.c
  $CODELETS/small_pow2/r8_n1_fwd.c
  $CODELETS/small_pow2/r8_t1s_dit_fwd.c
)

# Also need the FLAT-variant codelets because plan_executors.h's
# synthetic entry references them. Link them so the lookup-table
# function-pointer slots resolve at link time.
SOURCES+=(
  $CODELETS/small_pow2/r4_t1_dit_fwd.c
  $CODELETS/small_pow2/r8_t1_dit_fwd.c
  $CODELETS/small_pow2/r4_t1_dit_fwd_log3.c
  $CODELETS/small_pow2/r8_t1_dit_fwd_log3.c
)

echo "[build_demo] linking $(echo "${SOURCES[@]}" | wc -w) sources"
$CC $CFLAGS \
    -I $PROTO_CORE \
    -I $GENERATED \
    "${SOURCES[@]}" \
    -o $OUT -lm

echo "[build_demo] built $OUT"
ls -la $OUT
