#!/bin/bash
# Build measure_cpe against the prototype codelet tree.
#
# Usage:
#   bash cost_model/build_measure_cpe.sh                 # AVX-2 (default)
#   ISA=avx512 bash cost_model/build_measure_cpe.sh      # AVX-512
#
# Output: build_tuned/measure_cpe (AVX-2) or build_tuned/measure_cpe_avx512
#
# Codelets sourced from src/prototype/codelets/{isa}/{family}/r*.c for
# primes, small_pow2, mid_pow2, large_pow2, composites. xl_pow2 (R=1024)
# is intentionally NOT linked — research-only, out of cost-model scope.
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ISA=${ISA:-avx2}
CC=${CC:-gcc-15}
OUT_DIR=$ROOT/build_tuned
mkdir -p $OUT_DIR

if [ "$ISA" = "avx2" ]; then
  CFLAGS="-O3 -mavx2 -mfma -march=native -Wno-incompatible-pointer-types"
  OUT=$OUT_DIR/measure_cpe
  DEFS=""
elif [ "$ISA" = "avx512" ]; then
  CFLAGS="-O3 -mavx512f -mavx512dq -mfma -march=icelake-server -Wno-incompatible-pointer-types"
  OUT=$OUT_DIR/measure_cpe_avx512
  DEFS="-DVFFT_ISA_AVX512=1"
else
  echo "ERROR: unknown ISA '$ISA' (use avx2 or avx512)"
  exit 1
fi

# Gather codelet sources for this ISA. xl_pow2 (R=1024) included now —
# registry.h externs every codelet (including R=1024's 4 codelets), so the
# linker needs them all. R=1024 codelets aren't bench'd (excluded from
# RADIX_LIST in measure_cpe.c) but their addresses get taken by the
# registry init function.
CODELETS=""
for fam in primes small_pow2 mid_pow2 large_pow2 xl_pow2 composites; do
  for f in $ROOT/codelets/$ISA/$fam/r*.c; do
    [ -f "$f" ] && CODELETS="$CODELETS $f"
  done
done

n=$(echo $CODELETS | wc -w)
echo "[build_measure_cpe] ISA=$ISA codelet count=$n"
echo "[build_measure_cpe] CC=$CC"
echo "[build_measure_cpe] CFLAGS=$CFLAGS"
echo "[build_measure_cpe] compiling..."

$CC $CFLAGS $DEFS \
  -I $ROOT/cost_model \
  $ROOT/cost_model/measure_cpe.c $CODELETS \
  -o $OUT -lm

echo "[build_measure_cpe] built $OUT"
ls -la $OUT
