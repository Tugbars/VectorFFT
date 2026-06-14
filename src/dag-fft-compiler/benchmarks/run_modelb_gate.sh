#!/usr/bin/env bash
# Model (b) END-TO-END gate: the fused last-stage path (ls_fwd) vs the default
# r2c executor, full N=256 transform. PRIMARY = brute-Hermitian consistency
# (model-b reschedules + 0.5-folds, so bit-identity is wrong by construction;
# few-ulp vs default is the bar). Requires the bench harness at /tmp/rprof/bh.c.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
GEN="$REPO/generator/_build/default/bin/gen_radix.exe"
CC="${CC:-gcc-13}"
CFLAGS="-O3 -march=native -mavx512f -mavx512dq -mfma -ffp-contract=fast"
FFTW="${FFTW_A:-/home/claude/fftw-3.3.10/.libs/libfftw3.a}"
WORK="${TMPDIR:-/tmp}/mbgate"; mkdir -p "$WORK"
"$GEN" 256 --r2c-term-ls --r2c-term-ls-r 8 --emit-c --isa avx512 2>/dev/null > "$WORK/mb.c"
# the harness (bh.c) + rt codelet are session artifacts; gate runs if present
if [ -f /tmp/rprof/bh.c ] && [ -f /tmp/rt_gate/rt256.c ]; then
  $CC $CFLAGS -I "$REPO/core" -I "$REPO/generator/generated" \
    /tmp/rprof/bh.c /tmp/rt_gate/rt256.c "$WORK/mb.c" /tmp/libcodelets.a "$FFTW" \
    -L/usr/local/lib -lmkl_rt -lm -o "$WORK/g" 2>/dev/null
  LD_LIBRARY_PATH=/usr/local/lib "$WORK/g" 2>&1 | grep -E 'MODEL-B'
else
  echo "SKIP: bench harness (/tmp/rprof/bh.c) not present in this environment"
fi
