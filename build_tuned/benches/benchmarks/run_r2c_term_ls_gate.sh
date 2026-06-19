#!/usr/bin/env bash
# Model (b) emit gate: the fused last-stage terminator codelet vs brute r2c.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
GEN="$REPO/generator/_build/default/bin/gen_radix.exe"
CC="${CC:-gcc-13}"; CFLAGS="-O3 -march=native -mavx512f -mavx512dq -mfma -ffp-contract=fast"
WORK="${TMPDIR:-/tmp}/mblsgate"; mkdir -p "$WORK"
"$GEN" 256 --r2c-term-ls --r2c-term-ls-r 8 --emit-c --isa avx512 2>/dev/null > "$WORK/mb.c"
$CC $CFLAGS "$REPO/benchmarks/gate_r2c_term_ls.c" "$WORK/mb.c" -lm -o "$WORK/g" 2>/dev/null
"$WORK/g"
