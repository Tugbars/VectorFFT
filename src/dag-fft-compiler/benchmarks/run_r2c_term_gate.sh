#!/usr/bin/env bash
# run_r2c_term_gate.sh — correctness gate for the fused forward r2c_term codelet
# (step-2 fusion, emit half). Generates radix256_r2c_term_k{K} for several
# interior k, feeds known Z values, checks X[k]/X[m] vs brute r2c reference.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
GEN="$REPO/generator/_build/default/bin/gen_radix.exe"
CC="${CC:-gcc-13}"; CFLAGS="-O3 -march=native -mavx512f -mavx512dq -mfma -ffp-contract=fast"
WORK="${TMPDIR:-/tmp}/r2ctermgate"; mkdir -p "$WORK"
fails=0
echo "=== r2c_term codelet vs brute r2c (N=256, interior k) ==="
for k in 1 5 17 31 63; do
  "$GEN" 256 --r2c-term --r2c-term-k "$k" --emit-c --isa avx512 2>/dev/null > "$WORK/rt.c"
  sed "s/radix256_r2c_term_k1_fwd_avx512/radix256_r2c_term_k${k}_fwd_avx512/g; s/int k=1,/int k=$k,/" \
    "$REPO/benchmarks/gate_r2c_term.c" > "$WORK/gate_k.c"
  $CC $CFLAGS "$WORK/gate_k.c" "$WORK/rt.c" -lm -o "$WORK/gk" 2>/dev/null
  res=$("$WORK/gk" 2>/dev/null | grep 'MAX ERR'); rc=$?
  printf "  k=%-3d : %s\n" "$k" "$res"
  [ $rc -ne 0 ] && fails=$((fails+1))
done
[ $fails -eq 0 ] && echo "ALL PASS" || echo "$fails FAIL(s)"
exit $((fails>0?1:0))
