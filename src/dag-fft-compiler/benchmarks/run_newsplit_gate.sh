#!/usr/bin/env bash
# run_newsplit_gate.sh — correctness gate for the NEWSPLIT (scaled conjugate-pair
# split-radix) construction, both monolithic and blocked (dft_expand_newsplit_blocked).
# NEWSPLIT is gated behind VFFT_NEWSPLIT=1; default (off) behavior is unaffected
# (verified separately by diff_step2_baseline.sh staying 48/48 identical).
# n1 (no twiddles) → reference is a plain forward DFT (gate_fuse_n1.c).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
GEN="$REPO/generator/_build/default/bin/gen_radix.exe"
CC="${CC:-gcc-13}"; CFLAGS="-O3 -march=native -ffp-contract=fast"
WORK="${TMPDIR:-/tmp}/nsgate"; mkdir -p "$WORK"
fails=0
echo "=== NEWSPLIT blocked correctness (VFFT_NEWSPLIT=1, fwd, R=32/64/128) ==="
for R in 32 64 128; do
  VFFT_NEWSPLIT=1 "$GEN" "$R" --in-place --isa avx512 --su --emit-c 2>/dev/null > "$WORK/ns_${R}.c"
  fn="radix${R}_n1_fwd_avx512"
  $CC $CFLAGS -DRN="$R" -DFN="$fn" "$REPO/benchmarks/gate_fuse_n1.c" "$WORK/ns_${R}.c" -lm -o "$WORK/g" 2>/dev/null
  res=$("$WORK/g" 2>/dev/null); rc=$?
  spills=$(grep -cE 'spill_re\[|spill_im\[' "$WORK/ns_${R}.c")
  printf "  R=%-3s : %s  %s  (%s spill refs)\n" "$R" "$res" "$([ $rc -eq 0 ] && echo PASS || echo FAIL)" "$spills"
  [ $rc -ne 0 ] && fails=$((fails+1))
done
[ $fails -eq 0 ] && echo "ALL PASS" || echo "$fails FAIL(s)"
exit $((fails>0?1:0))
