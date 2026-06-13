#!/usr/bin/env bash
# run_t1_twiddle_gate.sh — full t1 twiddle-addressing correctness matrix.
#
# Verifies OOP/in-place t1 twiddle ADDRESSING is numerically correct
# across all four conventions x both directions x R in {16,32,64}:
#   flat (t1), log3, t1s         -> gate_t1_twiddle.c  (6-arg in-place)
#   t1p (per-position, OOP)       -> gate_t1p_twiddle.c (9-arg OOP)
#
# This is the REGRESSION GATE for the emit_c-helper extraction (de-dup of
# the "Mirror of emit_c.ml line NNNN" blocks in codelet_oop.ml). Run it
# before and after that refactor; every cell must stay PASS and the
# max_abs values must not change (the refactor is behavior-preserving).
#
# Usage:  ./benchmarks/run_t1_twiddle_gate.sh
# Requires: generator built (generator/_build/default/bin/gen_radix.exe).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
GEN="$REPO/generator/_build/default/bin/gen_radix.exe"
GATE="$REPO/benchmarks/gate_t1_twiddle.c"
GATEP="$REPO/benchmarks/gate_t1p_twiddle.c"
CC="${CC:-gcc-13}"
CFLAGS="-O3 -march=native -ffp-contract=fast"
WORK="${TMPDIR:-/tmp}/t1gate"; mkdir -p "$WORK"

if [ ! -x "$GEN" ]; then echo "ERROR: build generator first (cd generator && dune build)"; exit 2; fi

fails=0
run(){ # R kind dir t1sflag fn extra_genflags
  local R="$1" kind="$2" dir="$3" t1s="$4" fn="$5" extra="$6"
  local dflag=""; [ "$dir" = bwd ] && dflag="--bwd"
  local src="$WORK/${kind}_${R}_${dir}.c"
  # shellcheck disable=SC2086
  $GEN "$R" --twiddled $extra --in-place $dflag --isa avx512 --su --emit-c > "$src" 2>/dev/null
  local bwd=0; [ "$dir" = bwd ] && bwd=1
  $CC $CFLAGS -DRN="$R" -DFN="$fn" -DBWD=$bwd -DT1S="$t1s" "$GATE" "$src" -lm -o "$WORK/g" 2>/dev/null
  local res rc; res=$("$WORK/g" 2>/dev/null); rc=$?
  printf "  R=%-3s %-5s %-3s : %s  %s\n" "$R" "$kind" "$dir" "$res" "$([ $rc -eq 0 ] && echo PASS || { echo FAIL; })"
  [ $rc -ne 0 ] && fails=$((fails+1))
}
runp(){ # R dir   (t1p, OOP path)
  local R="$1" dir="$2"
  local dflag=""; [ "$dir" = bwd ] && dflag="--bwd"
  local src="$WORK/t1p_${R}_${dir}.c"
  $GEN "$R" --oop --twiddled-pos $dflag --oop-load UG --oop-store UG --isa avx512 --emit-c > "$src" 2>/dev/null
  local fn="radix${R}_t1p_inplace_${dir}_avx512_UG_UG"
  local bwd=0; [ "$dir" = bwd ] && bwd=1
  $CC $CFLAGS -DRN="$R" -DFN="$fn" -DBWD=$bwd "$GATEP" "$src" -lm -o "$WORK/gp" 2>/dev/null
  local res rc; res=$("$WORK/gp" 2>/dev/null); rc=$?
  printf "  R=%-3s %-5s %-3s : %s  %s\n" "$R" "t1p" "$dir" "$res" "$([ $rc -eq 0 ] && echo PASS || { echo FAIL; })"
  [ $rc -ne 0 ] && fails=$((fails+1))
}

echo "=== t1 twiddle-addressing correctness gate (avx512) ==="
for R in 16 32 64; do
  for dir in fwd bwd; do
    run "$R" flat "$dir" 0 "radix${R}_t1_dit_${dir}_avx512" ""
    run "$R" log3 "$dir" 0 "radix${R}_t1_dit_log3_${dir}_avx512" "--log3"
    run "$R" t1s  "$dir" 1 "radix${R}_t1s_dit_${dir}_avx512" "--t1s"
    runp "$R" "$dir"
  done
done
echo ""
if [ $fails -eq 0 ]; then echo "ALL PASS"; else echo "$fails FAIL(s)"; fi
exit $((fails > 0 ? 1 : 0))
