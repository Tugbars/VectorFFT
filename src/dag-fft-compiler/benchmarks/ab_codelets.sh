#!/bin/bash
# ab_codelets.sh — A/B microbench of OLD vs NEW in-place c2c AVX2 codelets.
#
#   OLD = src/prototype/codelets/avx2          (pre-holiday DAG compiler; M-pins on)
#   NEW = src/dag-fft-compiler/codelets/inplace/avx2  (new DAG compiler; pins off)
#
# For every codelet SYMBOL present in BOTH trees, build the same symbol from each
# tree into its own tiny exe (microbench_codelet.c) and time it. The driver,
# microbench, buffer, compiler, and flags are identical for both sides, so the
# ONLY variable is the codelet machine code.
#
# Two-phase by design: build everything first (noisy, CPU-hot), THEN measure in
# paced rounds (cool, low duty cycle) so the build heat does not bleed into the
# timings. Thermal defenses: a cooldown sleep before every run, A/B order
# flipped each round so slow drift hits both sides equally, best-of-11 inside
# each exe, and min-over-rounds across the whole sweep.
#
# RUN (from WSL; box should be cpufreq-locked + core-isolated for paper numbers):
#   wsl --cd /mnt/c/tmp bash /mnt/c/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler/benchmarks/ab_codelets.sh
#
# Env knobs:
#   ROUNDS=3          measurement rounds (min taken across them)
#   COOLDOWN_MS=200   idle sleep before every exe invocation
#   CC=gcc            compiler (this WSL has gcc-15 only; same compiler both sides
#                     => the gcc-15-vs-gcc-11 fidelity caveat is common-mode here)
#   PIN="taskset -c 2"  pin the timed runs to an isolated core (recommended)
#   LIMIT=0           if >0, only the first N common symbols (smoke the harness)
#   RADICES="6 7 11"  if set, restrict to these radices (focused re-measure)
#   OUT=/path.csv     override the output CSV path (don't clobber the full sweep)
#   MB_REPS_BUDGET / MB_BESTOF  (read by microbench_codelet.c) raise per-run reps
#                     and internal best-of for a tighter floor at small radices
#
# Output: results/ab_old_vs_new_avx2.csv  +  console summary (wins + geomean).

set -u

ROOT=/mnt/c/Users/Tugbars/Desktop/highSpeedFFT/src
OLD=$ROOT/prototype/codelets/avx2
NEW=$ROOT/dag-fft-compiler/codelets/inplace/avx2
MB=$ROOT/dag-fft-compiler/benchmarks/microbench_codelet.c
OUTDIR=$ROOT/dag-fft-compiler/benchmarks/results
OUT=${OUT:-$OUTDIR/ab_old_vs_new_avx2.csv}
WORK=/tmp/abcl

CC=${CC:-gcc}
CF="-O3 -mavx2 -mfma -march=haswell -ffp-contract=fast -Wno-incompatible-pointer-types -Wno-unused-result"
ROUNDS=${ROUNDS:-3}
COOLDOWN_MS=${COOLDOWN_MS:-200}
PIN=${PIN:-}
LIMIT=${LIMIT:-0}

mkdir -p "$WORK/bin" "$OUTDIR"
cool() { sleep "$(awk "BEGIN{printf \"%.3f\", $COOLDOWN_MS/1000.0}")"; }
sym_of() { grep -ohE 'radix[0-9]+_[a-z0-9_]+_avx2' "$1" | head -1; }

# ---- 1. symbol -> path maps (in-place c2c only) ----------------------------
declare -A OLDMAP NEWMAP

# OLD tree mixes families under avx2/; restrict to the in-place c2c subdirs by
# excluding the families with a different ABI (trig/strided/oop/rfft/c2r).
while IFS= read -r f; do
    s=$(sym_of "$f"); [ -n "$s" ] && OLDMAP[$s]=$f
done < <(find "$OLD" -name '*.c' \
            ! -path '*/trig/*'    ! -path '*/strided/*' \
            ! -path '*/oop/*'     ! -path '*/rfft/*' ! -path '*/c2r/*')

# NEW inplace/avx2 is already pure in-place c2c.
while IFS= read -r f; do
    s=$(sym_of "$f"); [ -n "$s" ] && NEWMAP[$s]=$f
done < <(find "$NEW" -name '*.c')

# intersection, sorted for stable output
common=()
for s in "${!NEWMAP[@]}"; do [ -n "${OLDMAP[$s]:-}" ] && common+=("$s"); done
IFS=$'\n' common=($(sort <<<"${common[*]}")); unset IFS
if [ -n "${RADICES:-}" ]; then              # keep only these radices (space-sep)
    filt=()
    for sym in "${common[@]}"; do
        R=$(echo "$sym" | sed -E 's/^radix([0-9]+)_.*/\1/')
        for want in $RADICES; do [ "$R" = "$want" ] && { filt+=("$sym"); break; }; done
    done
    common=("${filt[@]}")
fi
[ "$LIMIT" -gt 0 ] && common=("${common[@]:0:$LIMIT}")

echo "OLD symbols: ${#OLDMAP[@]}   NEW symbols: ${#NEWMAP[@]}   common: ${#common[@]}"
[ "${#common[@]}" -eq 0 ] && { echo "no common symbols — check tree paths"; exit 1; }

# ---- 2. build phase --------------------------------------------------------
echo "building ${#common[@]} pairs ..."
built=(); bfail=0
for sym in "${common[@]}"; do
    R=$(echo "$sym" | sed -E 's/^radix([0-9]+)_.*/\1/')
    t1s=0; case "$sym" in *_t1s_*) t1s=1;; esac
    ok=1
    $CC $CF -DRN=$R -DFN=$sym -DT1S=$t1s "$MB" "${OLDMAP[$sym]}" -o "$WORK/bin/old_$sym" -lm \
        2>"$WORK/bin/old_$sym.err" || { echo "  BUILD-FAIL old $sym"; ok=0; }
    $CC $CF -DRN=$R -DFN=$sym -DT1S=$t1s "$MB" "${NEWMAP[$sym]}" -o "$WORK/bin/new_$sym" -lm \
        2>"$WORK/bin/new_$sym.err" || { echo "  BUILD-FAIL new $sym"; ok=0; }
    if [ "$ok" -eq 1 ]; then built+=("$sym"); else bfail=$((bfail+1)); fi
done
echo "built ${#built[@]} pairs, $bfail failed"
[ "${#built[@]}" -eq 0 ] && { echo "nothing built"; exit 1; }

# ---- 3. measure phase (paced, interleaved, min-over-rounds) ----------------
declare -A OLDNS NEWNS
measure() { $PIN "$1" 2>/dev/null | sed -nE 's/^ns=([0-9.]+)$/\1/p'; }

for ((r=1; r<=ROUNDS; r++)); do
    echo "round $r/$ROUNDS"
    for sym in "${built[@]}"; do
        if [ $((r % 2)) -eq 1 ]; then
            cool; o=$(measure "$WORK/bin/old_$sym")
            cool; n=$(measure "$WORK/bin/new_$sym")
        else
            cool; n=$(measure "$WORK/bin/new_$sym")
            cool; o=$(measure "$WORK/bin/old_$sym")
        fi
        [ -z "$o" ] && o=nan; [ -z "$n" ] && n=nan
        OLDNS[$sym]=$(awk -v a="${OLDNS[$sym]:-1e30}" -v b="$o" 'BEGIN{print (b<a)?b:a}')
        NEWNS[$sym]=$(awk -v a="${NEWNS[$sym]:-1e30}" -v b="$n" 'BEGIN{print (b<a)?b:a}')
    done
done

# ---- 4. report -------------------------------------------------------------
echo "symbol,radix,old_ns,new_ns,ratio_old_over_new" > "$OUT"
for sym in "${built[@]}"; do
    R=$(echo "$sym" | sed -E 's/^radix([0-9]+)_.*/\1/')
    o=${OLDNS[$sym]}; n=${NEWNS[$sym]}
    ratio=$(awk -v o="$o" -v n="$n" 'BEGIN{print (n>0)?o/n:0}')
    printf "%s,%s,%.4f,%.4f,%.4f\n" "$sym" "$R" "$o" "$n" "$ratio" >> "$OUT"
done

echo ""; echo "=== summary (ratio = old_ns / new_ns; >1 means NEW faster) ==="
awk -F, 'NR>1 && $5+0>0 {
    n++; sl+=log($5);
    if ($5>=1.0) win++; else loss++;
    if ($5>mx){mx=$5; mxs=$1}
    if (mn==0||$5<mn){mn=$5; mns=$1}
}
END{
    if(n==0){print "no valid rows"; exit}
    printf "cells:        %d\n", n;
    printf "NEW faster:   %d   (OLD faster: %d)\n", win, loss;
    printf "geomean:      %.4fx\n", exp(sl/n);
    printf "best  NEW:    %.3fx  (%s)\n", mx, mxs;
    printf "worst NEW:    %.3fx  (%s)\n", mn, mns;
}' "$OUT"
echo ""; echo "CSV: $OUT"
