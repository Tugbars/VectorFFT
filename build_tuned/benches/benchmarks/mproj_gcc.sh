#!/bin/bash
# mproj_gcc.sh — M-project mode A/B on the NEW generator, gcc only.
#
# For a focused set of (regressive-under-ICX) in-place c2c codelets, regenerate
# each from the NEW generator in all THREE M-modes and bench under gcc:
#   M-off   = default            : const __m256d t = expr;
#   M-fence = VFFT_FORCE_FENCE=1  : register t = expr; asm volatile("":"+v"(t));
#   M-on    = VFFT_PIN_FORCE=1    : register t asm("ymmK") = expr; asm volatile(...);
#
# Question: does re-enabling M-fence or M-on beat the M-off default on gcc for
# these cells? (The generator flipped M-off-by-default from gcc-13 data; this
# validates that on gcc-15 / this hardware and flags any cell that wants M back.)
#
# Generation recipe (coverage.ml ip_families): t1 cell =
#   gen_radix <R> --twiddled --in-place --isa avx2 [--dif|--bwd] --emit-c
# Pinned to a FAR P-core (cpu14) by default to stay clear of foreground load.
#
# Env knobs: PIN (default "taskset -c 14"), ROUNDS(5), COOLDOWN_MS(150),
#            MB_REPS_BUDGET(4e6), MB_BESTOF(15), CC(gcc).
# Output: results/mproj_gcc.csv + console table.

export PATH=/home/tugbars/.opam/5.2.0/bin:/usr/bin:/bin
ROOT=/mnt/c/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler
GENROOT=$ROOT/generator
MB=$ROOT/benchmarks/microbench_codelet.c
GEN=/mnt/c/tmp/mproj/gen
BIN=/mnt/c/tmp/mproj/bin
OUT=$ROOT/benchmarks/results/mproj_gcc.csv

PIN=${PIN:-taskset -c 14}
CC=${CC:-gcc}
CF="-O3 -mavx2 -mfma -march=haswell -Wno-incompatible-pointer-types -Wno-unused-result"
ROUNDS=${ROUNDS:-5}
COOLDOWN_MS=${COOLDOWN_MS:-150}
export MB_REPS_BUDGET=${MB_REPS_BUDGET:-4000000}
export MB_BESTOF=${MB_BESTOF:-15}

mkdir -p "$GEN/moff" "$GEN/mfence" "$GEN/mon" "$BIN" "$(dirname "$OUT")"
cool(){ sleep "$(awk "BEGIN{printf \"%.3f\", $COOLDOWN_MS/1000.0}")"; }

cd "$GENROOT" && dune build 2>/tmp/dune.err || { echo "DUNE BUILD FAILED"; tail -15 /tmp/dune.err; exit 1; }
GR=$GENROOT/_build/default/bin/gen_radix.exe
[ -x "$GR" ] || { echo "no gen_radix.exe"; exit 1; }

# focused regressive set: plain-t1 POST-twiddle (dit_bwd / dif_fwd) at R8..64
cells=(
  "8  --bwd radix8_t1_dit_bwd_avx2"
  "8  --dif radix8_t1_dif_fwd_avx2"
  "16 --bwd radix16_t1_dit_bwd_avx2"
  "16 --dif radix16_t1_dif_fwd_avx2"
  "32 --bwd radix32_t1_dit_bwd_avx2"
  "32 --dif radix32_t1_dif_fwd_avx2"
  "64 --bwd radix64_t1_dit_bwd_avx2"
  "64 --dif radix64_t1_dif_fwd_avx2"
)
modes=( "moff:" "mfence:VFFT_FORCE_FENCE=1" "mon:VFFT_PIN_FORCE=1" )

echo "=== generate + compile (gcc) ==="
for cell in "${cells[@]}"; do
  set -- $cell; R=$1; FLAGS=$2; SYM=$3
  for m in "${modes[@]}"; do
    mode="${m%%:*}"; menv="${m#*:}"
    src=$GEN/$mode/$SYM.c
    env $menv "$GR" $R --twiddled --in-place --isa avx2 $FLAGS --emit-c > "$src" 2>/tmp/gen_$mode.err
    if [ ! -s "$src" ]; then echo "  GEN-FAIL $SYM $mode"; head -3 /tmp/gen_$mode.err; continue; fi
    pins=$(grep -c 'asm("ymm' "$src"); fen=$(grep -c 'asm volatile' "$src")
    exe=$BIN/${mode}_${SYM}
    if $CC $CF -DRN=$R -DFN=$SYM -DT1S=0 "$MB" "$src" -o "$exe" -lm 2>/tmp/cc.err; then
      printf "  %-28s %-7s pins=%-4s fence=%-5s OK\n" "$SYM" "$mode" "$pins" "$fen"
    else
      printf "  %-28s %-7s BUILD-FAIL\n" "$SYM" "$mode"; grep -i error /tmp/cc.err | head -2
    fi
  done
done

echo ""; echo "=== measure (pinned: $PIN, $ROUNDS rounds, best-of $MB_BESTOF) ==="
declare -A NS
measure(){ $PIN "$1" 2>/dev/null | sed -nE 's/^ns=([0-9.]+)$/\1/p'; }
for ((r=1; r<=ROUNDS; r++)); do
  echo "round $r/$ROUNDS"
  order=(moff mfence mon)
  case $((r % 3)) in 2) order=(mfence mon moff);; 0) order=(mon moff mfence);; esac
  for cell in "${cells[@]}"; do
    set -- $cell; SYM=$3
    for mode in "${order[@]}"; do
      exe=$BIN/${mode}_${SYM}; [ -x "$exe" ] || continue
      cool; v=$(measure "$exe"); [ -z "$v" ] && v=nan
      key="$SYM|$mode"
      NS[$key]=$(awk -v a="${NS[$key]:-1e30}" -v b="$v" 'BEGIN{print (b<a)?b:a}')
    done
  done
done

echo "symbol,radix,moff_ns,mfence_ns,mon_ns,best_mode,best_over_moff" > "$OUT"
echo ""; echo "=== RESULTS (best_over_moff > 1 => M-fence/M-on beats default) ==="
for cell in "${cells[@]}"; do
  set -- $cell; R=$1; SYM=$3
  o=${NS[$SYM|moff]:-nan}; f=${NS[$SYM|mfence]:-nan}; n=${NS[$SYM|mon]:-nan}
  read best_mode best_ns < <(awk -v o="$o" -v f="$f" -v n="$n" 'BEGIN{
    bm="moff"; bn=o; if(f<bn){bm="mfence";bn=f} if(n<bn){bm="mon";bn=n} print bm, bn}')
  ratio=$(awk -v o="$o" -v b="$best_ns" 'BEGIN{print (b>0)?o/b:0}')
  printf "  %-28s moff=%8.2f mfence=%8.2f mon=%8.2f  win=%-7s %.3fx\n" "$SYM" "$o" "$f" "$n" "$best_mode" "$ratio"
  printf "%s,%d,%.3f,%.3f,%.3f,%s,%.4f\n" "$SYM" "$R" "$o" "$f" "$n" "$best_mode" "$ratio" >> "$OUT"
done
echo ""; echo "CSV: $OUT"
