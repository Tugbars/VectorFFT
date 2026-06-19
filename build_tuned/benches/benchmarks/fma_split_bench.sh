#!/bin/bash
# fma_split_bench.sh — quantify what fully-fusing FMAs costs vs letting mul+add
# split across the FP ports (port-5 adder on Golden/Raptor Cove). gcc only.
#
# fused   = default generator (fma_lift on)  -> explicit _mm256_fmadd_pd
# unfused = VFFT_DISABLE_FMA_LIFT=1           -> separate _mm256_mul_pd + add/sub
#
# BOTH compiled -ffp-contract=off so gcc does NOT re-contract the split form back
# into FMAs (explicit FMA intrinsics are unaffected by ffp-contract, so the fused
# variant stays fused). Thus we compare genuinely-fused vs genuinely-split asm.
# ratio = fused_ns / unfused_ns ; >1 means SPLIT is faster (port 5 buys something).
#
# Pinned to a FAR P-core (cpu14). Env: PIN, ROUNDS(7), COOLDOWN_MS(150),
# MB_REPS_BUDGET(4e6), MB_BESTOF(15).

export PATH=/home/tugbars/.opam/5.2.0/bin:/usr/bin:/bin
ROOT=/mnt/c/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler
GENROOT=$ROOT/generator
MB=$ROOT/benchmarks/microbench_codelet.c
GEN=/mnt/c/tmp/fma_split/gen
BIN=/mnt/c/tmp/fma_split/bin
OUT=$ROOT/benchmarks/results/fma_split_gcc.csv

PIN=${PIN:-taskset -c 14}
CC=${CC:-gcc}
CF="-O3 -mavx2 -mfma -march=haswell -ffp-contract=off -Wno-incompatible-pointer-types -Wno-unused-result"
ROUNDS=${ROUNDS:-7}
COOLDOWN_MS=${COOLDOWN_MS:-150}
export MB_REPS_BUDGET=${MB_REPS_BUDGET:-4000000}
export MB_BESTOF=${MB_BESTOF:-15}

mkdir -p "$GEN/fused" "$GEN/unfused" "$BIN" "$(dirname "$OUT")"
cool(){ sleep "$(awk "BEGIN{printf \"%.3f\", $COOLDOWN_MS/1000.0}")"; }
FMA='vf(n)?m(add|sub)(add|sub)?[0-9]*pd'; MUL='vmulpd'; ADD='v(add|sub)pd'

cd "$GENROOT" && dune build 2>/tmp/d.err || { echo "DUNE FAIL"; tail /tmp/d.err; exit 1; }
GR=$GENROOT/_build/default/bin/gen_radix.exe

# "R T1S <genflags...> SYM"
cells=(
  "16 0 --in-place --isa avx2 --su radix16_n1_fwd_avx2"
  "32 0 --in-place --isa avx2 --su radix32_n1_fwd_avx2"
  "64 0 --in-place --isa avx2 --su radix64_n1_fwd_avx2"
  "32 0 --twiddled --in-place --isa avx2 radix32_t1_dit_fwd_avx2"
  "64 0 --twiddled --in-place --isa avx2 radix64_t1_dit_fwd_avx2"
)

echo "=== generate + asm structure + compile ==="
declare -A AF AM
for cell in "${cells[@]}"; do
  read -r R T1S rest <<< "$cell"
  SYM="${rest##* }"; FLAGS="${rest% *}"
  "$GR" $R $FLAGS --emit-c > "$GEN/fused/$SYM.c"   2>/dev/null
  VFFT_DISABLE_FMA_LIFT=1 "$GR" $R $FLAGS --emit-c > "$GEN/unfused/$SYM.c" 2>/dev/null
  for v in fused unfused; do
    src=$GEN/$v/$SYM.c
    $CC $CF -S "$src" -o /tmp/s_$v.s 2>/dev/null
    AF[$v]=$(grep -oE "$FMA" /tmp/s_$v.s | wc -l)
    AM[$v]=$(grep -oE "$MUL" /tmp/s_$v.s | wc -l)
    aadd=$(grep -oE "$ADD" /tmp/s_$v.s | wc -l)
    $CC $CF -DRN=$R -DFN=$SYM -DT1S=$T1S "$MB" "$src" -o "$BIN/${v}_$SYM" -lm 2>/tmp/cc.err \
      || echo "  BUILD-FAIL $v $SYM"
    printf "  %-26s %-8s asmFMA=%-4s asmMUL=%-4s asmADD=%-4s\n" "$SYM" "$v" "${AF[$v]}" "${AM[$v]}" "$aadd"
  done
done

echo ""; echo "=== measure (pinned $PIN, $ROUNDS rounds) ==="
declare -A NS
measure(){ $PIN "$1" 2>/dev/null | sed -nE 's/^ns=([0-9.]+)$/\1/p'; }
for ((r=1; r<=ROUNDS; r++)); do
  echo "round $r/$ROUNDS"
  order=(fused unfused); [ $((r%2)) -eq 0 ] && order=(unfused fused)
  for cell in "${cells[@]}"; do
    read -r R T1S rest <<< "$cell"; SYM="${rest##* }"
    for v in "${order[@]}"; do
      exe=$BIN/${v}_$SYM; [ -x "$exe" ] || continue
      cool; x=$(measure "$exe"); [ -z "$x" ] && x=nan
      NS[$SYM|$v]=$(awk -v a="${NS[$SYM|$v]:-1e30}" -v b="$x" 'BEGIN{print (b<a)?b:a}')
    done
  done
done

echo "symbol,radix,fused_ns,unfused_ns,split_speedup" > "$OUT"
echo ""; echo "=== RESULTS (split_speedup = fused/unfused; >1 => SPLIT faster => port5 helps) ==="
for cell in "${cells[@]}"; do
  read -r R T1S rest <<< "$cell"; SYM="${rest##* }"
  fu=${NS[$SYM|fused]:-nan}; un=${NS[$SYM|unfused]:-nan}
  sp=$(awk -v f="$fu" -v u="$un" 'BEGIN{print (u>0)?f/u:0}')
  printf "  %-26s fused=%8.2f  split=%8.2f   %.3fx\n" "$SYM" "$fu" "$un" "$sp"
  printf "%s,%d,%.3f,%.3f,%.4f\n" "$SYM" "$R" "$fu" "$un" "$sp" >> "$OUT"
done
echo ""; echo "CSV: $OUT"
