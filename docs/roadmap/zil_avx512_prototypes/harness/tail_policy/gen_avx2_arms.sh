#!/bin/bash
# gen_avx2_arms.sh OUTDIR -- emit the AVX2 remainder arms (VFFT_TAIL256) of the
# odd n1 / n1t / t2 kernels, one file per (arm, radix, kind), the symbol suffixed
# _<arm> so every arm links into one bench:
#   narrow       the shipped remainder (the monolithic DAG at VEX-128)
#   masked       the monolithic DAG in one ymm pass, lane 1 masked off
#   blk_narrow   the odd blocked passes at VEX-128              (radix >= 9)
#   blk_masked   the odd blocked passes in ymm, lane 1 masked   (radix >= 9)
#   blk_overrun  the odd blocked passes unmasked: a cost floor  (radix >= 9)
# n1t (the corner-turned leaf of the 2p route) has no odd blocked form: narrow
# and masked only.
# Run under WSL from anywhere; needs the generator built (dune build).
set -u
OUT=${1:?usage: gen_avx2_arms.sh OUTDIR}
G=$(cd "$(dirname "$0")/../../../../../src/dag-fft-compiler/generator" && pwd)
X=$G/_build/default/bin/gen_radix.exe
[ -x "$X" ] || { echo "no generator at $X"; exit 1; }
mkdir -p "$OUT"
RADICES="3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 37 41 43 47"
n=0
for R in $RADICES; do
  for K in n1 n1t t2; do
    ARMS="narrow masked"
    [ "$R" -ge 9 ] && [ "$K" != n1t ] && ARMS="$ARMS blk_narrow blk_masked blk_overrun"
    for A in $ARMS; do
      f="$OUT/radix${R}_z_${K}_fwd_avx2_${A}.c"
      if ! (cd "$G" && VFFT_TAIL256=$A "$X" "$R" --cil-$K --isa avx2 --uarch raptor_lake_avx2 --emit-c) > "$f.tmp" 2> "$f.err"; then
        echo "REFUSED $R $K $A: $(head -c 300 "$f.err")"; rm -f "$f.tmp"; continue
      fi
      sed "s/\bradix${R}_z_${K}_fwd_avx2(/radix${R}_z_${K}_fwd_avx2_${A}(/" "$f.tmp" > "$f"
      rm -f "$f.tmp" "$f.err"
      grep -q "radix${R}_z_${K}_fwd_avx2_${A}(" "$f" || echo "RENAME FAILED $f"
      n=$((n+1))
    done
  done
done
echo "emitted $n kernels into $OUT"
