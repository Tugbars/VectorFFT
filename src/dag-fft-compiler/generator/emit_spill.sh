#!/bin/sh
cd /mnt/c/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler/generator || exit 1
G=./_build/default/bin/gen_radix.exe
OUT=/mnt/c/Users/Tugbars/AppData/Local/Temp/claude/c--Users-Tugbars-Desktop-highSpeedFFT/dfcd8b2a-b554-435a-b1f8-81f8ebd8e66f/scratchpad/emit_tg
for B in 15 14 13; do
  VFFT_CX_WING=1 VFFT_CX_SCHED=asis VFFT_CX_SPILL=$B $G 16 --cil-t2 --cil-tangent --isa avx2 --uarch raptor_lake_avx2 --emit-c > "$OUT/wing_sp$B.c" 2> "$OUT/wing_sp$B.err" || echo "spill$B FAILED: $(head -1 $OUT/wing_sp$B.err)"
done
# OFF-safety: no knob => byte-identical to prior asis
VFFT_CX_WING=1 VFFT_CX_SCHED=asis $G 16 --cil-t2 --cil-tangent --isa avx2 --uarch raptor_lake_avx2 --emit-c > "$OUT/wing_nospill.c" 2>/dev/null
echo "S[] decl + spill traffic per budget:"
for B in 15 14 13; do echo -n "  b$B: "; grep -c "double S\[\|_mm256_storeu_pd(&S\|_mm256_loadu_pd(&S" "$OUT/wing_sp$B.c" 2>/dev/null | tr '\n' ' '; grep -o "double S\[[0-9]*\]" "$OUT/wing_sp$B.c" 2>/dev/null; done
