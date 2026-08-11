#!/bin/sh
cd /mnt/c/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler/generator || exit 1
G=./_build/default/bin/gen_radix.exe
OUT=/mnt/c/Users/Tugbars/AppData/Local/Temp/claude/c--Users-Tugbars-Desktop-highSpeedFFT/dfcd8b2a-b554-435a-b1f8-81f8ebd8e66f/scratchpad/emit_tg
VFFT_CX_SCHED=cpl2 $G 16 --cil-t2 --cil-tangent --cil-blocked --cil-split 2.8 --isa avx2 --uarch raptor_lake_avx2 --emit-c > "$OUT/t2b28_cpl2.c" 2> "$OUT/t2b28_cpl2.err" || echo "cpl2 halves FAILED"
VFFT_CX_SCHED=cpl2 $G 16 --cil-t2 --cil-tangent --isa avx2 --uarch raptor_lake_avx2 --emit-c > "$OUT/t2_16_cpl2.c" 2> "$OUT/t2_16_cpl2.err" || echo "cpl2 mono FAILED"
wc -l "$OUT/t2b28_cpl2.c" "$OUT/t2_16_cpl2.c" 2>/dev/null
