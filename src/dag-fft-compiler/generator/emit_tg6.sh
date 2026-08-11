#!/bin/sh
cd /mnt/c/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler/generator || exit 1
G=./_build/default/bin/gen_radix.exe
OUT=/mnt/c/Users/Tugbars/AppData/Local/Temp/claude/c--Users-Tugbars-Desktop-highSpeedFFT/dfcd8b2a-b554-435a-b1f8-81f8ebd8e66f/scratchpad/emit_tg
VFFT_CX_WING=1 $G 16 --cil-t2 --cil-tangent --isa avx2 --uarch raptor_lake_avx2 --emit-c > "$OUT/t2_wing_sr.c" 2> "$OUT/t2_wing_sr.err" || echo "wing SR FAILED"
VFFT_CX_WING=1 VFFT_CX_SCHED=cpl $G 16 --cil-t2 --cil-tangent --isa avx2 --uarch raptor_lake_avx2 --emit-c > "$OUT/t2_wing_cpl.c" 2> "$OUT/t2_wing_cpl.err" || echo "wing cpl FAILED"
VFFT_CX_WING=1 VFFT_CX_SCHED=cpl2 $G 16 --cil-t2 --cil-tangent --isa avx2 --uarch raptor_lake_avx2 --emit-c > "$OUT/t2_wing_cpl2.c" 2> "$OUT/t2_wing_cpl2.err" || echo "wing cpl2 FAILED"
wc -l "$OUT"/t2_wing_*.c 2>/dev/null
head -3 "$OUT/t2_wing_sr.err" 2>/dev/null
