#!/bin/sh
cd /mnt/c/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler/generator || exit 1
G=./_build/default/bin/gen_radix.exe
OUT=/mnt/c/Users/Tugbars/AppData/Local/Temp/claude/c--Users-Tugbars-Desktop-highSpeedFFT/dfcd8b2a-b554-435a-b1f8-81f8ebd8e66f/scratchpad/emit_tg
for C in 10 12 16; do
  VFFT_CX_SCHED=cpl VFFT_CX_CPL_CAP=$C $G 16 --cil-t2 --cil-tangent --cil-blocked --cil-split 2.8 --isa avx2 --uarch raptor_lake_avx2 --emit-c > "$OUT/t2b28_cap$C.c" 2>/dev/null || echo "cap$C FAILED"
done
wc -l "$OUT"/t2b28_cap*.c
