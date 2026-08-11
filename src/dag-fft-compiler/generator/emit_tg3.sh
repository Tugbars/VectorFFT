#!/bin/sh
cd /mnt/c/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler/generator || exit 1
G=./_build/default/bin/gen_radix.exe
OUT=/mnt/c/Users/Tugbars/AppData/Local/Temp/claude/c--Users-Tugbars-Desktop-highSpeedFFT/dfcd8b2a-b554-435a-b1f8-81f8ebd8e66f/scratchpad/emit_tg
# CPL-scheduled tangent emissions
VFFT_CX_SCHED=cpl $G 16 --cil-t2 --cil-tangent --isa avx2 --uarch raptor_lake_avx2 --emit-c > "$OUT/t2_16_tgc.c" 2> "$OUT/t2_16_tgc.err" || echo "cpl mono FAILED"
VFFT_CX_SCHED=cpl $G 16 --cil-t2 --cil-tangent --cil-blocked --cil-split 2.8 --isa avx2 --uarch raptor_lake_avx2 --emit-c > "$OUT/t2b28_16_tgc.c" 2> "$OUT/t2b28_16_tgc.err" || echo "cpl halves FAILED"
# OFF-safety: SR default still byte-identical
$G 16 --cil-t2 --cil-blocked --cil-split 4.4 --isa avx2 --uarch raptor_lake_avx2 --emit-c > "$OUT/t2b44_16_off3.c" 2>/dev/null
cmp -s "$OUT/t2b44_16_off1.c" "$OUT/t2b44_16_off3.c" && echo "OFF (SR default): BYTE-IDENTICAL" || echo "OFF: DIFFER (BUG)"
wc -l "$OUT/t2_16_tgc.c" "$OUT/t2b28_16_tgc.c" 2>/dev/null
head -3 "$OUT/t2_16_tgc.err" 2>/dev/null
