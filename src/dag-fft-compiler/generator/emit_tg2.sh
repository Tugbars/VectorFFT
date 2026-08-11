#!/bin/sh
cd /mnt/c/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler/generator || exit 1
G=./_build/default/bin/gen_radix.exe
OUT=/mnt/c/Users/Tugbars/AppData/Local/Temp/claude/c--Users-Tugbars-Desktop-highSpeedFFT/dfcd8b2a-b554-435a-b1f8-81f8ebd8e66f/scratchpad/emit_tg
$G 16 --cil-t2 --cil-tangent --cil-blocked --cil-split 2.8 --isa avx2 --uarch raptor_lake_avx2 --emit-c > "$OUT/t2b28_16_tg.c" 2> "$OUT/t2b28_16_tg.err" || echo "t2b28 FAILED"
$G 16 --cil-n1t --cil-tangent --cil-blocked --cil-split 2.8 --isa avx2 --uarch raptor_lake_avx2 --emit-c > "$OUT/n1tb28_16_tg.c" 2> "$OUT/n1tb28_16_tg.err" || echo "n1tb28 FAILED"
$G 32 --cil-t2 --cil-tangent --cil-blocked --cil-split 2.16 --isa avx2 --uarch raptor_lake_avx2 --emit-c > "$OUT/t2b216_32_tg.c" 2> "$OUT/t2b216_32_tg.err" || echo "t2b216 FAILED"
$G 32 --cil-n1t --cil-tangent --cil-blocked --cil-split 2.16 --isa avx2 --uarch raptor_lake_avx2 --emit-c > "$OUT/n1tb216_32_tg.c" 2> "$OUT/n1tb216_32_tg.err" || echo "n1tb216 FAILED"
echo "tan check:"; grep -l "0.4142" "$OUT"/t2b28_16_tg.c "$OUT"/n1tb28_16_tg.c "$OUT"/t2b216_32_tg.c "$OUT"/n1tb216_32_tg.c 2>/dev/null | sed 's|.*/||'
