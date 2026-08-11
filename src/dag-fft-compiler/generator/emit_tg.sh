#!/bin/sh
# emit tangent + classic reference codelets into the scratchpad for gating
cd /mnt/c/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler/generator || exit 1
G=./_build/default/bin/gen_radix.exe
OUT=/mnt/c/Users/Tugbars/AppData/Local/Temp/claude/c--Users-Tugbars-Desktop-highSpeedFFT/dfcd8b2a-b554-435a-b1f8-81f8ebd8e66f/scratchpad/emit_tg
mkdir -p "$OUT"
ls -la $G
# flag OFF reference (byte-identity check vs a second OFF emission)
$G 16 --cil-t2 --isa avx2 --uarch raptor_lake_avx2 --emit-c > "$OUT/t2_16_classic.c" 2> "$OUT/t2_16_classic.err" || echo "classic t2 emit FAILED"
# flag ON: tangent t2 (monolithic) + n1t (leaf)
$G 16 --cil-t2 --cil-tangent --isa avx2 --uarch raptor_lake_avx2 --emit-c > "$OUT/t2_16_tg.c" 2> "$OUT/t2_16_tg.err" || echo "tangent t2 emit FAILED"
$G 16 --cil-n1t --cil-tangent --isa avx2 --uarch raptor_lake_avx2 --emit-c > "$OUT/n1t_16_tg.c" 2> "$OUT/n1t_16_tg.err" || echo "tangent n1t emit FAILED"
# blocked tangent variants (the production-shape bet)
$G 16 --cil-t2 --cil-tangent --cil-blocked --cil-split 4.4 --isa avx2 --uarch raptor_lake_avx2 --emit-c > "$OUT/t2b44_16_tg.c" 2> "$OUT/t2b44_16_tg.err" || echo "tangent t2b44 emit FAILED"
$G 16 --cil-n1t --cil-tangent --cil-blocked --cil-split 4.4 --isa avx2 --uarch raptor_lake_avx2 --emit-c > "$OUT/n1tb44_16_tg.c" 2> "$OUT/n1tb44_16_tg.err" || echo "tangent n1tb44 emit FAILED"
# flag OFF byte-identity pair: emit classic t2b44 twice
$G 16 --cil-t2 --cil-blocked --cil-split 4.4 --isa avx2 --uarch raptor_lake_avx2 --emit-c > "$OUT/t2b44_16_off1.c" 2>/dev/null
$G 16 --cil-t2 --cil-blocked --cil-split 4.4 --isa avx2 --uarch raptor_lake_avx2 --emit-c > "$OUT/t2b44_16_off2.c" 2>/dev/null
cmp -s "$OUT/t2b44_16_off1.c" "$OUT/t2b44_16_off2.c" && echo "OFF determinism: BYTE-IDENTICAL" || echo "OFF determinism: DIFFER"
wc -l "$OUT"/*.c
