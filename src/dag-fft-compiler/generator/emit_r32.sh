#!/bin/sh
cd /mnt/c/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler/generator || exit 1
G=./_build/default/bin/gen_radix.exe
O=/mnt/c/Users/Tugbars/AppData/Local/Temp/claude/c--Users-Tugbars-Desktop-highSpeedFFT/dfcd8b2a-b554-435a-b1f8-81f8ebd8e66f/scratchpad/emit_tg
# OFF-safety first: classic blocked byte-identity
$G 16 --cil-t2 --cil-blocked --cil-split 4.4 --isa avx2 --uarch raptor_lake_avx2 --emit-c > "$O/off_t2b44_v2.c" 2>/dev/null
cmp -s "$O/t2b44_16_off1.c" "$O/off_t2b44_v2.c" && echo "OFF blocked: BYTE-IDENTICAL" || echo "OFF blocked: DIFFER (BUG)"
# R32 blocked tangent, halves split, wing interior in passes, lazy everything
VFFT_CX_WING=1 VFFT_CX_LAZYLOAD=1 VFFT_CX_LAZYSTORE=1 $G 32 --cil-t2 --cil-tangent --cil-blocked --cil-split 2.16 --isa avx2 --uarch raptor_lake_avx2 --emit-c > "$O/r32_lazy.c" 2>"$O/r32_lazy.err" || echo "FAIL: $(head -2 $O/r32_lazy.err)"
# reference: same without lazy knobs
VFFT_CX_WING=1 $G 32 --cil-t2 --cil-tangent --cil-blocked --cil-split 2.16 --isa avx2 --uarch raptor_lake_avx2 --emit-c > "$O/r32_plain.c" 2>/dev/null
# leaf too
VFFT_CX_WING=1 VFFT_CX_LAZYLOAD=1 VFFT_CX_LAZYSTORE=1 $G 32 --cil-n1t --cil-tangent --cil-blocked --cil-split 2.16 --isa avx2 --uarch raptor_lake_avx2 --emit-c > "$O/r32L_lazy.c" 2>/dev/null
wc -l "$O/r32_lazy.c" "$O/r32_plain.c" "$O/r32L_lazy.c"
