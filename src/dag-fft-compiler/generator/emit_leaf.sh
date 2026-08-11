#!/bin/sh
cd /mnt/c/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler/generator || exit 1
G=./_build/default/bin/gen_radix.exe
O=/mnt/c/Users/Tugbars/AppData/Local/Temp/claude/c--Users-Tugbars-Desktop-highSpeedFFT/dfcd8b2a-b554-435a-b1f8-81f8ebd8e66f/scratchpad/emit_tg
# leaf (n1t) wing: no ingest, loads+interior+corner-turn. lazy loads, asis order.
VFFT_CX_WING=1 VFFT_CX_LAZYLOAD=1 VFFT_CX_SCHED=asis $G 16 --cil-n1t --cil-tangent --isa avx2 --uarch raptor_lake_avx2 --emit-c > "$O/leaf_wing_lazy.c" 2>/dev/null
VFFT_CX_WING=1 VFFT_CX_LAZYLOAD=1 $G 16 --cil-n1t --cil-tangent --isa avx2 --uarch raptor_lake_avx2 --emit-c > "$O/leaf_wing_srlazy.c" 2>/dev/null
echo ok
