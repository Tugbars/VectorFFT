#!/bin/sh
cd /mnt/c/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler/generator || exit 1
G=./_build/default/bin/gen_radix.exe
O=/mnt/c/Users/Tugbars/AppData/Local/Temp/claude/c--Users-Tugbars-Desktop-highSpeedFFT/dfcd8b2a-b554-435a-b1f8-81f8ebd8e66f/scratchpad
U="--isa avx2 --uarch raptor_lake_avx2 --emit-c"
VFFT_CX_WING=1 VFFT_CX_LAZYLOAD=1 VFFT_CX_SCHED=asis $G 16 --cil-n1t --cil-tangent $U > $O/lf16.c 2>/dev/null
VFFT_CX_WING=1 VFFT_CX_LAZYLOAD=1 VFFT_CX_LAZYSTORE=1 $G 32 --cil-n1t --cil-tangent --cil-blocked --cil-split 2.16 $U > $O/lf32.c 2>/dev/null
echo emitted
