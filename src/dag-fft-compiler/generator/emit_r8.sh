#!/bin/sh
cd /mnt/c/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler/generator || exit 1
G=./_build/default/bin/gen_radix.exe
O=/mnt/c/Users/Tugbars/AppData/Local/Temp/claude/c--Users-Tugbars-Desktop-highSpeedFFT/dfcd8b2a-b554-435a-b1f8-81f8ebd8e66f/scratchpad
U="--isa avx2 --uarch raptor_lake_avx2 --emit-c"
$G 8 --cil-t2 $U > $O/r8_cls.c 2>/dev/null
VFFT_CX_LAZYLOAD=1 $G 8 --cil-t2 --cil-tangent $U > $O/r8_tan.c 2>/dev/null
$G 8 --cil-n1t $U > $O/r8L_cls.c 2>/dev/null
VFFT_CX_LAZYLOAD=1 $G 8 --cil-n1t --cil-tangent $U > $O/r8L_tan.c 2>/dev/null
echo emitted
