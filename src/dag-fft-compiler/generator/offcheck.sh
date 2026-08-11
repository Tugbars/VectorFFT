#!/bin/sh
cd /mnt/c/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler/generator || exit 1
G=./_build/default/bin/gen_radix.exe
O=/mnt/c/Users/Tugbars/AppData/Local/Temp/claude/c--Users-Tugbars-Desktop-highSpeedFFT/dfcd8b2a-b554-435a-b1f8-81f8ebd8e66f/scratchpad
# no spill knob: must byte-match a pre-cx_spill classic emission
$G 16 --cil-t2 --isa avx2 --uarch raptor_lake_avx2 --emit-c > "$O/off_t2_new.c" 2>/dev/null
$G 32 --cil-t2 --cil-blocked --cil-split 4.8 --isa avx2 --uarch raptor_lake_avx2 --emit-c > "$O/off_t2b48_new.c" 2>/dev/null
md5sum "$O/off_t2_new.c" "$O/off_t2b48_new.c"
