#!/bin/sh
cd /mnt/c/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler/generator || exit 1
G=./_build/default/bin/gen_radix.exe
O=/mnt/c/Users/Tugbars/AppData/Local/Temp/claude/c--Users-Tugbars-Desktop-highSpeedFFT/dfcd8b2a-b554-435a-b1f8-81f8ebd8e66f/scratchpad/emit_tg
VFFT_CX_WING=1 VFFT_CX_LAZYLOAD=1 VFFT_CX_SCHED=asis $G 16 --cil-t2 --cil-tangent --isa avx2 --uarch raptor_lake_avx2 --emit-c > "$O/wingt2_asis.c" 2>"$O/wingt2.err" || echo "FAIL: $(head -2 $O/wingt2.err)"
VFFT_CX_WING=1 VFFT_CX_LAZYLOAD=1 VFFT_CX_SCHED=asis VFFT_CX_NO_SUBDEDUP=1 $G 16 --cil-t2 --cil-tangent --isa avx2 --uarch raptor_lake_avx2 --emit-c > "$O/wingt2_nd.c" 2>/dev/null
echo "first 10 stmts (should interleave load/ingest/butterfly):"
awk '/k \+ 2 <= count/{f=1} f&&/_mm256/{print ++n": "$0}' "$O/wingt2_asis.c" | head -10 | sed 's/  */ /g' | cut -c1-70
