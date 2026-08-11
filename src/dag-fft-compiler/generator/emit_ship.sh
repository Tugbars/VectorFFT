#!/bin/sh
cd /mnt/c/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler/generator || exit 1
G=./_build/default/bin/gen_radix.exe
D=../codelets/zil/avx2/pure_il/tangent
U="--isa avx2 --uarch raptor_lake_avx2 --emit-c"

# 1. R16 mono T2 — the PARITY kernel (wing full-kernel + origin order + lazy stores)
VFFT_CX_WING=1 VFFT_CX_LAZYLOAD=1 VFFT_CX_SCHED=asis \
  $G 16 --cil-t2 --cil-tangent $U > $D/_r16_t2.c 2>$D/_e1 || echo "FAIL r16 t2"
# 2. R16 mono N1T leaf (wing interior; turned stores keep the batched edge by design)
VFFT_CX_WING=1 VFFT_CX_LAZYLOAD=1 VFFT_CX_SCHED=asis \
  $G 16 --cil-n1t --cil-tangent $U > $D/_r16_n1t.c 2>$D/_e2 || echo "FAIL r16 n1t"
# 3/4. R16 blocked halves (2.8), SR-scheduled
VFFT_CX_WING=1 VFFT_CX_LAZYLOAD=1 VFFT_CX_LAZYSTORE=1 \
  $G 16 --cil-t2 --cil-tangent --cil-blocked --cil-split 2.8 $U > $D/_r16_t2b.c 2>$D/_e3 || echo "FAIL r16 t2b"
VFFT_CX_WING=1 VFFT_CX_LAZYLOAD=1 VFFT_CX_LAZYSTORE=1 \
  $G 16 --cil-n1t --cil-tangent --cil-blocked --cil-split 2.8 $U > $D/_r16_n1tb.c 2>$D/_e4 || echo "FAIL r16 n1tb"
# 5/6. R32 blocked halves (2.16) — the raced R32 pair
VFFT_CX_WING=1 VFFT_CX_LAZYLOAD=1 VFFT_CX_LAZYSTORE=1 \
  $G 32 --cil-t2 --cil-tangent --cil-blocked --cil-split 2.16 $U > $D/_r32_t2b.c 2>$D/_e5 || echo "FAIL r32 t2b"
VFFT_CX_WING=1 VFFT_CX_LAZYLOAD=1 VFFT_CX_LAZYSTORE=1 \
  $G 32 --cil-n1t --cil-tangent --cil-blocked --cil-split 2.16 $U > $D/_r32_n1tb.c 2>$D/_e6 || echo "FAIL r32 n1tb"
# 7. R16 mono T2 BACKWARD (wing is fwd-only => plain tangent path)
VFFT_CX_LAZYLOAD=1 \
  $G 16 --cil-t2 --cil-tangent --cil-bwd $U > $D/_r16_t2_bwd.c 2>$D/_e7 || echo "FAIL r16 t2 bwd"
wc -l $D/_*.c
