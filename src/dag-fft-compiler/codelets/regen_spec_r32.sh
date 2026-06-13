#!/usr/bin/env bash
# Regenerate the stride-specialized radix-32 codelets for the 32x32 one-call
# engine. Three optimizations are baked in:
#   --oop-strides L,G,OL,OG : strides as compile-time constants (folds address
#                             arithmetic to constant displacements, frees regs)
#   --fuse 8                : M-project, keep trailing PASS-2 sub-DFTs register
#                             resident across the pass boundary
#   --oop-store-fused       : store-on-compute, write each output to memory the
#                             moment it is final instead of accumulating into
#                             out_lane_* and storing in a separate phase
# Strides for this engine's call sites (R=32, V=8):
#   leaf n1 : in_leg=256 in_grp=1 out_leg=8   out_grp=32   (R*V,1,V,R)
#   t1p     : in_leg=256 in_grp=1 out_leg=256 out_grp=1    (in-place twiddle)
# radix-32 leaf result: 90 stack spills / 189 vmovapd (vs 162 / 342 unoptimized,
# below even the in-place codelet's 107 / 238). relerr ~8e-15.
set -e
GEN="${GEN:-../generator/_build/default/bin/gen_radix.exe}"
OOP="--oop --oop-buffer-oop --oop-load UG --oop-store UG --isa avx512 --emit-c"
OPT="--fuse 8 --oop-store-fused"
$GEN 32 $OOP $OPT --oop-strides 256,1,8,32                       > radix32_n1_oop_avx512_spec.c
$GEN 32 $OOP $OPT --oop-strides 256,1,256,1 --twiddled-pos        > radix32_t1p_oop_avx512_spec.c
$GEN 32 $OOP $OPT --oop-strides 256,1,256,1 --twiddled-pos --log3 > radix32_t1p_log3_oop_avx512_spec.c
echo "regenerated 3 specialized r32 codelets (stride-specialized + M-project + store-on-compute)"
