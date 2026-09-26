#!/bin/bash
# Build the two ZTURN-T calibration binaries used for zil_avx512_design.md §11.7.
# Inputs: the repo (avx2 kernels + fused drivers), and an AVX-512 kernel set
# emitted by a generator patched with ../../zsplit_generator.diff:
#   K512  = dir with radix*_z_{t0tp,tmg,tlf,tlfi,tld,t0d,tmgd,msz..}_avx512.c
#   F512  = dir with the avx512 fused drivers + ztt_registry_avx512.h
#   RT    = dir with the runtime prototype (ztt_vw.h from ../../zsplit_ztt_h_runtime.diff,
#           il_reg_kinds.h, reg2.h, reg512.h, tw_exact.h, ztt_qw16384.h; see ../ztt_msz/)
# MKL from MKLROOT (pip install mkl-devel works). Run pinned on an idle machine.
set -e
R=${REPO:-$(git rev-parse --show-toplevel)}/src/dag-fft-compiler
: "${K512:?}" "${F512:?}" "${RT:?}" "${MKLROOT:?}"
W=${W:-./out}; mkdir -p $W/o2 $W/o512
ls $R/codelets/zil/avx2/ztt/*.c $R/codelets/zil/avx2/flat/odd_mid/*.c $R/generator/generated/fused_codelets/*.c |
  xargs -P4 -I{} sh -c 'gcc -O3 -march=native -mno-avx512f -mavx2 -mfma -w -I'"$R"'/generator/generated -c {} -o '"$W"'/o2/$(basename {} .c).o'
ls $K512/*.c $F512/*.c | xargs -P4 -I{} sh -c 'gcc -O3 -march=native -w -I'"$F512"' -c {} -o '"$W"'/o512/$(basename {} .c).o'
M="-I$MKLROOT/include -L$MKLROOT/lib -Wl,-rpath,$MKLROOT/lib -lmkl_rt -lm"
gcc -O2 -march=native -mno-avx512f -DVFFT_IL_VW=4 -DVFFT_ISA_SFX=avx2 -DVFFT_ZTT_REGISTRY_H='"reg2.h"' \
    -DVFFT_IL_REGISTRY_H='"il_reg_kinds.h"' -I$RT -I$R/generator/generated zttcal.c $W/o2/*.o $M -o $W/zttcal_avx2
gcc -O2 -march=native -DVFFT_IL_VW=8 -DVFFT_ISA_SFX=avx512 -DVFFT_ZTT_REGISTRY_H='"reg512.h"' \
    -DVFFT_IL_REGISTRY_H='"il_reg_kinds.h"' -I$RT -I$F512 zttcal.c $W/o512/*.o $M -o $W/zttcal_avx512
