#!/bin/sh
# scoped build of gen_radix with the tangent patch (never bare dune build)
export PATH="/home/tugbars/.opam/5.2.0/bin:$PATH"
cd /mnt/c/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler/generator || exit 1
DUNE_CACHE=disabled dune build bin/gen_radix.exe 2>&1 | tail -30
