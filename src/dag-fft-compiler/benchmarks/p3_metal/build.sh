#!/bin/bash
# P3 metal build. Run from repo root: bash benchmarks/p3_metal/build.sh
# Requires: gcc with AVX-512 (or edit ISA), FFTW3 (-lfftw3), optionally MKL.
set -e
ISA="-mavx512f -mavx512dq -mfma"
mkdir -p /tmp/p3_o
for f in codelets/rfft/avx512/*.c; do
  o=/tmp/p3_o/$(basename "$f" .c).o
  [ -f "$o" ] || gcc -O3 $ISA -c "$f" -o "$o"
done
OBJS=$(ls /tmp/p3_o/*.o)
MKL=""
if [ -n "$MKLROOT" ] || [ -f /usr/local/lib/libmkl_rt.so ]; then
  MKL="-DUSE_MKL -I${MKLROOT:-/usr/local}/include -L${MKLROOT:-/usr/local}/lib -lmkl_rt -Wl,-rpath,${MKLROOT:-/usr/local}/lib -ldl"
fi
for V in base ranged prefetch ranged_prefetch; do
  FLAGS=""
  [[ $V == *ranged* ]] && FLAGS="$FLAGS -DVFFT_RFFT_RANGED=1"
  [[ $V == *prefetch* ]] && FLAGS="$FLAGS -DVFFT_RFFT_PREFETCH=1"
  gcc -O3 $ISA $FLAGS -Icore benchmarks/p3_metal/bench_p3.c $OBJS \
      -lfftw3 $MKL -lm -o /tmp/p3_$V
  echo "built /tmp/p3_$V"
done
