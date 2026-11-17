#!/bin/sh
# run_pow2.sh -- VectorFFT against FFTW on the 1D transform at every power of two from 64
# to 4,194,304 with the estimate's plans, about a minute (README.txt).
# Options go to the program:  ./run_pow2.sh --cells 1000,4096
# FFTW is the system's: install libfftw3 (Debian / Ubuntu: sudo apt install libfftw3-3 libfftw3-single3;
# Fedora: sudo dnf install fftw-libs).
cd "$(dirname "$0")" || exit 1
echo
echo " VectorFFT against FFTW: the 1D transform at every power of two from 64 to 4,194,304, about a minute."
echo " Close other programs while it runs."
echo " The first run takes longer: FFTW measures its own plans once and keeps them."
echo
if [ ! -x ./vectorfft_vs_fftw_pow2 ]; then
    echo " vectorfft_vs_fftw_pow2 is not beside this file: extract the whole download into one folder first."
    exit 2
fi
exec ./vectorfft_vs_fftw_pow2 "$@"
