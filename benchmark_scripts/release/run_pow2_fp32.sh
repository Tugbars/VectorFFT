#!/bin/sh
# run_pow2_fp32.sh -- VectorFFT against FFTW on the powers of two in single precision (FP32)
# with the planner's plans: about a minute, plus about 7 the first time on this CPU (README.txt).
# Options go to the program:  ./run_pow2_fp32.sh --cells 1024,8192
# FFTW is the system's: install libfftw3 (Debian / Ubuntu: sudo apt install libfftw3-3 libfftw3-single3;
# Fedora: sudo dnf install fftw-libs).
cd "$(dirname "$0")" || exit 1
echo
echo " VectorFFT against FFTW: the powers of two in single precision (FP32), about a minute."
echo " Lengths new to this CPU are planned first and kept: about 7 more minutes the first time."
echo " Close other programs while it runs."
echo " The first run takes longer: FFTW measures its own plans once and keeps them."
echo
if [ ! -x ./vectorfft_vs_fftw_pow2_fp32 ]; then
    echo " vectorfft_vs_fftw_pow2_fp32 is not beside this file: extract the whole download into one folder first."
    exit 2
fi
exec ./vectorfft_vs_fftw_pow2_fp32 "$@"
