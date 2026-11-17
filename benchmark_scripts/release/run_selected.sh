#!/bin/sh
# run_selected.sh -- VectorFFT against FFTW on 18 commonly used transforms (1D, 2D, 3D)
# with the estimate's plans, about 2 minutes (README.txt).
# Options go to the program:  ./run_selected.sh --cells 1000,4096
# FFTW is the system's: install libfftw3 (Debian / Ubuntu: sudo apt install libfftw3-3 libfftw3-single3;
# Fedora: sudo dnf install fftw-libs).
cd "$(dirname "$0")" || exit 1
echo
echo " VectorFFT against FFTW: 18 commonly used transforms (1D, 2D and 3D), about 2 minutes."
echo " Close other programs while it runs."
echo " The first run takes longer: FFTW measures its own plans once and keeps them."
echo
if [ ! -x ./vectorfft_vs_fftw_selected ]; then
    echo " vectorfft_vs_fftw_selected is not beside this file: extract the whole download into one folder first."
    exit 2
fi
exec ./vectorfft_vs_fftw_selected "$@"
