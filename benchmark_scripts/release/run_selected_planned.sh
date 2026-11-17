#!/bin/sh
# run_selected_planned.sh -- VectorFFT against FFTW on the 18 transforms with the plans of
# VectorFFT's planner, which measures: about 2 minutes, plus up to about 20 the first time on this CPU (README.txt).
# Options go to the program:  ./run_selected_planned.sh --cells 1000,4096
# FFTW is the system's: install libfftw3 (Debian / Ubuntu: sudo apt install libfftw3-3 libfftw3-single3;
# Fedora: sudo dnf install fftw-libs).
cd "$(dirname "$0")" || exit 1
echo
echo " VectorFFT against FFTW: the 18 transforms with the plans of VectorFFT's planner, about 2 minutes."
echo " Transforms new to this CPU are planned first and kept: up to about 20 more minutes the first time."
echo " Close other programs while it runs."
echo " The first run takes longer: FFTW measures its own plans once and keeps them."
echo
if [ ! -x ./vectorfft_vs_fftw_selected_planned ]; then
    echo " vectorfft_vs_fftw_selected_planned is not beside this file: extract the whole download into one folder first."
    exit 2
fi
exec ./vectorfft_vs_fftw_selected_planned "$@"
