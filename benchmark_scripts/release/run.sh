#!/bin/sh
# run.sh -- the VectorFFT benchmark programs of this folder, from a menu (README.txt).
#   ./run.sh              the menu
#   ./run.sh 1 [options]  a choice directly, with the program's options (--cells LIST, --inplace, --avx2 / --avx512, --out DIR)
# The results land in vectorfft_results/ beside it. FFTW is the system's: install libfftw3 (Debian / Ubuntu:
# sudo apt install libfftw3-3 libfftw3-single3; Fedora: sudo dnf install fftw-libs).
cd "$(dirname "$0")" || exit 1
choice=${1:-}
if [ -z "$choice" ]; then
    echo
    echo " VectorFFT against FFTW -- one thread, complex to complex"
    echo
    echo "  1  selected transforms (1D, 2D, 3D), estimate plans          about 2 minutes   [start here]"
    echo "  2  1D powers of two from 64 to 4M, estimate plans            about a minute"
    echo "  3  selected transforms, the planner's plans                  2 minutes + planning of new transforms"
    [ -x ./vectorfft_vs_fftw_pow2_fp32 ] && echo "  4  1D powers of two in single precision, the planner's plans  a minute + planning of new lengths"
    echo
    echo " The first run against FFTW takes longer: FFTW measures its own plans once and keeps them."
    echo
    printf "Choice [1]: "
    read -r choice
    choice=${choice:-1}
else
    shift
fi
case "$choice" in
    1) prog=vectorfft_vs_fftw_selected ;;
    2) prog=vectorfft_vs_fftw_pow2 ;;
    3) prog=vectorfft_vs_fftw_selected_planned ;;
    4) prog=vectorfft_vs_fftw_pow2_fp32 ;;
    *) echo "No such choice: $choice"; exit 2 ;;
esac
if [ ! -x "./$prog" ]; then
    echo "$prog is not in this package."
    exit 2
fi
exec "./$prog" "$@"
