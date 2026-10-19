#!/usr/bin/env bash
# run_pow2.sh -- the quick check: the 1D complex transform at every power of two from 2 to 2^23 (23 lengths),
# VectorFFT against FFTW (default) or MKL, in minutes.
#   ./benchmark_scripts/run_pow2.sh [fftw|mkl] [run_suite.py options, e.g. --name-prefix NAME]
# The comparator is the first argument (default fftw); every further argument goes to run_suite.py.
set -euo pipefail
cd "$(dirname "$0")/.."
cmp=fftw
case "${1:-}" in fftw|mkl) cmp="$1"; shift ;; esac
py=python3; "$py" -c "import sys" >/dev/null 2>&1 || py=python
exec "$py" benchmark_scripts/run_suite.py --cmp "$cmp" --sets pow2 "$@"
