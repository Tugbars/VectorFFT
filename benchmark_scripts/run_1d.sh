#!/usr/bin/env bash
# run_1d.sh -- the 1D picture: the 1D complex transform at every N from 2 to 4,096, VectorFFT against FFTW (default) or MKL.
#   ./benchmark_scripts/run_1d.sh [fftw|mkl] [run_suite.py options, e.g. --yes --name-prefix NAME]
# The comparator is the first argument (default fftw); every further argument goes to run_suite.py.
set -euo pipefail
cd "$(dirname "$0")/.."
cmp=fftw
case "${1:-}" in fftw|mkl) cmp="$1"; shift ;; esac
py=python3; "$py" -c "import sys" >/dev/null 2>&1 || py=python
exec "$py" benchmark_scripts/run_suite.py --cmp "$cmp" --sets 1d "$@"
