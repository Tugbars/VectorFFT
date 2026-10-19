#!/usr/bin/env bash
# run_3d.sh -- the 3D picture: every power-of-two volume up to 4M points (1,288 volumes), VectorFFT against FFTW (default) or MKL.
#   ./benchmark_scripts/run_3d.sh [fftw|mkl] [run_suite.py options, e.g. --yes --name-prefix NAME]
# The comparator is the first argument (default fftw); every further argument goes to run_suite.py.
set -euo pipefail
cd "$(dirname "$0")/.."
cmp=fftw
case "${1:-}" in fftw|mkl) cmp="$1"; shift ;; esac
py=python3; "$py" -c "import sys" >/dev/null 2>&1 || py=python
exec "$py" benchmark_scripts/run_suite.py --cmp "$cmp" --sets 3d "$@"
