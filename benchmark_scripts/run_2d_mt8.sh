#!/usr/bin/env bash
# run_2d_mt8.sh -- the 2D picture at eight threads: the 159 planes at eight threads beside the one-thread run (run_2d.sh first, or run both); --threads N changes the count.
#   ./benchmark_scripts/run_2d_mt8.sh [fftw|mkl] [run_suite.py options, e.g. --yes --name-prefix NAME]
# The comparator is the first argument (default fftw); every further argument goes to run_suite.py.
set -euo pipefail
cd "$(dirname "$0")/.."
cmp=fftw
case "${1:-}" in fftw|mkl) cmp="$1"; shift ;; esac
py=python3; "$py" -c "import sys" >/dev/null 2>&1 || py=python
exec "$py" benchmark_scripts/run_suite.py --cmp "$cmp" --sets 2d-mt8 "$@"
