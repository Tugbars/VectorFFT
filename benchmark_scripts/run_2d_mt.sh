#!/usr/bin/env bash
# run_2d_mt.sh -- the 2D picture threaded: the 159 planes at T threads beside the one-thread run (run_2d.sh first, or run_all.sh); T is the physical cores of the fastest core class, --threads N overrides it.
#   ./benchmark_scripts/run_2d_mt.sh [fftw|mkl] [run_suite.py options, e.g. --yes --threads 6 --name-prefix NAME]
# The comparator is the first argument (default fftw); every further argument goes to run_suite.py.
set -euo pipefail
cd "$(dirname "$0")/.."
cmp=fftw
case "${1:-}" in fftw|mkl) cmp="$1"; shift ;; esac
py=python3; "$py" -c "import sys" >/dev/null 2>&1 || py=python
exec "$py" benchmark_scripts/run_suite.py --cmp "$cmp" --sets 2d-mt "$@"
