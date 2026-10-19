#!/usr/bin/env bash
# build.sh -- build the gauntlet tools for the benchmark scripts (benchmark_scripts/README.md).
#   ./benchmark_scripts/build.sh [build_tools.py options, e.g. --isa avx512 --fftw-dll /path/libfftw3.so]
set -euo pipefail
cd "$(dirname "$0")/.."
py=python3; "$py" -c "import sys" >/dev/null 2>&1 || py=python
exec "$py" benchmark_scripts/build_tools.py "$@"
