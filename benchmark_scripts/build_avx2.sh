#!/usr/bin/env bash
# build_avx2.sh -- build the gauntlet tools for the benchmark scripts with the AVX2 codelet tree, on a CPU
# with AVX2 and FMA (benchmark_scripts/README.md). build_avx512.sh is the AVX-512 twin.
#   ./benchmark_scripts/build_avx2.sh [build_tools.py options, e.g. --fftw-dll /path/libfftw3.so]
set -euo pipefail
cd "$(dirname "$0")/.."
py=python3; "$py" -c "import sys" >/dev/null 2>&1 || py=python
exec "$py" benchmark_scripts/build_tools.py --isa avx2 "$@"
