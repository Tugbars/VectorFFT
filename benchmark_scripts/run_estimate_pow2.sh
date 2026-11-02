#!/usr/bin/env bash
# run_estimate_pow2.sh -- the estimate against the planner at powers of two: the 1D, 2D and 3D complex transform,
# VFFT_FLAG_ESTIMATE's plan against the planner's banked plan, E/B per cell and a SUMMARY.md (estimate_pow2.py).
#   ./benchmark_scripts/run_estimate_pow2.sh [estimate_pow2.py options, e.g. --rounds 7 --sets 2d,3d --isa avx2]
# Builds its two tools with the AVX-512 tree by default; Windows only; run on a quiet machine.
set -euo pipefail
cd "$(dirname "$0")/.."
py=python3; "$py" -c "import sys" >/dev/null 2>&1 || py=python
exec "$py" benchmark_scripts/estimate_pow2.py "$@"
