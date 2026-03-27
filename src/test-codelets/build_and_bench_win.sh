#!/bin/bash
# build_and_bench_win.sh — Windows version of build_and_bench.sh
#
# Prerequisites:
#   - FFTW via vcpkg (auto-detected from project build dir)
#   - Python 3
#   - Intel ICX compiler (oneAPI)
#
# Usage (from Git Bash):
#   ./build_and_bench_win.sh
#
# Output: threshold calibration data for the VectorFFT dispatch table
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_bench"
HDR_DIR="${BUILD_DIR}/headers"

# Auto-detect FFTW from vcpkg
VCPKG_INSTALLED="${PROJECT_ROOT}/build/vcpkg_installed/x64-windows"
FFTW_INC="${VCPKG_INSTALLED}/include"
FFTW_LIB="${VCPKG_INSTALLED}/lib"
FFTW_BIN="${VCPKG_INSTALLED}/bin"

if [ ! -f "${FFTW_INC}/fftw3.h" ]; then
    echo "ERROR: fftw3.h not found at ${FFTW_INC}"
    echo "Make sure you've run cmake --build once so vcpkg installs FFTW."
    exit 1
fi

echo "==================================================="
echo "  VectorFFT Codelet Benchmark Suite (Windows/ICX)"
echo "  FFTW: ${VCPKG_INSTALLED}"
echo "==================================================="
echo ""

mkdir -p "${HDR_DIR}"

# --- Step 1: Generate all codelet headers ----------------

echo ">>> Generating codelet headers..."

# R=4
for isa in avx2 avx512; do
    python "${SCRIPT_DIR}/gen_radix4.py" $isa > "${HDR_DIR}/fft_radix4_${isa}.h"
done

# R=8
for isa in avx2 avx512; do
    python "${SCRIPT_DIR}/gen_radix8.py" $isa > "${HDR_DIR}/fft_radix8_${isa}.h"
done

# R=16
for isa in avx2 avx512; do
    for var in notw dit_tw dif_tw dit_tw_log3 dif_tw_log3; do
        python "${SCRIPT_DIR}/gen_radix16.py" --isa $isa --variant $var \
            > "${HDR_DIR}/fft_radix16_${isa}_${var}.h" 2>/dev/null
    done
done

# R=32
for isa in avx2 avx512; do
    for var in notw dit_tw dif_tw ladder; do
        python "${SCRIPT_DIR}/gen_radix32.py" --isa $isa --variant $var \
            > "${HDR_DIR}/fft_radix32_${isa}_${var}.h" 2>/dev/null
    done
done

# K=1 codelets
cp "${SCRIPT_DIR}/fft_n1_k1.h" "${HDR_DIR}/"
python "${SCRIPT_DIR}/gen_n1_k1_simd.py" all > "${HDR_DIR}/fft_n1_k1_simd.h"

echo "  Generated $(ls ${HDR_DIR}/*.h | wc -l) headers in ${HDR_DIR}/"
echo ""

# --- Step 2: Compile benchmarks -------------------------

CC="icx"
CFLAGS="-O3 -march=native -D_USE_MATH_DEFINES"
INCLUDES="-I${HDR_DIR} -I${FFTW_INC}"
LIBS="${FFTW_LIB}/fftw3.lib"

echo ">>> Compiling benchmarks with ICX..."

compile_bench() {
    local name="$1"
    local src="$2"
    $CC $CFLAGS $INCLUDES -o "${BUILD_DIR}/${name}.exe" "${SCRIPT_DIR}/${src}" $LIBS
    echo "  ${name} OK"
}

# Benchmark 1: All radixes honest
compile_bench bench_honest_all bench_honest_all.c

# Benchmark 2: R=8 flat vs log3
compile_bench bench_r8_log3 bench_r8_log3.c

# Benchmark 3: R=16 flat vs log3
compile_bench bench_r16_log3 bench_r16_log3.c

# Benchmark 4: R=32 flat vs ladder
compile_bench bench_r32_ladder_honest bench_r32_ladder_honest.c

# Benchmark 5: K=1 codelets
compile_bench bench_k1 bench_k1.c

echo ""

# --- Step 3: Run benchmarks -----------------------------

# Add FFTW DLL dir to PATH so executables can find fftw3.dll
export PATH="${FFTW_BIN}:${PATH}"
OUT="${BUILD_DIR}/results.txt"

CPU_NAME=$(wmic cpu get name 2>/dev/null | sed -n '2p' | sed 's/[[:space:]]*$//' || echo 'unknown')

echo ">>> Running benchmarks (this takes 5-10 minutes)..."
echo "" > "$OUT"

echo "============================================================" >> "$OUT"
echo "  VectorFFT Codelet Benchmark Results" >> "$OUT"
echo "  $(date)" >> "$OUT"
echo "  CPU: ${CPU_NAME}" >> "$OUT"
echo "  FFTW: ${VCPKG_INSTALLED}" >> "$OUT"
echo "  Compiler: ICX (Intel oneAPI)" >> "$OUT"
echo "============================================================" >> "$OUT"
echo "" >> "$OUT"

run_bench() {
    local idx="$1"
    local total="$2"
    local label="$3"
    local exe="$4"
    echo "  [${idx}/${total}] ${label}..."
    echo "===========================================================" >> "$OUT"
    echo "  BENCHMARK ${idx}: ${label}" >> "$OUT"
    echo "===========================================================" >> "$OUT"
    "${BUILD_DIR}/${exe}.exe" >> "$OUT" 2>&1
    echo "" >> "$OUT"
}

run_bench 1 5 "All radixes — our stride-K vs FFTW SIMD stride-1" bench_honest_all
run_bench 2 5 "R=8 flat vs log3 vs FFTW SIMD" bench_r8_log3
run_bench 3 5 "R=16 flat vs log3 vs FFTW SIMD" bench_r16_log3
run_bench 4 5 "R=32 flat vs ladder vs FFTW SIMD" bench_r32_ladder_honest
run_bench 5 5 "K=1 specialized codelets vs FFTW" bench_k1

echo ""
echo "============================================================"
echo "  Results saved to: ${OUT}"
echo "============================================================"
echo ""
cat "$OUT"
