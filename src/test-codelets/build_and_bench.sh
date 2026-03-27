#!/bin/bash
# build_and_bench.sh — Generate all codelets, build benchmarks, run on 14900KF
#
# Prerequisites:
#   - FFTW 3.3.10 installed at $FFTW_PREFIX (default: /usr/local)
#   - Python 3
#   - GCC with AVX2+AVX-512 support
#
# Usage:
#   ./build_and_bench.sh [fftw_prefix]
#
# Output: threshold calibration data for the VectorFFT dispatch table
set -e

FFTW_PREFIX="${1:-/usr/local}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_bench"
HDR_DIR="${BUILD_DIR}/headers"

echo "═══════════════════════════════════════════════════════════"
echo "  VectorFFT Codelet Benchmark Suite"
echo "  FFTW prefix: ${FFTW_PREFIX}"
echo "═══════════════════════════════════════════════════════════"
echo ""

mkdir -p "${HDR_DIR}"

# ─── Step 1: Generate all codelet headers ────────────────────

echo ">>> Generating codelet headers..."

# R=4
for isa in avx2 avx512; do
    python3 "${SCRIPT_DIR}/gen_radix4.py" $isa > "${HDR_DIR}/fft_radix4_${isa}.h"
done

# R=8
for isa in avx2 avx512; do
    python3 "${SCRIPT_DIR}/gen_radix8.py" $isa > "${HDR_DIR}/fft_radix8_${isa}.h"
done

# R=16
for isa in avx2 avx512; do
    for var in notw dit_tw dif_tw dit_tw_log3 dif_tw_log3; do
        python3 "${SCRIPT_DIR}/gen_radix16.py" --isa $isa --variant $var \
            > "${HDR_DIR}/fft_radix16_${isa}_${var}.h" 2>/dev/null
    done
done

# R=32
for isa in avx2 avx512; do
    for var in notw dit_tw dif_tw ladder; do
        python3 "${SCRIPT_DIR}/gen_radix32.py" --isa $isa --variant $var \
            > "${HDR_DIR}/fft_radix32_${isa}_${var}.h" 2>/dev/null
    done
done

# K=1 codelets
cp "${SCRIPT_DIR}/fft_n1_k1.h" "${HDR_DIR}/"
python3 "${SCRIPT_DIR}/gen_n1_k1_simd.py" all > "${HDR_DIR}/fft_n1_k1_simd.h"

echo "  Generated $(ls ${HDR_DIR}/*.h | wc -l) headers in ${HDR_DIR}/"
echo ""

# ─── Step 2: Compile benchmarks ─────────────────────────────

CFLAGS="-O3 -march=native -mavx2 -mavx512f -mavx512dq -mfma"
LDFLAGS="-I${FFTW_PREFIX}/include -L${FFTW_PREFIX}/lib -lfftw3 -lm"

echo ">>> Compiling benchmarks..."

# Benchmark 1: All radixes honest (notw + flat DIT, AVX-512 + AVX2 vs FFTW SIMD)
gcc $CFLAGS -I"${HDR_DIR}" -o "${BUILD_DIR}/bench_honest_all" \
    "${SCRIPT_DIR}/bench_honest_all.c" $LDFLAGS
echo "  bench_honest_all OK"

# Benchmark 2: R=8 flat vs log3
gcc $CFLAGS -I"${HDR_DIR}" -o "${BUILD_DIR}/bench_r8_log3" \
    "${SCRIPT_DIR}/bench_r8_log3.c" $LDFLAGS
echo "  bench_r8_log3 OK"

# Benchmark 3: R=16 flat vs log3
gcc $CFLAGS -I"${HDR_DIR}" -o "${BUILD_DIR}/bench_r16_log3" \
    "${SCRIPT_DIR}/bench_r16_log3.c" $LDFLAGS
echo "  bench_r16_log3 OK"

# Benchmark 4: R=32 flat vs ladder
gcc $CFLAGS -I"${HDR_DIR}" -o "${BUILD_DIR}/bench_r32_ladder_honest" \
    "${SCRIPT_DIR}/bench_r32_ladder_honest.c" $LDFLAGS
echo "  bench_r32_ladder_honest OK"

# Benchmark 5: K=1 codelets
gcc $CFLAGS -I"${HDR_DIR}" -o "${BUILD_DIR}/bench_k1" \
    "${SCRIPT_DIR}/bench_k1.c" $LDFLAGS
echo "  bench_k1 OK"

echo ""

# ─── Step 3: Run benchmarks ─────────────────────────────────

export LD_LIBRARY_PATH="${FFTW_PREFIX}/lib:${LD_LIBRARY_PATH}"
OUT="${BUILD_DIR}/results.txt"

echo ">>> Running benchmarks (this takes 5-10 minutes)..."
echo "" > "$OUT"

echo "════════════════════════════════════════════════════════════" >> "$OUT"
echo "  VectorFFT Codelet Benchmark Results" >> "$OUT"
echo "  $(date)" >> "$OUT"
echo "  CPU: $(lscpu 2>/dev/null | grep 'Model name' | sed 's/.*: *//' || echo 'unknown')" >> "$OUT"
echo "  FFTW: ${FFTW_PREFIX}" >> "$OUT"
echo "════════════════════════════════════════════════════════════" >> "$OUT"
echo "" >> "$OUT"

echo "  [1/5] All radixes honest (notw + flat DIT vs FFTW SIMD)..."
echo "═══════════════════════════════════════════════════════════" >> "$OUT"
echo "  BENCHMARK 1: All radixes — our stride-K vs FFTW SIMD stride-1" >> "$OUT"
echo "═══════════════════════════════════════════════════════════" >> "$OUT"
"${BUILD_DIR}/bench_honest_all" >> "$OUT" 2>&1
echo "" >> "$OUT"

echo "  [2/5] R=8 flat vs log3..."
echo "═══════════════════════════════════════════════════════════" >> "$OUT"
echo "  BENCHMARK 2: R=8 flat vs log3 vs FFTW SIMD" >> "$OUT"
echo "═══════════════════════════════════════════════════════════" >> "$OUT"
"${BUILD_DIR}/bench_r8_log3" >> "$OUT" 2>&1
echo "" >> "$OUT"

echo "  [3/5] R=16 flat vs log3..."
echo "═══════════════════════════════════════════════════════════" >> "$OUT"
echo "  BENCHMARK 3: R=16 flat vs log3 vs FFTW SIMD" >> "$OUT"
echo "═══════════════════════════════════════════════════════════" >> "$OUT"
"${BUILD_DIR}/bench_r16_log3" >> "$OUT" 2>&1
echo "" >> "$OUT"

echo "  [4/5] R=32 flat vs ladder..."
echo "═══════════════════════════════════════════════════════════" >> "$OUT"
echo "  BENCHMARK 4: R=32 flat vs ladder vs FFTW SIMD" >> "$OUT"
echo "═══════════════════════════════════════════════════════════" >> "$OUT"
"${BUILD_DIR}/bench_r32_ladder_honest" >> "$OUT" 2>&1
echo "" >> "$OUT"

echo "  [5/5] K=1 codelets..."
echo "═══════════════════════════════════════════════════════════" >> "$OUT"
echo "  BENCHMARK 5: K=1 specialized codelets vs FFTW" >> "$OUT"
echo "═══════════════════════════════════════════════════════════" >> "$OUT"
"${BUILD_DIR}/bench_k1" >> "$OUT" 2>&1
echo "" >> "$OUT"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Results saved to: ${OUT}"
echo "════════════════════════════════════════════════════════════"
echo ""
cat "$OUT"
