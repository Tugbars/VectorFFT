#!/usr/bin/env bash
# ==========================================================================
# VectorFFT — Linux/macOS build script
# Requires: C compiler (gcc/clang/icx), CMake 3.16+, Ninja, Git
# ==========================================================================
set -euo pipefail

echo "[VectorFFT] Build script"
echo

# ── Detect compiler ────────────────────────────────────────────────────────
if command -v icx &>/dev/null; then
    CC=icx
    echo "[INFO] Using Intel ICX compiler"
elif command -v gcc &>/dev/null; then
    CC=gcc
    echo "[INFO] Using GCC compiler"
elif command -v clang &>/dev/null; then
    CC=clang
    echo "[INFO] Using Clang compiler"
else
    echo "[ERROR] No C compiler found. Install gcc, clang, or Intel oneAPI."
    exit 1
fi

# ── Check prerequisites ───────────────────────────────────────────────────
for tool in cmake git; do
    if ! command -v "$tool" &>/dev/null; then
        echo "[ERROR] $tool not found. Please install it."
        exit 1
    fi
done

if ! command -v ninja &>/dev/null; then
    echo "[INFO] Ninja not found. Installing via pip..."
    pip install ninja
    NINJA_DIR=$(python3 -c "import ninja; print(ninja.BIN_DIR)")
    export PATH="$PATH:$NINJA_DIR"
    if ! command -v ninja &>/dev/null; then
        echo "[ERROR] Could not install Ninja. Install manually: pip install ninja"
        exit 1
    fi
fi

# ── Bootstrap vcpkg if needed ──────────────────────────────────────────────
if [ ! -x "vcpkg/vcpkg" ]; then
    if [ ! -f "vcpkg/bootstrap-vcpkg.sh" ]; then
        echo "[VectorFFT] Cloning vcpkg..."
        git clone https://github.com/microsoft/vcpkg.git vcpkg
    fi
    echo "[VectorFFT] Bootstrapping vcpkg..."
    ./vcpkg/bootstrap-vcpkg.sh -disableMetrics
fi

# ── Install FFTW3 with AVX2 via vcpkg manifest ────────────────────────────
echo "[VectorFFT] Installing dependencies via vcpkg (FFTW3 with AVX2)..."
./vcpkg/vcpkg install --triplet x64-linux
if [ $? -ne 0 ]; then
    echo "[ERROR] vcpkg install failed."
    exit 1
fi

# ── Configure ──────────────────────────────────────────────────────────────
echo "[VectorFFT] Configuring with CMake..."
mkdir -p build
cd build

cmake .. -G Ninja \
    -DCMAKE_C_COMPILER="$CC" \
    -DCMAKE_BUILD_TYPE=Release

# ── Build ──────────────────────────────────────────────────────────────────
echo "[VectorFFT] Building..."
cmake --build . --config Release

echo
echo "[VectorFFT] Build complete!"
echo "  Benchmark: build/test/bench_full_fft"
cd ..
