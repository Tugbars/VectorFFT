#!/usr/bin/env bash
# ==========================================================================
# VectorFFT — Linux/macOS build script
#
# If this script fails, here is what it does step by step:
#
#   Prerequisites:
#     1. Install a C compiler: gcc, clang, or Intel ICX
#        - Ubuntu/Debian:  sudo apt install build-essential
#        - Fedora:         sudo dnf install gcc
#        - macOS:          xcode-select --install
#        - Intel oneAPI:   https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html
#     2. Install CMake 3.16+:  sudo apt install cmake  (or https://cmake.org/download/)
#     3. Install Git:          sudo apt install git
#     4. Install Ninja:        pip3 install ninja
#        Then find where it went:  python3 -c "import ninja; print(ninja.BIN_DIR)"
#        Add that directory to your PATH.
#
#   Build steps:
#     1. Clone vcpkg:              git clone https://github.com/microsoft/vcpkg.git vcpkg
#     2. Bootstrap it:             ./vcpkg/bootstrap-vcpkg.sh -disableMetrics
#     3. Install FFTW3 with AVX2:  ./vcpkg/vcpkg install --triplet x64-linux
#        (reads vcpkg.json in project root automatically)
#     4. Configure:
#          mkdir -p build && cd build
#          cmake .. -G Ninja -DCMAKE_C_COMPILER=gcc -DCMAKE_BUILD_TYPE=Release \
#                -DCMAKE_PREFIX_PATH="../vcpkg_installed/x64-linux"
#     5. Build:                    cmake --build . --config Release
#     6. Run benchmark:
#          LD_LIBRARY_PATH="../vcpkg_installed/x64-linux/lib" ./test/bench_full_fft
#
#   Troubleshooting:
#     - "Ninja not found": Ninja may be installed but not on PATH.
#       Check: pip3 show ninja / python3 -c "import ninja; print(ninja.BIN_DIR)"
#       You can also pass -DCMAKE_MAKE_PROGRAM=<path-to-ninja> to cmake.
#     - "vcpkg install failed" with empty log: vcpkg binary may be corrupt (0 bytes).
#       Fix: cd vcpkg && git pull && ./bootstrap-vcpkg.sh -disableMetrics
# ==========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

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
    pip3 install ninja
    NINJA_DIR=$(python3 -c "import ninja; print(ninja.BIN_DIR)")
    export PATH="$PATH:$NINJA_DIR"
    if ! command -v ninja &>/dev/null; then
        echo "[ERROR] Could not install Ninja. Install manually: pip install ninja"
        exit 1
    fi
fi

# ── Bootstrap vcpkg if needed ──────────────────────────────────────────────
# Re-bootstrap if vcpkg binary is 0 bytes (corrupt download)
if [ -f "vcpkg/vcpkg" ] && [ ! -s "vcpkg/vcpkg" ]; then
    rm -f "vcpkg/vcpkg"
fi
if [ ! -x "vcpkg/vcpkg" ]; then
    if [ ! -f "vcpkg/bootstrap-vcpkg.sh" ]; then
        echo "[VectorFFT] Cloning vcpkg..."
        git clone https://github.com/microsoft/vcpkg.git vcpkg
    fi
    echo "[VectorFFT] Bootstrapping vcpkg..."
    ./vcpkg/bootstrap-vcpkg.sh -disableMetrics
fi

# ── Detect vcpkg triplet ──────────────────────────────────────────────────
ARCH=$(uname -m)
OS=$(uname -s)
case "$OS" in
    Linux)  TRIPLET="${ARCH/x86_64/x64}-linux" ;;
    Darwin) TRIPLET="${ARCH/x86_64/x64}-osx" ;;
    *)      TRIPLET="x64-linux" ;;
esac
echo "[INFO] vcpkg triplet: $TRIPLET"

# ── Install FFTW3 with AVX2 via vcpkg manifest ────────────────────────────
echo "[VectorFFT] Installing dependencies via vcpkg (FFTW3 with AVX2)..."
./vcpkg/vcpkg install --triplet "$TRIPLET"

# ── Configure ──────────────────────────────────────────────────────────────
echo "[VectorFFT] Configuring with CMake..."
mkdir -p build
cd build

# vcpkg_installed path for CMAKE_PREFIX_PATH (library discovery)
VCPKG_INSTALLED="$SCRIPT_DIR/vcpkg_installed/$TRIPLET"

cmake .. -G Ninja \
    -DCMAKE_C_COMPILER="$CC" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$VCPKG_INSTALLED"

# ── Build ──────────────────────────────────────────────────────────────────
echo "[VectorFFT] Building..."
cmake --build . --config Release

# ── Set up runtime library path ───────────────────────────────────────────
FFTW_LIB_DIR="$VCPKG_INSTALLED/lib"
if [ -d "$FFTW_LIB_DIR" ]; then
    echo
    echo "[INFO] To run benchmarks, add FFTW to your library path:"
    echo "  export LD_LIBRARY_PATH=\"$FFTW_LIB_DIR:\$LD_LIBRARY_PATH\""
    echo "  or run:"
    echo "  LD_LIBRARY_PATH=\"$FFTW_LIB_DIR\" ./build/test/bench_full_fft"
fi

echo
echo "[VectorFFT] Build complete!"
echo "  Benchmark: build/test/bench_full_fft"
cd ..
