@echo off
REM ==========================================================================
REM VectorFFT — Windows build script
REM Requires: Intel oneAPI (icx), CMake 3.16+, Ninja, Git
REM ==========================================================================
setlocal enabledelayedexpansion

echo [VectorFFT] Windows build script
echo.

REM ── Check prerequisites ──────────────────────────────────────────────────
where icx >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Intel ICX compiler not found.
    echo         Install Intel oneAPI Base Toolkit and run from oneAPI Command Prompt,
    echo         or run: "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
    exit /b 1
)

where cmake >nul 2>&1
if errorlevel 1 (
    echo [ERROR] CMake not found. Install from https://cmake.org/download/
    exit /b 1
)

where ninja >nul 2>&1
if errorlevel 1 (
    echo [INFO] Ninja not found. Installing via pip...
    pip install ninja >nul 2>&1
    for /f "delims=" %%i in ('python -c "import ninja; print(ninja.BIN_DIR)"') do set "NINJA_DIR=%%i"
    set "PATH=!PATH!;!NINJA_DIR!"
    where ninja >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Could not install Ninja. Install manually: pip install ninja
        exit /b 1
    )
)

where git >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git not found. Install from https://git-scm.com/
    exit /b 1
)

REM ── Bootstrap vcpkg if needed ────────────────────────────────────────────
if not exist "vcpkg\vcpkg.exe" (
    echo [VectorFFT] Bootstrapping vcpkg...
    if not exist "vcpkg\bootstrap-vcpkg.bat" (
        echo [INFO] Cloning vcpkg...
        git clone https://github.com/microsoft/vcpkg.git vcpkg
    )
    call vcpkg\bootstrap-vcpkg.bat -disableMetrics
    if errorlevel 1 (
        echo [ERROR] vcpkg bootstrap failed.
        exit /b 1
    )
)

REM ── Install FFTW3 with AVX2 via vcpkg manifest ──────────────────────────
echo [VectorFFT] Installing dependencies via vcpkg (FFTW3 with AVX2)...
vcpkg\vcpkg.exe install --triplet x64-windows
if errorlevel 1 (
    echo [ERROR] vcpkg install failed.
    exit /b 1
)

REM ── Configure ────────────────────────────────────────────────────────────
echo [VectorFFT] Configuring with CMake...
if not exist build mkdir build
cd build

cmake .. -G Ninja ^
    -DCMAKE_C_COMPILER=icx ^
    -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 (
    echo [ERROR] CMake configure failed.
    exit /b 1
)

REM ── Build ────────────────────────────────────────────────────────────────
echo [VectorFFT] Building...
cmake --build . --config Release
if errorlevel 1 (
    echo [ERROR] Build failed.
    exit /b 1
)

REM ── Copy FFTW DLL next to benchmarks ─────────────────────────────────────
if exist "..\vcpkg_installed\x64-windows\bin\fftw3.dll" (
    copy /y "..\vcpkg_installed\x64-windows\bin\fftw3.dll" test\ >nul 2>&1
    echo [VectorFFT] Copied fftw3.dll to build\test\
)

echo.
echo [VectorFFT] Build complete!
echo   Benchmark: build\test\bench_full_fft.exe
cd ..
