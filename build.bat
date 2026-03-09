@echo off
REM ==========================================================================
REM VectorFFT — Windows build script
REM
REM If this script fails, here is what it does step by step:
REM
REM   Prerequisites:
REM     1. Install Intel oneAPI Base Toolkit (provides icx compiler)
REM        https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html
REM     2. Install CMake 3.16+       https://cmake.org/download/
REM     3. Install Git               https://git-scm.com/
REM     4. Install Ninja:            pip install ninja
REM        Then find where it went:  python -c "import ninja; print(ninja.BIN_DIR)"
REM        Add that directory to your PATH.
REM
REM   Build steps (run from Intel oneAPI Command Prompt):
REM     1. Clone vcpkg:              git clone https://github.com/microsoft/vcpkg.git vcpkg
REM     2. Bootstrap it:             vcpkg\bootstrap-vcpkg.bat -disableMetrics
REM     3. Install FFTW3 with AVX2:  vcpkg\vcpkg.exe install --triplet x64-windows
REM        (reads vcpkg.json in project root automatically)
REM     4. Configure:
REM          mkdir build && cd build
REM          cmake .. -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_BUILD_TYPE=Release ^
REM                -DCMAKE_PREFIX_PATH="..\vcpkg_installed\x64-windows"
REM     5. Build:                    cmake --build . --config Release
REM     6. Copy FFTW DLL:           copy ..\vcpkg_installed\x64-windows\bin\fftw3.dll test\
REM     7. Run benchmark:            test\bench_full_fft.exe
REM
REM   Troubleshooting:
REM     - "Ninja not found": Ninja may be installed but not on PATH. Known locations:
REM         * Visual Studio:  C:\Program Files\Microsoft Visual Studio\2022\Community\
REM                           Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe
REM         * pip install:    python -c "import ninja; print(ninja.BIN_DIR)"
REM       You can also pass -DCMAKE_MAKE_PROGRAM=<path-to-ninja.exe> to cmake.
REM       WARNING: Do NOT use "setx PATH" to extend PATH — it truncates to 1024 chars
REM       and can corrupt your PATH. Use the Windows Environment Variables GUI instead.
REM     - "vcpkg install failed" with empty log: vcpkg.exe may be 0 bytes (corrupt).
REM       Fix: cd vcpkg && git pull && bootstrap-vcpkg.bat -disableMetrics
REM ==========================================================================
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

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
REM Also re-bootstrap if vcpkg.exe is 0 bytes (corrupt download)
for %%F in (vcpkg\vcpkg.exe) do if %%~zF==0 del "vcpkg\vcpkg.exe"
if not exist "vcpkg\vcpkg.exe" (
    if not exist "vcpkg\bootstrap-vcpkg.bat" (
        echo [VectorFFT] Cloning vcpkg...
        git clone https://github.com/microsoft/vcpkg.git vcpkg
    )
    echo [VectorFFT] Bootstrapping vcpkg...
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

set "VCPKG_INSTALLED=%SCRIPT_DIR%vcpkg_installed\x64-windows"

cmake .. -G Ninja ^
    -DCMAKE_C_COMPILER=icx ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_PREFIX_PATH="%VCPKG_INSTALLED%"
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
if exist "%VCPKG_INSTALLED%\bin\fftw3.dll" (
    copy /y "%VCPKG_INSTALLED%\bin\fftw3.dll" test\ >nul 2>&1
    echo [VectorFFT] Copied fftw3.dll to build\test\
)

echo.
echo [VectorFFT] Build complete!
echo   Benchmark: build\test\bench_full_fft.exe
cd ..
