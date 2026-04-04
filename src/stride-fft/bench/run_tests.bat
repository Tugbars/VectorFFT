@echo off
setlocal enabledelayedexpansion

REM -- Resolve paths relative to this script (bench/) --
set "BENCH_DIR=%~dp0"
for %%I in ("%BENCH_DIR%..\..\..") do set "ROOT=%%~fI"
set "BIN_DIR=%ROOT%\build\bin"
set "BIN_DIR2=%ROOT%\build\src\stride-fft"

REM -- Copy FFTW DLL if needed --
if exist "%ROOT%\vcpkg_installed\x64-windows\bin\fftw3.dll" (
    copy /y "%ROOT%\vcpkg_installed\x64-windows\bin\fftw3.dll" "%BIN_DIR%\" >nul 2>&1
    copy /y "%ROOT%\vcpkg_installed\x64-windows\bin\fftw3.dll" "%BIN_DIR2%\" >nul 2>&1
)

REM -- Run tests --
echo ========================================
echo  Log3 Exhaustive Correctness
echo ========================================
if exist "%BIN_DIR2%\vfft_test_log3.exe" (
    "%BIN_DIR2%\vfft_test_log3.exe"
) else if exist "%BIN_DIR%\vfft_test_log3.exe" (
    "%BIN_DIR%\vfft_test_log3.exe"
) else (
    echo [SKIP] vfft_test_log3.exe not found
)
echo.

echo ========================================
echo  Bluestein Standalone
echo ========================================
if exist "%BIN_DIR%\vfft_bench_bluestein.exe" (
    "%BIN_DIR%\vfft_bench_bluestein.exe"
) else (
    echo [SKIP] vfft_bench_bluestein.exe not found
)
echo.

echo ========================================
echo  Rader Standalone
echo ========================================
if exist "%BIN_DIR%\vfft_bench_rader.exe" (
    "%BIN_DIR%\vfft_bench_rader.exe"
) else (
    echo [SKIP] vfft_bench_rader.exe not found
)
echo.

echo ========================================
echo  R=32 Blocked Test
echo ========================================
if exist "%BIN_DIR2%\vfft_bench_r32.exe" (
    "%BIN_DIR2%\vfft_bench_r32.exe"
) else if exist "%BIN_DIR%\vfft_bench_r32.exe" (
    "%BIN_DIR%\vfft_bench_r32.exe"
) else (
    echo [SKIP] vfft_bench_r32.exe not found
)
echo.

echo [VectorFFT] All tests complete.
cd /d "%BENCH_DIR%"
