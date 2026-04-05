@echo off
setlocal enabledelayedexpansion

REM -- Resolve paths relative to this script (bench/) --
set "BENCH_DIR=%~dp0"
for %%I in ("%BENCH_DIR%..\..\..") do set "ROOT=%%~fI"
set "BIN_DIR=%ROOT%\build\bin"

REM -- Copy FFTW DLL if needed --
if exist "%ROOT%\vcpkg_installed\x64-windows\bin\fftw3.dll" (
    copy /y "%ROOT%\vcpkg_installed\x64-windows\bin\fftw3.dll" "%BIN_DIR%\" >nul 2>&1
)

echo ========================================
echo  Log3 Calibration + Estimate Comparison
echo ========================================
echo.
if exist "%BIN_DIR%\vfft_bench_log3_calib.exe" (
    "%BIN_DIR%\vfft_bench_log3_calib.exe"
) else (
    echo [ERROR] vfft_bench_log3_calib.exe not found in %BIN_DIR%
    echo Build it first: cmake --build build --target vfft_bench_log3_calib
)
echo.
echo Done.
cd /d "%BENCH_DIR%"
