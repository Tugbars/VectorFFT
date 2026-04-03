@echo off
setlocal

set "BENCH_DIR=%~dp0"
for %%I in ("%BENCH_DIR%..") do set "SFFT_DIR=%%~fI"
for %%I in ("%SFFT_DIR%\..\..") do set "PROJECT_ROOT=%%~fI"
set "CORE_DIR=%SFFT_DIR%\core"
set "CODELET_DIR=%SFFT_DIR%\codelets\avx2"
set "GEN_DIR=%SFFT_DIR%\generators"

call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" --force >nul 2>&1

set "VCPKG=%PROJECT_ROOT%\build\vcpkg_installed\x64-windows"
set "FFTW_INC=%VCPKG%\include"
set "FFTW_LIB=%VCPKG%\lib\fftw3.lib"
set "FFTW_BIN=%VCPKG%\bin"

echo.
echo VectorFFT Stride-FFT Benchmark
echo.

:: Generate codelets if needed
if not exist "%CODELET_DIR%\fft_radix16_avx2_ct_n1.h" (
    echo Generating codelets...
    call "%GEN_DIR%\generate_all.bat"
    echo.
)

echo Compiling bench_planner...
icx -O3 -march=native -mavx2 -mfma -D_USE_MATH_DEFINES -D_CRT_SECURE_NO_WARNINGS -I%CODELET_DIR% -I%CORE_DIR% -I%FFTW_INC% -o %BENCH_DIR%\bench_planner.exe %BENCH_DIR%\bench_planner.c %FFTW_LIB%
if errorlevel 1 ( echo FAILED & exit /b 1 )
echo   OK
echo.

set "PATH=%FFTW_BIN%;%PATH%"
echo Running...
echo.
"%BENCH_DIR%\bench_planner.exe"

endlocal
