@echo off
setlocal

set "PLANNER_DIR=%~dp0"
for %%I in ("%PLANNER_DIR%..") do set "CODELET_DIR=%%~fI"
for %%I in ("%CODELET_DIR%\..\..") do set "PROJECT_ROOT=%%~fI"
set "BUILD_DIR=%CODELET_DIR%\build_bench"
set "HDR_DIR=%BUILD_DIR%\headers"

call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" --force >nul 2>&1

set "VCPKG=%PROJECT_ROOT%\build\vcpkg_installed\x64-windows"
set "FFTW_INC=%VCPKG%\include"
set "FFTW_LIB=%VCPKG%\lib\fftw3.lib"
set "FFTW_BIN=%VCPKG%\bin"

echo.
echo VectorFFT Log3 vs Flat Benchmark
echo.

if not exist "%HDR_DIR%" mkdir "%HDR_DIR%"

set "PYTHONIOENCODING=utf-8"
echo Generating R=64 headers (n1 + t1_dit + t1_dit_log3)...
python "%CODELET_DIR%\gen_radix64.py" --isa avx2 --variant ct_n1 > "%HDR_DIR%\fft_radix64_avx2_ct_n1.h" 2>nul
python "%CODELET_DIR%\gen_radix64.py" --isa avx2 --variant ct_t1_dit > "%HDR_DIR%\fft_radix64_avx2_ct_t1_dit.h" 2>nul
python "%CODELET_DIR%\gen_radix64.py" --isa avx2 --variant ct_t1_dit_log3 > "%HDR_DIR%\fft_radix64_avx2_ct_t1_dit_log3.h" 2>nul
echo   Done.
echo.

echo Compiling...
icx -O3 -march=native -mavx2 -mfma -D_USE_MATH_DEFINES -D_CRT_SECURE_NO_WARNINGS -I%HDR_DIR% -I%FFTW_INC% -I%CODELET_DIR% -I%PLANNER_DIR% -o %BUILD_DIR%\bench_log3.exe %PLANNER_DIR%bench_log3.c %FFTW_LIB%
if errorlevel 1 ( echo FAILED & exit /b 1 )
echo   OK
echo.

set "PATH=%FFTW_BIN%;%PATH%"
echo Running...
echo.
%BUILD_DIR%\bench_log3.exe

endlocal
