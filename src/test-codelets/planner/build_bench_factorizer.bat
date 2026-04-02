@echo off
setlocal

set "PLANNER_DIR=%~dp0"
:: Resolve CODELET_DIR to absolute
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
echo VectorFFT Factorizer Comparison Benchmark
echo.

if not exist "%HDR_DIR%" mkdir "%HDR_DIR%"

set "PYTHONIOENCODING=utf-8"
echo Generating ALL codelet headers...
python "%CODELET_DIR%\gen_radix2.py" avx2 > "%HDR_DIR%\fft_radix2_avx2.h" 2>nul
python "%CODELET_DIR%\gen_radix3.py" --isa avx2 --variant ct_n1 > "%HDR_DIR%\fft_radix3_avx2_ct_n1.h" 2>nul
python "%CODELET_DIR%\gen_radix3.py" --isa avx2 --variant ct_t1_dit > "%HDR_DIR%\fft_radix3_avx2_ct_t1_dit.h" 2>nul
python "%CODELET_DIR%\gen_radix4.py" avx2 > "%HDR_DIR%\fft_radix4_avx2.h" 2>nul
python "%CODELET_DIR%\gen_radix5.py" --isa avx2 --variant ct_n1 > "%HDR_DIR%\fft_radix5_avx2_ct_n1.h" 2>nul
python "%CODELET_DIR%\gen_radix5.py" --isa avx2 --variant ct_t1_dit > "%HDR_DIR%\fft_radix5_avx2_ct_t1_dit.h" 2>nul
python "%CODELET_DIR%\gen_radix6.py" --isa avx2 --variant ct_n1 > "%HDR_DIR%\fft_radix6_avx2_ct_n1.h" 2>nul
python "%CODELET_DIR%\gen_radix6.py" --isa avx2 --variant ct_t1_dit > "%HDR_DIR%\fft_radix6_avx2_ct_t1_dit.h" 2>nul
python "%CODELET_DIR%\gen_radix7.py" --isa avx2 --variant ct_n1 > "%HDR_DIR%\fft_radix7_avx2_ct_n1.h" 2>nul
python "%CODELET_DIR%\gen_radix7.py" --isa avx2 --variant ct_t1_dit > "%HDR_DIR%\fft_radix7_avx2_ct_t1_dit.h" 2>nul
python "%CODELET_DIR%\gen_radix8.py" avx2 > "%HDR_DIR%\fft_radix8_avx2.h" 2>nul
python "%CODELET_DIR%\gen_radix10.py" --isa avx2 --variant ct_n1 > "%HDR_DIR%\fft_radix10_avx2_ct_n1.h" 2>nul
python "%CODELET_DIR%\gen_radix10.py" --isa avx2 --variant ct_t1_dit > "%HDR_DIR%\fft_radix10_avx2_ct_t1_dit.h" 2>nul
python "%CODELET_DIR%\gen_radix11.py" --isa avx2 --variant ct_n1 > "%HDR_DIR%\fft_radix11_avx2_ct_n1.h" 2>nul
python "%CODELET_DIR%\gen_radix11.py" --isa avx2 --variant ct_t1_dit > "%HDR_DIR%\fft_radix11_avx2_ct_t1_dit.h" 2>nul
python "%CODELET_DIR%\gen_radix12.py" --isa avx2 --variant ct_n1 > "%HDR_DIR%\fft_radix12_avx2_ct_n1.h" 2>nul
python "%CODELET_DIR%\gen_radix12.py" --isa avx2 --variant ct_t1_dit > "%HDR_DIR%\fft_radix12_avx2_ct_t1_dit.h" 2>nul
python "%CODELET_DIR%\gen_radix13.py" --isa avx2 --variant ct_n1 > "%HDR_DIR%\fft_radix13_avx2_ct_n1.h" 2>nul
python "%CODELET_DIR%\gen_radix13.py" --isa avx2 --variant ct_t1_dit > "%HDR_DIR%\fft_radix13_avx2_ct_t1_dit.h" 2>nul
python "%CODELET_DIR%\gen_radix16.py" --isa avx2 --variant ct_n1 > "%HDR_DIR%\fft_radix16_avx2_ct_n1.h" 2>nul
python "%CODELET_DIR%\gen_radix16.py" --isa avx2 --variant ct_t1_dit > "%HDR_DIR%\fft_radix16_avx2_ct_t1_dit.h" 2>nul
python "%CODELET_DIR%\gen_radix17.py" --isa avx2 --variant ct_n1 > "%HDR_DIR%\fft_radix17_avx2_ct_n1.h" 2>nul
python "%CODELET_DIR%\gen_radix17.py" --isa avx2 --variant ct_t1_dit > "%HDR_DIR%\fft_radix17_avx2_ct_t1_dit.h" 2>nul
python "%CODELET_DIR%\gen_radix19.py" --isa avx2 --variant ct_n1 > "%HDR_DIR%\fft_radix19_avx2_ct_n1.h" 2>nul
python "%CODELET_DIR%\gen_radix19.py" --isa avx2 --variant ct_t1_dit > "%HDR_DIR%\fft_radix19_avx2_ct_t1_dit.h" 2>nul
python "%CODELET_DIR%\gen_radix20.py" --isa avx2 --variant ct_n1 > "%HDR_DIR%\fft_radix20_avx2_ct_n1.h" 2>nul
python "%CODELET_DIR%\gen_radix20.py" --isa avx2 --variant ct_t1_dit > "%HDR_DIR%\fft_radix20_avx2_ct_t1_dit.h" 2>nul
python "%CODELET_DIR%\gen_radix25.py" --isa avx2 --variant ct_n1 > "%HDR_DIR%\fft_radix25_avx2_ct_n1.h" 2>nul
python "%CODELET_DIR%\gen_radix25.py" --isa avx2 --variant ct_t1_dit > "%HDR_DIR%\fft_radix25_avx2_ct_t1_dit.h" 2>nul
python "%CODELET_DIR%\gen_radix32.py" --isa avx2 --variant ct_n1 > "%HDR_DIR%\fft_radix32_avx2_ct_n1.h" 2>nul
python "%CODELET_DIR%\gen_radix32.py" --isa avx2 --variant ct_t1_dit > "%HDR_DIR%\fft_radix32_avx2_ct_t1_dit.h" 2>nul
python "%CODELET_DIR%\gen_radix64.py" --isa avx2 --variant ct_n1 > "%HDR_DIR%\fft_radix64_avx2_ct_n1.h" 2>nul
python "%CODELET_DIR%\gen_radix64.py" --isa avx2 --variant ct_t1_dit > "%HDR_DIR%\fft_radix64_avx2_ct_t1_dit.h" 2>nul
echo   Done.
echo.

echo Compiling...
icx -O3 -march=native -mavx2 -mfma -D_USE_MATH_DEFINES -D_CRT_SECURE_NO_WARNINGS "-I%HDR_DIR%" "-I%FFTW_INC%" "-I%CODELET_DIR%" "-I%PLANNER_DIR%" -o "%BUILD_DIR%\bench_factorizer.exe" "%PLANNER_DIR%bench_factorizer.c" "%FFTW_LIB%"
if errorlevel 1 ( echo FAILED & exit /b 1 )
echo   OK
echo.

set "PATH=%FFTW_BIN%;%PATH%"
echo Running...
echo.
"%BUILD_DIR%\bench_factorizer.exe"

endlocal
