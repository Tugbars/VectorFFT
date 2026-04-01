@echo off
setlocal EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%..\.."
set "BUILD_DIR=%SCRIPT_DIR%build_bench"
set "HDR_DIR=%BUILD_DIR%\headers"

REM -- Find oneAPI setvars --
set "ONEAPI_ROOT=C:\Program Files (x86)\Intel\oneAPI"
if not exist "!ONEAPI_ROOT!\setvars.bat" echo ERROR: Intel oneAPI not found at !ONEAPI_ROOT! & exit /b 1
echo Setting up Intel oneAPI environment...
call "!ONEAPI_ROOT!\setvars.bat" --force >nul 2>&1

REM -- Find FFTW via vcpkg --
set "VCPKG=!PROJECT_ROOT!\build\vcpkg_installed\x64-windows"
set "FFTW_INC=!VCPKG!\include"
set "FFTW_LIB=!VCPKG!\lib\fftw3.lib"
set "FFTW_BIN=!VCPKG!\bin"

if not exist "!FFTW_INC!\fftw3.h" echo ERROR: fftw3.h not found. Run cmake --build once so vcpkg installs FFTW. & exit /b 1

echo.
echo VectorFFT Odd-Radix Benchmark - Windows/ICX
echo FFTW: %VCPKG%
echo.

if not exist "%HDR_DIR%" mkdir "%HDR_DIR%"

REM -- Step 1: Generate odd-radix codelet headers --

set "PYTHONIOENCODING=utf-8"

echo Generating odd-radix codelet headers (AVX2)...

REM R=5
python "%SCRIPT_DIR%gen_radix5.py" --isa avx2 --variant notw > "%HDR_DIR%\fft_radix5_avx2_notw.h" 2>nul
python "%SCRIPT_DIR%gen_radix5.py" --isa avx2 --variant dit_tw > "%HDR_DIR%\fft_radix5_avx2_dit_tw.h" 2>nul
python "%SCRIPT_DIR%gen_radix5.py" --isa avx2 --variant dit_tw_log3 > "%HDR_DIR%\fft_radix5_avx2_dit_tw_log3.h" 2>nul

REM R=3
python "%SCRIPT_DIR%gen_radix3.py" --isa avx2 --variant notw > "%HDR_DIR%\fft_radix3_avx2_notw.h" 2>nul
python "%SCRIPT_DIR%gen_radix3.py" --isa avx2 --variant dit_tw > "%HDR_DIR%\fft_radix3_avx2_dit_tw.h" 2>nul
python "%SCRIPT_DIR%gen_radix3.py" --isa avx2 --variant dit_tw_log3 > "%HDR_DIR%\fft_radix3_avx2_dit_tw_log3.h" 2>nul

REM R=7
python "%SCRIPT_DIR%gen_radix7.py" --isa avx2 --variant notw > "%HDR_DIR%\fft_radix7_avx2_notw.h" 2>nul
python "%SCRIPT_DIR%gen_radix7.py" --isa avx2 --variant dit_tw > "%HDR_DIR%\fft_radix7_avx2_dit_tw.h" 2>nul
python "%SCRIPT_DIR%gen_radix7.py" --isa avx2 --variant dit_tw_log3 > "%HDR_DIR%\fft_radix7_avx2_dit_tw_log3.h" 2>nul

REM R=25
python "%SCRIPT_DIR%gen_radix25.py" --isa avx2 --variant notw > "%HDR_DIR%\fft_radix25_avx2_notw.h" 2>nul
python "%SCRIPT_DIR%gen_radix25.py" --isa avx2 --variant dit_tw > "%HDR_DIR%\fft_radix25_avx2_dit_tw.h" 2>nul
python "%SCRIPT_DIR%gen_radix25.py" --isa avx2 --variant dit_tw_log3 > "%HDR_DIR%\fft_radix25_avx2_dit_tw_log3.h" 2>nul

REM R=11
python "%SCRIPT_DIR%gen_radix11.py" --isa avx2 --variant notw > "%HDR_DIR%\fft_radix11_avx2_notw.h" 2>nul
python "%SCRIPT_DIR%gen_radix11.py" --isa avx2 --variant dit_tw > "%HDR_DIR%\fft_radix11_avx2_dit_tw.h" 2>nul
python "%SCRIPT_DIR%gen_radix11.py" --isa avx2 --variant dit_tw_log3 > "%HDR_DIR%\fft_radix11_avx2_dit_tw_log3.h" 2>nul

echo   Headers generated.
echo.

REM -- Step 2: Compile --

set "CC=icx"
set "CFLAGS=-O3 -march=native -mavx2 -mfma -D_USE_MATH_DEFINES -D_CRT_SECURE_NO_WARNINGS"

echo Compiling bench_odd_radix with ICX...
%CC% %CFLAGS% -I"%HDR_DIR%" -I"%FFTW_INC%" -I"%SCRIPT_DIR%." -o "%BUILD_DIR%\bench_odd_radix.exe" "%SCRIPT_DIR%bench_odd_radix.c" "%FFTW_LIB%"
if errorlevel 1 ( echo FAILED: bench_odd_radix & exit /b 1 )
echo   bench_odd_radix OK

echo.

REM -- Step 3: Run --

set "PATH=%FFTW_BIN%;%PATH%"

echo Running bench_odd_radix...
echo.
"%BUILD_DIR%\bench_odd_radix.exe"

endlocal
