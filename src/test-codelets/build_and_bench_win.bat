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
echo VectorFFT Codelet Benchmark Suite - Windows/ICX
echo FFTW: %VCPKG%
echo.

if not exist "%HDR_DIR%" mkdir "%HDR_DIR%"

REM -- Step 1: Generate codelet headers --

set "PYTHONIOENCODING=utf-8"

echo Generating codelet headers (AVX2 only)...

python "%SCRIPT_DIR%gen_radix2.py" avx2 > "%HDR_DIR%\fft_radix2_avx2.h"

python "%SCRIPT_DIR%gen_radix4.py" avx2 > "%HDR_DIR%\fft_radix4_avx2.h"

python "%SCRIPT_DIR%gen_radix8.py" avx2 > "%HDR_DIR%\fft_radix8_avx2.h"

python "%SCRIPT_DIR%gen_radix16.py" --isa avx2 --variant notw > "%HDR_DIR%\fft_radix16_avx2_notw.h" 2>nul
python "%SCRIPT_DIR%gen_radix16.py" --isa avx2 --variant dit_tw > "%HDR_DIR%\fft_radix16_avx2_dit_tw.h" 2>nul
python "%SCRIPT_DIR%gen_radix16.py" --isa avx2 --variant dif_tw > "%HDR_DIR%\fft_radix16_avx2_dif_tw.h" 2>nul
python "%SCRIPT_DIR%gen_radix16.py" --isa avx2 --variant dit_tw_log3 > "%HDR_DIR%\fft_radix16_avx2_dit_tw_log3.h" 2>nul
python "%SCRIPT_DIR%gen_radix16.py" --isa avx2 --variant dif_tw_log3 > "%HDR_DIR%\fft_radix16_avx2_dif_tw_log3.h" 2>nul
python "%SCRIPT_DIR%gen_radix16.py" --isa avx2 --variant ct_n1 > "%HDR_DIR%\fft_radix16_avx2_ct_n1.h" 2>nul
python "%SCRIPT_DIR%gen_radix16.py" --isa avx2 --variant ct_t1_dit > "%HDR_DIR%\fft_radix16_avx2_ct_t1_dit.h" 2>nul

python "%SCRIPT_DIR%gen_radix32.py" --isa avx2 --variant notw > "%HDR_DIR%\fft_radix32_avx2_notw.h" 2>nul
python "%SCRIPT_DIR%gen_radix32.py" --isa avx2 --variant dit_tw > "%HDR_DIR%\fft_radix32_avx2_dit_tw.h" 2>nul
python "%SCRIPT_DIR%gen_radix32.py" --isa avx2 --variant dif_tw > "%HDR_DIR%\fft_radix32_avx2_dif_tw.h" 2>nul
python "%SCRIPT_DIR%gen_radix32.py" --isa avx2 --variant ladder > "%HDR_DIR%\fft_radix32_avx2_ladder.h" 2>nul

python "%SCRIPT_DIR%gen_radix64.py" --isa avx2 --variant all > "%HDR_DIR%\r64_unified_avx2.h" 2>nul

copy "%SCRIPT_DIR%fft_n1_k1.h" "%HDR_DIR%\" >nul
python "%SCRIPT_DIR%gen_n1_k1_simd.py" all > "%HDR_DIR%\fft_n1_k1_simd.h"

echo   Headers generated.
echo.

REM -- Step 2: Compile benchmarks --

set "CC=icx"
set "CFLAGS=-O3 -march=native -mavx2 -mfma -D_USE_MATH_DEFINES"

echo Compiling benchmarks with ICX...

%CC% %CFLAGS% -I"%HDR_DIR%" -I"%FFTW_INC%" -o "%BUILD_DIR%\bench_honest_all.exe" "%SCRIPT_DIR%bench_honest_all.c" "%FFTW_LIB%"
if errorlevel 1 ( echo FAILED: bench_honest_all & exit /b 1 )
echo   bench_honest_all OK

%CC% %CFLAGS% -I"%HDR_DIR%" -I"%FFTW_INC%" -o "%BUILD_DIR%\bench_r8_log3.exe" "%SCRIPT_DIR%bench_r8_log3.c" "%FFTW_LIB%"
if errorlevel 1 ( echo FAILED: bench_r8_log3 & exit /b 1 )
echo   bench_r8_log3 OK

%CC% %CFLAGS% -I"%HDR_DIR%" -I"%FFTW_INC%" -o "%BUILD_DIR%\bench_r16_log3.exe" "%SCRIPT_DIR%bench_r16_log3.c" "%FFTW_LIB%"
if errorlevel 1 ( echo FAILED: bench_r16_log3 & exit /b 1 )
echo   bench_r16_log3 OK

%CC% %CFLAGS% -I"%HDR_DIR%" -I"%FFTW_INC%" -o "%BUILD_DIR%\bench_r32_ladder_honest.exe" "%SCRIPT_DIR%bench_r32_ladder_honest.c" "%FFTW_LIB%"
if errorlevel 1 ( echo FAILED: bench_r32_ladder_honest & exit /b 1 )
echo   bench_r32_ladder_honest OK

%CC% %CFLAGS% -I"%HDR_DIR%" -I"%FFTW_INC%" -o "%BUILD_DIR%\bench_k1.exe" "%SCRIPT_DIR%bench_k1.c" "%FFTW_LIB%"
if errorlevel 1 ( echo FAILED: bench_k1 & exit /b 1 )
echo   bench_k1 OK

%CC% %CFLAGS% -I"%HDR_DIR%" -I"%FFTW_INC%" -I"%SCRIPT_DIR%." -o "%BUILD_DIR%\bench_r64.exe" "%SCRIPT_DIR%bench_r64.c" "%FFTW_LIB%"
if errorlevel 1 ( echo FAILED: bench_r64 & exit /b 1 )
echo   bench_r64 OK

%CC% %CFLAGS% -I"%HDR_DIR%" -I"%FFTW_INC%" -I"%SCRIPT_DIR%." -o "%BUILD_DIR%\bench_r64_blocked.exe" "%SCRIPT_DIR%bench_r64_blocked.c" "%FFTW_LIB%"
if errorlevel 1 ( echo FAILED: bench_r64_blocked & exit /b 1 )
echo   bench_r64_blocked OK

%CC% %CFLAGS% -I"%HDR_DIR%" -I"%FFTW_INC%" -o "%BUILD_DIR%\bench_sv.exe" "%SCRIPT_DIR%bench_sv.c" "%FFTW_LIB%"
if errorlevel 1 ( echo FAILED: bench_sv & exit /b 1 )
echo   bench_sv OK

%CC% %CFLAGS% -I"%HDR_DIR%" -I"%FFTW_INC%" -o "%BUILD_DIR%\bench_fftw_style.exe" "%SCRIPT_DIR%bench_fftw_style.c" "%FFTW_LIB%"
if errorlevel 1 ( echo FAILED: bench_fftw_style & exit /b 1 )
echo   bench_fftw_style OK

%CC% %CFLAGS% -I"%HDR_DIR%" -I"%FFTW_INC%" -o "%BUILD_DIR%\bench_recursive_ct.exe" "%SCRIPT_DIR%bench_recursive_ct.c" "%FFTW_LIB%"
if errorlevel 1 ( echo FAILED: bench_recursive_ct & exit /b 1 )
echo   bench_recursive_ct OK

echo.

REM -- Step 3: Run benchmarks --

set "PATH=%FFTW_BIN%;%PATH%"
set "OUT=%BUILD_DIR%\results.txt"

echo Running bench_recursive_ct only...
echo.> "%OUT%"

echo ============================================================>> "%OUT%"
echo   VectorFFT Codelet Benchmark Results>> "%OUT%"
echo   %DATE% %TIME%>> "%OUT%"
echo   Compiler: ICX (Intel oneAPI)>> "%OUT%"
echo ============================================================>> "%OUT%"
echo.>> "%OUT%"

REM Skip benchmarks 1-9 for fast iteration
goto :run_ct

echo   [1/8] bench_honest_all...
echo BENCHMARK 1: bench_honest_all>> "%OUT%"
"%BUILD_DIR%\bench_honest_all.exe" >> "%OUT%" 2>&1
echo.>> "%OUT%"

echo   [2/8] bench_r8_log3...
echo BENCHMARK 2: bench_r8_log3>> "%OUT%"
"%BUILD_DIR%\bench_r8_log3.exe" >> "%OUT%" 2>&1
echo.>> "%OUT%"

echo   [3/8] bench_r16_log3...
echo BENCHMARK 3: bench_r16_log3>> "%OUT%"
"%BUILD_DIR%\bench_r16_log3.exe" >> "%OUT%" 2>&1
echo.>> "%OUT%"

echo   [4/8] bench_r32_ladder_honest...
echo BENCHMARK 4: bench_r32_ladder_honest>> "%OUT%"
"%BUILD_DIR%\bench_r32_ladder_honest.exe" >> "%OUT%" 2>&1
echo.>> "%OUT%"

echo   [5/8] bench_k1...
echo BENCHMARK 5: bench_k1>> "%OUT%"
"%BUILD_DIR%\bench_k1.exe" >> "%OUT%" 2>&1
echo.>> "%OUT%"

echo   [6/8] bench_r64...
echo BENCHMARK 6: bench_r64>> "%OUT%"
"%BUILD_DIR%\bench_r64.exe" >> "%OUT%" 2>&1
echo.>> "%OUT%"

echo   [7/8] bench_r64_blocked...
echo BENCHMARK 7: bench_r64_blocked>> "%OUT%"
"%BUILD_DIR%\bench_r64_blocked.exe" >> "%OUT%" 2>&1
echo.>> "%OUT%"

echo   [8/9] bench_sv...
echo BENCHMARK 8: bench_sv>> "%OUT%"
"%BUILD_DIR%\bench_sv.exe" >> "%OUT%" 2>&1
echo.>> "%OUT%"

echo   [9/10] bench_fftw_style...
echo BENCHMARK 9: bench_fftw_style>> "%OUT%"
"%BUILD_DIR%\bench_fftw_style.exe" >> "%OUT%" 2>&1
echo.>> "%OUT%"

:run_ct
echo   [10/10] bench_recursive_ct...
echo BENCHMARK 10: bench_recursive_ct>> "%OUT%"
"%BUILD_DIR%\bench_recursive_ct.exe" >> "%OUT%" 2>&1
echo.>> "%OUT%"

echo.
echo ============================================================
echo   Results saved to: %OUT%
echo ============================================================
echo.
type "%OUT%"

endlocal
