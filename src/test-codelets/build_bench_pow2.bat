@echo off
setlocal EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%..\.."
set "BUILD_DIR=%SCRIPT_DIR%build_bench"
set "HDR_DIR=%BUILD_DIR%\headers"

set "ONEAPI_ROOT=C:\Program Files (x86)\Intel\oneAPI"
if not exist "!ONEAPI_ROOT!\setvars.bat" echo ERROR: Intel oneAPI not found & exit /b 1
echo Setting up Intel oneAPI environment...
call "!ONEAPI_ROOT!\setvars.bat" --force >nul 2>&1

set "VCPKG=!PROJECT_ROOT!\build\vcpkg_installed\x64-windows"
set "FFTW_INC=!VCPKG!\include"
set "FFTW_LIB=!VCPKG!\lib\fftw3.lib"
set "FFTW_BIN=!VCPKG!\bin"

if not exist "!FFTW_INC!\fftw3.h" echo ERROR: fftw3.h not found & exit /b 1

echo.
echo VectorFFT Power-of-2 Executor Stress Test
echo.

if not exist "%HDR_DIR%" mkdir "%HDR_DIR%"

set "PYTHONIOENCODING=utf-8"
echo Generating headers...

python "%SCRIPT_DIR%gen_radix2.py" avx2 > "%HDR_DIR%\fft_radix2_avx2.h" 2>nul
python "%SCRIPT_DIR%gen_radix4.py" avx2 > "%HDR_DIR%\fft_radix4_avx2.h" 2>nul
python "%SCRIPT_DIR%gen_radix16.py" --isa avx2 --variant ct_n1 > "%HDR_DIR%\fft_radix16_avx2_ct_n1.h" 2>nul
python "%SCRIPT_DIR%gen_radix16.py" --isa avx2 --variant ct_t1_dit > "%HDR_DIR%\fft_radix16_avx2_ct_t1_dit.h" 2>nul
python "%SCRIPT_DIR%gen_radix32.py" --isa avx2 --variant ct_n1 > "%HDR_DIR%\fft_radix32_avx2_ct_n1.h" 2>nul
python "%SCRIPT_DIR%gen_radix32.py" --isa avx2 --variant ct_t1_dit > "%HDR_DIR%\fft_radix32_avx2_ct_t1_dit.h" 2>nul
python "%SCRIPT_DIR%gen_radix64.py" --isa avx2 --variant ct_n1 > "%HDR_DIR%\fft_radix64_avx2_ct_n1.h" 2>nul
python "%SCRIPT_DIR%gen_radix64.py" --isa avx2 --variant ct_t1_dit > "%HDR_DIR%\fft_radix64_avx2_ct_t1_dit.h" 2>nul

echo   Headers generated.
echo.

set "CC=icx"
set "CFLAGS=-O3 -march=native -mavx2 -mfma -D_USE_MATH_DEFINES -D_CRT_SECURE_NO_WARNINGS"

echo Compiling bench_pow2_executor...
%CC% %CFLAGS% -I"%HDR_DIR%" -I"%FFTW_INC%" -I"%SCRIPT_DIR%." -o "%BUILD_DIR%\bench_pow2_executor.exe" "%SCRIPT_DIR%bench_pow2_executor.c" "%FFTW_LIB%"
if errorlevel 1 ( echo FAILED & exit /b 1 )
echo   OK
echo.

set "PATH=%FFTW_BIN%;%PATH%"
echo Running...
echo.
"%BUILD_DIR%\bench_pow2_executor.exe"

endlocal
