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
echo VectorFFT Stride-Based Executor Prototype
echo.

if not exist "%HDR_DIR%" mkdir "%HDR_DIR%"

REM -- Generate codelet headers --
set "PYTHONIOENCODING=utf-8"
echo Generating headers...

REM R=3 ct_n1 (stride-based n1 with is/os/vl)
python "%SCRIPT_DIR%gen_radix3.py" --isa avx2 --variant ct_n1 > "%HDR_DIR%\fft_radix3_avx2_ct_n1.h" 2>nul

REM R=4 (includes t1_dit with ios/me)
python "%SCRIPT_DIR%gen_radix4.py" avx2 > "%HDR_DIR%\fft_radix4_avx2.h" 2>nul

echo   Headers generated.
echo.

REM -- Compile --
set "CC=icx"
set "CFLAGS=-O3 -march=native -mavx2 -mfma -D_USE_MATH_DEFINES -D_CRT_SECURE_NO_WARNINGS"

echo Compiling bench_stride_executor...
%CC% %CFLAGS% -I"%HDR_DIR%" -I"%FFTW_INC%" -I"%SCRIPT_DIR%." -o "%BUILD_DIR%\bench_stride_executor.exe" "%SCRIPT_DIR%bench_stride_executor.c" "%FFTW_LIB%"
if errorlevel 1 ( echo FAILED: bench_stride_executor & exit /b 1 )
echo   bench_stride_executor OK
echo.

REM -- Run --
set "PATH=%FFTW_BIN%;%PATH%"

echo Running...
echo.
"%BUILD_DIR%\bench_stride_executor.exe"

endlocal
