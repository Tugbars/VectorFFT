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
echo VectorFFT Odd-Radix CT Executor Benchmark
echo.

if not exist "%HDR_DIR%" mkdir "%HDR_DIR%"

REM -- Generate headers --
set "PYTHONIOENCODING=utf-8"
echo Generating odd-radix CT headers...

REM Odd radix CT codelets (t1_dit + n1_ovs)
python "%SCRIPT_DIR%gen_radix3.py" --isa avx2 --variant ct_n1 > "%HDR_DIR%\fft_radix3_avx2_ct_n1.h" 2>nul
python "%SCRIPT_DIR%gen_radix3.py" --isa avx2 --variant ct_t1_dit > "%HDR_DIR%\fft_radix3_avx2_ct_t1_dit.h" 2>nul
python "%SCRIPT_DIR%gen_radix5.py" --isa avx2 --variant ct_n1 > "%HDR_DIR%\fft_radix5_avx2_ct_n1.h" 2>nul
python "%SCRIPT_DIR%gen_radix5.py" --isa avx2 --variant ct_t1_dit > "%HDR_DIR%\fft_radix5_avx2_ct_t1_dit.h" 2>nul
python "%SCRIPT_DIR%gen_radix7.py" --isa avx2 --variant ct_n1 > "%HDR_DIR%\fft_radix7_avx2_ct_n1.h" 2>nul
python "%SCRIPT_DIR%gen_radix7.py" --isa avx2 --variant ct_t1_dit > "%HDR_DIR%\fft_radix7_avx2_ct_t1_dit.h" 2>nul
python "%SCRIPT_DIR%gen_radix25.py" --isa avx2 --variant ct_n1 > "%HDR_DIR%\fft_radix25_avx2_ct_n1.h" 2>nul
python "%SCRIPT_DIR%gen_radix25.py" --isa avx2 --variant ct_t1_dit > "%HDR_DIR%\fft_radix25_avx2_ct_t1_dit.h" 2>nul

REM Pow2 CT codelets (should already exist from build_and_bench_win.bat, but regenerate)
python "%SCRIPT_DIR%gen_radix16.py" --isa avx2 --variant ct_n1 > "%HDR_DIR%\fft_radix16_avx2_ct_n1.h" 2>nul
python "%SCRIPT_DIR%gen_radix16.py" --isa avx2 --variant ct_t1_dit > "%HDR_DIR%\fft_radix16_avx2_ct_t1_dit.h" 2>nul
python "%SCRIPT_DIR%gen_radix32.py" --isa avx2 --variant ct_n1 > "%HDR_DIR%\fft_radix32_avx2_ct_n1.h" 2>nul
python "%SCRIPT_DIR%gen_radix32.py" --isa avx2 --variant ct_t1_dit > "%HDR_DIR%\fft_radix32_avx2_ct_t1_dit.h" 2>nul

REM R=4 and R=8 are in the unified headers
python "%SCRIPT_DIR%gen_radix4.py" avx2 > "%HDR_DIR%\fft_radix4_avx2.h" 2>nul
python "%SCRIPT_DIR%gen_radix8.py" avx2 > "%HDR_DIR%\fft_radix8_avx2.h" 2>nul

echo   Headers generated.
echo.

REM -- Compile --
set "CC=icx"
set "CFLAGS=-O3 -march=native -mavx2 -mfma -D_USE_MATH_DEFINES -D_CRT_SECURE_NO_WARNINGS"

echo Compiling debug_ct_layout with ICX...
%CC% %CFLAGS% -I"%HDR_DIR%" -I"%FFTW_INC%" -I"%SCRIPT_DIR%." -o "%BUILD_DIR%\debug_ct_layout.exe" "%SCRIPT_DIR%debug_ct_layout.c" "%FFTW_LIB%"
if errorlevel 1 ( echo FAILED: debug_ct_layout & exit /b 1 )
echo   debug_ct_layout OK

echo Compiling bench_ct_odd with ICX...
%CC% %CFLAGS% -I"%HDR_DIR%" -I"%FFTW_INC%" -I"%SCRIPT_DIR%." -o "%BUILD_DIR%\bench_ct_odd.exe" "%SCRIPT_DIR%bench_ct_odd.c" "%FFTW_LIB%"
if errorlevel 1 ( echo FAILED: bench_ct_odd & exit /b 1 )
echo   bench_ct_odd OK
echo.

REM -- Run --
set "PATH=%FFTW_BIN%;%PATH%"

echo Running debug_ct_layout first...
echo.
"%BUILD_DIR%\debug_ct_layout.exe"

echo.
echo Running bench_ct_odd...
echo.
"%BUILD_DIR%\bench_ct_odd.exe"

endlocal
