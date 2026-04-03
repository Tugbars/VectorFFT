@echo off
setlocal

set "PLANNER_DIR=%~dp0"
for %%I in ("%PLANNER_DIR%..") do set "CODELET_DIR=%%~fI"
for %%I in ("%CODELET_DIR%\..\..") do set "PROJECT_ROOT=%%~fI"
set "BUILD_DIR=%CODELET_DIR%\build_bench"
set "HDR_DIR=%BUILD_DIR%\headers"

call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" --force >nul 2>&1

echo.
echo VectorFFT: IL vs Split-Complex R=16 Benchmark
echo.

if not exist "%HDR_DIR%" mkdir "%HDR_DIR%"

set "PYTHONIOENCODING=utf-8"
echo Generating R=16 headers...
python "%CODELET_DIR%\gen_radix16.py" --isa avx2 --variant ct_n1 > "%HDR_DIR%\fft_radix16_avx2_ct_n1.h" 2>nul
python "%CODELET_DIR%\gen_radix16.py" --isa avx2 --variant ct_n1_il > "%HDR_DIR%\fft_radix16_avx2_ct_n1_il.h" 2>nul
echo   Done.
echo.

echo Compiling...
icx -O3 -march=native -mavx2 -mfma -D_USE_MATH_DEFINES -D_CRT_SECURE_NO_WARNINGS -I%HDR_DIR% -I%CODELET_DIR% -I%PLANNER_DIR% -o %BUILD_DIR%\bench_il_vs_split.exe %PLANNER_DIR%bench_il_vs_split.c
if errorlevel 1 ( echo FAILED & exit /b 1 )
echo   OK
echo.

echo Running...
echo.
%BUILD_DIR%\bench_il_vs_split.exe

endlocal
