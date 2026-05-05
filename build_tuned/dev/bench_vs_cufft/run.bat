@echo off
setlocal enabledelayedexpansion

rem ─────────────────────────────────────────────────────────────────
rem cuFFT vs VectorFFT (T=8) latency sweep
rem
rem Runs three measurements per cell (cuFFT compute-only, +D2H, full
rem round-trip) and one VectorFFT T=8 number per cell. Saves CSVs and
rem prints a summary table.
rem
rem Requires:
rem   - CUDA toolkit (nvcc on PATH or in default location)
rem   - VectorFFT vfft.lib built (run cmake build at project root first)
rem   - Admin rights for powercfg /setactive (or run from elevated cmd)
rem ─────────────────────────────────────────────────────────────────

set "ROOT=C:\Users\Tugbars\Desktop\highSpeedFFT"
set "OUTDIR=%ROOT%\build_tuned\dev\bench_vs_cufft"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"

rem ── Set up MSVC + Intel oneAPI env so nvcc finds cl.exe and the
rem    cl.exe link of bench_vfft_mt finds libircmt.lib (vfft.lib was
rem    built with ICX so it depends on Intel's runtime). ─────────────
set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer;%PATH%"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (echo vcvars64.bat failed & exit /b 1)
set "ONEAPI_COMPILER=C:\Program Files (x86)\Intel\oneAPI\compiler\2025.3"
set "LIB=%ONEAPI_COMPILER%\lib;%LIB%"

rem ── Capture current power plan and switch to High Performance ────
echo === Saving current power plan ===
for /f "tokens=4" %%g in ('powercfg /getactivescheme') do set "ORIG_SCHEME=%%g"
echo Original scheme: %ORIG_SCHEME%

echo Switching to High Performance (8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c)
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c
if errorlevel 1 (
    echo [warn] powercfg failed — continuing anyway. For best numbers
    echo [warn] run this script from an elevated command prompt.
)

rem ── Build cuFFT bench ─────────────────────────────────────────────
echo.
echo === Building bench_vs_cufft.cu (CUDA + cuFFT) ===
"%CUDA_PATH%\bin\nvcc.exe" -O3 -arch=native ^
    "%OUTDIR%\bench_vs_cufft.cu" ^
    -lcufft ^
    -o "%OUTDIR%\bench_vs_cufft.exe"
if errorlevel 1 (echo nvcc build failed & goto :restore_power)

rem ── Build VectorFFT T=8 bench ─────────────────────────────────────
echo.
echo === Building bench_vfft_mt.c (MSVC + vfft.lib) ===
cl /nologo /O2 /MD /arch:AVX2 /I "%ROOT%\include" ^
    "%OUTDIR%\bench_vfft_mt.c" ^
    /Fe:"%OUTDIR%\bench_vfft_mt.exe" /Fo"%OUTDIR%\\" ^
    /link "%ROOT%\build\lib\vfft.lib"
if errorlevel 1 (echo vfft.lib build failed & goto :restore_power)

rem ── Run cuFFT bench, capture CSV ──────────────────────────────────
echo.
echo === Running cuFFT bench (3 measurements per cell) ===
"%OUTDIR%\bench_vs_cufft.exe" > "%OUTDIR%\cufft_results.csv"
if errorlevel 1 (echo cuFFT bench failed & goto :restore_power)
echo Saved: %OUTDIR%\cufft_results.csv

rem ── Run VectorFFT T=8 bench, capture CSV ──────────────────────────
echo.
echo === Running VectorFFT T=8 bench ===
"%OUTDIR%\bench_vfft_mt.exe" > "%OUTDIR%\vfft_results.csv"
if errorlevel 1 (echo VectorFFT bench failed & goto :restore_power)
echo Saved: %OUTDIR%\vfft_results.csv

rem ── Print combined summary ────────────────────────────────────────
echo.
echo === Combined results ===
type "%OUTDIR%\vfft_results.csv"
echo.
type "%OUTDIR%\cufft_results.csv"

:restore_power
echo.
echo === Restoring original power plan ===
powercfg /setactive %ORIG_SCHEME%

endlocal
