@echo off
setlocal enabledelayedexpansion

rem -----------------------------------------------------------------
rem Build + run bench_1d_vs_mkl with the calibrator-derived Bluestein
rem wisdom installed. Compares vfft (new core, wisdom-driven) against
rem MKL on the full 207-cell grid.
rem -----------------------------------------------------------------

set "ROOT=C:\Users\Tugbars\Desktop\highSpeedFFT"
set "OUTDIR=%ROOT%\build_tuned\benches"
set "ONEAPI_COMPILER=C:\Program Files (x86)\Intel\oneAPI\compiler\2025.3"
set "MKLROOT=C:\Program Files (x86)\Intel\oneAPI\mkl\latest"

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (echo vcvars64.bat failed & exit /b 1)

set "PATH=%ONEAPI_COMPILER%\bin;%PATH%"
set "LIB=%ONEAPI_COMPILER%\lib;%MKLROOT%\lib;%LIB%"

echo === Saving current power plan ===
for /f "tokens=4" %%g in ('powercfg /getactivescheme') do set "ORIG_SCHEME=%%g"
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c

echo.
echo === Building bench_1d_vs_mkl.exe ===
icx-cl /nologo /O2 /MD /arch:AVX2 ^
    /I "%ROOT%\include" ^
    /I "%ROOT%\src\core" ^
    /I "%MKLROOT%\include" ^
    /I "%ROOT%\src\vectorfft_tune\generated\r2"  ^
    /I "%ROOT%\src\vectorfft_tune\generated\r3"  ^
    /I "%ROOT%\src\vectorfft_tune\generated\r4"  ^
    /I "%ROOT%\src\vectorfft_tune\generated\r5"  ^
    /I "%ROOT%\src\vectorfft_tune\generated\r6"  ^
    /I "%ROOT%\src\vectorfft_tune\generated\r7"  ^
    /I "%ROOT%\src\vectorfft_tune\generated\r8"  ^
    /I "%ROOT%\src\vectorfft_tune\generated\r10" ^
    /I "%ROOT%\src\vectorfft_tune\generated\r11" ^
    /I "%ROOT%\src\vectorfft_tune\generated\r12" ^
    /I "%ROOT%\src\vectorfft_tune\generated\r13" ^
    /I "%ROOT%\src\vectorfft_tune\generated\r16" ^
    /I "%ROOT%\src\vectorfft_tune\generated\r17" ^
    /I "%ROOT%\src\vectorfft_tune\generated\r19" ^
    /I "%ROOT%\src\vectorfft_tune\generated\r20" ^
    /I "%ROOT%\src\vectorfft_tune\generated\r25" ^
    /I "%ROOT%\src\vectorfft_tune\generated\r32" ^
    /I "%ROOT%\src\vectorfft_tune\generated\r64" ^
    /I "%ROOT%\src\vectorfft_tune\generated\dct8" ^
    /D__FMA__ /DVFFT_ISA_AVX2 /D_CRT_SECURE_NO_WARNINGS ^
    /DVFFT_HAS_MKL /DMKL_ILP64 ^
    "%OUTDIR%\bench_1d_vs_mkl.c" ^
    /Fe:"%OUTDIR%\bench_1d_vs_mkl.exe" /Fo"%OUTDIR%\\" ^
    /link "%ROOT%\build\lib\Release\vfft.lib" ^
    mkl_intel_ilp64.lib mkl_sequential.lib mkl_core.lib
if errorlevel 1 (echo build failed & goto :restore_power)

echo.
echo === Running bench_1d_vs_mkl.exe ===
rem cd to build_tuned/ so the bench finds vfft_wisdom_tuned.txt
rem (and its _bluestein companion) at the default relative path.
cd /d "%ROOT%\build_tuned"
"%OUTDIR%\bench_1d_vs_mkl.exe" %*

:restore_power
echo.
echo === Restoring original power plan ===
powercfg /setactive %ORIG_SCHEME%

endlocal
