@echo off
setlocal enabledelayedexpansion

rem -----------------------------------------------------------------
rem Build + run the Bluestein M and B sweep harness.
rem
rem Bypasses the public vfft API; calls stride_bluestein_plan with an
rem explicit M to compare M-selection strategies.
rem
rem Usage:
rem   run.bat                              -- sweeps default cells
rem   run.bat 107 256                      -- single cell
rem   run.bat 107 256 179 256              -- multiple
rem -----------------------------------------------------------------

set "ROOT=C:\Users\Tugbars\Desktop\highSpeedFFT"
set "OUTDIR=%ROOT%\build_tuned\dev\bluestein_msweep"
set "ONEAPI_COMPILER=C:\Program Files (x86)\Intel\oneAPI\compiler\2025.3"

rem MSVC env (linker, system headers)
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (echo vcvars64.bat failed & exit /b 1)

rem ICX env: codelet headers use GCC __attribute__ syntax which only ICX
rem and gcc/clang can parse. Cl.exe doesn't. So we drive ICX with cl-style
rem flags via icx-cl.exe, falling back to MSVC's link.exe.
set "PATH=%ONEAPI_COMPILER%\bin;%PATH%"
set "LIB=%ONEAPI_COMPILER%\lib;%LIB%"

rem Power plan: High Performance for stable measurements.
echo === Saving current power plan ===
for /f "tokens=4" %%g in ('powercfg /getactivescheme') do set "ORIG_SCHEME=%%g"
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c

echo.
echo === Building sweep.exe ===
icx-cl /nologo /O2 /MD /arch:AVX2 ^
    /I "%ROOT%\include" ^
    /I "%ROOT%\src\core" ^
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
    "%OUTDIR%\sweep.c" ^
    /Fe:"%OUTDIR%\sweep.exe" /Fo"%OUTDIR%\\" ^
    /link "%ROOT%\build\lib\Release\vfft.lib"
if errorlevel 1 (echo build failed & goto :restore_power)

echo.
echo === Running sweep ===
if "%~1"=="" (
    rem Default: the 6 verification cells from bench_vtune
    "%OUTDIR%\sweep.exe" 47 256 59 256 83 256 107 256 179 256 311 256
) else (
    "%OUTDIR%\sweep.exe" %*
)

:restore_power
echo.
echo === Restoring original power plan ===
powercfg /setactive %ORIG_SCHEME%

endlocal
