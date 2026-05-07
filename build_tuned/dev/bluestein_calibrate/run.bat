@echo off
setlocal enabledelayedexpansion

rem -----------------------------------------------------------------
rem Build + run the Bluestein/Rader calibrator.
rem
rem Sweeps the prime cells in the production bench grid and writes
rem vfft_wisdom_tuned_bluestein.txt with the best (M, B) per cell.
rem
rem Usage:
rem   run.bat                                -- full grid, default output
rem   run.bat --output PATH                  -- override output path
rem   run.bat --primes 47,59 --Ks 4,256      -- subset for quick tests
rem -----------------------------------------------------------------

set "ROOT=C:\Users\Tugbars\Desktop\highSpeedFFT"
set "OUTDIR=%ROOT%\build_tuned\dev\bluestein_calibrate"
set "ONEAPI_COMPILER=C:\Program Files (x86)\Intel\oneAPI\compiler\2025.3"

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (echo vcvars64.bat failed & exit /b 1)

set "PATH=%ONEAPI_COMPILER%\bin;%PATH%"
set "LIB=%ONEAPI_COMPILER%\lib;%LIB%"

echo === Saving current power plan ===
for /f "tokens=4" %%g in ('powercfg /getactivescheme') do set "ORIG_SCHEME=%%g"
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c

echo.
echo === Building calibrate.exe ===
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
    "%OUTDIR%\calibrate.c" ^
    /Fe:"%OUTDIR%\calibrate.exe" /Fo"%OUTDIR%\\" ^
    /link "%ROOT%\build\lib\Release\vfft.lib"
if errorlevel 1 (echo build failed & goto :restore_power)

echo.
echo === Running calibrator ===
rem Pin process affinity to P-core 2 (0x4 = bit 2). Combined with
rem stride_pin_thread inside the binary for belt-and-suspenders. /B
rem runs without a new window; /WAIT blocks until exit so we can
rem capture exit code and restore the power plan.
rem Default output path: alongside production wisdom file.
if "%~1"=="" (
    start /B /WAIT /AFFINITY 0x4 "" "%OUTDIR%\calibrate.exe" --output "%ROOT%\build_tuned\vfft_wisdom_tuned_bluestein.txt"
) else (
    start /B /WAIT /AFFINITY 0x4 "" "%OUTDIR%\calibrate.exe" %*
)

:restore_power
echo.
echo === Restoring original power plan ===
powercfg /setactive %ORIG_SCHEME%

endlocal
