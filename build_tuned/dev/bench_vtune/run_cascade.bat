@echo off
setlocal enabledelayedexpansion

rem Build + run VTune-instrumented Path C cascade attribution bench.
rem Standalone — links only against the prototype's AVX2 codelets + ITT.
rem
rem Usage:
rem   run_cascade.bat                      build + run, no VTune
rem   run_cascade.bat --collect uarch-exploration   build + profile

set "ROOT=C:\Users\Tugbars\Desktop\highSpeedFFT"
set "OUTDIR=%ROOT%\build_tuned\dev\bench_vtune"
set "PROTO=%ROOT%\src\prototype"
set "VTUNE_ROOT=C:\Program Files (x86)\Intel\oneAPI\vtune\latest"
set "ONEAPI_SETVARS=C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
set "ICX=C:\Program Files (x86)\Intel\oneAPI\compiler\2025.3\bin\icx.exe"
set "VTUNE_EXE=%VTUNE_ROOT%\bin64\vtune.exe"

set "COLLECT="
:parseargs
if "%~1"=="" goto args_done
if "%~1"=="--collect" (set "COLLECT=%~2" & shift & shift & goto parseargs)
shift
goto parseargs
:args_done

rem Assumes setvars.bat has already been sourced in this cmd session.
rem If you see "lnk1104: cannot open libircmt.lib" or similar, run
rem `"%ONEAPI_SETVARS%" intel64` first, then re-invoke this script.

echo === Saving power plan ===
for /f "tokens=4" %%g in ('powercfg /getactivescheme') do set "ORIG_SCHEME=%%g"
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c
if errorlevel 1 echo [warn] powercfg failed - continuing

echo.
echo === Building bench_vtune_cascade.exe (icx, clang driver) ===
"%ICX%" -O3 -mavx2 -mfma -g -Wno-incompatible-pointer-types ^
    -I "%VTUNE_ROOT%\sdk\include" ^
    "%OUTDIR%\bench_vtune_cascade.c" ^
    "%PROTO%\codelets\avx2\large_pow2\r128_r2c_fwd.c" ^
    "%PROTO%\codelets\avx2\mid_pow2\r16_rdft_fwd.c" ^
    "%PROTO%\codelets\avx2\small_pow2\r8_hc2c_dit_fwd.c" ^
    "%VTUNE_ROOT%\sdk\lib64\libittnotify.lib" ^
    -o "%OUTDIR%\bench_vtune_cascade.exe"
if errorlevel 1 (echo build failed & goto :restore_power)

if "%COLLECT%"=="" (
    echo.
    echo === Running [no VTune collection] ===
    "%OUTDIR%\bench_vtune_cascade.exe"
    goto :restore_power
)

set "RESULT_DIR=%OUTDIR%\vt_cascade_%COLLECT%"
if exist "%RESULT_DIR%" rd /s /q "%RESULT_DIR%"
echo.
echo === Running under vtune -collect %COLLECT% ===
"%VTUNE_EXE%" -collect %COLLECT% ^
    -result-dir "%RESULT_DIR%" ^
    -- "%OUTDIR%\bench_vtune_cascade.exe"

echo.
echo === Exporting reports ===
"%VTUNE_EXE%" -report summary  -result-dir "%RESULT_DIR%" -format=csv -report-output "%RESULT_DIR%\summary.csv"
"%VTUNE_EXE%" -report hotspots -result-dir "%RESULT_DIR%" -format=csv -group-by task,function -report-output "%RESULT_DIR%\hotspots.csv"
if /i "%COLLECT%"=="uarch-exploration" (
    "%VTUNE_EXE%" -report top-down -result-dir "%RESULT_DIR%" -format=csv -group-by task,function -report-output "%RESULT_DIR%\topdown.csv"
)

echo.
echo Result dir: %RESULT_DIR%
echo Open GUI:   "%VTUNE_ROOT%\bin64\vtune-gui.exe" "%RESULT_DIR%"

:restore_power
echo.
echo === Restoring power plan ===
powercfg /setactive %ORIG_SCHEME%

endlocal
