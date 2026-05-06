@echo off
setlocal enabledelayedexpansion

rem ─────────────────────────────────────────────────────────────────
rem Build + run VTune-instrumented 1D C2C bench across selected cells.
rem
rem Builds bench_vtune.exe with ITT linkage. Optionally launches VTune
rem CLI for microarchitecture-exploration analysis. Output result-dir
rem is opened with vtune-gui or summarized to terminal.
rem
rem Usage:
rem   run.bat                       — build and run, no VTune collection
rem   run.bat --collect uarch       — build + run under VTune uarch-exploration
rem   run.bat --collect hotspots    — build + run under VTune hotspots
rem   run.bat --collect threading   — build + run under VTune threading
rem ─────────────────────────────────────────────────────────────────

set "ROOT=C:\Users\Tugbars\Desktop\highSpeedFFT"
set "OUTDIR=%ROOT%\build_tuned\dev\bench_vtune"
set "VTUNE_ROOT=C:\Program Files (x86)\Intel\oneAPI\vtune\latest"
set "ONEAPI_COMPILER=C:\Program Files (x86)\Intel\oneAPI\compiler\2025.3"
set "MKLROOT=C:\Program Files (x86)\Intel\oneAPI\mkl\latest"

set "COLLECT="
:parseargs
if "%~1"=="" goto args_done
if "%~1"=="--collect" (set "COLLECT=%~2" & shift & shift & goto parseargs)
shift
goto parseargs
:args_done

rem ── Set up MSVC + Intel oneAPI env so cl.exe links against the
rem    ICX-built vfft.lib (libircmt.lib) + MKL (if requested). ─────────
set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer;%PATH%"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (echo vcvars64.bat failed & exit /b 1)
set "LIB=%ONEAPI_COMPILER%\lib;%MKLROOT%\lib;%LIB%"

rem ── Power plan: switch to High Performance for stable VTune samples ──
echo === Saving current power plan ===
for /f "tokens=4" %%g in ('powercfg /getactivescheme') do set "ORIG_SCHEME=%%g"
echo Original scheme: %ORIG_SCHEME%
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c
if errorlevel 1 echo [warn] powercfg failed (admin needed) — continuing

rem ── Build with ITT + MKL + debug symbols (PDB).
rem    /Zi + /DEBUG keeps optimization on (/O2) but emits a PDB so
rem    VTune can resolve function names instead of showing
rem    `func@0x1402ad480`. /Z7 would inline the debug info into the
rem    .obj; /Zi puts it in a separate .pdb which VTune prefers. ──────
echo.
echo === Building bench_vtune.exe with ITT + MKL + debug symbols ===
cl /nologo /O2 /Zi /MD /arch:AVX2 ^
    /I "%ROOT%\include" ^
    /I "%VTUNE_ROOT%\sdk\include" ^
    /I "%MKLROOT%\include" ^
    /DVFFT_HAS_MKL /DMKL_ILP64 ^
    "%OUTDIR%\bench_vtune.c" ^
    /Fe:"%OUTDIR%\bench_vtune.exe" /Fo"%OUTDIR%\\" /Fd"%OUTDIR%\bench_vtune.pdb" ^
    /link /DEBUG ^
    "%ROOT%\build\lib\Release\vfft.lib" ^
    "%VTUNE_ROOT%\sdk\lib64\libittnotify.lib" ^
    mkl_intel_ilp64.lib mkl_sequential.lib mkl_core.lib
if errorlevel 1 (echo build failed & goto :restore_power)

rem ── Run, optionally under VTune ────────────────────────────────────
if "%COLLECT%"=="" (
    echo.
    echo === Running bench_vtune.exe --mkl [no VTune collection] ===
    "%OUTDIR%\bench_vtune.exe" --mkl
    goto :restore_power
)

rem ── VTune collection path: capture bench output, export report
rem    artifacts as CSV, then compose a markdown report. ────────────────
set "RESULT_DIR=%OUTDIR%\vt_%COLLECT%"
if exist "!RESULT_DIR!" rmdir /s /q "!RESULT_DIR!"
echo.
echo === Running under vtune -collect %COLLECT% ===
echo Result dir: !RESULT_DIR!
rem Bench writes its summary table directly to a file via --output
rem (VTune captures+suppresses the wrapped program's stdout, so we
rem can't rely on > redirection to get the bench results).
set "BENCH_OUT=%OUTDIR%\bench_output_%COLLECT%.txt"
"%VTUNE_ROOT%\bin64\vtune.exe" -collect %COLLECT% ^
    -result-dir "!RESULT_DIR!" ^
    -- "%OUTDIR%\bench_vtune.exe" --mkl --output "!BENCH_OUT!"
type "!BENCH_OUT!"
rem Move into result dir for the report builder
if exist "!BENCH_OUT!" copy /y "!BENCH_OUT!" "!RESULT_DIR!\bench_output.txt" >nul

echo.
echo === Exporting VTune reports as CSV ===
"%VTUNE_ROOT%\bin64\vtune.exe" -report summary  -result-dir "!RESULT_DIR!" -format=csv -report-output "!RESULT_DIR!\summary.csv"
rem -group-by task,function gives per-function breakdown WITHIN each ITT
rem task — i.e. "which codelet ran for which cell". With just `task`
rem grouping we'd only see the task's total time, not the functions
rem inside it. Both files are exported so the report builder can
rem present per-cell drill-down.
"%VTUNE_ROOT%\bin64\vtune.exe" -report hotspots -result-dir "!RESULT_DIR!" -format=csv -group-by task,function -report-output "!RESULT_DIR!\hotspots.csv"
if /i "%COLLECT%"=="uarch-exploration" (
    "%VTUNE_ROOT%\bin64\vtune.exe" -report top-down -result-dir "!RESULT_DIR!" -format=csv -group-by task,function -report-output "!RESULT_DIR!\topdown.csv"
)

echo.
echo === Composing markdown report ===
python "%OUTDIR%\make_report.py" "!RESULT_DIR!" --collect-mode "%COLLECT%"

echo.
echo Result dir:  !RESULT_DIR!
echo Markdown:    !RESULT_DIR!\report.md
echo Open GUI:    "%VTUNE_ROOT%\bin64\vtune-gui.exe" "!RESULT_DIR!"

:restore_power
echo.
echo === Restoring original power plan ===
powercfg /setactive %ORIG_SCHEME%

endlocal
