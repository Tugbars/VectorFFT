@echo off
rem check.bat -- is VectorFFT working on this PC? Double-click it. It runs check.exe from this folder (and builds
rem it from check.c first when it is missing, which needs MinGW-w64 GCC on PATH). On a CPU with AVX-512 it runs
rem the check twice: once with the library the CPU gets by itself, once forced to the AVX2 library.
setlocal
cd /d "%~dp0"
set VFFT_ISA=
set CHECK_FLOAT=
if exist "%~dp0vfftf.h" set CHECK_FLOAT=-DCHECK_FLOAT
if not exist "%~dp0check.exe" (
    where gcc >nul 2>nul
    if errorlevel 1 (
        echo check.exe is missing and gcc was not found: install MinGW-w64 GCC and put its bin folder on PATH.
        goto :end
    )
    gcc -O2 check.c %CHECK_FLOAT% -I. -L. -lvfft -o check.exe
    if errorlevel 1 (
        echo check.c did not compile.
        goto :end
    )
)
echo ==== the library this CPU gets by itself
"%~dp0check.exe" > "%~dp0check_result.txt" 2>&1
set RESULT=%errorlevel%
type "%~dp0check_result.txt"
findstr /c:"library loaded    avx512" "%~dp0check_result.txt" >nul
if not errorlevel 1 (
    echo.
    echo ==== the same CPU, forced to the AVX2 library
    set VFFT_ISA=avx2
    "%~dp0check.exe"
    if errorlevel 1 set RESULT=1
)
echo.
if "%RESULT%"=="0" (echo RESULT: everything passed.) else (echo RESULT: something failed. Send the text above.)
:end
echo.
pause
