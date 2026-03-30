@echo off
setlocal EnableDelayedExpansion

set "ONEAPI_ROOT=C:\Program Files (x86)\Intel\oneAPI"
call "!ONEAPI_ROOT!\setvars.bat" --force >nul 2>&1

set "SCRIPT_DIR=%~dp0.."
set "HDR_DIR=%~dp0headers"
set "FFTW_INC=C:\Users\Tugbars\Desktop\highSpeedFFT\build\vcpkg_installed\x64-windows\include"
set "FFTW_LIB=C:\Users\Tugbars\Desktop\highSpeedFFT\build\vcpkg_installed\x64-windows\lib\fftw3.lib"
set "FFTW_BIN=C:\Users\Tugbars\Desktop\highSpeedFFT\build\vcpkg_installed\x64-windows\bin"

echo Compiling with ASan...
icx -O1 -g -fsanitize=address -mavx2 -mfma -D_USE_MATH_DEFINES -I"%HDR_DIR%" -I"%FFTW_INC%" -I"%SCRIPT_DIR%" -o "%~dp0bench_ct_factor_asan.exe" "%SCRIPT_DIR%\bench_ct_factor.c" "%FFTW_LIB%"
if errorlevel 1 ( echo FAILED & exit /b 1 )
echo Compiled OK.

echo Running with ASan...
set "PATH=%FFTW_BIN%;%PATH%"
"%~dp0bench_ct_factor_asan.exe"
