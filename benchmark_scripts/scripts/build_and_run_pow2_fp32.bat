@echo off
rem build_and_run_pow2_fp32.bat -- the single-precision (FP32) quick check from one double-click on Windows. It builds
rem the program vectorfft_vs_fftw_pow2_fp32 (README.md, "Build targets") for this computer's CPU with CMake
rem (-march=native, AVX-512 when the CPU runs it, -DVFFT_ENABLE_FLOAT=ON) into build-fp32\, then runs it: the 1D
rem transform at every power of two from 64 to 4,194,304 against single-precision FFTW, with the planner's plans.
rem CMake does not have to be on PATH: the script looks beside the compiler, on PATH and in C:\Program Files\CMake.
rem Needs Python 3.8 or newer, GCC (MSYS2's UCRT64 compiler, or one named by the CC variable), CMake, and Ninja or
rem mingw32-make. With MSYS2, its "MSYS2 UCRT64" terminal installs all three:
rem   pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja
rem Further arguments go to the program: --cells LIST, --out DIR, --inplace, --avx2 / --avx512.
setlocal
cd /d "%~dp0..\.."
if not exist "gauntlet\build.py" (
    echo This script must stay in benchmark_scripts\scripts\ of the VectorFFT sources.
    goto fail
)

rem -- Python: the py launcher first (python.org installs it even when python.exe is not on PATH), then python
set "PY="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>&1
if not errorlevel 1 set "PY=py -3"
if defined PY goto have_py
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>&1
if not errorlevel 1 set "PY=python"
if defined PY goto have_py
echo  Python 3.8 or newer was not found. Install it from https://www.python.org/downloads/
echo  and run this again.
goto fail
:have_py
set "TMPF=%TEMP%\vectorfft_py_%RANDOM%.txt"
%PY% -c "import sys; print(sys.executable)" > "%TMPF%" 2>nul
set "PY_EXE="
set /p PY_EXE=<"%TMPF%"
del "%TMPF%" >nul 2>&1

rem -- the compiler: the one gauntlet\build.py uses (its list of MSYS2 and MinGW-w64 folders, or CC)
set "TMPF=%TEMP%\vectorfft_cc_%RANDOM%.txt"
%PY% -c "import importlib.util as u, os, shutil, sys; s = u.spec_from_file_location('build', os.path.join('gauntlet', 'build.py')); m = u.module_from_spec(s); s.loader.exec_module(m); cc = m.detect_toolchain()['cc']; p = cc if os.path.isfile(cc) else shutil.which(cc); print(p or cc); sys.exit(0 if p else 1)" > "%TMPF%" 2>nul
set "CCRC=%errorlevel%"
set "CC_EXE="
set /p CC_EXE=<"%TMPF%"
del "%TMPF%" >nul 2>&1
if not "%CCRC%"=="0" goto no_cc
rem the compiler's folder first on PATH: its helper programs find their DLLs through PATH, and CMake and Ninja
rem often sit beside it
for %%D in ("%CC_EXE%") do set "CC_DIR=%%~dpD"
set "PATH=%CC_DIR%;%PATH%"

rem -- CMake: beside the compiler or on PATH, else its installer's default folder
set "CMAKE_EXE="
for /f "delims=" %%P in ('where cmake 2^>nul') do if not defined CMAKE_EXE set "CMAKE_EXE=%%P"
if not defined CMAKE_EXE if exist "%ProgramFiles%\CMake\bin\cmake.exe" set "CMAKE_EXE=%ProgramFiles%\CMake\bin\cmake.exe"
if not defined CMAKE_EXE goto no_cmake

rem -- the build tool: Ninja (beside the compiler, on PATH or beside CMake), else the compiler's mingw32-make
set "GEN="
set "MAKE_EXE="
for /f "delims=" %%P in ('where ninja 2^>nul') do if not defined MAKE_EXE set "MAKE_EXE=%%P"
if not defined MAKE_EXE for %%D in ("%CMAKE_EXE%") do if exist "%%~dpDninja.exe" set "MAKE_EXE=%%~dpDninja.exe"
if defined MAKE_EXE set "GEN=Ninja"
if not defined GEN if exist "%CC_DIR%mingw32-make.exe" set "MAKE_EXE=%CC_DIR%mingw32-make.exe"
if not defined GEN if defined MAKE_EXE set "GEN=MinGW Makefiles"
if not defined GEN goto no_ninja

echo.
echo  Compiler: %CC_EXE%
echo  CMake:    %CMAKE_EXE%
echo  Build:    %MAKE_EXE%
echo  The first build compiles the library in both precisions: some minutes, longer on a small computer.
echo  Later builds reuse it. Then the benchmark, the 17 powers of two from 64 to 4,194,304 against FFTW:
echo  about a minute, plus the planning of the lengths new to this computer, about 7 minutes when all are new.
echo  The first run against FFTW takes longer: FFTW measures its own plans once and keeps them.
echo.

rem CMake reads forward slashes in paths everywhere
set "CC_F=%CC_EXE:\=/%"
set "MAKE_F=%MAKE_EXE:\=/%"
set "PY_F=%PY_EXE:\=/%"
"%CMAKE_EXE%" -S . -B build-fp32 -G "%GEN%" "-DCMAKE_MAKE_PROGRAM=%MAKE_F%" "-DCMAKE_C_COMPILER=%CC_F%" "-DPython3_EXECUTABLE=%PY_F%" -DVFFT_ENABLE_FLOAT=ON -DVFFT_BENCH_BOTH_ISAS=OFF
if errorlevel 1 goto cmake_failed
echo.
"%CMAKE_EXE%" --build build-fp32 --target vectorfft_vs_fftw_pow2_fp32 --parallel %NUMBER_OF_PROCESSORS%
if errorlevel 1 goto fail
echo.
"build-fp32\benchmark_scripts\vectorfft_vs_fftw_pow2_fp32.exe" %*
if errorlevel 1 goto fail
echo.
echo  Done. The table is in the run folder named above, under vectorfft_results\.
goto end

:no_cc
echo  GCC was not found in the folders gauntlet\build.py looks in.
echo  Install MSYS2 from https://www.msys2.org/ in its default folder, open its "MSYS2 UCRT64" terminal
echo  and run:  pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja
echo  Or name a GCC of your own: open a command prompt, run  set CC=C:\path\to\gcc.exe
echo  and start this script from that window.
goto fail

:no_cmake
echo  CMake was not found: not beside the compiler, not on PATH, not in %ProgramFiles%\CMake\bin.
echo  With MSYS2, run in its "MSYS2 UCRT64" terminal:  pacman -S mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja
echo  or install CMake from https://cmake.org/download/ in its default folder.
goto fail

:no_ninja
echo  Neither Ninja nor mingw32-make was found: not beside the compiler, not on PATH, not beside CMake.
echo  With MSYS2, run in its "MSYS2 UCRT64" terminal:  pacman -S mingw-w64-ucrt-x86_64-ninja
goto fail

:cmake_failed
echo.
echo  CMake could not configure the build. If the tools changed since an earlier run of this script,
echo  delete the folder build-fp32 and run it again.
goto fail

:fail
echo.
echo  Stopped: see the lines above.
:end
echo.
pause
endlocal
