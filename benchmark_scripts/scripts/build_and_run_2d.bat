@echo off
rem build_and_run_2d.bat -- the 2D picture from one double-click on Windows. It builds the benchmark tools for this
rem computer's CPU (build_tools.py: -march=native, the AVX-512 version when the CPU runs AVX-512, else the AVX2 one),
rem then runs the 2D complex transform on every power-of-two plane up to 4M points against FFTW (run_suite.py --sets 2d).
rem Needs Python 3.8 or newer, and GCC: MSYS2 in its default folder with its UCRT64 compiler
rem (pacman -S mingw-w64-ucrt-x86_64-gcc), or a compiler named by the CC variable. It installs nothing,
rem except the Python package matplotlib for the picture, and only when you say yes.
rem Further arguments go to run_suite.py: build_and_run_2d.bat --cmp mkl compares with MKL instead.
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

rem -- the compiler: the one gauntlet\build.py will use (its list of MSYS2 and MinGW-w64 folders, or CC)
set "TMPF=%TEMP%\vectorfft_cc_%RANDOM%.txt"
%PY% -c "import importlib.util as u, os, shutil, sys; s = u.spec_from_file_location('build', os.path.join('gauntlet', 'build.py')); m = u.module_from_spec(s); s.loader.exec_module(m); cc = m.detect_toolchain()['cc']; p = cc if os.path.isfile(cc) else shutil.which(cc); print(p or cc); sys.exit(0 if p else 1)" > "%TMPF%" 2>nul
set "CCRC=%errorlevel%"
set "CC_EXE="
set /p CC_EXE=<"%TMPF%"
del "%TMPF%" >nul 2>&1
if not "%CCRC%"=="0" goto no_cc

rem -- the CPU, as the compiler sees it with -march=native (the flag the build uses); the compiler's folder goes on
rem    PATH first, as build.py does, because its helper programs find their DLLs through PATH
for %%D in ("%CC_EXE%") do set "CC_DIR=%%~dpD"
set "PATH=%CC_DIR%;%PATH%"
set "TMPF=%TEMP%\vectorfft_isa_%RANDOM%.txt"
echo.| "%CC_EXE%" -march=native -dM -E -x c - > "%TMPF%" 2>nul
if errorlevel 1 goto cc_broken
set "ISA="
findstr /b /c:"#define __AVX2__ 1" "%TMPF%" >nul && findstr /b /c:"#define __FMA__ 1" "%TMPF%" >nul && set "ISA=avx2"
findstr /b /c:"#define __AVX512F__ 1" "%TMPF%" >nul && findstr /b /c:"#define __AVX512DQ__ 1" "%TMPF%" >nul && if defined ISA set "ISA=avx512"
del "%TMPF%" >nul 2>&1
if not defined ISA goto no_avx2

echo.
echo  Compiler: %CC_EXE%
if "%ISA%"=="avx512" echo  This CPU runs AVX-512: building the AVX-512 version.
if "%ISA%"=="avx2" echo  This CPU runs AVX2 but not AVX-512: building the AVX2 version.
echo  The first build compiles the library's kernels, over a thousand files: a few minutes. Later builds reuse them.
echo  Then the benchmark, the 2D transform on every power-of-two plane up to 4M points, 159 planes,
echo  against FFTW: about 10 minutes.
echo  The first run against FFTW takes longer: FFTW measures its own plans once and keeps them.
echo.

rem -- matplotlib draws the picture; installed only when the user says yes
%PY% -c "import matplotlib" >nul 2>&1
if not errorlevel 1 goto have_mpl
echo  The picture is drawn with the Python package matplotlib, which is not installed.
set "ANS=Y"
set /p "ANS=Install it now with pip? [Y/n]: "
if /i "%ANS%"=="n" goto no_mpl
%PY% -m pip install matplotlib
if errorlevel 1 goto no_mpl
:have_mpl

%PY% benchmark_scripts\scripts\build_tools.py --isa %ISA%
if errorlevel 1 goto fail
echo.
%PY% benchmark_scripts\scripts\run_suite.py --cmp fftw --sets 2d %*
if errorlevel 1 goto fail
echo.
echo  Done. The table and the picture are in the run folder named above, under gauntlet\results\.
goto end

:no_cc
echo  GCC was not found in the folders gauntlet\build.py looks in.
echo  Install MSYS2 from https://www.msys2.org/ in its default folder, open its "MSYS2 UCRT64" terminal
echo  and run:  pacman -S mingw-w64-ucrt-x86_64-gcc
echo  Or name a GCC of your own: open a command prompt, run  set CC=C:\path\to\gcc.exe
echo  and start this script from that window.
goto fail

:cc_broken
echo  The compiler %CC_EXE% did not run.
goto fail

:no_avx2
echo  The compiler reports no AVX2 and FMA for this CPU. VectorFFT needs both: Intel from Haswell, 2013,
echo  AMD from Zen, 2017. A GCC too old to know this CPU reports the same: a current MSYS2 GCC knows it.
goto fail

:no_mpl
echo  matplotlib is not installed. Install it with:  %PY% -m pip install matplotlib
echo  and run this again.
goto fail

:fail
echo.
echo  Stopped: see the lines above.
:end
echo.
pause
endlocal
