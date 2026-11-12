@echo off
rem run_benchmark.bat -- the VectorFFT benchmark programs of this folder, from a menu (README.txt).
rem Double-click it, or run it from a command prompt. The results land in vectorfft_results\ beside it.
setlocal
cd /d "%~dp0"
echo.
echo  VectorFFT against FFTW -- one thread, complex to complex
echo.
echo   1  selected transforms (1D, 2D, 3D), estimate plans          about 2 minutes   [start here]
echo   2  1D powers of two from 64 to 4M, estimate plans            about a minute
echo   3  selected transforms, the planner's plans                  2 minutes + planning of new transforms
if exist vectorfft_vs_fftw_pow2_fp32.exe echo   4  1D powers of two in single precision, the planner's plans  a minute + planning of new lengths
echo.
echo  The first run against FFTW takes longer: FFTW measures its own plans once and keeps them.
echo  Options of every program: --cells LIST, --inplace, --avx2 / --avx512, --out DIR (README.txt).
echo.
set "choice=1"
set /p "choice=Choice [1]: "
set "prog="
if "%choice%"=="1" set "prog=vectorfft_vs_fftw_selected.exe"
if "%choice%"=="2" set "prog=vectorfft_vs_fftw_pow2.exe"
if "%choice%"=="3" set "prog=vectorfft_vs_fftw_selected_planned.exe"
if "%choice%"=="4" set "prog=vectorfft_vs_fftw_pow2_fp32.exe"
if not defined prog (
    echo No such choice: %choice%
    goto done
)
if not exist "%prog%" (
    echo %prog% is not in this package.
    goto done
)
echo.
"%prog%" %*
:done
echo.
pause
endlocal
