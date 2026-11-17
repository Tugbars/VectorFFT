@echo off
rem run_pow2.bat -- double-click it: VectorFFT against FFTW on the 1D transform at every power of two from 64
rem to 4,194,304 with the estimate's plans, about a minute (README.txt).
rem From a command prompt it passes options to the program:  run_pow2.bat --cells 1000,4096
setlocal
cd /d "%~dp0"
echo.
echo  VectorFFT against FFTW: the 1D transform at every power of two from 64 to 4,194,304, about a minute.
echo  Close other programs while it runs.
echo  The first run takes longer: FFTW measures its own plans once and keeps them.
echo.
if exist "%~dp0vectorfft_vs_fftw_pow2.exe" (
    "%~dp0vectorfft_vs_fftw_pow2.exe" %*
) else (
    echo  vectorfft_vs_fftw_pow2.exe is not beside this file: unpack the whole download into one folder first.
)
echo.
pause
endlocal
