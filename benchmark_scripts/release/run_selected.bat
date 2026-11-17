@echo off
rem run_selected.bat -- double-click it: VectorFFT against FFTW on 18 commonly used transforms (1D, 2D, 3D)
rem with the estimate's plans, about 2 minutes (README.txt).
rem From a command prompt it passes options to the program:  run_selected.bat --cells 1000,4096
setlocal
cd /d "%~dp0"
echo.
echo  VectorFFT against FFTW: 18 commonly used transforms (1D, 2D and 3D), about 2 minutes.
echo  Close other programs while it runs.
echo  The first run takes longer: FFTW measures its own plans once and keeps them.
echo.
if exist "%~dp0vectorfft_vs_fftw_selected.exe" (
    "%~dp0vectorfft_vs_fftw_selected.exe" %*
) else (
    echo  vectorfft_vs_fftw_selected.exe is not beside this file: unpack the whole download into one folder first.
)
echo.
pause
endlocal
