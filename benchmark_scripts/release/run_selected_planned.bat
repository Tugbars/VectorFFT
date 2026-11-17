@echo off
rem run_selected_planned.bat -- double-click it: VectorFFT against FFTW on the 18 transforms with the plans of
rem VectorFFT's planner, which measures: about 2 minutes, plus up to about 20 the first time on this CPU (README.txt).
rem From a command prompt it passes options to the program:  run_selected_planned.bat --cells 1000,4096
setlocal
cd /d "%~dp0"
echo.
echo  VectorFFT against FFTW: the 18 transforms with the plans of VectorFFT's planner, about 2 minutes.
echo  Transforms new to this CPU are planned first and kept: up to about 20 more minutes the first time.
echo  Close other programs while it runs.
echo  The first run takes longer: FFTW measures its own plans once and keeps them.
echo.
if exist "%~dp0vectorfft_vs_fftw_selected_planned.exe" (
    "%~dp0vectorfft_vs_fftw_selected_planned.exe" %*
) else (
    echo  vectorfft_vs_fftw_selected_planned.exe is not beside this file: unpack the whole download into one folder first.
)
echo.
pause
endlocal
