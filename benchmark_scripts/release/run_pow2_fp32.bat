@echo off
rem run_pow2_fp32.bat -- double-click it: VectorFFT against FFTW on the powers of two in single precision (FP32)
rem with the planner's plans: about a minute, plus about 7 the first time on this CPU (README.txt).
rem From a command prompt it passes options to the program:  run_pow2_fp32.bat --cells 1024,8192
setlocal
cd /d "%~dp0"
echo.
echo  VectorFFT against FFTW: the powers of two in single precision (FP32), about a minute.
echo  Lengths new to this CPU are planned first and kept: about 7 more minutes the first time.
echo  Close other programs while it runs.
echo  The first run takes longer: FFTW measures its own plans once and keeps them.
echo.
if exist "%~dp0vectorfft_vs_fftw_pow2_fp32.exe" (
    "%~dp0vectorfft_vs_fftw_pow2_fp32.exe" %*
) else (
    echo  vectorfft_vs_fftw_pow2_fp32.exe is not beside this file: unpack the whole download into one folder first.
)
echo.
pause
endlocal
