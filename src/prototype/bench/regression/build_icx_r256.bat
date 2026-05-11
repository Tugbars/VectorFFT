@echo off
REM Build bench_r256_avx2 with ICX under the oneAPI environment.
REM Run from cmd.exe (not PowerShell) so the oneAPI env survives.

set "SETVARS=C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
call "%SETVARS%" intel64 > nul
if errorlevel 1 (
    echo setvars.bat failed
    exit /b 1
)

pushd "%~dp0..\..\"
if not exist _build_bench mkdir _build_bench

icx -O3 -mavx2 -mfma -Wno-incompatible-pointer-types ^
  codelets/avx2/large_pow2/r256_n1_fwd.c ^
  codelets/avx2/large_pow2/r256_t1_dit_fwd.c ^
  codelets/avx2/large_pow2/r256_t1s_dit_fwd.c ^
  codelets/avx2/large_pow2/r256_t1_dit_fwd_log3.c ^
  codelets/avx2/large_pow2/r256_t1s_dit_fwd_log3.c ^
  codelets/avx2/mid_pow2/r16_n1_fwd.c ^
  bench/regression/bench_r256_avx2.c ^
  -o _build_bench/bench_r256_avx2_icx.exe

set RC=%errorlevel%
popd
exit /b %RC%