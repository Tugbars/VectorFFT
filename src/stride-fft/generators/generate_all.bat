@echo off
setlocal

set "GEN_DIR=%~dp0"
for %%I in ("%GEN_DIR%..") do set "SFFT_DIR=%%~fI"
set "AVX2_DIR=%SFFT_DIR%\codelets\avx2"
set "AVX512_DIR=%SFFT_DIR%\codelets\avx512"
set "SCALAR_DIR=%SFFT_DIR%\codelets\scalar"

set "PYTHONIOENCODING=utf-8"

if not exist "%AVX2_DIR%" mkdir "%AVX2_DIR%"
if not exist "%AVX512_DIR%" mkdir "%AVX512_DIR%"
if not exist "%SCALAR_DIR%" mkdir "%SCALAR_DIR%"

echo Generating AVX2 codelets...

:: R=2, R=4, R=8: legacy generators (single file per ISA)
for %%R in (2 4 8) do (
    python "%GEN_DIR%gen_radix%%R.py" avx2 > "%AVX2_DIR%\fft_radix%%R_avx2.h" 2>nul
)

:: R=3,5,6,7,10,11,12,13,16,17,19,20,25: ct_n1 + ct_t1_dit + ct_t1s_dit + ct_t1_dit_log3 + ct_t1_oop_dit
for %%R in (3 5 6 7 10 11 12 13 16 17 19 20 25) do (
    python "%GEN_DIR%gen_radix%%R.py" --isa avx2 --variant ct_n1 > "%AVX2_DIR%\fft_radix%%R_avx2_ct_n1.h" 2>nul
    python "%GEN_DIR%gen_radix%%R.py" --isa avx2 --variant ct_t1_dit > "%AVX2_DIR%\fft_radix%%R_avx2_ct_t1_dit.h" 2>nul
    python "%GEN_DIR%gen_radix%%R.py" --isa avx2 --variant ct_t1s_dit > "%AVX2_DIR%\fft_radix%%R_avx2_ct_t1s_dit.h" 2>nul
    python "%GEN_DIR%gen_radix%%R.py" --isa avx2 --variant ct_t1_dit_log3 > "%AVX2_DIR%\fft_radix%%R_avx2_ct_t1_dit_log3.h" 2>nul
    python "%GEN_DIR%gen_radix%%R.py" --isa avx2 --variant ct_t1_oop_dit > "%AVX2_DIR%\fft_radix%%R_avx2_ct_t1_oop_dit.h" 2>nul
)

:: R=32: ct_n1 + ct_t1_dit + ct_t1_oop_dit (no log3)
python "%GEN_DIR%gen_radix32.py" --isa avx2 --variant ct_n1 > "%AVX2_DIR%\fft_radix32_avx2_ct_n1.h" 2>nul
python "%GEN_DIR%gen_radix32.py" --isa avx2 --variant ct_t1_dit > "%AVX2_DIR%\fft_radix32_avx2_ct_t1_dit.h" 2>nul
python "%GEN_DIR%gen_radix32.py" --isa avx2 --variant ct_t1_oop_dit > "%AVX2_DIR%\fft_radix32_avx2_ct_t1_oop_dit.h" 2>nul

:: R=64: ct_n1 + ct_t1_dit + ct_t1_dit_log3 + ct_t1_oop_dit
python "%GEN_DIR%gen_radix64.py" --isa avx2 --variant ct_n1 > "%AVX2_DIR%\fft_radix64_avx2_ct_n1.h" 2>nul
python "%GEN_DIR%gen_radix64.py" --isa avx2 --variant ct_t1_dit > "%AVX2_DIR%\fft_radix64_avx2_ct_t1_dit.h" 2>nul
python "%GEN_DIR%gen_radix64.py" --isa avx2 --variant ct_t1_dit_log3 > "%AVX2_DIR%\fft_radix64_avx2_ct_t1_dit_log3.h" 2>nul
python "%GEN_DIR%gen_radix64.py" --isa avx2 --variant ct_t1_oop_dit > "%AVX2_DIR%\fft_radix64_avx2_ct_t1_oop_dit.h" 2>nul

echo   Done. %AVX2_DIR%

echo.
echo Generating AVX-512 codelets...

:: R=2, R=4, R=8: legacy generators
for %%R in (2 4 8) do (
    python "%GEN_DIR%gen_radix%%R.py" avx512 > "%AVX512_DIR%\fft_radix%%R_avx512.h" 2>nul
)

:: R=3,5,6,7,10,11,12,13,16,17,19,20,25: ct_n1 + ct_t1_dit + ct_t1s_dit + ct_t1_dit_log3 + ct_t1_oop_dit
for %%R in (3 5 6 7 10 11 12 13 16 17 19 20 25) do (
    python "%GEN_DIR%gen_radix%%R.py" --isa avx512 --variant ct_n1 > "%AVX512_DIR%\fft_radix%%R_avx512_ct_n1.h" 2>nul
    python "%GEN_DIR%gen_radix%%R.py" --isa avx512 --variant ct_t1_dit > "%AVX512_DIR%\fft_radix%%R_avx512_ct_t1_dit.h" 2>nul
    python "%GEN_DIR%gen_radix%%R.py" --isa avx512 --variant ct_t1s_dit > "%AVX512_DIR%\fft_radix%%R_avx512_ct_t1s_dit.h" 2>nul
    python "%GEN_DIR%gen_radix%%R.py" --isa avx512 --variant ct_t1_dit_log3 > "%AVX512_DIR%\fft_radix%%R_avx512_ct_t1_dit_log3.h" 2>nul
    python "%GEN_DIR%gen_radix%%R.py" --isa avx512 --variant ct_t1_oop_dit > "%AVX512_DIR%\fft_radix%%R_avx512_ct_t1_oop_dit.h" 2>nul
)

:: R=32: ct_n1 + ct_t1_dit + ct_t1_oop_dit (no log3)
python "%GEN_DIR%gen_radix32.py" --isa avx512 --variant ct_n1 > "%AVX512_DIR%\fft_radix32_avx512_ct_n1.h" 2>nul
python "%GEN_DIR%gen_radix32.py" --isa avx512 --variant ct_t1_dit > "%AVX512_DIR%\fft_radix32_avx512_ct_t1_dit.h" 2>nul
python "%GEN_DIR%gen_radix32.py" --isa avx512 --variant ct_t1_oop_dit > "%AVX512_DIR%\fft_radix32_avx512_ct_t1_oop_dit.h" 2>nul

:: R=64: ct_n1 + ct_t1_dit + ct_t1_dit_log3 + ct_t1_oop_dit
python "%GEN_DIR%gen_radix64.py" --isa avx512 --variant ct_n1 > "%AVX512_DIR%\fft_radix64_avx512_ct_n1.h" 2>nul
python "%GEN_DIR%gen_radix64.py" --isa avx512 --variant ct_t1_dit > "%AVX512_DIR%\fft_radix64_avx512_ct_t1_dit.h" 2>nul
python "%GEN_DIR%gen_radix64.py" --isa avx512 --variant ct_t1_dit_log3 > "%AVX512_DIR%\fft_radix64_avx512_ct_t1_dit_log3.h" 2>nul
python "%GEN_DIR%gen_radix64.py" --isa avx512 --variant ct_t1_oop_dit > "%AVX512_DIR%\fft_radix64_avx512_ct_t1_oop_dit.h" 2>nul

echo   Done. %AVX512_DIR%

echo.
echo Generating scalar codelets...

:: R=2, R=4, R=8: legacy generators
for %%R in (2 4 8) do (
    python "%GEN_DIR%gen_radix%%R.py" scalar > "%SCALAR_DIR%\fft_radix%%R_scalar.h" 2>nul
)

:: R=3,5,6,7,10,11,12,13,16,17,19,20,25: ct_n1 + ct_t1_dit + ct_t1s_dit + ct_t1_dit_log3 + ct_t1_oop_dit
for %%R in (3 5 6 7 10 11 12 13 16 17 19 20 25) do (
    python "%GEN_DIR%gen_radix%%R.py" --isa scalar --variant ct_n1 > "%SCALAR_DIR%\fft_radix%%R_scalar_ct_n1.h" 2>nul
    python "%GEN_DIR%gen_radix%%R.py" --isa scalar --variant ct_t1_dit > "%SCALAR_DIR%\fft_radix%%R_scalar_ct_t1_dit.h" 2>nul
    python "%GEN_DIR%gen_radix%%R.py" --isa scalar --variant ct_t1s_dit > "%SCALAR_DIR%\fft_radix%%R_scalar_ct_t1s_dit.h" 2>nul
    python "%GEN_DIR%gen_radix%%R.py" --isa scalar --variant ct_t1_dit_log3 > "%SCALAR_DIR%\fft_radix%%R_scalar_ct_t1_dit_log3.h" 2>nul
    python "%GEN_DIR%gen_radix%%R.py" --isa scalar --variant ct_t1_oop_dit > "%SCALAR_DIR%\fft_radix%%R_scalar_ct_t1_oop_dit.h" 2>nul
)

:: R=32: ct_n1 + ct_t1_dit + ct_t1_oop_dit (no log3)
python "%GEN_DIR%gen_radix32.py" --isa scalar --variant ct_n1 > "%SCALAR_DIR%\fft_radix32_scalar_ct_n1.h" 2>nul
python "%GEN_DIR%gen_radix32.py" --isa scalar --variant ct_t1_dit > "%SCALAR_DIR%\fft_radix32_scalar_ct_t1_dit.h" 2>nul
python "%GEN_DIR%gen_radix32.py" --isa scalar --variant ct_t1_oop_dit > "%SCALAR_DIR%\fft_radix32_scalar_ct_t1_oop_dit.h" 2>nul

:: R=64: ct_n1 + ct_t1_dit + ct_t1_dit_log3 + ct_t1_oop_dit
python "%GEN_DIR%gen_radix64.py" --isa scalar --variant ct_n1 > "%SCALAR_DIR%\fft_radix64_scalar_ct_n1.h" 2>nul
python "%GEN_DIR%gen_radix64.py" --isa scalar --variant ct_t1_dit > "%SCALAR_DIR%\fft_radix64_scalar_ct_t1_dit.h" 2>nul
python "%GEN_DIR%gen_radix64.py" --isa scalar --variant ct_t1_dit_log3 > "%SCALAR_DIR%\fft_radix64_scalar_ct_t1_dit_log3.h" 2>nul
python "%GEN_DIR%gen_radix64.py" --isa scalar --variant ct_t1_oop_dit > "%SCALAR_DIR%\fft_radix64_scalar_ct_t1_oop_dit.h" 2>nul

echo   Done. %SCALAR_DIR%

endlocal
