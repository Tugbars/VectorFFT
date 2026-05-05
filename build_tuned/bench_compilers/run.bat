@echo off
setlocal enabledelayedexpansion

rem Compare MSVC / ICX / GCC builds of the same N=1024 K=256 bench.
rem Run from project root: build_tuned\bench_compilers\run.bat

rem Set up MSVC env (gives cl.exe + Windows SDK LIB/INCLUDE)
set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer;!PATH!"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (echo vcvars64.bat failed & exit /b 1)

rem Layer Intel oneAPI on top
set "ONEAPI_COMPILER=C:\Program Files (x86)\Intel\oneAPI\compiler\2025.3"
set "PATH=!ONEAPI_COMPILER!\bin;!PATH!"
set "LIB=!ONEAPI_COMPILER!\lib;!LIB!"

set "ROOT=C:\Users\Tugbars\Desktop\highSpeedFFT"
set "INC=/I!ROOT!\include"
set "OUTDIR=!ROOT!\build_tuned\bench_compilers"
set "SRC=!OUTDIR!\bench_compiler.c"

echo === Building bench against each vfft lib ===

echo.
echo [build] MSVC
cl /nologo /O2 /MD /arch:AVX2 !INC! "!SRC!" /Fe:"!OUTDIR!\bench_compiler_msvc.exe" /Fo"!OUTDIR!\\" /link !ROOT!\build_msvc\lib\Release\vfft.lib

echo.
echo [build] ICX
icx /nologo /O2 /MD /arch:AVX2 !INC! "!SRC!" /Fe:"!OUTDIR!\bench_compiler_icx.exe" /Fo"!OUTDIR!\\" /link !ROOT!\build_icx\lib\vfft.lib

echo.
echo [build] GCC
"C:\ProgramData\mingw64\mingw64\bin\gcc.exe" -O3 -mavx2 -mfma -DNDEBUG -I"!ROOT!\include" "!SRC!" -L"!ROOT!\build_gcc\lib" -lvfft -o "!OUTDIR!\bench_compiler_gcc.exe" -lm

echo.
echo === Running benches (best-of-3) ===
echo.
for %%i in (1 2 3) do (
    "!OUTDIR!\bench_compiler_msvc.exe" "MSVC#%%i"
    "!OUTDIR!\bench_compiler_icx.exe"  "ICX#%%i"
    set "PATH=C:\ProgramData\mingw64\mingw64\bin;!PATH!"
    "!OUTDIR!\bench_compiler_gcc.exe"  "GCC#%%i"
)

endlocal
