@echo off
setlocal enabledelayedexpansion

rem Set up MSVC env (gives cl.exe + Windows SDK LIB/INCLUDE)
set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer;!PATH!"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (echo vcvars64.bat failed & exit /b 1)

rem Layer Intel oneAPI
set "ONEAPI_COMPILER=C:\Program Files (x86)\Intel\oneAPI\compiler\2025.3"
set "PATH=!ONEAPI_COMPILER!\bin;!PATH!"
set "LIB=!ONEAPI_COMPILER!\lib;!LIB!"

set "ROOT=C:\Users\Tugbars\Desktop\highSpeedFFT"
set "INC=/I!ROOT!\include"
set "SRC=!ROOT!\bench_compiler.c"

echo === Building bench against each vfft lib ===

rem MSVC
echo.
echo [build] MSVC
cl /nologo /O2 /arch:AVX2 !INC! "!SRC!" /Fe:bench_compiler_msvc.exe /link !ROOT!\build_msvc\lib\Release\vfft.lib >nul
if errorlevel 1 (echo MSVC build failed & exit /b 1)

rem ICX
echo [build] ICX
icx /nologo /O2 /arch:AVX2 !INC! "!SRC!" /Fe:bench_compiler_icx.exe /link !ROOT!\build_icx\lib\vfft.lib >nul
if errorlevel 1 (echo ICX build failed & exit /b 1)

rem GCC (MinGW). Need separate PATH so cl/icx don't shadow gcc.
set "GCC_PATH=C:\ProgramData\mingw64\mingw64\bin"
echo [build] GCC
"!GCC_PATH!\gcc.exe" -O3 -mavx2 -mfma -DNDEBUG -I"!ROOT!\include" "!SRC!" -L"!ROOT!\build_gcc\lib" -lvfft -o bench_compiler_gcc.exe -lm
if errorlevel 1 (echo GCC build failed & exit /b 1)

echo.
echo === Running benches ===
echo.
.\bench_compiler_msvc.exe MSVC
.\bench_compiler_icx.exe  ICX
.\bench_compiler_gcc.exe  GCC

echo.
echo === Production toolchain (build_tuned/build.py via ICX) ===
echo (Reference number for comparison — same lib path the production benches use.)

endlocal
