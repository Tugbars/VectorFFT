@echo off
setlocal enabledelayedexpansion

rem Pre-add vswhere.exe location so vcvars64.bat can find it
set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer;!PATH!"

rem Source MSVC env (gives us cl.exe, link.exe, MSVC LIB + INCLUDE, Windows SDK)
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 (echo vcvars64.bat failed & exit /b 1)

rem Layer Intel oneAPI on top: PATH for icx.exe, LIB for libircmt.lib / svml_dispmt.lib / libmmt.lib
set "ONEAPI_COMPILER=C:\Program Files (x86)\Intel\oneAPI\compiler\2025.3"
if not exist "!ONEAPI_COMPILER!\bin\icx.exe" (echo icx.exe not found & exit /b 1)
set "PATH=!ONEAPI_COMPILER!\bin;!PATH!"
set "LIB=!ONEAPI_COMPILER!\lib;!LIB!"

rem Configure + build
if exist build_icx rmdir /s /q build_icx
"C:\vcpkg\downloads\tools\cmake-3.30.1-windows\cmake-3.30.1-windows-i386\bin\cmake.exe" -S . -B build_icx -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=icx -DCMAKE_MAKE_PROGRAM="C:\vcpkg\downloads\tools\ninja\1.13.1-windows\ninja.exe"
if errorlevel 1 (echo cmake configure failed & exit /b 1)

"C:\vcpkg\downloads\tools\cmake-3.30.1-windows\cmake-3.30.1-windows-i386\bin\cmake.exe" --build build_icx
if errorlevel 1 (echo build failed & exit /b 1)

echo.
echo === ICX BUILD SUCCESS ===
dir build_icx\lib 2>nul
endlocal
