@echo off
set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer;%PATH%"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >/dev/null
set "PATH=C:\Program Files (x86)\Intel\oneAPI\compiler\2025.3\bin;%PATH%"
set "LIB=C:\Program Files (x86)\Intel\oneAPI\compiler\2025.3\lib;%LIB%"
if exist build_icx_ex rmdir /s /q build_icx_ex
"C:\vcpkg\downloads\tools\cmake-3.30.1-windows\cmake-3.30.1-windows-i386\bin\cmake.exe" -S . -B build_icx_ex -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=icx -DCMAKE_MAKE_PROGRAM="C:\vcpkg\downloads\tools\ninja\1.13.1-windows\ninja.exe"
"C:\vcpkg\downloads\tools\cmake-3.30.1-windows\cmake-3.30.1-windows-i386\bin\cmake.exe" --build build_icx_ex
