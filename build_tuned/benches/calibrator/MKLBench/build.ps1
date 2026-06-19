# build.ps1 — build bench_jit_vs_mkl.exe (Windows).
# Single gcc executable: gcc exe + in-repo gcc codelets + gcc-JIT'd .dlls at runtime
# + Intel MKL via mkl_rt. KEY: do NOT define MKL_ILP64 — mkl_rt's DFTI is LP64
# (4-byte MKL_LONG); ILP64 corrupts the strides array -> DftiCommit "Inconsistent
# configuration parameters".
param(
  [string]$Gcc = "C:\mingw152\mingw64\bin\gcc.exe",
  [string]$Mkl = "C:\Program Files (x86)\Intel\oneAPI\mkl\latest"
)
$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
$root = Split-Path $here -Parent          # ...\dag-fft-compiler
$rsp  = "@$root/jit/generated/codelets.rsp"   # in-repo codelet objects (build_codelets.ps1)
$cf = @("-O3","-mavx2","-mfma","-march=haswell","-D_GNU_SOURCE",
        "-Wno-incompatible-pointer-types","-Wno-unused-result")
& $Gcc @cf "-I$Mkl\include" -c "$here\bench_jit_vs_mkl.c" -o "$here\bench_jit_vs_mkl.o"
& $Gcc "$here\bench_jit_vs_mkl.o" $rsp "$Mkl\lib\mkl_rt.lib" -lm -o "$here\bench_jit_vs_mkl.exe"
Write-Host "built $here\bench_jit_vs_mkl.exe"
