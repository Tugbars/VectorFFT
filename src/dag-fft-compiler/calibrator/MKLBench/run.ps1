# run.ps1 - dag-fft-compiler (JIT) vs Intel MKL, 1D C2C forward.
# Runs each wisdom cell in an ISOLATED process (fresh per cell) to avoid the
# cross-cell cache/thermal carryover that contaminates a single sequential run
# (cachebust alone does not clear it). Collects all rows into ONE results CSV.
#
#   powershell -ExecutionPolicy Bypass -File MKLBench\run.ps1 [wisdom] [csv]
#
# Requires: build.ps1 run first; mkl_rt.dll (mkl\latest\bin) + libwinpthread-1.dll
# (mingw bin) on PATH - set here automatically.
param(
  [string]$Wisdom = "",
  [string]$Csv = "",
  [string]$Mkl = "C:\Program Files (x86)\Intel\oneAPI\mkl\latest",
  [string]$MingwBin = "C:\mingw152\mingw64\bin"
)
$here = $PSScriptRoot
$root = Split-Path $here -Parent
if (-not $Wisdom) { $Wisdom = "$root\generator\generated\spike_wisdom.txt" }
if (-not $Csv)    { $Csv    = "$here\results.csv" }
$exe = "$here\bench_jit_vs_mkl.exe"
if (-not (Test-Path $exe)) { Write-Host "build first:  powershell -File MKLBench\build.ps1"; exit 1 }
$env:PATH = "$Mkl\bin;$MingwBin;$env:PATH"

"N,K,factors,path,vfft_ns,mkl_ns,vfft_gflops,ratio_vs_mkl,rt_err" | Set-Content $Csv -Encoding ascii
$lines = Get-Content $Wisdom | Where-Object { $_ -notmatch '^[#@]' -and $_.Trim() -ne '' }
Write-Host "=== dag-JIT vs MKL (isolated per cell)  wisdom: $Wisdom ==="
Write-Host ("{0,-8} {1,-5} {2,-18} {3,-6} {4,12} {5,12} {6,7}" -f "N","K","factors","path","vfft_ns","mkl_ns","ratio")
foreach ($ln in $lines) {
  Set-Content "$here\_one.txt" -Encoding ascii -Value @("@version 5", $ln)
  & $exe "$here\_one.txt" "$here\_one.csv" 0 *> $null
  $row = Get-Content "$here\_one.csv" -ErrorAction SilentlyContinue | Select-Object -Skip 1
  if ($row) {
    $row | Add-Content $Csv -Encoding ascii
    $c = $row -split ","
    Write-Host ("{0,-8} {1,-5} {2,-18} {3,-6} {4,12} {5,12} {6,6}x" -f $c[0],$c[1],$c[2],$c[3],$c[4],$c[5],$c[7])
  }
}
Write-Host ""
Write-Host "results CSV -> $Csv"
