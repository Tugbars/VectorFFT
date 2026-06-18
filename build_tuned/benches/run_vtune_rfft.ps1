# run_vtune_rfft.ps1 - uarch-exploration on the rfft high-K loss cell (N=256 K=256),
# OURS vs MKL r2c. EBS needs admin -- RUN FROM AN ADMINISTRATOR PowerShell:
#   powershell -ExecutionPolicy Bypass -File build_tuned\benches\run_vtune_rfft.ps1
#
# Profiles both engines (10s each, pinned core 2), writes summary_{ours,mkl}.txt,
# and prints the Top-Down / memory-bound lines for each. Assumes vtune_rfft.exe is
# already built (build_tuned/build.py --src benches/vtune_rfft.c --mkl --compile).
param(
  [int]$Seconds = 10,
  [int]$Core    = 2,
  [string]$Vtune = "C:\Program Files (x86)\Intel\oneAPI\vtune\latest\bin64\vtune.exe"
)
$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
$exe  = Join-Path $here "vtune_rfft.exe"
if (-not (Test-Path $exe))   { throw "build first: build_tuned/build.py --src benches/vtune_rfft.c --mkl --compile" }
if (-not (Test-Path $Vtune)) { throw "vtune.exe not found: $Vtune" }
$env:PATH = "C:\Program Files (x86)\Intel\oneAPI\mkl\latest\bin;C:\mingw152\mingw64\bin;" + $env:PATH

$modes = @("ours","mkl")
foreach ($mode in $modes) {
  $rd = Join-Path $here ("vt_rfft_" + $mode)
  if (Test-Path $rd) { Remove-Item $rd -Recurse -Force }
  Write-Host ("=== vtune uarch-exploration: " + $mode + " (" + $Seconds + "s, core " + $Core + ") ===")
  & $Vtune -collect uarch-exploration -knob sampling-interval=0.5 -result-dir $rd -- $exe $mode $Seconds $Core
  if ($LASTEXITCODE -ne 0) {
    Write-Host ("WARN: collect returned " + $LASTEXITCODE + ". If it says Cannot enable Hardware Event-Based Sampling,")
    Write-Host "      this shell is NOT elevated. Re-run from an Administrator PowerShell."
    continue
  }
  $sum = Join-Path $here ("summary_" + $mode + ".txt")
  & $Vtune -report summary -result-dir $rd | Tee-Object -FilePath $sum | Out-Null
  Write-Host ("  summary -> " + $sum)
}

Write-Host ""
Write-Host "==================== KEY METRICS (ours vs mkl) ===================="
$keys = "Retiring|Front-End Bound|Bad Speculation|Back-End Bound|Memory Bound|Core Bound|DRAM Bound|Memory Bandwidth|Memory Latency|L1 Bound|L2 Bound|L3 Bound|SQ Full|FB Full|Store Bound|DTLB|Split Stores|CPI|Port 0|Port 1|Port 5|Port 6|Average CPU Frequency|Elapsed Time"
foreach ($mode in $modes) {
  $sum = Join-Path $here ("summary_" + $mode + ".txt")
  if (-not (Test-Path $sum)) { continue }
  Write-Host ("---- " + $mode + " ----")
  Select-String -Path $sum -Pattern $keys | ForEach-Object { $_.Line.Trim() }
}
