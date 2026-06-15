# run_vtune.ps1 — build + VTune-profile ONE in-place codelet (uarch-exploration).
# Answers: do the gcc reg->reg vmovapd cost port cycles, or are they move-eliminated?
# See vtune_codelet.c header for how to read the result.
#
#   powershell -ExecutionPolicy Bypass -File benchmarks\run_vtune.ps1
#   powershell -ExecutionPolicy Bypass -File benchmarks\run_vtune.ps1 -Codelet radix64_t1_dit_fwd_avx2 -R 64 -Src ...\r64_t1_dit_fwd.c
#
# Requires: gcc (mingw), Intel VTune (oneAPI).
# IMPORTANT: uarch-exploration uses HARDWARE event-based sampling -> RUN THIS FROM AN
# ADMINISTRATOR PowerShell (or load the VTune sampling driver). A non-elevated shell
# fails with "Cannot enable Hardware Event-Based Sampling". (Driverless sw sampling,
# -collect hotspots -knob sampling-mode=sw, works unelevated but gives only time/line
# attribution, NOT port utilization — so it can't answer the vmovapd-cost question.)
param(
  [string]$Codelet = "radix32_t1_dit_fwd_avx2",
  [int]   $R       = 32,
  [int]   $T1S     = 0,
  [string]$Src     = "",          # codelet .c; default = R32 t1 dit fwd avx2
  [int]   $Seconds = 8,
  [int]   $Core    = 2,           # P-core thread on the 14900KF
  [string]$Gcc     = "C:\mingw152\mingw64\bin\gcc.exe",
  [string]$Vtune   = "C:\Program Files (x86)\Intel\oneAPI\vtune\latest\bin64\vtune.exe"
)
$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
$root = Split-Path $here -Parent                       # ...\dag-fft-compiler
if (-not $Src) { $Src = "$root\codelets\inplace\avx2\r32_t1_dit_fwd.c" }
if (-not (Test-Path $Src))   { throw "codelet source not found: $Src" }
if (-not (Test-Path $Vtune)) { throw "vtune.exe not found: $Vtune" }

$out = "$here\vtune_out"; New-Item -ItemType Directory -Force $out | Out-Null
$exe = "$out\vtune_$Codelet.exe"
$cf  = @("-O3","-g","-mavx2","-mfma","-march=native","-Wno-incompatible-pointer-types","-Wno-unused-result")

Write-Host "=== build ($Codelet) ==="
& $Gcc @cf "-DRN=$R" "-DFN=$Codelet" "-DT1S=$T1S" "$here\vtune_codelet.c" $Src -lm -o $exe
if ($LASTEXITCODE -ne 0) { throw "gcc build failed" }

Write-Host "=== sanity run (1s) ==="
& $exe 1 $Core

$rdir = "$out\vt_$Codelet"
if (Test-Path $rdir) { Remove-Item $rdir -Recurse -Force }
Write-Host "=== vtune uarch-exploration ($Seconds s, core $Core) ==="
& $Vtune -collect uarch-exploration -knob sampling-interval=0.5 -result-dir $rdir -- $exe $Seconds $Core
if ($LASTEXITCODE -ne 0) {
  Write-Host "WARN: vtune collect returned $LASTEXITCODE."
  Write-Host "      If it says 'Cannot enable Hardware Event-Based Sampling', re-run this"
  Write-Host "      script from an ADMINISTRATOR PowerShell (or load the sampling driver)."
}

Write-Host "=== summary (top-down + port utilization; codelet dominates the process) ==="
$rep = "$out\summary_$Codelet.txt"
& $Vtune -report summary -result-dir $rdir | Tee-Object -FilePath $rep
Write-Host ""
Write-Host "report saved -> $rep"
Write-Host "per-source-line drill-down: `"$Vtune`-gui`" $rdir   (find vmovapd lines, see if any are hot)"
