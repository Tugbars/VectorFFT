<#
  ab_codelets_icx.ps1 — A/B microbench of OLD vs NEW in-place c2c codelets, built
  with Intel ICX (the production Windows compiler), Windows-native.

    OLD = src/prototype/codelets/avx2          (pre-holiday DAG compiler; M-pins on)
    NEW = src/dag-fft-compiler/codelets/inplace/avx2  (new DAG compiler; pins off)

  The Windows/ICX sibling of ab_codelets.sh. Same idea: for every codelet SYMBOL
  present in BOTH trees (restricted here to pow2 radices >= 4), build that symbol
  from each tree into its own tiny exe (microbench_codelet.c) and time it. Driver,
  microbench, compiler, and flags identical for both sides => the only variable is
  the codelet machine code.

  Two-phase: build everything first, THEN measure pinned + paced (cooldown before
  every run, A/B order flipped each round, best-of-N inside each exe, min over
  rounds). Self-contained: imports oneAPI setvars (PS shell state does not persist
  across the harness's own process boundary, so we set the env in-process here).

  RUN (from a normal PowerShell):
    powershell -ExecutionPolicy Bypass -File ab_codelets_icx.ps1
  Params: -Radices "4 8 16 32 64"  -Rounds 5  -CooldownMs 150  -PinCpu 2
          -RepsBudget 4000000  -BestOf 15  -Limit 0   (Limit>0 smokes the harness)
  Output: results/ab_old_vs_new_icx_pow2.csv  +  console summary.
#>
param(
  [string]$Radices    = "4 8 16 32 64",   # pow2 >= 4 (NEW inplace tops out at 64)
  [int]   $Rounds     = 5,
  [int]   $CooldownMs = 150,
  [int]   $PinCpu     = 2,                 # logical CPU to pin measurement to; -1 = none
  [string]$RepsBudget = "4000000",
  [string]$BestOf     = "15",
  [int]   $Limit      = 0
)
# Default ErrorActionPreference (Continue): native-tool stderr must NOT abort us.

$REPO      = "C:\Users\Tugbars\Desktop\highSpeedFFT\src"
$OLD       = "$REPO\prototype\codelets\avx2"
$NEW       = "$REPO\dag-fft-compiler\codelets\inplace\avx2"
$MB        = "$REPO\dag-fft-compiler\benchmarks\microbench_codelet.c"
$OUTDIR    = "$REPO\dag-fft-compiler\benchmarks\results"
$OUT       = "$OUTDIR\ab_old_vs_new_icx_pow2.csv"
$WORK      = "C:\tmp\abicx\bin"
$SETVARS   = "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
$INTEL_LIB = "C:\Program Files (x86)\Intel\oneAPI\compiler\2025.3\lib"

New-Item -ItemType Directory -Force $WORK   | Out-Null
New-Item -ItemType Directory -Force $OUTDIR | Out-Null

# ---- 1. oneAPI environment ------------------------------------------------
# setvars gives MSVC/WinSDK INCLUDE+LIB and PATH to icx; it does NOT add the
# Intel runtime lib dir (libircmt.lib), so prepend it explicitly.
cmd /c "`"$SETVARS`" 1>nul 2>nul && set" | ForEach-Object {
  if ($_ -match '^([^=]+)=(.*)$') { Set-Item "env:$($matches[1])" $matches[2] }
}
if ($env:LIB -notlike "*$INTEL_LIB*") { $env:LIB = "$INTEL_LIB;$env:LIB" }
$icx = (Get-Command icx -ErrorAction SilentlyContinue)
if (-not $icx) { Write-Output "FATAL: icx not on PATH after setvars"; exit 1 }
Write-Output ("icx: {0}" -f $icx.Source)

# ---- 2. scan -> symbol maps (in-place c2c only, pow2 radix in -Radices) ----
$want = @{}
foreach ($r in ($Radices -split '\s+')) { if ($r) { $want[[int]$r] = $true } }

function Scan-Tree($dir) {
  $m = @{}
  Get-ChildItem -Path $dir -Recurse -Filter *.c -ErrorAction SilentlyContinue | ForEach-Object {
    $f = $_.FullName
    if ($f -match 'strided|[\\/]oop[\\/]|[\\/]rfft[\\/]|[\\/]c2r[\\/]|[\\/]trig[\\/]') { return }
    $hit = Select-String -LiteralPath $f -Pattern 'radix(\d+)_(n1|t1s|t1)_[a-z0-9_]*avx2' -List
    if (-not $hit) { return }
    $sym = $hit.Matches[0].Value
    $rad = [int]$hit.Matches[0].Groups[1].Value
    if (-not $want.ContainsKey($rad)) { return }
    if (-not $m.ContainsKey($sym)) { $m[$sym] = $f }
  }
  return $m
}

$oldm = Scan-Tree $OLD
$newm = Scan-Tree $NEW
$common = @()
foreach ($s in $newm.Keys) { if ($oldm.ContainsKey($s)) { $common += $s } }
$common = $common | Sort-Object
if ($Limit -gt 0) { $common = @($common | Select-Object -First $Limit) }
Write-Output ("OLD symbols: {0}   NEW symbols: {1}   common: {2}" -f $oldm.Count, $newm.Count, $common.Count)
if ($common.Count -eq 0) { Write-Output "no common symbols"; exit 1 }

# ---- 3. build phase (cmd /c isolates native streams from PS error stream) --
$CF = "-O3 -mavx2 -mfma -w"
function Build-One($sym, $src, $exe) {
  $rad = [int]([regex]::Match($sym, '^radix(\d+)_').Groups[1].Value)
  $t1s = 0; if ($sym -match '_t1s_') { $t1s = 1 }
  $line = "icx $CF -DRN=$rad -DFN=$sym -DT1S=$t1s `"$MB`" `"$src`" -o `"$exe`" 1>nul 2>nul"
  cmd /c $line
  return (($LASTEXITCODE -eq 0) -and (Test-Path $exe))
}

Write-Output ("building {0} pairs ..." -f $common.Count)
$built = @()
foreach ($sym in $common) {
  $oe = "$WORK\old_$sym.exe"; $ne = "$WORK\new_$sym.exe"
  $ok1 = Build-One $sym $oldm[$sym] $oe
  $ok2 = Build-One $sym $newm[$sym] $ne
  if ($ok1 -and $ok2) { $built += $sym } else { Write-Output ("  BUILD-FAIL {0} (old={1} new={2})" -f $sym, $ok1, $ok2) }
}
Write-Output ("built {0}/{1} pairs" -f $built.Count, $common.Count)
if ($built.Count -eq 0) { Write-Output "nothing built"; exit 1 }

# ---- 4. measure phase (pinned, paced, min-over-rounds) --------------------
$env:MB_REPS_BUDGET = $RepsBudget
$env:MB_BESTOF      = $BestOf
if ($PinCpu -ge 0) {
  try {
    (Get-Process -Id $PID).ProcessorAffinity = [IntPtr][long][math]::Pow(2, $PinCpu)
    Write-Output ("pinned measurement to cpu{0}" -f $PinCpu)
  } catch { Write-Output ("WARN: could not pin (cpu{0}); running unpinned" -f $PinCpu) }
}

function Run-Ns($exe) {
  $o = (& $exe) -join "`n"
  if ($o -match 'ns=([0-9.]+)') { return [double]$matches[1] } else { return [double]::NaN }
}

$oldNs = @{}; $newNs = @{}
foreach ($s in $built) { $oldNs[$s] = [double]1e30; $newNs[$s] = [double]1e30 }

for ($r = 1; $r -le $Rounds; $r++) {
  Write-Output ("round {0}/{1}" -f $r, $Rounds)
  foreach ($sym in $built) {
    if ($r % 2 -eq 1) {
      Start-Sleep -Milliseconds $CooldownMs; $o = Run-Ns "$WORK\old_$sym.exe"
      Start-Sleep -Milliseconds $CooldownMs; $n = Run-Ns "$WORK\new_$sym.exe"
    } else {
      Start-Sleep -Milliseconds $CooldownMs; $n = Run-Ns "$WORK\new_$sym.exe"
      Start-Sleep -Milliseconds $CooldownMs; $o = Run-Ns "$WORK\old_$sym.exe"
    }
    if (-not [double]::IsNaN($o) -and $o -lt $oldNs[$sym]) { $oldNs[$sym] = $o }
    if (-not [double]::IsNaN($n) -and $n -lt $newNs[$sym]) { $newNs[$sym] = $n }
  }
}

# ---- 5. report ------------------------------------------------------------
"symbol,radix,old_ns,new_ns,ratio_old_over_new" | Set-Content -Encoding ascii $OUT
$rows = @()
foreach ($sym in $built) {
  $rad = [int]([regex]::Match($sym, '^radix(\d+)_').Groups[1].Value)
  $o = $oldNs[$sym]; $n = $newNs[$sym]
  $ratio = if ($n -gt 0) { $o / $n } else { 0 }
  ("{0},{1},{2:N4},{3:N4},{4:N4}" -f $sym, $rad, $o, $n, $ratio) | Add-Content -Encoding ascii $OUT
  $rows += [pscustomobject]@{ sym = $sym; rad = $rad; ratio = $ratio }
}

$valid = @($rows | Where-Object { $_.ratio -gt 0 })
Write-Output ""
Write-Output "================ BY RADIX (ratio=old/new; >1 NEW faster) ================"
$valid | Group-Object rad | Sort-Object { [int]$_.Name } | ForEach-Object {
  $g = $_.Group; $gn = $g.Count
  $gsl = ($g | ForEach-Object { [math]::Log($_.ratio) } | Measure-Object -Sum).Sum
  $gw  = @($g | Where-Object { $_.ratio -ge 1.0 }).Count
  Write-Output ("  radix {0,-3} cells={1,-3} geomean={2:N4}  NEWwin={3}/{1}" -f $_.Name, $gn, [math]::Exp($gsl/$gn), $gw)
}

$n  = $valid.Count
$sl = ($valid | ForEach-Object { [math]::Log($_.ratio) } | Measure-Object -Sum).Sum
$wins = @($valid | Where-Object { $_.ratio -ge 1.0 }).Count
$best = $valid | Sort-Object ratio -Descending | Select-Object -First 1
$worst = $valid | Sort-Object ratio | Select-Object -First 1
Write-Output ""
Write-Output "================ OVERALL (ICX) ================"
Write-Output ("  cells={0}  NEWfaster={1}  OLDfaster={2}  geomean={3:N4}" -f $n, $wins, ($n-$wins), [math]::Exp($sl/$n))
Write-Output ("  best  NEW: {0:N3}x  ({1})" -f $best.ratio, $best.sym)
Write-Output ("  worst NEW: {0:N3}x  ({1})" -f $worst.ratio, $worst.sym)
Write-Output ""
Write-Output ("CSV: {0}" -f $OUT)
