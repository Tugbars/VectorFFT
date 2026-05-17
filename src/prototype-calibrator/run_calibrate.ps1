# run_calibrate.ps1 — orchestrate prototype-core wisdom calibration.
#
# Runs the calibrate.exe binary on (N, K) cells, pinned to CPU 2 (first
# clean P-core), priority=High. Edits src/prototype/generated/spike_wisdom.txt.
#
# Three operation modes:
#
#   1. Single cell (default behavior when -N/-K is given): always OVERWRITE
#      the entry for the requested (N, K, orient). For ad-hoc recalibration.
#
#        run_calibrate.ps1 -N 1024 -K 4
#
#   2. Batch fill-missing (default when -Cells is given): only calibrate
#      cells that don't already have a wisdom entry for the chosen orient.
#      Cells already present are skipped. Use -Force to override.
#
#        run_calibrate.ps1 -Cells "8:4,16:4,32:4,64:4"
#        run_calibrate.ps1 -Cells "8:4,16:4" -Force   # overwrite even if present
#
#   3. Overwrite-all: backup the current wisdom file to backup/, then
#      recalibrate every cell currently in the file (same orient as each
#      existing entry).
#
#        run_calibrate.ps1 -OverwriteAll
#
# Other flags:
#   -Mode best|patient|dp|estimate   (default: best — runs all 3, picks fastest)
#   -Orient dit|dif|both             (default: dit — DIF code path plumbed but inactive)
#
# Prereqs: power plan = High Performance; calibrate.exe built.

param(
    [int]$N,
    [int]$K,
    [string]$Cells,
    [switch]$Force,
    [switch]$OverwriteAll,
    [string]$Mode = "best",
    [string]$Orient = "dit",
    [int]$CpuAffinity = 0x4   # CPU 2 only (skip CPU 0/1 which handle interrupts)
)

$exe = "c:\Users\Tugbars\Desktop\highSpeedFFT\src\prototype\build_tuned\calibrate.exe"
$wisdomFile = "c:\Users\Tugbars\Desktop\highSpeedFFT\src\prototype\generated\spike_wisdom.txt"
$sessionLog = "c:\tmp\calibrate_session.log"

# Parse a wisdom data line into (N, K, nf, use_dif_forward). Returns
# $null for comments / blank lines / parse failures.
function Parse-WisdomLine {
    param([string]$Line)
    $trimmed = $Line.TrimStart()
    if ($trimmed.Length -eq 0 -or $trimmed[0] -eq "#" -or $trimmed[0] -eq "@") {
        return $null
    }
    $toks = $trimmed -split '\s+'
    if ($toks.Count -lt 3) { return $null }
    try {
        $lineN  = [int]$toks[0]
        $lineK  = [int]$toks[1]
        $lineNf = [int]$toks[2]
        $useDifIdx = $lineNf + 7
        if ($toks.Count -le $useDifIdx) { return $null }
        $lineDif = [int]$toks[$useDifIdx]
        return @{ N = $lineN; K = $lineK; Nf = $lineNf; UseDif = $lineDif }
    } catch {
        return $null
    }
}

# Check if a cell (N, K, use_dif_forward) is already in the wisdom file.
function Test-WisdomEntryExists {
    param([string]$WisdomPath, [int]$TargetN, [int]$TargetK, [int]$TargetDif)
    if (-not (Test-Path $WisdomPath)) { return $false }
    foreach ($line in Get-Content $WisdomPath) {
        $p = Parse-WisdomLine -Line $line
        if ($null -eq $p) { continue }
        if ($p.N -eq $TargetN -and $p.K -eq $TargetK -and $p.UseDif -eq $TargetDif) {
            return $true
        }
    }
    return $false
}

# Read all (N, K, UseDif) cells from the wisdom file.
function Get-WisdomCells {
    param([string]$WisdomPath)
    $cells = New-Object System.Collections.Generic.List[hashtable]
    if (-not (Test-Path $WisdomPath)) { return $cells }
    foreach ($line in Get-Content $WisdomPath) {
        $p = Parse-WisdomLine -Line $line
        if ($null -eq $p) { continue }
        $cells.Add(@{ N = $p.N; K = $p.K; UseDif = $p.UseDif }) | Out-Null
    }
    return $cells
}

# Backup the wisdom file to a timestamped file under backup/.
function Backup-WisdomFile {
    param([string]$WisdomPath)
    if (-not (Test-Path $WisdomPath)) {
        Write-Host "[backup] no wisdom file at $WisdomPath — nothing to back up"
        return
    }
    $wisdomDir = Split-Path -Parent $WisdomPath
    $backupDir = Join-Path $wisdomDir "backup"
    if (-not (Test-Path $backupDir)) {
        New-Item -ItemType Directory -Path $backupDir | Out-Null
    }
    $stamp = [DateTime]::Now.ToString('yyyyMMdd_HHmmss')
    $backupName = "spike_wisdom_$stamp.txt"
    $backupPath = Join-Path $backupDir $backupName
    Copy-Item -Path $WisdomPath -Destination $backupPath
    Write-Host "[backup] wisdom saved to $backupPath"
}

# Remove any existing wisdom entry matching (N, K, use_dif_forward).
# Production wisdom format: "N K nf factors... best_ns use_blocked split block_groups use_dif_forward variants..."
# use_dif_forward is at token index (nf + 7) (zero-indexed): N=0, K=1, nf=2, factors=3..3+nf-1, best_ns=3+nf, use_blocked, split, block_groups, use_dif_forward.
function Remove-MatchingWisdomEntry {
    param([string]$WisdomPath, [int]$TargetN, [int]$TargetK, [int]$TargetDif)
    if (-not (Test-Path $WisdomPath)) { return }
    $lines = Get-Content $WisdomPath
    $kept = New-Object System.Collections.Generic.List[string]
    $removed = 0
    for ($i = 0; $i -lt $lines.Count; $i++) {
        $line = $lines[$i]
        $trimmed = $line.TrimStart()
        if ($trimmed.Length -eq 0 -or $trimmed[0] -eq "#" -or $trimmed[0] -eq "@") {
            # Comment / version line — keep (unless it's the provenance comment
            # immediately preceding the line we're about to remove; handled below).
            $kept.Add($line) | Out-Null
            continue
        }
        # Parse data line.
        $toks = $trimmed -split '\s+'
        if ($toks.Count -lt 3) { $kept.Add($line) | Out-Null; continue }
        try {
            $lineN  = [int]$toks[0]
            $lineK  = [int]$toks[1]
            $lineNf = [int]$toks[2]
            $useDifIdx = $lineNf + 7
            if ($toks.Count -le $useDifIdx) { $kept.Add($line) | Out-Null; continue }
            $lineDif = [int]$toks[$useDifIdx]
        } catch {
            $kept.Add($line) | Out-Null
            continue
        }
        if ($lineN -eq $TargetN -and $lineK -eq $TargetK -and $lineDif -eq $TargetDif) {
            # Drop this line. Also drop the immediately-preceding line if it's an
            # auto-generated provenance comment (starts with "# patient" or "# calibrate").
            if ($kept.Count -gt 0) {
                $last = $kept[$kept.Count - 1]
                if ($last -match '^\s*#\s*(patient|calibrate)\s+') {
                    $kept.RemoveAt($kept.Count - 1)
                }
            }
            $removed++
            continue
        }
        $kept.Add($line) | Out-Null
    }
    if ($removed -gt 0) {
        Set-Content -Path $WisdomPath -Value $kept
        Write-Host "  [orchestrator] removed $removed existing entry/entries for N=$TargetN K=$TargetK use_dif_forward=$TargetDif"
    }
}

if (-not (Test-Path $exe)) {
    Write-Host "[ERROR] calibrate.exe not found at $exe"
    Write-Host "Build it first: bash src/prototype-calibrator/build_calibrate.sh"
    exit 1
}

# Collect cells: either single (N, K) or comma-separated "N:K" list.
$cellList = @()
if ($Cells) {
    foreach ($pair in $Cells -split ",") {
        $parts = $pair.Trim() -split ":"
        if ($parts.Count -ne 2) { Write-Host "[ERROR] bad cell '$pair' (expected N:K)"; exit 1 }
        $cellList += @{ N = [int]$parts[0]; K = [int]$parts[1] }
    }
} elseif ($N -and $K) {
    $cellList += @{ N = $N; K = $K }
} else {
    Write-Host "[ERROR] specify -N and -K, OR -Cells 'N1:K1,N2:K2,...'"
    exit 1
}

Write-Host "[orchestrator] $($cellList.Count) cells, mode=$Mode orient=$Orient cpu_affinity=0x$([Convert]::ToString($CpuAffinity, 16))"
$activePlan = (powercfg /getactivescheme) -replace ".*\((.*)\)\s*$", '$1'
Write-Host "[orchestrator] active power plan: $activePlan"
if ($activePlan -notmatch "High|Ultimate") {
    Write-Host "[WARN] power plan is not High Performance/Ultimate. Set with: powercfg /setactive SCHEME_MIN"
}

foreach ($cell in $cellList) {
    $cellN = $cell.N
    $cellK = $cell.K
    $logErr = "c:\tmp\calib_n${cellN}_k${cellK}.err"
    $logOut = "c:\tmp\calib_n${cellN}_k${cellK}.txt"

    Write-Host "[$([DateTime]::Now.ToString('HH:mm:ss'))] N=$cellN K=$cellK starting..."

    $proc = Start-Process -FilePath $exe `
        -ArgumentList $cellN, $cellK, "--mode", $Mode, "--orient", $Orient `
        -PassThru -NoNewWindow `
        -RedirectStandardOutput $logOut `
        -RedirectStandardError $logErr
    $proc.ProcessorAffinity = $CpuAffinity
    $proc.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::High
    $proc.WaitForExit()
    # Start-Process + RedirectStandardOutput sometimes leaves ExitCode null.
    # Use the file content as the success signal: a wisdom line starts with
    # the cell's N and is non-empty.
    $rawOut = Get-Content $logOut -Raw
    if (-not $rawOut -or $rawOut.Trim() -eq "" -or $rawOut.Trim() -notmatch "^$cellN\s") {
        Write-Host "[$([DateTime]::Now.ToString('HH:mm:ss'))] N=$cellN K=$cellK FAILED (no wisdom line)"
        Get-Content $logErr | Select-Object -Last 10
        continue
    }

    $wisdomLine = (Get-Content $logOut -Raw).Trim()
    $winnerLine = Get-Content $logErr | Select-String "WINNER" | Select-Object -Last 1

    Write-Host "[$([DateTime]::Now.ToString('HH:mm:ss'))] DONE: $winnerLine"
    Write-Host "  wisdom -> $wisdomLine"

    # Parse winning use_dif_forward from the new wisdom line — we replace
    # the entry matching (N, K, that orient) so DIT and DIF entries for
    # the same (N, K) can coexist.
    $newToks = $wisdomLine -split '\s+'
    $newNf = [int]$newToks[2]
    $newDif = [int]$newToks[$newNf + 7]

    # Append to session log (audit trail; never trimmed).
    $timestamp = [DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss')
    Add-Content -Path $sessionLog -Value "# $timestamp  N=$cellN K=$cellK  mode=$Mode orient=$Orient"
    Add-Content -Path $sessionLog -Value $wisdomLine

    # Remove existing entry for (N, K, use_dif_forward), then append new one.
    Remove-MatchingWisdomEntry -WisdomPath $wisdomFile -TargetN $cellN -TargetK $cellK -TargetDif $newDif
    Add-Content -Path $wisdomFile -Value "# calibrate $timestamp  N=$cellN K=$cellK mode=$Mode orient=$Orient CPU2-pinned"
    Add-Content -Path $wisdomFile -Value $wisdomLine
}

Write-Host "[orchestrator] done. wisdom appended to $wisdomFile"
Write-Host "[orchestrator] regen plan_executors.h next: wsl --cd /mnt/c/tmp bash regen_executor.sh"
