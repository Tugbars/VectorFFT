<#
.SYNOPSIS
    Activate Windows High Performance power scheme, run measure_cpe via WSL,
    restore previous scheme.

.DESCRIPTION
    For meaningful CPE measurements on Raptor Lake and similar boosting CPUs,
    the Windows host needs to be locked into High Performance (no idle states
    that disable boost, no per-core frequency capping). This script:

      1. Reads the currently-active power scheme GUID.
      2. Switches to High Performance ("8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c").
      3. Launches measure_cpe under WSL (the binary itself is an ELF executable
         built by gcc-15; powercfg on the Windows host affects the same CPU
         that WSL workloads run on).
      4. Restores the previous power scheme on exit (even on Ctrl-C).

    Run as administrator for powercfg to take effect — without elevation, the
    /setactive call may silently no-op.

.PARAMETER Force
    Pass `--force` through to measure_cpe (bypass the 5% CV refuse threshold).

.PARAMETER NoEmit
    Pass `--no-emit` through (smoke run, don't overwrite radix_cpe.h).

.EXAMPLE
    .\scripts\run_measure_cpe.ps1 -NoEmit

.EXAMPLE
    .\scripts\run_measure_cpe.ps1 -Force
#>
param(
    [switch]$Force,
    [switch]$NoEmit
)

$HighPerfGuid = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"

Write-Host "[run_measure_cpe] Reading current power scheme..."
$active = & powercfg /getactivescheme
$previousGuid = ($active -split ":")[1].Trim().Split()[0]
Write-Host "[run_measure_cpe] Previous scheme: $previousGuid"

Write-Host "[run_measure_cpe] Activating High Performance ($HighPerfGuid)..."
& powercfg /setactive $HighPerfGuid
if ($LASTEXITCODE -ne 0) {
    Write-Host "[run_measure_cpe] WARNING: powercfg /setactive returned $LASTEXITCODE." -ForegroundColor Yellow
    Write-Host "[run_measure_cpe] Try running this script as Administrator." -ForegroundColor Yellow
}

# Brief settle delay so the scheme switch propagates before bench starts.
Start-Sleep -Seconds 2

try {
    $extraArgs = ""
    if ($Force) { $extraArgs += " --force" }
    if ($NoEmit) { $extraArgs += " --no-emit" }

    # Pin the bench to a single P-core. On Raptor Lake the OS scheduler
    # can hop the thread between P-core and E-core between batches, which
    # produces 10-25% CV at small radixes. Default = CPU 2, which is the
    # first thread of the second physical P-core — avoids HT contention
    # with CPU 0 (where Windows tends to schedule system threads).
    $cpu = if ($env:VFFT_PIN_CPU) { $env:VFFT_PIN_CPU } else { "2" }
    Write-Host "[run_measure_cpe] Pinning to logical CPU $cpu (set VFFT_PIN_CPU to override)"
    Write-Host "[run_measure_cpe] Launching measure_cpe via WSL..."
    # measure_cpe is an ELF binary; invoke via WSL bash with taskset.
    # chrt --rr 99 needs root, so try with sudo -n first and fall back
    # to plain taskset if sudo isn't passwordless-configured.
    $cmd = "cd /mnt/c/Users/Tugbars/Desktop/highSpeedFFT/src/prototype && " +
           "taskset -c $cpu build_tuned/measure_cpe$extraArgs"
    & wsl.exe bash -c $cmd
} finally {
    Write-Host "[run_measure_cpe] Restoring previous power scheme: $previousGuid"
    & powercfg /setactive $previousGuid
}
