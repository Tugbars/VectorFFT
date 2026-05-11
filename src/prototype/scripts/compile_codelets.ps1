<#
.SYNOPSIS
    Compile all generated codelets using the production compiler config
    from doc 38 (gcc-11 + -flive-range-shrinkage).

.DESCRIPTION
    Windows PowerShell port of scripts/compile_codelets.sh. Walks the
    codelets\ tree and compiles each .c into a .o file in place, using
    gcc-11 or whatever GCC is configured.

    Note: this script only handles compilation. The CSE/Algsimp pass
    selection per radix happens upstream in the GENERATOR
    (gen_radix.exe), NOT here. The generator gates several optimization
    passes by algorithm class — fma_lift is primes-only (doc 28),
    share_subsums and the transposition fixed-point loop are
    pow2/composite-only (doc 23, 28), and the spill recipe + SU
    scheduler are universal for any twiddled CT codelet ≥ R=5 (doc 13).
    See generate_codelets.ps1's help block for the full breakdown.

    By the time we get to compilation, every .c file already encodes
    the right algorithm-class-specific optimizations baked into its
    expression tree. This script just picks the right compiler config
    to translate them into asm without losing the wins.

    Compiler choice rationale (doc 38):
    - gcc-11 + -flive-range-shrinkage: 29% fewer stack ops at R=512
      AVX-512 vs gcc-13 default; 5-8% runtime gain at moderate B.
    - gcc-12 introduced an AVX-512 register allocator regression vs
      gcc-11 (-9.4% to -14% worse). gcc-13 inherits this.
    - Clang-18 is significantly worse on AVX-512 (3× more spills at R=512).
    - The -flive-range-shrinkage flag is asymmetric on AVX2 (helps
      small R, mildly hurts large R) but we apply it everywhere for
      CI simplicity.

    Windows GCC setup:
    - MSYS2 (https://www.msys2.org/) — recommended; provides gcc-11+
      via `pacman -S mingw-w64-x86_64-gcc`
    - WinLibs (https://winlibs.com/) — standalone gcc distribution
    - TDM-GCC — older but stable
    Make sure gcc.exe (and ideally gcc-11.exe specifically) is on PATH.

.PARAMETER VerifyOnly
    Switch: check sources compile but don't save .o files.

.EXAMPLE
    .\compile_codelets.ps1
    Compile all codelets with default config (gcc-11 + shrinkage).

.EXAMPLE
    $env:CC = "gcc"; $env:EXTRA_CFLAGS = ""; .\compile_codelets.ps1
    Use whatever gcc is on PATH, no extra flags (baseline comparison).

.EXAMPLE
    .\compile_codelets.ps1 -VerifyOnly
    Verify all sources compile without emitting .o files.

.NOTES
    Environment variables:
      CC            — compiler (default: gcc-11, falls back to gcc if not found)
      EXTRA_CFLAGS  — additional flags (default: -flive-range-shrinkage)
      JOBS          — parallelism (default: $env:NUMBER_OF_PROCESSORS)
      CODELETS_DIR  — input directory (default: <project_root>\codelets)
#>

param(
    [switch]$VerifyOnly
)

$ErrorActionPreference = "Stop"

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
$Root = (Resolve-Path "$PSScriptRoot\..").Path
$CodeletsDir = if ($env:CODELETS_DIR) { $env:CODELETS_DIR } else { Join-Path $Root "codelets" }
$CC = if ($env:CC) { $env:CC } else { "gcc-11" }
$ExtraCFlags = if ($env:EXTRA_CFLAGS -ne $null) { $env:EXTRA_CFLAGS } else { "-flive-range-shrinkage" }
$Jobs = if ($env:JOBS) { [int]$env:JOBS } else { [int]$env:NUMBER_OF_PROCESSORS }

# Per-ISA compile flags. Explicit -m* flags so codelets compiled on one
# machine match those compiled on another (no -march=native variance).
$Avx512Flags = "-mavx512f -mavx512dq -mfma -march=skylake-avx512"
$Avx2Flags   = "-mavx2 -mfma -march=haswell"

# ──────────────────────────────────────────────────────────────────────
# Sanity checks
# ──────────────────────────────────────────────────────────────────────
if (-not (Test-Path $CodeletsDir)) {
    Write-Host "ERROR: codelets directory not found: $CodeletsDir" -ForegroundColor Red
    Write-Host "Run .\scripts\generate_codelets.ps1 first."
    exit 1
}

# Check compiler is available; fall back to plain `gcc` if specific version not found
$CCResolved = Get-Command $CC -ErrorAction SilentlyContinue
if (-not $CCResolved) {
    Write-Host "WARNING: compiler '$CC' not found in PATH" -ForegroundColor Yellow
    $fallback = Get-Command "gcc" -ErrorAction SilentlyContinue
    if (-not $fallback) {
        Write-Host "ERROR: no gcc found in PATH at all." -ForegroundColor Red
        Write-Host "Install MSYS2 (https://www.msys2.org/) and run:"
        Write-Host "  pacman -S mingw-w64-x86_64-gcc"
        Write-Host "Then add C:\msys64\mingw64\bin to your PATH."
        exit 1
    }
    Write-Host "Falling back to plain 'gcc' (which may be different version)" -ForegroundColor Yellow
    $CC = "gcc"
}

Write-Host "═══════════════════════════════════════════════════════════════════"
Write-Host "  vfft_v2 codelet compilation (PowerShell)"
Write-Host "  CC:           $CC"
Write-Host "  EXTRA_CFLAGS: $ExtraCFlags"
Write-Host "  Codelets:     $CodeletsDir"
Write-Host "  Parallelism:  $Jobs jobs"
if ($VerifyOnly) { Write-Host "  Mode:         VerifyOnly (no .obj output)" -ForegroundColor Yellow }
Write-Host "═══════════════════════════════════════════════════════════════════"
Write-Host ""

# Find all .c files in the codelets tree
$Sources = Get-ChildItem -Path $CodeletsDir -Recurse -Filter "*.c" | Sort-Object FullName
$Total = $Sources.Count

if ($Total -eq 0) {
    Write-Host "No .c files found under $CodeletsDir"
    exit 1
}

Write-Host "Found $Total codelet sources. Compiling..."
Write-Host ""

# ──────────────────────────────────────────────────────────────────────
# Per-source compile function
# ──────────────────────────────────────────────────────────────────────
function Compile-One {
    param([System.IO.FileInfo]$Src)

    $srcPath = $Src.FullName
    # On Windows, .obj is the conventional object extension; gcc accepts both.
    # We use .o here to stay consistent with the bash script's output.
    $outPath = [System.IO.Path]::ChangeExtension($srcPath, ".o")
    if ($script:VerifyOnly) { $outPath = "NUL" }

    # Detect ISA from path
    $isaFlags = ""
    if ($srcPath -match '\\avx512\\') { $isaFlags = $script:Avx512Flags }
    elseif ($srcPath -match '\\avx2\\') { $isaFlags = $script:Avx2Flags }
    else {
        return @{ Success = $false; Path = $srcPath; Error = "Cannot determine ISA from path" }
    }

    # Compose argument list
    $argString = "-O3 $isaFlags $($script:ExtraCFlags) -c `"$srcPath`" -o `"$outPath`""

    $proc = Start-Process -FilePath $script:CC -ArgumentList $argString `
        -Wait -PassThru -NoNewWindow -RedirectStandardError "$srcPath.err" 2>$null

    if ($proc.ExitCode -ne 0) {
        $errText = if (Test-Path "$srcPath.err") { Get-Content "$srcPath.err" -Raw } else { "(no stderr)" }
        Remove-Item "$srcPath.err" -ErrorAction SilentlyContinue
        return @{ Success = $false; Path = $srcPath; Error = $errText }
    }
    Remove-Item "$srcPath.err" -ErrorAction SilentlyContinue
    return @{ Success = $true; Path = $srcPath }
}

# ──────────────────────────────────────────────────────────────────────
# Parallel compilation
# ──────────────────────────────────────────────────────────────────────
$TimeStart = Get-Date

# PowerShell 7+ has ForEach-Object -Parallel. PowerShell 5.1 doesn't, so
# we fall back to sequential there (slower but works everywhere).
$PSVersion = $PSVersionTable.PSVersion.Major

$Results = if ($PSVersion -ge 7) {
    # Parallel path (PowerShell 7+)
    $Sources | ForEach-Object -Parallel {
        $src = $_
        $srcPath = $src.FullName
        $outPath = [System.IO.Path]::ChangeExtension($srcPath, ".o")
        if ($using:VerifyOnly) { $outPath = "NUL" }

        $isaFlags = ""
        if ($srcPath -match '\\avx512\\') { $isaFlags = $using:Avx512Flags }
        elseif ($srcPath -match '\\avx2\\') { $isaFlags = $using:Avx2Flags }
        else { return @{ Success = $false; Path = $srcPath; Error = "Cannot determine ISA from path" } }

        $argString = "-O3 $isaFlags $($using:ExtraCFlags) -c `"$srcPath`" -o `"$outPath`""
        $errFile = "$srcPath.err"
        $proc = Start-Process -FilePath $using:CC -ArgumentList $argString `
            -Wait -PassThru -NoNewWindow -RedirectStandardError $errFile 2>$null

        if ($proc.ExitCode -ne 0) {
            $errText = if (Test-Path $errFile) { Get-Content $errFile -Raw } else { "(no stderr)" }
            Remove-Item $errFile -ErrorAction SilentlyContinue
            return @{ Success = $false; Path = $srcPath; Error = $errText }
        }
        Remove-Item $errFile -ErrorAction SilentlyContinue
        return @{ Success = $true; Path = $srcPath }
    } -ThrottleLimit $Jobs
} else {
    # Sequential fallback for PowerShell 5.1
    Write-Host "(Note: PowerShell $PSVersion detected; using sequential compilation." -ForegroundColor Yellow
    Write-Host " For parallel builds, install PowerShell 7+: https://aka.ms/powershell)" -ForegroundColor Yellow
    Write-Host ""
    $Sources | ForEach-Object { Compile-One -Src $_ }
}

$Elapsed = [int]((Get-Date) - $TimeStart).TotalSeconds

# ──────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────
$Successes = @($Results | Where-Object { $_.Success })
$Failures = @($Results | Where-Object { -not $_.Success })

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════════"
Write-Host "  Compilation complete in ${Elapsed}s"
Write-Host "  Compiled: $($Successes.Count) / $Total   Failed: $($Failures.Count)"
Write-Host "═══════════════════════════════════════════════════════════════════"

if ($Failures.Count -gt 0) {
    Write-Host ""
    Write-Host "Failures:" -ForegroundColor Red
    foreach ($f in $Failures) {
        Write-Host "  FAIL: $($f.Path)" -ForegroundColor Red
        if ($f.Error) { Write-Host "    $($f.Error -split "`n" | Select-Object -First 5)" -ForegroundColor DarkRed }
    }
}

# Stats per family
if (-not $VerifyOnly -and $Successes.Count -gt 0) {
    Write-Host ""
    Write-Host "  Object file sizes by family:"
    Get-ChildItem -Path $CodeletsDir -Directory | ForEach-Object {
        $isaName = $_.Name
        Get-ChildItem -Path $_.FullName -Directory | ForEach-Object {
            $family = $_.Name
            $objs = Get-ChildItem -Path $_.FullName -Filter "*.o"
            if ($objs.Count -gt 0) {
                $sizeKB = [math]::Round(($objs | Measure-Object -Property Length -Sum).Sum / 1024, 0)
                $isaFamily = "$isaName/$family"
                Write-Host ("    {0,-22}  {1,3} objects   {2} KB" -f $isaFamily, $objs.Count, $sizeKB)
            }
        }
    }
}

if ($Failures.Count -gt 0) { exit 1 }
exit 0
