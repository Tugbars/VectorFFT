<#
.SYNOPSIS
    Generate all production codelets using the best-known configurations
    from docs 09-42.

.DESCRIPTION
    Windows PowerShell port of scripts/generate_codelets.sh. Drives
    gen_radix.exe over all (R, ISA, variant) combinations, emitting
    .c files into a structured directory tree.

    Requires PowerShell 5.1 or later (PowerShell 7+ recommended for
    better parallelism support if needed).

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    CSE / Algsimp passes are GATED by algorithm class
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    The generator's optimization pipeline applies DIFFERENT passes
    depending on whether the codelet is Direct (primes) or Cooley-Tukey
    (composites, pow2). This is intentional — passes that help one
    class actively hurt the other. The dispatch happens in
    bin/gen_radix.ml via the `aggressive` flag, which is set to true
    ONLY when pick_algorithm returns Direct (n=2 or odd prime ≥3).

    PRIMES-ONLY passes (aggressive=true):
      - factor_common_muls    — Winograd-style constant factoring.
                                For composites: no-op or breaks
                                FMA-friendly butterfly structure.
      - factor_by_atom        — Factor by shared atom subexpressions.
                                Same rationale.
      - fma_lift              — Explicit Add(Mul,c) → NK_Fma lift.
                                ~1-2% prime win. REGRESSES composites
                                (doc 28): R=32 t1_dit llvm-mca SKX
                                312→226 cycles when DISABLED. Explicit
                                FMA atoms constrain GCC's register
                                allocator more than letting it auto-fuse.
      - Conjugate-pair DFT    — Pair sums/diffs + shared p_re/q_im/etc.
                                Doc 23: R=11 went from 300→190 ops.

    POW2/COMPOSITE-ONLY passes (aggressive=false):
      - share_subsums         — Factor common partial-sums across output
                                bins. HURTS direct primes (doc 23):
                                splits unified mixed-sign FMA chain
                                into separate ± sub-chains, costing 4
                                extra ops per pair output.
      - Transposition loop    — Frigo's network transposition, iterated
                                up to 6× with factor/share between
                                passes. Inner step uses share_subsums,
                                so it's skipped for primes. Also skipped
                                for twiddled codelets (t1_dit, t1_dif)
                                where Cmul nodes wrap symbolic twiddle
                                loads, making the network non-linear.

    UNIVERSAL passes (both classes):
      - dedup_sub_pairs       — Canonicalize Sub(a,b) vs Sub(b,a).
      - Sub(Neg(Mul),c)→fnmsub peephole at construction (doc 30).
      - Single-use inlining (doc 24) in the SU emit path.
      - Spill recipe + SU scheduler — auto-applies for any twiddled
        CT codelet meeting should_spill (n≥5 in practice). On AVX2
        with R≥32, GH (Goodman-Hsu) also auto-fires. Doc 13.

    The same gen_radix.exe N --twiddled ... command works for all
    radixes; the generator's internal dispatch picks the right pipeline
    based on pick_algorithm(N). Family separation in this script
    reflects different VARIANT MATRICES (log3 only on pow2, etc.) and
    per-family wisdom from docs 33-42, NOT optimization-pass gating.

    Reference: bin/gen_radix.ml around line 158 (the `aggressive` flag)
    and the doc 23 / 28 / 30 writeups for empirical motivation.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Per-radix wisdom (each entry cites the supporting doc):

      PRIMES (R ∈ {2,3,5,7,11,13,17,19})
        - Recipe auto-fires for R ≥ 5 (doc 29)
        - Conjugate-pair construction for odd primes ≥ 3 (doc 23)
        - fma_lift gated to primes only (doc 28)
        - 8 variants per R

      POW2 (R ∈ {4,8,16,32,64,128,256,512})
        - Recipe + SU auto-fires (doc 13)
        - AVX2 GH at R≥32 (doc 21)
        - Monolithic competitive up to R=512 (doc 33,34)
        - R=512 log3 crossover at B≈128 (doc 42) — generate both
        - 16 variants per R

      R=1024
        - Monolithic loses to multi-stage cascade (doc 41)
        - Generated for research/repro only; planner should cascade
        - 2 variants

      SMALL NON-PRIME COMPOSITES (R ∈ {6,10,12,20,25})
        - Recipe applies, 8 variants

      COMPILER (doc 38): gcc-11 + -flive-range-shrinkage on AVX-512.
        gcc-13 fine for AVX2 (flag's AVX2 effect varies by R).

.PARAMETER Families
    Optional list of families to generate. If omitted, all are generated.
    Available: primes, small_pow2, mid_pow2, large_pow2, xl_pow2, composites

.EXAMPLE
    .\generate_codelets.ps1
    Generate all families for AVX-512 (default).

.EXAMPLE
    $env:ISA = "both"; .\generate_codelets.ps1
    Generate all families for both ISAs.

.EXAMPLE
    .\generate_codelets.ps1 primes mid_pow2
    Generate only primes and mid_pow2 families.

.NOTES
    Environment variables:
      ISA     — "avx512" (default), "avx2", or "both"
      OUTDIR  — output directory (default: <project_root>\codelets)
      GEN     — override generator binary location
#>

param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Families
)

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
$ErrorActionPreference = "Stop"

$Root = (Resolve-Path "$PSScriptRoot\..").Path
$Gen  = if ($env:GEN) { $env:GEN } else { Join-Path $Root "_build\default\bin\gen_radix.exe" }
$OutDir = if ($env:OUTDIR) { $env:OUTDIR } else { Join-Path $Root "codelets" }
$Isa = if ($env:ISA) { $env:ISA } else { "avx512" }

$FamiliesAll = @("primes", "small_pow2", "mid_pow2", "large_pow2", "xl_pow2", "composites", "trig", "strided")
if (-not $Families -or $Families.Count -eq 0) {
    $Families = $FamiliesAll
}

# Per-family radix sets
$RadixSets = @{
    "primes"       = @(2, 3, 5, 7, 11, 13, 17, 19)
    "small_pow2"   = @(4, 8)
    "mid_pow2"     = @(16, 32, 64)
    "large_pow2"   = @(128, 256, 512)
    "xl_pow2"      = @(1024)
    "composites"   = @(6, 10, 12, 20, 25)
    # trig: validated cells from docs 55 + 56. See bash version for context.
    "trig"         = @(8, 16, 32, 64)
    # strided: Design C 2D row FFT codelets, AVX2 only. R=512/1024 out of
    # scope per feedback-strided-r512-overkill.
    "strided"      = @(16, 32, 64, 128, 256)
}

# ──────────────────────────────────────────────────────────────────────
# Sanity checks
# ──────────────────────────────────────────────────────────────────────
if (-not (Test-Path $Gen)) {
    Write-Host "ERROR: generator not built at $Gen" -ForegroundColor Red
    Write-Host "Run 'dune build' from the project root first."
    exit 1
}

$Isas = if ($Isa -eq "both") { @("avx512", "avx2") } else { @($Isa) }

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# ──────────────────────────────────────────────────────────────────────
# Generation helpers
# ──────────────────────────────────────────────────────────────────────
function Invoke-Codelet {
    param(
        [int]$R, [string]$IsaName, [string]$Family,
        [string[]]$Flags, [string]$Suffix
    )

    $dir = Join-Path $OutDir (Join-Path $IsaName $Family)
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $out = Join-Path $dir "r${R}_${Suffix}.c"

    # Build the argument list. Always includes --twiddled --in-place --emit-c.
    $args = @("$R", "--twiddled", "--in-place", "--isa", $IsaName, "--emit-c") + $Flags

    try {
        $output = & $Gen @args 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  FAIL: R=$R isa=$IsaName flags='$($Flags -join ' ')'" -ForegroundColor Red
            return $false
        }
        Set-Content -Path $out -Value $output -Encoding utf8
        return $true
    } catch {
        Write-Host "  FAIL: R=$R isa=$IsaName flags='$($Flags -join ' ')' (exception: $_)" -ForegroundColor Red
        return $false
    }
}

function Invoke-Variants {
    param([int]$R, [string]$IsaName, [string]$Family, [bool]$WithLog3)

    $ok = 0
    $variants = @(
        @{ Flags = @();                                      Suffix = "t1_dit_fwd"  },
        @{ Flags = @("--bwd");                               Suffix = "t1_dit_bwd"  },
        @{ Flags = @("--dif");                               Suffix = "t1_dif_fwd"  },
        @{ Flags = @("--dif", "--bwd");                      Suffix = "t1_dif_bwd"  },
        @{ Flags = @("--t1s");                               Suffix = "t1s_dit_fwd" },
        @{ Flags = @("--t1s", "--bwd");                      Suffix = "t1s_dit_bwd" },
        @{ Flags = @("--t1s", "--dif");                      Suffix = "t1s_dif_fwd" },
        @{ Flags = @("--t1s", "--dif", "--bwd");             Suffix = "t1s_dif_bwd" }
    )

    if ($WithLog3) {
        $variants += @(
            @{ Flags = @("--log3");                                Suffix = "t1_dit_fwd_log3"  },
            @{ Flags = @("--log3", "--bwd");                       Suffix = "t1_dit_bwd_log3"  },
            @{ Flags = @("--log3", "--dif");                       Suffix = "t1_dif_fwd_log3"  },
            @{ Flags = @("--log3", "--dif", "--bwd");              Suffix = "t1_dif_bwd_log3"  },
            @{ Flags = @("--log3", "--t1s");                       Suffix = "t1s_dit_fwd_log3" },
            @{ Flags = @("--log3", "--t1s", "--bwd");              Suffix = "t1s_dit_bwd_log3" },
            @{ Flags = @("--log3", "--t1s", "--dif");              Suffix = "t1s_dif_fwd_log3" },
            @{ Flags = @("--log3", "--t1s", "--dif", "--bwd");     Suffix = "t1s_dif_bwd_log3" }
        )
    }

    foreach ($v in $variants) {
        if (Invoke-Codelet -R $R -IsaName $IsaName -Family $Family -Flags $v.Flags -Suffix $v.Suffix) {
            $ok++
        }
    }
    return $ok
}

# Emit a no-twiddle (n1) codelet — used as the FFT's first stage on the DIT
# forward path and the last stage on the DIF backward path. Because there's
# no twiddle table, there's no t1s rendering choice, no dit/dif order to
# flip, and no log3 derivation: the variant matrix is just (fwd, bwd).
# Function name is radix{R}_n1_{fwd|bwd}_{isa}_gen_inplace_su_spill.
function Invoke-N1 {
    param(
        [int]$R, [string]$IsaName, [string]$Family,
        [string[]]$ExtraFlags, [string]$Suffix
    )

    $dir = Join-Path $OutDir (Join-Path $IsaName $Family)
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $out = Join-Path $dir "r${R}_${Suffix}.c"

    # No --twiddled flag — gen_radix's default is the n1 form.
    # --su explicitly engages the Sethi-Ullman scheduler (auto-rule is
    # twiddled-gated, so we ask for it by name on the n1 path).
    # --spill is twiddled-only at the math layer; passing it would be a no-op,
    # and the resulting symbol name is `..._gen_inplace_su` (no _spill).
    $args = @("$R", "--in-place", "--isa", $IsaName, "--su", "--emit-c") + $ExtraFlags
    try {
        $output = & $Gen @args 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  FAIL: R=$R isa=$IsaName n1 flags='$($ExtraFlags -join ' ')'" -ForegroundColor Red
            return $false
        }
        Set-Content -Path $out -Value $output -Encoding utf8
        return $true
    } catch {
        Write-Host "  FAIL: R=$R isa=$IsaName n1 (exception: $_)" -ForegroundColor Red
        return $false
    }
}

# Trig-transform codelet emitter. Each trig transform has its own algorithm
# (DCT-II/III/IV via Makhoul/inverse-Makhoul/Lee, DST-II/III via DCT
# wrappers, DHT via N-rdft+butterfly). No t1/t1s/dit/dif variants.
function Invoke-Trig {
    param(
        [int]$R, [string]$IsaName, [string]$Family,
        [string]$Flag, [string]$Suffix
    )

    $dir = Join-Path $OutDir (Join-Path $IsaName $Family)
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $out = Join-Path $dir "r${R}_${Suffix}.c"

    $args = @("$R", $Flag, "--isa", $IsaName, "--emit-c")
    try {
        $output = & $Gen @args 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  FAIL: R=$R isa=$IsaName trig=$Flag" -ForegroundColor Red
            return $false
        }
        Set-Content -Path $out -Value $output -Encoding utf8
        return $true
    } catch {
        Write-Host "  FAIL: R=$R isa=$IsaName trig=$Flag (exception: $_)" -ForegroundColor Red
        return $false
    }
}

# Strided codelet emitter (Design C 2D row FFT). Composes with --bwd.
function Invoke-Strided {
    param(
        [int]$R, [string]$IsaName, [string]$Family,
        [string[]]$ExtraFlags, [string]$Suffix
    )

    $dir = Join-Path $OutDir (Join-Path $IsaName $Family)
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $out = Join-Path $dir "r${R}_${Suffix}.c"

    $args = @("$R", "--strided", "--isa", $IsaName, "--emit-c") + $ExtraFlags
    try {
        $output = & $Gen @args 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  FAIL: R=$R isa=$IsaName strided flags='$($ExtraFlags -join ' ')'" -ForegroundColor Red
            return $false
        }
        Set-Content -Path $out -Value $output -Encoding utf8
        return $true
    } catch {
        Write-Host "  FAIL: R=$R isa=$IsaName strided (exception: $_)" -ForegroundColor Red
        return $false
    }
}

# ──────────────────────────────────────────────────────────────────────
# Main generation loop
# ──────────────────────────────────────────────────────────────────────
$TotalOK = 0
$TotalFail = 0
$TimeStart = Get-Date

Write-Host "═══════════════════════════════════════════════════════════════════"
Write-Host "  vfft_v2 codelet generation (PowerShell)"
Write-Host "  Generator: $Gen"
Write-Host "  ISAs:      $($Isas -join ' ')"
Write-Host "  Families:  $($Families -join ' ')"
Write-Host "  Output:    $OutDir"
Write-Host "═══════════════════════════════════════════════════════════════════"
Write-Host ""

foreach ($isaName in $Isas) {
    Write-Host "▶ ISA: $isaName"

    foreach ($family in $Families) {
        if (-not $RadixSets.ContainsKey($family)) {
            Write-Host "  WARNING: unknown family '$family' — skipping" -ForegroundColor Yellow
            continue
        }

        $radixes = $RadixSets[$family]
        $radixStr = $radixes -join ' '

        switch ($family) {
            "primes" {
                Write-Host "  └─ family: primes ($radixStr)"
                foreach ($r in $radixes) {
                    if (Invoke-N1 -R $r -IsaName $isaName -Family $family -ExtraFlags @()        -Suffix "n1_fwd") { $TotalOK++ } else { $TotalFail++ }
                    if (Invoke-N1 -R $r -IsaName $isaName -Family $family -ExtraFlags @("--bwd") -Suffix "n1_bwd") { $TotalOK++ } else { $TotalFail++ }
                    foreach ($v in @("t1_dit_fwd","t1_dit_bwd","t1_dif_fwd","t1_dif_bwd",
                                      "t1s_dit_fwd","t1s_dit_bwd","t1s_dif_fwd","t1s_dif_bwd")) {
                        $flags = @()
                        if ($v -like "*t1s*") { $flags += "--t1s" }
                        if ($v -like "*dif*") { $flags += "--dif" }
                        if ($v -like "*bwd*") { $flags += "--bwd" }
                        if (Invoke-Codelet -R $r -IsaName $isaName -Family $family -Flags $flags -Suffix $v) {
                            $TotalOK++
                        } else { $TotalFail++ }
                    }
                }
            }

            { $_ -eq "small_pow2" -or $_ -eq "mid_pow2" -or $_ -eq "large_pow2" } {
                Write-Host "  └─ family: $family ($radixStr)"
                foreach ($r in $radixes) {
                    if (Invoke-N1 -R $r -IsaName $isaName -Family $family -ExtraFlags @()        -Suffix "n1_fwd") { $TotalOK++ } else { $TotalFail++ }
                    if (Invoke-N1 -R $r -IsaName $isaName -Family $family -ExtraFlags @("--bwd") -Suffix "n1_bwd") { $TotalOK++ } else { $TotalFail++ }
                    $TotalOK += (Invoke-Variants -R $r -IsaName $isaName -Family $family -WithLog3 $true)
                }
            }

            "xl_pow2" {
                Write-Host "  └─ family: xl_pow2 ($radixStr) [research-only; planner prefers cascade]"
                foreach ($r in $radixes) {
                    if (Invoke-N1 -R $r -IsaName $isaName -Family $family -ExtraFlags @()        -Suffix "n1_fwd") { $TotalOK++ } else { $TotalFail++ }
                    if (Invoke-N1 -R $r -IsaName $isaName -Family $family -ExtraFlags @("--bwd") -Suffix "n1_bwd") { $TotalOK++ } else { $TotalFail++ }
                    if (Invoke-Codelet -R $r -IsaName $isaName -Family $family -Flags @() -Suffix "t1_dit_fwd") {
                        $TotalOK++
                    } else { $TotalFail++ }
                    if (Invoke-Codelet -R $r -IsaName $isaName -Family $family -Flags @("--log3") -Suffix "t1_dit_fwd_log3") {
                        $TotalOK++
                    } else { $TotalFail++ }
                }
            }

            "composites" {
                Write-Host "  └─ family: composites ($radixStr)"
                foreach ($r in $radixes) {
                    if (Invoke-N1 -R $r -IsaName $isaName -Family $family -ExtraFlags @()        -Suffix "n1_fwd") { $TotalOK++ } else { $TotalFail++ }
                    if (Invoke-N1 -R $r -IsaName $isaName -Family $family -ExtraFlags @("--bwd") -Suffix "n1_bwd") { $TotalOK++ } else { $TotalFail++ }
                    $TotalOK += (Invoke-Variants -R $r -IsaName $isaName -Family $family -WithLog3 $false)
                }
            }

            "trig" {
                # DCT-II/III/IV, DST-II/III, DHT. Validated cells from
                # docs 55 + 56. Forward-direction only (DCT-II/III are an
                # inverse pair; DST-II/III likewise; DCT-IV and DHT are
                # self-inverse up to scaling).
                Write-Host "  └─ family: trig ($radixStr — DCT-II/III/IV, DST-II/III, DHT)"
                $transforms = @(
                    @{ Flag = "--dct2"; Suffix = "dct2_fwd" },
                    @{ Flag = "--dct3"; Suffix = "dct3_fwd" },
                    @{ Flag = "--dct4"; Suffix = "dct4_fwd" },
                    @{ Flag = "--dst2"; Suffix = "dst2_fwd" },
                    @{ Flag = "--dst3"; Suffix = "dst3_fwd" },
                    @{ Flag = "--dht";  Suffix = "dht_fwd"  }
                )
                foreach ($r in $radixes) {
                    foreach ($t in $transforms) {
                        if (Invoke-Trig -R $r -IsaName $isaName -Family $family -Flag $t.Flag -Suffix $t.Suffix) {
                            $TotalOK++
                        } else { $TotalFail++ }
                    }
                }
            }

            "strided" {
                # Design C 2D row FFT codelets. AVX2 and AVX-512 both supported
                # (AVX-512 via 8×8 in-register transpose preamble/postamble,
                # validated 2026-05-13 on real AVX-512 hardware). See doc 56.
                Write-Host "  └─ family: strided ($radixStr — fwd+bwd, isa=$isaName)"
                foreach ($r in $radixes) {
                    if (Invoke-Strided -R $r -IsaName $isaName -Family $family -ExtraFlags @() -Suffix "n1_fwd_strided") {
                        $TotalOK++
                    } else { $TotalFail++ }
                    if (Invoke-Strided -R $r -IsaName $isaName -Family $family -ExtraFlags @("--bwd") -Suffix "n1_bwd_strided") {
                        $TotalOK++
                    } else { $TotalFail++ }
                }
            }
        }
    }
}

$Elapsed = [int]((Get-Date) - $TimeStart).TotalSeconds

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════════"
Write-Host "  Generation complete in ${Elapsed}s"
Write-Host "  Codelets emitted: $TotalOK   Failures: $TotalFail"
Write-Host "  Output tree: $OutDir"
Write-Host "═══════════════════════════════════════════════════════════════════"

# Summary by family
Write-Host ""
Write-Host "  Files by family:"
foreach ($isaName in $Isas) {
    foreach ($family in $Families) {
        $dir = Join-Path $OutDir (Join-Path $isaName $family)
        if (Test-Path $dir) {
            $files = Get-ChildItem -Path $dir -Filter "*.c"
            $count = $files.Count
            $totalLines = 0
            foreach ($f in $files) {
                $totalLines += (Get-Content $f.FullName | Measure-Object -Line).Lines
            }
            $isaFamily = "$isaName/$family"
            Write-Host ("    {0,-22}  {1,3} codelets   {2} total lines" -f $isaFamily, $count, $totalLines)
        }
    }
}

Write-Host ""
Write-Host "Next: run .\scripts\compile_codelets.ps1 to build .o files"
Write-Host "with the production compiler config (gcc-11 + -flive-range-shrinkage)."
