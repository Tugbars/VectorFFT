#!/bin/bash
# generate_codelets.sh — Generate all production codelets using the
# best-known configurations from docs 09-42.
#
# ──────────────────────────────────────────────────────────────────────
# CSE / Algsimp passes are GATED by algorithm class
# ──────────────────────────────────────────────────────────────────────
# The generator's optimization pipeline applies DIFFERENT passes
# depending on whether the codelet is Direct (primes) or Cooley-Tukey
# (composites, pow2). This is intentional — passes that help one class
# actively hurt the other. The dispatch happens in bin/gen_radix.ml
# via the `aggressive` flag, which is set to true ONLY when
# pick_algorithm returns Direct (n=2 or odd prime ≥3).
#
# PRIMES-ONLY passes (aggressive=true):
#   * factor_common_muls       — Winograd-style: factor out constants
#                                shared across output bins. For composites
#                                this either does nothing or breaks the
#                                FMA-friendly butterfly structure.
#   * factor_by_atom           — factors expressions by shared "atom"
#                                subexpressions. Same rationale.
#   * fma_lift                 — explicit `Add(Mul,c) → NK_Fma` lift.
#                                ~1-2% win on prime DAGs (Add(Mul,c)
#                                patterns benefit from explicit FMA).
#                                For composites this REGRESSES (doc 28):
#                                R=32 t1_dit llvm-mca SKX 312 → 226
#                                cycles when fma_lift is DISABLED.
#                                Explicit FMA atoms constrain GCC's
#                                register allocator more than letting
#                                it auto-fuse mul+add. See doc 28.
#   * conjugate-pair construction (in dft.ml) — pair sums/diffs and
#                                shared p_re/q_im/q_re/p_im intermediates.
#                                Only applies to direct-DFT primes ≥3.
#                                Doc 23 brought R=11 from 300→190 ops.
#
# POW2/COMPOSITE-ONLY passes (aggressive=false):
#   * share_subsums           — factors common partial-sums across
#                                output bins. Helps pow2 (lots of
#                                cross-output butterfly overlaps) but
#                                HURTS direct primes (doc 23): splits
#                                the unified mixed-sign FMA chain into
#                                separate positive/negative sub-chains,
#                                costing 4 extra ops per pair output.
#                                Skipped via `is_direct` check.
#   * transposition fixed-point loop — Frigo's network transposition
#                                pass, iterated up to 6× with factor/share
#                                between each pass. Skipped for primes
#                                because share_subsums is the inner step
#                                and we don't want share active there.
#                                ALSO skipped for twiddled codelets
#                                (t1_dit, t1_dif): Cmul nodes wrap
#                                symbolic twiddle loads, making the
#                                network non-linear in our representation.
#
# UNIVERSAL passes (apply to both):
#   * dedup_sub_pairs         — canonicalize Sub(a,b) vs Sub(b,a) and
#                                merge duplicates. Pure CSE.
#   * Sub(Neg(Mul(a,b)),c) → fnmsub peephole — at construction time
#                                via mk_sub_binary (doc 30). Universally
#                                beneficial; emits vfnmsub which GCC
#                                won't auto-derive.
#   * single-use inlining (doc 24) — values with one consumer get
#                                inlined into the consumer's expression
#                                rather than declared as separate
#                                `const __m512d t<N> = …`. Closes the
#                                nested-intrinsic gap to hand-written
#                                codelets. Applies in the SU emit path
#                                regardless of algorithm class.
#   * spill recipe + SU scheduler — auto-applied for any twiddled CT
#                                codelet meeting should_spill (n≥5).
#                                On AVX2 with R≥32, GH (Goodman-Hsu
#                                pressure-aware mode) also auto-fires.
#                                Doc 13: cost-model rule "if CT-decomposed
#                                AND (n+6 > vec_regs OR vec_regs >= 32)
#                                use full recipe".
#
# What this means for the script below:
#   The same `gen_radix.exe N --twiddled ...` invocation handles both
#   classes — the generator's internal dispatch picks the right pipeline
#   based on pick_algorithm(N). You don't need different flags per
#   family for this. The family separation in this script reflects
#   per-family wisdom from docs 33-42 about optimal CT factorizations,
#   not different optimization-pass gating. log3 now applies to every
#   family (primes, composites, pow2) — TP_Log3 is a Cmul-derivation
#   pass on EXTERNAL twiddles, orthogonal to the Direct/CT kernel split.
#
# Reference: bin/gen_radix.ml around line 158 (the `aggressive` flag)
# and the doc 28 / 29 / 30 writeups for the empirical motivation.
#
# ──────────────────────────────────────────────────────────────────────
# Per-radix wisdom encoded here (each section cites the supporting doc):
#
#   PRIMES (R ∈ {2,3,5,7,11,13,17,19})
#     - Recipe auto-fires for R ≥ 5 (doc 29)
#     - Conjugate-pair construction for odd primes ≥ 3 (doc 23)
#     - fma_lift gated to primes only (doc 28)
#     - Variants: t1/t1s × DIT/DIF × Fwd/Bwd × Flat/Log3 = 16 per radix
#       (log3 added 2026-05-16; prod wisdom uses primes-log3 for inner
#       stages of large plans, e.g. R=13 ×6 at N=2197.)
#
#   POW2 (R ∈ {4,8,16,32,64,128,256,512})
#     - Recipe + SU auto-fires (doc 13)
#     - Plus AVX2 GH at R≥32 (doc 21)
#     - Monolithic competitive up to R=512 (doc 33,34)
#     - R=512 log3 crossover at B≈128 (doc 42) — generate both flat & log3
#     - Variants: t1/t1s × DIT/DIF × Fwd/Bwd × Flat/Log3 = 16 per radix
#
#   R=1024
#     - Monolithic loses to multi-stage cascade (doc 41) — generate anyway
#       for research/repro, planner should route to cascade
#
#   SMALL NON-PRIME COMPOSITES (R ∈ {6,10,12,20,25})
#     - Generated for completeness; recipe applies
#     - log3 ON (2026-05-16): R=25 is the largest single log3 user in
#       production wisdom (15 selections). R=10/12/20 also picked.
#
#   COMPILER (doc 38): gcc-11 + -flive-range-shrinkage on AVX-512
#     gcc-13 is fine for AVX2 (the flag's AVX2 effect varies by R)
#
# Output: $OUTDIR/codelets/<isa>/<family>/<name>.c — organized by family
# for easy planner consumption.
#
# Usage:
#   ./generate_codelets.sh              # default: AVX-512 only, all families
#   ISA=avx2 ./generate_codelets.sh     # AVX2 build
#   ISA=both ./generate_codelets.sh     # both ISAs
#   ./generate_codelets.sh primes       # only primes family
#   ./generate_codelets.sh pow2 large_pow2  # specific families
#
# Generated codelets are emitted as .c files only — NOT compiled.
# Compile separately with the production gcc-11 + -flive-range-shrinkage
# config (see compile_codelets.sh for a batch compiler script).

set -e

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GEN="${GEN:-$ROOT/_build/default/bin/gen_radix.exe}"
OUTDIR="${OUTDIR:-$ROOT/codelets}"
ISA="${ISA:-avx512}"           # avx512 | avx2 | both

# Family selection: if any args provided, generate only those families.
# Available: primes, small_pow2, mid_pow2, large_pow2, xl_pow2, composites,
#            trig, strided
#
# large_pow2 (128/256/512) and xl_pow2 (1024) are NOT in the default set:
# they are experimental assembly-analysis sizes the planner must never
# select (a single large-radix codelet loses to multi-stage composition).
# They remain reachable on explicit request, e.g. `./generate_codelets.sh
# large_pow2`, but never generate by default.
FAMILIES_ALL="primes small_pow2 mid_pow2 composites trig strided"
if [ $# -eq 0 ]; then
  FAMILIES="$FAMILIES_ALL"
else
  FAMILIES="$@"
fi

# Radix sets per family (based on what's supported by the picker)
PRIMES="2 3 5 7 11 13 17 19"
SMALL_POW2="4 8"
MID_POW2="16 32 64"
LARGE_POW2="128 256 512"   # opt-in only; not in FAMILIES_ALL
XL_POW2="1024"             # opt-in only; not in FAMILIES_ALL
COMPOSITES="6 10 12 20 25"

# trig: discrete trig transforms (DCT-II/III/IV, DST-II/III, DHT).
# Validated cells from docs 55 + 56. N=8 ties or beats production at the
# specialized hand-codelet cells; N=16/32/64 fills general-N gap where
# production has no dedicated codelet.
TRIG_SIZES="8 16 32 64"

# strided: Design C 2D row FFT codelets (matrix→registers→matrix, no
# scratch). AVX2 only. Trimmed to 16/32/64 to match the c2c trim: the
# 128/256 strided sizes are dropped with the rest of the large radixes.
STRIDED_SIZES="16 32 64"

# ──────────────────────────────────────────────────────────────────────
# Sanity checks
# ──────────────────────────────────────────────────────────────────────
if [ ! -x "$GEN" ]; then
  echo "ERROR: generator not built at $GEN"
  echo "Run 'dune build' from the project root first."
  exit 1
fi

if [ "$ISA" = "both" ]; then
  ISAS="avx512 avx2"
else
  ISAS="$ISA"
fi

mkdir -p "$OUTDIR"

# ──────────────────────────────────────────────────────────────────────
# Generation helpers
# ──────────────────────────────────────────────────────────────────────
# emit_codelet R isa family variant_flags suffix
# Generates a single codelet file: <OUTDIR>/codelets/<isa>/<family>/r<R>_<suffix>.c
emit_codelet() {
  local R=$1
  local isa=$2
  local family=$3
  local flags=$4
  local suffix=$5

  local dir="$OUTDIR/$isa/$family"
  mkdir -p "$dir"
  local out="$dir/r${R}_${suffix}.c"

  # No per-radix --fuse here. The M-project fuse lever collapses the
  # source PASS-2 scratch arrays of the 2-pass codelets (R25/32/64) and is
  # bit-identical, but it was measured performance-neutral on the in-place
  # path: gcc reallocates regardless of how many scratch arrays the source
  # names, so real stack spills and cycles are unchanged (see the
  # store-on-compute / fuse in-place A/B). If a future compiler or target
  # makes it pay, the right home is auto-fuse-on-block in the generator,
  # which covers all blocking radixes uniformly, not a per-radix flag here.

  # Always include --in-place (default codelet calling convention) and
  # --emit-c. The variant flags carry direction/sign/policy/etc.
  if $GEN $R --twiddled --in-place --isa $isa $flags --emit-c > "$out" 2>/dev/null; then
    return 0
  else
    echo "  FAIL: R=$R isa=$isa flags='$flags'"
    rm -f "$out"
    return 1
  fi
}

# Wraps the 8-variant (or 16-variant) sweep per R.
# Args: R isa family generate_log3=(yes|no)
emit_variants() {
  local R=$1
  local isa=$2
  local family=$3
  local with_log3=$4

  # Base 8 variants: t1/t1s × dit/dif × fwd/bwd
  emit_codelet $R $isa $family ""                              "t1_dit_fwd"
  emit_codelet $R $isa $family "--bwd"                         "t1_dit_bwd"
  emit_codelet $R $isa $family "--dif"                         "t1_dif_fwd"
  emit_codelet $R $isa $family "--dif --bwd"                   "t1_dif_bwd"
  emit_codelet $R $isa $family "--t1s"                         "t1s_dit_fwd"
  emit_codelet $R $isa $family "--t1s --bwd"                   "t1s_dit_bwd"
  emit_codelet $R $isa $family "--t1s --dif"                   "t1s_dif_fwd"
  emit_codelet $R $isa $family "--t1s --dif --bwd"             "t1s_dif_bwd"

  # log3 variants: TP_Log3 derives external twiddles by binary
  # decomposition. Applies at every family (primes, composites, pow2);
  # R=2 trivially equals flat, so callers may skip it.
  if [ "$with_log3" = "yes" ]; then
    emit_codelet $R $isa $family "--log3"                      "t1_dit_fwd_log3"
    emit_codelet $R $isa $family "--log3 --bwd"                "t1_dit_bwd_log3"
    emit_codelet $R $isa $family "--log3 --dif"                "t1_dif_fwd_log3"
    emit_codelet $R $isa $family "--log3 --dif --bwd"          "t1_dif_bwd_log3"
    emit_codelet $R $isa $family "--log3 --t1s"                "t1s_dit_fwd_log3"
    emit_codelet $R $isa $family "--log3 --t1s --bwd"          "t1s_dit_bwd_log3"
    emit_codelet $R $isa $family "--log3 --t1s --dif"          "t1s_dif_fwd_log3"
    emit_codelet $R $isa $family "--log3 --t1s --dif --bwd"    "t1s_dif_bwd_log3"
  fi
}

# Emit a no-twiddle (n1) codelet — used as the FFT's first stage on the DIT
# forward path (stride S=1, no twiddle table read) and the last stage on the
# DIF backward path. Because there's no twiddle table, there's no t1s
# rendering choice, no DIT-vs-DIF order to flip, and no log3 derivation:
# the variant matrix is just (fwd, bwd). Function name is
# `radix{N}_n1_{fwd|bwd}_{isa}_gen_inplace_su_spill`.
# Args: R isa family extra_flags suffix
emit_n1_codelet() {
  local R=$1
  local isa=$2
  local family=$3
  local flags=$4
  local suffix=$5

  local dir="$OUTDIR/$isa/$family"
  mkdir -p "$dir"
  local out="$dir/r${R}_${suffix}.c"

  # No per-radix --fuse here (see emit_codelet): measured performance-neutral
  # on the in-place path, so it is not applied.

  # Note: NO --twiddled flag — gen_radix's default is the n1 form.
  # --su explicitly engages the Sethi-Ullman scheduler (auto-rule is
  # twiddled-gated, so we have to ask for it by name on the n1 path).
  # --spill is twiddled-only at the math layer; passing it here is a no-op
  # and the resulting symbol name is `..._gen_inplace_su` (no _spill).
  if $GEN $R --in-place --isa $isa --su $flags --emit-c > "$out" 2>/dev/null; then
    return 0
  else
    echo "  FAIL: R=$R isa=$isa n1 flags='$flags'"
    rm -f "$out"
    return 1
  fi
}

# Emit a single trig-transform codelet. The trig codelets have their own
# emit_codelet name convention: radix{N}_{transform}_{isa}_gen. No t1/t1s
# /dit/dif variants — each trig transform has its own algorithm. Forward
# direction only (inverse transforms have their own dedicated entry, e.g.
# DCT-II inverse is DCT-III, DST-II inverse is DST-III).
# Args: R isa family transform_flag suffix
emit_trig() {
  local R=$1
  local isa=$2
  local family=$3
  local flag=$4
  local suffix=$5

  local dir="$OUTDIR/$isa/$family"
  mkdir -p "$dir"
  local out="$dir/r${R}_${suffix}.c"

  if $GEN $R $flag --isa $isa --emit-c > "$out" 2>/dev/null; then
    return 0
  else
    echo "  FAIL: R=$R isa=$isa flag='$flag'"
    rm -f "$out"
    return 1
  fi
}

# Emit a strided-batch codelet (Design C for 2D row FFT). Out-of-place
# at the function level (reads from matrix, writes to matrix), n1 (no
# inter-stage twiddles) since v1 only supports single-stage strided.
# Args: R isa family extra_flags suffix
emit_strided() {
  local R=$1
  local isa=$2
  local family=$3
  local flags=$4
  local suffix=$5

  local dir="$OUTDIR/$isa/$family"
  mkdir -p "$dir"
  local out="$dir/r${R}_${suffix}.c"

  if $GEN $R --strided --isa $isa $flags --emit-c > "$out" 2>/dev/null; then
    return 0
  else
    echo "  FAIL: R=$R isa=$isa strided flags='$flags'"
    rm -f "$out"
    return 1
  fi
}

# ──────────────────────────────────────────────────────────────────────
# Main generation loop
# ──────────────────────────────────────────────────────────────────────
TOTAL_OK=0
TOTAL_FAIL=0
TIME_START=$(date +%s)

echo "═══════════════════════════════════════════════════════════════════"
echo "  vfft_v2 codelet generation"
echo "  Generator: $GEN"
echo "  ISAs:      $ISAS"
echo "  Families:  $FAMILIES"
echo "  Output:    $OUTDIR"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

for isa in $ISAS; do
  echo "▶ ISA: $isa"

  for family in $FAMILIES; do
    case $family in
      primes)
        # Primes: 8 t1/t1s variants + 8 log3 variants + 2 n1 variants per R.
        # log3 IS meaningful for primes: TP_Log3 applies at the external
        # twiddle layer (Cmul wrappers around the monolithic DFT), not at
        # the kernel level — so derived twiddles save bandwidth on the
        # innermost stage even when the kernel is a Winograd direct DFT.
        # Production wisdom selects log3 for R∈{3,5,7,11,13,17,19} on
        # innermost stages of large plans (see vfft_wisdom_tuned.txt).
        echo "  └─ family: primes ($PRIMES)"
        for R in $PRIMES; do
          # n1 first-stage codelets (DIT fwd / DIF bwd entry points).
          emit_n1_codelet $R $isa $family ""      "n1_fwd" && TOTAL_OK=$((TOTAL_OK+1)) || TOTAL_FAIL=$((TOTAL_FAIL+1))
          emit_n1_codelet $R $isa $family "--bwd" "n1_bwd" && TOTAL_OK=$((TOTAL_OK+1)) || TOTAL_FAIL=$((TOTAL_FAIL+1))

          emit_variants $R $isa $family yes
          TOTAL_OK=$((TOTAL_OK + 16))
        done
        ;;

      small_pow2)
        # R=4, R=8: log3 supported but marginal at these sizes
        echo "  └─ family: small_pow2 ($SMALL_POW2)"
        for R in $SMALL_POW2; do
          emit_n1_codelet $R $isa $family ""      "n1_fwd" && TOTAL_OK=$((TOTAL_OK+1)) || TOTAL_FAIL=$((TOTAL_FAIL+1))
          emit_n1_codelet $R $isa $family "--bwd" "n1_bwd" && TOTAL_OK=$((TOTAL_OK+1)) || TOTAL_FAIL=$((TOTAL_FAIL+1))
          emit_variants $R $isa $family yes
          TOTAL_OK=$((TOTAL_OK + 16))
        done
        ;;

      mid_pow2)
        # R=16, 32, 64: the workhorse codelets. Recipe + SU active.
        # On AVX2 with R ≥ 32, GH auto-fires (doc 21) — no flag needed.
        echo "  └─ family: mid_pow2 ($MID_POW2)"
        for R in $MID_POW2; do
          emit_n1_codelet $R $isa $family ""      "n1_fwd" && TOTAL_OK=$((TOTAL_OK+1)) || TOTAL_FAIL=$((TOTAL_FAIL+1))
          emit_n1_codelet $R $isa $family "--bwd" "n1_bwd" && TOTAL_OK=$((TOTAL_OK+1)) || TOTAL_FAIL=$((TOTAL_FAIL+1))
          emit_variants $R $isa $family yes
          TOTAL_OK=$((TOTAL_OK + 16))
        done
        ;;

      large_pow2)
        # R=128, 256, 512: monolithic codelets dominate up to R=512.
        # log3 crossover at R=512 B≈128 — generate both (doc 42).
        echo "  └─ family: large_pow2 ($LARGE_POW2)"
        for R in $LARGE_POW2; do
          emit_n1_codelet $R $isa $family ""      "n1_fwd" && TOTAL_OK=$((TOTAL_OK+1)) || TOTAL_FAIL=$((TOTAL_FAIL+1))
          emit_n1_codelet $R $isa $family "--bwd" "n1_bwd" && TOTAL_OK=$((TOTAL_OK+1)) || TOTAL_FAIL=$((TOTAL_FAIL+1))
          emit_variants $R $isa $family yes
          TOTAL_OK=$((TOTAL_OK + 16))
        done
        ;;

      xl_pow2)
        # R=1024: monolithic loses to multi-stage cascade (doc 41).
        # Generate anyway for research, but only 2 essential variants —
        # the planner should never pick this in practice. n1 still emitted
        # so the cost model has a measurable first-stage entry.
        echo "  └─ family: xl_pow2 ($XL_POW2) [research-only; planner prefers cascade]"
        for R in $XL_POW2; do
          emit_n1_codelet $R $isa $family ""      "n1_fwd" && TOTAL_OK=$((TOTAL_OK+1)) || TOTAL_FAIL=$((TOTAL_FAIL+1))
          emit_n1_codelet $R $isa $family "--bwd" "n1_bwd" && TOTAL_OK=$((TOTAL_OK+1)) || TOTAL_FAIL=$((TOTAL_FAIL+1))
          # Only fwd dit; this isn't a production path
          emit_codelet $R $isa $family ""                "t1_dit_fwd"
          emit_codelet $R $isa $family "--log3"          "t1_dit_fwd_log3"
          TOTAL_OK=$((TOTAL_OK + 2))
        done
        ;;

      composites)
        # Small non-prime composites used by mixed-radix planners.
        # log3 ON: production wisdom selects R=25 with log3 15 times (more
        # than any other radix) and R=10/12/20 once each as innermost
        # stages. The log3 derivation saves twiddle bandwidth on big-me
        # inner-stage runs, which is exactly where these composites land.
        echo "  └─ family: composites ($COMPOSITES)"
        for R in $COMPOSITES; do
          emit_n1_codelet $R $isa $family ""      "n1_fwd" && TOTAL_OK=$((TOTAL_OK+1)) || TOTAL_FAIL=$((TOTAL_FAIL+1))
          emit_n1_codelet $R $isa $family "--bwd" "n1_bwd" && TOTAL_OK=$((TOTAL_OK+1)) || TOTAL_FAIL=$((TOTAL_FAIL+1))
          emit_variants $R $isa $family yes
          TOTAL_OK=$((TOTAL_OK + 16))
        done
        ;;

      trig)
        # Discrete trig transforms (DCT-II/III/IV, DST-II/III, DHT).
        # Validated 2026-05-12 → 2026-05-13. See doc 55.
        #
        # Coverage:
        #   - N=8:  DCT-II ties production; DCT-III loses at K≤256, ties at
        #           K≥512; DCT-IV/DST-II/DST-III/DHT all WIN vs production
        #           runtime 3-pass.
        #   - N=16/32/64: production has no dedicated codelet at these sizes,
        #           our fused DAG fills the gap. DCT-IV verified at 24/24
        #           cells vs production runtime 3-pass.
        #
        # Forward-direction only — DCT-II/DCT-III are an inverse pair, same
        # for DST-II/DST-III. DCT-IV and DHT are self-inverse up to scaling.
        # Per-transform `--emit-c` invocation produces a single codelet per
        # (transform, N) pair, no t1/t1s/dit/dif variants.
        echo "  └─ family: trig ($TRIG_SIZES — DCT-II/III/IV, DST-II/III, DHT)"
        for R in $TRIG_SIZES; do
          emit_trig $R $isa $family "--dct2" "dct2_fwd" && TOTAL_OK=$((TOTAL_OK+1)) || TOTAL_FAIL=$((TOTAL_FAIL+1))
          emit_trig $R $isa $family "--dct3" "dct3_fwd" && TOTAL_OK=$((TOTAL_OK+1)) || TOTAL_FAIL=$((TOTAL_FAIL+1))
          emit_trig $R $isa $family "--dct4" "dct4_fwd" && TOTAL_OK=$((TOTAL_OK+1)) || TOTAL_FAIL=$((TOTAL_FAIL+1))
          emit_trig $R $isa $family "--dst2" "dst2_fwd" && TOTAL_OK=$((TOTAL_OK+1)) || TOTAL_FAIL=$((TOTAL_FAIL+1))
          emit_trig $R $isa $family "--dst3" "dst3_fwd" && TOTAL_OK=$((TOTAL_OK+1)) || TOTAL_FAIL=$((TOTAL_FAIL+1))
          emit_trig $R $isa $family "--dht"  "dht_fwd"  && TOTAL_OK=$((TOTAL_OK+1)) || TOTAL_FAIL=$((TOTAL_FAIL+1))
        done
        ;;

      strided)
        # Design C strided-batch 2D row FFT codelets. Validated 2026-05-13.
        # See doc 56. AVX2 only — AVX-512 8×8 transpose preamble deferred.
        #
        # Microbench: 40/40 directional cells WIN vs (gather + standard OOP
        # codelet + scatter) reference. Speedups 1.15×–3.67×, growing with B.
        # Roundtrip identity 20/20 PASS at FP noise.
        #
        # AVX-512 strided supported via 8×8 in-register transpose preamble
        # and postamble in emit_c.ml (validated 2026-05-13 on real
        # AVX-512 hardware — 60/60 PASS, see doc 56).
        echo "  └─ family: strided ($STRIDED_SIZES — fwd+bwd, isa=$isa)"
        for R in $STRIDED_SIZES; do
          emit_strided $R $isa $family ""        "n1_fwd_strided" && TOTAL_OK=$((TOTAL_OK+1)) || TOTAL_FAIL=$((TOTAL_FAIL+1))
          emit_strided $R $isa $family "--bwd"   "n1_bwd_strided" && TOTAL_OK=$((TOTAL_OK+1)) || TOTAL_FAIL=$((TOTAL_FAIL+1))
        done
        ;;

      *)
        echo "  WARNING: unknown family '$family' — skipping"
        ;;
    esac
  done
done

TIME_END=$(date +%s)
ELAPSED=$((TIME_END - TIME_START))

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Generation complete in ${ELAPSED}s"
echo "  Codelets emitted: $TOTAL_OK   Failures: $TOTAL_FAIL"
echo "  Output tree: $OUTDIR"
echo "═══════════════════════════════════════════════════════════════════"

# Summary by family
echo ""
echo "  Files by family:"
for isa in $ISAS; do
  for family in $FAMILIES; do
    dir="$OUTDIR/$isa/$family"
    if [ -d "$dir" ]; then
      count=$(ls "$dir"/*.c 2>/dev/null | wc -l)
      total_lines=$(cat "$dir"/*.c 2>/dev/null | wc -l)
      printf "    %s/%-12s  %3d codelets   %d total lines\n" "$isa" "$family" "$count" "$total_lines"
    fi
  done
done

echo ""
echo "Next: run ./scripts/compile_codelets.sh to build .o files with the"
echo "production compiler config (gcc-11 + -flive-range-shrinkage on AVX-512)."
