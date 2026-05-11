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
#   different VARIANT MATRICES (e.g., log3 only applies to pow2) and
#   per-family wisdom from docs 33-42 about optimal CT factorizations,
#   not different optimization-pass gating.
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
#     - Variants: t1/t1s × DIT/DIF × Fwd/Bwd = 8 per radix
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
# Available: primes, small_pow2, mid_pow2, large_pow2, xl_pow2, composites
FAMILIES_ALL="primes small_pow2 mid_pow2 large_pow2 xl_pow2 composites"
if [ $# -eq 0 ]; then
  FAMILIES="$FAMILIES_ALL"
else
  FAMILIES="$@"
fi

# Radix sets per family (based on what's supported by the picker)
PRIMES="2 3 5 7 11 13 17 19"
SMALL_POW2="4 8"
MID_POW2="16 32 64"
LARGE_POW2="128 256 512"
XL_POW2="1024"
COMPOSITES="6 10 12 20 25"

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

  # log3 variants: only for pow2 R ≥ 4 where TP_Log3 is meaningful.
  # Log3 derives twiddles by binary decomposition (R=2 → trivial, skip).
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
        # Primes: 8 variants per R, no log3 (twiddle reduction doesn't apply
        # to monolithic prime DFTs in our generator).
        echo "  └─ family: primes ($PRIMES)"
        for R in $PRIMES; do
          for v in t1_dit_fwd t1_dit_bwd t1_dif_fwd t1_dif_bwd \
                   t1s_dit_fwd t1s_dit_bwd t1s_dif_fwd t1s_dif_bwd; do
            # Build the flag string from variant name
            flags=""
            [[ "$v" == *"t1s"* ]] && flags="$flags --t1s"
            [[ "$v" == *"dif"* ]] && flags="$flags --dif"
            [[ "$v" == *"bwd"* ]] && flags="$flags --bwd"
            if emit_codelet $R $isa $family "$flags" "$v"; then
              TOTAL_OK=$((TOTAL_OK+1))
            else
              TOTAL_FAIL=$((TOTAL_FAIL+1))
            fi
          done
        done
        ;;

      small_pow2)
        # R=4, R=8: log3 supported but marginal at these sizes
        echo "  └─ family: small_pow2 ($SMALL_POW2)"
        for R in $SMALL_POW2; do
          emit_variants $R $isa $family yes
          TOTAL_OK=$((TOTAL_OK + 16))
        done
        ;;

      mid_pow2)
        # R=16, 32, 64: the workhorse codelets. Recipe + SU active.
        # On AVX2 with R ≥ 32, GH auto-fires (doc 21) — no flag needed.
        echo "  └─ family: mid_pow2 ($MID_POW2)"
        for R in $MID_POW2; do
          emit_variants $R $isa $family yes
          TOTAL_OK=$((TOTAL_OK + 16))
        done
        ;;

      large_pow2)
        # R=128, 256, 512: monolithic codelets dominate up to R=512.
        # log3 crossover at R=512 B≈128 — generate both (doc 42).
        echo "  └─ family: large_pow2 ($LARGE_POW2)"
        for R in $LARGE_POW2; do
          emit_variants $R $isa $family yes
          TOTAL_OK=$((TOTAL_OK + 16))
        done
        ;;

      xl_pow2)
        # R=1024: monolithic loses to multi-stage cascade (doc 41).
        # Generate anyway for research, but only 2 essential variants —
        # the planner should never pick this in practice.
        echo "  └─ family: xl_pow2 ($XL_POW2) [research-only; planner prefers cascade]"
        for R in $XL_POW2; do
          # Only fwd dit; this isn't a production path
          emit_codelet $R $isa $family ""                "t1_dit_fwd"
          emit_codelet $R $isa $family "--log3"          "t1_dit_fwd_log3"
          TOTAL_OK=$((TOTAL_OK + 2))
        done
        ;;

      composites)
        # Small non-prime composites used by mixed-radix planners.
        echo "  └─ family: composites ($COMPOSITES)"
        for R in $COMPOSITES; do
          emit_variants $R $isa $family no  # log3 supported only for pow2
          TOTAL_OK=$((TOTAL_OK + 8))
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
