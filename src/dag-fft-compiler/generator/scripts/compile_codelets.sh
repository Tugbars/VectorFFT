#!/bin/bash
# compile_codelets.sh — Compile all generated codelets using the
# production compiler config from doc 38: gcc-11 + -flive-range-shrinkage.
#
# Note: this script only handles compilation. The CSE/Algsimp pass
# selection per radix happens upstream in the GENERATOR (gen_radix.ml),
# NOT here. The generator gates several optimization passes by algorithm
# class — fma_lift is primes-only (doc 28), share_subsums and the
# transposition fixed-point loop are pow2/composite-only (doc 23, 28),
# and the spill recipe + SU scheduler are universal for any twiddled
# CT codelet ≥ R=5 (doc 13). See generate_codelets.sh's header comment
# for the full breakdown of which passes apply where.
#
# By the time we get to compilation, every .c file already encodes the
# right algorithm-class-specific optimizations baked into its expression
# tree. All we do here is pick the right compiler config to translate
# them into asm without losing the wins.
#
# Walks the codelets/ tree (output of generate_codelets.sh) and compiles
# each .c file into a .o file alongside it. Per-ISA flags are applied
# based on the directory the codelet lives in.
#
# Compiler choice rationale (doc 38):
#   - gcc-11 + -flive-range-shrinkage: 29% fewer stack ops on AVX-512 R=512
#     vs gcc-13 default; 5-8% runtime gain at moderate B.
#   - gcc-12 has an AVX-512 register allocator regression (-9.4% to -14%
#     vs gcc-11). gcc-13 inherits this.
#   - Clang-18 is significantly worse on AVX-512 (3× more spills at R=512).
#   - -flive-range-shrinkage helps AVX2 at small R, slightly hurts at large R,
#     but the production deployment uses it for all sizes to keep CI simple.
#
# Usage:
#   ./compile_codelets.sh                          # default config
#   CC=gcc-13 ./compile_codelets.sh                # override compiler
#   EXTRA_CFLAGS='' ./compile_codelets.sh          # disable shrinkage
#   CODELETS_DIR=./mydir ./compile_codelets.sh     # custom input tree

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CODELETS_DIR="${CODELETS_DIR:-$ROOT/codelets}"
CC="${CC:-gcc-11}"
EXTRA_CFLAGS="${EXTRA_CFLAGS:--flive-range-shrinkage}"
JOBS="${JOBS:-$(nproc)}"
VERIFY_ONLY="${VERIFY_ONLY:-no}"  # set to 'yes' to check sources compile but not save .o

# Per-ISA compile flags. -march=native lets the chip pick the right
# subset; the -m* flags are explicit so codelets compiled on one machine
# match those compiled on another (no -march=native variance).
AVX512_FLAGS="-mavx512f -mavx512dq -mfma -march=skylake-avx512"
AVX2_FLAGS="-mavx2 -mfma -march=haswell"

if [ ! -d "$CODELETS_DIR" ]; then
  echo "ERROR: codelets directory not found: $CODELETS_DIR"
  echo "Run ./scripts/generate_codelets.sh first."
  exit 1
fi

if ! command -v "$CC" >/dev/null 2>&1; then
  echo "ERROR: compiler '$CC' not found in PATH"
  echo "Install with: sudo apt install gcc-11   (Ubuntu/Debian)"
  echo "Or override with: CC=gcc-13 $0"
  exit 1
fi

echo "═══════════════════════════════════════════════════════════════════"
echo "  vfft_v2 codelet compilation"
echo "  CC:           $CC"
echo "  EXTRA_CFLAGS: $EXTRA_CFLAGS"
echo "  Codelets:     $CODELETS_DIR"
echo "  Parallelism:  $JOBS jobs"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Find all .c files under the codelets tree
mapfile -t SOURCES < <(find "$CODELETS_DIR" -name "*.c" | sort)
TOTAL=${#SOURCES[@]}

if [ "$TOTAL" = "0" ]; then
  echo "No .c files found under $CODELETS_DIR"
  exit 1
fi

echo "Found $TOTAL codelet sources. Compiling..."
echo ""

# Build per-source command list, parallelized via xargs
TIME_START=$(date +%s)
FAIL_LOG=$(mktemp)
trap "rm -f $FAIL_LOG" EXIT

compile_one() {
  local src=$1
  local out="${src%.c}.o"
  # Determine ISA from the path (codelets/<isa>/...)
  local isa_flags=""
  case "$src" in
    */avx512/*) isa_flags="$AVX512_FLAGS" ;;
    */avx2/*)   isa_flags="$AVX2_FLAGS"  ;;
    *)
      echo "  WARNING: can't determine ISA for $src — skipping" >&2
      return 1
      ;;
  esac

  if [ "$VERIFY_ONLY" = "yes" ]; then
    out=/dev/null
  fi

  if ! "$CC" -O3 $isa_flags $EXTRA_CFLAGS -c "$src" -o "$out" 2>"$FAIL_LOG.$$" ; then
    {
      echo "FAIL: $src"
      cat "$FAIL_LOG.$$"
      echo "---"
    } >> "$FAIL_LOG"
    rm -f "$FAIL_LOG.$$"
    return 1
  fi
  rm -f "$FAIL_LOG.$$"
  return 0
}

export -f compile_one
export CC EXTRA_CFLAGS AVX512_FLAGS AVX2_FLAGS VERIFY_ONLY FAIL_LOG

# Run compilations in parallel. Progress: count finished files.
printf '%s\n' "${SOURCES[@]}" | xargs -P "$JOBS" -I {} bash -c 'compile_one "$@"' _ {} || true

TIME_END=$(date +%s)
ELAPSED=$((TIME_END - TIME_START))

# Count successes
OK_COUNT=$(find "$CODELETS_DIR" -name "*.o" | wc -l)
FAIL_COUNT=$((TOTAL - OK_COUNT))

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Compilation complete in ${ELAPSED}s"
echo "  Compiled: $OK_COUNT / $TOTAL   Failed: $FAIL_COUNT"
echo "═══════════════════════════════════════════════════════════════════"

if [ "$FAIL_COUNT" -gt 0 ] && [ -s "$FAIL_LOG" ]; then
  echo ""
  echo "Failures:"
  cat "$FAIL_LOG"
fi

# Quick stats on the resulting object files
if [ "$VERIFY_ONLY" != "yes" ] && [ "$OK_COUNT" -gt 0 ]; then
  echo ""
  echo "  Object file sizes by family:"
  for isa_dir in "$CODELETS_DIR"/*/; do
    isa=$(basename "$isa_dir")
    for family_dir in "$isa_dir"*/; do
      family=$(basename "$family_dir")
      count=$(find "$family_dir" -name "*.o" | wc -l)
      size=$(du -ck "$family_dir"/*.o 2>/dev/null | tail -1 | awk '{print $1}')
      printf "    %s/%-12s  %3d objects   %s KB\n" "$isa" "$family" "$count" "${size:-0}"
    done
  done
fi

[ "$FAIL_COUNT" = "0" ]
