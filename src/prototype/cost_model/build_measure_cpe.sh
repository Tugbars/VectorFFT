#!/bin/bash
# build_measure_cpe.sh — Build the CPE measurement tool.
#
# ═════════════════════════════════════════════════════════════════════
# Quick start
# ═════════════════════════════════════════════════════════════════════
#
#   bash cost_model/build_measure_cpe.sh
#       Default build: AVX-2, gcc-15, parallelism = $(nproc).
#       Produces: build_tuned/measure_cpe (~9.7 MB executable).
#       Fresh build: ~70-90 s on a modern desktop.
#       Cached rebuild: ~5-7 s (link-only).
#
# ═════════════════════════════════════════════════════════════════════
# Configuration via environment variables
# ═════════════════════════════════════════════════════════════════════
#
#   ISA=avx2|avx512       Target ISA. Default avx2.
#                         avx512 produces build_tuned/measure_cpe_avx512.
#                         The host must support the requested ISA; this
#                         script picks codelets from codelets/$ISA/.
#
#   CC=<compiler>         C compiler. Default gcc-15. Override to gcc-11
#                         (production-aligned), gcc-13, clang-18, etc.
#                         (icx works too but slower at -O3 on R≥128).
#
#   NJOBS=<int>           Number of parallel compile jobs. Default
#                         $(nproc) on Linux, $(sysctl -n hw.physicalcpu)
#                         on macOS, 4 if neither found.
#                         Set NJOBS=1 to serialize (useful for debugging
#                         compile errors that get interleaved across
#                         parallel jobs).
#
#   CLEAN=1               Discard all cached .o files and re-compile
#                         everything from scratch. Use after sweeping
#                         changes you can't be sure mtime caught — e.g.
#                         compiler upgrades, CFLAGS changes, or recipe
#                         engine updates that affect every codelet.
#                         Equivalent to: rm -rf build_tuned/obj && ...
#
# ═════════════════════════════════════════════════════════════════════
# Examples
# ═════════════════════════════════════════════════════════════════════
#
#   # Default (AVX-2, gcc-15, max parallelism)
#   bash cost_model/build_measure_cpe.sh
#
#   # AVX-512 build on a Sapphire Rapids / Zen 4 host
#   ISA=avx512 bash cost_model/build_measure_cpe.sh
#
#   # Use production-aligned gcc-11
#   CC=gcc-11 bash cost_model/build_measure_cpe.sh
#
#   # Force clean rebuild (after gcc upgrade or recipe-engine changes)
#   CLEAN=1 bash cost_model/build_measure_cpe.sh
#
#   # Single-threaded for clearer compile error output
#   NJOBS=1 bash cost_model/build_measure_cpe.sh
#
#   # Combine: clean AVX-512 build on production compiler
#   CLEAN=1 ISA=avx512 CC=gcc-11 bash cost_model/build_measure_cpe.sh
#
# ═════════════════════════════════════════════════════════════════════
# What gets built
# ═════════════════════════════════════════════════════════════════════
#
#   Output:           build_tuned/measure_cpe (or measure_cpe_avx512)
#   .o cache:         build_tuned/obj/$ISA/r*.o (~378 files, ~50 MB)
#   R=1024 stubs:     build_tuned/obj/$ISA/r1024_stubs.c (auto-generated)
#
#   The cache survives across runs. Each codelet .o gets recompiled
#   only if its source .c is newer than the .o file (mtime-based).
#   Delete build_tuned/obj/ to force a full rebuild (or use CLEAN=1).
#
# ═════════════════════════════════════════════════════════════════════
# Performance design
# ═════════════════════════════════════════════════════════════════════
#
# (1) PARALLEL COMPILE via xargs -P NJOBS. Each codelet .c → .o is an
#     independent gcc invocation; they run concurrently. On 32 hardware
#     threads, 4-6× wall-time speedup vs the old single-gcc-invocation
#     style. Real perf data from a 32-thread i9-14900KF:
#         Fresh build (no cache):  ~75 s wall  (was ~5 min single-thread)
#         Cached rebuild:          ~7 s
#
# (2) .o CACHING in build_tuned/obj/$ISA/. A codelet recompiles only if
#     its source .c is newer than its .o. Means edits to one codelet (or
#     a fresh registry regen that didn't touch codelet bodies) only
#     recompile what actually changed.
#
# (3) xl_pow2 (R=1024) NOT compiled or linked. The registry references
#     R=1024 codelet symbols (n1_fwd, n1_bwd, t1_dit_fwd, t1_dit_log3_fwd)
#     so the linker needs SOMETHING at those names — we provide trivial
#     stubs. measure_cpe never bench's R=1024 (RADIX_LIST stops at 512),
#     so the stubs are never called. Saves ~30-60 s per fresh build —
#     R=1024 monolithic codelets are 76K-line straight-line bodies that
#     dominate compile time at -O3.
#
# (4) -march=native handles AMD and Intel uniformly. Auto-picks Intel
#     {haswell, skylake-avx512, icelake-server, sapphire-rapids, ...}
#     or AMD {znver1, znver2, znver3, znver4} based on the host CPU.
#     For ISA=avx512 we also assert -mavx512f -mavx512dq so the build
#     fails loudly if the host lacks AVX-512 (rather than silently
#     degrading and producing a binary that references undefined
#     AVX-512 codelets).
#
# ═════════════════════════════════════════════════════════════════════
# Cross-platform notes
# ═════════════════════════════════════════════════════════════════════
#
#   Linux:    Tested on Ubuntu 24.04 + gcc-15. WSL2 works.
#   macOS:    Untested but should work; uses sysctl for nproc fallback.
#   Windows:  Run via WSL or Git Bash. PowerShell version not provided
#             (the codelet tree itself is OS-independent; only build
#             scripting differs).
#
#   AMD vs Intel: Both work. -march=native does the right thing on each.
#   Specific tested platforms:
#     - Intel i9-14900KF (Raptor Lake) — AVX-2 only, used by this repo
#     - AMD Zen 3/4: should work via -march=native
#     - Intel Sapphire Rapids: should work for ISA=avx512
set -e

ROOT=$(cd "$(dirname "$0")/.." && pwd)
ISA=${ISA:-avx2}
CC=${CC:-gcc-15}
CLEAN=${CLEAN:-0}
# Detect parallelism. nproc on Linux, sysctl on macOS, fall back to 4.
NJOBS=${NJOBS:-$(nproc 2>/dev/null || sysctl -n hw.physicalcpu 2>/dev/null || echo 4)}

OUT_DIR=$ROOT/build_tuned
OBJ_DIR=$OUT_DIR/obj/$ISA

# Clean cache if requested.
if [ "$CLEAN" = "1" ] || [ "$CLEAN" = "true" ]; then
  echo "[build_measure_cpe] CLEAN=1: discarding cached .o files in $OBJ_DIR"
  rm -rf "$OBJ_DIR"
fi

mkdir -p $OBJ_DIR

if [ "$ISA" = "avx2" ]; then
  CFLAGS="-O3 -mavx2 -mfma -march=native -Wno-incompatible-pointer-types"
  OUT=$OUT_DIR/measure_cpe
  DEFS=""
elif [ "$ISA" = "avx512" ]; then
  CFLAGS="-O3 -mavx512f -mavx512dq -mfma -march=native -Wno-incompatible-pointer-types"
  OUT=$OUT_DIR/measure_cpe_avx512
  DEFS="-DVFFT_ISA_AVX512=1"
else
  echo "ERROR: unknown ISA '$ISA' (use avx2 or avx512)"
  exit 1
fi

# Gather codelet sources for this ISA — primes/composites/small/mid/large
# pow2 only. xl_pow2 (R=1024) intentionally skipped; we stub those symbols
# below for the registry's sake (see note 3 above).
CODELETS=()
for fam in primes small_pow2 mid_pow2 large_pow2 composites; do
  for f in $ROOT/codelets/$ISA/$fam/r*.c; do
    [ -f "$f" ] && CODELETS+=("$f")
  done
done
n=${#CODELETS[@]}

echo "[build_measure_cpe] ISA=$ISA, CC=$CC, NJOBS=$NJOBS"
echo "[build_measure_cpe] CFLAGS=$CFLAGS"
echo "[build_measure_cpe] codelet count=$n  (xl_pow2 skipped, stubbed)"

# ── Stage 1: emit R=1024 stubs ────────────────────────────────────────
# Tiny no-op definitions so the linker can resolve registry.h's R=1024
# externs without dragging in the 76K-line monolithic codelets.
STUBS_C=$OBJ_DIR/r1024_stubs.c
cat > $STUBS_C <<EOF
/* Auto-generated by build_measure_cpe.sh. Stubs for R=1024 codelets
 * referenced by the registry. measure_cpe never bench's R=1024
 * (RADIX_LIST stops at 512), so these are never called. */
#include <stddef.h>
__attribute__((target("$ISA")))
void radix1024_n1_fwd_$ISA(double *a, double *b, const double *c,
                            const double *d, size_t e, size_t f)
    { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; }
__attribute__((target("$ISA")))
void radix1024_n1_bwd_$ISA(double *a, double *b, const double *c,
                            const double *d, size_t e, size_t f)
    { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; }
__attribute__((target("$ISA")))
void radix1024_t1_dit_fwd_$ISA(double *a, double *b, const double *c,
                                const double *d, size_t e, size_t f)
    { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; }
__attribute__((target("$ISA")))
void radix1024_t1_dit_log3_fwd_$ISA(double *a, double *b, const double *c,
                                     const double *d, size_t e, size_t f)
    { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; }
EOF

# ── Stage 2: parallel compile ─────────────────────────────────────────
# Per-file freshness check: rebuild .o only if .c is newer OR .o missing.
# xargs -P N runs N gcc invocations in parallel.
compile_one() {
  local src=$1
  local obj=$OBJ_DIR/$(basename "$src" .c).o
  if [ ! -f "$obj" ] || [ "$src" -nt "$obj" ]; then
    $CC $CFLAGS $DEFS -I $ROOT/cost_model -c "$src" -o "$obj"
  fi
}
export -f compile_one
export CC CFLAGS DEFS OBJ_DIR ROOT

# All sources to compile: codelets + stubs + measure_cpe itself.
ALL_SOURCES=("${CODELETS[@]}" "$STUBS_C" "$ROOT/cost_model/measure_cpe.c")
TOTAL=${#ALL_SOURCES[@]}

# Count what needs work (informational only).
NEED=0
for src in "${ALL_SOURCES[@]}"; do
  obj=$OBJ_DIR/$(basename "$src" .c).o
  if [ ! -f "$obj" ] || [ "$src" -nt "$obj" ]; then
    NEED=$((NEED+1))
  fi
done
echo "[build_measure_cpe] compiling $NEED / $TOTAL sources ($((TOTAL-NEED)) cached)"

T0=$(date +%s)
printf '%s\n' "${ALL_SOURCES[@]}" | xargs -P $NJOBS -I {} bash -c 'compile_one "$@"' _ {}
T1=$(date +%s)
echo "[build_measure_cpe] compile phase: $((T1-T0))s"

# ── Stage 3: link ─────────────────────────────────────────────────────
echo "[build_measure_cpe] linking..."
T0=$(date +%s)
ALL_OBJS=()
for src in "${ALL_SOURCES[@]}"; do
  ALL_OBJS+=("$OBJ_DIR/$(basename "$src" .c).o")
done
$CC $CFLAGS "${ALL_OBJS[@]}" -o $OUT -lm
T1=$(date +%s)
echo "[build_measure_cpe] link phase: $((T1-T0))s"

echo "[build_measure_cpe] built $OUT"
ls -la $OUT
