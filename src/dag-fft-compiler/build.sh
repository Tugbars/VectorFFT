#!/usr/bin/env bash
# Build the VectorFFT OOP engine + all investigation benchmarks.
#
# Requirements:
#   - gcc with AVX-512 support (gcc-13 used for the measurements)
#   - FFTW 3.x built with AVX-512 (for the comparison baselines)
#
# Point these at your FFTW build, or override on the command line:
#   FFTW_INC=/path/to/fftw/api FFTW_LIB=/path/to/libfftw3.a ./build.sh
set -e
FFTW_INC="${FFTW_INC:-/home/claude/fftw-3.3.10/api}"
FFTW_LIB="${FFTW_LIB:-/home/claude/fftw-3.3.10/.libs/libfftw3.a}"
CC="${CC:-gcc}"
FLAGS="-O3 -mavx512f -mavx512dq -mfma -I${FFTW_INC}"
OUT=bin
mkdir -p "$OUT"

C=codelets
echo "== engine =="
$CC $FLAGS engine/engine_natural_oop.c \
    $C/radix64_n1_oop_avx512.c $C/radix16_t1s_oop_avx512.c \
    "$FFTW_LIB" -lm -o "$OUT/engine"

# in-place-twiddle (FFTW-method) engine, the recommended structure (see docs/OOP_DESIGN.md)
$CC $FLAGS engine/engine_natural_oop_inplace_twiddle.c \
    $C/radix16_n1_oop_avx512.c $C/radix64_t1s_oop_avx512.c \
    $C/radix64_n1_oop_avx512.c $C/radix16_t1s_oop_avx512.c \
    "$FFTW_LIB" -lm -o "$OUT/engine_inplace_twiddle"

# one-call engine (t1p per-position broadcast); 32x32 balanced with a log3 twiddle stage is the fastest natural-order OOP path (see docs/OOP_DESIGN.md)
$CC $FLAGS engine/engine_natural_oop_onecall.c \
    $C/radix32_n1_oop_avx512.c $C/radix32_t1p_oop_avx512.c \
    $C/radix32_t1p_log3_oop_avx512.c \
    "$FFTW_LIB" -lm -o "$OUT/engine_onecall"

# stride-specialized + M-project one-call engine: strides baked as compile-time
# constants and trailing PASS-2 sub-DFTs kept register-resident. ~6-10% faster
# than engine_onecall, log3_spec beats FFTW ~1.15-1.23x. See
# docs/oop_stride_specialization.md. (Regenerate the *_spec codelets with
# codelets/regen_spec_r32.sh after a generator change.)
$CC $FLAGS engine/engine_natural_oop_onecall_spec.c \
    $C/radix32_n1_oop_avx512_spec.c $C/radix32_t1p_oop_avx512_spec.c \
    $C/radix32_t1p_log3_oop_avx512_spec.c \
    "$FFTW_LIB" -lm -o "$OUT/engine_onecall_spec"

# 4-stage 4x4x4x16 engine (OOP stage-count study; a confirming negative, see docs/OOP_DESIGN.md)
$CC $FLAGS engine/engine_natural_oop_4stage.c \
    $C/radix4_n1_oop_avx512.c $C/radix4_t1p_oop_avx512.c $C/radix16_t1p_oop_avx512.c \
    $C/radix32_n1_oop_avx512.c $C/radix32_t1p_oop_avx512.c \
    "$FFTW_LIB" -lm -o "$OUT/engine_4stage"

echo "== benchmarks =="
$CC $FLAGS benchmarks/01_leaf_spill.c \
    $C/radix64_n1_inplace_avx512.c $C/radix64_n1_oop_avx512.c -lm -o "$OUT/01_leaf_spill"
$CC $FLAGS benchmarks/02_stage_split_and_aliasing.c \
    $C/radix64_n1_oop_avx512.c $C/radix16_t1s_oop_avx512.c "$FFTW_LIB" -lm -o "$OUT/02_stage_split_and_aliasing"
$CC $FLAGS benchmarks/03_call_fragmentation.c \
    $C/radix16_n1_oop_avx512.c -lm -o "$OUT/03_call_fragmentation"
$CC $FLAGS benchmarks/04_leaf_codelet_quality.c \
    $C/radix16_n1_oop_avx512.c $C/radix64_n1_oop_avx512.c "$FFTW_LIB" -lm -o "$OUT/04_leaf_codelet_quality"
$CC $FLAGS benchmarks/05_twiddle_overhead.c \
    $C/radix16_n1_oop_avx512.c $C/radix16_t1s_oop_avx512.c \
    $C/radix64_n1_oop_avx512.c $C/radix64_t1s_oop_avx512.c -lm -o "$OUT/05_twiddle_overhead"
$CC $FLAGS benchmarks/06_block_width_ilp.c \
    $C/radix64_n1_oop_avx512.c $C/radix16_t1s_oop_avx512.c "$FFTW_LIB" -lm -o "$OUT/06_block_width_ilp_V8"
# 06 is also built at V=16 and V=32 to show the block-width sweep
$CC $FLAGS -DV=16 benchmarks/06_block_width_ilp.c \
    $C/radix64_n1_oop_avx512.c $C/radix16_t1s_oop_avx512.c "$FFTW_LIB" -lm -o "$OUT/06_block_width_ilp_V16"
$CC $FLAGS -DV=32 benchmarks/06_block_width_ilp.c \
    $C/radix64_n1_oop_avx512.c $C/radix16_t1s_oop_avx512.c "$FFTW_LIB" -lm -o "$OUT/06_block_width_ilp_V32"
$CC $FLAGS benchmarks/07_fftw_plan_dump.c "$FFTW_LIB" -lm -o "$OUT/07_fftw_plan_dump"
$CC $FLAGS benchmarks/08_compare_vs_fftw_patient.c \
    $C/radix64_n1_oop_avx512.c $C/radix16_t1s_oop_avx512.c "$FFTW_LIB" -lm -o "$OUT/08_compare_vs_fftw_patient"

echo
echo "Built into $OUT/. Run e.g.:"
echo "  $OUT/engine 2048                 # verify + bench at K=2048"
echo "  $OUT/07_fftw_plan_dump 512        # FFTW's actual plan tree"
echo "  $OUT/08_compare_vs_fftw_patient 2048"
