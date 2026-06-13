#!/usr/bin/env bash
# snapshot_step2_baseline.sh — regenerate + hash the step-2 byte baseline.
# Covers the 36 fuse=0 codelets AND 12 fuse=2 R>=25 variants that exercise the
# :926/emit_c fused-tag forward-declare path (dead code at the default fuse=0,
# so the original 36-codelet baseline gave the fused-tag mirror ZERO coverage).
# Usage: ./benchmarks/snapshot_step2_baseline.sh <outdir>   (default /tmp/step2_baseline)
set -uo pipefail
GEN="$(cd "$(dirname "$0")/.." && pwd)/generator/_build/default/bin/gen_radix.exe"
OUT="${1:-/tmp/step2_baseline}"; rm -rf "$OUT"; mkdir -p "$OUT"
for R in 16 32 64; do
  for dir in fwd bwd; do df=""; [ "$dir" = bwd ] && df="--bwd"
    $GEN $R --twiddled $df --in-place --isa avx512 --su --emit-c > "$OUT/t1_${R}_${dir}.c" 2>/dev/null
    $GEN $R --twiddled --log3 $df --in-place --isa avx512 --su --emit-c > "$OUT/log3_${R}_${dir}.c" 2>/dev/null
    $GEN $R --twiddled --t1s $df --in-place --isa avx512 --su --emit-c > "$OUT/t1s_${R}_${dir}.c" 2>/dev/null
    $GEN $R --oop --twiddled-pos $df --oop-load UG --oop-store UG --isa avx512 --emit-c > "$OUT/t1p_${R}_${dir}.c" 2>/dev/null
    $GEN $R $df --in-place --isa avx512 --su --emit-c > "$OUT/n1_${R}_${dir}.c" 2>/dev/null
  done
done
for R in 16 32 64; do
  $GEN $R --twiddled --in-place --isa avx2 --su --emit-c > "$OUT/t1_${R}_avx2.c" 2>/dev/null
  $GEN $R --in-place --isa avx2 --su --emit-c > "$OUT/n1_${R}_avx2.c" 2>/dev/null
done
# fuse=2 R>=25 — exercises the fused-tag forward-declare/no-store/skip-reload path
for R in 25 32 64; do
  $GEN $R --oop --twiddled-pos --fuse 2 --oop-load UG --oop-store UG --isa avx512 --emit-c > "$OUT/fuse2_t1p_${R}_fwd.c" 2>/dev/null
  $GEN $R --oop --twiddled-pos --fuse 2 --bwd --oop-load UG --oop-store UG --isa avx512 --emit-c > "$OUT/fuse2_t1p_${R}_bwd.c" 2>/dev/null
  $GEN $R --in-place --fuse 2 --isa avx512 --su --emit-c > "$OUT/fuse2_n1_${R}.c" 2>/dev/null
  $GEN $R --twiddled --in-place --fuse 2 --isa avx512 --su --emit-c > "$OUT/fuse2_t1_${R}.c" 2>/dev/null
done
n=$(ls "$OUT"/*.c | wc -l)
( cd "$OUT" && sha256sum *.c | sort -k2 > "$OUT.manifest" )
echo "$n codelets -> $OUT ; manifest -> $OUT.manifest"
echo "aggregate sha256: $(cat "$OUT"/*.c | sha256sum | cut -d' ' -f1)"
