#!/bin/sh
# Regenerate the shipped tangent-interior codelets.
#
#   src/dag-fft-compiler/codelets/zil/avx2/pure_il/tangent/
#
# Each emitted file is renamed (the emitter tags neither the interior variant
# nor the split in the symbol) and keeps the fact-sheet header that documents
# its op counts; re-add that header by hand if you regenerate, or just diff
# against the committed file to confirm the body is unchanged.
#
# Verify afterwards:
#   gcc -O2 -static -o tangent_gate build_tuned/benches/tangent_gate.c \
#       src/dag-fft-compiler/codelets/zil/avx2/pure_il/tangent/*.c -lm && ./tangent_gate
#
# Run from the generator directory (WSL): sh emit_ship.sh
set -e
G=./_build/default/bin/gen_radix.exe
OUT=../codelets/zil/avx2/pure_il/tangent
TMP=${TMPDIR:-/tmp}/tangent_emit
U="--isa avx2 --uarch raptor_lake_avx2 --emit-c"
mkdir -p "$TMP"

# radix 8 — no general-twiddle sites; tangent only rewrites the two sqrt(1/2)
# folds, and the result is bit-identical to the classic codelet.
VFFT_CX_LAZYLOAD=1 $G 8  --cil-t2  --cil-tangent $U > "$TMP/r8_t2.c"
VFFT_CX_LAZYLOAD=1 $G 8  --cil-n1t --cil-tangent $U > "$TMP/r8_n1t.c"

# radix 16 — the wing construction (loads + ingest + butterflies interleaved in
# construction order) plus interleaved stores; this is the parity kernel.
VFFT_CX_WING=1 VFFT_CX_LAZYLOAD=1 VFFT_CX_SCHED=asis \
  $G 16 --cil-t2  --cil-tangent $U > "$TMP/r16_t2.c"
VFFT_CX_WING=1 VFFT_CX_LAZYLOAD=1 VFFT_CX_SCHED=asis \
  $G 16 --cil-n1t --cil-tangent $U > "$TMP/r16_n1t.c"

# radix 32 — SUPERSEDED 2026-08-13 by the wing32 recipes below (kept for the
# bannered t2btan216 file until pool-sunset deletes it): blocked halves
# (split is m.p, so 2.16 = two 16-point halves whose combine pass uses the
# tangent butterfly; 16.2 would not).
VFFT_CX_WING=1 VFFT_CX_LAZYLOAD=1 VFFT_CX_LAZYSTORE=1 \
  $G 32 --cil-t2 --cil-tangent --cil-blocked --cil-split 2.16 $U > "$TMP/r32_t2b.c"

# radix 32 wing32 (A-1, docs/roadmap/r32_tangent_parity_plan.md):
# W32TG = canonical-angle pass-B combine (ulp-twin table collapsed),
# ROTFMA = -i rotations folded into sign-folded FMA constants (zero xors),
# TURN128 (leaf only) = per-output split-128 corner-turn stores, no
# permute2f128 — the old paired edge was the +32.4% killed-leaf tax.
VFFT_CX_WING=1 VFFT_CX_LAZYLOAD=1 VFFT_CX_LAZYSTORE=1 VFFT_CX_W32TG=1 VFFT_CX_ROTFMA=1 \
  $G 32 --cil-t2  --cil-tangent --cil-blocked --cil-split 2.16 $U > "$TMP/r32_t2bw32.c"
VFFT_CX_WING=1 VFFT_CX_LAZYLOAD=1 VFFT_CX_LAZYSTORE=1 VFFT_CX_W32TG=1 VFFT_CX_ROTFMA=1 VFFT_CX_TURN128=1 \
  $G 32 --cil-n1t --cil-tangent --cil-blocked --cil-split 2.16 $U > "$TMP/r32_n1tbw32.c"

ren() { sed "s/$2/$3/g" "$TMP/$1" > "$TMP/$1.ren" && mv "$TMP/$1.ren" "$TMP/$1"; }
ren r8_t2.c        radix8_z_t2_fwd_avx2    radix8_z_t2tan_fwd_avx2
ren r8_n1t.c       radix8_z_n1t_fwd_avx2   radix8_z_n1ttan_fwd_avx2
ren r16_t2.c       radix16_z_t2_fwd_avx2   radix16_z_t2tan_fwd_avx2
ren r16_n1t.c      radix16_z_n1t_fwd_avx2  radix16_z_n1ttan_fwd_avx2
ren r32_t2b.c      radix32_z_t2b_fwd_avx2  radix32_z_t2btan216_fwd_avx2
ren r32_t2bw32.c   radix32_z_t2b_fwd_avx2  radix32_z_t2bw32_fwd_avx2
ren r32_n1tbw32.c  radix32_z_n1tb_fwd_avx2 radix32_z_n1tbw32_fwd_avx2

echo "emitted to $TMP — compare against $OUT before overwriting:"
for f in r8_t2 r8_n1t r16_t2 r16_n1t r32_t2b r32_t2bw32 r32_n1tbw32; do echo "  $TMP/$f.c"; done
