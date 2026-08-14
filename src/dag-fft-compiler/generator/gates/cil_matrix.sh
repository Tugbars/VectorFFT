#!/bin/bash
# cil_matrix.sh <outdir> — the 183-case codelet_cil.ml emission matrix.
#
# PROVENANCE: recovered verbatim from the scratchpad of session
# dfcd8b2a-b554-435a-b1f8-81f8ebd8e66f (`emit_matrix.sh`), the script that gated
# the cx_* decomposition and the cx_pipeline hosting as "183/183 byte-identical".
# It was living in a session-scoped temp dir; this is its durable home.
#
# Verified 2026-08-14: 183 cases, 10.2 s, run-A vs run-B byte-identical,
# 6 cases exit non-zero BY DESIGN (the refusal cases — those exits are part of
# the contract and the .out files record EXIT= plus stderr).
#
# Byte-identity gate: diff -r <before> <after> must be empty.
set -u
OUT="$1"; mkdir -p "$OUT"
GEN=./_build/default/bin/gen_radix.exe
ISA="--isa avx2 --uarch raptor_lake_avx2"
i=0
run() { # run <slug> <args...>
  local slug="$1"; shift
  i=$((i+1))
  local f
  f=$(printf "%s/%03d_%s.out" "$OUT" "$i" "$slug")
  $GEN "$@" $ISA --emit-c > "$f" 2> "$f.err"
  echo "EXIT=$?" >> "$f"
  cat "$f.err" >> "$f"; rm -f "$f.err"
}

KINDS="n1 n1t t2"
for R in 2 4 8 16 32 64 3 5 7 9 11 13 15 25 49 6 10 12 20; do
  for K in $KINDS; do
    run "r${R}_${K}_fwd" "$R" --cil-$K
    run "r${R}_${K}_bwd" "$R" --cil-$K --cil-bwd
  done
done

# blocked, pow2 default splits
for R in 16 32 64; do
  for K in $KINDS; do
    run "r${R}_${K}b_fwd" "$R" --cil-$K --cil-blocked
    run "r${R}_${K}b_bwd" "$R" --cil-$K --cil-blocked --cil-bwd
  done
done

# blocked, explicit splits
for S in "16 4.4" "16 2.8" "16 8.2" "32 4.8" "64 8.8"; do
  set -- $S
  for K in n1t t2; do
    run "r${1}_${K}b${2}_fwd" "$1" --cil-$K --cil-blocked --cil-split "$2"
    run "r${1}_${K}b${2}_bwd" "$1" --cil-$K --cil-blocked --cil-split "$2" --cil-bwd
  done
done
# odd blocked (+ the odd-p turned refusal + the no-split refusal)
for S in "25 5.5" "15 3.5" "49 7.7"; do
  set -- $S
  run "r${1}_n1b${2}_fwd" "$1" --cil-n1 --cil-blocked --cil-split "$2"
  run "r${1}_t2b${2}_fwd" "$1" --cil-t2 --cil-blocked --cil-split "$2"
done
run "r15_n1tb3.5_fwd_REFUSE" 15 --cil-n1t --cil-blocked --cil-split 3.5
run "r25_n1b_nosplit_REFUSE" 25 --cil-n1 --cil-blocked

# log3 (t2 only; + the n1 refusal)
for R in 8 32; do
  run "r${R}_t2_log3_fwd" "$R" --cil-t2 --cil-log3
  run "r${R}_t2_log3_bwd" "$R" --cil-t2 --cil-log3 --cil-bwd
done
run "r8_n1_log3_REFUSE" 8 --cil-n1 --cil-log3

# turned stores: t2t / t2tg; blocked turned; blocked t2tg refusal
for R in 8 16; do
  run "r${R}_t2t_fwd" "$R" --cil-t2 --cil-turnst
  run "r${R}_t2t_bwd" "$R" --cil-t2 --cil-turnst --cil-bwd
done
run "r8_t2tg_fwd" 8 --cil-t2 --cil-turnst-gs
run "r8_t2tg_bwd" 8 --cil-t2 --cil-turnst-gs --cil-bwd
run "r16_t2bt4.4_fwd" 16 --cil-t2 --cil-blocked --cil-split 4.4 --cil-turnst
run "r16_t2bt4.4_bwd" 16 --cil-t2 --cil-blocked --cil-split 4.4 --cil-turnst --cil-bwd
run "r16_t2btg_REFUSE" 16 --cil-t2 --cil-blocked --cil-split 4.4 --cil-turnst-gs

# pretw is a hard parse-time refusal
run "r8_t2p_REFUSE" 8 --cil-t2 --cil-pretw

# fused K=1
run "k1_16_a4.2_b2_fwd"    16   --cil-k1 --cil-chain "4.2:2"
run "k1_64_a8_b8_fwd"      64   --cil-k1 --cil-chain "8:8"
run "k1_64_a8_b8_bwd"      64   --cil-k1 --cil-chain "8:8" --cil-bwd
run "k1_256_a16_b16_fwd"   256  --cil-k1 --cil-chain "16:16"
run "k1_512_a8.4_b4.4_fwd" 512  --cil-k1 --cil-chain "8.4:4.4"
run "k1_512_a8.4_b4.4_bwd" 512  --cil-k1 --cil-chain "8.4:4.4" --cil-bwd
run "k1_1024_a32_b32_fwd"  1024 --cil-k1 --cil-chain "32:32"
run "k1_nochain_REFUSE"    64   --cil-k1

echo "cases: $i"
