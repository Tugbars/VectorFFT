#!/bin/bash
# Build prime codelet bench: Hand-coded (Python gen_radix*.py) vs OCaml-generated.
#
# Prerequisites:
#   - vfft_v2_pack built (dune build)
#   - gen_radix5.py / gen_radix7.py / gen_radix11.py in $PYGEN dir
#
# Usage:
#   PYGEN=/path/to/python/generators ./build.sh

set -e
PYGEN=${PYGEN:-.}
GEN_OCAML=../../_build/default/bin/gen_radix.exe

# Hand-coded headers (Python generators)
for R in 5 7 11; do
  for V in ct_t1_dit ct_t1_dif ct_t1_dit_log3 ct_t1s_dit; do
    NAME=$(echo $V | sed 's/^ct_//')
    python3 "$PYGEN/gen_radix${R}.py" --isa avx512 --variant $V > "hand_r${R}_${NAME}.h" 2>/dev/null
    # Strip any stderr banners that leaked into output
    sed -i '/^=== /d' "hand_r${R}_${NAME}.h"
  done
done

# OCaml-generated codelets
for R in 5 7 11; do
  $GEN_OCAML $R --twiddled --in-place --emit-c              > gen_r${R}_t1_dit.c
  $GEN_OCAML $R --twiddled --dif --in-place --emit-c        > gen_r${R}_t1_dif.c
  $GEN_OCAML $R --twiddled --log3 --in-place --emit-c       > gen_r${R}_t1_dit_log3.c
  $GEN_OCAML $R --twiddled --t1s --in-place --emit-c        > gen_r${R}_t1s_dit.c
done

# Build bench harness
gcc -O3 -mavx512f -mavx512dq -mfma -march=native bench_prime.c \
    gen_r5_t1_dit.c gen_r5_t1_dif.c gen_r5_t1_dit_log3.c gen_r5_t1s_dit.c \
    gen_r7_t1_dit.c gen_r7_t1_dif.c gen_r7_t1_dit_log3.c gen_r7_t1s_dit.c \
    gen_r11_t1_dit.c gen_r11_t1_dif.c gen_r11_t1_dit_log3.c gen_r11_t1s_dit.c \
    -o bench_prime -lm

echo "Built. Run: ./bench_prime"
