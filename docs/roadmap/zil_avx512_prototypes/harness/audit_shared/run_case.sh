#!/bin/bash
# usage: run_case.sh <folder rel to zil/avx2> <base name e.g. radix3_z_n1> <kind> [bwd]
set -e
Z=/home/user/VectorFFT/src/dag-fft-compiler/codelets/zil/avx2
S=/tmp/claude-0/-home-user-VectorFFT/207882f8-0ee5-5dd9-bdc8-3e5ddb7a3abd/scratchpad
H=$S/wf1/audit_a/h
folder=$1; base=$2; kind=$3; bwd=${4:-0}
tag=$(echo "$folder" | tr / _)_$base
A=$Z/$folder/${base}_avx2.c
B=$S/wf1/audit_a/regen2/$(echo "$folder" | tr / _)_${base}_avx2.c
C=${CFILE:-$S/a512/out/$(echo "$folder" | tr / _)_${base}_avx512.c}
W=$H/build/$tag${WSUF:-}; mkdir -p $W
CF="-O2 -ffp-contract=off -mavx512f -mavx512dq -mfma"
symA=$(grep -oE "radix[0-9]+_z_[a-z0-9_]+_avx2\b" $A | sort -u | head -1)
symC=$(grep -oE "radix[0-9]+_z_[a-z0-9_]+_avx512\b" $C | sort -u | head -1)
gcc $CF -c $A -o $W/a.o
gcc $CF -c $B -o $W/b0.o
objcopy --redefine-sym $symA=B_$symA $W/b0.o $W/b.o
gcc $CF -c $C -o $W/c.o
gcc -O1 -ffp-contract=off -mavx512f -mavx512dq -mfma -DKIND=$kind -DRADIX=$(echo $base | sed -E 's/radix([0-9]+)_.*/\1/') -DBWD=$bwd \
    -DSYM_A=$symA -DSYM_B=B_$symA -DSYM_C=$symC $H/${HSRC:-harness.c} $W/a.o $W/b.o $W/c.o -o $W/h -lm
$W/h
