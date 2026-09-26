#!/bin/bash
# ab_run.sh <rel>   -> builds + runs the bitwise A/B for one turned recipe
W=/tmp/claude-0/-home-user-VectorFFT/207882f8-0ee5-5dd9-bdc8-3e5ddb7a3abd/scratchpad/wf1/turn
rel="$1"; b=$(echo ${rel%.c} | tr / _)
f2=$W/ab_avx2/$b.base.c; f5=$W/ab_avx512/$b.new.c
s2=$(grep -m1 -oE '^void [a-z0-9_]+' $f2 | cut -d' ' -f2); s5=$(grep -m1 -oE '^void [a-z0-9_]+' $f5 | cut -d' ' -f2)
R=$(echo $s2 | sed -E 's/^radix([0-9]+)_.*/\1/')
cls=0; case "$rel" in chain3/*) cls=1;; rows/*) cls=2;; esac
tw=0; case "$s2" in *_z_t2*) tw=1;; esac
blk=0; case "$rel" in *blocked*|*n1tbw32t256*) blk=1;; esac
d=$W/abrun/$b; mkdir -p $d
gcc -O2 -w -c $f2 -o $d/k2.o && gcc -O2 -w -c $f5 -o $d/k5.o && \
gcc -O2 -w -DF2=$s2 -DF5=$s5 -DRAD=$R -DCLS=$cls -DTW=$tw -DBLK=$blk $W/ab_harness.c $d/k2.o $d/k5.o -lm -o $d/ab 2> $d/build.err || { echo "BUILDFAIL $rel $(head -c 200 $d/build.err)"; exit 0; }
$d/ab 2>&1 | tail -6
