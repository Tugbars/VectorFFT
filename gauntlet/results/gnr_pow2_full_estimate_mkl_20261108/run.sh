#!/bin/bash
# full pow2 1D sweep: VFFT_FLAG_ESTIMATE vs MKL, one process per cell, 3 passes
cd /home/claude/VectorFFT
R=gauntlet/results/gnr_pow2_full_estimate_mkl_20261108
for p in 1 2 3; do
  for e in $(seq 1 23); do
    n=$((1<<e))
    ./gauntlet/bench_estimate_vs_mkl - $R/runs.csv $n 2>&1 | grep -E "^(N=|# N=)" >> $R/run_pass$p.log
  done
  echo "pass $p done $(date +%T)" >> $R/progress.log
  sleep 20
done
echo ALLDONE >> $R/progress.log
