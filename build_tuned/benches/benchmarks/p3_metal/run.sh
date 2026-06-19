#!/bin/bash
# P3 metal run: taskset-pinned, all variants, optional perf counters.
CPU=${CPU:-2}
OUT=${OUT:-p3_results.csv}
echo "variant,lane,N,K,ns" > "$OUT"
for V in base ranged prefetch ranged_prefetch; do
  echo "=== $V ==="
  if command -v perf >/dev/null && [ "${PERF:-0}" = "1" ]; then
    taskset -c $CPU perf stat -e cycles,instructions,L1-dcache-load-misses,LLC-load-misses \
      /tmp/p3_$V 2>&1 | tee /tmp/p3_${V}.log
  else
    taskset -c $CPU /tmp/p3_$V | tee /tmp/p3_${V}.log
  fi
  grep '^csv,' /tmp/p3_${V}.log | sed "s/^csv,/$V,/" >> "$OUT"
done
echo "results -> $OUT (same-run ratios only)"
