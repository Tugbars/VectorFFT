#!/bin/sh
# band_recal_check.sh — the band-map regression check (2026-09-18).
#
# For one representative N per region of policy.h's K=1 interleaved band map:
#   1. recalibrate the cell on a SCRATCH copy of the shipped store
#      (recal_1d_probe: cfg.recalibrate = 1 -> re-race, overwrite the verdict);
#   2. read the route the store BANKED for it (il_route=...) -- the direct
#      check that the region got the method policy.h says it gets;
#   3. bench the cell with the canonical bench, --k1noop (natural, out of
#      place, T = 1), one process per cell, the recorded protocol:
#      pace 300 ms, cool 300 ms, core 2 + HIGH, MKL single-threaded;
#   4. print route + vfft/MKL ratio beside the ratio recorded in
#      docs/performance/v1_0_results.md, so a wrong verdict shows twice: by
#      name, and by cost.
#
# Run from build_tuned/, machine QUIET:  sh band_recal_check.sh <out-dir>
set -u
OUT="${1:?out-dir}"
HERE="$(cd "$(dirname "$0")" && pwd)"
G="$HERE/../src/dag-fft-compiler/generator/generated"
ST="$OUT/store"
rm -rf "$ST"; mkdir -p "$ST"
cp "$G"/*.txt "$ST"/
export VFFT_WISDOM_DIR="$ST"
RES="$OUT/band_recal.txt"
: > "$RES"
printf "%-8s %-10s %-34s %-9s %s\n" "N" "region" "banked route (after recal)" "vs MKL" "recorded" | tee -a "$RES"

# N region recorded-ratio   (natural OOP T=1, docs/performance/v1_0_results.md)
CELLS="
16      mono         1.18
32      mono         0.90
64      mono         0.91
128     pair+ztt     1.01
256     pair+ztt     0.99
512     pair+ztt     0.98
1024    pair+ztt     1.18
2048    ztt          1.30
4096    ztt          1.02
8192    ztt          1.05
16384   ztt          1.02
262144  ztt|fs       0.94-1.12
524288  fourstep     1.00-1.08
1048576 fourstep     1.26-1.34
2097152 fourstep     1.40-1.43
3072    ztt-odd      1.85
12288   ztt-odd      1.91
24576   ztt-odd      1.94
1215    chain3       1.06
4095    chain3       1.24
3125    flat         1.04
6561    flat         1.14
15625   flat         1.10
19683   flat         1.20
"
echo "$CELLS" | while read N REG REC; do
  [ -z "$N" ] && continue
  # 1. recalibrate (natural, out of place, T=1)
  ./benches/recal_1d_probe.exe "$ST" "$N" 0 0 1 1 > "$OUT/recal_$N.txt" 2>&1
  # 2. the banked route, from the store
  ROUTE=$(grep -E "n=$N q=1 ord=nat place=oop role=comp lay=il \|" "$ST/wisdom2_oop.txt" | grep -v "dir=bwd" | head -1 \
          | grep -oE "il_route=[a-z0-9]+( il_(pair|ztt|flat)=[0-9.]+)?" | head -1)
  [ -z "$ROUTE" ] && ROUTE="(no row)"
  # 3. the canonical bench, one process, the recorded invocation
  ./benches/bench_1d_vs_mkl.exe --k1noop "$ST/spike_wisdom.txt" "$OUT/csv_$N.csv" 300 "$N" 1 300 0 2 > "$OUT/bench_$N.txt" 2>&1
  SP=$(grep -E "^$N,1," "$OUT/csv_$N.csv" 2>/dev/null | tail -1 | awk -F, '{print $(NF-1)}')
  [ -z "$SP" ] && SP="?"
  printf "%-8s %-10s %-34s %-9s %s\n" "$N" "$REG" "$ROUTE" "$SP" "$REC" | tee -a "$RES"
done
echo "done -> $RES"
