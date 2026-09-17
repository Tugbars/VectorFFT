#!/bin/sh
# prime_vs_mkl.sh — the prime cells against MKL (2026-09-18).
#
# For each prime N: on a SCRATCH copy of the shipped store, one create through
# the front door (natural, out of place, T = 1, PATIENT) races the prime cell's
# inner and banks it (recal_1d_probe, recalibrate = 0: a miss races); then the
# canonical bench in a FRESH process replays the banked verdict and times it
# beside MKL (--k1noop: natural OOP T = 1, pace 300 ms, cool 300 ms, core 2 +
# HIGH, MKL single-threaded) -- never an in-process timing after a cold race.
#
# Run from build_tuned/, machine QUIET:  sh prime_vs_mkl.sh <out-dir>
set -u
OUT="${1:?out-dir}"
HERE="$(cd "$(dirname "$0")" && pwd)"
G="$HERE/../src/dag-fft-compiler/generator/generated"
ST="$OUT/store"
rm -rf "$ST"; mkdir -p "$ST"
cp "$G"/*.txt "$ST"/
export VFFT_WISDOM_DIR="$ST"
export VFFT_ILPR_LOG=1
RES="$OUT/prime_vs_mkl.txt"
: > "$RES"
printf "%-8s %-52s %-8s %s\n" "N" "banked verdict (eng / inner)" "vs MKL" "err" | tee -a "$RES"
for N in 31 127 257 1021 4099 8191 65537 131071; do
  ./benches/recal_1d_probe.exe "$ST" "$N" 0 0 1 0 > "$OUT/create_$N.txt" 2>&1
  ROW=$(grep -E "n=$N " "$ST/wisdom2_prime.txt" | grep -oE "eng=[a-z]+ in=[a-z0-9]+ in_sh=[0-9.]+ in_tw=[0-9]+" | head -1)
  [ -z "$ROW" ] && ROW="(no prime row: $(grep -oE 'refused|banked' "$OUT/create_$N.txt" | head -1))"
  ./benches/bench_1d_vs_mkl.exe --k1noop "$ST/spike_wisdom.txt" "$OUT/csv_$N.csv" 300 "$N" 1 300 0 2 > "$OUT/bench_$N.txt" 2>&1
  L=$(grep -E "^$N,1,.*,nat-oop," "$OUT/csv_$N.csv" 2>/dev/null | tail -1)
  SP=$(echo "$L" | awk -F, '{print $(NF-1)}')
  ERR=$(echo "$L" | awk -F, '{print $NF}')
  printf "%-8s %-52s %-8s %s\n" "$N" "$ROW" "${SP:-?}" "${ERR:-?}" | tee -a "$RES"
done
echo "done -> $RES"
