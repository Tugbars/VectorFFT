#!/bin/sh
# gauntlet_1d.sh — the 1D C2C vs MKL gauntlet (2026-09-19; resume 2026-09-20).
#
# Two phases, because a cold race is a PLANNING cost that must not sit inside
# a timed process, and because a verdict banked from one sample is what the
# bench will then measure:
#   1. CALIBRATE — one front-door create per cell (recal_1d_probe, recal = 0:
#      a miss races and banks, a hit replays), on a SCRATCH copy of the
#      shipped store. Records wall-clock per cell and whether it was served.
#   2. BENCH — one process per cell, the canonical bench with --k1noop
#      (natural, out of place, T = 1), pace 300 ms, cool 300 ms, core 2 +
#      HIGH, MKL single-threaded. Every process APPENDS to ONE csv; whichever
#      process creates it writes the header.
#
# TWO FAIRNESS RULES, both from measured house findings:
#   - BOTH ENGINE ORDERS AT EVERY CELL. Whichever engine runs second is timed
#     on a hotter core; a fixed order was measured at up to +81% on a long
#     run. Every cell gets flip 0 and flip 1 and the csv's flip column keeps
#     them apart.
#   - A CONTROL CELL every CONTROL_EVERY cells, into control.csv, same
#     protocol, one fixed length, rows in run order: if the first and the
#     last agree the run is internally comparable. Absolute ns are never
#     comparable across a long run; the per-cell RATIO is, because both
#     engines are timed back to back in one process.
#
# RESUME (2026-09-20): `resume` keeps the store and skips every cell already
# in calibrate.log, then benches. A stopped run loses only the cell in flight
# -- every finished cell's verdict is already banked in the store.
#
# Run from build_tuned/, machine QUIET:
#   sh gauntlet_1d.sh <out-dir> <cell-list> [calibrate|bench|both|resume]
set -u
OUT="${1:?out-dir}"
LIST="${2:?cell-list file}"
PHASE="${3:-both}"
CONTROL_N=${CONTROL_N:-4096}
CONTROL_EVERY=${CONTROL_EVERY:-100}
HERE="$(cd "$(dirname "$0")" && pwd)"
G="$HERE/../src/dag-fft-compiler/generator/generated"
ST="$OUT/store"
CSV="$OUT/gauntlet.csv"
CTL="$OUT/control.csv"
CAL="$OUT/calibrate.log"
mkdir -p "$OUT"
export VFFT_WISDOM_DIR="$ST"

cells() { grep -vE '^\s*(#|$)' "$LIST" | awk '{print $1}'; }

bench_cell() {   # $1 = N, $2 = csv path
  for F in 0 1; do
    "$HERE/benches/bench_1d_vs_mkl.exe" --k1noop "$ST/spike_wisdom.txt" "$2" \
        300 "$1" 1 300 "$F" 2 > /dev/null 2>&1
  done
}

if [ "$PHASE" = calibrate ] || [ "$PHASE" = both ] || [ "$PHASE" = resume ]; then
  if [ "$PHASE" != resume ]; then
    rm -rf "$ST"; mkdir -p "$ST"
    cp "$G"/*.txt "$ST"/
    : > "$CAL"
  fi
  n=0; skipped=0
  t0=$(date +%s)
  for N in $(cells); do
    if [ "$PHASE" = resume ] && grep -qE "^$N +(banked|REFUSED)" "$CAL"; then
      skipped=$((skipped + 1)); continue
    fi
    s0=$(date +%s%3N)
    "$HERE/benches/recal_1d_probe.exe" "$ST" "$N" 0 0 1 0 > "$OUT/.cal.tmp" 2>&1
    s1=$(date +%s%3N)
    st=$(grep -oE 'banked|REFUSED' "$OUT/.cal.tmp" | tail -1)
    printf "%-10s %-8s %s\n" "$N" "${st:-ERROR}" "$((s1 - s0))ms" >> "$CAL"
    n=$((n + 1))
    [ $((n % 100)) -eq 0 ] && echo "  calibrated $n cells this pass, $(($(date +%s) - t0))s elapsed" >&2
  done
  rm -f "$OUT/.cal.tmp"
  echo "calibrate: $n cells this pass ($skipped already done) in $(($(date +%s) - t0))s -> $CAL" >&2
  echo "  served: $(grep -c banked "$CAL")   refused: $(grep -c REFUSED "$CAL")" >&2
fi

if [ "$PHASE" = bench ] || [ "$PHASE" = both ] || [ "$PHASE" = resume ]; then
  rm -f "$CSV" "$CTL"
  "$HERE/benches/recal_1d_probe.exe" "$ST" "$CONTROL_N" 0 0 1 0 > /dev/null 2>&1
  bench_cell "$CONTROL_N" "$CTL"
  n=0
  t0=$(date +%s)
  for N in $(cells); do
    grep -qE "^$N +banked" "$CAL" 2>/dev/null || continue   # never bench a refused cell
    bench_cell "$N" "$CSV"
    n=$((n + 1))
    if [ $((n % CONTROL_EVERY)) -eq 0 ]; then
      bench_cell "$CONTROL_N" "$CTL"
      echo "  benched $n cells, $(($(date +%s) - t0))s elapsed (control taken)" >&2
    fi
  done
  bench_cell "$CONTROL_N" "$CTL"
  echo "bench: $n cells in $(($(date +%s) - t0))s -> $CSV" >&2
  echo "  control N=$CONTROL_N every $CONTROL_EVERY cells -> $CTL (rows are in run order)" >&2
fi
