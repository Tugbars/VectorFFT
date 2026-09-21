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
# RERUN (2026-09-21): `rerun` re-races every listed cell on the SAME store
# (recal = 1: the race runs again and re-banks over the old verdict), replaces
# the cell's calibrate.log line, then benches both flips; the bench REPLACES
# the cell's csv rows in place (k1z_csv_replace), so gauntlet.csv keeps one
# row per (cell, flip) and every untouched cell keeps its original rows. A
# control pair is taken before and after, into control.csv, in run order.
#
# PLACE (2026-09-21): PLACE=ip runs the same cells IN PLACE, natural order --
# probe ip=1, bench --k1nat (DFTI_INPLACE on MKL's side too) -- into its OWN
# files (gauntlet_ip.csv, control_ip.csv, calibrate_ip.log) on the SAME store,
# so one merge later carries both contracts. Use `resume` so the store with
# the out-of-place verdicts is kept (calibrate/both wipe it).
#
# THREADS (2026-09-21): THREADS=8 runs the same cells at nthreads = 8 -- the
# probe creates at T (the door's per-T race: the flat DIT's, ZTURN-T's and the
# four-step's threaded arms against the serial verdict, banked as per-T tokens on
# the cell's own row; pair / chain3 / prime / mono have no threaded arm and
# replay serial at once), the bench runs --k1noop --mt with VFFT_MT=8 (our arm at
# T on the snapshot pool, MKL at T -- which is serial below 8192 by its own
# rule) -- into its OWN files (gauntlet_mt8.csv, control_mt8.csv,
# calibrate_mt8.log) on the SAME store. `resume` keeps the store.
#
# Run from build_tuned/, machine QUIET:
#   sh gauntlet_1d.sh <out-dir> <cell-list> [calibrate|bench|both|resume|rerun|retime]
#   PLACE=ip sh gauntlet_1d.sh <out-dir> <cell-list> resume
#   THREADS=8 sh gauntlet_1d.sh <out-dir> <cell-list> resume
set -u
OUT="${1:?out-dir}"
LIST="${2:?cell-list file}"
PHASE="${3:-both}"
CONTROL_N=${CONTROL_N:-4096}
CONTROL_EVERY=${CONTROL_EVERY:-100}
HERE="$(cd "$(dirname "$0")" && pwd)"
G="$HERE/../src/dag-fft-compiler/generator/generated"
ST="$OUT/store"
PLACE="${PLACE:-oop}"
THREADS="${THREADS:-1}"
if [ "$PLACE" = ip ]; then
  SFX="_ip"; IP=1; K1FLAG=--k1nat
else
  SFX=""; IP=0; K1FLAG=--k1noop
fi
if [ "$THREADS" -gt 1 ]; then
  SFX="${SFX}_mt$THREADS"; K1FLAG="$K1FLAG --mt"; export VFFT_MT="$THREADS"
fi
CSV="$OUT/gauntlet$SFX.csv"
CTL="$OUT/control$SFX.csv"
CAL="$OUT/calibrate$SFX.log"
mkdir -p "$OUT"
export VFFT_WISDOM_DIR="$ST"

cells() { grep -vE '^\s*(#|$)' "$LIST" | awk '{print $1}'; }

bench_cell() {   # $1 = N, $2 = csv path
  for F in 0 1; do
    "$HERE/benches/bench_1d_vs_mkl.exe" $K1FLAG "$ST/spike_wisdom.txt" "$2" \
        300 "$1" 1 300 "$F" 2 > /dev/null 2>&1   # K1FLAG may be two words (--k1noop --mt)
  done
}
# The control file is a TIME SERIES of one cell, so its rows must ACCUMULATE:
# the bench replaces a (cell, flip) row in place (2026-09-21), which is right
# for gauntlet.csv and wrong here -- the in-place run's control_ip.csv kept
# only its last pair. Bench into a scratch csv, then append the rows.
control_cell() {   # $1 = N, $2 = control csv path
  rm -f "$OUT/.ctl.tmp"
  bench_cell "$1" "$OUT/.ctl.tmp"
  [ -f "$2" ] || head -1 "$OUT/.ctl.tmp" > "$2"
  tail -n +2 "$OUT/.ctl.tmp" >> "$2"
  rm -f "$OUT/.ctl.tmp"
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
    "$HERE/benches/recal_1d_probe.exe" "$ST" "$N" 0 "$IP" "$THREADS" 0 > "$OUT/.cal.tmp" 2>&1
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

if [ "$PHASE" = rerun ]; then
  n=0; t0=$(date +%s)
  control_cell "$CONTROL_N" "$CTL"
  for N in $(cells); do
    s0=$(date +%s%3N)
    "$HERE/benches/recal_1d_probe.exe" "$ST" "$N" 0 "$IP" "$THREADS" 1 > "$OUT/.cal.tmp" 2>&1
    s1=$(date +%s%3N)
    st=$(grep -oE 'banked|REFUSED' "$OUT/.cal.tmp" | tail -1)
    grep -vE "^$N " "$CAL" > "$OUT/.cal.new" 2>/dev/null; mv "$OUT/.cal.new" "$CAL"
    printf "%-10s %-8s %s rerun\n" "$N" "${st:-ERROR}" "$((s1 - s0))ms" >> "$CAL"
    [ "$st" = banked ] && bench_cell "$N" "$CSV"
    n=$((n + 1))
    [ $((n % 25)) -eq 0 ] && echo "  rerun $n cells, $(($(date +%s) - t0))s elapsed" >&2
  done
  rm -f "$OUT/.cal.tmp"
  control_cell "$CONTROL_N" "$CTL"
  echo "rerun: $n cells in $(($(date +%s) - t0))s -> $CSV (rows replaced in place)" >&2
fi

if [ "$PHASE" = retime ]; then
  # RETIME (2026-09-21): bench only, the banked plans as they stand (no race),
  # rows replaced in place -- for a bench-protocol change (the sibling guard,
  # the two timing windows) that changes what is MEASURED, not what is planned.
  n=0; t0=$(date +%s)
  control_cell "$CONTROL_N" "$CTL"
  for N in $(cells); do
    bench_cell "$N" "$CSV"
    n=$((n + 1))
    [ $((n % 25)) -eq 0 ] && echo "  retime $n cells, $(($(date +%s) - t0))s elapsed" >&2
  done
  control_cell "$CONTROL_N" "$CTL"
  echo "retime: $n cells in $(($(date +%s) - t0))s -> $CSV (rows replaced in place)" >&2
fi

if [ "$PHASE" = bench ] || [ "$PHASE" = both ] || [ "$PHASE" = resume ]; then
  rm -f "$CSV" "$CTL"
  "$HERE/benches/recal_1d_probe.exe" "$ST" "$CONTROL_N" 0 "$IP" "$THREADS" 0 > /dev/null 2>&1
  control_cell "$CONTROL_N" "$CTL"
  n=0
  t0=$(date +%s)
  for N in $(cells); do
    grep -qE "^$N +banked" "$CAL" 2>/dev/null || continue   # never bench a refused cell
    bench_cell "$N" "$CSV"
    n=$((n + 1))
    if [ $((n % CONTROL_EVERY)) -eq 0 ]; then
      control_cell "$CONTROL_N" "$CTL"
      echo "  benched $n cells, $(($(date +%s) - t0))s elapsed (control taken)" >&2
    fi
  done
  control_cell "$CONTROL_N" "$CTL"
  echo "bench: $n cells in $(($(date +%s) - t0))s -> $CSV" >&2
  echo "  control N=$CONTROL_N every $CONTROL_EVERY cells -> $CTL (rows are in run order)" >&2
fi
