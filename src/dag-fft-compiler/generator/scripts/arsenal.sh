#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# arsenal.sh — the one entry point, tiered.
#
#   bash arsenal.sh codelets [quadrants...]   committed codelet tree only (fast)
#   bash arsenal.sh tools                     scratch tree + headers + cost-model binaries
#   bash arsenal.sh measure                   RUN the long CPE / memboundness measurements
#   bash arsenal.sh all                       everything, in order (default)
#
# The fast path is also available directly: regen_codelets.sh.
# This wrapper exists for the full arsenal; nobody regenerating
# codelets has to touch the long measurement pipeline, and nobody
# calibrating the cost model has to remember four scripts.
#
# Env: GEN (generator path), CC (compiler, bootstrap default gcc-15),
#      CPU (core to pin measurements to, default 4), DRY=1 (print only).
# ═══════════════════════════════════════════════════════════════════
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
GENROOT=$(cd "$HERE/.." && pwd)
CPU="${CPU:-4}"
run() { if [ "${DRY:-0}" = "1" ]; then echo "DRY: $*"; else "$@"; fi }

stage_codelets() {
  echo "━━ stage: codelets (committed tree)"
  run bash "$HERE/regen_codelets.sh" "$@"
}

stage_tools() {
  echo "━━ stage: tools (scratch tree, headers, cost-model binaries)"
  run bash "$HERE/bootstrap.sh"
  # bootstrap builds measure_cpe + score_and_time; memboundness is ours:
  run bash "$HERE/build_measure_memboundness.sh"
}

stage_measure() {
  echo "━━ stage: measure (LONG; pin to an isolated core, quiet machine)"
  echo "   pinning to CPU $CPU (override with CPU=n); output: $GENROOT/build_tuned/"
  # Q3-style live prelaunch screen (section 40): READY/WAIT verdict on
  # load, governor, SMT sibling; writes a preflight snapshot sidecar.
  # Non-interactive runs: YES=1 arsenal.sh measure (or no-TTY + YES=1).
  if [ "${DRY:-0}" != "1" ]; then
    bash "$HERE/preflight.sh" "$CPU" "$GENROOT/build_tuned" \
      $([ "${YES:-0}" = "1" ] && echo --yes) || {
        echo "   aborted at preflight"; return 1; }
  fi
  for bin in measure_cpe measure_memboundness; do
    if [ -x "$GENROOT/build_tuned/$bin" ]; then
      run taskset -c "$CPU" "$GENROOT/build_tuned/$bin"
    else
      echo "   missing $GENROOT/build_tuned/$bin — run the tools stage first"
      return 1
    fi
  done
}

TARGET="${1:-all}"; shift 2>/dev/null || true
case "$TARGET" in
  codelets) stage_codelets "$@" ;;
  tools)    stage_tools ;;
  measure)  stage_measure ;;
  all)      stage_codelets && stage_tools && stage_measure ;;
  *) echo "usage: arsenal.sh [codelets [quadrants...]|tools|measure|all]"; exit 1 ;;
esac
