#!/bin/bash
# bootstrap.sh — wire the full DAG-FFT machinery in one go.
#
# Quake-III-style boot log: each phase announces what it's doing, posts
# real-time progress, ends with a summary. Phases are cheap if the cache
# is hot; a fresh full run on a 32-thread desktop takes ~3 minutes.
#
# Phases (each can be skipped):
#   1. Codelet regen           — scripts/generate_codelets.sh (slow)
#   2. Auto-emitted headers    — radix_profile, plan_executors, registry
#   3. Build cost-model tools  — measure_cpe, score_and_time_plans
#
# Usage:
#   bash scripts/bootstrap.sh                  # default: all phases
#   SKIP_CODELETS=1 bash scripts/bootstrap.sh  # use existing codelet tree
#   SKIP_HEADERS=1  bash scripts/bootstrap.sh  # use existing headers
#   SKIP_BUILD=1    bash scripts/bootstrap.sh  # use existing binaries
#   CLEAN=1         bash scripts/bootstrap.sh  # force full rebuild
#
# Environment passthrough:
#   ISA=avx2|avx512  CC=<compiler>  NJOBS=<int>  (forwarded to children)
#
set -e

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

# ── ANSI color setup ────────────────────────────────────────────────────
if [ -t 1 ]; then
  C_RESET="\033[0m"
  C_BOLD="\033[1m"
  C_DIM="\033[2m"
  C_RED="\033[31m"
  C_GREEN="\033[32m"
  C_YELLOW="\033[33m"
  C_BLUE="\033[34m"
  C_CYAN="\033[36m"
  C_GRAY="\033[90m"
else
  C_RESET="" C_BOLD="" C_DIM="" C_RED="" C_GREEN="" C_YELLOW="" C_BLUE="" C_CYAN="" C_GRAY=""
fi

# ── Helpers ─────────────────────────────────────────────────────────────
now_s() { date +%s.%N; }
fmt_secs() { printf "%5.1fs" "$1"; }

elapsed() {
  local t0=$1
  local t1=$(now_s)
  awk "BEGIN { printf \"%.1f\", $t1 - $t0 }"
}

banner_top() {
  echo -e "${C_BOLD}${C_BLUE}═══════════════════════════════════════════════════════════════════════════════${C_RESET}"
  echo -e "${C_BOLD}${C_CYAN}  ▓▒░  V F F T _ V 2   B O O T S T R A P   ░▒▓${C_RESET}"
  echo -e "${C_BOLD}${C_BLUE}═══════════════════════════════════════════════════════════════════════════════${C_RESET}"
  echo
}

phase_header() {
  echo
  echo -e "${C_BOLD}${C_YELLOW}━━━━━━━━━━━━ Phase $1: $2 ━━━━━━━━━━━━${C_RESET}"
  echo
}

ok()   { echo -e "  ${C_GREEN}[ ok ]${C_RESET} $1"; }
skip() { echo -e "  ${C_GRAY}[skip]${C_RESET} ${C_DIM}$1${C_RESET}"; }
warn() { echo -e "  ${C_YELLOW}[warn]${C_RESET} $1"; }
fail() { echo -e "  ${C_RED}[fail]${C_RESET} $1"; }

# Print a status line with right-aligned timing
status_line() {
  local label="$1"
  local detail="$2"
  local timing="$3"
  printf "  ${C_GREEN}[ ok ]${C_RESET} %-22s %-40s ${C_DIM}%s${C_RESET}\n" "$label" "$detail" "$timing"
}

# ── [init] block ────────────────────────────────────────────────────────
banner_top

# Host
host_os=$(uname -srm 2>/dev/null || echo unknown)
host_cpu=$(awk -F: '/^model name/ {gsub(/^[ \t]+/, "", $2); print $2; exit}' /proc/cpuinfo 2>/dev/null || echo unknown)
host_threads=$(nproc 2>/dev/null || echo unknown)

# Toolchain
cc=${CC:-gcc-15}
cc_version=$($cc --version 2>/dev/null | head -1 | sed -E 's/.*\(.*\) //; s/ Copyright.*//' || echo unknown)
ocaml_version=$(opam config var ocaml-version 2>/dev/null \
                || ocaml --version 2>/dev/null | awk '{print $NF}' \
                || echo unknown)

# Build params
isa=${ISA:-avx2}
njobs=${NJOBS:-$(nproc 2>/dev/null || echo 4)}

printf "${C_DIM}[init]${C_RESET} %-12s %s\n" "Host:"      "$host_os"
printf "${C_DIM}[init]${C_RESET} %-12s %s\n" "CPU:"       "$host_cpu ($host_threads threads)"
printf "${C_DIM}[init]${C_RESET} %-12s %s\n" "Compiler:"  "$cc $cc_version"
printf "${C_DIM}[init]${C_RESET} %-12s %s\n" "OCaml:"     "$ocaml_version"
printf "${C_DIM}[init]${C_RESET} %-12s %s\n" "ISA:"       "$isa"
printf "${C_DIM}[init]${C_RESET} %-12s %s\n" "NJOBS:"     "$njobs"
printf "${C_DIM}[init]${C_RESET} %-12s %s\n" "Repo:"      "$ROOT"

T_TOTAL_START=$(now_s)

# Make sure OCaml binaries are built (silent, idempotent).
echo
echo -e "${C_DIM}[init] ensuring OCaml binaries are built...${C_RESET}"
T_DUNE=$(now_s)
eval "$(opam env --switch=5.2.0 --set-switch 2>/dev/null)" 2>/dev/null || true
if dune build 2>&1 | tail -3; then
  dune_t=$(elapsed $T_DUNE)
  ok "dune build  (${dune_t}s)"
else
  fail "dune build"
  exit 1
fi

# ── Phase 1: codelet regen ──────────────────────────────────────────────
if [ "${SKIP_CODELETS:-0}" = "1" ]; then
  phase_header 1 "codelet generation"
  skip "SKIP_CODELETS=1 — using existing tree at codelets/"
  if [ -d "$ROOT/codelets" ]; then
    count=$(find $ROOT/codelets -name 'r*.c' 2>/dev/null | wc -l)
    ok "found $count existing codelets"
  fi
else
  phase_header 1 "codelet generation"
  T_PHASE1=$(now_s)
  echo -e "${C_DIM}  invoking generate_codelets.sh ISA=both ...${C_RESET}"
  echo
  # Run with M-active env vars (per session-established convention)
  if VFFT_USE_REGALLOC=1 VFFT_USE_REGALLOC_M5=1 ISA=both \
       bash scripts/generate_codelets.sh 2>&1 \
       | sed -E "s/^/    ${C_DIM}/; s/$/${C_RESET}/" \
       | tail -25
  then
    P1_T=$(elapsed $T_PHASE1)
    total_codelets=$(find $ROOT/codelets -name 'r*.c' | wc -l)
    echo
    status_line "gen_radix TOTAL"  "$total_codelets codelets emitted"  "${P1_T}s"
  else
    fail "generate_codelets.sh failed"
    exit 1
  fi
fi

# ── Phase 2: auto-emit headers ──────────────────────────────────────────
phase_header 2 "auto-emit headers"
if [ "${SKIP_HEADERS:-0}" = "1" ]; then
  skip "SKIP_HEADERS=1 — using existing generated/ tree"
  ls generated/*.h cost_model/generated/*.h 2>/dev/null | while read f; do
    lines=$(wc -l < "$f" 2>/dev/null || echo 0)
    rel="${f#$ROOT/}"
    printf "  ${C_GRAY}[skip]${C_RESET} %-44s ${C_DIM}%5d lines${C_RESET}\n" "$rel" "$lines"
  done
else
  mkdir -p generated cost_model/generated

  # 2a. radix_profile.h (both ISAs in one file)
  T=$(now_s)
  ./_build/default/bin/emit_profile_h.exe --isa both > cost_model/generated/radix_profile.h
  lines=$(wc -l < cost_model/generated/radix_profile.h)
  status_line "emit_profile_h"  "cost_model/generated/radix_profile.h ($lines lines)"  "$(elapsed $T)s"

  # 2b. plan_executors.h (avx2)
  T=$(now_s)
  ./_build/default/bin/emit_executor_h.exe --isa avx2 > generated/plan_executors.h
  lines=$(wc -l < generated/plan_executors.h)
  status_line "emit_executor_h"  "generated/plan_executors.h ($lines lines)"  "$(elapsed $T)s"

  # 2c. registry_avx2.h
  T=$(now_s)
  ./_build/default/bin/emit_registry_h.exe --isa avx2 > generated/registry_avx2.h
  lines=$(wc -l < generated/registry_avx2.h)
  externs=$(grep -c "^extern void radix" generated/registry_avx2.h)
  status_line "emit_registry_h avx2"  "$externs externs, $lines lines"  "$(elapsed $T)s"

  # 2d. registry_avx512.h
  T=$(now_s)
  ./_build/default/bin/emit_registry_h.exe --isa avx512 > generated/registry_avx512.h
  lines=$(wc -l < generated/registry_avx512.h)
  externs=$(grep -c "^extern void radix" generated/registry_avx512.h)
  status_line "emit_registry_h avx512"  "$externs externs, $lines lines"  "$(elapsed $T)s"

  # 2e. dispatcher registry.h is hand-written — sanity check it exists
  if [ -f generated/registry.h ]; then
    lines=$(wc -l < generated/registry.h)
    status_line "registry.h (manual)"  "ISA-dispatcher present ($lines lines)"  "  cached"
  else
    warn "generated/registry.h missing — hand-written ISA dispatcher should be in tree"
  fi
fi

# ── Phase 3: build cost-model tools ─────────────────────────────────────
phase_header 3 "build cost-model tools"
if [ "${SKIP_BUILD:-0}" = "1" ]; then
  skip "SKIP_BUILD=1 — using existing binaries in build_tuned/"
  for b in measure_cpe score_and_time_plans; do
    if [ -x "$ROOT/build_tuned/$b" ]; then
      sz=$(du -h "$ROOT/build_tuned/$b" | awk '{print $1}')
      status_line "$b"  "$sz binary (existing)"  "  cached"
    fi
  done
else
  CLEAN_ARG=""
  [ "${CLEAN:-0}" = "1" ] && CLEAN_ARG="CLEAN=1 "

  # 3a. measure_cpe (parallel + cached + R=1024 stubbed)
  T=$(now_s)
  echo -e "${C_DIM}  building measure_cpe (parallel, NJOBS=$njobs)...${C_RESET}"
  if eval "$CLEAN_ARG bash cost_model/build_measure_cpe.sh" 2>&1 \
       | sed -E "s/^/    ${C_DIM}/; s/$/${C_RESET}/" \
       | tail -6; then
    sz=$(du -h build_tuned/measure_cpe | awk '{print $1}')
    status_line "measure_cpe"  "$sz binary"  "$(elapsed $T)s"
  else
    fail "build_measure_cpe.sh failed"
    exit 1
  fi

  # 3b. score_and_time_plans (smaller, fast)
  T=$(now_s)
  echo -e "${C_DIM}  building score_and_time_plans...${C_RESET}"
  if bash cost_model/build_score_and_time.sh 2>&1 \
       | sed -E "s/^/    ${C_DIM}/; s/$/${C_RESET}/" \
       | tail -4; then
    sz=$(du -h build_tuned/score_and_time_plans | awk '{print $1}')
    status_line "score_and_time_plans"  "$sz binary"  "$(elapsed $T)s"
  else
    fail "build_score_and_time.sh failed"
    exit 1
  fi
fi

# ── Final banner ────────────────────────────────────────────────────────
TOTAL=$(elapsed $T_TOTAL_START)
echo
echo -e "${C_BOLD}${C_BLUE}═══════════════════════════════════════════════════════════════════════════════${C_RESET}"
echo -e "${C_BOLD}${C_GREEN}  ▶ All systems initialized.${C_RESET}  ${C_DIM}Total: ${TOTAL}s${C_RESET}"
echo -e "${C_BOLD}${C_BLUE}═══════════════════════════════════════════════════════════════════════════════${C_RESET}"
echo
echo -e "${C_BOLD}Next steps:${C_RESET}"
echo -e "  ${C_CYAN}•${C_RESET} ${C_DIM}calibrate CPE on a quiet host:${C_RESET}"
echo -e "      ${C_BOLD}taskset -c 2 build_tuned/measure_cpe${C_RESET}  ${C_GRAY}# ~10 min, --force to overwrite${C_RESET}"
echo -e "  ${C_CYAN}•${C_RESET} ${C_DIM}validate cost-model ranking:${C_RESET}"
echo -e "      ${C_BOLD}taskset -c 2 build_tuned/score_and_time_plans${C_RESET}  ${C_GRAY}# ~5 min${C_RESET}"
echo -e "  ${C_CYAN}•${C_RESET} ${C_DIM}re-run this script with SKIP_CODELETS=1 to skip the slow regen:${C_RESET}"
echo -e "      ${C_BOLD}SKIP_CODELETS=1 bash scripts/bootstrap.sh${C_RESET}"
echo
