#!/bin/bash
# preflight.sh — live prelaunch screen for the measure tier (section 40).
#
# Quake-3-prelaunch-style console: a full-screen terminal dashboard that
# refreshes once per second showing everything that makes CPE /
# memboundness measurements unfair — system load, the top CPU consumers
# by name, the frequency governor, the SMT sibling of the pinned core,
# thermal state — with a live READY / WAIT verdict. The user closes
# programs and WATCHES the machine go quiet, then launches.
#
# Keys:  [enter] launch now   [a] auto-launch when ready (3s countdown)
#        [q] abort
#
# Non-interactive: --yes (or no TTY + --yes) prints one snapshot and
# proceeds; no TTY without --yes aborts with instructions. Either way a
# preflight snapshot sidecar is written next to the measurement output
# (measurement provenance, same philosophy as the codelet stamps).
#
# Usage: preflight.sh <pinned_cpu> <output_dir> [--yes]
set -u
CPU="${1:?pinned cpu}"
OUTDIR="${2:?output dir}"
YES=0; [ "${3:-}" = "--yes" ] && YES=1

C0=$'\033[0m'; CB=$'\033[1m'; CG=$'\033[32m'; CY=$'\033[33m'; CR=$'\033[31m'; CD=$'\033[2m'; CC=$'\033[36m'

# ── probes (each degrades to n/a where the host lacks the interface) ──
p_load()    { cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || echo "n/a"; }
p_runq()    { cut -d' ' -f4 /proc/loadavg 2>/dev/null | cut -d/ -f1 || echo "?"; }
p_ncpu()    { nproc 2>/dev/null || echo 1; }
p_gov()     { cat "/sys/devices/system/cpu/cpu${CPU}/cpufreq/scaling_governor" 2>/dev/null || echo "n/a"; }
p_sibling() {
  local s
  s=$(cat "/sys/devices/system/cpu/cpu${CPU}/topology/thread_siblings_list" 2>/dev/null) || { echo ""; return; }
  echo "$s" | tr ',-' '\n\n' | grep -vx "$CPU" | head -1
}
p_cpu_busy() { # busy% of cpu $1 over ~0.2s
  local a b ia ib
  a=$(grep "^cpu$1 " /proc/stat 2>/dev/null) || { echo "?"; return; }
  sleep 0.2
  b=$(grep "^cpu$1 " /proc/stat)
  ia=$(echo "$a" | awk '{print $5+$6}'); sa=$(echo "$a" | awk '{s=0;for(i=2;i<=NF;i++)s+=$i;print s}')
  ib=$(echo "$b" | awk '{print $5+$6}'); sb=$(echo "$b" | awk '{s=0;for(i=2;i<=NF;i++)s+=$i;print s}')
  local dt=$((sb-sa)); [ "$dt" -le 0 ] && { echo 0; return; }
  echo $(( (dt-(ib-ia))*100/dt ))
}
p_top3()    { ps -eo pcpu,comm --no-headers --sort=-pcpu 2>/dev/null | awk '$1>0.5 && $2!="ps"' | head -3 | awk '{printf "%s %.0f%%  ", $2, $1}'; }
p_temp()    {
  local t
  t=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -rn | head -1) || true
  [ -n "${t:-}" ] && echo "$((t/1000))C" || echo "n/a"
}

verdict() { # sets V_OK plus per-check marks
  local load runq ncpu gov sib sibbusy
  load=$(p_load | cut -d' ' -f1); runq=$(p_runq); ncpu=$(p_ncpu)
  gov=$(p_gov); sib=$(p_sibling)
  M_LOAD="${CG}ok${C0}"; M_GOV="${CG}ok${C0}"; M_SIB="${CG}ok${C0}"; V_OK=1
  # 1-minute load is the quiet-machine signal; instantaneous runq is
  # jitter (it flapped the verdict in pty testing) and stays display-only.
  awk "BEGIN{exit !($load > $ncpu*0.25)}" && { M_LOAD="${CY}busy${C0}"; V_OK=0; }
  : "$runq"
  case "$gov" in performance|n/a) ;; *) M_GOV="${CY}$gov${C0}"; V_OK=0;; esac
  SIBBUSY="-"
  if [ -n "$sib" ]; then
    sibbusy=$(p_cpu_busy "$sib"); SIBBUSY="cpu$sib ${sibbusy}%"
    [ "$sibbusy" != "?" ] && [ "$sibbusy" -gt 10 ] && { M_SIB="${CY}${sibbusy}%${C0}"; V_OK=0; }
  fi
}

snapshot() {
  mkdir -p "$OUTDIR"
  {
    echo "# measurement preflight snapshot (section 40)"
    echo "timestamp: $(date -Is)"
    echo "pinned_cpu: $CPU"
    echo "load: $(p_load)"
    echo "governor: $(p_gov)"
    echo "smt_sibling: $(p_sibling || true) busy: ${SIBBUSY:-n/a}"
    echo "temp_max: $(p_temp)"
    echo "top_consumers: $(p_top3)"
    echo "host: $(uname -srm 2>/dev/null)"
  } > "$OUTDIR/preflight_snapshot.txt"
}

# ── non-interactive paths ─────────────────────────────────────────────
if [ ! -t 1 ] || [ "$YES" = 1 ]; then
  verdict
  echo "[preflight] load=$(p_load) governor=$(p_gov) sibling=${SIBBUSY} temp=$(p_temp)"
  if [ "$YES" = 1 ]; then snapshot; exit 0; fi
  echo "[preflight] no TTY: pass --yes to proceed non-interactively"
  exit 1
fi

# ── GUI dispatch (section 40b) ────────────────────────────────────────
# A display + the built raylib binary = the literal-window prelaunch
# (WSLg puts it on the Windows desktop). PREFLIGHT_TUI=1 forces the
# terminal screen; no display or no binary falls through to it.
GUI_BIN="$(cd "$(dirname "$0")/.." && pwd)/build_tuned/preflight_gui"
if [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ] && [ -x "$GUI_BIN" ] \
   && [ "${PREFLIGHT_TUI:-0}" != "1" ]; then
  if "$GUI_BIN" "$CPU" "$OUTDIR"; then
    echo "[preflight] launched (gui); snapshot in $OUTDIR/preflight_snapshot.txt"
    exit 0
  else
    exit 1
  fi
fi

# ── the live screen ───────────────────────────────────────────────────
tput smcup; tput civis
trap 'tput cnorm; tput rmcup' EXIT
AUTO=0; COUNT=-1; RESULT=1

while :; do
  verdict
  tput home
  printf "%s\n" "${CB}${CC}═══════════════════════════════════════════════════════════════${C0}"
  printf "%s\n" "${CB}${CC}  ▓▒░  V F F T   M E A S U R E M E N T   P R E L A U N C H  ░▒▓${C0}"
  printf "%s\n" "${CB}${CC}═══════════════════════════════════════════════════════════════${C0}"
  printf "\n"
  printf "  %-18s %-32s [%b]\033[K\n" "load (1/5/15m)"  "$(p_load)"   "$M_LOAD"
  printf "  %-18s %-32s\033[K\n"      "top consumers"   "$(p_top3 | cut -c1-44)"
  printf "  %-18s %-32s [%b]\033[K\n" "governor cpu$CPU" "$(p_gov)"   "$M_GOV"
  printf "  %-18s %-32s [%b]\033[K\n" "smt sibling"      "${SIBBUSY}" "$M_SIB"
  printf "  %-18s %-32s\033[K\n"      "max temp"         "$(p_temp)"
  printf "\n"
  if [ "$V_OK" = 1 ]; then
    if [ "$AUTO" = 1 ]; then
      [ "$COUNT" -lt 0 ] && COUNT=3
      printf "  %b\033[K\n" "${CG}${CB}READY — launching in ${COUNT}...${C0}"
      [ "$COUNT" -eq 0 ] && { RESULT=0; break; }
      COUNT=$((COUNT-1))
    else
      printf "  %b\033[K\n" "${CG}${CB}READY${C0} ${CD}— machine is quiet${C0}"
    fi
  else
    COUNT=-1
    printf "  %b\033[K\n" "${CY}${CB}WAIT${C0} ${CD}— close marked items, screen updates live${C0}"
  fi
  printf "\n  %b\033[K\n" "${CD}[enter] launch now   [a] auto-launch when ready   [q] abort${C0}"
  IFS= read -rsn1 -t 1 key; rc=$?
  if [ $rc -ne 0 ]; then
    key=""
    # rc>128 = the 1s timeout (the normal tick). Anything else is EOF
    # (headless pty): sleep explicitly so the loop still ticks at 1Hz.
    [ $rc -le 128 ] && sleep 1
  fi
  case "$key" in
    "")  [ "$AUTO" = 1 ] && continue || continue ;;
    a|A) AUTO=1 ;;
    q|Q) RESULT=1; break ;;
    *)   RESULT=0; break ;;   # enter / any other key = launch now
  esac
done
tput cnorm; tput rmcup; trap - EXIT
if [ "$RESULT" = 0 ]; then
  snapshot
  echo "[preflight] launched; snapshot written to $OUTDIR/preflight_snapshot.txt"
fi
exit $RESULT
