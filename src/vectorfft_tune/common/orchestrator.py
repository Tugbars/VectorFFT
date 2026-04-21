#!/usr/bin/env python3
"""
common/orchestrator.py — run all radix benches in one sweep.

Discovers radix directories under radixes/, runs each through common/bench.py
sequentially, handles failures per-radix, generates bench_out/ALL_SUMMARY.md.

Benchmark hygiene:
  - CPU affinity via wrapper (taskset on Linux, start /AFFINITY on Windows)
  - Pre-flight check: performance governor (Linux) or high-perf power plan (Windows)
  - Thermal pacing between radixes
  - Variance reporting deferred (future work item)

Config file: orchestrator.json in project root (next to common/). Missing fields
fall through to CLI defaults. CLI args override config overrides builtins.

Methodology follows FFTW-style timing (min of 8 blocks, 10ms each, 2s cap), with
external affinity wrapper matching the benchFFT script pattern. The variance-
reporting extension and pre-flight governor check go beyond FFTW's own tooling.
"""
from __future__ import annotations
import argparse
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ────────────────────── project layout ──────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCH_PY     = PROJECT_ROOT / 'common' / 'bench.py'
RADIXES_DIR  = PROJECT_ROOT / 'radixes'
BENCH_OUT    = PROJECT_ROOT / 'bench_out'
CONFIG_FILE_DEFAULT = PROJECT_ROOT / 'orchestrator.json'


# ────────────────────── defaults ──────────────────────
# The default CPU index (2) targets the first "clean" P-core on most consumer
# Intel configurations: CPU 0 handles most interrupts on Linux, CPU 1 is its
# SMT sibling. CPU 2 is typically on P-core 1 (a clean P-core with its own
# cache hierarchy). For other hardware (AMD, older Intel, servers, or if the
# default produces surprising results), override via --cpu or config file.
# Try different cores during calibration runs to find the best one for your chip.

DEFAULTS = {
    'cpu':              2,
    'pace_seconds':     2,
    'phase':            'all',
    'radix':            None,     # None means all radixes
    'skip':             None,
    'force':            False,
    'fail_fast':        False,
    'quiet':            False,
    'dry_run':          False,
    'no_summary':       False,
    'auto_performance': False,    # opt-in: switch to high-perf plan, restore on exit
}


# ────────────────────── data types ──────────────────────

@dataclass
class RadixRun:
    name: str                    # "r10"
    radix: int                   # 10
    dir_path: Path
    status: str = 'pending'      # pending|success|failure|skipped
    reason: Optional[str] = None
    elapsed_s: float = 0.0
    measurements: int = 0
    validation_cases: int = 0
    validation_failed: int = 0


@dataclass
class SweepReport:
    runs: list[RadixRun] = field(default_factory=list)
    overall_elapsed_s: float = 0.0
    host_info: dict = field(default_factory=dict)


# ────────────────────── discovery ──────────────────────

_RE_RADIX_DIR = re.compile(r'^r(\d+)$')

def discover_radixes() -> list[RadixRun]:
    """Walk radixes/*/candidates.py, return RadixRun stubs sorted by radix."""
    if not RADIXES_DIR.is_dir():
        raise SystemExit(f"orchestrator: radixes dir not found: {RADIXES_DIR}")
    found = []
    for p in sorted(RADIXES_DIR.iterdir()):
        if not p.is_dir():
            continue
        m = _RE_RADIX_DIR.match(p.name)
        if not m:
            continue
        cand = p / 'candidates.py'
        if not cand.is_file():
            # directory named rN but no candidates.py; not a tunable radix
            continue
        found.append(RadixRun(name=p.name, radix=int(m.group(1)), dir_path=p))
    found.sort(key=lambda r: r.radix)
    return found


def filter_radixes(all_runs: list[RadixRun], include: Optional[str],
                   skip: Optional[str]) -> list[RadixRun]:
    if include and skip:
        raise SystemExit("orchestrator: --radix and --skip are mutually exclusive")
    if include:
        names = {s.strip() for s in include.split(',') if s.strip()}
        result = [r for r in all_runs if r.name in names]
        missing = names - {r.name for r in result}
        if missing:
            raise SystemExit(f"orchestrator: requested radixes not discovered: {sorted(missing)}")
        return result
    if skip:
        names = {s.strip() for s in skip.split(',') if s.strip()}
        return [r for r in all_runs if r.name not in names]
    return all_runs


# ────────────────────── pre-flight ──────────────────────

def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding='utf-8').strip()
    except Exception:
        return None


def preflight_linux() -> tuple[bool, str]:
    """Return (ok, human message)."""
    # Check a few CPUs; all should be "performance" for confidence
    gov_files = sorted(Path('/sys/devices/system/cpu').glob('cpu[0-9]*/cpufreq/scaling_governor'))
    if not gov_files:
        return False, "no /sys/devices/system/cpu/*/cpufreq/scaling_governor files found; cannot verify CPU governor"
    governors = {}
    for g in gov_files[:8]:  # sample first 8 CPUs; assume homogeneous
        val = _read_text(g)
        if val is not None:
            governors[g.parts[-3]] = val
    if not governors:
        return False, "could not read any scaling_governor files"
    non_perf = {cpu: gov for cpu, gov in governors.items() if gov != 'performance'}
    if non_perf:
        sample = dict(list(non_perf.items())[:3])
        return False, (
            f"CPU governor is not 'performance' on: {sample}. "
            f"Fix with: sudo cpupower frequency-set -g performance  "
            f"(or write 'performance' to /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor). "
            f"Run with --force to bypass."
        )
    return True, f"governor=performance on {len(governors)} CPUs checked"


# GUIDs from Windows powercfg for the canonical Microsoft-distributed plans.
# Duplicated instances (e.g., via `powercfg -duplicatescheme`) get fresh GUIDs,
# so we also fall back to name matching for "high performance" / "ultimate
# performance" as identifiers.
_WIN_HIGHPERF_GUIDS = {
    '8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c': 'High performance',
    'e9a42b02-d5df-448d-aa00-03f14749eb61': 'Ultimate Performance',
}

# Name tokens that identify a high-perf-class plan, in multiple spellings.
# Matched case-insensitively against the plan name reported by powercfg.
# English only; users on other locales may need --force (documented).
_WIN_HIGHPERF_NAME_TOKENS = (
    'high performance',
    'ultimate performance',
)

def preflight_windows() -> tuple[bool, str]:
    try:
        out = subprocess.run(['powercfg', '/getactivescheme'],
                             capture_output=True, text=True, timeout=5)
    except Exception as e:
        return False, f"powercfg failed: {e}"
    if out.returncode != 0:
        return False, f"powercfg returned {out.returncode}: {out.stderr.strip()}"
    text_lower = out.stdout.lower()
    # Primary: canonical Microsoft GUID
    for guid, name in _WIN_HIGHPERF_GUIDS.items():
        if guid in text_lower:
            return True, f"active power plan is '{name}' (canonical GUID {guid})"
    # Fallback: name contains a high-perf token (handles duplicated/vendor plans
    # with non-canonical GUIDs but recognizable names).
    # powercfg output format: "Power Scheme GUID: <guid>  (<name>)"
    for token in _WIN_HIGHPERF_NAME_TOKENS:
        if token in text_lower:
            return True, f"active power plan name matches '{token}' (non-canonical GUID, accepted by name)"
    return False, (
        f"active power plan is not a high-performance scheme. "
        f"Fix with: powercfg /setactive SCHEME_MIN  "
        f"(or enable Ultimate Performance via: "
        f"powercfg -duplicatescheme e9a42b02-d5df-448d-aa00-03f14749eb61). "
        f"If your plan is high-perf but not in English, run with --force. "
        f"Output was: {out.stdout.strip()}"
    )


def preflight_affinity_tool(system: str) -> tuple[bool, str]:
    if system == 'Linux':
        if shutil.which('taskset') is None:
            return False, "'taskset' not found on PATH; install util-linux or disable pinning"
        return True, "taskset available"
    if system == 'Windows':
        # We use ctypes.windll.kernel32.SetProcessAffinityMask for pinning.
        # ctypes is stdlib, so this is a sanity load check.
        try:
            import ctypes
            ctypes.windll.kernel32  # probe
            return True, "ctypes available for SetProcessAffinityMask"
        except Exception as e:
            return False, f"ctypes load failed: {e}"
    return False, f"unsupported OS for affinity pinning: {system}"


def run_preflight(force: bool) -> tuple[bool, list[str]]:
    """Run all pre-flight checks. Return (ok, messages)."""
    messages = []
    ok = True
    system = platform.system()
    if system == 'Linux':
        good, msg = preflight_linux()
    elif system == 'Windows':
        good, msg = preflight_windows()
    else:
        good, msg = True, f"OS={system}: governor check not implemented (proceeding)"
    messages.append(('governor', good, msg))
    ok = ok and good

    good, msg = preflight_affinity_tool(system)
    messages.append(('affinity_tool', good, msg))
    ok = ok and good

    if force:
        return True, messages  # --force ignores all failures
    return ok, messages


# ────────────────────── power-plan auto-switch ──────────────────────
#
# When --auto-performance is on, the orchestrator captures the current
# power plan / governor, forces high-performance for the duration of the
# sweep, and restores on exit. A backup file at bench_out/.power_state_backup
# records the original state so ungraceful exits can be detected and
# recovered on next launch.
#
# Limitations:
#   - SIGKILL, kernel panic, or power loss: backup stays on disk; next run
#     detects it and offers manual restore.
#   - Linux writes /sys/.../scaling_governor, which usually requires root.
#     Without root the switch fails; we warn and proceed without switching
#     rather than abort (the preflight check will still catch the wrong
#     governor and refuse to run).

_BACKUP_FILENAME = '.power_state_backup'
# Canonical High Performance GUID; works on all Windows editions that have
# a default plan set. Consumer Windows ships this built in.
_WIN_HIGHPERF_TARGET_GUID = '8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c'


def _backup_path() -> Path:
    return BENCH_OUT / _BACKUP_FILENAME


def capture_current_plan(system: str) -> Optional[dict]:
    """Return current state as a dict, or None if it can't be determined.
    Shape: {'system': 'Linux'|'Windows', 'value': <serialized state>}
    """
    if system == 'Windows':
        try:
            out = subprocess.run(['powercfg', '/getactivescheme'],
                                 capture_output=True, text=True, timeout=5)
        except Exception:
            return None
        if out.returncode != 0:
            return None
        # Parse: "Power Scheme GUID: <guid>  (<name>)"
        m = re.search(r'GUID:\s*([0-9a-fA-F-]{36})', out.stdout)
        if not m:
            return None
        return {'system': 'Windows', 'guid': m.group(1), 'raw': out.stdout.strip()}

    if system == 'Linux':
        # Capture governors for all CPUs (they may differ).
        gov_files = sorted(Path('/sys/devices/system/cpu').glob(
            'cpu[0-9]*/cpufreq/scaling_governor'))
        if not gov_files:
            return None
        per_cpu = {}
        for g in gov_files:
            val = _read_text(g)
            if val is not None:
                per_cpu[str(g)] = val
        if not per_cpu:
            return None
        return {'system': 'Linux', 'governors': per_cpu}

    return None


def apply_performance_plan(system: str) -> tuple[bool, str]:
    """Switch to high-performance. Return (ok, message)."""
    if system == 'Windows':
        try:
            out = subprocess.run(
                ['powercfg', '/setactive', _WIN_HIGHPERF_TARGET_GUID],
                capture_output=True, text=True, timeout=5)
        except Exception as e:
            return False, f"powercfg /setactive failed: {e}"
        if out.returncode != 0:
            return False, f"powercfg /setactive returned {out.returncode}: {out.stderr.strip()}"
        return True, f"switched to High Performance (GUID {_WIN_HIGHPERF_TARGET_GUID})"

    if system == 'Linux':
        if os.geteuid() != 0:
            return False, ("writing /sys/.../scaling_governor requires root. "
                           "Re-run with sudo, or set governor manually and omit --auto-performance.")
        gov_files = sorted(Path('/sys/devices/system/cpu').glob(
            'cpu[0-9]*/cpufreq/scaling_governor'))
        if not gov_files:
            return False, "no scaling_governor files found"
        errors = []
        for g in gov_files:
            try:
                g.write_text('performance\n')
            except Exception as e:
                errors.append(f"{g}: {e}")
        if errors:
            return False, f"wrote performance to {len(gov_files) - len(errors)} CPUs, "\
                          f"failed on {len(errors)}: {errors[0]}"
        return True, f"set performance governor on {len(gov_files)} CPUs"

    return False, f"auto-performance not supported on {system}"


def restore_plan(state: dict) -> tuple[bool, str]:
    """Restore the captured state. Return (ok, message)."""
    system = state.get('system')
    if system == 'Windows':
        guid = state.get('guid')
        if not guid:
            return False, "backup has no GUID"
        try:
            out = subprocess.run(['powercfg', '/setactive', guid],
                                 capture_output=True, text=True, timeout=5)
        except Exception as e:
            return False, f"powercfg /setactive failed: {e}"
        if out.returncode != 0:
            return False, f"powercfg returned {out.returncode}: {out.stderr.strip()}"
        return True, f"restored power plan GUID {guid}"

    if system == 'Linux':
        governors = state.get('governors', {})
        if not governors:
            return False, "backup has no governor records"
        if os.geteuid() != 0:
            return False, "restoring governor requires root"
        errors = []
        for path_str, gov in governors.items():
            try:
                Path(path_str).write_text(gov + '\n')
            except Exception as e:
                errors.append(f"{path_str}: {e}")
        if errors:
            return False, f"restored {len(governors) - len(errors)} governors, "\
                          f"failed on {len(errors)}: {errors[0]}"
        return True, f"restored {len(governors)} governors"

    return False, f"restore not supported on {system}"


def write_backup(state: dict) -> None:
    BENCH_OUT.mkdir(parents=True, exist_ok=True)
    _backup_path().write_text(json.dumps(state, indent=2), encoding='utf-8')


def delete_backup() -> None:
    p = _backup_path()
    if p.exists():
        try:
            p.unlink()
        except Exception:
            pass


def check_stale_backup(auto_performance: bool) -> None:
    """If a backup exists at startup, prior run ended ungracefully.
    Warn the user and offer to restore."""
    p = _backup_path()
    if not p.exists():
        return
    try:
        state = json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        print(f'[auto-perf] found unreadable backup at {p}; leaving it alone',
              file=sys.stderr)
        return
    print(f'[auto-perf] WARNING: stale power-state backup found at {p}')
    print(f'[auto-perf] A previous run ended ungracefully without restoring '
          f'the system power state.')
    print(f'[auto-perf] Captured state: {state}')
    if not auto_performance:
        print(f'[auto-perf] Re-run with --auto-performance to restore, '
              f'or delete the backup manually: {p}')
        return
    # With --auto-performance, prompt for restore
    try:
        answer = input('[auto-perf] Restore this state now? [y/N]: ').strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = 'n'
    if answer == 'y':
        ok, msg = restore_plan(state)
        marker = 'OK' if ok else 'FAIL'
        print(f'[auto-perf] restore {marker}: {msg}')
        if ok:
            delete_backup()
    else:
        print(f'[auto-perf] skipped restore; backup remains at {p}')


# Global to coordinate cleanup between signal handlers and normal exit.
_AUTO_PERF_ACTIVE = {'state': None, 'done': False}


def _restore_and_mark_done() -> None:
    """Idempotent restore helper used by all exit paths."""
    if _AUTO_PERF_ACTIVE['done']:
        return
    state = _AUTO_PERF_ACTIVE['state']
    if state is None:
        _AUTO_PERF_ACTIVE['done'] = True
        return
    ok, msg = restore_plan(state)
    marker = 'OK' if ok else 'FAIL'
    print(f'[auto-perf] restore on exit {marker}: {msg}', file=sys.stderr)
    if ok:
        delete_backup()
    _AUTO_PERF_ACTIVE['done'] = True


def _signal_restore_handler(signum, frame):
    """Signal handler: restore state, then re-raise the signal at default
    disposition so the process actually exits."""
    print(f'\n[auto-perf] signal {signum} received, restoring power state...',
          file=sys.stderr)
    _restore_and_mark_done()
    # Re-raise at default disposition
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def _is_state_already_highperf(state: dict) -> bool:
    """True if the captured state is already a high-performance plan/governor.
    When true, we skip the apply step (don't lateral-move away from Ultimate
    Performance to High Performance, for example) and only register restore
    tracking for the duration of the sweep."""
    if state.get('system') == 'Windows':
        guid = state.get('guid', '').lower()
        raw = state.get('raw', '').lower()
        # Canonical GUID match
        if guid in (g.lower() for g in _WIN_HIGHPERF_GUIDS):
            return True
        # Name fallback — handles duplicated / vendor plans
        for token in _WIN_HIGHPERF_NAME_TOKENS:
            if token in raw:
                return True
        return False
    if state.get('system') == 'Linux':
        governors = state.get('governors', {})
        if not governors:
            return False
        # Consider already-high-perf if ALL sampled governors are 'performance'
        return all(g == 'performance' for g in governors.values())
    return False


def install_auto_performance(system: str) -> bool:
    """Capture state, switch to high-perf (if needed), install cleanup handlers.
    Return True on success, False if we should proceed without the switch."""
    state = capture_current_plan(system)
    if state is None:
        print('[auto-perf] FAIL: could not capture current power state; '
              'proceeding without auto-switch', file=sys.stderr)
        return False

    already_highperf = _is_state_already_highperf(state)
    apply_msg = ''
    if already_highperf:
        apply_msg = 'current plan is already high-performance; skipping apply (tracking only)'
    else:
        ok, apply_msg = apply_performance_plan(system)
        if not ok:
            print(f'[auto-perf] FAIL to apply: {apply_msg}', file=sys.stderr)
            if system == 'Linux' and os.geteuid() != 0:
                print('[auto-perf] Linux auto-performance requires root. '
                      'Re-run with sudo, or use --force if you have already '
                      'set the governor manually.', file=sys.stderr)
            return False

    # Register cleanup handlers, write backup, mark active.
    # We always write a backup so SIGKILL recovery works even for the
    # "already high-perf, no apply" case — if some external process
    # modifies the plan during our sweep, restore brings it back.
    write_backup(state)
    _AUTO_PERF_ACTIVE['state'] = state
    _AUTO_PERF_ACTIVE['done'] = False

    import atexit
    atexit.register(_restore_and_mark_done)
    # SIGINT and SIGTERM — on Windows signal.SIGTERM exists but only
    # raise-from-python works; external termination is harder to catch.
    try:
        signal.signal(signal.SIGINT, _signal_restore_handler)
    except Exception:
        pass
    try:
        signal.signal(signal.SIGTERM, _signal_restore_handler)
    except Exception:
        pass

    print(f'[auto-perf] OK: {apply_msg}')
    if not already_highperf:
        print('[auto-perf] WARNING: if this process is killed (SIGKILL, power loss, '
              'crash), the power plan will NOT be auto-restored.')
        print(f'[auto-perf]          To restore manually afterward, check '
              f'{_backup_path()} or run the orchestrator again and accept the prompt.')
    return True


# ────────────────────── subprocess invocation ──────────────────────

def set_parent_affinity_windows(cpu: int) -> tuple[bool, str]:
    """Set the orchestrator's own affinity on Windows via ctypes.
    Child processes inherit the mask by default on Windows, so setting
    the orchestrator once pins every subprocess we spawn.

    Using ctypes instead of `start /AFFINITY` has two advantages:
      1. Exit codes propagate through subprocess.Popen.wait() correctly
         (the `cmd /c "start /WAIT ..."` chain can lose exit codes in
         some Windows configurations).
      2. No shell quoting, no `start` quirks around first-quoted-arg-as-
         window-title, no delayed-expansion ERRORLEVEL tricks.
    """
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        k32.SetProcessAffinityMask.restype  = ctypes.c_int
        handle = k32.GetCurrentProcess()
        mask = 1 << cpu
        if k32.SetProcessAffinityMask(handle, mask) == 0:
            err = ctypes.get_last_error()
            return False, f"SetProcessAffinityMask failed (GetLastError={err})"
        return True, f"orchestrator pinned to CPU {cpu} (mask 0x{mask:x}); children inherit"
    except Exception as e:
        return False, f"ctypes call failed: {e}"


def build_command(radix_dir: Path, phase: str, cpu: int,
                  system: str) -> list[str]:
    """Return the argv to exec. On Linux we prepend taskset; on Windows
    affinity is inherited from the orchestrator's own mask (set at startup
    via set_parent_affinity_windows)."""
    bench_args = [
        sys.executable, str(BENCH_PY),
        '--radix-dir', str(radix_dir.relative_to(PROJECT_ROOT)),
        '--phase', phase,
    ]
    if system == 'Linux':
        return ['taskset', '-c', str(cpu)] + bench_args
    # Windows and other OSes: bare argv; affinity inheritance handles pinning.
    return bench_args


def run_radix(radix: RadixRun, phase: str, cpu: int, quiet: bool,
              system: str) -> None:
    """Invoke bench.py for one radix with pinning, capture logs, update state."""
    log_path = BENCH_OUT / radix.name / 'orchestrator.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)

    argv = build_command(radix.dir_path, phase, cpu, system)

    t0 = time.time()
    log_f = open(log_path, 'w', encoding='utf-8', errors='replace')
    try:
        proc = subprocess.Popen(
            argv, cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log_f.write(line)
            log_f.flush()
            if not quiet:
                sys.stdout.write(f'[{radix.name}] {line}')
                sys.stdout.flush()
        rc = proc.wait()
    finally:
        log_f.close()

    radix.elapsed_s = time.time() - t0
    _populate_run_stats(radix, log_path)

    if rc != 0:
        radix.status = 'failure'
        radix.reason = f'bench.py exited {rc}'
        return
    if radix.validation_failed > 0:
        radix.status = 'failure'
        radix.reason = f'{radix.validation_failed} validation case(s) failed'
        return
    radix.status = 'success'


_RE_MEAS_LOADED = re.compile(r'loaded (\d+) measurements')
_RE_VALIDATE    = re.compile(r'\[validate\] (\d+) cases, (\d+) failed')


def _populate_run_stats(radix: RadixRun, log_path: Path) -> None:
    """Parse log for measurement/validation counts. Best-effort."""
    try:
        text = log_path.read_text(encoding='utf-8', errors='replace')
    except Exception:
        return
    m = _RE_MEAS_LOADED.search(text)
    if m:
        radix.measurements = int(m.group(1))
    total_cases = 0
    total_failed = 0
    for vm in _RE_VALIDATE.finditer(text):
        total_cases += int(vm.group(1))
        total_failed += int(vm.group(2))
    radix.validation_cases = total_cases
    radix.validation_failed = total_failed


# ────────────────────── summary generation ──────────────────────

_DISP = {'flat': 'flat', 't1s': 't1s', 'log3': 'log3'}

def aggregate_all_measurements() -> dict:
    """Walk bench_out/*/measurements.jsonl, compute per-radix per-ISA per-protocol
    win counts, and per-radix last-modified timestamps. Returns a dict:
      {
        'r10': {
          'last_run': epoch_seconds,
          'measurements': N,
          'avx2':   {'flat': wins, 't1s': wins, 'log3': wins, 'total_points': P},
          'avx512': {...},
        }, ...
      }
    """
    import json as _json  # shadow-safe local
    result = {}
    if not BENCH_OUT.is_dir():
        return result
    for radix_out in sorted(BENCH_OUT.iterdir()):
        if not radix_out.is_dir():
            continue
        m = _RE_RADIX_DIR.match(radix_out.name)
        if not m:
            continue
        meas_file = radix_out / 'measurements.jsonl'
        if not meas_file.is_file():
            continue

        records = []
        try:
            for line in meas_file.read_text(encoding='utf-8', errors='replace').splitlines():
                if line.strip():
                    records.append(_json.loads(line))
        except Exception:
            continue

        # Filter to fwd, non-skipped
        records = [r for r in records if r.get('dir') == 'fwd' and not r.get('skipped')]

        per_isa = {}
        for isa in ('avx2', 'avx512'):
            # Group by (me, ios), keep best ns per protocol at each point
            by_pt = defaultdict(dict)
            for rec in records:
                if rec.get('isa') != isa:
                    continue
                proto = rec.get('protocol')
                if proto not in _DISP:
                    continue  # skip log1_tight and exotic protocols for summary
                key = (rec['me'], rec['ios'])
                ns = rec.get('ns', 0)
                if proto not in by_pt[key] or ns < by_pt[key][proto]:
                    by_pt[key][proto] = ns

            # Count winners across points that have at least 2 protocols compared
            win = defaultdict(int)
            total_points = 0
            for pt, protos in by_pt.items():
                if len(protos) < 1:
                    continue
                winner = min(protos, key=protos.get)
                win[winner] += 1
                total_points += 1
            per_isa[isa] = {
                'flat':         win.get('flat', 0),
                't1s':          win.get('t1s', 0),
                'log3':         win.get('log3', 0),
                'total_points': total_points,
            }

        result[radix_out.name] = {
            'last_run':     meas_file.stat().st_mtime,
            'measurements': len(records),
            **per_isa,
        }
    return result


def _find_transition(records: list[dict], isa: str) -> Optional[int]:
    """Estimate crossover me where t1s starts winning vs flat.
    Returns the smallest me at which t1s wins a majority, or None."""
    by_me = defaultdict(lambda: defaultdict(list))  # me -> proto -> [ns values]
    for rec in records:
        if rec.get('isa') != isa or rec.get('dir') != 'fwd' or rec.get('skipped'):
            continue
        proto = rec.get('protocol')
        if proto in ('flat', 't1s'):
            by_me[rec['me']][proto].append(rec.get('ns', 0))
    # For each me, count how many ios points t1s beats flat
    t1s_majority_me = None
    for me in sorted(by_me):
        flat_wins = 0
        t1s_wins = 0
        # Group again by (me, ios) actually... this is simplification
        # Count how many ios points have t1s < flat
        # We need aligned pairs per (me, ios); do a second pass
        pass
    # Simpler approach: walk records grouped by (isa, me, ios), best-ns per proto
    pts_by_me = defaultdict(list)  # me -> [(flat_ns, t1s_ns)]
    by_pt = defaultdict(dict)
    for rec in records:
        if rec.get('isa') != isa or rec.get('dir') != 'fwd' or rec.get('skipped'):
            continue
        proto = rec.get('protocol')
        if proto not in ('flat', 't1s'):
            continue
        k = (rec['me'], rec['ios'])
        ns = rec.get('ns', 0)
        if proto not in by_pt[k] or ns < by_pt[k][proto]:
            by_pt[k][proto] = ns
    for (me, ios), protos in by_pt.items():
        if 'flat' in protos and 't1s' in protos:
            pts_by_me[me].append((protos['flat'], protos['t1s']))
    for me in sorted(pts_by_me):
        pts = pts_by_me[me]
        t1s_wins = sum(1 for f, t in pts if t < f)
        if t1s_wins > len(pts) / 2:
            return me
    return None


def regime_transitions() -> dict:
    """For each radix + ISA, return the smallest me where t1s wins a majority of ios points."""
    result = {}
    for radix_out in sorted(BENCH_OUT.iterdir()):
        if not radix_out.is_dir():
            continue
        m = _RE_RADIX_DIR.match(radix_out.name)
        if not m:
            continue
        meas_file = radix_out / 'measurements.jsonl'
        if not meas_file.is_file():
            continue
        try:
            records = [json.loads(L) for L in meas_file.read_text(encoding='utf-8', errors='replace').splitlines() if L.strip()]
        except Exception:
            continue
        result[radix_out.name] = {
            'avx2':   _find_transition(records, 'avx2'),
            'avx512': _find_transition(records, 'avx512'),
        }
    return result


def write_summary(report: SweepReport, summary_path: Path) -> None:
    """Generate bench_out/ALL_SUMMARY.md aggregating all existing measurement data."""
    stats = aggregate_all_measurements()
    trans = regime_transitions()

    lines = []
    lines.append('# VectorFFT portfolio tuning summary')
    lines.append('')
    lines.append(f'Generated: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
    if report.host_info:
        lines.append(f'Host: {report.host_info.get("system", "?")} '
                     f'{report.host_info.get("machine", "?")} '
                     f'(pinned to CPU {report.host_info.get("cpu", "?")})')
    lines.append('')

    # Portfolio stats
    total_radixes = len(stats)
    total_meas = sum(s['measurements'] for s in stats.values())
    passed_this_run = sum(1 for r in report.runs if r.status == 'success')
    failed_this_run = sum(1 for r in report.runs if r.status == 'failure')
    lines.append('## Portfolio')
    lines.append('')
    lines.append(f'- **Radixes tuned** (with measurement data on disk): {total_radixes}')
    lines.append(f'- **Total measurements** (fwd + bwd, across all radixes): {total_meas}')
    if report.runs:
        lines.append(f'- **This invocation**: {passed_this_run} passed, '
                     f'{failed_this_run} failed, {len(report.runs)} total')
        if report.overall_elapsed_s:
            lines.append(f'- **Elapsed**: {report.overall_elapsed_s:.1f}s '
                         f'({report.overall_elapsed_s/60:.1f} min)')
    lines.append('')

    # This-invocation outcomes (only if any radixes ran)
    if report.runs:
        lines.append('## This invocation')
        lines.append('')
        lines.append('| Radix | Status | Measurements | Validated | Failed | Elapsed | Reason |')
        lines.append('|---|---|---|---|---|---|---|')
        for r in report.runs:
            reason = r.reason or ''
            lines.append(f'| {r.name} | {r.status} | {r.measurements} | '
                         f'{r.validation_cases} | {r.validation_failed} | '
                         f'{r.elapsed_s:.1f}s | {reason} |')
        lines.append('')

    # Win counts per radix per ISA
    lines.append('## Dispatcher winners by radix (fwd direction)')
    lines.append('')
    lines.append('For each (radix, ISA), count of sweep points each protocol wins. '
                 '"Total" is the number of (me, ios) grid points where at least '
                 'one protocol was measured.')
    lines.append('')
    lines.append('### AVX2')
    lines.append('')
    lines.append('| Radix | flat | t1s | log3 | Total | Transition me |')
    lines.append('|---|---|---|---|---|---|')
    for rname in sorted(stats, key=lambda s: int(_RE_RADIX_DIR.match(s).group(1))):
        s = stats[rname]
        a = s.get('avx2', {})
        tm = trans.get(rname, {}).get('avx2')
        tm_str = f'me≥{tm}' if tm is not None else '—'
        if a.get('total_points', 0) == 0:
            continue
        lines.append(f'| {rname} | {a.get("flat",0)} | {a.get("t1s",0)} | '
                     f'{a.get("log3",0)} | {a.get("total_points",0)} | {tm_str} |')
    lines.append('')
    lines.append('### AVX-512')
    lines.append('')
    lines.append('| Radix | flat | t1s | log3 | Total | Transition me |')
    lines.append('|---|---|---|---|---|---|')
    for rname in sorted(stats, key=lambda s: int(_RE_RADIX_DIR.match(s).group(1))):
        s = stats[rname]
        a = s.get('avx512', {})
        tm = trans.get(rname, {}).get('avx512')
        tm_str = f'me≥{tm}' if tm is not None else '—'
        if a.get('total_points', 0) == 0:
            continue
        lines.append(f'| {rname} | {a.get("flat",0)} | {a.get("t1s",0)} | '
                     f'{a.get("log3",0)} | {a.get("total_points",0)} | {tm_str} |')
    lines.append('')

    # Last-run timestamps (detect stale data)
    lines.append('## Last run per radix')
    lines.append('')
    lines.append('| Radix | Last measurement written |')
    lines.append('|---|---|')
    for rname in sorted(stats, key=lambda s: int(_RE_RADIX_DIR.match(s).group(1))):
        ts = stats[rname]['last_run']
        lines.append(f'| {rname} | {time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))} |')
    lines.append('')

    lines.append('---')
    lines.append('')
    lines.append('Per-radix reports: `generated/rN/vfft_rN_report.md`. '
                 'Raw measurements: `bench_out/rN/measurements.jsonl`. '
                 'Orchestrator logs: `bench_out/rN/orchestrator.log`.')

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


# ────────────────────── config ──────────────────────

def load_config(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except Exception as e:
        print(f"orchestrator: failed to parse {path}: {e}", file=sys.stderr)
        return {}
    if not isinstance(data, dict):
        print(f"orchestrator: {path} is not a JSON object", file=sys.stderr)
        return {}
    return data


# ────────────────────── CLI ──────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description='VectorFFT per-radix bench orchestrator.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--phase', choices=['emit', 'build', 'run', 'validate', 'all'],
                    help='phase passed through to each per-radix bench.py (default: all)')
    ap.add_argument('--radix', help='comma-separated radix names (e.g. r10,r12)')
    ap.add_argument('--skip',  help='comma-separated radix names to skip')
    ap.add_argument('--cpu', type=int, help='CPU core to pin bench runs to (default: 2)')
    ap.add_argument('--pace-seconds', type=int,
                    help='pause between radix runs for thermal pacing (default: 2)')
    ap.add_argument('--quiet', action='store_true',
                    help='capture per-radix output to log only (default: live stream)')
    ap.add_argument('--force', action='store_true',
                    help='skip pre-flight governor/power-plan check')
    ap.add_argument('--fail-fast', action='store_true',
                    help='stop on first radix failure (default: continue)')
    ap.add_argument('--dry-run', action='store_true',
                    help='print discovered radixes and commands, do not execute')
    ap.add_argument('--no-summary', action='store_true',
                    help='skip regenerating bench_out/ALL_SUMMARY.md')
    ap.add_argument('--summary-only', action='store_true',
                    help='regenerate ALL_SUMMARY.md from existing measurement data '
                         'and exit; skip preflight and bench runs entirely')
    ap.add_argument('--auto-performance', action='store_true',
                    help='before sweep: capture current power plan/governor, '
                         'switch to high-performance, restore on exit. Opt-in; '
                         'Linux requires root. Backup written to '
                         'bench_out/.power_state_backup for recovery if the '
                         'process is killed.')
    ap.add_argument('--config', type=Path, default=CONFIG_FILE_DEFAULT,
                    help=f'config file path (default: {CONFIG_FILE_DEFAULT.name})')
    return ap.parse_args()


def merge_settings(args) -> dict:
    """Precedence: CLI > config file > builtin defaults."""
    settings = dict(DEFAULTS)
    config = load_config(args.config)
    settings.update({k: v for k, v in config.items() if k in DEFAULTS and v is not None})
    # CLI overrides — only update keys the user explicitly set
    for k in DEFAULTS:
        arg_val = getattr(args, k, None) if k not in ('fail_fast', 'no_summary') else None
        # argparse translates --fail-fast -> args.fail_fast; same convention
        arg_val = getattr(args, k.replace('-', '_'), None)
        if arg_val is None:
            continue
        # For boolean store_true flags, argparse gives False when absent; treat
        # False as "CLI didn't flip it, so use config/default"
        if isinstance(DEFAULTS[k], bool):
            if arg_val is True:
                settings[k] = True
        else:
            settings[k] = arg_val
    return settings


# ────────────────────── main ──────────────────────

def main():
    args = parse_args()
    s = merge_settings(args)
    system = platform.system()

    # --summary-only: regenerate the summary from whatever data is on disk,
    # then exit. No preflight, no bench runs. Useful when a previous sweep
    # completed successfully but the summary write failed (e.g. a bug in the
    # summary code), or when the user wants a fresh summary after manually
    # deleting stale per-radix data.
    if getattr(args, 'summary_only', False):
        empty_report = SweepReport()
        empty_report.host_info = {
            'system':  system,
            'machine': platform.machine(),
            'cpu':     s['cpu'],
        }
        summary_path = BENCH_OUT / 'ALL_SUMMARY.md'
        try:
            write_summary(empty_report, summary_path)
        except Exception as e:
            print(f'[orch] summary regeneration FAILED: {type(e).__name__}: {e}',
                  file=sys.stderr)
            return 1
        print(f'[orch] summary regenerated at {summary_path}')
        return 0

    # Check for stale backup from a prior ungraceful exit. Do this BEFORE
    # pre-flight because if the stale state is Balanced but we're now running
    # in performance (leftover from a killed previous run), the user probably
    # wants the restore offered regardless of whether this invocation uses
    # --auto-performance.
    check_stale_backup(s['auto_performance'])

    # Pre-flight
    ok, messages = run_preflight(s['force'])
    for tag, good, msg in messages:
        marker = 'OK' if good else 'FAIL'
        print(f'[preflight] {marker}  {tag}: {msg}')

    # If --auto-performance is on and pre-flight failed because of governor,
    # try to fix by switching plan; then re-run pre-flight.
    # (Skip the actual switch during --dry-run; we only want to show what
    # would be done, not modify system state.)
    if not ok and s['auto_performance'] and not s['force'] and not s['dry_run']:
        print('[preflight] governor check failed; attempting auto-switch to high-performance')
        if install_auto_performance(system):
            # Re-run pre-flight to confirm the switch worked
            ok2, messages2 = run_preflight(s['force'])
            for tag, good, msg in messages2:
                marker = 'OK' if good else 'FAIL'
                print(f'[preflight/retry] {marker}  {tag}: {msg}')
            ok = ok2

    if not ok and not s['force']:
        print('[preflight] FAIL — run with --force to bypass, or use --auto-performance '
              'to auto-switch (Linux requires root). Measurements in a non-perf plan may be noisy.',
              file=sys.stderr)
        return 2

    # If auto-performance was requested but wasn't already installed above
    # (pre-flight passed without needing the switch), install it now so the
    # restore-on-exit behavior still applies — prior state is preserved
    # regardless of whether the current state was already high-perf.
    # Skip during --dry-run: we don't modify system state when just showing plans.
    if s['auto_performance'] and not s['dry_run'] and _AUTO_PERF_ACTIVE['state'] is None:
        install_auto_performance(system)

    # Windows: set orchestrator's own affinity so child processes inherit.
    # (Linux pinning happens per-subprocess via the taskset wrapper; this
    # block is Windows-only.) Skip during --dry-run.
    if system == 'Windows' and not s['dry_run']:
        ok_aff, msg_aff = set_parent_affinity_windows(s['cpu'])
        marker = 'OK' if ok_aff else 'WARN'
        print(f'[affinity] {marker}  {msg_aff}')
        if not ok_aff:
            print('[affinity] proceeding without parent-affinity pinning; '
                  'measurements may be noisier', file=sys.stderr)

    # Discover
    all_runs = discover_radixes()
    if not all_runs:
        print('[orch] no radixes discovered under radixes/; nothing to do')
        return 0
    selected = filter_radixes(all_runs, s['radix'], s['skip'])
    if not selected:
        print('[orch] radix filter matched zero radixes; nothing to do')
        return 0

    print(f'[orch] discovered {len(all_runs)} radixes, running {len(selected)}: '
          f'{", ".join(r.name for r in selected)}')

    if s['dry_run']:
        print('[orch] --dry-run: would execute:')
        if system == 'Windows':
            print(f'  (orchestrator affinity → CPU {s["cpu"]} via SetProcessAffinityMask; '
                  f'children inherit)')
        for r in selected:
            argv = build_command(r.dir_path, s['phase'], s['cpu'], system)
            print('  ' + ' '.join(argv))
        return 0

    # Run
    report = SweepReport()
    report.runs = selected
    report.host_info = {
        'system':  system,
        'machine': platform.machine(),
        'cpu':     s['cpu'],
    }
    BENCH_OUT.mkdir(parents=True, exist_ok=True)

    t_overall = time.time()
    any_failed = False
    for i, r in enumerate(selected):
        if i > 0:
            time.sleep(s['pace_seconds'])
        print(f'[orch] ({i+1}/{len(selected)}) starting {r.name} '
              f'(pinned CPU {s["cpu"]}, phase={s["phase"]})')
        try:
            run_radix(r, s['phase'], s['cpu'], s['quiet'], system)
        except KeyboardInterrupt:
            r.status = 'failure'
            r.reason = 'KeyboardInterrupt'
            print(f'\n[orch] interrupted during {r.name}; stopping sweep', file=sys.stderr)
            any_failed = True
            break
        except Exception as e:
            r.status = 'failure'
            r.reason = f'{type(e).__name__}: {e}'
        status_str = r.status.upper()
        note = f' ({r.reason})' if r.reason else ''
        print(f'[orch] {r.name} {status_str} in {r.elapsed_s:.1f}s: '
              f'{r.measurements} measurements, {r.validation_cases} validated, '
              f'{r.validation_failed} failed{note}')
        if r.status == 'failure':
            any_failed = True
            if s['fail_fast']:
                print(f'[orch] --fail-fast: stopping after {r.name} failed')
                break

    report.overall_elapsed_s = time.time() - t_overall

    # Summary
    if not s['no_summary']:
        summary_path = BENCH_OUT / 'ALL_SUMMARY.md'
        write_summary(report, summary_path)
        print(f'[orch] summary written to {summary_path.relative_to(PROJECT_ROOT)}')

    # Overall log
    overall_log = BENCH_OUT / 'orchestrator.log'
    with open(overall_log, 'w', encoding='utf-8') as f:
        f.write(f'Sweep at {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'System: {system} {platform.machine()}  CPU pin: {s["cpu"]}\n')
        f.write(f'Phase: {s["phase"]}  Elapsed: {report.overall_elapsed_s:.1f}s\n\n')
        for r in report.runs:
            f.write(f'{r.name}: {r.status} in {r.elapsed_s:.1f}s  '
                    f'meas={r.measurements} val={r.validation_cases} '
                    f'failed={r.validation_failed}  {r.reason or ""}\n')

    print(f'[orch] done in {report.overall_elapsed_s:.1f}s '
          f'({report.overall_elapsed_s/60:.1f} min). '
          f'Exit {1 if any_failed else 0}.')
    return 1 if any_failed else 0


if __name__ == '__main__':
    sys.exit(main())
