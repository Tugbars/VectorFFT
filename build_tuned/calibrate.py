"""
calibrate.py — orchestrate new-core calibration with powercfg control.

Mirrors what src/vectorfft_tune/common/orchestrator.py does for the
codelet sweep, but for the top-level (N, K) plan calibration:

  1. Verify Windows is on a high-performance plan (preflight). Capture
     the current plan to a backup file so we can restore on exit.
  2. Switch to High Performance.
  3. Compile build_tuned/calibrate_tuned.c via the existing build.py.
  4. Run calibrate_tuned.exe — produces vfft_wisdom_tuned.txt and
     vfft_wisdom_tuned_codelets.csv next to the binary.
  5. Restore the original power plan, even if the C run fails.

Usage:
    python build_tuned/calibrate.py
    python build_tuned/calibrate.py --no-powercfg   # skip power switch
    python build_tuned/calibrate.py --skip-build    # reuse existing exe
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent  # repo root

# Reuse the orchestrator's power-plan helpers — same behaviour as the
# codelet calibration so this calibrator runs under matching conditions.
sys.path.insert(0, str(ROOT / 'src' / 'vectorfft_tune' / 'common'))
try:
    from orchestrator import (
        preflight_windows,
        capture_current_plan,
        apply_performance_plan,
        restore_plan,
        set_parent_affinity_windows,
    )
    HAVE_ORCHESTRATOR = True
except Exception as e:  # pragma: no cover — orchestrator may move
    print(f'[warn] could not import orchestrator helpers: {e}',
          file=sys.stderr)
    HAVE_ORCHESTRATOR = False


# Default pin = CPU 2: matches orchestrator.py default. On consumer Intel,
# CPU 0 handles interrupts and CPU 1 is its SMT sibling; CPU 2 is the first
# clean P-core. The codelet wisdom in vectorfft_tune/generated/ was
# calibrated on this same core, so the top-level (N, K) calibration must
# pin here too — otherwise the codelet predicates may pick a variant tuned
# for a different cache/branch behaviour than what we measure.
DEFAULT_CPU = 2


BACKUP_FILE = HERE / '.power_state_backup'


def save_backup(state: dict) -> None:
    BACKUP_FILE.write_text(json.dumps(state, indent=2), encoding='utf-8')


def load_backup() -> dict | None:
    if not BACKUP_FILE.exists():
        return None
    try:
        return json.loads(BACKUP_FILE.read_text(encoding='utf-8'))
    except Exception:
        return None


def clear_backup() -> None:
    if BACKUP_FILE.exists():
        BACKUP_FILE.unlink()


class PowercfgError(RuntimeError):
    pass


def setup_powercfg() -> dict | None:
    """Capture, switch to high-perf, return saved state for later restore.

    Strict: raises PowercfgError if we cannot guarantee a high-perf plan
    is active (orchestrator missing, switch fails, OS unsupported on a
    machine where we needed to switch). The codelet wisdom was calibrated
    under high-perf, so running without it would let frequency throttling
    distort plan-search timing and the resulting top-level wisdom would
    diverge from what we'd see at deploy time."""
    if not HAVE_ORCHESTRATOR:
        raise PowercfgError(
            'orchestrator helpers unavailable — cannot verify or set power plan')

    system = platform.system()
    if system != 'Windows':
        # On Linux orchestrator's preflight_linux checks the governor; we
        # skip auto-switch (needs root) and assume the user has set
        # `performance` governor manually.
        print(f'[powercfg] OS={system}: relying on user-managed governor')
        return None

    # Stale backup from a prior crashed run? Warn but don't auto-restore —
    # the user may have already manually fixed it.
    stale = load_backup()
    if stale is not None:
        print(f'[powercfg] WARN: stale backup found at {BACKUP_FILE}')
        print(f'[powercfg]       contents: {stale}')
        print(f'[powercfg]       a previous calibration may have crashed without restoring.')
        print(f'[powercfg]       continuing — will overwrite this backup.')

    state = capture_current_plan(system)
    if state is None:
        raise PowercfgError('could not capture current plan via powercfg')

    # Already on high-perf? Don't overwrite — just record so we don't restore
    # to the wrong thing on exit.
    ok, msg = preflight_windows()
    if ok:
        print(f'[powercfg] {msg} — leaving as-is')
        save_backup(state)
        return state  # state is the already-active high-perf plan; restore is a no-op

    save_backup(state)
    ok, msg = apply_performance_plan(system)
    if not ok:
        clear_backup()
        raise PowercfgError(f'switch to High Performance failed: {msg}')
    print(f'[powercfg] {msg}')
    return state


def teardown_powercfg(state: dict | None) -> None:
    if state is None:
        return
    if not HAVE_ORCHESTRATOR:
        print('[powercfg] orchestrator helpers unavailable; cannot restore')
        return
    ok, msg = restore_plan(state)
    if ok:
        print(f'[powercfg] {msg}')
        clear_backup()
    else:
        print(f'[powercfg] WARN: restore failed: {msg}')
        print(f'[powercfg]       backup retained at {BACKUP_FILE}')
        print(f'[powercfg]       restore manually with: powercfg /setactive <guid>')


def compile_calibrator() -> int:
    """Run build.py with --compile and our source. Returns rc."""
    cmd = [sys.executable, str(HERE / 'build.py'),
           '--compile',
           '--src', str(HERE / 'calibrate_tuned.c')]
    return subprocess.run(cmd).returncode


def setup_affinity(cpu: int) -> bool:
    """Pin this Python process to a single CPU. Child processes inherit
    the affinity mask on Windows (we use SetProcessAffinityMask) and on
    Linux (we prepend `taskset -c <cpu>` per child).

    Returns True on success, False on failure. Caller decides whether to
    proceed without affinity."""
    system = platform.system()
    if system == 'Windows':
        if not HAVE_ORCHESTRATOR:
            print('[affinity] orchestrator helpers unavailable; cannot pin',
                  file=sys.stderr)
            return False
        ok, msg = set_parent_affinity_windows(cpu)
        print(f'[affinity] {msg}')
        return ok
    if system == 'Linux':
        # On Linux we'll prepend `taskset -c <cpu>` to the run command
        # rather than mask the launcher itself (matches orchestrator).
        import shutil
        if shutil.which('taskset') is None:
            print('[affinity] taskset not on PATH; cannot pin', file=sys.stderr)
            return False
        print(f'[affinity] will prepend `taskset -c {cpu}` to calibrator')
        return True
    print(f'[affinity] OS={system}: pinning not implemented; proceeding unpinned')
    return False


def run_calibrator(cpu: int, pinned: bool) -> int:
    exe = HERE / ('calibrate_tuned.exe' if os.name == 'nt' else 'calibrate_tuned')
    if not exe.exists():
        print(f'[error] calibrator binary not found: {exe}', file=sys.stderr)
        return 1
    out = HERE / 'vfft_wisdom_tuned.txt'
    info = HERE / 'vfft_wisdom_tuned_codelets.csv'

    argv: list[str] = []
    if pinned and platform.system() == 'Linux':
        argv += ['taskset', '-c', str(cpu)]
    argv += [str(exe), str(out), str(info)]

    print(f'[run] {" ".join(argv)}')
    t0 = time.time()
    rc = subprocess.run(argv).returncode
    print(f'[run] exit {rc}  elapsed {time.time()-t0:.1f}s')
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-powercfg', action='store_true',
                    help='Skip the auto powercfg switch. The script will '
                         'still verify a high-perf plan is active and '
                         'refuse to run otherwise.')
    ap.add_argument('--skip-build', action='store_true',
                    help='Skip the compile step (reuse existing exe).')
    ap.add_argument('--cpu', type=int, default=DEFAULT_CPU,
                    help=f'CPU core to pin the calibrator to. Default: '
                         f'{DEFAULT_CPU} (matches orchestrator.py — first '
                         f'clean P-core on consumer Intel).')
    ap.add_argument('--no-affinity', action='store_true',
                    help='Skip CPU pinning. NOT recommended — codelet '
                         'wisdom was tuned under pinned conditions.')
    args = ap.parse_args()

    print('=' * 60)
    print(' new-core calibration: A/B wisdom generator')
    print('=' * 60)

    # ── Power plan ────────────────────────────────────────────────────
    state = None
    try:
        if args.no_powercfg:
            # Don't auto-switch, but still require a high-perf plan.
            if not HAVE_ORCHESTRATOR:
                print('[powercfg] orchestrator unavailable; cannot verify plan')
                return 2
            if platform.system() == 'Windows':
                ok, msg = preflight_windows()
                if not ok:
                    print(f'[powercfg] {msg}')
                    print('[powercfg] refusing to calibrate on a non-high-perf plan.')
                    print('[powercfg] either drop --no-powercfg, or set the plan manually.')
                    return 2
                print(f'[powercfg] {msg} — user-managed (--no-powercfg)')
        else:
            state = setup_powercfg()
    except PowercfgError as e:
        print(f'[powercfg] FATAL: {e}', file=sys.stderr)
        print('[powercfg] refusing to calibrate without a verified high-perf plan.',
              file=sys.stderr)
        return 2

    # ── CPU affinity ──────────────────────────────────────────────────
    pinned = False
    if not args.no_affinity:
        pinned = setup_affinity(args.cpu)
        if not pinned:
            print('[affinity] FATAL: could not pin to core; refusing to run.',
                  file=sys.stderr)
            print('[affinity] use --no-affinity to bypass (codelet wisdom was '
                  'tuned with pinning, so unpinned runs may be misleading).',
                  file=sys.stderr)
            teardown_powercfg(state)
            return 2
    else:
        print('[affinity] WARN: --no-affinity given; running unpinned')

    # ── Compile + run ────────────────────────────────────────────────
    rc = 0
    try:
        if not args.skip_build:
            rc = compile_calibrator()
            if rc != 0:
                print(f'[error] compile failed with rc={rc}')
                return rc
        rc = run_calibrator(args.cpu, pinned)
    finally:
        teardown_powercfg(state)

    return rc


if __name__ == '__main__':
    sys.exit(main())
