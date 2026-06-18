"""
calibrate_remaining.py - fill spike_wisdom.txt up to the old-complete
vfft_wisdom_tuned.txt cell set, for an UNATTENDED run.

Difference from calibrate.py (which is one-K-grid-per-invocation):
  - TARGET-DRIVEN: the cell list is computed as (cells in vfft_wisdom_tuned.txt)
    minus (cells already in spike_wisdom.txt). No hardcoded grid.
  - GLOBAL cost order: all remaining (N,K) sorted by N*K ascending, so cheap
    cells across every K finish first (broad coverage) and the monster K=256
    composites land last.
  - RESUMABLE: re-reads spike before each cell and skips any now present, so a
    restart (or a cell another process just finished) is a no-op.
  - TIMEOUT-GUARDED: a cell exceeding --timeout is killed and skipped, so one
    pathological search can't eat the whole window.
  - Reuses calibrate.py's build()/verify_power() and the High-priority +
    pinned-core + cooldown discipline (thermal-order-bias fix).

Writes to the calibrator's DEFAULT wisdom (generated/spike_wisdom.txt) - we do
NOT set VFFT_PROTO_WIS, so it loads+appends to the real spike file.

usage:
  python build_tuned/calibrate_remaining.py                 # all remaining, cost-ordered
  python build_tuned/calibrate_remaining.py --max-nk 5000000 # skip the biggest tail
  python build_tuned/calibrate_remaining.py --dry-run        # just print the plan
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time
from pathlib import Path

import calibrate as cal   # reuse build(), verify_power(), DRIVER_EXE, paths

HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent
SPIKE  = ROOT / 'src' / 'dag-fft-compiler' / 'generator' / 'generated' / 'spike_wisdom.txt'
TARGET = HERE / 'vfft_wisdom_tuned.txt'   # the old-complete 198-cell production wisdom


def parse_cells(path: Path) -> set[tuple[int, int]]:
    s: set[tuple[int, int]] = set()
    if not path.exists():
        return s
    for ln in path.read_text(encoding='utf-8', errors='replace').splitlines():
        ln = ln.strip()
        if not ln or ln[0] in '#@':
            continue
        p = ln.split()
        if len(p) < 2:
            continue
        try:
            s.add((int(p[0]), int(p[1])))
        except ValueError:
            pass
    return s


def run_cell(N, K, core, timeout) -> tuple[str, float]:
    flags = subprocess.HIGH_PRIORITY_CLASS if os.name == 'nt' else 0
    t0 = time.time()
    # do NOT set VFFT_PROTO_WIS -> calibrator uses its absolute default (spike).
    p = subprocess.Popen([str(cal.DRIVER_EXE), str(N), str(K), str(core), '1'],
                         creationflags=flags)
    try:
        rc = p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        try: p.wait(5)
        except Exception: pass
        return ('TIMEOUT', time.time() - t0)
    return (('OK' if rc == 0 else f'rc={rc}'), time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--core', type=int, default=2)
    ap.add_argument('--cooldown', type=float, default=15.0)
    ap.add_argument('--timeout', type=float, default=2700.0, help='per-cell kill (s)')
    ap.add_argument('--max-nk', type=int, default=0, help='skip cells with N*K above this (0=no cap)')
    ap.add_argument('--skip-build', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    target = parse_cells(TARGET)
    if not target:
        print(f'[fatal] target wisdom empty/missing: {TARGET}', file=sys.stderr); sys.exit(1)
    done = parse_cells(SPIKE)
    remaining = sorted(target - done, key=lambda nk: (nk[0] * nk[1], nk[1], nk[0]))
    if args.max_nk:
        remaining = [nk for nk in remaining if nk[0] * nk[1] <= args.max_nk]

    print('=' * 70)
    print(f' calibrate_remaining  core={args.core}  cooldown={args.cooldown}s  timeout={args.timeout:.0f}s')
    print(f' target={len(target)}  done(spike)={len(done)}  REMAINING={len(remaining)}')
    print(f' order: N*K ascending (cheap first; monster K=256 last)')
    print('=' * 70)
    for i, (N, K) in enumerate(remaining):
        print(f'  [{i+1:>3}/{len(remaining)}] N={N:<7} K={K:<4} (N*K={N*K})')
    if args.dry_run:
        print('[dry-run] not executing.'); return

    cal.verify_power()
    if not args.skip_build:
        cal.build()

    t_start = time.time()
    for i, (N, K) in enumerate(remaining):
        # resumable: re-check spike each iteration
        if (N, K) in parse_cells(SPIKE):
            print(f'\n----- [{i+1}/{len(remaining)}] N={N} K={K}  (already present - skip) -----', flush=True)
            continue
        print(f'\n----- [{i+1}/{len(remaining)}] N={N} K={K}  (N*K={N*K}) -----', flush=True)
        status, secs = run_cell(N, K, args.core, args.timeout)
        tag = '' if status == 'OK' else f'  <<< {status}'
        print(f'  -> {status} in {secs:.1f}s{tag}', flush=True)
        if i < len(remaining) - 1 and args.cooldown > 0:
            time.sleep(args.cooldown)
    print(f'\n[done] elapsed {(time.time()-t_start)/60:.1f} min. '
          f'spike now has {len(parse_cells(SPIKE))} entries.')


if __name__ == '__main__':
    main()
