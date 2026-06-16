"""
calibrate.py - orchestrate dag-fft-compiler wisdom calibration.

Re-pointed from production for the dag tree (2026-06-16):
  - Production switched to "High Performance" and imported powercfg helpers
    from src/vectorfft_tune/common/orchestrator.py - BOTH retired. We saw this
    session that High Performance got corrupted (turbo latched off) and that
    powercfg freq-capping backfired.
  - So this VERIFIES a turbo-capable scheme (Ultimate Performance) is active and
    WARNS if not - it does NOT switch schemes (needs admin; you set it once).
  - Runs the driver ONE CELL PER PROCESS (isolation), High priority, with an
    inter-cell cooldown to reset the thermal baseline (prevents the cross-cell
    heat-soak that inflated numbers ~1.5x this session).

usage:
  python build_tuned/calibrate.py                       # default small subset, K=4
  python build_tuned/calibrate.py --cells 8,64,128,256  # specific cells
  python build_tuned/calibrate.py --all                 # full 18-cell K=4 grid
  python build_tuned/calibrate.py --skip-build --wisdom C:/tmp/test_wisdom.txt
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent
DAG  = ROOT / 'src' / 'dag-fft-compiler'
DRIVER_SRC = DAG / 'calibrator' / 'calibrate.c'
DRIVER_EXE = DAG / 'calibrator' / ('calibrate.exe' if os.name == 'nt' else 'calibrate')

# Ultimate Performance scheme GUID (the one that turbos on this 14900KF).
ULTIMATE_GUID = '9addb3a6-5b21-4f19-9686-4f5825f2ff53'

# K=4 grid, smallest-first.
GRID_K4 = [8, 16, 32, 64, 126, 128, 250, 256, 400, 512,
           1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
# "run a few first" default: a fast, varied spread (single-codelet, 2-stage, 3-stage).
DEFAULT_FEW = [8, 64, 128, 256, 512, 1024]


def verify_power():
    """Warn (don't switch) if a turbo-capable scheme isn't active."""
    if os.name != 'nt':
        print('[power] non-Windows: relying on user-set performance governor')
        return
    try:
        out = subprocess.run(['powercfg', '/getactivescheme'], capture_output=True,
                             text=True).stdout
    except Exception as e:
        print(f'[power] could not query scheme: {e}'); return
    print(f'[power] active: {out.strip()}')
    if ULTIMATE_GUID not in out.lower():
        print('[power] WARNING: Ultimate Performance is NOT active. Turbo may be '
              'capped/variable. Set it once (elevated):', file=sys.stderr)
        print(f'        powercfg /setactive {ULTIMATE_GUID}', file=sys.stderr)
        print('        (calibrate.py does not switch schemes - see header.)', file=sys.stderr)
    else:
        print('[power] Ultimate Performance active - turbo available.')


def build():
    print('[build] compiling driver via build.py (codelet lib cached) ...', flush=True)
    r = subprocess.run([sys.executable, str(HERE / 'build.py'),
                        '--src', str(DRIVER_SRC), '--compile'])
    if r.returncode != 0:
        print('[build] FAILED', file=sys.stderr); sys.exit(1)
    if not DRIVER_EXE.exists():
        print(f'[build] driver exe missing: {DRIVER_EXE}', file=sys.stderr); sys.exit(1)


def calibrate_cell(N, K, core, wisdom):
    env = os.environ.copy()
    if wisdom:
        env['VFFT_PROTO_WIS'] = wisdom
    flags = subprocess.HIGH_PRIORITY_CLASS if os.name == 'nt' else 0
    p = subprocess.Popen([str(DRIVER_EXE), str(N), str(K), str(core), '1'],
                         env=env, creationflags=flags)
    p.wait()
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', help='comma list of N (K=4 grid)')
    ap.add_argument('--all', action='store_true', help='full 18-cell grid')
    ap.add_argument('--K', type=int, default=4)
    ap.add_argument('--core', type=int, default=2)
    ap.add_argument('--cooldown', type=float, default=15.0, help='idle seconds between cells')
    ap.add_argument('--skip-build', action='store_true')
    ap.add_argument('--wisdom', help='output wisdom path (default: dag spike_wisdom)')
    args = ap.parse_args()

    if args.cells:
        cells = [int(x) for x in args.cells.split(',')]
    elif args.all:
        cells = GRID_K4
    else:
        cells = DEFAULT_FEW

    print('=' * 64)
    print(f' dag calibration - K={args.K}  core={args.core}  cooldown={args.cooldown}s')
    print(f' cells ({len(cells)}): {cells}')
    print(f' wisdom: {args.wisdom or "(dag spike_wisdom default)"}')
    print('=' * 64)
    verify_power()
    if not args.skip_build:
        build()

    for i, N in enumerate(cells):
        print(f'\n----- [{i+1}/{len(cells)}] cell N={N} K={args.K} -----', flush=True)
        rc = calibrate_cell(N, args.K, args.core, args.wisdom)
        if rc != 0:
            print(f'  [warn] cell N={N} returned {rc}', file=sys.stderr)
        if i < len(cells) - 1 and args.cooldown > 0:
            print(f'  ...cooldown {args.cooldown:.0f}s', flush=True)
            time.sleep(args.cooldown)
    print('\n[done] calibration complete.')


if __name__ == '__main__':
    main()
