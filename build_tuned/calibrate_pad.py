"""
calibrate_pad.py — orchestrate the PADDED pad-vs-tail calibration (dev tooling).

Companion to calibrate.py (the tight calibrator). Runs benches/calibrator/calibrate_pad.c
ONE CELL PER PROCESS over the MISALIGNED-K grid (K%4 != 0) with thermal pacing, writing the
per-cell verdict (exec_me + factorization + variants) into spike_wisdom_padded.txt — the
SEPARATE padded wisdom file the production dispatch (vfft.c, config.batch) consults.

Scope (deliberately NOT the ~246-cell tight grid): padding only matters for misaligned K, and
aligned K needs no padded entry at all (Kp==K -> a padded buffer IS a tight buffer). So this
sweeps the small misaligned candidate grid; the pad win concentrates at small K (7, 11, 15).
Each leg uses measure+deploy with the widened PATIENT DP (== the tight calibrator's quality bar).

Built with --jit so the pad leg is benched on the baked/JIT fast path it will actually run
(wrinkle C). Tail-verdict cells are written too (they pre-store factK so the runtime skips its
own calibrate-on-miss; functionally identical to the no-entry fallback).

usage:
  python build_tuned/calibrate_pad.py                       # full misaligned grid (35 cells)
  python build_tuned/calibrate_pad.py --Ks 7,11,15          # subset of K
  python build_tuned/calibrate_pad.py --Ns 256,512,1024     # subset of N
  python build_tuned/calibrate_pad.py --wisdom C:/tmp/pad.txt --skip-build
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent
DAG  = ROOT / 'src' / 'dag-fft-compiler'
DRIVER_SRC = HERE / 'benches' / 'calibrator' / 'calibrate_pad.c'
DRIVER_EXE = HERE / 'benches' / 'calibrator' / ('calibrate_pad.exe' if os.name == 'nt' else 'calibrate_pad')
PAD_WISDOM = DAG / 'generator' / 'generated' / 'spike_wisdom_padded.txt'

ULTIMATE_GUID = '9addb3a6-5b21-4f19-9686-4f5825f2ff53'

# Misaligned-K candidate grid (the Step-0 rem=3 set) × the pow2 N range. Smallest-first so the
# cheap cells (and the small-K pad-winners) report first. Aligned K is skipped by the driver.
NS_DEFAULT = [256, 512, 1024, 2048, 4096]
KS_DEFAULT = [7, 11, 15, 19, 23, 27, 31]


def verify_power():
    if os.name != 'nt':
        print('[power] non-Windows: relying on user-set performance governor'); return
    try:
        out = subprocess.run(['powercfg', '/getactivescheme'], capture_output=True, text=True).stdout
    except Exception as e:
        print(f'[power] could not query scheme: {e}'); return
    print(f'[power] active: {out.strip()}')
    if ULTIMATE_GUID not in out.lower():
        print('[power] WARNING: Ultimate Performance NOT active — turbo may be capped. '
              f'Set once (elevated): powercfg /setactive {ULTIMATE_GUID}', file=sys.stderr)
    else:
        print('[power] Ultimate Performance active — turbo available.')


def build():
    print('[build] compiling calibrate_pad (--jit; codelet lib cached) ...', flush=True)
    r = subprocess.run([sys.executable, str(HERE / 'build.py'),
                        '--src', str(DRIVER_SRC), '--jit', '--compile'])
    if r.returncode != 0:
        print('[build] FAILED', file=sys.stderr); sys.exit(1)
    if not DRIVER_EXE.exists():
        print(f'[build] driver exe missing: {DRIVER_EXE}', file=sys.stderr); sys.exit(1)


def calibrate_cell(N, K, core, wisdom):
    env = os.environ.copy()
    env['VFFT_PROTO_PAD_WIS'] = wisdom
    flags = subprocess.HIGH_PRIORITY_CLASS if os.name == 'nt' else 0
    p = subprocess.Popen([str(DRIVER_EXE), str(N), str(K), str(core), '0'],
                         env=env, creationflags=flags)
    p.wait()
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--Ns', help='comma list of N (default 256,512,1024,2048,4096)')
    ap.add_argument('--Ks', help='comma list of misaligned K (default 7,11,15,19,23,27,31)')
    ap.add_argument('--core', type=int, default=2)
    ap.add_argument('--cooldown', type=float, default=8.0, help='idle seconds between cells')
    ap.add_argument('--skip-build', action='store_true')
    ap.add_argument('--wisdom', help='output padded wisdom path (default: dag spike_wisdom_padded.txt)')
    args = ap.parse_args()

    Ns = [int(x) for x in args.Ns.split(',')] if args.Ns else NS_DEFAULT
    Ks = [int(x) for x in args.Ks.split(',')] if args.Ks else KS_DEFAULT
    wisdom = args.wisdom or str(PAD_WISDOM)
    cells = [(N, K) for N in Ns for K in Ks]   # N-major: cheap small-N cells first

    print('=' * 68)
    print(f' dag PADDED calibration — core={args.core}  cooldown={args.cooldown}s')
    print(f' N ({len(Ns)}): {Ns}')
    print(f' K ({len(Ks)}): {Ks}   (misaligned; aligned K skipped by the driver)')
    print(f' cells: {len(cells)}   wisdom: {wisdom}')
    print('=' * 68)
    verify_power()
    if not args.skip_build:
        build()

    t_start = time.time()
    for i, (N, K) in enumerate(cells):
        print(f'\n----- [{i+1}/{len(cells)}] cell N={N} K={K} -----', flush=True)
        rc = calibrate_cell(N, K, args.core, wisdom)
        if rc != 0:
            print(f'  [warn] cell N={N} K={K} returned {rc}', file=sys.stderr)
        if i < len(cells) - 1 and args.cooldown > 0:
            time.sleep(args.cooldown)
    print(f'\n[done] padded calibration complete in {time.time()-t_start:.0f}s -> {wisdom}')


if __name__ == '__main__':
    main()
