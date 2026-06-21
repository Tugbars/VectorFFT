#!/usr/bin/env python3
"""calibrate_fft2d_c2r.py -- batch 2D C2R calibration (convenience wrapper).

Mirror of calibrate_fft2d_r2c.py, but the planner scores the BACKWARD (c2r)
direction and writes a SEPARATE c2r wisdom file (r2c != c2r optima). One isolated
process per (N1,N2) cell. Resumable.

  usage:  python calibrate_fft2d_c2r.py [--patient] [--force] [--core N]
  build:  python build.py --src benches/calibrator/calibrate_fft2d_c2r.c --compile
"""
import os
import sys
import subprocess
import time
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXE = HERE / 'benches' / 'calibrator' / 'calibrate_fft2d_c2r.exe'
WIS = Path('C:/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler/'
           'generator/generated/fft2d_c2r_wisdom.txt')
MINGW_BIN = r'C:\mingw152\mingw64\bin'

CELLS = [(64, 64), (128, 128), (256, 256), (512, 512)]


def done_cells(path):
    s = set()
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line[0] in '#@':
                continue
            t = line.split()
            if len(t) >= 2:
                try:
                    s.add((int(t[0]), int(t[1])))
                except ValueError:
                    pass
    return s


def run_cell(n1, n2, core, patient, timeout):
    env = os.environ.copy()
    env['PATH'] = MINGW_BIN + os.pathsep + env.get('PATH', '')
    args = [str(EXE), str(n1), str(n2), str(core), '1', '1' if patient else '0']
    flags = subprocess.HIGH_PRIORITY_CLASS if os.name == 'nt' else 0
    t0 = time.time()
    try:
        p = subprocess.Popen(args, env=env, creationflags=flags)
        p.wait(timeout=timeout)
        return ('ok' if p.returncode == 0 else f'rc={p.returncode}'), time.time() - t0
    except subprocess.TimeoutExpired:
        p.kill()
        return 'timeout', time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--patient', action='store_true')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--core', type=int, default=2)
    args = ap.parse_args()

    if not EXE.exists():
        print(f'[error] {EXE.name} not built. Run:\n'
              f'  python build.py --src benches/calibrator/calibrate_fft2d_c2r.c --compile')
        return 1

    cells = sorted(CELLS, key=lambda c: c[0] * c[1])
    mode = 'PATIENT' if args.patient else 'MEASURE'
    print(f'2D C2R calibration ({mode}) -- {len(cells)} cells -> {WIS.name}')
    for (n1, n2) in cells:
        if not args.force and (n1, n2) in done_cells(WIS):
            print(f'  {n1}x{n2}: present, skip (--force to redo)')
            continue
        total = n1 * n2
        timeout = max(180, total // 500)
        st, secs = run_cell(n1, n2, args.core, args.patient, timeout)
        cool = min(30, max(3, total // 40000))
        print(f'  {n1}x{n2}: {st} ({secs:.0f}s); cool {cool}s')
        time.sleep(cool)
    print('done.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
