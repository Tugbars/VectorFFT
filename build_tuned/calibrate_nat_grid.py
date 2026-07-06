#!/usr/bin/env python3
"""calibrate_nat_grid.py -- fire-and-forget NATURAL-order grid calibrator (1D + 2D c2c).

Drives build_tuned/test/calibrate_nat.exe over the production cell grid with the CANONICAL
measurement discipline (modeled on bench_1d_vs_mkl.c + run_bench.py):
  - ONE ISOLATED PROCESS PER CELL (fresh process -> no cache/thermal carryover; trusted numbers).
  - core-2 pinned + HIGH_PRIORITY  (the exe SetThreadAffinityMask(1<<2) + SetPriorityClass itself).
  - N*K-scaled INTER-CELL cooldown (reset the thermal baseline between cells).
  - per-cell pre-measure cool via VFFT_CAL_COOL_MS (the exe idles before it measures).
  - the exe's own robust per-cell measurement: 1D natorder race (5% margin, interleaved rounds);
    2D interleaved + min-of-rounds + 5% margin + natural-intrinsic tie-break.
  - RESUMABLE: a cell already carrying an @nat / @nat2d record in the staging file is skipped.

Writes to a STAGING copy of the canonical wisdom (grid -> STAGING -> inspect -> promote), so canonical
is never touched until you cp the staged files over. The 1D grid is read from the staging
spike_wisdom.txt (every scrambled (N,K) cell); the 2D grid is the staging fft2d_c2c_wisdom.txt cells
plus a narrow-row set where natorder palindromes win.

  usage:  python calibrate_nat_grid.py [--dim 1d|2d|both] [--staging DIR] [--force] [--dry-run]
  build:  python build.py --src test/calibrate_nat.c --vfft --jit
  promote (after inspecting STAGING):
          cp STAGING/spike_wisdom.txt STAGING/fft2d_c2c_wisdom.txt <repo>/src/dag-fft-compiler/generator/generated/
"""
import os, sys, time, shutil, subprocess, argparse
from pathlib import Path

REPO   = Path('C:/Users/Tugbars/Desktop/highSpeedFFT')
HERE   = Path(__file__).resolve().parent
EXE    = HERE / 'test' / 'calibrate_nat.exe'
GEN    = REPO / 'src' / 'dag-fft-compiler' / 'generator' / 'generated'
SPIKE  = 'spike_wisdom.txt'
FFT2D  = 'fft2d_c2c_wisdom.txt'
MINGW  = r'C:\mingw152\mingw64\bin'

# 2D narrow-row cells where natorder palindromes/FREE-leaf win (beyond the square cells already banked).
NARROW_2D = [(64,16),(128,16),(256,16),(512,16),(128,64),(256,64),(512,64),(128,32),(256,32)]


def parse_scrambled_1d(path):
    """Every scrambled (N,K) cell (plain digit-leading line) in spike_wisdom.txt."""
    cells = []
    if path.exists():
        for ln in path.read_text().splitlines():
            ln = ln.strip()
            if not ln or ln[0] in '#@':
                continue
            t = ln.split()
            if len(t) >= 2:
                try: cells.append((int(t[0]), int(t[1])))
                except ValueError: pass
    return cells


def parse_scrambled_2d(path):
    cells = []
    if path.exists():
        for ln in path.read_text().splitlines():
            ln = ln.strip()
            if not ln or ln[0] in '#@':
                continue
            t = ln.split()
            if len(t) >= 2:
                try: cells.append((int(t[0]), int(t[1])))
                except ValueError: pass
    return cells


def done_nat(path, tag):
    """(N,K) cells that already carry an @nat / @nat2d record (resume)."""
    s = set()
    if path.exists():
        for ln in path.read_text().splitlines():
            ln = ln.strip()
            if ln.startswith(tag + ' '):
                t = ln.split()
                if len(t) >= 3:
                    try: s.add((int(t[1]), int(t[2])))
                    except ValueError: pass
    return s


def cooldown_for(nk):
    return min(20, max(2, nk // 50000))     # seconds; bigger cells cool longer


def run_cell(dim, a, b, staging, log):
    env = os.environ.copy()
    env['PATH'] = MINGW + os.pathsep + env.get('PATH', '')
    env['VFFT_WISDOM_DIR'] = str(staging)
    env['VFFT_CAL_COOL_MS'] = '300'                 # exe idles before it measures
    flags = subprocess.HIGH_PRIORITY_CLASS if os.name == 'nt' else 0
    timeout = max(240, (a * b) // 400)
    t0 = time.time()
    try:
        p = subprocess.run([str(EXE), dim, str(a), str(b)], env=env, creationflags=flags,
                           capture_output=True, text=True, timeout=timeout)
        secs = time.time() - t0
        # last PASS/FAIL/NULL-carrying line from the exe
        tail = [l for l in p.stdout.splitlines() if any(k in l for k in ('PASS','FAIL','NULL'))]
        verdict = tail[-1].strip() if tail else f'rc={p.returncode}'
        return ('ok' if p.returncode == 0 else f'rc={p.returncode}'), secs, verdict
    except subprocess.TimeoutExpired:
        return 'timeout', time.time() - t0, '(timed out)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dim', choices=['1d','2d','both'], default='both')
    ap.add_argument('--staging', default=str(GEN),   # write STRAIGHT TO CANONICAL (user-approved)
                    help='wisdom dir to write @nat/@nat2d into (default = canonical generated/)')
    ap.add_argument('--force', action='store_true', help='recalibrate cells that already have @nat/@nat2d')
    ap.add_argument('--ks', default='4,32', help='1D K values to calibrate (comma list). default 4,32')
    ap.add_argument('--max-n', type=int, default=2048,
                    help='skip cells with N (or N1) > this. default 2048 (keeps the run short; the large-N '
                         'tail dominates wall time)')
    ap.add_argument('--max-total', type=int, default=4_000_000, help='skip cells with N*K > this')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    KS = set(int(x) for x in args.ks.split(',') if x.strip())

    def in_range(a, nk):
        return a <= args.max_n and nk <= args.max_total

    if not EXE.exists():
        print(f'[error] {EXE} not built. Run:\n  python build.py --src test/calibrate_nat.c --vfft --jit')
        return 1

    staging = Path(args.staging); staging.mkdir(parents=True, exist_ok=True)
    for f in (SPIKE, FFT2D):
        dst = staging / f
        if not dst.exists():
            shutil.copy2(GEN / f, dst)                 # seed staging from canonical (once)
    logpath = staging / 'calibrate_nat_grid.log'
    log = open(logpath, 'a', buffering=1)
    def out(msg):
        print(msg); log.write(msg + '\n')

    out(f'\n===== calibrate_nat_grid  dim={args.dim}  staging={staging}  {time.strftime("%Y-%m-%d %H:%M:%S")} =====')

    grid = []   # (dim, a, b, nk)
    if args.dim in ('1d','both'):
        cells = parse_scrambled_1d(staging / SPIKE)
        done = set() if args.force else done_nat(staging / SPIKE, '@nat')
        for (n,k) in cells:
            if k in KS and (n,k) not in done and in_range(n, n*k): grid.append(('1d', n, k, n*k))
    if args.dim in ('2d','both'):
        cells = list(dict.fromkeys(parse_scrambled_2d(staging / FFT2D) + NARROW_2D))
        done = set() if args.force else done_nat(staging / FFT2D, '@nat2d')
        for (a,b) in cells:
            if (a,b) not in done and in_range(a, a*b): grid.append(('2d', a, b, a*b))
    grid.sort(key=lambda g: g[3])                      # smallest first (fast feedback)

    out(f'{len(grid)} cells to calibrate (skipping already-@nat cells; --force to redo).')
    if args.dry_run:
        for (dim,a,b,nk) in grid: out(f'  {dim} {a} {b}')
        return 0

    t_start = time.time()
    for i,(dim,a,b,nk) in enumerate(grid, 1):
        st, secs, verdict = run_cell(dim, a, b, staging, log)
        cool = cooldown_for(nk)
        out(f'[{i}/{len(grid)}] {dim} {a}x{b}: {st} ({secs:.0f}s)  {verdict}   cool {cool}s')
        time.sleep(cool)
    tgt = 'CANONICAL' if Path(args.staging) == GEN else str(staging)
    out(f'DONE {len(grid)} cells in {(time.time()-t_start)/60:.1f} min. '
        f'@nat/@nat2d records written to {tgt}: {staging/SPIKE} + {staging/FFT2D}.')
    log.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
