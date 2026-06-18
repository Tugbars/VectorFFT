"""
run_bench.py - paper-grade dag-vs-MKL 1D C2C sweep, ISOLATED per cell.

Why isolated: a single in-process loop has cross-cell cache/thermal carryover
that cachebust() can't clear (the bench itself says "trust only isolated
numbers"). So we spawn ONE FRESH PROCESS per (N,K) cell — mirroring calibrate.py's
discipline — pinned to core 2, High priority, with:
  - N*K-scaled cooldown between cells (reset the thermal baseline; big cells
    heat-soak more, like the calibrator's N*K gate),
  - per-cell ALTERNATING engine order (flip = cell_index % 2) so the residual
    within-cell vfft-vs-MKL order bias averages out across the suite,
  - an inter-engine cooldown (--cool-ms) so each engine starts from a comparable
    baseline (the legacy fixed-order loop measured MKL on a warmed core -> ratio
    optimistic for us),
  - an inter-trial idle for big cells (VFFT_TRIAL_PACE_MS) so the best-of-5 min
    reflects a cooler core.

Every cell appends one row to ONE CSV (header written here once); resumable
(skips (N,K) already in the CSV). Prints a per-K summary at the end.

usage:
  python build_tuned/run_bench.py                       # K=4,32,256 from spike_wisdom
  python build_tuned/run_bench.py --K 4,256             # subset of K
  python build_tuned/run_bench.py --no-primes --dry-run
  python build_tuned/run_bench.py --csv C:/tmp/run1.csv
"""
from __future__ import annotations
import argparse, os, statistics, subprocess, sys, time
from pathlib import Path

import calibrate as cal   # verify_power(), HIGH_PRIORITY discipline, paths

HERE  = Path(__file__).parent.resolve()
ROOT  = HERE.parent
DAG   = ROOT / 'src' / 'dag-fft-compiler'
BENCH_SRC = HERE / 'benches' / 'bench_1d_vs_mkl.c'
BENCH_EXE = HERE / 'benches' / ('bench_1d_vs_mkl.exe' if os.name == 'nt' else 'bench_1d_vs_mkl')
SPIKE = DAG / 'generator' / 'generated' / 'spike_wisdom.txt'

# Must match bench_1d_vs_mkl.c's MAX_TOTAL_ELEMS (cells above it the bench SKIPs).
MAX_TOTAL_ELEMS = 16_777_216
# Prime cells benched via the override (Rader/Bluestein) path — must match the
# prime_N[] list in bench_1d_vs_mkl.c.
PRIME_N = [127, 251, 257, 401, 641, 1009, 2801, 4001,
            47,  59,  83, 107,  167,  179,  263,  311]
# MKL runtime dll (mkl_rt.2.dll) + mingw runtime (libwinpthread) must be on PATH.
MKL_BIN   = r'C:\Program Files (x86)\Intel\oneAPI\mkl\latest\bin'
MINGW_BIN = r'C:\mingw152\mingw64\bin'

CSV_HEADER = "N,K,plan,path,vfft_ns,mkl_ns,vfft_gflops,ratio_vs_mkl,rt_err\n"


def wisdom_cells(path: Path, kset: set[int]) -> set[tuple[int, int]]:
    s: set[tuple[int, int]] = set()
    if not path.exists():
        return s
    for ln in path.read_text(encoding='utf-8', errors='replace').splitlines():
        t = ln.strip()
        if not t or t[0] in '#@':
            continue
        p = t.split()
        if len(p) < 2:
            continue
        try:
            N, K = int(p[0]), int(p[1])
        except ValueError:
            continue
        if K in kset:
            s.add((N, K))
    return s


def csv_cells(path: Path) -> set[tuple[int, int]]:
    s: set[tuple[int, int]] = set()
    if not path.exists():
        return s
    for ln in path.read_text(encoding='utf-8', errors='replace').splitlines()[1:]:
        p = ln.split(',')
        if len(p) < 2:
            continue
        try:
            s.add((int(p[0]), int(p[1])))
        except ValueError:
            pass
    return s


def cooldown_for(nk: int) -> float:
    if nk < 100_000:   return 3.0
    if nk < 1_000_000: return 6.0
    return 12.0


def trial_pace_for(nk: int) -> int:
    return 50 if nk > 1_000_000 else 0


def build():
    print('[build] bench_1d_vs_mkl (--mkl --jit) ...', flush=True)
    r = subprocess.run([sys.executable, str(HERE / 'build.py'),
                        '--src', str(BENCH_SRC), '--mkl', '--jit', '--compile'])
    if r.returncode != 0 or not BENCH_EXE.exists():
        print('[build] FAILED', file=sys.stderr); sys.exit(1)


def run_cell(N, K, csv, core, cool_ms, flip, trial_pace, wisdom):
    env = os.environ.copy()
    env['PATH'] = MKL_BIN + os.pathsep + MINGW_BIN + os.pathsep + env.get('PATH', '')
    env['VFFT_TRIAL_PACE_MS'] = str(trial_pace)
    flags = subprocess.HIGH_PRIORITY_CLASS if os.name == 'nt' else 0
    # bench args: wisdom csv pace_ms N K cool_ms flip core
    p = subprocess.Popen([str(BENCH_EXE), str(wisdom), str(csv), '0',
                          str(N), str(K), str(cool_ms), str(flip), str(core)],
                         env=env, creationflags=flags)
    p.wait()
    return p.returncode


def summarize(csv: Path):
    if not csv.exists():
        return
    rows = []
    for ln in csv.read_text(encoding='utf-8', errors='replace').splitlines()[1:]:
        p = ln.split(',')
        if len(p) < 8:
            continue
        try:
            rows.append((int(p[0]), int(p[1]), float(p[7])))   # N, K, ratio
        except ValueError:
            pass
    if not rows:
        return
    print('\n' + '=' * 56)
    print(f' SUMMARY ({len(rows)} cells)   ratio = MKL_ns / dag_ns  (>1 = dag faster)')
    print('=' * 56)
    bykey = {}
    for N, K, r in rows:
        bykey.setdefault(K, []).append(r)
    for K in sorted(bykey):
        rs = [r for r in bykey[K] if r > 0]
        if not rs:
            continue
        wins = sum(1 for r in rs if r > 1.0)
        print(f'  K={K:<4}  n={len(rs):<3}  wins={wins}/{len(rs)}  '
              f'median={statistics.median(rs):.2f}x  min={min(rs):.2f}x  max={max(rs):.2f}x')
    allr = [r for _, _, r in rows if r > 0]
    wins = sum(1 for r in allr if r > 1.0)
    print(f'  ALL    n={len(allr):<3}  wins={wins}/{len(allr)}  '
          f'median={statistics.median(allr):.2f}x  min={min(allr):.2f}x  max={max(allr):.2f}x')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--K', default='4,32,256', help='comma list of batch widths')
    ap.add_argument('--core', type=int, default=2)
    ap.add_argument('--cool-ms', type=int, default=250, help='inter-engine idle (order-bias)')
    ap.add_argument('--csv', default=str(HERE / 'benches' / 'vfft_perf_tuned_1d.csv'))
    ap.add_argument('--wisdom', default=str(SPIKE))
    ap.add_argument('--max-nk', type=int, default=MAX_TOTAL_ELEMS)
    ap.add_argument('--no-primes', action='store_true')
    ap.add_argument('--fresh', action='store_true', help='overwrite CSV (else resume/append)')
    ap.add_argument('--skip-build', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    kset = set(int(x) for x in args.K.split(','))
    csv = Path(args.csv)
    wisdom = Path(args.wisdom)

    cells = set(wisdom_cells(wisdom, kset))
    if not args.no_primes:
        for K in kset:
            for N in PRIME_N:
                cells.add((N, K))
    cells = sorted((nk for nk in cells if nk[0] * nk[1] <= args.max_nk),
                   key=lambda nk: (nk[0] * nk[1], nk[1], nk[0]))

    done = set() if args.fresh else csv_cells(csv)
    todo = [nk for nk in cells if nk not in done]

    print('=' * 64)
    print(f' run_bench  K={sorted(kset)}  core={args.core}  cool-ms={args.cool_ms}')
    print(f' wisdom={wisdom.name}  csv={csv}')
    print(f' cells={len(cells)}  already-in-csv={len(cells)-len(todo)}  TODO={len(todo)}')
    print(f' order: N*K ascending; engine order ALTERNATES per cell (flip=i%2)')
    print('=' * 64)
    for i, (N, K) in enumerate(todo):
        print(f'  [{i+1:>3}/{len(todo)}] N={N:<7} K={K:<4} (N*K={N*K}) flip={i%2}')
    if args.dry_run:
        print('[dry-run] not executing.'); return

    cal.verify_power()
    if not args.skip_build:
        build()

    if args.fresh or not csv.exists():
        csv.parent.mkdir(parents=True, exist_ok=True)
        csv.write_text(CSV_HEADER, encoding='ascii')

    t0 = time.time()
    for i, (N, K) in enumerate(todo):
        nk = N * K
        flip = i % 2
        print(f'\n----- [{i+1}/{len(todo)}] N={N} K={K} (N*K={nk}) flip={flip} -----', flush=True)
        rc = run_cell(N, K, csv, args.core, args.cool_ms, flip, trial_pace_for(nk), wisdom)
        if rc != 0:
            print(f'  [warn] cell N={N} K={K} returned {rc}', file=sys.stderr)
        if i < len(todo) - 1:
            time.sleep(cooldown_for(nk))
    print(f'\n[done] elapsed {(time.time()-t0)/60:.1f} min.  CSV -> {csv}')
    summarize(csv)


if __name__ == '__main__':
    main()
