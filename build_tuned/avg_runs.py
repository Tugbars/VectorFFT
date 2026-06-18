"""
avg_runs.py - average two isolated dag-vs-MKL runs to cut run-to-run variance.

Cells are matched on (N, K, plan). A cell measured in BOTH runs gets its vfft_ns
and mkl_ns averaged (ratio + gflops recomputed from the averages); a cell in only
one run is passed through. Because we match on the PLAN too, run #1's bug rows
(K=32/256 CT cells mislabeled K=4 with a different factorization) match nothing in
the clean run #2 and are dropped automatically — no special-casing.

run #1 source can be either a CSV (your saved copy) or the run's .output LOG
(reconstructed: the table prints ns at the same integer precision as the CSV).

usage:
  python build_tuned/avg_runs.py --new <run2.csv> --old <run1.csv|run1.output> --out <avg.csv>
"""
from __future__ import annotations
import argparse, math, re, statistics
from pathlib import Path

HDR = "N,K,plan,path,vfft_ns,mkl_ns,vfft_gflops,ratio_vs_mkl,rt_err\n"


def load_csv(path: Path) -> dict:
    """(N,K,plan) -> (vfft_ns, mkl_ns, path, rt_err)."""
    d = {}
    for ln in path.read_text(encoding='utf-8', errors='replace').splitlines()[1:]:
        p = ln.split(',')
        if len(p) < 9:
            continue
        try:
            N, K, plan, pth = int(p[0]), int(p[1]), p[2], p[3]
            vns, mns, rt = float(p[4]), float(p[5]), float(p[8])
        except ValueError:
            continue
        d[(N, K, plan)] = (vns, mns, pth, rt)
    return d


def load_log(path: Path, ct_k4: bool = False) -> dict:
    """Reconstruct (N,K,plan) -> (vfft_ns, mkl_ns, path, rt_err) from a run log:
    pair each '----- [i/n] N=X K=Y' header with the next data row.

    HAZARD: the header K is the REQUESTED K. If the source run had the K-logging
    bug (run #1: CT cells measured at K=4 but requested at K=32/256), keying by
    the header K would make those rows FALSE-MATCH the clean run's correct rows
    and contaminate the average. Pass ct_k4=True to replicate run #1's actual CSV
    keys — CT cells keyed at K=4 (where they were really measured), override
    (prime) cells at the true K — so bugged CT rows match nothing in the clean run."""
    d = {}
    curK = None
    hdr = re.compile(r'^-----\s*\[\d+/\d+\]\s*N=(\d+)\s+K=(\d+)')
    for ln in path.read_text(encoding='utf-8', errors='replace').splitlines():
        m = hdr.match(ln.strip())
        if m:
            curK = int(m.group(2))
            continue
        t = ln.split()
        # data row: N plan path vns mns vgf ratiox rterr  (8 tokens, ratio ends 'x')
        if curK is None or len(t) != 8 or not t[0].isdigit() or not t[6].endswith('x'):
            continue
        try:
            N, plan, pth = int(t[0]), t[1], t[2]
            vns, mns, rt = float(t[3]), float(t[4]), float(t[7])
        except ValueError:
            continue
        key_K = 4 if (ct_k4 and plan != '[override]') else curK
        d[(N, key_K, plan)] = (vns, mns, pth, rt)
        curK = None   # one data row per header
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--new', required=True, help='clean run #2 CSV')
    ap.add_argument('--old', required=True, help='run #1 CSV or .output log')
    ap.add_argument('--out', required=True, help='averaged CSV out')
    ap.add_argument('--old-ct-k4', action='store_true',
                    help='source run logged CT cells at K=4 (run #1 K-bug); only with --old <log>.output')
    args = ap.parse_args()

    new = load_csv(Path(args.new))
    op = Path(args.old)
    old = load_log(op, ct_k4=args.old_ct_k4) if op.suffix == '.output' else load_csv(op)
    print(f'new(run2)={len(new)} cells   old(run1)={len(old)} cells')

    lines = [HDR.rstrip('\n')]
    averaged = single = 0
    rows = []
    for key in sorted(new, key=lambda k: (k[0] * k[1], k[1], k[0])):
        N, K, plan = key
        vns, mns, pth, rt = new[key]
        if key in old:
            ovns, omns, _, ort = old[key]
            vns = (vns + ovns) / 2.0
            mns = (mns + omns) / 2.0
            rt = max(rt, ort)
            averaged += 1
            tag = 'avg2'
        else:
            single += 1
            tag = pth
        ratio = (mns / vns) if vns > 0 else 0.0
        vgf = (5.0 * N * math.log2(N) * K / vns) if vns > 0 and N > 1 else 0.0
        lines.append(f'{N},{K},{plan},{tag},{vns:.0f},{mns:.0f},{vgf:.3f},{ratio:.3f},{rt:.3e}')
        rows.append((N, K, ratio))
    Path(args.out).write_text('\n'.join(lines) + '\n', encoding='ascii')

    print(f'averaged(2 runs)={averaged}   run2-only={single}   -> {args.out}')
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


if __name__ == '__main__':
    main()
