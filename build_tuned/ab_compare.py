"""
ab_compare.py — A/B compare production stride-fft wisdom against the
new-core tuned wisdom on (N, K) cells they share.

Both files are produced by the same `_stride_refine_bench` protocol
(10 warmup + 5 trials × reps clamped to [20, 100000], take min), so the
`best_ns` columns are directly comparable. Production wisdom comes from
src/stride-fft/bench/bench_planner.c (Phase 1); new-core wisdom comes
from build_tuned/calibrate_tuned.c.

Usage:
    python build_tuned/ab_compare.py
    python build_tuned/ab_compare.py --prod path/to/vfft_wisdom.txt \\
                                     --tuned path/to/vfft_wisdom_tuned.txt
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent

DEFAULT_PROD  = ROOT / 'src' / 'stride-fft' / 'bench' / 'vfft_wisdom.txt'
DEFAULT_TUNED = HERE / 'vfft_wisdom_tuned.txt'
DEFAULT_INFO  = HERE / 'vfft_wisdom_tuned_codelets.csv'
DEFAULT_OUT   = HERE / 'ab_comparison.csv'


def parse_wisdom(path: Path) -> dict[tuple[int, int], dict]:
    """Load v3 or v4 wisdom file. Returns {(N, K): {factors, best_ns, ...}}.

    v3 columns: N K nf factors... best_ns use_blocked split_stage block_groups
    v4 columns: ...above + use_dif_forward
    """
    if not path.exists():
        raise FileNotFoundError(f'wisdom file not found: {path}')
    out: dict[tuple[int, int], dict] = {}
    version_seen = False
    with path.open('r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('@'):
                if line.startswith('@version'):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1] in ('3', '4'):
                        version_seen = True
                    else:
                        raise ValueError(
                            f'{path}: expected @version 3 or 4, got: {line}')
                continue
            if not version_seen:
                raise ValueError(f'{path}: missing @version header')
            tokens = line.split()
            if len(tokens) < 5:
                continue
            N      = int(tokens[0])
            K      = int(tokens[1])
            nf     = int(tokens[2])
            if len(tokens) < 3 + nf + 1:
                continue
            factors = [int(t) for t in tokens[3:3 + nf]]
            best_ns = float(tokens[3 + nf])
            entry = {
                'N': N, 'K': K,
                'factors': factors,
                'best_ns': best_ns,
                'use_blocked': 0, 'split_stage': 0, 'block_groups': 0,
                'use_dif_forward': 0,
            }
            if len(tokens) >= 3 + nf + 4:
                entry['use_blocked']  = int(tokens[3 + nf + 1])
                entry['split_stage']  = int(tokens[3 + nf + 2])
                entry['block_groups'] = int(tokens[3 + nf + 3])
            if len(tokens) >= 3 + nf + 5:
                entry['use_dif_forward'] = int(tokens[3 + nf + 4])
            out[(N, K)] = entry
    return out


def parse_codelets(path: Path) -> dict[tuple[int, int], dict]:
    """Load the sidecar CSV from calibrate_tuned.c. Returns {(N,K): {...}}."""
    if not path.exists():
        return {}
    out: dict[tuple[int, int], dict] = {}
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                key = (int(row['N']), int(row['K']))
            except (KeyError, ValueError):
                continue
            out[key] = row
    return out


def fmt_factors(factors: list[int]) -> str:
    return 'x'.join(str(f) for f in factors)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--prod',  default=str(DEFAULT_PROD))
    ap.add_argument('--tuned', default=str(DEFAULT_TUNED))
    ap.add_argument('--info',  default=str(DEFAULT_INFO),
                    help='Sidecar codelets CSV from calibrate_tuned.c')
    ap.add_argument('--out',   default=str(DEFAULT_OUT))
    ap.add_argument('--regression-threshold', type=float, default=1.05,
                    help='Flag a cell as regression when tuned/prod >= this. '
                         'Default 1.05 (5%% slower than production).')
    args = ap.parse_args()

    prod_path  = Path(args.prod).resolve()
    tuned_path = Path(args.tuned).resolve()
    info_path  = Path(args.info).resolve()
    out_path   = Path(args.out).resolve()

    print(f'production wisdom : {prod_path}')
    print(f'tuned wisdom      : {tuned_path}')
    print(f'tuned codelets    : {info_path}')
    print(f'output            : {out_path}')
    print()

    try:
        prod  = parse_wisdom(prod_path)
        tuned = parse_wisdom(tuned_path)
    except (FileNotFoundError, ValueError) as e:
        print(f'[error] {e}', file=sys.stderr)
        return 2

    codelets = parse_codelets(info_path)

    shared = sorted(set(prod) & set(tuned))
    only_prod  = sorted(set(prod)  - set(tuned))
    only_tuned = sorted(set(tuned) - set(prod))

    print(f'shared cells   : {len(shared)}')
    print(f'only in prod   : {len(only_prod)}')
    print(f'only in tuned  : {len(only_tuned)}')
    print()

    rows: list[dict] = []
    n_regressions = 0
    n_wins        = 0
    n_tie         = 0

    for key in shared:
        N, K = key
        p = prod[key]
        t = tuned[key]
        ratio = t['best_ns'] / p['best_ns'] if p['best_ns'] > 0 else float('inf')

        # Speedup framing: > 1.0 means tuned is faster than prod.
        speedup = p['best_ns'] / t['best_ns'] if t['best_ns'] > 0 else 0.0

        is_regression = ratio >= args.regression_threshold
        if is_regression:
            n_regressions += 1
            verdict = 'REGRESSION'
        elif ratio <= 1 / args.regression_threshold:
            n_wins += 1
            verdict = 'WIN'
        else:
            n_tie += 1
            verdict = 'tie'

        info = codelets.get(key, {})

        # Blocked-executor flag for each side. Wisdom file has the
        # use_blocked/split/groups columns; sidecar CSV duplicates them.
        def fmt_blocked(entry: dict) -> str:
            if entry.get('use_blocked'):
                return f'blk@s={entry["split_stage"]},g={entry["block_groups"]}'
            return ''

        # DIF orientation flag (v4+). Production is always v3 (DIT only).
        tuned_dif = bool(t.get('use_dif_forward', 0))

        rows.append({
            'N': N, 'K': K,
            'prod_factors':   fmt_factors(p['factors']),
            'tuned_factors':  fmt_factors(t['factors']),
            'tuned_codelets': info.get('codelets', ''),
            'tuned_method':   info.get('method', ''),
            'prod_blocked':   fmt_blocked(p),
            'tuned_blocked':  fmt_blocked(t),
            'tuned_dif':      'DIF' if tuned_dif else '',
            'prod_ns':  f'{p["best_ns"]:.2f}',
            'tuned_ns': f'{t["best_ns"]:.2f}',
            'speedup':  f'{speedup:.3f}',
            'ratio_t/p': f'{ratio:.3f}',
            'verdict':  verdict,
        })

    # Write CSV
    if rows:
        with out_path.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f'wrote {len(rows)} rows to {out_path}')
    else:
        print('no shared cells — nothing to write')

    # Console summary
    print()
    print(f'{"N":>6} {"K":>4} {"prod_ns":>11} {"tuned_ns":>11} {"speedup":>8} {"verdict":>10}  factors_p / factors_t / codelets_t [tags]')
    print('-' * 130)
    for r in rows:
        tags = []
        if r['prod_blocked'] or r['tuned_blocked']:
            tags.append(f'P:{r["prod_blocked"] or "-"} | T:{r["tuned_blocked"] or "-"}')
        if r['tuned_dif']:
            tags.append(r['tuned_dif'])
        tag_str = f' [{"; ".join(tags)}]' if tags else ''
        print(f'{r["N"]:>6} {r["K"]:>4} {r["prod_ns"]:>11} {r["tuned_ns"]:>11} '
              f'{r["speedup"]:>8} {r["verdict"]:>10}  '
              f'{r["prod_factors"]} / {r["tuned_factors"]} / {r["tuned_codelets"]}{tag_str}')

    print()
    print(f'wins        : {n_wins}')
    print(f'ties        : {n_tie}')
    print(f'regressions : {n_regressions}')

    return 1 if n_regressions > 0 else 0


if __name__ == '__main__':
    sys.exit(main())
