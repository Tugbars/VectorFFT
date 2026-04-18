"""
select_codelets.py — pick winners per (R, ios, me, direction) region.

Reads measurements.json, applies argmin over candidates at each sweep point,
uses a configurable tie threshold (default 2%) to prefer simpler variants
when close calls would cause selection-table churn.

Output: selection.json listing only the candidates that won at least one
region. Downstream emit_selector.py uses this to decide which .h files to
ship and what the lookup function should return.

Tie-breaking preference order (simpler first):
  ct_t1_dit > ct_t1s_dit > ct_t1_dit_log3 > ct_t1_buf_dit
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
import candidates


# Simpler-first preference. Lower index = preferred when within tie threshold.
FAMILY_PREFERENCE = [
    'ct_t1_dit',
    'ct_t1_dif',
    'ct_t1s_dit',
    'ct_t1_dit_log3',
    'ct_t1_dit_prefetch',
    'ct_t1_ladder_dit',
    'ct_t1_buf_dit',
]


def family_rank(family: str) -> int:
    try:
        return FAMILY_PREFERENCE.index(family)
    except ValueError:
        return len(FAMILY_PREFERENCE)  # unknown families come last


def id_to_family(cid: str) -> str:
    """Extract family name from candidate id ('ct_t1_buf_dit__avx2__...' → 'ct_t1_buf_dit')."""
    return cid.split('__')[0]


def id_to_isa(cid: str) -> str:
    return cid.split('__')[1]


def pick_winner(measurements_at_point, direction: str, tie_threshold: float):
    """
    measurements_at_point: list of {id, ios, me, fwd_ns, bwd_ns}
    Returns winner id, or None if no valid candidates.
    """
    key = f'{direction}_ns'
    valid = [m for m in measurements_at_point if m[key] > 0]
    if not valid:
        return None

    best_ns = min(m[key] for m in valid)
    # Anyone within tie_threshold of best is a "near-winner"
    threshold_ns = best_ns * (1.0 + tie_threshold)
    contenders = [m for m in valid if m[key] <= threshold_ns]

    # Among contenders, prefer simpler family, then faster
    contenders.sort(key=lambda m: (family_rank(id_to_family(m['id'])), m[key]))
    return contenders[0]['id']


def _load_measurements(path: Path) -> tuple:
    """Load measurements from either format:
      - JSON (R=16 style): {'avx512_available': bool, 'measurements': [...]}
        where each measurement has fwd_ns and bwd_ns paired in one record.
      - JSONL (R=32 style): one {id, ios, me, dir, ns} per line,
        emitted incrementally by the bench harness for checkpointing.
    Returns (measurements_list, avx512_ok) where measurements_list has
    {id, ios, me, fwd_ns, bwd_ns} records regardless of input format.
    """
    text = path.read_text(encoding='utf-8')
    if path.suffix == '.jsonl' or (text.lstrip().startswith('{') and '\n{' in text.strip()):
        # JSONL format: one record per line, separate fwd/bwd
        grouped = {}
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r['id'], r['ios'], r['me'])
            grouped.setdefault(key, {'id': r['id'], 'ios': r['ios'], 'me': r['me']})
            grouped[key][f"{r['dir']}_ns"] = r['ns']
        # Keep only fully-paired records (both fwd_ns and bwd_ns present)
        measurements = [m for m in grouped.values()
                        if 'fwd_ns' in m and 'bwd_ns' in m]
        # avx512 availability: infer from presence of any avx512 candidate
        avx512_ok = any('__avx512' in m['id'] for m in measurements)
        return measurements, avx512_ok
    else:
        data = json.loads(text)
        return data['measurements'], data.get('avx512_available', True)


def select(measurements_path: Path, output_path: Path,
           tie_threshold: float = 0.02, verbose: bool = True):
    measurements, avx512_ok = _load_measurements(measurements_path)

    # Index measurements by (ios, me, isa)
    by_point_isa = defaultdict(list)
    for m in measurements:
        isa = id_to_isa(m['id'])
        by_point_isa[(m['ios'], m['me'], isa)].append(m)

    # For each (ios, me, isa, direction), pick a winner
    decisions = []  # list of {ios, me, isa, direction, winner_id}
    for (ios, me, isa), pts in sorted(by_point_isa.items()):
        if isa == 'avx512' and not avx512_ok:
            continue
        for direction in ('fwd', 'bwd'):
            winner = pick_winner(pts, direction, tie_threshold)
            if winner:
                decisions.append({
                    'ios': ios, 'me': me, 'isa': isa,
                    'direction': direction, 'winner': winner,
                })

    # Gather the distinct set of winning candidate ids
    winning_ids = sorted({d['winner'] for d in decisions})

    # Build output structure
    out = {
        'tie_threshold': tie_threshold,
        'decisions': decisions,
        'winners': winning_ids,
        'isas_measured': sorted({id_to_isa(i) for i in winning_ids}),
    }
    output_path.write_text(json.dumps(out, indent=2), encoding='utf-8')

    if verbose:
        print(f"[select] {len(decisions)} decisions, "
              f"{len(winning_ids)} distinct winners kept")
        # Report win counts per candidate
        win_counts = defaultdict(int)
        for d in decisions:
            win_counts[d['winner']] += 1
        print("[select] win counts (candidate -> regions won):")
        for cid, count in sorted(win_counts.items(), key=lambda x: -x[1]):
            print(f"  {count:>4} wins  {cid}")
        print(f"[select] written: {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default=None,
                    help='measurements file (auto-detects jsonl vs json; '
                         'defaults to measurements.jsonl if present, '
                         'else measurements.json)')
    ap.add_argument('--output', default='selection.json')
    ap.add_argument('--tie-threshold', type=float, default=0.02,
                    help='fractional margin; within this is considered a tie '
                         '(default 0.02 = 2%%)')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    # Auto-detect input format: prefer .jsonl (R=32 incremental) if present
    if args.input is None:
        if Path('measurements.jsonl').exists():
            args.input = 'measurements.jsonl'
        else:
            args.input = 'measurements.json'
    select(Path(args.input), Path(args.output),
           tie_threshold=args.tie_threshold,
           verbose=not args.quiet)


if __name__ == '__main__':
    main()
