"""
select_and_emit.py — read bench measurements, emit dispatchers + plan wisdom.

Produces (in generated/):
  vfft_r{R}_t1_dit_dispatch.h         flat protocol dispatcher
  vfft_r{R}_t1_dit_log3_dispatch.h    log3 protocol dispatcher
  vfft_r{R}_t1s_dit_dispatch.h        t1s protocol dispatcher   (if applicable)
  vfft_r{R}_plan_wisdom.h             protocol selection for the planner
  vfft_r{R}_report.md                 human-readable summary

Dispatcher naming (for Phase 4/5 wiring — not yet final):
  The dispatcher exposes `vfft_r{R}_{protocol}_dispatch_{dir}_{isa}`.
  Production wiring later rebinds these to the canonical symbol names the
  registry expects (`radix{R}_t1_dit_fwd_avx2` etc). Until then the
  dispatchers are standalone and testable in isolation.

Selection logic:
  - Within a protocol: for each (isa, me, ios, dir), pick the fastest variant.
  - Tie threshold: within 2% of best counts as a tie; prefer simpler variant
    (ct_t1_dit > ct_t1_dit_log1 > ct_t1_dit_u2 > ...).
  - Across protocols: for plan wisdom, at each (me, ios), compare the best of
    each protocol. Emit decision functions.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).parent.resolve()


# ═══════════════════════════════════════════════════════════════
# Tie-breaking preference: simpler variant wins close calls.
# Lower index = preferred.
# ═══════════════════════════════════════════════════════════════

FAMILY_PREFERENCE = [
    'ct_t1_dit',          # flat baseline — simplest
    'ct_t1_dit_log1',     # 2 loads + derive w3
    'ct_t1_dit_log3',     # 1 load + derive w2, w3
    'ct_t1_dit_u2',       # unroll — trickier codegen
    'ct_t1s_dit',         # broadcast — different protocol
]

def family_rank(variant: str) -> int:
    try:
        return FAMILY_PREFERENCE.index(variant)
    except ValueError:
        return 1000


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def _load_measurements(jsonl: Path) -> list[dict]:
    out = []
    with jsonl.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                m = json.loads(line)
            except Exception:
                continue
            if m.get('skipped'):
                continue
            if 'ns' not in m:
                continue
            out.append(m)
    return out


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ═══════════════════════════════════════════════════════════════
# Winner selection
# ═══════════════════════════════════════════════════════════════

def pick_winner(entries: list[dict], tie_threshold: float = 0.02) -> dict:
    """Pick the best entry. Entries within tie_threshold of the best are
    considered tied; among ties, prefer the simpler variant."""
    if not entries:
        return None
    best_ns = min(e['ns'] for e in entries)
    limit = best_ns * (1.0 + tie_threshold)
    contenders = [e for e in entries if e['ns'] <= limit]
    contenders.sort(key=lambda e: (family_rank(e['variant']), e['ns']))
    return contenders[0]


def _group_per_point(ms: list[dict], dispatcher_fn) -> dict:
    """Group measurements by (isa, dispatcher, me, ios, dir) -> list of entries.

    dispatcher_fn: callable(variant) -> dispatcher_key. Passed in so the
    loader can reach back to the candidates module.
    """
    g = defaultdict(list)
    for m in ms:
        variant = m['variant']
        try:
            disp = dispatcher_fn(variant)
        except Exception:
            # Handicap experiments (log1_tight) and unknown variants: skip
            # the dispatcher grouping; they remain available for the
            # handicap report via raw measurement list.
            continue
        k = (m['isa'], disp, m['me'], m['ios'], m['dir'])
        g[k].append(m)
    return g


def winners_per_dispatcher(ms: list[dict], dispatcher_fn) -> dict:
    """Return {(isa, dispatcher, me, ios, dir) -> winner_entry}."""
    grouped = _group_per_point(ms, dispatcher_fn)
    return {k: pick_winner(v) for k, v in grouped.items() if v}


def winners_cross_protocol(ms: list[dict]) -> dict:
    """For each (isa, me, ios, dir), return {protocol -> best_entry}.
    Used to build plan-wisdom decision functions. This groups by PROTOCOL
    (twiddle layout) — a different axis than dispatcher (codelet family).

    The same codelet can belong to different protocols depending on how
    its twiddle buffer is allocated (the log1_tight handicap experiment
    uses 'log1_tight' as a synthetic protocol name to keep it separate
    from real 'flat')."""
    g: dict = defaultdict(list)
    for m in ms:
        k = (m['isa'], m['protocol'], m['me'], m['ios'], m['dir'])
        g[k].append(m)
    out: dict = defaultdict(dict)
    for (isa, protocol, me, ios, d), entries in g.items():
        w = pick_winner(entries)
        if w is not None:
            out[(isa, me, ios, d)][protocol] = w
    return dict(out)


# ═══════════════════════════════════════════════════════════════
# Dispatcher emit (per protocol, per ISA, per direction)
# ═══════════════════════════════════════════════════════════════

def _rules_from_winners(winners: list[tuple[int, int, str, float]],
                        fallback_variant: str) -> list[dict]:
    """Given [(me, ios, variant, ns), ...] for one (isa, protocol, dir),
    cluster into dispatch rules. Coarse-grained: one rule per me band,
    ios-aware only when bench shows ios-dependent winners within a band.

    Returns a list of dicts: [{'me_min', 'me_max', 'ios_pow2_variant',
    'ios_padded_variant'}]. Simple rules use the same variant regardless
    of ios; complex rules differentiate power-of-2 from padded strides.
    """
    if not winners:
        return [{'me_min': 0, 'me_max': 2**31 - 1,
                 'ios_pow2_variant': fallback_variant,
                 'ios_padded_variant': fallback_variant}]

    # Group by me.
    by_me: dict[int, list[tuple[int, str, float]]] = defaultdict(list)
    for (me, ios, v, ns) in winners:
        by_me[me].append((ios, v, ns))

    # For each me, compute the winning variant per ios-class.
    # ios class: 'pow2' if ios == me, else 'padded'.
    bands: list[tuple[int, str, str]] = []
    for me in sorted(by_me):
        pow2_winners = [(v, ns) for (ios, v, ns) in by_me[me] if ios == me]
        pad_winners  = [(v, ns) for (ios, v, ns) in by_me[me] if ios != me]
        pw = pick_variant_across(pow2_winners) if pow2_winners else None
        pd = pick_variant_across(pad_winners)  if pad_winners  else None
        bands.append((me, pw or fallback_variant, pd or fallback_variant))

    # Collapse consecutive bands with identical variants.
    rules: list[dict] = []
    for (me, pw, pd) in bands:
        if rules and rules[-1]['ios_pow2_variant'] == pw and \
                     rules[-1]['ios_padded_variant'] == pd:
            rules[-1]['me_max'] = me
        else:
            rules.append({'me_min': me, 'me_max': me,
                          'ios_pow2_variant': pw,
                          'ios_padded_variant': pd})
    # Extend the last rule's me_max to +inf so out-of-range points fall through.
    if rules:
        rules[-1]['me_max'] = 2**31 - 1

    return rules


def pick_variant_across(entries: list[tuple[str, float]]) -> str | None:
    """From [(variant, ns), ...], pick the simpler-preferring winner."""
    if not entries:
        return None
    best_ns = min(ns for (_, ns) in entries)
    limit = best_ns * 1.02
    cands = [v for (v, ns) in entries if ns <= limit]
    cands.sort(key=family_rank)
    return cands[0]


def emit_dispatch_header(R: int, dispatcher_key: str, isa: str,
                         candidates_mod, winners_by_disp: dict,
                         out_path: Path):
    """Emit `vfft_r{R}_{dispatcher}_dispatch_{isa}.h` with static inline
    fwd/bwd dispatchers that pick the winning variant per (me, ios).

    Dispatcher name: vfft_r{R}_{dispatcher_key}_dispatch_{dir}_{isa}.
    Variants sharing a dispatcher key compute the same mathematical
    function and are mutually exchangeable based on measured ns/call.
    """
    dispatcher_name = lambda d: f'vfft_r{R}_{dispatcher_key}_dispatch_{d}_{isa}'

    # Pick a fallback variant for the dispatcher: the first variant
    # declared for this dispatcher key in the candidates module.
    fallback_variant = None
    for c in candidates_mod.enumerate_all():
        if candidates_mod.dispatcher(c.variant) == dispatcher_key:
            fallback_variant = c.variant
            break
    if fallback_variant is None:
        raise RuntimeError(f'no variants for dispatcher {dispatcher_key}')

    lines = [
        f'/* vfft_r{R}_{dispatcher_key}_dispatch_{isa}.h',
        f' *',
        f' * Auto-generated codelet dispatcher for R={R} / dispatcher={dispatcher_key} / isa={isa}.',
        f' * Derived from a bench run on this host. The dispatcher picks the',
        f' * fastest variant per (me, ios) based on measured ns/call.',
        f' *',
        f' * To retune: re-run common/bench.py.',
        f' */',
        f'#ifndef VFFT_R{R}_{dispatcher_key.upper()}_DISPATCH_{isa.upper()}_H',
        f'#define VFFT_R{R}_{dispatcher_key.upper()}_DISPATCH_{isa.upper()}_H',
        f'',
        f'#include <stddef.h>',
        f'#include "fft_radix{R}_{isa}.h"',
        f'',
    ]

    for direction in ('fwd', 'bwd'):
        key = (isa, dispatcher_key, direction)
        winners = winners_by_disp.get(key, [])
        rules = _rules_from_winners(winners, fallback_variant=fallback_variant)

        lines += [
            f'static inline void {dispatcher_name(direction)}(',
            f'    double * __restrict__ rio_re, double * __restrict__ rio_im,',
            f'    const double * __restrict__ W_re, const double * __restrict__ W_im,',
            f'    size_t ios, size_t me)',
            f'{{',
        ]

        # Rule table as a comment block.
        lines.append(f'    /* dispatch rules (per bench):')
        for r in rules:
            upper = r['me_max']
            upper_str = '∞' if upper == 2**31 - 1 else str(upper)
            if r['ios_pow2_variant'] == r['ios_padded_variant']:
                lines.append(f'     *   me∈[{r["me_min"]}..{upper_str}]: {r["ios_pow2_variant"]}')
            else:
                lines.append(f'     *   me∈[{r["me_min"]}..{upper_str}] pow2 ios: {r["ios_pow2_variant"]}')
                lines.append(f'     *   me∈[{r["me_min"]}..{upper_str}] padded ios: {r["ios_padded_variant"]}')
        lines.append(f'     */')

        # Emit the actual branching.
        for i, r in enumerate(rules):
            cond_me = []
            if i > 0:
                cond_me.append(f'me >= {r["me_min"]}')
            if r['me_max'] != 2**31 - 1:
                cond_me.append(f'me <= {r["me_max"]}')
            prefix = '    if (' if i == 0 else '    else if ('
            if not cond_me:
                cond_line = None
            else:
                cond_line = ' && '.join(cond_me)

            same = (r['ios_pow2_variant'] == r['ios_padded_variant'])
            if cond_line:
                lines.append(f'{prefix}{cond_line}) {{')
            else:
                lines.append(f'    {{')
            if same:
                fn = candidates_mod.function_name(r['ios_pow2_variant'], isa, direction)
                lines.append(f'        {fn}(rio_re, rio_im, W_re, W_im, ios, me);')
                lines.append(f'        return;')
            else:
                fn_p2 = candidates_mod.function_name(r['ios_pow2_variant'], isa, direction)
                fn_pd = candidates_mod.function_name(r['ios_padded_variant'], isa, direction)
                lines.append(f'        if (ios == me) {{')
                lines.append(f'            {fn_p2}(rio_re, rio_im, W_re, W_im, ios, me);')
                lines.append(f'            return;')
                lines.append(f'        }} else {{')
                lines.append(f'            {fn_pd}(rio_re, rio_im, W_re, W_im, ios, me);')
                lines.append(f'            return;')
                lines.append(f'        }}')
            lines.append('    }')

        lines += [
            f'}}',
            f'',
        ]

    lines += [
        f'#endif /* VFFT_R{R}_{dispatcher_key.upper()}_DISPATCH_{isa.upper()}_H */',
        f'',
    ]

    out_path.write_text('\n'.join(lines), encoding='utf-8')


def _protocol_fallback_variant(protocol: str) -> str:
    # Legacy helper, retained for compatibility with any older imports.
    if protocol == 'flat': return 'ct_t1_dit'
    if protocol == 'log3': return 'ct_t1_dit_log3'
    if protocol == 't1s':  return 'ct_t1s_dit'
    raise ValueError(protocol)


# ═══════════════════════════════════════════════════════════════
# Plan wisdom emit (protocol selection functions)
# ═══════════════════════════════════════════════════════════════

def emit_plan_wisdom(R: int, cross: dict, out_path: Path, host_desc: str):
    """Produce static inline bool functions the planner queries to decide
    which protocol to use for a (me, ios) stage. Only AVX2 data is used
    for plan wisdom (the planner's decisions are protocol-level, not
    ISA-level; if an AVX-512 host's split points differ they can
    regenerate wisdom for that host)."""

    # Build a per-(me, ios, dir) table of protocol decisions.
    # Decision: at this (me, ios), which protocol's best-ns wins?
    decisions_fwd: list[tuple[int, int, str, float]] = []
    decisions_bwd: list[tuple[int, int, str, float]] = []
    for (isa, me, ios, d), by_protocol in cross.items():
        if isa != 'avx2':
            continue
        if not by_protocol:
            continue
        # Exclude log1_tight from plan-protocol comparison — it's not a
        # real production protocol; it was only for the handicap experiment.
        candidates_for_plan = {k: v for k, v in by_protocol.items()
                               if k != 'log1_tight'}
        if not candidates_for_plan:
            continue
        winning_proto = min(candidates_for_plan,
                            key=lambda p: candidates_for_plan[p]['ns'])
        winning_ns = candidates_for_plan[winning_proto]['ns']
        tgt = decisions_fwd if d == 'fwd' else decisions_bwd
        tgt.append((me, ios, winning_proto, winning_ns))

    lines = [
        f'/* vfft_r{R}_plan_wisdom.h',
        f' *',
        f' * Auto-generated plan-protocol selection for R={R}.',
        f' * Bench host: {host_desc}',
        f' *',
        f' * Each function returns 1 if the planner should use the named protocol',
        f' * for a stage with the given (me, ios), 0 otherwise.',
        f' *',
        f' * Derived from cross-protocol comparison at each sweep point. The',
        f' * planner should consult these AFTER the codelet-level dispatcher',
        f' * has been selected — they drive twiddle-table layout and the',
        f' * K-blocked execution path.',
        f' */',
        f'#ifndef VFFT_R{R}_PLAN_WISDOM_H',
        f'#define VFFT_R{R}_PLAN_WISDOM_H',
        f'',
        f'#include <stddef.h>',
        f'',
    ]

    # Helper: table of (me, ios) -> protocol strings for comments
    lines.append(f'/* Cross-protocol comparison (fwd direction, AVX2):')
    lines.append(f' *   me    ios   winning_protocol   winning_ns')
    for (me, ios, p, ns) in sorted(decisions_fwd)[:30]:  # cap length
        lines.append(f' *   {me:5d} {ios:5d}   {p:12s}   {ns:8.1f}')
    lines.append(f' */')
    lines.append(f'')

    # Emit three functions. Each maps (me, ios) -> 1 if protocol wins there.
    for wanted in ('log3', 't1s'):
        # Collect (me, ios) where wanted wins, using fwd decisions.
        wins = [(me, ios) for (me, ios, p, _) in decisions_fwd if p == wanted]
        fn_name = f'radix{R}_prefer_{wanted}'
        lines += [
            f'/* Should the planner use {wanted} protocol at (me, ios)? */',
            f'static inline int {fn_name}(size_t me, size_t ios) {{',
            f'    (void)ios;  /* may be unused if rules are me-only */',
        ]
        if not wins:
            lines += [
                f'    /* Bench showed {wanted} never wins on this host. */',
                f'    return 0;',
                f'}}',
                f'',
            ]
            continue

        # Summarize wins as me-ranges. Coarse: find the me-range in which
        # {wanted} wins a plurality of (me, ios) sweep points.
        wins_by_me: dict[int, int] = defaultdict(int)
        total_by_me: dict[int, int] = defaultdict(int)
        for (me, ios, p, _) in decisions_fwd:
            total_by_me[me] += 1
            if p == wanted:
                wins_by_me[me] += 1

        # Emit rules per me where {wanted} is majority winner.
        rule_mes = sorted(me for me in wins_by_me
                          if wins_by_me[me] >= total_by_me[me] / 2)
        if rule_mes:
            # Try a contiguous range [lo..hi]
            lo, hi = min(rule_mes), max(rule_mes)
            # Confirm contiguous within the sweep. If not, we use explicit enumeration.
            lines.append(f'    /* Bench wins at me ∈ {{{", ".join(str(m) for m in rule_mes)}}} */')
            if rule_mes == list(range(lo, hi + 1)) or len(rule_mes) <= 3:
                conds = ' || '.join(f'me == {m}' for m in rule_mes)
                lines.append(f'    if ({conds}) return 1;')
            else:
                # Use a range if contiguous in sweep grid terms, else enumerate
                lines.append(f'    if (me >= {lo} && me <= {hi}) return 1;')
        lines += [
            f'    return 0;',
            f'}}',
            f'',
        ]

    lines += [
        f'#endif /* VFFT_R{R}_PLAN_WISDOM_H */',
        f'',
    ]
    out_path.write_text('\n'.join(lines), encoding='utf-8')


# ═══════════════════════════════════════════════════════════════
# Human-readable report
# ═══════════════════════════════════════════════════════════════

def emit_report(R: int, ms: list[dict], cross: dict, out_path: Path,
                dispatcher_fn=None):
    lines = [
        f'# VectorFFT R={R} tuning report',
        f'',
        f'Total measurements: **{len(ms)}**',
        f'',
        f'## Cross-protocol winners (fwd direction)',
        f'',
        f'Best-of-protocol ns/call at each sweep point (informs plan-level',
        f'choice of twiddle-table layout).',
        f'',
        f'| isa | me | ios | flat | log3 | t1s | log1_tight | plan winner |',
        f'|---|---|---|---|---|---|---|---|',
    ]
    rows = []
    for (isa, me, ios, d), by_p in sorted(cross.items()):
        if d != 'fwd': continue
        flat  = by_p.get('flat', {}).get('ns', None)
        log3  = by_p.get('log3', {}).get('ns', None)
        t1s   = by_p.get('t1s',  {}).get('ns', None)
        tight = by_p.get('log1_tight', {}).get('ns', None)
        cells = [f'{v:.0f}' if v is not None else '—'
                 for v in (flat, log3, t1s, tight)]
        plan_candidates = {p: e['ns'] for p, e in by_p.items()
                           if p != 'log1_tight'}
        winner = min(plan_candidates, key=plan_candidates.get) if plan_candidates else '—'
        rows.append(f'| {isa} | {me} | {ios} | '
                    + ' | '.join(cells) + f' | {winner} |')
    lines += rows[:40]
    if len(rows) > 40:
        lines.append(f'| ... | ... ({len(rows) - 40} rows omitted) | | | | | | |')

    if dispatcher_fn is not None:
        lines += [
            f'',
            f'## Per-dispatcher winners (fwd)',
            f'',
            f'Within each dispatcher slot (variants that compute the same',
            f'mathematical function), which variant wins each point?',
            f'',
            f'| isa | dispatcher | me | ios | winner | ns |',
            f'|---|---|---|---|---|---|',
        ]
        per_disp = winners_per_dispatcher(ms, dispatcher_fn)
        for (isa, disp, me, ios, d), w in sorted(per_disp.items()):
            if d != 'fwd': continue
            lines.append(
                f'| {isa} | `{disp}` | {me} | {ios} | `{w["variant"]}` | {w["ns"]:.0f} |')

    # Handicap experiment: only relevant for R=4 (has log1_tight variant)
    if any(m.get('protocol') == 'log1_tight' for m in ms):
        lines += [
            f'',
            f'## log1 vs log1_tight (handicap experiment)',
            f'',
            f'Tests whether the log1 variant in flat protocol is handicapped',
            f'by a full (R-1)*me twiddle table vs a tight 2*me table. Same',
            f'codelet body; different harness allocation.',
            f'',
            f'| isa | me | ios | log1 (full) ns | log1_tight ns | delta % |',
            f'|---|---|---|---|---|---|',
        ]
        tight_cmp: list[tuple] = []
        for (isa, me, ios, d), by_p in cross.items():
            if d != 'fwd': continue
            flat_entries = [m for m in ms if m['isa']==isa and m['me']==me
                            and m['ios']==ios and m['dir']=='fwd'
                            and m['protocol']=='flat' and m['variant']=='ct_t1_dit_log1']
            tight_entries = [m for m in ms if m['isa']==isa and m['me']==me
                             and m['ios']==ios and m['dir']=='fwd'
                             and m['protocol']=='log1_tight']
            if not flat_entries or not tight_entries:
                continue
            a = flat_entries[0]['ns']
            b = tight_entries[0]['ns']
            delta = (b - a) / a * 100.0
            tight_cmp.append((isa, me, ios, a, b, delta))
        for (isa, me, ios, a, b, delta) in sorted(tight_cmp):
            sign = '+' if delta >= 0 else ''
            lines.append(f'| {isa} | {me} | {ios} | {a:.0f} | {b:.0f} | {sign}{delta:.1f}% |')

    out_path.write_text('\n'.join(lines), encoding='utf-8')


# ═══════════════════════════════════════════════════════════════
# Top-level driver
# ═══════════════════════════════════════════════════════════════

def emit_all(candidates_mod, measurements: Path, out_root: Path,
             host_desc: str = 'unknown'):
    R = candidates_mod.RADIX
    ms = _load_measurements(measurements)
    if not ms:
        raise SystemExit(f'no measurements found in {measurements}')
    print(f'  loaded {len(ms)} measurements')

    dispatcher_fn = candidates_mod.dispatcher  # variant -> dispatcher key

    # Per-dispatcher winner tables: {(isa, dispatcher, direction) ->
    # [(me, ios, winner_variant, ns), ...]}
    winners_by_disp: dict[tuple[str, str, str], list] = defaultdict(list)
    per_disp = winners_per_dispatcher(ms, dispatcher_fn)
    for (isa, disp, me, ios, d), w in per_disp.items():
        winners_by_disp[(isa, disp, d)].append((me, ios, w['variant'], w['ns']))

    cross = winners_cross_protocol(ms)

    out_root.mkdir(parents=True, exist_ok=True)

    # Emit one dispatcher header per (dispatcher, isa).
    disp_keys_by_isa: dict[str, set] = defaultdict(set)
    for (isa, disp, _) in winners_by_disp:
        disp_keys_by_isa[isa].add(disp)

    for isa in sorted(disp_keys_by_isa):
        for disp in sorted(disp_keys_by_isa[isa]):
            path = out_root / f'vfft_r{R}_{disp}_dispatch_{isa}.h'
            emit_dispatch_header(R, disp, isa, candidates_mod,
                                 winners_by_disp, path)
            print(f'  [emit] {path.name}')

    # Plan wisdom (cross-protocol comparison, unchanged).
    path = out_root / f'vfft_r{R}_plan_wisdom.h'
    emit_plan_wisdom(R, cross, path, host_desc)
    print(f'  [emit] {path.name}')

    # Report
    path = out_root / f'vfft_r{R}_report.md'
    emit_report(R, ms, cross, path, dispatcher_fn=dispatcher_fn)
    print(f'  [emit] {path.name}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--radix-dir', required=True)
    ap.add_argument('--measurements', default=None,
                    help='override path to measurements.jsonl')
    ap.add_argument('--out', default='generated',
                    help='output directory for emitted headers')
    ap.add_argument('--host-desc', default='',
                    help='description (cpu, compiler, date) embedded in headers')
    args = ap.parse_args()

    radix_dir = Path(args.radix_dir).resolve()
    cand_py = radix_dir / 'candidates.py'
    cand_mod = _load_module(cand_py, 'candidates')

    if args.measurements:
        ms_path = Path(args.measurements).resolve()
    else:
        # Default: bench_out/r{R}/measurements.jsonl in CWD
        ms_path = Path(f'bench_out/r{cand_mod.RADIX}/measurements.jsonl').resolve()

    out_dir = Path(args.out).resolve() / f'r{cand_mod.RADIX}'

    host = args.host_desc or os.environ.get('VFFT_HOST_DESC', '')
    if not host:
        try:
            with open('/proc/cpuinfo') as f:
                for line in f:
                    if line.startswith('model name'):
                        host = line.split(':', 1)[1].strip()
                        break
        except Exception:
            host = 'unknown'

    emit_all(cand_mod, ms_path, out_dir, host_desc=host)


if __name__ == '__main__':
    main()
