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
# DIF filter — DIF codelets are not executor-dispatchable per stage
#
# DIF and DIT compute different output buffers given the same input
# and twiddles, so they are not interchangeable when the planner picks
# a codelet for a stage. The DIT-structured forward executor cannot
# substitute a DIF variant; if wisdom ranks a DIF variant as the winner
# at some cell, the planner can't reach it and runs DIT-baseline
# instead — but the cross-protocol comparison was made against the
# unreachable DIF ns, leading to wrong protocol selection.
#
# Filter applied wherever wisdom selects a "winner" the planner will
# consume. See dit_dif_design_note.md.
# ═══════════════════════════════════════════════════════════════

def _is_executor_dispatchable(variant: str, candidates_mod) -> bool:
    """Is this codelet structurally callable by the current DIT executor?

    DIF codelets (those routing through a 't1_dif*' dispatcher) compute a
    different transform from DIT (pre- vs post-butterfly twiddle placement)
    and are not interchangeable per-stage. See dit_dif_design_note.md.
    Until a DIF executor path exists, these variants must be excluded from
    planner wisdom — the planner cannot reach them, so wisdom that ranks
    them as winners is misleading.
    """
    try:
        disp = candidates_mod.dispatcher(variant)
    except Exception:
        # Unknown variant (e.g. handicap experiments). Don't filter —
        # callers that care will have already grouped it out.
        return True
    return 'dif' not in disp


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


def winners_cross_protocol(ms: list[dict],
                           candidates_mod=None) -> dict:
    """For each (isa, me, ios, dir), return {protocol -> best_entry}.
    Used to build plan-wisdom decision functions. This groups by PROTOCOL
    (twiddle layout) — a different axis than dispatcher (codelet family).

    The same codelet can belong to different protocols depending on how
    its twiddle buffer is allocated (the log1_tight handicap experiment
    uses 'log1_tight' as a synthetic protocol name to keep it separate
    from real 'flat').

    DIF filter (v1.0): when candidates_mod is provided, DIF variants are
    excluded before the per-protocol pick_winner call. Both the flat
    pool and the log3 pool see the filter applied. This means
    prefer_dif_log3 collapses to always-zero (intentional — Phase 3
    DIF executor work is needed to consume those wins). Without the
    filter the silent bug from `dit_dif_design_note.md` remains: the
    planner makes flat-vs-log3 decisions against DIF baselines it
    cannot reach.

    candidates_mod=None preserves the pre-change behaviour for any
    caller that doesn't have access to a candidates module — callers
    that emit planner-consumed wisdom MUST pass candidates_mod.
    """
    g: dict = defaultdict(list)
    for m in ms:
        if candidates_mod is not None and \
                not _is_executor_dispatchable(m['variant'], candidates_mod):
            continue
        k = (m['isa'], m['protocol'], m['me'], m['ios'], m['dir'])
        g[k].append(m)
    out: dict = defaultdict(dict)
    for (isa, protocol, me, ios, d), entries in g.items():
        w = pick_winner(entries)
        if w is not None:
            out[(isa, me, ios, d)][protocol] = w
    return dict(out)


# ═══════════════════════════════════════════════════════════════
# Within-flat dispatcher selection
#
# The flat protocol contains multiple dispatchers per radix when the
# bench portfolio includes buffered variants:
#   t1_dit       — baseline DIT-flat (and any non-buf log/u2 variants
#                  that share its dispatcher key)
#   t1_buf_dit   — buffered DIT-flat (tile-based output staging)
#
# When flat wins cross-protocol, the planner needs to choose WHICH
# flat dispatcher to call. winners_cross_protocol's per-protocol
# pick_winner collapses this — it returns one "flat winner", losing
# the dispatcher axis. This function preserves it.
#
# Tie-break: 2% threshold favouring the simpler dispatcher (t1_dit
# over t1_buf_dit). Buf has more complex code shape (the tile-loop +
# drain phase), and at cells where the win is in the noise, the
# baseline DIT path is preferred. Genuine buf wins (10-40% on
# strided cells) are far outside any noise threshold.
# ═══════════════════════════════════════════════════════════════

# Cross-dispatcher preference within the flat protocol. Lower index
# wins on near-ties (within 2% of best). Order matches stylistic
# simplicity, not measured speed: t1_dit < t1_buf_dit < others.
_FLAT_DISPATCHER_PREFERENCE = ['t1_dit', 't1_buf_dit']


def _flat_dispatcher_rank(disp: str) -> int:
    try:
        return _FLAT_DISPATCHER_PREFERENCE.index(disp)
    except ValueError:
        return 1000


def winners_within_flat_per_dispatcher(
        ms: list[dict], candidates_mod) -> dict:
    """Within the flat protocol, return the winning DISPATCHER per cell.

    Returns: {(isa, me, ios, dir) -> dispatcher_key}

    Mirrors the log3_family accounting that emit_plan_wisdom does for
    log3, but applied to the flat protocol's dispatcher axis. Only
    considers executor-dispatchable variants (DIF filtered out).

    Tie-break: cross-dispatcher comparison applies a 2% threshold biased
    toward the simpler dispatcher per _FLAT_DISPATCHER_PREFERENCE.
    Within each (isa, me, ios, dir, dispatcher) bucket the per-bucket
    winner is selected by pick_winner (which already applies its own
    within-dispatcher 2% tie threshold favouring simpler variants).
    """
    # Filter to flat protocol AND executor-dispatchable.
    flat = [m for m in ms
            if m.get('protocol') == 'flat'
            and _is_executor_dispatchable(m['variant'], candidates_mod)]

    # Bucket by (isa, me, ios, dir, dispatcher).
    g: dict = defaultdict(list)
    for m in flat:
        try:
            disp = candidates_mod.dispatcher(m['variant'])
        except Exception:
            continue
        k = (m['isa'], m['me'], m['ios'], m['dir'], disp)
        g[k].append(m)

    # Within each bucket, pick the winner.
    bucket_winner: dict = {}
    for k, entries in g.items():
        w = pick_winner(entries)
        if w is not None:
            bucket_winner[k] = w

    # Cross-dispatcher selection per cell with tie threshold.
    by_cell: dict = defaultdict(list)
    for (isa, me, ios, d, disp), w in bucket_winner.items():
        by_cell[(isa, me, ios, d)].append((disp, w['ns']))

    out: dict = {}
    for cell, options in by_cell.items():
        best_ns = min(ns for (_, ns) in options)
        limit = best_ns * 1.02  # 2% tie threshold
        contenders = [(disp, ns) for (disp, ns) in options if ns <= limit]
        # Within-tie ordering: simpler dispatcher first, ns as fallback.
        contenders.sort(key=lambda x: (_flat_dispatcher_rank(x[0]), x[1]))
        out[cell] = contenders[0][0]

    return out


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

    # Collapse consecutive bands with identical variants, and extend each
    # rule's me_max to cover runtime me values between benched points.
    # A rule for benched me=16 (with next benched point at me=32) covers
    # the runtime interval [16, 31]. The last rule extends to +inf.
    # Without this, runtime me=24 would fall through all rules.
    rules: list[dict] = []
    for (me, pw, pd) in bands:
        if rules and rules[-1]['ios_pow2_variant'] == pw and \
                     rules[-1]['ios_padded_variant'] == pd:
            rules[-1]['me_max'] = me  # will be overwritten below with interval upper bound
        else:
            rules.append({'me_min': me, 'me_max': me,
                          'ios_pow2_variant': pw,
                          'ios_padded_variant': pd})
    # Expand each rule's me_max to just-below-next-rule's me_min so the
    # ladder partitions the positive integers (no gaps).
    for i in range(len(rules) - 1):
        rules[i]['me_max'] = rules[i + 1]['me_min'] - 1
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

def emit_plan_wisdom(R: int, cross: dict,
                     flat_disp_winners: dict | None,
                     out_path: Path, host_desc: str,
                     has_buf_dispatcher: bool = False):
    """Produce static inline bool functions the planner queries to decide
    which protocol to use for a (me, ios) stage. Only AVX2 data is used
    for plan wisdom (the planner's decisions are protocol-level, not
    ISA-level; if an AVX-512 host's split points differ they can
    regenerate wisdom for that host).

    Emits five queries:
      prefer_dit_log3  — true where ct_t1_dit_log3_* won as the log3
                         representative AND log3 was cross-protocol winner.
                         Safe for today's executor (forward path is DIT-
                         structured, calls t1_fwd which points to DIT-log3
                         codelet after Phase 2 integration).
      prefer_dif_log3  — collapses to always-zero in v1.0 because the
                         DIF filter (see _is_executor_dispatchable) excludes
                         DIF variants from cross-protocol winner selection.
                         Kept in the API for v1.1 when a DIF-forward
                         executor path lands.
      prefer_log3      — union of the above (== prefer_dit_log3 in v1.0).
                         Kept for backward compatibility with Phase 2.
      prefer_t1s       — true where t1s wins cross-protocol.
      prefer_buf       — true where flat wins cross-protocol AND the
                         t1_buf_dit dispatcher beats t1_dit baseline within
                         flat (per flat_disp_winners). For radixes with no
                         t1_buf_dit dispatcher (has_buf_dispatcher=False),
                         emits a no-op returning 0 — same pattern as t1s
                         for radixes without a t1s variant.

    flat_disp_winners: output of winners_within_flat_per_dispatcher.
                      Required when has_buf_dispatcher is True; ignored
                      otherwise. None is allowed for the no-buf case.
    """

    # Build per-(me, ios, dir) tables. Each entry captures:
    #   winning_protocol:   which protocol won the cross-protocol comparison
    #   winning_ns:         that protocol's best ns at this point
    #   log3_family:        when log3 is the winning protocol, which DIT/DIF
    #                       family the log3 winner came from ('dit' | 'dif')
    decisions_fwd: list[tuple[int, int, str, float, str]] = []
    decisions_bwd: list[tuple[int, int, str, float, str]] = []
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

        # When log3 wins, figure out whether the winning log3 variant was
        # a DIT flavor or a DIF flavor. The variant name carries this:
        #   ct_t1_dit_log3[_u2*|_isub2|_log_half]   -> 'dit'
        #   ct_t1_dif_log3                          -> 'dif'
        # If anything else shows up, tag as 'dit' conservatively (it means
        # the protocol was log3 but the variant didn't match our naming
        # — shouldn't happen in practice, but conservative default means
        # Phase 2 activates DIT-log3 there and the planner still wins).
        log3_family = ''
        if winning_proto == 'log3':
            winning_variant = candidates_for_plan['log3']['variant']
            if 'dif_log3' in winning_variant:
                log3_family = 'dif'
            else:
                log3_family = 'dit'

        tgt = decisions_fwd if d == 'fwd' else decisions_bwd
        tgt.append((me, ios, winning_proto, winning_ns, log3_family))

    lines = [
        f'/* vfft_r{R}_plan_wisdom.h',
        f' *',
        f' * Auto-generated plan-protocol selection for R={R}.',
        f' * Bench host: {host_desc}',
        f' *',
        f' * Each function returns 1 if the planner should use the named protocol',
        f' * for a stage with the given (me, ios), 0 otherwise.',
        f' *',
        f' * Predicates:',
        f' *   radix{R}_prefer_dit_log3 — DIT-log3 codelet won as log3 representative',
        f' *                              AND log3 was cross-protocol winner.',
        f' *   radix{R}_prefer_dif_log3 — collapsed to always-0 in v1.0 (DIF filter).',
        f' *   radix{R}_prefer_log3     — union of the above (= dit_log3 in v1.0).',
        f' *   radix{R}_prefer_t1s      — t1s protocol won.',
        f' *   radix{R}_prefer_buf      — flat won cross-protocol AND t1_buf_dit',
        f' *                              dispatcher won within flat (vs t1_dit).',
        f' *                              Always-0 for radixes without buf variants.',
        f' *',
        f' * DIF filter (v1.0): DIF variants are excluded from winner selection',
        f' * because the DIT-structured forward executor cannot substitute them',
        f' * per stage. See dit_dif_design_note.md.',
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
    lines.append(f' *   me    ios   winning_protocol   log3_family   winning_ns')
    for (me, ios, p, ns, fam) in sorted(decisions_fwd)[:30]:  # cap length
        fam_s = fam if fam else '-'
        lines.append(f' *   {me:5d} {ios:5d}   {p:12s}   {fam_s:11s}   {ns:8.1f}')
    lines.append(f' */')
    lines.append(f'')

    # -- Helper: given a list of (me, ios) wins, emit a predicate body --
    # Returns the list of lines forming the function body (excluding the
    # signature and closing brace).
    #
    # Two-tier emit:
    #   1. me-majority rule:    `if (me >= LO && me <= HI) return 1;`
    #      Used where the predicate fires at the majority of ios values
    #      for a given me. Dense wins (e.g. t1s at small me on every
    #      stride) collapse into a clean me-range rule.
    #   2. cell-list rule:      `if ((me == 1024 && ios == 8192) || ...) return 1;`
    #      Used for sparse wins at specific (me, ios) pairs that don't
    #      belong to a majority me. Required for buf and similar predicates
    #      that fire only at high-stride cells. Without this, sparse wins
    #      get silently dropped — that bug emitted prefer_buf == 0 even
    #      when buf had real wins on R=64 at (1024, 8192) and (2048, 16384).
    def _emit_predicate_body(wins: list[tuple[int, int]],
                             total_fwd: list[tuple[int, int, str, float, str]],
                             wanted_label: str) -> list[str]:
        body: list[str] = []
        if not wins:
            body.append(f'    (void)me; (void)ios;')
            body.append(f'    /* Bench showed {wanted_label} never wins on this host. */')
            body.append(f'    return 0;')
            return body

        wins_by_me: dict[int, int] = defaultdict(int)
        total_by_me: dict[int, int] = defaultdict(int)
        for (me, _ios, *_rest) in total_fwd:
            total_by_me[me] += 1
        for (me, _ios) in wins:
            wins_by_me[me] += 1

        # Tier 1: me values where the predicate is majority winner across ios.
        rule_mes = sorted(me for me in wins_by_me
                          if wins_by_me[me] >= total_by_me[me] / 2)
        me_rule_set = set(rule_mes)

        # Tier 2: cells outside the majority-me set — sparse wins at
        # specific (me, ios) pairs.
        sparse_cells = sorted([(me, ios) for (me, ios) in wins
                               if me not in me_rule_set])

        # Decide whether ios is referenced by any rule.
        rules_use_ios = bool(sparse_cells)
        if not rules_use_ios:
            body.append(f'    (void)ios;  /* rules are me-only */')

        if sparse_cells:
            # Document the cells so a reader can audit them quickly.
            cell_strs = [f'({me},{ios})' for (me, ios) in sparse_cells[:8]]
            tail = '' if len(sparse_cells) <= 8 else f', ... +{len(sparse_cells) - 8} more'
            body.append(f'    /* Sparse {wanted_label} wins at {len(sparse_cells)} '
                        f'specific (me, ios) cells: {", ".join(cell_strs)}{tail} */')
            # Emit the OR-chain. C compilers handle long ||-chains fine.
            cell_conds = ' || '.join(
                f'(me == {me} && ios == {ios})'
                for (me, ios) in sparse_cells)
            body.append(f'    if ({cell_conds}) return 1;')

        if rule_mes:
            lo, hi = min(rule_mes), max(rule_mes)
            body.append(f'    /* Bench wins at me ∈ {{{", ".join(str(m) for m in rule_mes)}}} */')
            if rule_mes == list(range(lo, hi + 1)) or len(rule_mes) <= 3:
                conds = ' || '.join(f'me == {m}' for m in rule_mes)
                body.append(f'    if ({conds}) return 1;')
            else:
                body.append(f'    if (me >= {lo} && me <= {hi}) return 1;')

        body.append(f'    return 0;')
        return body

    # --- prefer_dit_log3: log3 won AND winning variant was DIT flavor ---
    dit_log3_wins = [(me, ios) for (me, ios, p, _, fam) in decisions_fwd
                     if p == 'log3' and fam == 'dit']
    lines += [
        f'/* Should the planner use DIT-log3 protocol at (me, ios)? */',
        f'/* Safe for today\'s executor: activates DIT-log3 codelet on the forward path. */',
        f'static inline int radix{R}_prefer_dit_log3(size_t me, size_t ios) {{',
    ]
    lines += _emit_predicate_body(dit_log3_wins, decisions_fwd, 'DIT-log3')
    lines += [f'}}', f'']

    # --- prefer_dif_log3: log3 won AND winning variant was DIF flavor ---
    dif_log3_wins = [(me, ios) for (me, ios, p, _, fam) in decisions_fwd
                     if p == 'log3' and fam == 'dif']
    lines += [
        f'/* Should the planner use DIF-log3 protocol at (me, ios)? */',
        f'/* NOT yet consumable by the default executor (forward path is DIT).',
        f' * Requires a DIF-forward executor path to activate. Informational. */',
        f'static inline int radix{R}_prefer_dif_log3(size_t me, size_t ios) {{',
    ]
    lines += _emit_predicate_body(dif_log3_wins, decisions_fwd, 'DIF-log3')
    lines += [f'}}', f'']

    # --- prefer_log3: union of the two (backward-compatible) ---
    lines += [
        f'/* Should the planner use any log3 protocol at (me, ios)?',
        f' * Backward-compatible union: returns 1 if DIT-log3 or DIF-log3 wins. */',
        f'static inline int radix{R}_prefer_log3(size_t me, size_t ios) {{',
        f'    return radix{R}_prefer_dit_log3(me, ios)',
        f'        || radix{R}_prefer_dif_log3(me, ios);',
        f'}}',
        f'',
    ]

    # --- prefer_t1s (unchanged) ---
    t1s_wins = [(me, ios) for (me, ios, p, _, _) in decisions_fwd if p == 't1s']
    lines += [
        f'/* Should the planner use t1s protocol at (me, ios)? */',
        f'static inline int radix{R}_prefer_t1s(size_t me, size_t ios) {{',
    ]
    lines += _emit_predicate_body(t1s_wins, decisions_fwd, 't1s')
    lines += [f'}}', f'']

    # --- prefer_buf: t1_buf_dit dispatcher won within flat AND flat won cross ---
    # For radixes without a t1_buf_dit dispatcher, emit a no-op stub.
    lines += [
        f'/* Should the planner use t1_buf_dit (buffered flat) at (me, ios)?',
        f' * True when the flat protocol wins cross-protocol AND the t1_buf_dit',
        f' * dispatcher beats t1_dit baseline within the flat protocol (per the',
        f' * within-flat per-dispatcher comparison with 2%% tie threshold). */',
        f'static inline int radix{R}_prefer_buf(size_t me, size_t ios) {{',
    ]
    if has_buf_dispatcher and flat_disp_winners is not None:
        # Cells where t1_buf_dit won within flat AND flat won cross-protocol.
        flat_wins_set = {(me, ios) for (me, ios, p, _, _) in decisions_fwd
                         if p == 'flat'}
        buf_wins = []
        for (isa, me, ios, d), winner_disp in flat_disp_winners.items():
            if isa != 'avx2' or d != 'fwd':
                continue
            if winner_disp != 't1_buf_dit':
                continue
            if (me, ios) not in flat_wins_set:
                continue
            buf_wins.append((me, ios))
        lines += _emit_predicate_body(buf_wins, decisions_fwd, 'buf')
    else:
        lines += [
            f'    (void)me; (void)ios;',
            f'    /* No t1_buf_dit dispatcher in this radix\'s portfolio. */',
            f'    return 0;',
        ]
    lines += [f'}}', f'']

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

    # cross-protocol winners (DIF filtered for planner consumption)
    cross = winners_cross_protocol(ms, candidates_mod)

    # Within-flat dispatcher winners (drives prefer_buf). DIF filtered.
    flat_disp_winners = winners_within_flat_per_dispatcher(ms, candidates_mod)

    # Does this radix have a t1_buf_dit dispatcher in its candidate matrix?
    # If not, prefer_buf emits as a no-op returning 0.
    has_buf_dispatcher = any(
        candidates_mod.dispatcher(c.variant) == 't1_buf_dit'
        for c in candidates_mod.enumerate_all())

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

    # Plan wisdom (cross-protocol + within-flat-per-dispatcher).
    path = out_root / f'vfft_r{R}_plan_wisdom.h'
    emit_plan_wisdom(R, cross, flat_disp_winners, path, host_desc,
                     has_buf_dispatcher=has_buf_dispatcher)
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
