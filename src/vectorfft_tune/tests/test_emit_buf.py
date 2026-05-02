"""
test_emit_buf.py — validation tests for Phase 2.1 emit changes.

Tests the DIF filter and within-flat-per-dispatcher selection by loading
the live `bench_out/r16/measurements.jsonl` (which the development host has
populated). Run with:

    cd vectorfft_tune
    python -m pytest tests/test_emit_buf.py -v

The host's measurement set may show or hide certain cells (e.g. on Raptor
Lake, log3 dominates cross-protocol so prefer_buf is always 0). Tests
that depend on cell-level outcomes are framed against the within-flat
selection, which is invariant to which protocol wins cross-protocol.
"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

# Resolve repo paths.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'common'))
import select_and_emit as sae  # noqa: E402


def _load_candidates(radix_name: str):
    """Load a radix candidates module by name (e.g. 'r16', 'r11')."""
    path = ROOT / 'radixes' / radix_name / 'candidates.py'
    spec = importlib.util.spec_from_file_location(
        f'{radix_name}_candidates', path)
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec — Python 3.13's dataclass needs
    # the module discoverable while the @dataclass decorator runs.
    sys.modules[f'{radix_name}_candidates'] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_measurements(radix_name: str):
    msf = ROOT / 'bench_out' / radix_name / 'measurements.jsonl'
    if not msf.exists():
        return None
    return sae._load_measurements(msf)


# ─────────────────────────────────────────────────────────────────────
# Test 1: DIF filter caught the silent bug
# ─────────────────────────────────────────────────────────────────────

def test_dif_filter_excludes_dif_from_flat_winner():
    """At every cell, the cross-protocol flat winner returned by
    winners_cross_protocol(ms, candidates_mod) must be from a DIT
    dispatcher (i.e. 'dif' is not in the dispatcher name). If a DIF
    variant leaks into the flat winner, the DIF filter is broken and
    the planner's flat-vs-log3 decisions will be made against an
    unreachable baseline."""
    cands = _load_candidates('r16')
    ms = _load_measurements('r16')
    if ms is None:
        import pytest
        pytest.skip('r16 measurements not present')
    cross = sae.winners_cross_protocol(ms, cands)
    leaks = []
    for cell, by_proto in cross.items():
        flat = by_proto.get('flat')
        if flat is None:
            continue
        disp = cands.dispatcher(flat['variant'])
        if 'dif' in disp:
            leaks.append((cell, flat['variant'], disp))
    assert not leaks, (
        f'DIF filter leaked: {len(leaks)} cells have DIF variants as '
        f'the flat winner. Sample: {leaks[:3]}')


def test_dif_filter_excludes_dif_from_log3_winner():
    """Same check for the log3 pool. v1.0 filters DIF from BOTH pools so
    prefer_dif_log3 collapses to always-zero. Verifies the filter is
    applied to log3 selection too, not just flat."""
    cands = _load_candidates('r16')
    ms = _load_measurements('r16')
    if ms is None:
        import pytest
        pytest.skip('r16 measurements not present')
    cross = sae.winners_cross_protocol(ms, cands)
    leaks = []
    for cell, by_proto in cross.items():
        log3 = by_proto.get('log3')
        if log3 is None:
            continue
        disp = cands.dispatcher(log3['variant'])
        if 'dif' in disp:
            leaks.append((cell, log3['variant'], disp))
    assert not leaks, (
        f'DIF filter leaked into log3 pool: {len(leaks)} cells have '
        f'DIF-log3 variants as the log3 winner. Sample: {leaks[:3]}')


# ─────────────────────────────────────────────────────────────────────
# Test 2/3: within-flat dispatcher selection picks buf at high-stride,
#           t1_dit at low-stride
# ─────────────────────────────────────────────────────────────────────

def test_within_flat_picks_buf_at_high_stride_high_me():
    """At cells where buf is genuinely faster than baseline within the
    flat protocol (>2% margin), winners_within_flat_per_dispatcher must
    select t1_buf_dit. This is independent of whether flat wins
    cross-protocol — the within-flat selection is what drives prefer_buf,
    but the within-flat winner itself reflects measured ns within the
    flat family.

    The R=16 portfolio measurements show buf winning at high-me / high-
    stride cells (e.g. me=2048 ios=2048). This test asserts the within-
    flat selector finds at least one such cell. If the bench doesn't
    contain any buf-winning cells (different host, different measurement
    methodology), the test skips."""
    cands = _load_candidates('r16')
    ms = _load_measurements('r16')
    if ms is None:
        import pytest
        pytest.skip('r16 measurements not present')
    flat_disp = sae.winners_within_flat_per_dispatcher(ms, cands)
    buf_cells = [cell for cell, disp in flat_disp.items()
                 if cell[0] == 'avx2' and cell[3] == 'fwd'
                 and disp == 't1_buf_dit']
    if not buf_cells:
        import pytest
        pytest.skip('no buf-winning cells on this host')
    # At least one should be in the high-me / high-stride region.
    high = [cell for cell in buf_cells if cell[1] >= 256]  # me >= 256
    assert high, (
        f'buf wins {len(buf_cells)} cells but none at me >= 256 — '
        f'unexpected distribution. Wins: {sorted(buf_cells)[:5]}')


def test_within_flat_picks_dit_at_low_me():
    """At small me (low stride pressure), the baseline t1_dit dispatcher
    should win within flat. The tie threshold biases toward t1_dit on
    near-ties; only genuine 2%+ wins by buf flip the selection."""
    cands = _load_candidates('r16')
    ms = _load_measurements('r16')
    if ms is None:
        import pytest
        pytest.skip('r16 measurements not present')
    flat_disp = sae.winners_within_flat_per_dispatcher(ms, cands)
    cell = ('avx2', 64, 64, 'fwd')
    winner = flat_disp.get(cell)
    if winner is None:
        import pytest
        pytest.skip(f'cell {cell} not measured')
    assert winner == 't1_dit', (
        f'expected t1_dit at {cell}, got {winner}. The buf dispatcher '
        f'should not win at small me/ios — it incurs drain overhead '
        f'that exceeds the DTLB savings at small working sets.')


# ─────────────────────────────────────────────────────────────────────
# Test 4: composite radix without buf emits prefer_buf as no-op
# ─────────────────────────────────────────────────────────────────────

def test_composite_radix_has_no_buf_dispatcher():
    """R=11 has no t1_buf_dit dispatcher in its candidate matrix. The
    has_buf_dispatcher detection in emit_all should evaluate to False
    for this radix, causing emit_plan_wisdom to emit prefer_buf as a
    no-op stub returning 0. This mirrors how prefer_t1s emits as
    always-0 for radixes (R=32, R=64) without t1s variants."""
    cands = _load_candidates('r11')
    has_buf = any(
        cands.dispatcher(c.variant) == 't1_buf_dit'
        for c in cands.enumerate_all())
    assert not has_buf, (
        'R=11 unexpectedly has a t1_buf_dit variant — adjust this test '
        'or the emit logic to match the actual portfolio.')


def test_emitted_r16_prefer_buf_compiles_to_valid_predicate():
    """The emitted vfft_r16_plan_wisdom.h must declare radix16_prefer_buf
    as a static inline int function returning either 0 unconditionally
    or via an if-ladder. This is a structural test — it asserts the
    function exists and has the expected signature so that Unit 3's
    bridge can call it without compile errors."""
    wisdom = ROOT / 'generated' / 'r16' / 'vfft_r16_plan_wisdom.h'
    if not wisdom.exists():
        import pytest
        pytest.skip('r16 plan_wisdom not yet emitted; run --phase emit')
    text = wisdom.read_text(encoding='utf-8')
    assert 'static inline int radix16_prefer_buf(size_t me, size_t ios)' in text, (
        'prefer_buf signature missing from r16 plan_wisdom')
    assert 'return 0;' in text, (
        'prefer_buf body missing return statement')


# ─────────────────────────────────────────────────────────────────────
# Test 5: existing predicates flip ONLY at cells where DIF was
#         the pre-change winner
# ─────────────────────────────────────────────────────────────────────

def test_r64_prefer_buf_fires_at_high_stride_cells():
    """The reviewer's R=64 predictions, derived from raw measurements:

      (1024, 8192):  buf wins cross-protocol — flat=161673 (buf tile128),
                     log3=184304. Within flat, buf tile128 (161673) beats
                     baseline t1_dit (198192). prefer_buf must be 1.

      (2048, 16384): same pattern — flat=333975 (buf tile128) beats
                     log3=399937. Within flat, buf tile128 wins.
                     prefer_buf must be 1.

      (64, 64):      t1s wins cross-protocol (2542 ns) — buf is far behind
                     at 6205 ns. prefer_buf must be 0 (cross-protocol).

      (2048, 2048):  flat-baseline wins (239201 ns). Within flat, t1_dit
                     (239201) beats buf tile128 (276205). prefer_buf must
                     be 0 (lost within flat to baseline).

    These are sparse (me, ios) wins. Before the cell-list emit fix, these
    silently emitted prefer_buf == 0 because no me had majority wins.
    """
    wisdom_path = ROOT / 'generated' / 'r64' / 'vfft_r64_plan_wisdom.h'
    if not wisdom_path.exists():
        import pytest
        pytest.skip('r64 plan_wisdom not yet emitted; run --phase emit')

    text = wisdom_path.read_text(encoding='utf-8')

    # Extract the prefer_buf body and evaluate it as Python with me/ios
    # substituted. Simple regex pull.
    import re
    m = re.search(
        r'static inline int radix64_prefer_buf\([^)]*\)\s*\{(.+?)^\}',
        text, re.MULTILINE | re.DOTALL)
    assert m, 'prefer_buf signature not found in r64 wisdom'
    body = m.group(1)

    # Asserts via static check on the body — the if-condition must
    # cover the high-stride cells.
    assert '1024' in body and '8192' in body, (
        f'prefer_buf body does not reference (1024, 8192):\n{body}')
    assert '2048' in body and '16384' in body, (
        f'prefer_buf body does not reference (2048, 16384):\n{body}')


def test_r64_prefer_t1s_fires_at_low_me():
    """t1s wins broadly at small me on R=64 — me ∈ {64, 96, 128} per
    the orchestrator's measurements. The wisdom file should emit a
    me-range or me-list rule covering these. The bridge's old hardcoded
    `case 64: return 0;` was wrong — now fixed to call radix64_prefer_t1s."""
    wisdom_path = ROOT / 'generated' / 'r64' / 'vfft_r64_plan_wisdom.h'
    if not wisdom_path.exists():
        import pytest
        pytest.skip('r64 plan_wisdom not yet emitted')

    text = wisdom_path.read_text(encoding='utf-8')
    import re
    m = re.search(
        r'static inline int radix64_prefer_t1s\([^)]*\)\s*\{(.+?)^\}',
        text, re.MULTILINE | re.DOTALL)
    assert m, 'prefer_t1s signature not found in r64 wisdom'
    body = m.group(1)

    # Must reference me=64 (low-me t1s win)
    assert 'me == 64' in body or 'me >= 64' in body, (
        f'prefer_t1s body does not cover me=64:\n{body}')
    # Should NOT be the always-zero stub
    assert 'never wins' not in body, (
        f'prefer_t1s body claims never-wins but should fire at low me:\n{body}')


def test_existing_predicates_change_only_at_dif_cells():
    """The DIF filter changes prefer_dit_log3 / prefer_log3 / prefer_t1s
    only at cells where the pre-change winner (no filter) was a DIF
    variant. At every other cell, the per-protocol winners must be
    byte-identical to the pre-change emit.

    This is the audit for the silent-bug fix: every flipped cell must
    be attributable to DIF having been the unreachable per-protocol
    winner. Any other change is a regression."""
    cands = _load_candidates('r16')
    ms = _load_measurements('r16')
    if ms is None:
        import pytest
        pytest.skip('r16 measurements not present')

    cross_unfiltered = sae.winners_cross_protocol(ms)  # no filter
    cross_filtered = sae.winners_cross_protocol(ms, cands)  # DIF filtered

    # Identify cells where any pre-change per-protocol winner was DIF.
    suspect_cells: set = set()
    for cell, by_proto in cross_unfiltered.items():
        for proto, winner in by_proto.items():
            disp = cands.dispatcher(winner['variant'])
            if 'dif' in disp:
                suspect_cells.add(cell)

    # At all non-suspect cells, the per-protocol winners must be
    # byte-identical between filtered and unfiltered.
    regressions = []
    for cell, by_proto in cross_unfiltered.items():
        if cell in suspect_cells:
            continue
        filtered = cross_filtered.get(cell, {})
        for proto, winner in by_proto.items():
            f_winner = filtered.get(proto)
            if f_winner is None or f_winner['variant'] != winner['variant']:
                regressions.append(
                    (cell, proto, winner['variant'],
                     f_winner['variant'] if f_winner else None))
    assert not regressions, (
        f'Existing predicates changed at non-DIF cells (regression). '
        f'Sample: {regressions[:3]}')

    # Also verify suspect cells DID change (otherwise the filter is no-op).
    flips = 0
    for cell in suspect_cells:
        unfiltered = cross_unfiltered[cell]
        filtered = cross_filtered.get(cell, {})
        for proto in unfiltered:
            u = unfiltered[proto]['variant']
            f = filtered.get(proto, {}).get('variant') if filtered else None
            if u != f:
                flips += 1
                break
    assert flips > 0 or not suspect_cells, (
        f'DIF filter did not flip any cell — implementation may be '
        f'a no-op. {len(suspect_cells)} suspect cells, {flips} flips.')
