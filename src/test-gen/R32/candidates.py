"""
candidates.py — candidate matrix for R=16 codelet bench.

Single source of truth for the three downstream scripts:
  - bench_codelets.py   iterates candidates, generates .h files, runs bench
  - select_codelets.py  iterates winners, writes selection.json
  - emit_selector.py    iterates kept candidates, writes codelet_select_r16.h

Every candidate is a (family, knob-config) tuple describing ONE specific
generated codelet. The bench measures all of them across (ios, me) points
and the selector picks winners per region.

Scope: R=16, ct_t1_dit signature (mid-plan in-place DIT). Extend this file
to add more radixes or codelet signatures.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Candidate:
    """One generator invocation producing one codelet."""
    family: str                    # e.g. 'ct_t1_dit', 'ct_t1_buf_dit'
    isa: str                       # 'avx2' or 'avx512'
    knobs: tuple = ()              # frozen list of (flag, value) pairs

    def cli_args(self) -> list:
        """Return the generator CLI args to produce this codelet."""
        args = ['--isa', self.isa, '--variant', self.family]
        for flag, value in self.knobs:
            if isinstance(value, bool):
                if value:
                    args.append(flag)
            else:
                args.extend([flag, str(value)])
        return args

    def id(self) -> str:
        """Stable identifier — used as dispatch index in the harness."""
        knob_str = '_'.join(f"{k.lstrip('-').replace('-', '_')}{v}"
                            for k, v in self.knobs)
        if knob_str:
            return f"{self.family}__{self.isa}__{knob_str}"
        return f"{self.family}__{self.isa}"

    def header_name(self) -> str:
        """Output .h filename for this candidate's generated code."""
        return f"{self.id()}.h"

    def function_names(self) -> dict:
        """
        Expected function symbol names emitted by the generator.
        Returns {'fwd': '...', 'bwd': '...'}.

        This is coupled to the generator's naming convention. If the
        generator changes how it names functions, update here.
        """
        f = self.family
        isa = self.isa

        # Base symbol derived from family name
        if f == 'ct_t1_dit':
            base = 'radix16_t1_dit'
        elif f == 'ct_t1_dit_log3':
            base = 'radix16_t1_dit_log3'
        elif f == 'ct_t1s_dit':
            base = 'radix16_t1s_dit'
        elif f == 'ct_t1_buf_dit':
            tile = next(v for k, v in self.knobs if k == '--tile')
            drain = next(v for k, v in self.knobs if k == '--drain')
            base = f'radix16_t1_buf_dit_tile{tile}_{drain}'
        else:
            raise ValueError(f"unknown family {f}")

        return {
            'fwd': f'{base}_fwd_{isa}',
            'bwd': f'{base}_bwd_{isa}',
        }


# ─────────────────────────────────────────────────────────────────────
# KNOB ENUMERATION
# ─────────────────────────────────────────────────────────────────────

BUF_TILES = [16, 32, 64, 128]
BUF_DRAINS = ['temporal', 'stream']


def _enumerate_buf_candidates(isa: str) -> list:
    """All (tile × drain) combos of ct_t1_buf_dit for one ISA."""
    return [
        Candidate(
            family='ct_t1_buf_dit',
            isa=isa,
            knobs=(('--tile', t), ('--drain', d)),
        )
        for t in BUF_TILES
        for d in BUF_DRAINS
    ]


def _enumerate_simple_candidates(isa: str) -> list:
    """Families without extra knobs: one candidate each."""
    return [
        Candidate(family='ct_t1_dit', isa=isa),
        Candidate(family='ct_t1_dit_log3', isa=isa),
        Candidate(family='ct_t1s_dit', isa=isa),
    ]


def enumerate_all() -> list:
    """All candidates across ISAs we might want to bench."""
    out = []
    for isa in ('avx2', 'avx512'):
        out.extend(_enumerate_simple_candidates(isa))
        out.extend(_enumerate_buf_candidates(isa))
    return out


# ─────────────────────────────────────────────────────────────────────
# SWEEP SHAPE — the (ios, me) grid the bench measures at
# ─────────────────────────────────────────────────────────────────────

# me values span a range covering small (L1), medium (L2), and large (L3).
ME_VALUES = [64, 128, 256, 512, 1024, 2048]

# For each me, we measure at several ios values:
#   ios == me         (the "default" case — what plans usually call with)
#   ios == me + 8     (one cacheline of padding — tests stride aliasing)
#   ios == me + 64    (larger padding — confirms alignment effects)
def ios_variants_for(me: int) -> list:
    return [me, me + 8, me + 64]


def sweep_points() -> list:
    """Full list of (ios, me) points to measure."""
    return [(ios, me) for me in ME_VALUES for ios in ios_variants_for(me)]


if __name__ == '__main__':
    # Sanity summary when run directly
    cands = enumerate_all()
    pts = sweep_points()
    print(f"R=16 candidate matrix: {len(cands)} candidates")
    by_isa = {}
    for c in cands:
        by_isa.setdefault(c.isa, []).append(c)
    for isa, cs in by_isa.items():
        print(f"  {isa}: {len(cs)}")
    print(f"Sweep points: {len(pts)} (ios, me) combinations")
    print(f"Total measurements: {len(cands)} × {len(pts)} × 2 (fwd/bwd) = "
          f"{len(cands) * len(pts) * 2}")
    print()
    print("Sample candidates:")
    for c in cands[:5]:
        print(f"  {c.id()}")
        print(f"    CLI:  {' '.join(c.cli_args())}")
        print(f"    fwd:  {c.function_names()['fwd']}")
