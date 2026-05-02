"""
radixes/r6/candidates.py — R=6 candidate enumeration.

Tuning scope:
  - Twiddle strategies: flat / t1s / log3
  - Unroll factor: U=1 only on AVX2 (16 YMM budget saturated by
                   2× DFT-3 butterfly + 2 W6 twiddle pairs + sign_flip
                   already ~15/16 YMM at U=1)
  - AVX-512 takes U=1 and U=2 (pipelined). Register math: always-live
    ~7 ZMM, U=1 peak ~17 ZMM, U=2 peak ~25 ZMM (fits 32).
  - U=3 conservatively omitted: would push to ~33 ZMM, outside the
    comfortable 3-6 spill absorption window on Intel P-cores.
  - Skipped: DIF, oop_dit, ct_n1 family, tile/buf, prefetch
    (n1 family is planner-specific; DIF mathematically dominated for
     small composite radixes; buf doesn't help at R=6 since output
     footprint is small.)

Dispatcher layout:
  t1_dit       ← ct_t1_dit      (AVX2 U=1; AVX-512 U=1, U=2)
  t1s_dit      ← ct_t1s_dit     (AVX2 U=1; AVX-512 U=1, U=2)
  t1_dit_log3  ← ct_t1_dit_log3 (AVX2 U=1; AVX-512 U=1, U=2)

Sweep grid:
  me ∈ {8, 16, 32, 64, 128, 256}
    Compatible with VL=4 (AVX2) and VL=8 (AVX-512). U=2 skips me=8
    because the main loop requires me >= 2*VL = 16 before the U=2 body
    is taken (U=1 tail handles the remainder, but if me=8 the main loop
    runs zero iterations).
  ios ∈ {me, me+8, 10·me, 32·me}
    Matches R=10/R=12 pattern. Covers contiguous, small-offset (stride),
    large-offset (mid-plan), and extreme-offset (DTLB pressure) regimes.
"""
from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
import gen_radix6 as _gen  # noqa: E402


RADIX = 6
GEN_SCRIPT = str(_HERE / 'gen_radix6.py')


@dataclass(frozen=True)
class Candidate:
    variant: str
    isa: str
    id: str
    protocol_override: str | None = None
    requires_avx512: bool = False


_ME_GRID = [8, 16, 32, 64, 128, 256]


def _ios_for_me(me: int) -> list[int]:
    return [me, me + 8, 10 * me, 32 * me]


def sweep_grid(variant_id: str) -> list[tuple[int, int]]:
    """Grid for a given candidate. U=2 variants skip me=8 (main loop
    requires me >= 2*VL = 16 on AVX-512)."""
    pts = [(me, ios) for me in _ME_GRID for ios in _ios_for_me(me)]
    if variant_id.endswith('_u2'):
        pts = [(me, ios) for (me, ios) in pts if me >= 16]
    return pts


# AVX2 takes U=1 only
_AVX2_VARIANTS = ['ct_t1_dit', 'ct_t1s_dit', 'ct_t1_dit_log3']
# AVX-512 takes U=1 base triple + U=2 for all three protocols
_AVX512_VARIANTS = [
    'ct_t1_dit', 'ct_t1s_dit', 'ct_t1_dit_log3',
    'ct_t1_dit_u2', 'ct_t1s_dit_u2', 'ct_t1_dit_log3_u2',
]


def enumerate_all() -> list[Candidate]:
    out: list[Candidate] = []
    for v in _AVX2_VARIANTS:
        out.append(Candidate(variant=v, isa='avx2', id=f'{v}__avx2',
                             requires_avx512=False))
    for v in _AVX512_VARIANTS:
        out.append(Candidate(variant=v, isa='avx512', id=f'{v}__avx512',
                             requires_avx512=True))
    return out


def function_name(variant_id: str, isa: str, direction: str) -> str:
    return _gen.function_name(variant_id, isa, direction)


def protocol(variant_id: str) -> str:
    return _gen.protocol(variant_id)


def dispatcher(variant_id: str) -> str:
    return _gen.dispatcher(variant_id)
