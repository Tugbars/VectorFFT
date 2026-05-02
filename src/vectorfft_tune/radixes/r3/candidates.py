"""
radixes/r3/candidates.py — R=3 candidate enumeration.

Tuning scope (finalized, see project discussion):
  - Twiddle strategies: flat / t1s / log3
  - Unroll factor: U=1 only on AVX2 (16 YMM budget too tight for U=2)
                   U=1, U=2, U=3 on AVX-512 (32 ZMM accommodates all)
  - Skipped: DIF, oop_dit, ct_n1 family, tile/buf, prefetch
    (DIF mathematically dominated at R=3; n1 family is called at
     planner-specific stages, not dispatcher-selected.)

Dispatcher layout:
  t1_dit       ← ct_t1_dit_u1 (AVX2); ct_t1_dit_u{1,2,3} (AVX-512)
  t1s_dit      ← ct_t1s_dit_u1 (AVX2); ct_t1s_dit_u{1,2,3} (AVX-512)
  t1_dit_log3  ← ct_t1_dit_log3_u1 (AVX2); ct_t1_dit_log3_u{1,2,3} (AVX-512)

Sweep grid:
  me ∈ {24, 48, 96, 192, 384, 768, 1536, 3072}
    (multiples of 24 so that U=3 can always be exercised without falling
     entirely through to the U=1 tail; these are also representative me
     values R=3 sees inside mixed-radix plans for N = 2^a · 3 · 5^b · ...)
  ios ∈ {me, me+8, 3*me, 8*me}
"""
from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
import gen_radix3 as _gen  # noqa: E402


RADIX = 3
GEN_SCRIPT = str(_HERE / 'gen_radix3.py')


@dataclass(frozen=True)
class Candidate:
    variant: str
    isa: str
    id: str
    protocol_override: str | None = None
    requires_avx512: bool = False


_ME_GRID = [24, 48, 96, 192, 384, 768, 1536, 3072]

def _ios_for_me(me: int) -> list[int]:
    return [me, me + 8, 3 * me, 8 * me]


def sweep_grid(variant_id: str) -> list[tuple[int, int]]:
    # No radix-specific skips: R=3 has no t1s-only me restriction
    # (the t1s variant hoists only 2 scalars at R=3, works at any me)
    return [(me, ios) for me in _ME_GRID for ios in _ios_for_me(me)]


# AVX2 only takes U=1; AVX-512 takes U=1/U=2/U=3
_AVX2_VARIANTS = ['ct_t1_dit_u1', 'ct_t1s_dit_u1', 'ct_t1_dit_log3_u1']
_AVX512_VARIANTS = [
    'ct_t1_dit_u1', 'ct_t1_dit_u2', 'ct_t1_dit_u3',
    'ct_t1s_dit_u1', 'ct_t1s_dit_u2', 'ct_t1s_dit_u3',
    'ct_t1_dit_log3_u1', 'ct_t1_dit_log3_u2', 'ct_t1_dit_log3_u3',
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
