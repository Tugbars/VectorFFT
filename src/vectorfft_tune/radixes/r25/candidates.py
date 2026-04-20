"""
radixes/r25/candidates.py — R=25 composite candidate enumeration.

R=25 is the first composite radix in the tuning framework. Internally it
uses a fused 5×5 Cooley-Tukey decomposition:
  - Pass 1: 5 parallel radix-5 sub-FFTs
  - Internal W25 twiddle application (9 unique exponents, all requiring
    cmul, hoisted as compile-time constants)
  - Pass 2: 5 radix-5 column combines
  - 25-slot stack spill buffer holds the intermediate grid between passes

From the planner's perspective R=25 is a single radix. The internal
5×5 structure is opaque.

Tuning scope:
  - Twiddle strategies: flat / t1s / log3 (3 variants)
  - Unroll factor: U=1 only (spill buffer is already large; U=2 doubles it)
  - Skip: DIF (dominated), oop_dit, ct_n1 family

Note on t1s at R=25: external twiddle count is (R-1) = 24 scalars, so
t1s can only partially hoist (5/24 on AVX2, 12/24 on AVX-512). This is
the same partial-hoist regime as R=32/R=64 on power-of-two radixes.
Partial t1s typically wins at small me (where broadcast amortization
still helps) and loses at large me.

Sweep grid (working-set analysis, Raptor Lake 48KB L1):
  me ∈ {8, 16, 32, 64, 128, 256}
    me=8..32:  rio 3.2-12.8 KB, L1-resident
    me=64:     rio 25.6 KB, L1-tight
    me=128:    rio 51.2 KB, exceeds L1, L2 hits
    me=256:    rio 102 KB, L2-resident (VTune characterized this point)
  All divisible by both AVX2 k_step=4 and AVX-512 k_step=8.
  ios ∈ {me, me+8, 25*me, 32*me}
    pow2 stride / padded / natural R=25 next-stage stride / power-of-two
    large stride.
"""
from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
import gen_radix25 as _gen  # noqa: E402


RADIX = 25
GEN_SCRIPT = str(_HERE / 'gen_radix25.py')


@dataclass(frozen=True)
class Candidate:
    variant: str
    isa: str
    id: str
    protocol_override: str | None = None
    requires_avx512: bool = False


_ME_GRID = [8, 16, 32, 64, 128, 256]

def _ios_for_me(me: int) -> list[int]:
    return [me, me + 8, 25 * me, 32 * me]


def _min_me_for_variant(variant_id: str) -> int:
    """Buf variants require me >= tile so that at least one full tile fits.
    Base variants have no me floor."""
    if 'tile32' in variant_id:
        return 32
    if 'tile64' in variant_id:
        return 64
    return 0


def sweep_grid(variant_id: str) -> list[tuple[int, int]]:
    floor = _min_me_for_variant(variant_id)
    return [(me, ios) for me in _ME_GRID if me >= floor for ios in _ios_for_me(me)]


_ISAS = ['avx2', 'avx512']


def enumerate_all() -> list[Candidate]:
    out: list[Candidate] = []
    for variant in _gen.VARIANTS:
        for isa in _ISAS:
            out.append(Candidate(
                variant=variant,
                isa=isa,
                id=f'{variant}__{isa}',
                requires_avx512=(isa == 'avx512'),
            ))
    return out


def function_name(variant_id: str, isa: str, direction: str) -> str:
    return _gen.function_name(variant_id, isa, direction)


def protocol(variant_id: str) -> str:
    return _gen.protocol(variant_id)


def dispatcher(variant_id: str) -> str:
    return _gen.dispatcher(variant_id)
