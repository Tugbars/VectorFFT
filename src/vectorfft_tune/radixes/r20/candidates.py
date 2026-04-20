"""
radixes/r20/candidates.py — R=20 composite candidate enumeration.

R=20 is the second composite radix (4×5 = R=4 × R=5 Cooley-Tukey):
  - Pass 1: 5 parallel radix-4 sub-FFTs (zero constants, 8 add/sub each,
    peak 8 YMM — very light)
  - Internal W20 twiddle application (8 unique exponents, compiled
    constants, hoisted as broadcast registers)
  - Pass 2: 4 parallel radix-5 column combines (4 constants, peak 16 YMM)
  - 20-slot stack spill buffer holds the intermediate grid between passes

Comparison to R=25 (5×5):
  - R=25 internal twiddles: 9 unique exponents
  - R=20 internal twiddles: 8 unique exponents (one fewer)
  - R=25 external twiddles per iteration: 24 cmuls for flat
  - R=20 external twiddles per iteration: 19 cmuls for flat
  - R=20 Pass 1 is cheaper (DFT-4 has no multiplies) while Pass 2 is
    the same DFT-5 used at R=5. Net arithmetic: significantly lower
    per-iteration cost than R=25.

Tuning scope:
  - Twiddle strategies: flat / t1s / log3 (3 variants)
  - Unroll factor: U=1 only (ct_ family doesn't emit U=2 at R=20)
  - Skip: DIF, oop_dit, ct_n1 family (planner-selected)

Note on t1s at R=20: external twiddle count (R-1) = 19, hoist capacity
is 5 on AVX2 / 12 on AVX-512. Same partial-hoist regime as R=25.

Sweep grid:
  me ∈ {8, 16, 32, 64, 128, 256}
    me=8..32:  rio 2.6-10.2 KB, L1-resident
    me=64:     rio 20.5 KB, comfortably L1
    me=128:    rio 41 KB, fits L1 on RL (48 KB) but tight
    me=256:    rio 82 KB, L2-resident (matches VTune characterization)
  All divisible by both AVX2 k_step=4 and AVX-512 k_step=8.
  ios ∈ {me, me+8, 20*me, 32*me}
    pow2 stride / padded / natural R=20 next-stage stride / large pow2.
"""
from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
import gen_radix20 as _gen  # noqa: E402


RADIX = 20
GEN_SCRIPT = str(_HERE / 'gen_radix20.py')


@dataclass(frozen=True)
class Candidate:
    variant: str
    isa: str
    id: str
    protocol_override: str | None = None
    requires_avx512: bool = False


_ME_GRID = [8, 16, 32, 64, 128, 256]

def _ios_for_me(me: int) -> list[int]:
    return [me, me + 8, 20 * me, 32 * me]


def sweep_grid(variant_id: str) -> list[tuple[int, int]]:
    return [(me, ios) for me in _ME_GRID for ios in _ios_for_me(me)]


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
