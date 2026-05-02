"""
radixes/r7/candidates.py — R=7 candidate enumeration.

Tuning scope:
  - Twiddle strategies: flat / t1s / log3  (3 variants)
  - Unroll factor: U=1 only
    AVX2 butterfly already uses an explicit 6-slot stack spill buffer
    within the inner m-loop; U=2 would double the spill traffic.
    AVX-512 no-spill path uses 26/32 ZMMs peak; U=2 would need ~52 ZMMs.
  - Skipped: DIF (strictly dominated at small R), oop_dit, ct_n1 family

  Note on t1s at AVX2: the hoisted twiddle broadcasts (6 complex = 12 YMMs)
  compete with the butterfly's own register working set. The existing
  spill plan is already tight at 16 YMMs — t1s may spill worse than flat.
  We include t1s at AVX2 anyway to get empirical confirmation; if it loses
  at every point the tuner will route around it.

Sweep grid:
  me ∈ {56, 112, 224, 448, 896, 1792}
    All divisible by both AVX2 k_step=4 and AVX-512 k_step=8.
    56 is the smallest useful AVX-512 sweep point (56 = 7·8).
    Pure 7^k me values {7, 49, 343, 2401} are excluded: the codelet's
    tight `for (m; m<me; m+=k_step)` loop has no tail handling, so
    non-multiples of k_step would either overrun or under-process.
  ios ∈ {me, me+8, 7*me, 8*me}
    pow2 stride / padded stride / natural R=7 next-stage stride /
    power-of-two large stride (mirrors the convention used at R=3..R=64).
"""
from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
import gen_radix7 as _gen  # noqa: E402


RADIX = 7
GEN_SCRIPT = str(_HERE / 'gen_radix7.py')


@dataclass(frozen=True)
class Candidate:
    variant: str
    isa: str
    id: str
    protocol_override: str | None = None
    requires_avx512: bool = False


_ME_GRID = [56, 112, 224, 448, 896, 1792]

def _ios_for_me(me: int) -> list[int]:
    return [me, me + 8, 7 * me, 8 * me]


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
