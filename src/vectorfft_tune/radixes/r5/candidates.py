"""
radixes/r5/candidates.py — R=5 candidate enumeration.

Tuning scope (same pattern as R=7):
  - Twiddle strategies: flat / t1s / log3  (3 variants)
  - Unroll factor: U=1 only
    AVX2 butterfly is zero-spill at 16 YMMs; U=2 would need ~32 YMMs
    and spill heavily. AVX-512 has headroom but the butterfly is small
    enough that U=2 amortization is likely below measurement noise
    and not worth doubling the code size.
  - Skipped: DIF (strictly dominated at small R), oop_dit, ct_n1 family

Sweep grid:
  me ∈ {40, 80, 160, 320, 640, 1280}
    Multiples of lcm(5, 8) = 40 — all divisible by both AVX2 k_step=4
    and AVX-512 k_step=8. 40 is the smallest useful AVX-512 sweep point
    (40 = 5·8). Pure 5^k me values {5, 25, 125, 625, 3125} are excluded:
    the codelet's tight `for (m; m<me; m+=k_step)` loop has no tail
    handling, so non-multiples of k_step would either overrun or
    under-process.
  ios ∈ {me, me+8, 5*me, 8*me}
    pow2 stride / padded / natural R=5 next-stage stride / power-of-two
    large stride (mirrors the R=7 convention with 5 replacing 7).
"""
from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
import gen_radix5 as _gen  # noqa: E402


RADIX = 5
GEN_SCRIPT = str(_HERE / 'gen_radix5.py')


@dataclass(frozen=True)
class Candidate:
    variant: str
    isa: str
    id: str
    protocol_override: str | None = None
    requires_avx512: bool = False


_ME_GRID = [40, 80, 160, 320, 640, 1280]

def _ios_for_me(me: int) -> list[int]:
    return [me, me + 8, 5 * me, 8 * me]


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
