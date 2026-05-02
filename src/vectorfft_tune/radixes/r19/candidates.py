"""
radixes/r19/candidates.py — R=19 candidate enumeration.

Tuning scope:
  - Twiddle strategies: flat / t1s / log3
  - Unroll factor: U=1 only on BOTH ISAs.

    R=19 is a prime radix with a monolithic DFT-19 butterfly: 19 inputs
    (38 registers for re/im pairs) + 26 genfft constants + sign_flip
    ≈ 64+ registers needed at U=1. Both AVX2 (16 YMM) and AVX-512 (32 ZMM)
    spill extremely heavily. Heaviest prime butterfly in the portfolio.
    The compiler handles via Sethi-Ullman scheduling and store-to-load
    forwarding. U=N unrolling is categorically infeasible — pressure is
    already 32 over the AVX-512 budget at U=1.

  - Skipped: DIF, oop_dit, ct_n1 family (prime-direct; no composite
    factorization).

Dispatcher layout:
  t1_dit       ← ct_t1_dit      (AVX2 U=1; AVX-512 U=1)
  t1s_dit      ← ct_t1s_dit     (AVX2 U=1; AVX-512 U=1)
  t1_dit_log3  ← ct_t1_dit_log3 (AVX2 U=1; AVX-512 U=1)

Sweep grid:
  me ∈ {8, 16, 32, 64, 128, 256}
  ios ∈ {me, me+8, 19·me, 32·me}
    Stride 19·me matches natural R=19 packing; 32·me exercises DTLB.

Notes:
  With 64+ register peak, AVX2 compiler spills ~50 temporaries; AVX-512
  spills ~32. This is well past any "absorbed spill" threshold. Expect
  meaningful performance gaps between protocols and the highest run-to-run
  variance in the prime portfolio. The protocol selection here stresses
  the compiler's scheduler more than any other radix — interesting data
  point for the paper about where hand-scheduled beats compiler-scheduled.
"""
from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
import gen_radix19 as _gen  # noqa: E402


RADIX = 19
GEN_SCRIPT = str(_HERE / 'gen_radix19.py')


@dataclass(frozen=True)
class Candidate:
    variant: str
    isa: str
    id: str
    protocol_override: str | None = None
    requires_avx512: bool = False


_ME_GRID = [8, 16, 32, 64, 128, 256]


def _ios_for_me(me: int) -> list[int]:
    return [me, me + 8, 19 * me, 32 * me]


def sweep_grid(variant_id: str) -> list[tuple[int, int]]:
    return [(me, ios) for me in _ME_GRID for ios in _ios_for_me(me)]


_AVX2_VARIANTS   = ['ct_t1_dit', 'ct_t1s_dit', 'ct_t1_dit_log3']
_AVX512_VARIANTS = ['ct_t1_dit', 'ct_t1s_dit', 'ct_t1_dit_log3']


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
