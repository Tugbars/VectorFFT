"""
radixes/r13/candidates.py — R=13 candidate enumeration.

Tuning scope:
  - Twiddle strategies: flat / t1s / log3
  - Unroll factor: U=1 only on BOTH ISAs.

    R=13 is a prime radix with a monolithic DFT-13 butterfly: 13 inputs
    (26 registers for re/im pairs) + 21 genfft constants + sign_flip
    ≈ 47 registers needed at U=1. Both AVX2 (16 YMM) and AVX-512 (32 ZMM)
    spill heavily already. The compiler handles via Sethi-Ullman
    scheduling and store-to-load forwarding for spilled values. U=N
    unrolling is categorically infeasible — pressure is already 15 over
    the AVX-512 budget; U=2 would double the peak live set.

  - Skipped: DIF, oop_dit, ct_n1 family (prime-direct DFT has no
    composite factorization, so n1 variants are not called by the
    executor for R=13).

Dispatcher layout:
  t1_dit       ← ct_t1_dit      (AVX2 U=1; AVX-512 U=1)
  t1s_dit      ← ct_t1s_dit     (AVX2 U=1; AVX-512 U=1)
  t1_dit_log3  ← ct_t1_dit_log3 (AVX2 U=1; AVX-512 U=1)

Sweep grid:
  me ∈ {8, 16, 32, 64, 128, 256}
    Compatible with VL=4 (AVX2) and VL=8 (AVX-512).
  ios ∈ {me, me+8, 13·me, 32·me}
    Stride 13·me reflects the natural R=13 packing (13 rows × me doubles);
    32·me exercises DTLB-pressure regime.

Notes:
  Expected register pressure effects: with 47-register peak need, the
  AVX2 compiler will spill ~30 temporaries; AVX-512 will spill ~15.
  Store-to-load forwarding absorbs these reasonably well on modern
  Intel P-cores, but performance variance across protocols may be
  higher than for R=11 (10 constants, milder pressure). Expect t1s
  to struggle more than it does for R=11 since every twiddle broadcast
  adds to the already-stressed live set.
"""
from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
import gen_radix13 as _gen  # noqa: E402


RADIX = 13
GEN_SCRIPT = str(_HERE / 'gen_radix13.py')


@dataclass(frozen=True)
class Candidate:
    variant: str
    isa: str
    id: str
    protocol_override: str | None = None
    requires_avx512: bool = False


_ME_GRID = [8, 16, 32, 64, 128, 256]


def _ios_for_me(me: int) -> list[int]:
    return [me, me + 8, 13 * me, 32 * me]


def sweep_grid(variant_id: str) -> list[tuple[int, int]]:
    """Grid for a given candidate. Prime radix has no U=N variants so
    no me-based skips needed."""
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
