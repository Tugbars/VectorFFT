"""
radixes/r11/candidates.py — R=11 prime candidate enumeration.

R=11 is a prime radix implemented as a monolithic genfft DAG (Sethi-Ullman
scheduled straight-line code). No internal twiddles — no N1×N2 factorization
exists for primes.

AVX2: 3-phase spill-based butterfly, peak 16 YMM
AVX-512: single-pass no-spill butterfly, exactly 32 ZMM

Tuning scope:
  - flat / t1s / log3 × U=1, both ISAs (3 variants per ISA)
  - No U=2: AVX-512 butterfly already saturates the register file
  - No buf: 22 output pages well below DTLB threshold

Known concern: t1s on AVX-512 adds 20 ZMMs of hoisted twiddle broadcasts
on top of 11 ZMMs of constants (31 ZMMs live) before the no-spill butterfly
starts. Compiler will spill. We bench anyway to let data decide whether
amortization beats the spill cost.

Sweep grid:
  me ∈ {8, 16, 32, 64, 128, 256}   (matches prime and composite conventions)
  ios ∈ {me, me+8, 11*me, 32*me}   (pow2, padded, 11×me natural, large pow2)
"""
from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
import gen_radix11 as _gen  # noqa: E402


RADIX = 11
GEN_SCRIPT = str(_HERE / 'gen_radix11.py')


@dataclass(frozen=True)
class Candidate:
    variant: str
    isa: str
    id: str
    protocol_override: str | None = None
    requires_avx512: bool = False


_ME_GRID = [8, 16, 32, 64, 128, 256]

def _ios_for_me(me: int) -> list[int]:
    return [me, me + 8, 11 * me, 32 * me]


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
