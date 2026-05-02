"""
radixes/r12/candidates.py — R=12 composite candidate enumeration.

R=12 is the lightest composite we tune (4×3 = DFT-4 × DFT-3):
  - Pass 1: 3 parallel DFT-4 sub-FFTs (zero constants, 8 YMM peak per sub-FFT)
  - Pass 2: 4 parallel DFT-3 columns (2 constants KHALF, KS, 10 YMM peak)
  - 12-slot spill buffer between passes
  - Internal W12: 5 unique exponents but only 3 require cmul (W12^3 = -j and
    W12^6 = -1 are implemented as free swap-and-negate / negation)
  - 3 hoisted internal twiddle pairs × 2 = 6 ZMMs for W12 constants
  - Plus sign_flip mask for the j-rotation = 1 ZMM
  - Plus DFT-3 constants KHALF, KS = 2 ZMMs
  - Total always-live (non-t1s): 9 ZMMs on AVX-512

Tuning scope:
  AVX2:    flat / t1s / log3 (U=1 only, 3 variants)
  AVX-512: flat / t1s / log3 (U=1) + flat U=2a/U=2b + log3 U=2a/U=2b
           (7 variants)

U=2 register analysis (AVX-512):
  Per-butterfly working set: ~12-14 ZMMs peak (smaller than R=10's ~22)
  U=2 total estimate: 9 always-live + 24-28 two-lane = 33-37 ZMMs
  Within 1-5 of the 32-ZMM budget — A and B should be register-clean or
  close to it. First composite where U=2 is predicted to fit cleanly.

t1s U=2 infeasible: 11 hoisted twiddle pairs alone = 22 ZMMs, plus 9
  always-live = 31 ZMMs before the butterfly starts.

U=3 excluded: estimated 9 + ~40 = 49 ZMMs, moderate spill similar to
  R=10 U=2 which lost on container. Signal-to-noise too low to justify.

Sweep grid:
  me ∈ {8, 16, 32, 64, 128, 256}   (matches R=10/R=20/R=25 for comparability)
  ios ∈ {me, me+8, 12*me, 32*me}   (pow2, padded, 12×me natural, large pow2)
"""
from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
import gen_radix12 as _gen  # noqa: E402


RADIX = 12
GEN_SCRIPT = str(_HERE / 'gen_radix12.py')


@dataclass(frozen=True)
class Candidate:
    variant: str
    isa: str
    id: str
    protocol_override: str | None = None
    requires_avx512: bool = False


_ME_GRID = [8, 16, 32, 64, 128, 256]

def _ios_for_me(me: int) -> list[int]:
    return [me, me + 8, 12 * me, 32 * me]


def sweep_grid(variant_id: str) -> list[tuple[int, int]]:
    # U=2 variants require me >= 16 so the 2*VL=16 main loop runs at least once.
    if '_u2' in variant_id:
        return [(me, ios) for me in _ME_GRID if me >= 16 for ios in _ios_for_me(me)]
    return [(me, ios) for me in _ME_GRID for ios in _ios_for_me(me)]


_AVX512_ONLY = {'ct_t1_dit_u2a', 'ct_t1_dit_u2b',
                'ct_t1_dit_log3_u2a', 'ct_t1_dit_log3_u2b'}


def enumerate_all() -> list[Candidate]:
    out: list[Candidate] = []
    for variant in _gen.VARIANTS:
        if variant in _AVX512_ONLY:
            isas = ['avx512']
        else:
            isas = ['avx2', 'avx512']
        for isa in isas:
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
