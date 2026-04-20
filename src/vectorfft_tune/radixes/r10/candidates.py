"""
radixes/r10/candidates.py — R=10 composite candidate enumeration.

R=10 is the smallest composite in the tuned portfolio (5×2 = DFT-5 × DFT-2):
  - Pass 1: 2 parallel DFT-5 sub-FFTs (4 constants, peak 16 YMM / 22 ZMM)
  - Pass 2: 5 parallel DFT-2 columns (zero constants, add/sub only)
  - 10-slot spill buffer between passes
  - 4 unique W10 internal twiddle exponents (hoisted as broadcasts)

Tuning scope:
  AVX2:    flat / t1s / log3 (U=1 only, 3 variants)
  AVX-512: flat / t1s / log3 (U=1) + flat U=2a/U=2b + log3 U=2a/U=2b
           (7 variants)

U=2 approaches (AVX-512 only):
  u2a — interleaved, R=3-style. Two lanes woven at the operation level with
        per-lane spill buffers. Higher ILP; both lanes' live sets coexist.
  u2b — pipelined, n1-style. Two sequential { } blocks sharing the per-m
        spill buffer. Modest ILP gain, register pressure near U=1.
  Empirical bench to determine which approach wins where (or if U=1 wins).

t1s U=2 excluded: 9 hoisted twiddle broadcasts consume 18 ZMMs already
  at U=1; with 12 other always-live constants (DFT-5 + W10), headroom is
  too tight to add a second lane's butterfly working set.
U=3 excluded at any radix variant: per-butterfly peak ~22 ZMMs + 12 always-live
  means three lanes would need ~52 ZMMs, well over 32.

Sweep grid:
  me ∈ {8, 16, 32, 64, 128, 256}  (matches R=20/R=25 for cross-composite
                                   comparability)
  ios ∈ {me, me+8, 10*me, 32*me}  (pow2, padded, 10×me natural stride,
                                   large pow2)
"""
from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
import gen_radix10 as _gen  # noqa: E402


RADIX = 10
GEN_SCRIPT = str(_HERE / 'gen_radix10.py')


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
    # U=2 variants require me >= 16 so the 2*VL=16 main loop runs at least once.
    # At me=8 only the VL=8 tail would execute, which is just U=1 — pointless.
    if '_u2' in variant_id:
        return [(me, ios) for me in _ME_GRID if me >= 16 for ios in _ios_for_me(me)]
    return [(me, ios) for me in _ME_GRID for ios in _ios_for_me(me)]


# AVX-512-only variants
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
