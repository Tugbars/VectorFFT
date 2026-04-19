"""
radixes/r8/candidates.py — R=8 candidate enumeration (Phase A).

Current candidate matrix (small, intentionally):
  ct_t1_dit            — flat baseline (DIT)
  ct_t1_dif            — flat DIF (VTune: ~10% faster than DIT on Raptor Lake)
  ct_t1_dit_prefetch   — DIT with SW twiddle prefetch; NEGATIVE CONTROL
                          (VTune on Raptor Lake showed +8-15% regression; we
                          still bench it to see if that holds across chips —
                          Zen with weaker HW prefetch might flip it)

Phase B candidates (not yet ported to gen_radix8):
  ct_t1s_dit           — scalar-broadcast twiddles (R-1=7 scalars)
  ct_t1_dit_log1       — load w1+w2+...+w6, derive w7
  ct_t1_dit_log3       — derive w2..w7 from 3 bases (exists but OOP signature)
  ct_t1_dit_u2         — 2x m-loop unroll (AVX-512 only per VTune — 32 ZMM)

Sweep grid: same as R=4. No reason to differ (codelet is called at the same
range of (me, ios) in larger plans).
"""
from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
import gen_radix8 as _gen  # noqa: E402


RADIX = 8
GEN_SCRIPT = str(_HERE / 'gen_radix8.py')


@dataclass(frozen=True)
class Candidate:
    variant: str
    isa: str
    id: str
    protocol_override: str | None = None
    requires_avx512: bool = False


# Sweep grid (shared with R=4).
_ME_RANGES = [64, 128, 256, 512, 1024, 2048]

def _ios_for_me(me: int) -> list[int]:
    return [me, me + 8, 8 * me]

def sweep_grid(variant_id: str) -> list[tuple[int, int]]:
    return [(me, ios) for me in _ME_RANGES for ios in _ios_for_me(me)]


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
