"""
radixes/r16/candidates.py — R=16 candidate enumeration (Phase A + Phase B).

Phase A (4 variants, one per dispatcher):
  ct_t1_dit            flat DIT baseline
  ct_t1_dif            flat DIF
  ct_t1s_dit           (R-1)=15 scalars broadcast, K-blocked
  ct_t1_dit_log3       sparse-log3 reading W^1,W^2,W^4,W^8

Phase B (6 parameterized buf variants, all in one dispatcher):
  ct_t1_buf_dit_tile{T}_{drain}
    tile ∈ {64, 128, 256}, drain ∈ {temporal, stream}

  All 6 compete for the 't1_buf_dit' dispatcher — planner picks the
  fastest (tile, drain) per (me, ios) from bench.

Still deferred to later:
  ct_t1_oop_dit        out-of-place signature differs
  ct_t1_dit_log1       not emitted by R=16 generator yet

Sweep grid:
  Standard: me ∈ {64, 128, 256, 512, 1024, 2048}, ios ∈ {me, me+8, 8*me}
  t1s:      me ∈ {64, 128} only (K-blocked scalar broadcasts)
  buf:      standard grid, but skip combinations where me < tile
            (the buf kernel requires me ≥ tile for a full tile iteration)
"""
from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
import gen_radix16 as _gen  # noqa: E402


RADIX = 16
GEN_SCRIPT = str(_HERE / 'gen_radix16.py')


@dataclass(frozen=True)
class Candidate:
    variant: str
    isa: str
    id: str
    protocol_override: str | None = None
    requires_avx512: bool = False


_ME_RANGES = [64, 128, 256, 512, 1024, 2048]
_ME_T1S    = [64, 128]

def _ios_for_me(me: int) -> list[int]:
    return [me, me + 8, 8 * me]


def sweep_grid(variant_id: str) -> list[tuple[int, int]]:
    if variant_id == 'ct_t1s_dit':
        mes = _ME_T1S
    elif _gen.is_buf_variant(variant_id):
        # buf requires me ≥ tile (kernel processes tile-sized m-blocks)
        tile, _ = _gen._parse_buf_variant(variant_id)
        mes = [me for me in _ME_RANGES if me >= tile]
    else:
        mes = _ME_RANGES
    return [(me, ios) for me in mes for ios in _ios_for_me(me)]


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
