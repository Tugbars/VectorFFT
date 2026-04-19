"""
radixes/r64/candidates.py — R=64 candidate enumeration (Phase A).

Phase A matrix (4 variants, each its own dispatcher):

  ct_t1_dit          flat DIT baseline — 63 twiddle loads per butterfly
  ct_t1_dif          flat DIF
  ct_t1s_dit         (R-1)=63 scalars broadcast, K-blocked
  ct_t1_dit_log3     sparse-log3 reading W^1, W^8 from flat buffer;
                     derives remaining 61 twiddles via cmul chains.
                     (Different sparse pattern from R=16/R=32 — the R=64
                     FFTW-style kernel uses a 2-level decomposition with
                     only 2 base loads instead of log2(R)=6.)

Phase B (not yet ported):
  ct_t1_buf_dit       tile × drain matrix
  ct_t1_oop_dit       out-of-place signature
  ct_t1_dit_prefetch  SW prefetch variant

Sweep grid:
  Standard: me ∈ {64, 128, 256, 512, 1024, 2048}, ios ∈ {me, me+8, 8*me}
  t1s:      me ∈ {64, 128} only (K-blocked pattern)

Note: R=64 codelets are large (13K+ lines per ISA). Compile time per
candidate is noticeably longer than R=32. Expected bench runtime: ~3-5
min on Raptor Lake.
"""
from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
import gen_radix64 as _gen  # noqa: E402


RADIX = 64
GEN_SCRIPT = str(_HERE / 'gen_radix64.py')


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
    mes = _ME_T1S if variant_id == 'ct_t1s_dit' else _ME_RANGES
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
