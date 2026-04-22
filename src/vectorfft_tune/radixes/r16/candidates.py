"""
radixes/r16/candidates.py — R=16 candidate enumeration (Phase A + Phase B).

Phase A variants:
  ct_t1_dit              flat DIT baseline
  ct_t1_dif              flat DIF (required by transpose-free roundtrip)
  ct_t1s_dit             (R-1)=15 scalars broadcast, K-blocked
  ct_t1_dit_log3         sparse-log3 reading W^1, W^2, W^4, W^8
  ct_t1_dif_log3         DIF counterpart of the log3 variant — twiddles
                         applied in PASS 2 instead of PASS 1. Same 11
                         derivations, same protocol. Competes in its
                         own dispatcher 't1_dif_log3'.

AVX-512-only specialists (gated via _gen.supported_isas()):
  ct_t1_dit_log3_isub2   paired sub-FFT scheduling (~30 ZMM peak)
  ct_t1_dit_log_half     8 loaded bases + 7 depth-1 derivations (16 ZMM
                         for bases alone)

Phase B buf variants (pruned to temporal + tile64/128 based on measured data):
  ct_t1_buf_dit_tile64_temporal
  ct_t1_buf_dit_tile128_temporal

  tile256_*: pruned. outbuf (64 KiB) > 48 KiB L1d on Raptor Lake;
             44–58% slower than tile128 at every me ≥ 256 cell.
  *_stream:  pruned. Next stage reads output immediately in recursive
             FFT; cache-bypass hint is actively wrong. 3–4× slower
             than temporal cross-radix.

Sweep grid:
  Standard: me ∈ {64, 128, 256, 512, 1024, 2048}, ios ∈ {me, me+8, 8*me}
  t1s:      me ∈ {64, 128} only (K-blocked scalar broadcasts)
  buf:      standard grid, but skip combinations where me < tile

Per-variant ISA gating:
  enumerate_all() consults _gen.supported_isas(variant) and skips the
  variant entirely for incompatible ISAs. This keeps AVX-512 specialists
  out of AVX2 bench runs at enumeration time, not just at runtime via
  host-capability detection.
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
        # buf requires me >= tile (kernel processes tile-sized m-blocks)
        tile, _ = _gen._parse_buf_variant(variant_id)
        mes = [me for me in _ME_RANGES if me >= tile]
    else:
        mes = _ME_RANGES
    return [(me, ios) for me in mes for ios in _ios_for_me(me)]


_ISAS = ['avx2', 'avx512']


def enumerate_all() -> list[Candidate]:
    out: list[Candidate] = []
    for variant in _gen.VARIANTS:
        # Per-variant ISA gating: if the generator declares supported_isas,
        # honour the restriction at enumeration time. This keeps AVX-512-only
        # specialists (isub2, log_half) out of AVX2 bench runs entirely.
        allowed = _gen.supported_isas(variant)
        for isa in _ISAS:
            if allowed is not None and isa not in allowed:
                continue
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
