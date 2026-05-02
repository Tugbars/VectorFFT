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
sys.path.insert(0, str(_HERE.parent.parent / 'common'))
import gen_radix16 as _gen  # noqa: E402
import grids as _grids  # noqa: E402


RADIX = 16
GEN_SCRIPT = str(_HERE / 'gen_radix16.py')

# Per-radix grid density preference. Override with the _ME / _IOS env vars
# set by bench.py --me-density / --ios-density, or override globally by
# editing these defaults.
#
# R=16 defaults to 'fine' on me because it has the most competing variants
# (flat/t1s/log3/DIF-log3/isub2/log_half/buf × tile × drain) and the RL AVX2
# bench showed a 6/6/6 even split — the most potential for fine-grained
# winner-flip discovery anywhere in the portfolio.
_GRID_DENSITY_ME  = 'fine'
_GRID_DENSITY_IOS = 'medium'


def _effective_me_density() -> str:
    import os
    env = os.environ.get('VFFT_ME_DENSITY')
    if env:
        return env
    return _GRID_DENSITY_ME


def _effective_ios_density() -> str:
    import os
    env = os.environ.get('VFFT_IOS_DENSITY')
    if env:
        return env
    return _GRID_DENSITY_IOS


@dataclass(frozen=True)
class Candidate:
    variant: str
    isa: str
    id: str
    protocol_override: str | None = None
    requires_avx512: bool = False


def sweep_grid(variant_id: str) -> list[tuple[int, int]]:
    me_density = _effective_me_density()
    ios_density = _effective_ios_density()
    full_me_grid = _grids.me_grid(RADIX, density=me_density)

    if variant_id == 'ct_t1s_dit':
        # t1s only benches the two smallest me values (K-blocked broadcast
        # pattern; larger me values saturate cache with full twiddle table).
        # In 'fine'/'ultra' grids this picks up slightly more points.
        mes = [me for me in full_me_grid if me <= 128]
    elif _gen.is_buf_variant(variant_id):
        # buf requires me >= tile (kernel processes tile-sized m-blocks)
        tile, _ = _gen._parse_buf_variant(variant_id)
        mes = [me for me in full_me_grid if me >= tile]
    else:
        mes = full_me_grid

    return [(me, ios) for me in mes
                       for ios in _grids.ios_grid(me, ios_density)]


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
