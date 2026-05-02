"""
radixes/r8/candidates.py — R=8 candidate enumeration (Phase B).

Candidate matrix (8 variants):

  DIT family (dispatcher=t1_dit, flat protocol):
    ct_t1_dit              baseline — 7 twiddle loads per butterfly
    ct_t1_dit_prefetch     baseline + SW prefetch of next block's twiddles
    ct_t1_dit_log1         6 twiddle loads + derive W^7 = W^3 * W^4
    ct_t1_dit_u2           AVX-512 ONLY — 2x m-loop unroll, hides DFT-8
                           dependency chain via cross-block ILP

  DIF family (dispatcher=t1_dif, flat protocol):
    ct_t1_dif              baseline — butterfly first, post-twiddle outputs
    ct_t1_dif_prefetch     baseline + SW prefetch

  Log3 (dispatcher=t1_dit_log3, log3 protocol):
    ct_t1_dit_log3         CT-signature — load 3 bases W^1,W^2,W^4,
                           derive W^3,W^5,W^6,W^7

  T1s (dispatcher=t1s_dit, t1s protocol):
    ct_t1s_dit             hoisted 7 scalar broadcasts, K-blocked execution

Sweep grid:
  Most variants: me ∈ {64, 128, 256, 512, 1024, 2048}, ios ∈ {me, me+8, 8*me}
  t1s_dit:       me ∈ {64, 128} only (K-blocked pattern — scalar twiddles
                 amortize across K iterations, so t1s is only relevant at
                 small me where each m-iteration is cheap).
"""
from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent / 'common'))
import gen_radix8 as _gen  # noqa: E402
import grids as _grids  # noqa: E402


RADIX = 8
GEN_SCRIPT = str(_HERE / 'gen_radix8.py')

# R=8 defaults to 'fine' me grid because it's a power-of-2 radix with
# multiple unroll variants (U=1/U=2/U=3) that compete at short loops.
_GRID_DENSITY_ME  = 'fine'
_GRID_DENSITY_IOS = 'medium'


def _effective_me_density() -> str:
    import os
    return os.environ.get('VFFT_ME_DENSITY', _GRID_DENSITY_ME)


def _effective_ios_density() -> str:
    import os
    return os.environ.get('VFFT_IOS_DENSITY', _GRID_DENSITY_IOS)


@dataclass(frozen=True)
class Candidate:
    variant: str
    isa: str
    id: str
    protocol_override: str | None = None
    requires_avx512: bool = False


def sweep_grid(variant_id: str) -> list[tuple[int, int]]:
    me_density  = _effective_me_density()
    ios_density = _effective_ios_density()
    full_me_grid = _grids.me_grid(RADIX, density=me_density)

    if variant_id == 'ct_t1s_dit':
        mes = [me for me in full_me_grid if me <= 128]
    else:
        mes = full_me_grid

    return [(me, ios) for me in mes
                       for ios in _grids.ios_grid(me, ios_density)]


_ISAS = ['avx2', 'avx512']

# u2 is AVX-512 only (AVX2 has only 16 YMM — catastrophic spill with 2x unroll)
_AVX512_ONLY = {'ct_t1_dit_u2'}


def enumerate_all() -> list[Candidate]:
    out: list[Candidate] = []
    for variant in _gen.VARIANTS:
        for isa in _ISAS:
            if variant in _AVX512_ONLY and isa != 'avx512':
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
