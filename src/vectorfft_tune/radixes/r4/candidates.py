"""
radixes/r4/candidates.py — R=4 candidate enumeration.

Exposes the interface consumed by common/bench.py:
  RADIX:          integer radix value
  GEN_SCRIPT:     path to the code generator
  enumerate_all() -> list[Candidate]
  sweep_grid(variant_id) -> list[tuple[me, ios]]
  function_name(variant_id, isa, direction) -> str
  protocol(variant_id) -> str
"""
from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass, field

# Import the generator's VARIANTS metadata directly.
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent / 'common'))
import gen_radix4 as _gen  # noqa: E402
import grids as _grids  # noqa: E402


RADIX = 4
GEN_SCRIPT = str(_HERE / 'gen_radix4.py')

# Per-radix grid density preference. R=4 defaults to 'fine' on me because
# it's power-of-2 and has U=1/U=2/U=3 unroll variants that can flip winners
# across me values (compiler scheduling sensitivity on short loops).
# Override via bench.py --me-density/--ios-density, or by editing here.
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
    """One (variant, isa, protocol_override) tuple that the bench will measure.

    Notes:
      - `protocol_override` is None for regular candidates (uses the variant's
        declared protocol). It's non-None only for the log1_tight experiment:
        the same codelet body (`ct_t1_dit_log1`) is benched with a tighter
        twiddle buffer allocation to isolate memory-footprint handicap.
      - `isa` is resolved against the host's CPU capabilities at run time by
        the harness (auto-skip AVX-512 on non-AVX-512 hosts).
    """
    variant: str
    isa: str
    id: str                          # unique bench id, e.g. "ct_t1_dit_u2__avx2"
    protocol_override: str | None = None
    requires_avx512: bool = False


# ═══════════════════════════════════════════════════════════════
# Sweep grid (density-configurable — see common/grids.py)
#
# me axis:  controlled by _GRID_DENSITY_ME or VFFT_ME_DENSITY env.
#           Defaults to 'fine' for R=4 (power-of-2 radix with unroll variants).
# ios axis: controlled by _GRID_DENSITY_IOS or VFFT_IOS_DENSITY env.
#           Defaults to 'medium' which is {me, me+8, 8*me}:
#             me    — power-of-2 stride (worst-case aliasing)
#             me+8  — minimally padded stride (alias-free)
#             8*me  — large stride (memory-latency-bound regime)
# ═══════════════════════════════════════════════════════════════


def sweep_grid(variant_id: str) -> list[tuple[int, int]]:
    """Return the (me, ios) points where this variant should be benched."""
    me_density  = _effective_me_density()
    ios_density = _effective_ios_density()
    full_me_grid = _grids.me_grid(RADIX, density=me_density)

    if variant_id == 'ct_t1s_dit':
        # t1s benched only at small me (K-blocked pattern; t1s loses at
        # large me because scalar broadcasts serialize the inner loop).
        mes = [me for me in full_me_grid if me <= 128]
    else:
        mes = full_me_grid

    return [(me, ios) for me in mes
                       for ios in _grids.ios_grid(me, ios_density)]


# ═══════════════════════════════════════════════════════════════
# Candidate enumeration
# ═══════════════════════════════════════════════════════════════

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
    # Handicap experiment: log1 with tight twiddle buffer (2*me instead of 3*me).
    # Same codelet body as ct_t1_dit_log1; only the harness allocation differs.
    for isa in _ISAS:
        out.append(Candidate(
            variant='ct_t1_dit_log1',
            isa=isa,
            id=f'ct_t1_dit_log1_tight__{isa}',
            protocol_override='log1_tight',
            requires_avx512=(isa == 'avx512'),
        ))
    return out


# ═══════════════════════════════════════════════════════════════
# Naming & protocol passthrough
# ═══════════════════════════════════════════════════════════════

def function_name(variant_id: str, isa: str, direction: str) -> str:
    return _gen.function_name(variant_id, isa, direction)


def protocol(variant_id: str) -> str:
    return _gen.protocol(variant_id)


def dispatcher(variant_id: str) -> str:
    return _gen.dispatcher(variant_id)
