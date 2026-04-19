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
import gen_radix4 as _gen  # noqa: E402


RADIX = 4
GEN_SCRIPT = str(_HERE / 'gen_radix4.py')


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
# Sweep grid
#
# For each me, bench ios in {me, me+8, 8*me}:
#   me:    power-of-2 stride (worst-case aliasing)
#   me+8:  minimally padded stride (alias-free)
#   8*me:  large stride (memory-latency-bound regime)
# ═══════════════════════════════════════════════════════════════

_ME_RANGES = [64, 128, 256, 512, 1024, 2048]
_ME_T1S    = [64, 128]  # t1s only benched at small me (K-blocked pattern)


def _ios_for_me(me: int) -> list[int]:
    return [me, me + 8, 8 * me]


def sweep_grid(variant_id: str) -> list[tuple[int, int]]:
    """Return the (me, ios) points where this variant should be benched."""
    if variant_id == 'ct_t1s_dit':
        mes = _ME_T1S
    else:
        mes = _ME_RANGES
    return [(me, ios) for me in mes for ios in _ios_for_me(me)]


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
