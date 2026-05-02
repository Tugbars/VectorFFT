"""
common/grids.py — Configurable sweep grids for codelet benching.

Grid density controls how many (me, ios) points each variant gets benched at.
Denser grids catch more me-dependent winner flips but take longer. Coarser
grids are adequate when winners are stable across the me range (typical for
prime radixes with monolithic butterflies).

## Grid density levels

### me axis
- coarse: 3 points (quick sanity check; not for production wisdom)
- medium: 6 points = {64, 128, 256, 512, 1024, 2048}
          (paper default; adequate for radixes with stable winners)
- fine:   11 points = medium + {96, 192, 384, 768, 1536}
          (power-of-2 radixes where variant flips show up between dyadic
          boundaries; R=16 and R=32 are the clear beneficiaries)
- ultra:  24 points (research-grade, seldom needed)

### ios axis
- coarse:  2 points = {me, me+8}               (aligned + padded, minimum)
- medium:  3 points = {me, me+8, 8*me}         (current default)
- fine:    6 points = {me, me+8, me+64, 2*me, 4*me, 8*me}
           (characterizes prefetcher stream-tracking vs large-stride regimes)
- ultra:   9 points = fine + {16*me, 32*me, page-crossing stride}

## Per-radix defaults

Power-of-2 radixes (4, 8, 16, 32, 64) default to fine me grid because
their variant space has real me-dependent winners (U=1/U=2/U=3 variants,
log3/log_half/isub2, buf tiles). All other radixes default to medium.

Override per-radix in candidates.py by setting:
    _GRID_DENSITY_ME  = 'fine'   # or 'coarse', 'ultra'
    _GRID_DENSITY_IOS = 'medium' # or 'coarse', 'fine', 'ultra'

Override globally from the CLI:
    python bench.py --radix-dir radixes/r16 --me-density ultra --ios-density fine
"""
from __future__ import annotations


# Power-of-two ladder: where CPU microarchitecture boundaries actually live
# (cache line = 64 B = 8 doubles; L1d = 48 KiB = 6144 doubles; page = 4 KiB = 512 doubles).
# Non-power-of-2 intermediate values would sample regimes nothing in production
# runs at and provide little mechanistic insight.
ME_GRIDS = {
    'coarse': [64, 256, 1024],
    'medium': [64, 128, 256, 512, 1024, 2048],
    'fine':   [64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048],
    'ultra':  [64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512,
               640, 768, 896, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 4096],
}


def ios_grid(me: int, density: str = 'medium') -> list[int]:
    """Return ios points for a given me and density.

    Grids:
      coarse: [me, me+8]                               -- minimum viable
      medium: [me, me+8, 8*me]                         -- current default
      fine:   [me, me+8, me+64, 2*me, 4*me, 8*me]      -- prefetcher characterization
      ultra:  [me, me+8, me+64, 2*me, 4*me, 8*me, 16*me, 32*me]
    """
    if density == 'coarse':
        return [me, me + 8]
    if density == 'medium':
        return [me, me + 8, 8 * me]
    if density == 'fine':
        # Aligned, near-aligned (prefetcher stress), cache-line stride,
        # 2x/4x/8x multiples. Each point probes a distinct access regime.
        return [me, me + 8, me + 64, 2 * me, 4 * me, 8 * me]
    if density == 'ultra':
        return [me, me + 8, me + 64, 2 * me, 4 * me, 8 * me, 16 * me, 32 * me]
    raise ValueError(f"unknown ios grid density: {density!r}")


# Radixes that benefit from fine me-gridding by default.
# These have enough variant machinery (unroll factors, log3/log_half/isub2,
# buf tiles) that winners flip across me values — dense sampling pays off.
# All other radixes default to medium.
_POW2_RADIXES = {4, 8, 16, 32, 64}


def default_me_density(radix: int) -> str:
    return 'fine' if radix in _POW2_RADIXES else 'medium'


def default_ios_density(radix: int) -> str:
    # ios density doesn't need to scale with radix class; the prefetcher
    # effects are radix-agnostic. Medium is a good default; users who want
    # to characterize the prefetcher boundary bump it to fine.
    return 'medium'


def me_grid(radix: int, density: str | None = None,
            override: list[int] | None = None) -> list[int]:
    """Resolve the me grid for a radix.

    Precedence (highest first):
      1. `override` parameter (explicit numeric list from candidates.py)
      2. `density` parameter (from candidates.py _GRID_DENSITY_ME or CLI)
      3. default_me_density(radix) (fine for pow2, medium otherwise)
    """
    if override is not None:
        return list(override)
    if density is None:
        density = default_me_density(radix)
    return list(ME_GRIDS[density])
