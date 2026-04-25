"""
radixes/r64/candidates.py — R=64 candidate enumeration (Phase A + Phase B).

Phase A variants:
  ct_t1_dit              flat DIT baseline — 63 twiddle loads per butterfly
  ct_t1_dif              flat DIF (required by transpose-free roundtrip)
  ct_t1s_dit             (R-1)=63 scalars broadcast, K-blocked
  ct_t1_dit_log3         sparse-log3: reads W^1, W^8 from flat buffer,
                         derives remaining 61 twiddles via cmul chains
                         (aggressive 2-base FFTW-style log3 for R=64)
  ct_t1_dif_log3         DIF counterpart of log3 — twiddles applied in
                         PASS 2 after the column butterfly rather than in
                         PASS 1 before the sub-FFT. Same derivation chain,
                         same protocol; competes in its own dispatcher
                         't1_dif_log3'. Paper-result high-radix data point:
                         the decisive 17/18 DIF-log3 sweep on RL AVX2.

AVX-512-only specialists (gated via _gen.supported_isas()):
  ct_t1_dit_log3_isub2   paired sub-FFT scheduling. Intel container:
                         3W/9T/3L (mixed — less decisive than R=32's loss).
                         Carried for AMD EPYC Zen4/Zen5 evaluation where
                         the 3-load-port profile may favor the paired-
                         parallelism mechanism more clearly.
                         Shares dispatcher 't1_dit_log3' with plain log3.

Phase B buf variants (pruned to temporal + tile64/128 per cross-radix evidence):
  ct_t1_buf_dit_tile64_temporal
  ct_t1_buf_dit_tile128_temporal

  tile256_*: pruned. outbuf (64 KiB) > 48 KiB L1d on Raptor Lake;
             never wins vs tile128 at me >= 256 cells (cross-radix
             evidence R=16 + R=32 + R=64).
  *_stream:  pruned. Next stage reads output immediately in recursive
             FFT; cache-bypass hint is actively wrong. 3-4x slower
             than temporal cross-radix.

Caveat: at R=64, buf variants compile to very large code bodies (~25k
lines per variant on AVX2). Header file is now smaller (~40k lines total
per ISA after pruning) than the original 54k. Expected bench runtime:
~4-6 min on Raptor Lake.

Still deferred:
  ct_t1_buf_dit          drain_prefetch knob
  ct_t1_oop_dit          out-of-place signature
  ct_t1_dit_prefetch     SW prefetch variant

Sweep grid:
  Standard: me ∈ {64, 128, 256, 512, 1024, 2048}, ios ∈ {me, me+8, 8*me}
  t1s:      me ∈ {64, 128} only (K-blocked pattern)
  buf:      skip points where me < tile

Per-variant ISA gating:
  enumerate_all() consults _gen.supported_isas(variant) and skips the
  variant entirely for incompatible ISAs. This keeps AVX-512 specialists
  (isub2) out of AVX2 bench runs at enumeration time, not just at runtime
  via host-capability detection.
"""
from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent / 'common'))
import gen_radix64 as _gen  # noqa: E402
import grids as _grids  # noqa: E402


RADIX = 64
GEN_SCRIPT = str(_HERE / 'gen_radix64.py')

# R=64 defaults to 'fine' me grid. This is the paper's decisive-sweep
# data point (13/18 log3 wins on RL AVX2, 72% of the grid) — dense sampling
# here gives the cleanest confirmation that log3 wins everywhere, and lets
# us characterize any near-boundary flat/log3 crossovers not caught at
# the medium grid.
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
    elif _gen.is_buf_variant(variant_id):
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
        # specialists (isub2) out of AVX2 bench runs entirely.
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
