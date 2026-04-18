"""R=64 candidate enumeration.

The R=64 generator now has per-variant knobs for ct_t1_buf_dit:
  --tile (int, default 64/AVX2 or 32/AVX-512)
  --drain {temporal, stream}
  --drain-prefetch (flag)

Other variants still have no knobs.

Families benched:
  ct_t1_dit           - flat baseline (what VTune profiled)
  ct_t1_dif           - flat DIF variant
  ct_t1_dit_log3      - log3 derived twiddles
  ct_t1_dit_prefetch  - flat with built-in twiddle prefetch
                        VTune: regressed 2.5-6.6% on Raptor Lake AVX2
                        Keep as negative control to measure cross-chip
  ct_t1s_dit          - scalar-broadcast twiddles (NEW, just ported)
                        SPR AVX2 won 92% of regions in previous bench
  ct_t1_buf_dit       - buffered output via scratch tile (NEW, just ported)
                        Targets store DTLB pressure (43.7% of stalls per
                        VTune). Knob matrix: tile × drain × prefw.

Not included:
  ct_t1_ladder_dit    - not in R=64 generator; VTune shows port utilization
                        is LOWER at R=64 than R=32 (memory stalls dominate),
                        so register-pressure story doesn't apply
"""
from dataclasses import dataclass
from typing import List, Tuple

ASSUMED_L1D_BYTES = 48 * 1024
ASSUMED_L2_BYTES = 2 * 1024 * 1024
SIZEOF_COMPLEX = 16
R = 64


@dataclass(frozen=True)
class Candidate:
    family: str
    isa: str
    tile: int = None
    drain: str = None
    drain_prefetch: bool = None

    def id(self) -> str:
        parts = [self.family, self.isa]
        if self.family == 'ct_t1_buf_dit':
            parts.append(f'tile{self.tile}')
            parts.append(f'drain{self.drain}')
            if self.drain_prefetch:
                parts.append('prefw')
        return '__'.join(parts)

    def header_name(self) -> str:
        return f'{self.id()}.h'

    def cli_args(self) -> List[str]:
        args = ['--isa', self.isa, '--variant', self.family]
        if self.family == 'ct_t1_buf_dit':
            args += ['--tile', str(self.tile), '--drain', self.drain]
            if self.drain_prefetch:
                args.append('--drain-prefetch')
        return args

    def function_names(self) -> dict:
        base = 'radix64_'
        f = self.family
        if f == 'ct_t1_dit':
            base += 't1_dit'
        elif f == 'ct_t1_dif':
            base += 't1_dif'
        elif f == 'ct_t1_dit_log3':
            base += 't1_dit_log3'
        elif f == 'ct_t1_dit_prefetch':
            base += 't1_dit_prefetch'
        elif f == 'ct_t1s_dit':
            base += 't1s_dit'
        elif f == 'ct_t1_buf_dit':
            base += f't1_buf_dit_tile{self.tile}_{self.drain}'
            if self.drain_prefetch:
                base += '_prefw'
        else:
            raise ValueError(f"Unknown family: {f}")
        return {'fwd': f'{base}_fwd_{self.isa}',
                'bwd': f'{base}_bwd_{self.isa}'}


# Sweep
SWEEP_ME = [64, 128, 256, 512, 1024, 2048]
SWEEP_IOS_OFFSETS = [0, 8, 64]


def sweep_points() -> List[Tuple[int, int]]:
    return [(me + off, me) for me in SWEEP_ME for off in SWEEP_IOS_OFFSETS]


def _stream_useful() -> bool:
    # Stream stores bypass cache. Worthwhile when output approaches or
    # exceeds L2 — forcing cache won't help anyway (output won't be
    # re-read while in L2). At R=64 me=2048: 64 * 2048 * 16 = 2MB,
    # matches 2MB L2 exactly. Include stream variants.
    return R * max(SWEEP_ME) * SIZEOF_COMPLEX >= ASSUMED_L2_BYTES


def _drain_modes() -> List[str]:
    return ['temporal', 'stream'] if _stream_useful() else ['temporal']


def _buf_tiles() -> List[int]:
    # DTLB math for R=64:
    # TILE=32:  outbuf = 64*32 * 16 = 32KB   - fits L1 comfortably
    # TILE=64:  outbuf = 64*64 * 16 = 64KB   - near/spills L1 (48KB Raptor)
    # TILE=128: outbuf = 64*128 * 16 = 128KB - spills L1, fits L2
    # TILE=256: outbuf = 64*256 * 16 = 256KB - L2 pressure
    # Keep 256 out — likely to thrash.
    return [32, 64, 128]


def enumerate_all() -> List[Candidate]:
    """Per ISA: 5 simple variants + (3 tiles × N drain modes × 2 prefw) buf variants.

    With _stream_useful() == True at R=64 me=2048: 2 drain modes per tile.
    → 5 + 3*2*2 = 17 candidates per ISA
    → 34 candidates total (AVX2 + AVX-512)
    """
    cands = []
    for isa in ('avx2', 'avx512'):
        cands.append(Candidate(family='ct_t1_dit', isa=isa))
        cands.append(Candidate(family='ct_t1_dif', isa=isa))
        cands.append(Candidate(family='ct_t1_dit_log3', isa=isa))
        cands.append(Candidate(family='ct_t1_dit_prefetch', isa=isa))
        cands.append(Candidate(family='ct_t1s_dit', isa=isa))
        for tile in _buf_tiles():
            for drain in _drain_modes():
                for drain_pf in (False, True):
                    cands.append(Candidate(
                        family='ct_t1_buf_dit', isa=isa,
                        tile=tile, drain=drain, drain_prefetch=drain_pf))
    return cands


if __name__ == '__main__':
    cands = enumerate_all()
    print(f'{len(cands)} candidates')
    for c in cands:
        fns = c.function_names()
        print(f'  {c.id():60s} fwd={fns["fwd"]}')

