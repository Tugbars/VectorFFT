"""R=32 candidate enumeration with cache-spill gating."""
from dataclasses import dataclass
from typing import List, Tuple

ASSUMED_L1D_BYTES = 48 * 1024
ASSUMED_L2_BYTES = 2 * 1024 * 1024
SIZEOF_COMPLEX = 16
R = 32


@dataclass(frozen=True)
class Candidate:
    family: str
    isa: str
    tile: int = None
    drain: str = None
    drain_prefetch: bool = None
    twiddle_prefetch: int = 0
    twiddle_prefetch_rows: int = 1

    def id(self) -> str:
        parts = [self.family, self.isa]
        if self.family == 'ct_t1_buf_dit':
            parts.append(f'tile{self.tile}')
            parts.append(f'drain{self.drain}')
            if self.drain_prefetch:
                parts.append('prefw')
        if self.twiddle_prefetch > 0:
            parts.append(f'tpf{self.twiddle_prefetch}r{self.twiddle_prefetch_rows}')
        return '__'.join(parts)

    def header_name(self) -> str:
        return f'{self.id()}.h'

    def cli_args(self) -> List[str]:
        args = ['--isa', self.isa, '--variant', self.family]
        if self.family == 'ct_t1_buf_dit':
            args += ['--tile', str(self.tile), '--drain', self.drain]
            if self.drain_prefetch:
                args.append('--drain-prefetch')
        if self.twiddle_prefetch > 0:
            args += ['--twiddle-prefetch', str(self.twiddle_prefetch),
                     '--twiddle-prefetch-rows', str(self.twiddle_prefetch_rows)]
        return args

    def function_names(self) -> dict:
        base = 'radix32_'
        f = self.family
        if f == 'ct_t1_dit':
            base += 't1_dit'
        elif f == 'ct_t1_dit_log3':
            base += 't1_dit_log3'
        elif f == 'ct_t1_ladder_dit':
            base += 't1_ladder_dit'
        elif f == 'ct_t1_buf_dit':
            base += f't1_buf_dit_tile{self.tile}_{self.drain}'
            if self.drain_prefetch:
                base += '_prefw'
        else:
            raise ValueError(f"Unknown family: {f}")
        if self.twiddle_prefetch > 0:
            base += f'_tpf{self.twiddle_prefetch}r{self.twiddle_prefetch_rows}'
        return {'fwd': f'{base}_fwd_{self.isa}',
                'bwd': f'{base}_bwd_{self.isa}'}


SWEEP_ME = [64, 128, 256, 512, 1024, 2048]
SWEEP_IOS_OFFSETS = [0, 8, 64]

def sweep_points() -> List[Tuple[int, int]]:
    return [(me + off, me) for me in SWEEP_ME for off in SWEEP_IOS_OFFSETS]


def _stream_useful() -> bool:
    return R * max(SWEEP_ME) * SIZEOF_COMPLEX > ASSUMED_L2_BYTES

def _prefetch_useful() -> bool:
    return (R - 1) * max(SWEEP_ME) * SIZEOF_COMPLEX > ASSUMED_L1D_BYTES


def _prefetch_configs() -> List[Tuple[int, int]]:
    if not _prefetch_useful():
        return [(0, 1)]
    cfgs = [(0, 1)]
    for dist in (4, 8, 16, 32):
        cfgs.append((dist, 1))
        if dist >= 8:
            cfgs.append((dist, 2))
    return cfgs

def _drain_modes() -> List[str]:
    return ['temporal', 'stream'] if _stream_useful() else ['temporal']


def enumerate_all() -> List[Candidate]:
    cands = []
    for isa in ('avx2', 'avx512'):
        for tp, tpr in _prefetch_configs():
            cands.append(Candidate(family='ct_t1_dit', isa=isa,
                                   twiddle_prefetch=tp, twiddle_prefetch_rows=tpr))
        for tp, tpr in _prefetch_configs():
            cands.append(Candidate(family='ct_t1_dit_log3', isa=isa,
                                   twiddle_prefetch=tp, twiddle_prefetch_rows=tpr))
        if isa == 'avx512':
            cands.append(Candidate(family='ct_t1_ladder_dit', isa=isa))
        for tile in (32, 64, 128, 256):
            for drain in _drain_modes():
                for drain_pf in (False, True):
                    for tp, tpr in _prefetch_configs():
                        cands.append(Candidate(
                            family='ct_t1_buf_dit', isa=isa,
                            tile=tile, drain=drain, drain_prefetch=drain_pf,
                            twiddle_prefetch=tp, twiddle_prefetch_rows=tpr))
    return cands


if __name__ == '__main__':
    print(f'{len(enumerate_all())} candidates')
