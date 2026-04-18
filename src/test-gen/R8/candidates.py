"""R=8 candidate enumeration.

VTune profiling on Raptor Lake (K=256) showed R=8 is CPU-bound, not memory-bound:
  - Memory Bound: 1.4% (vs 30% at R=16, 66% at R=32)
  - Core Bound: 23.4% (dependency chains dominate)
  - L1 Latency Dependency: 36.8% of clockticks (chain stalls, not misses)

Memory-tier optimizations that dominated R=16/R=32 selectors all regressed
measurably on R=8 in VTune:
  - Aggressive twiddle prefetch: +15% worse
  - Lightweight twiddle prefetch:  +8% worse
  - Fused-half layout (regfile relief): +5.5% worse
  - U=2 AVX2 interleaving: +39% worse (spills 16->32 YMM needed)

So the R=8 matrix is tiny: no tile knob, no prefetch knob, no buf.
We bench the family choice and DIT-vs-DIF only, plus t1_dit_prefetch as
a negative control to confirm SW prefetch regresses on our chips too.
"""
from dataclasses import dataclass
from typing import List, Tuple

R = 8


@dataclass(frozen=True)
class Candidate:
    family: str
    isa: str

    def id(self) -> str:
        return f'{self.family}__{self.isa}'

    def header_name(self) -> str:
        return f'{self.id()}.h'

    def function_names(self) -> dict:
        """R=8 generator emits symbols like 'radix8_t1_dit_fwd_avx2',
        'radix8_t1_dif_fwd_avx2', 'radix8_t1_dit_prefetch_fwd_avx2'.
        Family names here strip 'ct_' prefix since R=8 generator doesn't
        use the ct_ namespace.
        """
        fam_map = {
            'ct_t1_dit':          'radix8_t1_dit',
            'ct_t1_dif':          'radix8_t1_dif',
            'ct_t1_dit_prefetch': 'radix8_t1_dit_prefetch',
            'ct_t1_dit_log3':     'radix8_tw_log3_dit_kernel',  # different signature! see notes
            'ct_t1_dif_log3':     'radix8_tw_log3_dif_kernel',
        }
        if self.family not in fam_map:
            raise ValueError(f'Unknown family: {self.family}')
        base = fam_map[self.family]
        return {'fwd': f'{base}_fwd_{self.isa}',
                'bwd': f'{base}_bwd_{self.isa}'}


# Sweep — same as R=32 for direct comparability.
# R=8 codelet is usually called as a sub-FFT in a larger plan, so me values
# 64..2048 span realistic call sizes.
SWEEP_ME = [64, 128, 256, 512, 1024, 2048]
SWEEP_IOS_OFFSETS = [0, 8, 64]

def sweep_points() -> List[Tuple[int, int]]:
    return [(me + off, me) for me in SWEEP_ME for off in SWEEP_IOS_OFFSETS]


def enumerate_all() -> List[Candidate]:
    """Three families (t1_dit, t1_dif, t1_dit_prefetch) × two ISAs = 6 candidates.

    The log3 variants (tw_log3_dit_kernel / tw_log3_dif_kernel) are SKIPPED
    because their generated signature is
      (in_re, in_im, out_re, out_im, base_tw_re, base_tw_im, K)
    which differs from the CT-style
      (rio_re, rio_im, W_re, W_im, ios, me)
    used by t1_dit/t1_dif. Adding them would require a second harness code
    path. Deferred until we know they're worth measuring.

    t1_dit_prefetch is included as a NEGATIVE CONTROL: VTune on Raptor Lake
    showed prefetch regresses R=8 performance. We expect it to lose to
    t1_dit everywhere. If it doesn't, the VTune conclusion may not
    generalize across micro-architectures.
    """
    cands = []
    for isa in ('avx2', 'avx512'):
        cands.append(Candidate(family='ct_t1_dit', isa=isa))
        cands.append(Candidate(family='ct_t1_dif', isa=isa))
        cands.append(Candidate(family='ct_t1_dit_prefetch', isa=isa))
    return cands


if __name__ == '__main__':
    cands = enumerate_all()
    print(f'{len(cands)} candidates')
    for c in cands:
        fns = c.function_names()
        print(f'  {c.id():30s} fwd={fns["fwd"]}')
