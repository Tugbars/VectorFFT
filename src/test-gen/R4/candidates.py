"""R=4 candidate enumeration.

gen_radix4.py's CLI is: python gen_radix4.py {scalar|avx2|avx512}
It emits ALL variants in a single header for the given ISA.

Each Candidate's cli_args() returns the positional ISA arg. All candidates
for the same ISA will generate the same header content; the wrapper
mechanism (in run_r4_bench.py) strips 'static inline' from only the
specific function we want to export as extern.

Families benched:
  ct_t1_dit           - flat baseline. Wins at small me across chips.
  ct_t1_dit_log3      - derive w2, w3 from w1. 3-deep dep chain.
                        On Raptor Lake, wins at large me by up to 2.6x.
                        On SPR, serializes and loses everywhere.
  ct_t1_dif           - post-twiddle butterfly. ~Same perf as DIT everywhere.

Phase 2 variants to add later (not yet in generator):
  ct_t1_dit_log1      - derive w3 from w1*w2 (2-deep dep chain)
  ct_t1_dit_u2        - unroll m-loop by 2x

Not included (would lose for R=4):
  ct_t1s_dit          - only 3 twiddles, broadcast setup not worth it
  ct_t1_buf_dit       - outbuf drain overhead dominates at R=4
  ct_t1_dit_prefetch  - twiddle table already fits L2
  ct_t1_ladder_dit    - no register pressure at R=4
"""
from dataclasses import dataclass
from typing import List, Tuple

R = 4


@dataclass(frozen=True)
class Candidate:
    family: str
    isa: str

    def id(self) -> str:
        return f'{self.family}__{self.isa}'

    def header_name(self) -> str:
        # All candidates for the same ISA share the same header content.
        # The wrapper C file stamps out only the needed function from a
        # per-candidate header file (which is just a copy of the ISA header).
        return f'{self.id()}.h'

    def cli_args(self) -> List[str]:
        # gen_radix4.py takes a single positional ISA arg.
        return [self.isa]

    def function_names(self) -> dict:
        base = 'radix4_'
        f = self.family
        if f == 'ct_t1_dit':
            base += 't1_dit'
        elif f == 'ct_t1_dit_log3':
            base += 't1_dit_log3'
        elif f == 'ct_t1_dit_log1':
            base += 't1_dit_log1'
        elif f == 'ct_t1_dit_u2':
            base += 't1_dit_u2'
        elif f == 'ct_t1_dif':
            base += 't1_dif'
        else:
            raise ValueError(f"Unknown family: {f}")
        return {'fwd': f'{base}_fwd_{self.isa}',
                'bwd': f'{base}_bwd_{self.isa}'}


# Regular sweep (VTune's small-me regime + small ios padding).
# Captures small-N plans and late-stage inner codelets of big-N plans.
SWEEP_ME = [64, 128, 256, 512, 1024, 2048]
SWEEP_IOS_OFFSETS = [0, 8, 64]

# Wisdom-motivated probe points: R=4 as stage 0/1/2 of big-N plans.
# These (me, ios) pairs don't fit the regular sweep-grid pattern.
# At these points, ios is much larger than me (R=4's 4 inputs span distant
# pages). Probe reveals large-ios behavior invisible in the regular sweep.
#
# Derived from wisdom-file plans where R=4 is non-last-stage:
#   N=131072 K=4   stage 2:  me=2048, ios=2048   (but ios_offsets=0 case)
#   N=131072 K=4   stage 1:  me=8192, ios=8192   (too big for us, use 2048 proxy)
#   N=131072 K=4   stage 0:  me=32768, ios=32768 (same - proxy)
#   N=131072 K=256 stage 1:  me=512, ios=512     (covered by regular sweep)
# And big-me/ios probes for N=1M-scale plans:
LARGE_IOS_PROBE = [
    (256, 2048),    # R=4 inner stage of N≈16384 plan
    (256, 8192),    # probe — intermediate ios
    (512, 8192),    # medium-N stage 1 proxy
    (1024, 16384),  # big-N stage 1 proxy (log3 wins 2.6x on RL here)
    (2048, 8192),   # medium-N K=4 stage 1 proxy
    (2048, 32768),  # big-N K=4 stage 1 proxy (extreme DTLB on server chips)
]


def sweep_points() -> List[Tuple[int, int]]:
    pts = [(me + off, me) for me in SWEEP_ME for off in SWEEP_IOS_OFFSETS]
    pts += [(ios, me) for (me, ios) in LARGE_IOS_PROBE]  # (ios, me) tuple format
    return pts


def enumerate_all() -> List[Candidate]:
    """Five families × one ISA = 5 candidates (AVX2 only for now).

    AVX-512 disabled: gen_radix4.py's n1_ovs path has a bug (uses
    _mm256_unpacklo_pd on __m512d types). n1_ovs isn't in our bench path,
    but the compile unit fails to build. Fix the generator separately.

    Phase 2 variants (log1, u2) are now included.
    """
    cands = []
    for isa in ('avx2',):
        cands.append(Candidate(family='ct_t1_dit', isa=isa))
        cands.append(Candidate(family='ct_t1_dit_log3', isa=isa))
        cands.append(Candidate(family='ct_t1_dit_log1', isa=isa))
        cands.append(Candidate(family='ct_t1_dit_u2', isa=isa))
        cands.append(Candidate(family='ct_t1_dif', isa=isa))
    return cands


if __name__ == '__main__':
    cands = enumerate_all()
    print(f'{len(cands)} candidates')
    for c in cands:
        fns = c.function_names()
        print(f'  {c.id():40s} fwd={fns["fwd"]}')
    pts = sweep_points()
    print(f'\n{len(pts)} sweep points:')
    for ios, me in pts:
        print(f'  ios={ios:>6} me={me:>6}')
