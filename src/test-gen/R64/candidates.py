"""R=64 candidate enumeration.

The R=64 generator has no per-variant knobs (no --tile, no --tpf-distance,
no --drain). Each variant is its own per-ISA emission. So the candidate
matrix is just (variant × isa).

Families benched:
  ct_t1_dit           - flat baseline (what VTune profiled)
  ct_t1_dif           - flat DIF variant
  ct_t1_dit_log3      - log3 derived twiddles
  ct_t1_dit_prefetch  - flat with built-in twiddle prefetch
                        VTune: regressed 2.5-6.6% on Raptor Lake AVX2
                        Keep as negative control to measure cross-chip
  ct_t1s_dit          - scalar-broadcast twiddles (NEW, just ported)
                        Expected to help: eliminates twiddle vector loads,
                        reduces twiddle DTLB footprint (~30 pages -> ~1-2)

Not included:
  ct_t1_buf_dit       - not ported to R=64 (would need separate port)
  ct_t1_ladder_dit    - not in R=64 generator; VTune shows port utilization
                        is LOWER at R=64 than R=32 (memory stalls dominate),
                        so register-pressure story doesn't apply
"""
from dataclasses import dataclass
from typing import List, Tuple

R = 64


@dataclass(frozen=True)
class Candidate:
    family: str
    isa: str

    def id(self) -> str:
        return f'{self.family}__{self.isa}'

    def header_name(self) -> str:
        return f'{self.id()}.h'

    def cli_args(self) -> List[str]:
        return ['--isa', self.isa, '--variant', self.family]

    def function_names(self) -> dict:
        """Map candidate family to generated function base names.

        R=64 generator emits one header per (variant, isa) call, with each
        containing the named fwd/bwd functions. We extract those for the
        bench harness.
        """
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
        else:
            raise ValueError(f"Unknown family: {f}")
        return {'fwd': f'{base}_fwd_{self.isa}',
                'bwd': f'{base}_bwd_{self.isa}'}


# Sweep — same as R=32 and R=16 for cross-radix comparability.
# Note: at R=64, twiddle table size = 63 * me * 16 bytes. At me=2048
# that's ~2 MB, well beyond L1/L2. DTLB pressure (per VTune) kicks in
# around me=256 where each output row starts on its own 4KB page.
SWEEP_ME = [64, 128, 256, 512, 1024, 2048]
SWEEP_IOS_OFFSETS = [0, 8, 64]


def sweep_points() -> List[Tuple[int, int]]:
    return [(me + off, me) for me in SWEEP_ME for off in SWEEP_IOS_OFFSETS]


def enumerate_all() -> List[Candidate]:
    """Five families × two ISAs = 10 candidates total.

    This is a much smaller matrix than R=32 (163 candidates) because
    the R=64 generator has no internal knob space. What we lose in
    sweep granularity, we gain in bench speed — 10 candidates × 18
    sweeps × 2 dirs = 360 measurements vs R=32's 5868. Should run in
    ~1-2 minutes total.
    """
    cands = []
    for isa in ('avx2', 'avx512'):
        cands.append(Candidate(family='ct_t1_dit', isa=isa))
        cands.append(Candidate(family='ct_t1_dif', isa=isa))
        cands.append(Candidate(family='ct_t1_dit_log3', isa=isa))
        cands.append(Candidate(family='ct_t1_dit_prefetch', isa=isa))
        cands.append(Candidate(family='ct_t1s_dit', isa=isa))
    return cands


if __name__ == '__main__':
    cands = enumerate_all()
    print(f'{len(cands)} candidates')
    for c in cands:
        fns = c.function_names()
        print(f'  {c.id():40s} fwd={fns["fwd"]}')
