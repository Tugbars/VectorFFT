# =============================================================================
# DEPRECATED for the inplace family (il_architecture.md 6a11).
# The inplace il_in/il_out codelets are now EMITTED by the generator
# (gen_radix.exe R --ip-il-in / --ip-il-out --bwd), which renders the z
# lattice inside the scheduler's ordering and — critically — handles the
# scalar / masked / sse2 TAIL passes, which this deriver never rewrote
# (derived il_in tails read the uninitialized output buffers for any
# me % VW != 0; latent, no shipped exposure — all production shapes were
# vector multiples). Kept for provenance only. Do not regenerate from here.
# =============================================================================
import sys
sys.path.insert(0, '/home/claude/vfft/VectorFFT-main/src/dag-fft-compiler/tools')
from il_derive import derive
B='/home/claude/vfft/VectorFFT-main/src/dag-fft-compiler/codelets'
SIG_RIO="    double       * __restrict__ rio_re,\n    double       * __restrict__ rio_im,"
for isa in ('avx2','avx512'):
    for R in (2,3,4,5,6,7,8,10,11,12,13,16,17,19,20,25,32,64):
        # fwd: interleaved loads, split stores
        derive(f'{B}/inplace/{isa}/r{R}_n1_fwd.c',
               f'{B}/il/{isa}/r{R}_n1_fwd_ip_il_in_{isa}.c', 'il_in', isa,
               'rio_re','rio_im','in_z',
               f'radix{R}_n1_fwd_{isa}', f'radix{R}_n1_fwd_{isa}_il_in',
               SIG_RIO,
               "    const double * __restrict__ in_z,\n"
               "    double       * __restrict__ out_re,\n"
               "    double       * __restrict__ out_im,")
        t=open(f'{B}/il/{isa}/r{R}_n1_fwd_ip_il_in_{isa}.c').read()
        t=t.replace('rio_re','out_re').replace('rio_im','out_im')
        open(f'{B}/il/{isa}/r{R}_n1_fwd_ip_il_in_{isa}.c','w').write(t)
        # bwd: split loads, interleaved stores
        derive(f'{B}/inplace/{isa}/r{R}_n1_bwd.c',
               f'{B}/il/{isa}/r{R}_n1_bwd_ip_il_out_{isa}.c', 'il_out', isa,
               'rio_re','rio_im','out_z',
               f'radix{R}_n1_bwd_{isa}', f'radix{R}_n1_bwd_{isa}_il_out',
               SIG_RIO,
               "    const double * __restrict__ in_re,\n"
               "    const double * __restrict__ in_im,\n"
               "    double       * __restrict__ out_z,")
        t=open(f'{B}/il/{isa}/r{R}_n1_bwd_ip_il_out_{isa}.c').read()
        t=t.replace('rio_re','in_re').replace('rio_im','in_im')
        open(f'{B}/il/{isa}/r{R}_n1_bwd_ip_il_out_{isa}.c','w').write(t)
    print(isa,'done')
