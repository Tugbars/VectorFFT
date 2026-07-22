import sys
sys.path.insert(0, '/home/claude/vfft/VectorFFT-main/src/dag-fft-compiler/tools')
from il_derive import derive
B='/home/claude/vfft/VectorFFT-main/src/dag-fft-compiler/codelets'
SIG_IN_OLD="    const double * __restrict__ in_re,\n    const double * __restrict__ in_im,"
SIG_OUT_OLD="    double       * __restrict__ out_re,\n    double       * __restrict__ out_im,"
SIG_RIO_OLD="    double       * __restrict__ rio_re,\n    double       * __restrict__ rio_im,"
for isa in ('avx2','avx512'):
    R=64
    oop=f'{B}/oop/{isa}/radix{R}_n1_oop_{isa}.c'
    fno=f'radix{R}_n1_oop_fwd_{isa}_UG_UG'
    # A) axis-0 fwd: IL input
    derive(oop, f'{B}/il/{isa}/radix{R}_n1_oop_il_in_{isa}.c', 'il_in', isa,
           'in_re','in_im','in_z', fno, fno+'_il_in',
           SIG_IN_OLD,
           "    const double * __restrict__ in_z,          /* interleaved pairs */\n"
           "    const double * __restrict__ in_unused,")
    # B) axis-0 bwd via swap trick: fwd math, IL output flipped (im,re)
    derive(oop, f'{B}/il/{isa}/radix{R}_n1_oop_il_out_sw_{isa}.c', 'il_out_flip', isa,
           'out_re','out_im','out_z', fno, fno+'_il_out_sw',
           SIG_OUT_OLD,
           "    double       * __restrict__ out_z,         /* interleaved, (im,re) for bwd-swap */\n"
           "    double       * __restrict__ out_unused,")
    # C) rows fwd: split in -> interleaved out
    sf=f'{B}/strided/{isa}/r{R}_n1_fwd_strided.c'
    fns=f'radix{R}_n1_fwd_{isa}_strided'
    derive(sf, f'{B}/il/{isa}/r{R}_n1_fwd_strided_il_out_{isa}.c', 'il_out', isa,
           'rio_re','rio_im','out_z', fns, fns+'_il_out',
           SIG_RIO_OLD,
           "    const double * __restrict__ in_re,\n"
           "    const double * __restrict__ in_im,\n"
           "    double       * __restrict__ out_z,")
    t=open(f'{B}/il/{isa}/r{R}_n1_fwd_strided_il_out_{isa}.c').read()
    t=t.replace('rio_re','in_re').replace('rio_im','in_im')
    open(f'{B}/il/{isa}/r{R}_n1_fwd_strided_il_out_{isa}.c','w').write(t)
    # D) rows bwd: interleaved in -> split out
    sb=f'{B}/strided/{isa}/r{R}_n1_bwd_strided.c'
    fnb=f'radix{R}_n1_bwd_{isa}_strided'
    derive(sb, f'{B}/il/{isa}/r{R}_n1_bwd_strided_il_in_{isa}.c', 'il_in', isa,
           'rio_re','rio_im','in_z', fnb, fnb+'_il_in',
           SIG_RIO_OLD,
           "    const double * __restrict__ in_z,\n"
           "    double       * __restrict__ out_re,\n"
           "    double       * __restrict__ out_im,")
    t=open(f'{B}/il/{isa}/r{R}_n1_bwd_strided_il_in_{isa}.c').read()
    t=t.replace('rio_re','out_re').replace('rio_im','out_im')
    open(f'{B}/il/{isa}/r{R}_n1_bwd_strided_il_in_{isa}.c','w').write(t)
    print(f'{isa} done')
