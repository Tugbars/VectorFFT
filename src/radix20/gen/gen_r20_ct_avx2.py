#!/usr/bin/env python3
"""
gen_r20_ct_avx2.py — DFT-20 zero-spill CT codelet generator

4×5 Cooley-Tukey decomposition designed for 16 YMM registers:
  Pass 1: 5× DFT-4 (peak 12 YMM) → explicit L1 spill buffer
  Twiddle: 12 W₂₀ internal twiddles (peak 6 YMM per entry)
  Pass 2: 4× DFT-5 (peak 12 YMM, constants from memory not regs)

Zero compiler spills. All temporaries fit in 16 YMM at every point.
Explicit spill buffer = sequential L1 access (predictable, unlike compiler spills).

Usage: python3 gen_r20_ct_avx2.py <avx2|avx512>
"""
import math, sys

R, N1, N2 = 20, 4, 5  # 4×5: Pass1=5×DFT-4, Pass2=4×DFT-5

# W20 internal twiddles: W20^(k1*n2) for k1=1..3, n2=1..4
def w20(e):
    e = e % 20
    a = 2.0 * math.pi * e / 20.0
    return (math.cos(a), -math.sin(a))  # VectorFFT convention: -sin

# Collect all non-trivial internal twiddle exponents
def internal_twiddles():
    tw = {}
    for k1 in range(1, N1):  # k1=1,2,3
        for n2 in range(1, N2):  # n2=1,2,3,4
            e = (k1 * n2) % R
            if e != 0 and e not in tw:
                tw[e] = w20(e)
    return tw

ITW = internal_twiddles()
# Exponents: {1,2,3,4,6,8,9,12} — 8 unique twiddles (not 12, some share)

# DFT-5 Rader constants
C_A = "0.55901699437494742"   # (sqrt(5)-1)/4
C_B = "0.95105651629515357"   # sin(2π/5)
C_C = "0.58778525229247313"   # sin(π/5)  
C_D = "0.25"

class Emitter:
    def __init__(self):
        self.L = []
        self.ind = 0
    def o(self, s=''):
        self.L.append('    ' * self.ind + s)
    def c(self, s):
        self.o(f'/* {s} */')
    def b(self):
        self.L.append('')


def isa_config(isa):
    if isa == 'avx512':
        return dict(T='__m512d', C=8, P='_mm512', VL=8, align=64,
                    attr='__attribute__((target("avx512f,fma")))')
    else:
        return dict(T='__m256d', C=4, P='_mm256', VL=4, align=32,
                    attr='__attribute__((target("avx2,fma")))')


def emit_dft4_pass(em, cfg, n2, has_tw, tw_mode):
    """Emit one DFT-4 sub-FFT for column n2.
    Loads inputs m = n2, n2+5, n2+10, n2+15.
    If has_tw: applies external twiddles before DFT-4.
    Stores results to spill buffer sp_re/im[k1*5 + n2].
    Peak: 12 YMM (8 data + 4 twiddle temps)."""
    T, P, VL = cfg['T'], cfg['P'], cfg['VL']
    inputs = [n2, n2 + 5, n2 + 10, n2 + 15]

    em.c(f'DFT-4 sub-FFT n2={n2} — inputs [{",".join(str(i) for i in inputs)}]')
    em.o(f'{{')  # open block scope

    # Load 4 complex inputs (8 YMM)
    for j, m in enumerate(inputs):
        em.o(f'{T} r{j} = {P}_loadu_pd(&in_re[{m}*K+k]);')
        em.o(f'{T} i{j} = {P}_loadu_pd(&in_im[{m}*K+k]);')

    # Apply external twiddles to inputs 1,2,3 (m>0 always for those)
    if has_tw:
        for j in range(1, 4):
            m = inputs[j]
            if m == 0:
                continue
            if tw_mode == 'contig':
                em.o(f'{{ {T} wr = {P}_load_pd(&tr[{m-1}*{VL}]);')
                em.o(f'  {T} wi = {P}_load_pd(&ti[{m-1}*{VL}]);')
            else:  # strided
                em.o(f'{{ {T} wr = {P}_loadu_pd(&tw_re[{m-1}*K+k]);')
                em.o(f'  {T} wi = {P}_loadu_pd(&tw_im[{m-1}*K+k]);')
            em.o(f'  {T} t = r{j};')
            em.o(f'  r{j} = {P}_fmsub_pd(t, wr, {P}_mul_pd(i{j}, wi));')
            em.o(f'  i{j} = {P}_fmadd_pd(t, wi, {P}_mul_pd(i{j}, wr)); }}')
    # Also twiddle input 0 if m>0
    if has_tw and inputs[0] > 0:
        m = inputs[0]
        if tw_mode == 'contig':
            em.o(f'{{ {T} wr = {P}_load_pd(&tr[{m-1}*{VL}]);')
            em.o(f'  {T} wi = {P}_load_pd(&ti[{m-1}*{VL}]);')
        else:
            em.o(f'{{ {T} wr = {P}_loadu_pd(&tw_re[{m-1}*K+k]);')
            em.o(f'  {T} wi = {P}_loadu_pd(&tw_im[{m-1}*K+k]);')
        em.o(f'  {T} t = r0;')
        em.o(f'  r0 = {P}_fmsub_pd(t, wr, {P}_mul_pd(i0, wi));')
        em.o(f'  i0 = {P}_fmadd_pd(t, wi, {P}_mul_pd(i0, wr)); }}')

    # DFT-4 butterfly (in-place on r0..r3, i0..i3)
    # s = r0+r2, d = r0-r2, t = r1+r3, u = r1-r3 (forward: j rotation on u)
    em.o(f'{{ {T} sr = {P}_add_pd(r0, r2), si = {P}_add_pd(i0, i2);')
    em.o(f'  {T} dr = {P}_sub_pd(r0, r2), di = {P}_sub_pd(i0, i2);')
    em.o(f'  {T} tr = {P}_add_pd(r1, r3), ti = {P}_add_pd(i1, i3);')
    em.o(f'  {T} ur = {P}_sub_pd(r1, r3), ui = {P}_sub_pd(i1, i3);')
    # out[0] = s+t, out[1] = d+j*u, out[2] = s-t, out[3] = d-j*u
    # j*u = (-ui, ur) in forward
    for k1, (rexpr, iexpr) in enumerate([
        (f'{P}_add_pd(sr, tr)', f'{P}_add_pd(si, ti)'),
        (f'{P}_add_pd(dr, ui)', f'{P}_sub_pd(di, ur)'),   # d + j*u
        (f'{P}_sub_pd(sr, tr)', f'{P}_sub_pd(si, ti)'),
        (f'{P}_sub_pd(dr, ui)', f'{P}_add_pd(di, ur)'),   # d - j*u
    ]):
        slot = k1 * 5 + n2
        em.o(f'  {P}_store_pd(&sp_re[{slot}*{VL}], {rexpr});')
        em.o(f'  {P}_store_pd(&sp_im[{slot}*{VL}], {iexpr});')
    em.o(f'}}')
    em.o(f'}}')  # close block scope
    em.b()


def emit_internal_twiddle_pass(em, cfg):
    """Apply internal W₂₀ twiddles to spill buffer entries.
    k1=0 row: no twiddle. k1=1,2,3 × n2=1,2,3,4: 12 entries.
    Peak per entry: 6 YMM (2 data + 2 const + 2 temp). Zero spill."""
    T, P, VL = cfg['T'], cfg['P'], cfg['VL']

    em.c('Internal W₂₀ twiddle apply (12 entries, peak 6 YMM each)')
    for k1 in range(1, N1):
        for n2 in range(1, N2):
            e = (k1 * n2) % R
            if e == 0:
                continue
            wr, wi = w20(e)
            slot = k1 * 5 + n2
            em.o(f'{{ {T} dr = {P}_load_pd(&sp_re[{slot}*{VL}]);')
            em.o(f'  {T} di = {P}_load_pd(&sp_im[{slot}*{VL}]);')
            em.o(f'  {P}_store_pd(&sp_re[{slot}*{VL}], {P}_fmsub_pd(dr, {P}_set1_pd({wr:.20e}), {P}_mul_pd(di, {P}_set1_pd({wi:.20e}))));')
            em.o(f'  {P}_store_pd(&sp_im[{slot}*{VL}], {P}_fmadd_pd(dr, {P}_set1_pd({wi:.20e}), {P}_mul_pd(di, {P}_set1_pd({wr:.20e})))); }}')
    em.b()


def emit_dft5_pass(em, cfg, k1, out_target, out_stride):
    """Emit one DFT-5 column butterfly for row k1.
    Loads from spill buffer sp[k1*5 + n2] for n2=0..4.
    Stores to output at indices k1+4*k2 for k2=0..4.
    Peak: 12 YMM (10 data + 2 temps, constants from memory).
    
    DFT-5 Rader butterfly (forward):
      s1=x1+x4, s2=x2+x3, d1=x1-x4, d2=x2-x3
      y0 = x0 + s1 + s2
      t0 = x0 - 0.25*(s1+s2)
      t1 = cA*(s1-s2)
      p1 = t0+t1, p2 = t0-t1
      U = cB*d1 + cC*d2   (cross-term for imaginary rotation)
      V = cB*d2 - cC*d1
      y1 = p1_re + U_im, p1_im - U_re  (i.e. p1 - j*U in forward)
      y4 = p1_re - U_im, p1_im + U_re  (i.e. p1 + j*U)
      y2 = p2_re - V_im, p2_im + V_re  (i.e. p2 + j*V)
      y3 = p2_re + V_im, p2_im - V_re  (i.e. p2 - j*V)
    """
    T, P, VL = cfg['T'], cfg['P'], cfg['VL']

    em.c(f'DFT-5 column k1={k1} — outputs [{k1},{k1+4},{k1+8},{k1+12},{k1+16}]')
    em.o(f'{{')  # block scope

    # Load 5 complex from spill buffer (10 YMM)
    for n2 in range(N2):
        slot = k1 * 5 + n2
        em.o(f'{T} x{n2}r = {P}_load_pd(&sp_re[{slot}*{VL}]);')
        em.o(f'{T} x{n2}i = {P}_load_pd(&sp_im[{slot}*{VL}]);')

    # DFT-5 butterfly — constants from memory (not registers)
    # Each _mm256_set1_pd compiles to vbroadcastsd from .rodata, ~4 cycle L1 hit
    em.o(f'{{ {T} s1r={P}_add_pd(x1r,x4r), s1i={P}_add_pd(x1i,x4i);')
    em.o(f'  {T} s2r={P}_add_pd(x2r,x3r), s2i={P}_add_pd(x2i,x3i);')
    em.o(f'  {T} d1r={P}_sub_pd(x1r,x4r), d1i={P}_sub_pd(x1i,x4i);')
    em.o(f'  {T} d2r={P}_sub_pd(x2r,x3r), d2i={P}_sub_pd(x2i,x3i);')
    # FREE x1,x2,x3,x4 → 2(x0) + 8(s1,s2,d1,d2) = 10 live

    em.o(f'  {T} ssr={P}_add_pd(s1r,s2r), ssi={P}_add_pd(s1i,s2i);')
    # y0 = x0 + ss → store immediately to free x0
    out_idx_0 = k1 + N1 * 0
    em.o(f'  {P}_storeu_pd(&{out_target}_re[{out_idx_0}*{out_stride}+k], {P}_add_pd(x0r, ssr));')
    em.o(f'  {P}_storeu_pd(&{out_target}_im[{out_idx_0}*{out_stride}+k], {P}_add_pd(x0i, ssi));')

    # t0 = x0 - 0.25*ss → FREE x0, ss
    em.o(f'  {T} t0r={P}_fnmadd_pd({P}_set1_pd({C_D}), ssr, x0r);')
    em.o(f'  {T} t0i={P}_fnmadd_pd({P}_set1_pd({C_D}), ssi, x0i);')
    # 8 live: t0(2), s1(2), s2(2), d1(2), d2(2) → wait, that's 10
    # But ss is dead, x0 is dead → s1,s2 still needed for ds

    # ds = s1-s2, t1 = cA*ds → FREE s1, s2
    em.o(f'  {T} t1r={P}_mul_pd({P}_set1_pd({C_A}), {P}_sub_pd(s1r,s2r));')
    em.o(f'  {T} t1i={P}_mul_pd({P}_set1_pd({C_A}), {P}_sub_pd(s1i,s2i));')
    # 8 live: t0(2), t1(2), d1(2), d2(2)

    # p1 = t0+t1, p2 = t0-t1 → FREE t0, t1
    em.o(f'  {T} p1r={P}_add_pd(t0r,t1r), p1i={P}_add_pd(t0i,t1i);')
    em.o(f'  {T} p2r={P}_sub_pd(t0r,t1r), p2i={P}_sub_pd(t0i,t1i);')
    # 8 live: p1(2), p2(2), d1(2), d2(2)

    # U = cB*d1 + cC*d2 → 2 YMM
    em.o(f'  {T} Ur={P}_fmadd_pd({P}_set1_pd({C_C}), d2r, {P}_mul_pd({P}_set1_pd({C_B}), d1r));')
    em.o(f'  {T} Ui={P}_fmadd_pd({P}_set1_pd({C_C}), d2i, {P}_mul_pd({P}_set1_pd({C_B}), d1i));')

    # V = cB*d2 - cC*d1 → 2 YMM, FREE d1, d2
    em.o(f'  {T} Vr={P}_fnmadd_pd({P}_set1_pd({C_C}), d1r, {P}_mul_pd({P}_set1_pd({C_B}), d2r));')
    em.o(f'  {T} Vi={P}_fnmadd_pd({P}_set1_pd({C_C}), d1i, {P}_mul_pd({P}_set1_pd({C_B}), d2i));')
    # 8 live: p1(2), p2(2), U(2), V(2)

    # y1 = p1 - j*U → (p1r + Ui, p1i - Ur)  [forward]
    out_idx_1 = k1 + N1 * 1
    em.o(f'  {P}_storeu_pd(&{out_target}_re[{out_idx_1}*{out_stride}+k], {P}_add_pd(p1r, Ui));')
    em.o(f'  {P}_storeu_pd(&{out_target}_im[{out_idx_1}*{out_stride}+k], {P}_sub_pd(p1i, Ur));')

    # y4 = p1 + j*U → (p1r - Ui, p1i + Ur)
    out_idx_4 = k1 + N1 * 4
    em.o(f'  {P}_storeu_pd(&{out_target}_re[{out_idx_4}*{out_stride}+k], {P}_sub_pd(p1r, Ui));')
    em.o(f'  {P}_storeu_pd(&{out_target}_im[{out_idx_4}*{out_stride}+k], {P}_add_pd(p1i, Ur));')

    # y2 = p2 + j*V → (p2r - Vi, p2i + Vr)
    out_idx_2 = k1 + N1 * 2
    em.o(f'  {P}_storeu_pd(&{out_target}_re[{out_idx_2}*{out_stride}+k], {P}_sub_pd(p2r, Vi));')
    em.o(f'  {P}_storeu_pd(&{out_target}_im[{out_idx_2}*{out_stride}+k], {P}_add_pd(p2i, Vr));')

    # y3 = p2 - j*V → (p2r + Vi, p2i - Vr)
    out_idx_3 = k1 + N1 * 3
    em.o(f'  {P}_storeu_pd(&{out_target}_re[{out_idx_3}*{out_stride}+k], {P}_add_pd(p2r, Vi));')
    em.o(f'  {P}_storeu_pd(&{out_target}_im[{out_idx_3}*{out_stride}+k], {P}_sub_pd(p2i, Vr));')

    em.o(f'}}')
    em.o(f'}}')  # close DFT-5 block scope
    em.b()


def gen_notw(cfg):
    """NOTW CT kernel — zero external twiddles."""
    T, P, C, VL = cfg['T'], cfg['P'], cfg['C'], cfg['VL']
    isa = 'avx2' if C == 4 else 'avx512'

    em = Emitter()
    em.c(f'DFT-20 NOTW (N1) — zero-spill 4×5 CT, {isa.upper()}')
    em.c(f'Pass 1: 5× DFT-4 (peak 12 YMM) → L1 spill buffer')
    em.c(f'Pass 2: 4× DFT-5 (peak 12 YMM, constants from memory)')
    em.c(f'Internal W₂₀ twiddles between passes (peak 6 YMM)')
    em.b()
    em.o(cfg['attr'])
    em.o(f'static void radix20_ct_n1_fwd_{isa}(')
    em.o(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
    em.o(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    em.o(f'    size_t K)')
    em.o(f'{{')
    em.ind = 1
    em.o(f'__attribute__((aligned({cfg["align"]}))) double sp_re[{R * VL}];')
    em.o(f'__attribute__((aligned({cfg["align"]}))) double sp_im[{R * VL}];')
    em.b()
    em.o(f'for (size_t k = 0; k < K; k += {C}) {{')
    em.ind += 1

    # Pass 1: 5× DFT-4
    em.c('═══ Pass 1: 5× DFT-4 → spill buffer ═══')
    for n2 in range(N2):
        emit_dft4_pass(em, cfg, n2, has_tw=False, tw_mode=None)

    # Internal twiddles
    em.c('═══ Internal W₂₀ twiddles ═══')
    emit_internal_twiddle_pass(em, cfg)

    # Pass 2: 4× DFT-5
    em.c('═══ Pass 2: 4× DFT-5 → output ═══')
    for k1 in range(N1):
        emit_dft5_pass(em, cfg, k1, 'out', 'K')

    em.ind -= 1
    em.o('}')
    em.ind -= 1
    em.o('}')
    em.b()
    # Backward
    em.o(cfg['attr'])
    em.o(f'static inline void radix20_ct_n1_bwd_{isa}(')
    em.o(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
    em.o(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    em.o(f'    size_t K)')
    em.o(f'{{')
    em.ind = 1
    em.o(f'radix20_ct_n1_fwd_{isa}(in_im, in_re, out_im, out_re, K);')
    em.ind -= 1
    em.o('}')
    em.b()
    return em.L


def gen_tw_contig(cfg):
    """TW DIT with contiguous twiddles — zero-spill CT."""
    T, P, C, VL = cfg['T'], cfg['P'], cfg['C'], cfg['VL']
    isa = 'avx2' if C == 4 else 'avx512'

    em = Emitter()
    em.c(f'DFT-20 TW DIT (contiguous) — zero-spill 4×5 CT, {isa.upper()}')
    em.c(f'Twiddle layout: VFFT_TW_CONTIG — 19 entries packed per k-group of {VL}')
    em.b()
    em.o(cfg['attr'])
    em.o(f'static void radix20_ct_tw_contig_fwd_{isa}(')
    em.o(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
    em.o(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    em.o(f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,')
    em.o(f'    size_t K)')
    em.o(f'{{')
    em.ind = 1
    em.o(f'__attribute__((aligned({cfg["align"]}))) double sp_re[{R * VL}];')
    em.o(f'__attribute__((aligned({cfg["align"]}))) double sp_im[{R * VL}];')
    em.b()
    em.o(f'for (size_t k = 0; k < K; k += {C}) {{')
    em.ind += 1
    em.o(f'const size_t kg = k / {VL};')
    em.o(f'const double *tr = &tw_re[kg * 19 * {VL}];')
    em.o(f'const double *ti = &tw_im[kg * 19 * {VL}];')
    em.b()

    em.c('═══ Pass 1: 5× DFT-4 with external twiddles → spill buffer ═══')
    for n2 in range(N2):
        emit_dft4_pass(em, cfg, n2, has_tw=True, tw_mode='contig')

    em.c('═══ Internal W₂₀ twiddles ═══')
    emit_internal_twiddle_pass(em, cfg)

    em.c('═══ Pass 2: 4× DFT-5 → output ═══')
    for k1 in range(N1):
        emit_dft5_pass(em, cfg, k1, 'out', 'K')

    em.ind -= 1
    em.o('}')
    em.ind -= 1
    em.o('}')
    em.b()
    em.o(cfg['attr'])
    em.o(f'static inline void radix20_ct_tw_contig_bwd_{isa}(')
    em.o(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
    em.o(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    em.o(f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,')
    em.o(f'    size_t K)')
    em.o(f'{{')
    em.ind = 1
    em.o(f'radix20_ct_tw_contig_fwd_{isa}(in_im, in_re, out_im, out_re, tw_re, tw_im, K);')
    em.ind -= 1
    em.o('}')
    em.b()
    return em.L


def gen_tw_strided(cfg):
    """TW DIT with strided twiddles — zero-spill CT (for ladder/walk integration)."""
    T, P, C, VL = cfg['T'], cfg['P'], cfg['C'], cfg['VL']
    isa = 'avx2' if C == 4 else 'avx512'

    em = Emitter()
    em.c(f'DFT-20 TW DIT (strided) — zero-spill 4×5 CT, {isa.upper()}')
    em.b()
    em.o(cfg['attr'])
    em.o(f'static void radix20_ct_tw_strided_fwd_{isa}(')
    em.o(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
    em.o(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    em.o(f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,')
    em.o(f'    size_t K)')
    em.o(f'{{')
    em.ind = 1
    em.o(f'__attribute__((aligned({cfg["align"]}))) double sp_re[{R * VL}];')
    em.o(f'__attribute__((aligned({cfg["align"]}))) double sp_im[{R * VL}];')
    em.b()
    em.o(f'for (size_t k = 0; k < K; k += {C}) {{')
    em.ind += 1

    em.c('═══ Pass 1: 5× DFT-4 with strided external twiddles ═══')
    for n2 in range(N2):
        emit_dft4_pass(em, cfg, n2, has_tw=True, tw_mode='strided')

    em.c('═══ Internal W₂₀ twiddles ═══')
    emit_internal_twiddle_pass(em, cfg)

    em.c('═══ Pass 2: 4× DFT-5 → output ═══')
    for k1 in range(N1):
        emit_dft5_pass(em, cfg, k1, 'out', 'K')

    em.ind -= 1
    em.o('}')
    em.ind -= 1
    em.o('}')
    em.b()
    em.o(cfg['attr'])
    em.o(f'static inline void radix20_ct_tw_strided_bwd_{isa}(')
    em.o(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
    em.o(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    em.o(f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,')
    em.o(f'    size_t K)')
    em.o(f'{{')
    em.ind = 1
    em.o(f'radix20_ct_tw_strided_fwd_{isa}(in_im, in_re, out_im, out_re, tw_re, tw_im, K);')
    em.ind -= 1
    em.o('}')
    em.b()
    return em.L


def emit_dft5_pass_dif(em, cfg, k1, tw_mode):
    """DIF DFT-5: butterfly → external twiddle on output → store.
    Same butterfly as DIT, but twiddle goes on each output m>0."""
    T, P, VL = cfg['T'], cfg['P'], cfg['VL']

    em.c(f'DFT-5 DIF column k1={k1}')
    em.o(f'{{')

    for n2 in range(N2):
        slot = k1 * 5 + n2
        em.o(f'{T} x{n2}r = {P}_load_pd(&sp_re[{slot}*{VL}]);')
        em.o(f'{T} x{n2}i = {P}_load_pd(&sp_im[{slot}*{VL}]);')

    em.o(f'{{ {T} s1r={P}_add_pd(x1r,x4r), s1i={P}_add_pd(x1i,x4i);')
    em.o(f'  {T} s2r={P}_add_pd(x2r,x3r), s2i={P}_add_pd(x2i,x3i);')
    em.o(f'  {T} d1r={P}_sub_pd(x1r,x4r), d1i={P}_sub_pd(x1i,x4i);')
    em.o(f'  {T} d2r={P}_sub_pd(x2r,x3r), d2i={P}_sub_pd(x2i,x3i);')
    em.o(f'  {T} ssr={P}_add_pd(s1r,s2r), ssi={P}_add_pd(s1i,s2i);')
    em.o(f'  {T} t0r={P}_fnmadd_pd({P}_set1_pd({C_D}), ssr, x0r);')
    em.o(f'  {T} t0i={P}_fnmadd_pd({P}_set1_pd({C_D}), ssi, x0i);')
    em.o(f'  {T} t1r={P}_mul_pd({P}_set1_pd({C_A}), {P}_sub_pd(s1r,s2r));')
    em.o(f'  {T} t1i={P}_mul_pd({P}_set1_pd({C_A}), {P}_sub_pd(s1i,s2i));')
    em.o(f'  {T} p1r={P}_add_pd(t0r,t1r), p1i={P}_add_pd(t0i,t1i);')
    em.o(f'  {T} p2r={P}_sub_pd(t0r,t1r), p2i={P}_sub_pd(t0i,t1i);')
    em.o(f'  {T} Ur={P}_fmadd_pd({P}_set1_pd({C_C}), d2r, {P}_mul_pd({P}_set1_pd({C_B}), d1r));')
    em.o(f'  {T} Ui={P}_fmadd_pd({P}_set1_pd({C_C}), d2i, {P}_mul_pd({P}_set1_pd({C_B}), d1i));')
    em.o(f'  {T} Vr={P}_fnmadd_pd({P}_set1_pd({C_C}), d1r, {P}_mul_pd({P}_set1_pd({C_B}), d2r));')
    em.o(f'  {T} Vi={P}_fnmadd_pd({P}_set1_pd({C_C}), d1i, {P}_mul_pd({P}_set1_pd({C_B}), d2i));')

    # DIF: compute butterfly output, apply external twiddle, store
    outputs = [
        (0, f'{P}_add_pd(x0r, ssr)', f'{P}_add_pd(x0i, ssi)'),
        (1, f'{P}_add_pd(p1r, Ui)',  f'{P}_sub_pd(p1i, Ur)'),
        (2, f'{P}_sub_pd(p2r, Vi)',  f'{P}_add_pd(p2i, Vr)'),
        (3, f'{P}_add_pd(p2r, Vi)',  f'{P}_sub_pd(p2i, Vr)'),
        (4, f'{P}_sub_pd(p1r, Ui)',  f'{P}_add_pd(p1i, Ur)'),
    ]

    for k2, rexpr, iexpr in outputs:
        m = k1 + N1 * k2
        em.o(f'  {{ {T} yr={rexpr}, yi={iexpr};')
        if m > 0:
            if tw_mode == 'contig':
                em.o(f'    {T} wr={P}_load_pd(&tr[{m-1}*{VL}]), wi={P}_load_pd(&ti[{m-1}*{VL}]);')
            else:
                em.o(f'    {T} wr={P}_loadu_pd(&tw_re[{m-1}*K+k]), wi={P}_loadu_pd(&tw_im[{m-1}*K+k]);')
            em.o(f'    {T} t=yr;')
            em.o(f'    yr={P}_fmsub_pd(t,wr,{P}_mul_pd(yi,wi));')
            em.o(f'    yi={P}_fmadd_pd(t,wi,{P}_mul_pd(yi,wr));')
        em.o(f'    {P}_storeu_pd(&out_re[{m}*K+k], yr);')
        em.o(f'    {P}_storeu_pd(&out_im[{m}*K+k], yi); }}')

    em.o(f'}}')
    em.o(f'}}')
    em.b()


def gen_dif_contig(cfg):
    """DIF with contiguous twiddles."""
    T, P, C, VL = cfg['T'], cfg['P'], cfg['C'], cfg['VL']
    isa = 'avx2' if C == 4 else 'avx512'

    em = Emitter()
    em.c(f'DFT-20 TW DIF (contiguous) — zero-spill 4×5 CT, {isa.upper()}')
    em.b()
    em.o(cfg['attr'])
    em.o(f'static void radix20_ct_dif_contig_fwd_{isa}(')
    em.o(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
    em.o(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    em.o(f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,')
    em.o(f'    size_t K)')
    em.o(f'{{')
    em.ind = 1
    em.o(f'__attribute__((aligned({cfg["align"]}))) double sp_re[{R * VL}];')
    em.o(f'__attribute__((aligned({cfg["align"]}))) double sp_im[{R * VL}];')
    em.b()
    em.o(f'for (size_t k = 0; k < K; k += {C}) {{')
    em.ind += 1
    em.o(f'const size_t kg = k / {VL};')
    em.o(f'const double *tr = &tw_re[kg * 19 * {VL}];')
    em.o(f'const double *ti = &tw_im[kg * 19 * {VL}];')
    em.b()

    em.c('Pass 1: 5× DFT-4 — NO external twiddles (DIF)')
    for n2 in range(N2):
        emit_dft4_pass(em, cfg, n2, has_tw=False, tw_mode=None)

    em.c('Internal W₂₀ twiddles')
    emit_internal_twiddle_pass(em, cfg)

    em.c('Pass 2: 4× DFT-5 → external twiddle on output → store (DIF)')
    for k1 in range(N1):
        emit_dft5_pass_dif(em, cfg, k1, 'contig')

    em.ind -= 1
    em.o('}')
    em.ind -= 1
    em.o('}')
    em.b()
    em.o(cfg['attr'])
    em.o(f'static inline void radix20_ct_dif_contig_bwd_{isa}(')
    em.o(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
    em.o(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    em.o(f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,')
    em.o(f'    size_t K)')
    em.o(f'{{')
    em.ind = 1
    em.o(f'radix20_ct_dif_contig_fwd_{isa}(in_im, in_re, out_im, out_re, tw_re, tw_im, K);')
    em.ind -= 1
    em.o('}')
    em.b()
    return em.L


def gen_dif_strided(cfg):
    """DIF with strided twiddles."""
    T, P, C, VL = cfg['T'], cfg['P'], cfg['C'], cfg['VL']
    isa = 'avx2' if C == 4 else 'avx512'

    em = Emitter()
    em.c(f'DFT-20 TW DIF (strided) — zero-spill 4×5 CT, {isa.upper()}')
    em.b()
    em.o(cfg['attr'])
    em.o(f'static void radix20_ct_dif_strided_fwd_{isa}(')
    em.o(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
    em.o(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    em.o(f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,')
    em.o(f'    size_t K)')
    em.o(f'{{')
    em.ind = 1
    em.o(f'__attribute__((aligned({cfg["align"]}))) double sp_re[{R * VL}];')
    em.o(f'__attribute__((aligned({cfg["align"]}))) double sp_im[{R * VL}];')
    em.b()
    em.o(f'for (size_t k = 0; k < K; k += {C}) {{')
    em.ind += 1

    em.c('Pass 1: 5× DFT-4 — NO external twiddles (DIF)')
    for n2 in range(N2):
        emit_dft4_pass(em, cfg, n2, has_tw=False, tw_mode=None)

    em.c('Internal W₂₀ twiddles')
    emit_internal_twiddle_pass(em, cfg)

    em.c('Pass 2: 4× DFT-5 → strided twiddle on output (DIF)')
    for k1 in range(N1):
        emit_dft5_pass_dif(em, cfg, k1, 'strided')

    em.ind -= 1
    em.o('}')
    em.ind -= 1
    em.o('}')
    em.b()
    em.o(cfg['attr'])
    em.o(f'static inline void radix20_ct_dif_strided_bwd_{isa}(')
    em.o(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
    em.o(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    em.o(f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,')
    em.o(f'    size_t K)')
    em.o(f'{{')
    em.ind = 1
    em.o(f'radix20_ct_dif_strided_fwd_{isa}(in_im, in_re, out_im, out_re, tw_re, tw_im, K);')
    em.ind -= 1
    em.o('}')
    em.b()
    return em.L


def gen_file(isa):
    cfg = isa_config(isa)
    ISA = isa.upper()
    guard = f'FFT_RADIX20_{ISA}_CT_H'

    L = [f'/**',
         f' * @file fft_radix20_{isa}_ct.h',
         f' * @brief DFT-20 {ISA} zero-spill 4×5 CT codelets (N1+DIT+DIF)',
         f' * Each pass fits 16 YMM registers. Explicit L1 spill buffer.',
         f' * Generated by gen_r20_ct_avx2.py',
         f' */', '',
         f'#ifndef {guard}', f'#define {guard}',
         f'#include <stddef.h>', f'#include <immintrin.h>', '']

    L.extend(gen_notw(cfg))
    L.extend(gen_tw_contig(cfg))
    L.extend(gen_tw_strided(cfg))
    L.extend(gen_dif_contig(cfg))
    L.extend(gen_dif_strided(cfg))

    L.append(f'#endif /* {guard} */')
    return L


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in ('avx2', 'avx512'):
        print("Usage: gen_r20_ct_avx2.py <avx2|avx512>", file=sys.stderr)
        sys.exit(1)
    isa = sys.argv[1]
    lines = gen_file(isa)
    print('\n'.join(lines))
    print(f"/* {len(lines)} lines */", file=sys.stderr)
