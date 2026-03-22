#!/usr/bin/env python3
"""
gen_r20_ct_avx512.py — DFT-20 AVX-512 zero-spill CT with ILP pairing.

4×5 Cooley-Tukey, 32 ZMM registers.
ILP strategy: pair two independent sub-FFTs to fill both FMA ports.
  Pass 1: [DFT-4 n2=0 + n2=1] [DFT-4 n2=2 + n2=3] [DFT-4 n2=4 alone]
  Pass 2: [DFT-5 k1=0 + k1=1] [DFT-5 k1=2 + k1=3]

Register budget per pair:
  DFT-4 pair: 2×12 = 24 ZMM, 8 free
  DFT-5 pair: 2×12 = 24 ZMM, 8 free

Internal W₂₀: static arrays with {1to8} broadcast, XOR negation for ×j/×(-1)/×(-j).

Usage: python3 gen_r20_ct_avx512.py
"""
import math

R, N1, N2 = 20, 4, 5  # 4×5: Pass1=5×DFT-4, Pass2=4×DFT-5

T = '__m512d'
C = 8  # doubles per ZMM
ATTR = '__attribute__((target("avx512f,avx512dq,fma")))'


# ════════════════════════════════════════
# Helpers
# ════════════════════════════════════════

def w20(e):
    e = e % 20
    a = 2.0 * math.pi * e / 20.0
    return (math.cos(a), -math.sin(a))

def neg(v):
    """XOR sign-bit negation — 1 cycle vs 4 for vsubpd."""
    return f'_mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512({v}), sign_mask))'

def classify_itw(e):
    """Classify W₂₀ exponent for trivial-twiddle fast paths."""
    e = e % 20
    if e == 0:  return 'one'       # W^0 = 1
    if e == 5:  return 'neg_j'     # W^5 = -j  (forward)
    if e == 10: return 'neg_one'   # W^10 = -1
    if e == 15: return 'pos_j'     # W^15 = +j  (forward)
    return 'general'


class Emitter:
    def __init__(self): self.L=[]; self.ind=0
    def o(self, s=''): self.L.append('    '*self.ind + s)
    def c(self, s): self.o(f'/* {s} */')
    def b(self): self.L.append('')


# ════════════════════════════════════════
# Static W₂₀ arrays (compiler emits {1to8} broadcast)
# ════════════════════════════════════════

def emit_itw_static_arrays():
    L = []
    L.append('/* Internal W20 twiddle constants — static arrays for {1to8} broadcast */')
    L.append('static const double __attribute__((aligned(8))) w20_re[20] = {')
    for i in range(0, 20, 4):
        vals = [f'{w20(e)[0]:.20e}' for e in range(i, min(i+4, 20))]
        L.append(f'    {", ".join(vals)},')
    L.append('};')
    L.append('static const double __attribute__((aligned(8))) w20_im[20] = {')
    for i in range(0, 20, 4):
        vals = [f'{w20(e)[1]:.20e}' for e in range(i, min(i+4, 20))]
        L.append(f'    {", ".join(vals)},')
    L.append('};')
    L.append('')
    return L


def emit_itw_apply(em, xr, xi, e, fwd):
    """Apply internal W₂₀^e in-place. XOR negation for trivial cases."""
    e = e % 20
    cls = classify_itw(e)

    if cls == 'one':
        return
    if cls == 'neg_j':
        if fwd:
            em.o(f'{{ {T} t={xr}; {xr}={xi}; {xi}={neg("t")}; }}')
        else:
            em.o(f'{{ {T} t={xr}; {xr}={neg(xi)}; {xi}=t; }}')
        return
    if cls == 'neg_one':
        em.o(f'{xr}={neg(xr)}; {xi}={neg(xi)};')
        return
    if cls == 'pos_j':
        if fwd:
            em.o(f'{{ {T} t={xr}; {xr}={neg(xi)}; {xi}=t; }}')
        else:
            em.o(f'{{ {T} t={xr}; {xr}={xi}; {xi}={neg("t")}; }}')
        return

    # General case: load from static array → {1to8} broadcast
    wr = f'_mm512_set1_pd(w20_re[{e}])'
    wi = f'_mm512_set1_pd(w20_im[{e}])'
    em.o(f'{{ {T} t={xr};')
    if fwd:
        em.o(f'  {xr}=_mm512_fmsub_pd({xr},{wr},_mm512_mul_pd({xi},{wi}));')
        em.o(f'  {xi}=_mm512_fmadd_pd(t,{wi},_mm512_mul_pd({xi},{wr})); }}')
    else:
        em.o(f'  {xr}=_mm512_fmadd_pd({xr},{wr},_mm512_mul_pd({xi},{wi}));')
        em.o(f'  {xi}=_mm512_fnmadd_pd(t,{wi},_mm512_mul_pd({xi},{wr})); }}')


# ════════════════════════════════════════
# ILP-paired DFT-4 (two sub-FFTs interleaved)
# ════════════════════════════════════════

def emit_dft4_paired(em, n2_a, n2_b, has_tw, fwd):
    """Emit two DFT-4 sub-FFTs interleaved for ILP.
    A uses variables a0r..a3r/a0i..a3i, B uses b0r..b3r/b0i..b3i.
    24 ZMM total, 8 free."""
    inputs_a = [n2_a, n2_a+5, n2_a+10, n2_a+15]
    inputs_b = [n2_b, n2_b+5, n2_b+10, n2_b+15]

    em.c(f'DFT-4 PAIR: n2={n2_a} + n2={n2_b} (24 ZMM, 8 free)')
    em.o(f'{{')  # block scope

    # Interleaved loads: A0, B0, A1, B1, ...
    for j in range(4):
        em.o(f'{T} a{j}r=_mm512_load_pd(&in_re[{inputs_a[j]}*K+k]), a{j}i=_mm512_load_pd(&in_im[{inputs_a[j]}*K+k]);')
        em.o(f'{T} b{j}r=_mm512_load_pd(&in_re[{inputs_b[j]}*K+k]), b{j}i=_mm512_load_pd(&in_im[{inputs_b[j]}*K+k]);')

    # External twiddle apply (interleaved)
    if has_tw:
        for j in range(4):
            ma, mb = inputs_a[j], inputs_b[j]
            for prefix, m in [('a', ma), ('b', mb)]:
                if m > 0:
                    em.o(f'{{ {T} wr=_mm512_load_pd(&tw_re[{m-1}*K+k]), wi=_mm512_load_pd(&tw_im[{m-1}*K+k]), t={prefix}{j}r;')
                    if fwd:
                        em.o(f'  {prefix}{j}r=_mm512_fmsub_pd(t,wr,_mm512_mul_pd({prefix}{j}i,wi));')
                        em.o(f'  {prefix}{j}i=_mm512_fmadd_pd(t,wi,_mm512_mul_pd({prefix}{j}i,wr)); }}')
                    else:
                        em.o(f'  {prefix}{j}r=_mm512_fmadd_pd(t,wr,_mm512_mul_pd({prefix}{j}i,wi));')
                        em.o(f'  {prefix}{j}i=_mm512_fnmadd_pd(t,wi,_mm512_mul_pd({prefix}{j}i,wr)); }}')

    # DFT-4 butterfly A (interleaved with B)
    # s=x0+x2, d=x0-x2, t=x1+x3, u=x1-x3
    em.o(f'{T} asr=_mm512_add_pd(a0r,a2r), asi=_mm512_add_pd(a0i,a2i);')
    em.o(f'{T} bsr=_mm512_add_pd(b0r,b2r), bsi=_mm512_add_pd(b0i,b2i);')
    em.o(f'{T} adr=_mm512_sub_pd(a0r,a2r), adi=_mm512_sub_pd(a0i,a2i);')
    em.o(f'{T} bdr=_mm512_sub_pd(b0r,b2r), bdi=_mm512_sub_pd(b0i,b2i);')
    em.o(f'{T} atr=_mm512_add_pd(a1r,a3r), ati=_mm512_add_pd(a1i,a3i);')
    em.o(f'{T} btr=_mm512_add_pd(b1r,b3r), bti=_mm512_add_pd(b1i,b3i);')
    em.o(f'{T} aur=_mm512_sub_pd(a1r,a3r), aui=_mm512_sub_pd(a1i,a3i);')
    em.o(f'{T} bur=_mm512_sub_pd(b1r,b3r), bui=_mm512_sub_pd(b1i,b3i);')

    # out[0]=s+t, out[1]=d+j*u, out[2]=s-t, out[3]=d-j*u (forward: j*u = (-ui,ur))
    for prefix, n2 in [('a', n2_a), ('b', n2_b)]:
        s, d, t, u = f'{prefix}sr', f'{prefix}dr', f'{prefix}tr', f'{prefix}ur'
        si, di, ti, ui = f'{prefix}si', f'{prefix}di', f'{prefix}ti', f'{prefix}ui'
        for k1, (rexpr, iexpr) in enumerate([
            (f'_mm512_add_pd({s},{t})', f'_mm512_add_pd({si},{ti})'),
            (f'_mm512_add_pd({d},{ui})', f'_mm512_sub_pd({di},{u})') if fwd else
            (f'_mm512_sub_pd({d},{ui})', f'_mm512_add_pd({di},{u})'),
            (f'_mm512_sub_pd({s},{t})', f'_mm512_sub_pd({si},{ti})'),
            (f'_mm512_sub_pd({d},{ui})', f'_mm512_add_pd({di},{u})') if fwd else
            (f'_mm512_add_pd({d},{ui})', f'_mm512_sub_pd({di},{u})'),
        ]):
            slot = k1 * 5 + n2
            em.o(f'_mm512_store_pd(&sp_re[{slot}*{C}], {rexpr}); _mm512_store_pd(&sp_im[{slot}*{C}], {iexpr});')

    em.o(f'}}')
    em.b()


def emit_dft4_single(em, n2, has_tw, fwd):
    """Single DFT-4 (for the odd-one-out n2=4). 12 ZMM."""
    inputs = [n2, n2+5, n2+10, n2+15]
    em.c(f'DFT-4 single: n2={n2} (12 ZMM)')
    em.o(f'{{')
    for j in range(4):
        em.o(f'{T} r{j}=_mm512_load_pd(&in_re[{inputs[j]}*K+k]), i{j}=_mm512_load_pd(&in_im[{inputs[j]}*K+k]);')

    if has_tw:
        for j in range(4):
            m = inputs[j]
            if m > 0:
                em.o(f'{{ {T} wr=_mm512_load_pd(&tw_re[{m-1}*K+k]), wi=_mm512_load_pd(&tw_im[{m-1}*K+k]), t=r{j};')
                if fwd:
                    em.o(f'  r{j}=_mm512_fmsub_pd(t,wr,_mm512_mul_pd(i{j},wi));')
                    em.o(f'  i{j}=_mm512_fmadd_pd(t,wi,_mm512_mul_pd(i{j},wr)); }}')
                else:
                    em.o(f'  r{j}=_mm512_fmadd_pd(t,wr,_mm512_mul_pd(i{j},wi));')
                    em.o(f'  i{j}=_mm512_fnmadd_pd(t,wi,_mm512_mul_pd(i{j},wr)); }}')

    em.o(f'{{ {T} sr=_mm512_add_pd(r0,r2),si=_mm512_add_pd(i0,i2);')
    em.o(f'  {T} dr=_mm512_sub_pd(r0,r2),di=_mm512_sub_pd(i0,i2);')
    em.o(f'  {T} tr=_mm512_add_pd(r1,r3),ti=_mm512_add_pd(i1,i3);')
    em.o(f'  {T} ur=_mm512_sub_pd(r1,r3),ui=_mm512_sub_pd(i1,i3);')
    for k1, (rexpr, iexpr) in enumerate([
        ('_mm512_add_pd(sr,tr)', '_mm512_add_pd(si,ti)'),
        ('_mm512_add_pd(dr,ui)', '_mm512_sub_pd(di,ur)') if fwd else
        ('_mm512_sub_pd(dr,ui)', '_mm512_add_pd(di,ur)'),
        ('_mm512_sub_pd(sr,tr)', '_mm512_sub_pd(si,ti)'),
        ('_mm512_sub_pd(dr,ui)', '_mm512_add_pd(di,ur)') if fwd else
        ('_mm512_add_pd(dr,ui)', '_mm512_sub_pd(di,ur)'),
    ]):
        slot = k1 * 5 + n2
        em.o(f'  _mm512_store_pd(&sp_re[{slot}*{C}], {rexpr}); _mm512_store_pd(&sp_im[{slot}*{C}], {iexpr});')
    em.o(f'}}')
    em.o(f'}}')
    em.b()


# ════════════════════════════════════════
# ILP-paired DFT-5 (two columns interleaved)
# ════════════════════════════════════════

def emit_dft5_paired(em, k1_a, k1_b, out_target, out_stride, fwd):
    """Two DFT-5 columns interleaved. 24 ZMM, 8 free.
    A uses a-prefixed vars, B uses b-prefixed vars.
    Constants from memory ({1to8} broadcast), not registers."""

    em.c(f'DFT-5 PAIR: k1={k1_a} + k1={k1_b} (24 ZMM, 8 free)')
    em.o(f'{{')

    # Load from spill buffer — interleaved A/B
    for n2 in range(5):
        slot_a = k1_a * 5 + n2
        slot_b = k1_b * 5 + n2
        em.o(f'{T} a{n2}r=_mm512_load_pd(&sp_re[{slot_a}*{C}]), a{n2}i=_mm512_load_pd(&sp_im[{slot_a}*{C}]);')
        em.o(f'{T} b{n2}r=_mm512_load_pd(&sp_re[{slot_b}*{C}]), b{n2}i=_mm512_load_pd(&sp_im[{slot_b}*{C}]);')

    # DFT-5 Rader butterfly — interleaved A/B line by line
    # s1=x1+x4, s2=x2+x3, d1=x1-x4, d2=x2-x3
    em.o(f'{T} as1r=_mm512_add_pd(a1r,a4r), as1i=_mm512_add_pd(a1i,a4i);')
    em.o(f'{T} bs1r=_mm512_add_pd(b1r,b4r), bs1i=_mm512_add_pd(b1i,b4i);')
    em.o(f'{T} as2r=_mm512_add_pd(a2r,a3r), as2i=_mm512_add_pd(a2i,a3i);')
    em.o(f'{T} bs2r=_mm512_add_pd(b2r,b3r), bs2i=_mm512_add_pd(b2i,b3i);')
    em.o(f'{T} ad1r=_mm512_sub_pd(a1r,a4r), ad1i=_mm512_sub_pd(a1i,a4i);')
    em.o(f'{T} bd1r=_mm512_sub_pd(b1r,b4r), bd1i=_mm512_sub_pd(b1i,b4i);')
    em.o(f'{T} ad2r=_mm512_sub_pd(a2r,a3r), ad2i=_mm512_sub_pd(a2i,a3i);')
    em.o(f'{T} bd2r=_mm512_sub_pd(b2r,b3r), bd2i=_mm512_sub_pd(b2i,b3i);')

    # ss = s1+s2, y0 = x0+ss → store immediately
    em.o(f'{T} assr=_mm512_add_pd(as1r,as2r), assi=_mm512_add_pd(as1i,as2i);')
    em.o(f'{T} bssr=_mm512_add_pd(bs1r,bs2r), bssi=_mm512_add_pd(bs1i,bs2i);')

    def out_idx(k1, k2):
        return k1 + N1 * k2

    # Store y0
    for prefix, k1 in [('a', k1_a), ('b', k1_b)]:
        idx = out_idx(k1, 0)
        em.o(f'_mm512_store_pd(&{out_target}_re[{idx}*{out_stride}+k], _mm512_add_pd({prefix}0r,{prefix}ssr));')
        em.o(f'_mm512_store_pd(&{out_target}_im[{idx}*{out_stride}+k], _mm512_add_pd({prefix}0i,{prefix}ssi));')

    # t0 = x0 - 0.25*ss
    cD = '_mm512_set1_pd(0.25)'
    em.o(f'{T} at0r=_mm512_fnmadd_pd({cD},assr,a0r), at0i=_mm512_fnmadd_pd({cD},assi,a0i);')
    em.o(f'{T} bt0r=_mm512_fnmadd_pd({cD},bssr,b0r), bt0i=_mm512_fnmadd_pd({cD},bssi,b0i);')

    # t1 = cA*(s1-s2)
    cA = '_mm512_set1_pd(0.55901699437494742)'
    em.o(f'{T} at1r=_mm512_mul_pd({cA},_mm512_sub_pd(as1r,as2r)), at1i=_mm512_mul_pd({cA},_mm512_sub_pd(as1i,as2i));')
    em.o(f'{T} bt1r=_mm512_mul_pd({cA},_mm512_sub_pd(bs1r,bs2r)), bt1i=_mm512_mul_pd({cA},_mm512_sub_pd(bs1i,bs2i));')

    # p1=t0+t1, p2=t0-t1
    em.o(f'{T} ap1r=_mm512_add_pd(at0r,at1r), ap1i=_mm512_add_pd(at0i,at1i);')
    em.o(f'{T} bp1r=_mm512_add_pd(bt0r,bt1r), bp1i=_mm512_add_pd(bt0i,bt1i);')
    em.o(f'{T} ap2r=_mm512_sub_pd(at0r,at1r), ap2i=_mm512_sub_pd(at0i,at1i);')
    em.o(f'{T} bp2r=_mm512_sub_pd(bt0r,bt1r), bp2i=_mm512_sub_pd(bt0i,bt1i);')

    # U = cB*d1 + cC*d2, V = cB*d2 - cC*d1
    cB = '_mm512_set1_pd(0.95105651629515357)'
    cC = '_mm512_set1_pd(0.58778525229247313)'
    em.o(f'{T} aUr=_mm512_fmadd_pd({cC},ad2r,_mm512_mul_pd({cB},ad1r));')
    em.o(f'{T} aUi=_mm512_fmadd_pd({cC},ad2i,_mm512_mul_pd({cB},ad1i));')
    em.o(f'{T} bUr=_mm512_fmadd_pd({cC},bd2r,_mm512_mul_pd({cB},bd1r));')
    em.o(f'{T} bUi=_mm512_fmadd_pd({cC},bd2i,_mm512_mul_pd({cB},bd1i));')
    em.o(f'{T} aVr=_mm512_fnmadd_pd({cC},ad1r,_mm512_mul_pd({cB},ad2r));')
    em.o(f'{T} aVi=_mm512_fnmadd_pd({cC},ad1i,_mm512_mul_pd({cB},ad2i));')
    em.o(f'{T} bVr=_mm512_fnmadd_pd({cC},bd1r,_mm512_mul_pd({cB},bd2r));')
    em.o(f'{T} bVi=_mm512_fnmadd_pd({cC},bd1i,_mm512_mul_pd({cB},bd2i));')

    # Store y1,y4 (p1 ∓ j*U) and y2,y3 (p2 ± j*V)
    # Forward: j*(Ur,Ui) = (-Ui,Ur)
    #   y1 = p1 - j*U = (p1r+Ui, p1i-Ur)
    #   y4 = p1 + j*U = (p1r-Ui, p1i+Ur)
    #   y2 = p2 + j*V = (p2r-Vi, p2i+Vr)
    #   y3 = p2 - j*V = (p2r+Vi, p2i-Vr)
    for prefix, k1 in [('a', k1_a), ('b', k1_b)]:
        p1r,p1i = f'{prefix}p1r', f'{prefix}p1i'
        p2r,p2i = f'{prefix}p2r', f'{prefix}p2i'
        Ur,Ui = f'{prefix}Ur', f'{prefix}Ui'
        Vr,Vi = f'{prefix}Vr', f'{prefix}Vi'

        if fwd:
            store_pairs = [
                (1, f'_mm512_add_pd({p1r},{Ui})', f'_mm512_sub_pd({p1i},{Ur})'),
                (4, f'_mm512_sub_pd({p1r},{Ui})', f'_mm512_add_pd({p1i},{Ur})'),
                (2, f'_mm512_sub_pd({p2r},{Vi})', f'_mm512_add_pd({p2i},{Vr})'),
                (3, f'_mm512_add_pd({p2r},{Vi})', f'_mm512_sub_pd({p2i},{Vr})'),
            ]
        else:
            store_pairs = [
                (1, f'_mm512_sub_pd({p1r},{Ui})', f'_mm512_add_pd({p1i},{Ur})'),
                (4, f'_mm512_add_pd({p1r},{Ui})', f'_mm512_sub_pd({p1i},{Ur})'),
                (2, f'_mm512_add_pd({p2r},{Vi})', f'_mm512_sub_pd({p2i},{Vr})'),
                (3, f'_mm512_sub_pd({p2r},{Vi})', f'_mm512_add_pd({p2i},{Vr})'),
            ]
        for k2, rexpr, iexpr in store_pairs:
            idx = out_idx(k1, k2)
            em.o(f'_mm512_store_pd(&{out_target}_re[{idx}*{out_stride}+k], {rexpr});')
            em.o(f'_mm512_store_pd(&{out_target}_im[{idx}*{out_stride}+k], {iexpr});')

    em.o(f'}}')
    em.b()


# ════════════════════════════════════════
# Internal twiddle pass
# ════════════════════════════════════════

def emit_internal_twiddles(em, fwd):
    """Apply W₂₀ internal twiddles to spill buffer.
    k1=0: no twiddle. k1=1,2,3 × n2=1,2,3,4: 12 entries, 6 ZMM peak each."""
    em.c('Internal W₂₀ twiddles (XOR negation for trivial cases)')
    for k1 in range(1, N1):
        for n2 in range(1, N2):
            e = (k1 * n2) % R
            if e == 0:
                continue
            slot = k1 * 5 + n2
            xr = f'_itwr_{slot}'
            xi = f'_itwi_{slot}'
            em.o(f'{{ {T} {xr}=_mm512_load_pd(&sp_re[{slot}*{C}]), {xi}=_mm512_load_pd(&sp_im[{slot}*{C}]);')
            emit_itw_apply(em, xr, xi, e, fwd)
            em.o(f'  _mm512_store_pd(&sp_re[{slot}*{C}],{xr}); _mm512_store_pd(&sp_im[{slot}*{C}],{xi}); }}')
    em.b()


# ════════════════════════════════════════
# Full kernel generators
# ════════════════════════════════════════

def gen_dit_tw(direction):
    """DIT: external tw → DFT-4 → internal tw → DFT-5 → store."""
    fwd = direction == 'fwd'
    em = Emitter()

    em.L.append(ATTR)
    em.L.append(f'static void')
    em.L.append(f'radix20_ct_tw_dit_kernel_{direction}_avx512(')
    em.L.append(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
    em.L.append(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    em.L.append(f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,')
    em.L.append(f'    size_t K)')
    em.L.append(f'{{')
    em.ind = 1

    em.o(f'const __m512i sign_mask = _mm512_set1_epi64(0x8000000000000000ULL);')
    em.o(f'__attribute__((aligned(64))) double sp_re[{R*C}], sp_im[{R*C}];')
    em.b()

    em.o(f'for (size_t k = 0; k < K; k += {C}) {{')
    em.ind += 1

    em.c('═══ Pass 1: 5× DFT-4 with external twiddles (ILP paired) ═══')
    emit_dft4_paired(em, 0, 1, has_tw=True, fwd=fwd)
    emit_dft4_paired(em, 2, 3, has_tw=True, fwd=fwd)
    emit_dft4_single(em, 4, has_tw=True, fwd=fwd)

    emit_internal_twiddles(em, fwd)

    em.c('═══ Pass 2: 4× DFT-5 (ILP paired) ═══')
    emit_dft5_paired(em, 0, 1, 'out', 'K', fwd)
    emit_dft5_paired(em, 2, 3, 'out', 'K', fwd)

    em.ind -= 1
    em.o('}')
    em.ind -= 1
    em.o('}')
    em.b()
    return em.L


def gen_notw(direction):
    """NOTW: no external twiddles."""
    fwd = direction == 'fwd'
    em = Emitter()

    em.L.append(ATTR)
    em.L.append(f'static void')
    em.L.append(f'radix20_ct_n1_kernel_{direction}_avx512(')
    em.L.append(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
    em.L.append(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    em.L.append(f'    size_t K)')
    em.L.append(f'{{')
    em.ind = 1

    em.o(f'const __m512i sign_mask = _mm512_set1_epi64(0x8000000000000000ULL);')
    em.o(f'__attribute__((aligned(64))) double sp_re[{R*C}], sp_im[{R*C}];')
    em.b()

    em.o(f'for (size_t k = 0; k < K; k += {C}) {{')
    em.ind += 1

    em.c('═══ Pass 1: 5× DFT-4 (ILP paired, no twiddles) ═══')
    emit_dft4_paired(em, 0, 1, has_tw=False, fwd=fwd)
    emit_dft4_paired(em, 2, 3, has_tw=False, fwd=fwd)
    emit_dft4_single(em, 4, has_tw=False, fwd=fwd)

    emit_internal_twiddles(em, fwd)

    em.c('═══ Pass 2: 4× DFT-5 (ILP paired) ═══')
    emit_dft5_paired(em, 0, 1, 'out', 'K', fwd)
    emit_dft5_paired(em, 2, 3, 'out', 'K', fwd)

    em.ind -= 1
    em.o('}')
    em.ind -= 1
    em.o('}')
    em.b()
    return em.L


def gen_dif_tw(direction):
    """DIF: load → DFT-4 → internal tw → DFT-5 → external tw → store."""
    fwd = direction == 'fwd'
    em = Emitter()

    em.L.append(ATTR)
    em.L.append(f'static void')
    em.L.append(f'radix20_ct_tw_dif_kernel_{direction}_avx512(')
    em.L.append(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
    em.L.append(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    em.L.append(f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,')
    em.L.append(f'    size_t K)')
    em.L.append(f'{{')
    em.ind = 1

    em.o(f'const __m512i sign_mask = _mm512_set1_epi64(0x8000000000000000ULL);')
    em.o(f'__attribute__((aligned(64))) double sp_re[{R*C}], sp_im[{R*C}];')
    em.b()

    em.o(f'for (size_t k = 0; k < K; k += {C}) {{')
    em.ind += 1

    em.c('═══ Pass 1: 5× DFT-4 — NO external twiddles (DIF) ═══')
    emit_dft4_paired(em, 0, 1, has_tw=False, fwd=fwd)
    emit_dft4_paired(em, 2, 3, has_tw=False, fwd=fwd)
    emit_dft4_single(em, 4, has_tw=False, fwd=fwd)

    emit_internal_twiddles(em, fwd)

    em.c('═══ Pass 2: 4× DFT-5 → external twiddle on output → store (DIF) ═══')
    # For DIF, we need a special DFT-5 that applies external tw AFTER the butterfly.
    # Emit paired DFT-5 to temp vars, then apply twiddle, then store.
    emit_dft5_paired_dif(em, 0, 1, fwd)
    emit_dft5_paired_dif(em, 2, 3, fwd)

    em.ind -= 1
    em.o('}')
    em.ind -= 1
    em.o('}')
    em.b()
    return em.L


def emit_dft5_paired_dif(em, k1_a, k1_b, fwd):
    """DIF variant: DFT-5 → external twiddle on outputs → store.
    Same ILP pairing as DIT but twiddle goes on outputs."""

    em.c(f'DFT-5 DIF PAIR: k1={k1_a} + k1={k1_b}')
    em.o(f'{{')

    # Load from spill buffer
    for n2 in range(5):
        slot_a = k1_a * 5 + n2
        slot_b = k1_b * 5 + n2
        em.o(f'{T} a{n2}r=_mm512_load_pd(&sp_re[{slot_a}*{C}]), a{n2}i=_mm512_load_pd(&sp_im[{slot_a}*{C}]);')
        em.o(f'{T} b{n2}r=_mm512_load_pd(&sp_re[{slot_b}*{C}]), b{n2}i=_mm512_load_pd(&sp_im[{slot_b}*{C}]);')

    # DFT-5 butterfly (same math as DIT, interleaved A/B)
    em.o(f'{T} as1r=_mm512_add_pd(a1r,a4r), as1i=_mm512_add_pd(a1i,a4i);')
    em.o(f'{T} bs1r=_mm512_add_pd(b1r,b4r), bs1i=_mm512_add_pd(b1i,b4i);')
    em.o(f'{T} as2r=_mm512_add_pd(a2r,a3r), as2i=_mm512_add_pd(a2i,a3i);')
    em.o(f'{T} bs2r=_mm512_add_pd(b2r,b3r), bs2i=_mm512_add_pd(b2i,b3i);')
    em.o(f'{T} ad1r=_mm512_sub_pd(a1r,a4r), ad1i=_mm512_sub_pd(a1i,a4i);')
    em.o(f'{T} bd1r=_mm512_sub_pd(b1r,b4r), bd1i=_mm512_sub_pd(b1i,b4i);')
    em.o(f'{T} ad2r=_mm512_sub_pd(a2r,a3r), ad2i=_mm512_sub_pd(a2i,a3i);')
    em.o(f'{T} bd2r=_mm512_sub_pd(b2r,b3r), bd2i=_mm512_sub_pd(b2i,b3i);')

    em.o(f'{T} assr=_mm512_add_pd(as1r,as2r), assi=_mm512_add_pd(as1i,as2i);')
    em.o(f'{T} bssr=_mm512_add_pd(bs1r,bs2r), bssi=_mm512_add_pd(bs1i,bs2i);')

    cD = '_mm512_set1_pd(0.25)'
    cA = '_mm512_set1_pd(0.55901699437494742)'
    cB = '_mm512_set1_pd(0.95105651629515357)'
    cC = '_mm512_set1_pd(0.58778525229247313)'

    em.o(f'{T} at0r=_mm512_fnmadd_pd({cD},assr,a0r), at0i=_mm512_fnmadd_pd({cD},assi,a0i);')
    em.o(f'{T} bt0r=_mm512_fnmadd_pd({cD},bssr,b0r), bt0i=_mm512_fnmadd_pd({cD},bssi,b0i);')
    em.o(f'{T} at1r=_mm512_mul_pd({cA},_mm512_sub_pd(as1r,as2r)), at1i=_mm512_mul_pd({cA},_mm512_sub_pd(as1i,as2i));')
    em.o(f'{T} bt1r=_mm512_mul_pd({cA},_mm512_sub_pd(bs1r,bs2r)), bt1i=_mm512_mul_pd({cA},_mm512_sub_pd(bs1i,bs2i));')
    em.o(f'{T} ap1r=_mm512_add_pd(at0r,at1r), ap1i=_mm512_add_pd(at0i,at1i);')
    em.o(f'{T} bp1r=_mm512_add_pd(bt0r,bt1r), bp1i=_mm512_add_pd(bt0i,bt1i);')
    em.o(f'{T} ap2r=_mm512_sub_pd(at0r,at1r), ap2i=_mm512_sub_pd(at0i,at1i);')
    em.o(f'{T} bp2r=_mm512_sub_pd(bt0r,bt1r), bp2i=_mm512_sub_pd(bt0i,bt1i);')
    em.o(f'{T} aUr=_mm512_fmadd_pd({cC},ad2r,_mm512_mul_pd({cB},ad1r));')
    em.o(f'{T} aUi=_mm512_fmadd_pd({cC},ad2i,_mm512_mul_pd({cB},ad1i));')
    em.o(f'{T} bUr=_mm512_fmadd_pd({cC},bd2r,_mm512_mul_pd({cB},bd1r));')
    em.o(f'{T} bUi=_mm512_fmadd_pd({cC},bd2i,_mm512_mul_pd({cB},bd1i));')
    em.o(f'{T} aVr=_mm512_fnmadd_pd({cC},ad1r,_mm512_mul_pd({cB},ad2r));')
    em.o(f'{T} aVi=_mm512_fnmadd_pd({cC},ad1i,_mm512_mul_pd({cB},ad2i));')
    em.o(f'{T} bVr=_mm512_fnmadd_pd({cC},bd1r,_mm512_mul_pd({cB},bd2r));')
    em.o(f'{T} bVi=_mm512_fnmadd_pd({cC},bd1i,_mm512_mul_pd({cB},bd2i));')

    # Compute outputs, apply external twiddle, store
    for prefix, k1 in [('a', k1_a), ('b', k1_b)]:
        p1r,p1i = f'{prefix}p1r', f'{prefix}p1i'
        p2r,p2i = f'{prefix}p2r', f'{prefix}p2i'
        Ur,Ui = f'{prefix}Ur', f'{prefix}Ui'
        Vr,Vi = f'{prefix}Vr', f'{prefix}Vi'
        x0r, x0i = f'{prefix}0r', f'{prefix}0i'
        ssr, ssi = f'{prefix}ssr', f'{prefix}ssi'

        if fwd:
            outputs = [
                (0, f'_mm512_add_pd({x0r},{ssr})', f'_mm512_add_pd({x0i},{ssi})'),
                (1, f'_mm512_add_pd({p1r},{Ui})', f'_mm512_sub_pd({p1i},{Ur})'),
                (2, f'_mm512_sub_pd({p2r},{Vi})', f'_mm512_add_pd({p2i},{Vr})'),
                (3, f'_mm512_add_pd({p2r},{Vi})', f'_mm512_sub_pd({p2i},{Vr})'),
                (4, f'_mm512_sub_pd({p1r},{Ui})', f'_mm512_add_pd({p1i},{Ur})'),
            ]
        else:
            outputs = [
                (0, f'_mm512_add_pd({x0r},{ssr})', f'_mm512_add_pd({x0i},{ssi})'),
                (1, f'_mm512_sub_pd({p1r},{Ui})', f'_mm512_add_pd({p1i},{Ur})'),
                (2, f'_mm512_add_pd({p2r},{Vi})', f'_mm512_sub_pd({p2i},{Vr})'),
                (3, f'_mm512_sub_pd({p2r},{Vi})', f'_mm512_add_pd({p2i},{Vr})'),
                (4, f'_mm512_add_pd({p1r},{Ui})', f'_mm512_sub_pd({p1i},{Ur})'),
            ]

        for k2, rexpr, iexpr in outputs:
            m = k1 + N1 * k2  # output index
            yr = f'{prefix}y{k2}r'
            yi = f'{prefix}y{k2}i'
            em.o(f'{T} {yr}={rexpr}, {yi}={iexpr};')
            # DIF: apply external twiddle to output (conjugate for bwd)
            if m > 0:
                em.o(f'{{ {T} wr=_mm512_load_pd(&tw_re[{m-1}*K+k]), wi=_mm512_load_pd(&tw_im[{m-1}*K+k]), t={yr};')
                if fwd:
                    em.o(f'  {yr}=_mm512_fmsub_pd(t,wr,_mm512_mul_pd({yi},wi));')
                    em.o(f'  {yi}=_mm512_fmadd_pd(t,wi,_mm512_mul_pd({yi},wr)); }}')
                else:
                    em.o(f'  {yr}=_mm512_fmadd_pd(t,wr,_mm512_mul_pd({yi},wi));')
                    em.o(f'  {yi}=_mm512_fnmadd_pd(t,wi,_mm512_mul_pd({yi},wr)); }}')
            em.o(f'_mm512_store_pd(&out_re[{m}*K+k],{yr}); _mm512_store_pd(&out_im[{m}*K+k],{yi});')

    em.o(f'}}')
    em.b()


# ════════════════════════════════════════
# File generation
# ════════════════════════════════════════

def gen_file():
    guard = 'FFT_RADIX20_AVX512_CT_H'
    L = ['/**',
         ' * @file fft_radix20_avx512_ct.h',
         ' * @brief DFT-20 AVX-512 — zero-spill 4×5 CT with ILP pairing',
         ' * 32 ZMM: pairs of sub-FFTs interleaved to fill both FMA ports.',
         ' * Pass 1: [DFT-4+DFT-4] × 2 + [DFT-4 alone] = 5 sub-FFTs',
         ' * Pass 2: [DFT-5+DFT-5] × 2 = 4 column FFTs',
         ' * Internal W₂₀: XOR negation for trivial, {1to8} broadcast for general.',
         ' * DIT + DIF variants. Generated by gen_r20_ct_avx512.py',
         ' */', '',
         f'#ifndef {guard}', f'#define {guard}',
         '#include <stddef.h>', '#include <immintrin.h>', '']

    L.extend(emit_itw_static_arrays())

    for d in ('fwd', 'bwd'):
        L.extend(gen_notw(d))
    for d in ('fwd', 'bwd'):
        L.extend(gen_dit_tw(d))
    for d in ('fwd', 'bwd'):
        L.extend(gen_dif_tw(d))

    L.append(f'#endif /* {guard} */')
    return L


if __name__ == '__main__':
    lines = gen_file()
    print('\n'.join(lines))
    import sys
    print(f'/* {len(lines)} lines */', file=sys.stderr)
