#!/usr/bin/env python3
"""
gen_r64_avx512_tw.py — DFT-64 AVX-512 split twiddled codelet (8×8 CT + log3).

v2 optimizations:
  1. Flattened twiddle derivation tree (depth 4→3 for ILP)
  2. Sequential cross twiddles (data×col×row, no temp regs)
  3. Static arrays for internal W64 constants (embedded broadcast {1to8})
  4. XOR negation for ×j/×(-j)/×(-1) (1-cycle vs 4-cycle latency)

8×8 Cooley-Tukey with radix-8 butterfly (DFT-4 + W8 combine).
External twiddles: log3 — load W^1, W^8 (2 bases), derive via depth-3 tree.
Internal W64 twiddles: static arrays → compiler emits {1to8} broadcast.

AVX-512: k-step=8, 32 ZMM registers.
Output: fft_radix64_avx512_tw.h (DIT fwd/bwd + DIF fwd/bwd)
"""
import math

T = '__m512d'
C = 8  # doubles per vector
ATTR = '__attribute__((target("avx512f,avx512dq,fma")))'

# ═══════════════════════════════════════
# Helpers
# ═══════════════════════════════════════

def w64(e):
    e = e % 64
    a = 2.0 * math.pi * e / 64
    return (math.cos(a), -math.sin(a))

def cmul_split(em, dst_r, dst_i, ar, ai, br, bi):
    """Emit split complex multiply: dst = a * b."""
    em.o(f'const {T} {dst_r} = _mm512_fmsub_pd({ar},{br},_mm512_mul_pd({ai},{bi}));')
    em.o(f'const {T} {dst_i} = _mm512_fmadd_pd({ar},{bi},_mm512_mul_pd({ai},{br}));')

def cmul_inplace_split(em, xr, xi, wr, wi, fwd):
    """Emit in-place twiddle: x *= w (fwd) or x *= conj(w) (bwd)."""
    em.o(f'{{ {T} tr = {xr};')
    if fwd:
        em.o(f'  {xr} = _mm512_fmsub_pd({xr},{wr},_mm512_mul_pd({xi},{wi}));')
        em.o(f'  {xi} = _mm512_fmadd_pd(tr,{wi},_mm512_mul_pd({xi},{wr})); }}')
    else:
        em.o(f'  {xr} = _mm512_fmadd_pd({xr},{wr},_mm512_mul_pd({xi},{wi}));')
        em.o(f'  {xi} = _mm512_fnmadd_pd(tr,{wi},_mm512_mul_pd({xi},{wr})); }}')


class Emitter:
    def __init__(self): self.L=[]; self.ind=0
    def o(self, s=''): self.L.append('    '*self.ind + s)
    def c(self, s): self.o(f'/* {s} */')
    def b(self): self.L.append('')


# ═══════════════════════════════════════
# [OPT 4] XOR negation helper
# ═══════════════════════════════════════

def neg(v):
    """Negate via XOR sign bit — 1-cycle latency vs 4 for vsubpd."""
    return f'_mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512({v}), sign_mask))'


# ═══════════════════════════════════════
# Radix-8 butterfly (split, inline)
# ═══════════════════════════════════════

def emit_r8_butterfly(em, fwd, xr, xi, yr, yi):
    """Radix-8 DFT butterfly. Uses 'vc','vnc','sign_mask' from outer scope."""
    add = lambda a,b: f'_mm512_add_pd({a},{b})'
    sub = lambda a,b: f'_mm512_sub_pd({a},{b})'
    mul = lambda a,b: f'_mm512_mul_pd({a},{b})'

    em.o('{')
    em.ind += 1

    # DFT-4 evens
    em.o(f'const {T} epr={add(xr[0],xr[4])}, epi={add(xi[0],xi[4])};')
    em.o(f'const {T} eqr={sub(xr[0],xr[4])}, eqi={sub(xi[0],xi[4])};')
    em.o(f'const {T} err={add(xr[2],xr[6])}, eri={add(xi[2],xi[6])};')
    em.o(f'const {T} esr={sub(xr[2],xr[6])}, esi={sub(xi[2],xi[6])};')
    em.o(f'const {T} A0r={add("epr","err")}, A0i={add("epi","eri")};')
    em.o(f'const {T} A2r={sub("epr","err")}, A2i={sub("epi","eri")};')
    if fwd:
        em.o(f'const {T} A1r={add("eqr","esi")}, A1i={sub("eqi","esr")};')
        em.o(f'const {T} A3r={sub("eqr","esi")}, A3i={add("eqi","esr")};')
    else:
        em.o(f'const {T} A1r={sub("eqr","esi")}, A1i={add("eqi","esr")};')
        em.o(f'const {T} A3r={add("eqr","esi")}, A3i={sub("eqi","esr")};')

    # DFT-4 odds
    em.o(f'const {T} opr={add(xr[1],xr[5])}, opi={add(xi[1],xi[5])};')
    em.o(f'const {T} oqr={sub(xr[1],xr[5])}, oqi={sub(xi[1],xi[5])};')
    em.o(f'const {T} orr={add(xr[3],xr[7])}, ori={add(xi[3],xi[7])};')
    em.o(f'const {T} osr={sub(xr[3],xr[7])}, osi={sub(xi[3],xi[7])};')
    em.o(f'const {T} B0r={add("opr","orr")}, B0i={add("opi","ori")};')
    em.o(f'const {T} B2r={sub("opr","orr")}, B2i={sub("opi","ori")};')
    if fwd:
        em.o(f'const {T} B1r={add("oqr","osi")}, B1i={sub("oqi","osr")};')
        em.o(f'const {T} B3r={sub("oqr","osi")}, B3i={add("oqi","osr")};')
    else:
        em.o(f'const {T} B1r={sub("oqr","osi")}, B1i={add("oqi","osr")};')
        em.o(f'const {T} B3r={add("oqr","osi")}, B3i={sub("oqi","osr")};')

    # W8 combine: y0/y4
    em.o(f'{yr[0]}={add("A0r","B0r")}; {yi[0]}={add("A0i","B0i")};')
    em.o(f'{yr[4]}={sub("A0r","B0r")}; {yi[4]}={sub("A0i","B0i")};')

    # y1/y5
    if fwd:
        em.o(f'{{ const {T} t1r={mul("vc",add("B1r","B1i"))}, t1i={mul("vc",sub("B1i","B1r"))};')
    else:
        em.o(f'{{ const {T} t1r={mul("vc",sub("B1r","B1i"))}, t1i={mul("vc",add("B1r","B1i"))};')
    em.o(f'  {yr[1]}={add("A1r","t1r")}; {yi[1]}={add("A1i","t1i")};')
    em.o(f'  {yr[5]}={sub("A1r","t1r")}; {yi[5]}={sub("A1i","t1i")}; }}')

    # y2/y6
    if fwd:
        em.o(f'{yr[2]}={add("A2r","B2i")}; {yi[2]}={sub("A2i","B2r")};')
        em.o(f'{yr[6]}={sub("A2r","B2i")}; {yi[6]}={add("A2i","B2r")};')
    else:
        em.o(f'{yr[2]}={sub("A2r","B2i")}; {yi[2]}={add("A2i","B2r")};')
        em.o(f'{yr[6]}={add("A2r","B2i")}; {yi[6]}={sub("A2i","B2r")};')

    # y3/y7
    if fwd:
        em.o(f'{{ const {T} t3r={mul("vnc",sub("B3r","B3i"))}, t3i={mul("vnc",add("B3r","B3i"))};')
    else:
        em.o(f'{{ const {T} t3r={mul("vnc",add("B3r","B3i"))}, t3i={mul("vc",sub("B3r","B3i"))};')
    em.o(f'  {yr[3]}={add("A3r","t3r")}; {yi[3]}={add("A3i","t3i")};')
    em.o(f'  {yr[7]}={sub("A3r","t3r")}; {yi[7]}={sub("A3i","t3i")}; }}')

    em.ind -= 1
    em.o('}')


# ═══════════════════════════════════════
# [OPT 3] Internal W64 constants as static arrays
# ═══════════════════════════════════════

def collect_internal_twiddles():
    exps = set()
    for k1 in range(1, 8):
        for n2 in range(1, 8):
            exps.add((n2 * k1) % 64)
    return sorted(exps)

ITW_EXPS = collect_internal_twiddles()

def classify_itw(e):
    e = e % 64
    if e == 0: return 'one'
    if e == 16: return 'neg_j'
    if e == 32: return 'neg_one'
    if e == 48: return 'pos_j'
    if e == 8 or e == 24: return 'w8'
    return 'general'


def emit_itw_static_arrays():
    """Emit file-scope static arrays for internal W64 constants."""
    L = []
    L.append('/* Internal W64 twiddle constants — static arrays for {1to8} broadcast */')
    L.append('static const double __attribute__((aligned(8))) iw_re[64] = {')
    vals_re = ['0.0'] * 64
    vals_im = ['0.0'] * 64
    for e in range(64):
        wr, wi = w64(e)
        vals_re[e] = f'{wr:.20e}'
        vals_im[e] = f'{wi:.20e}'
    # Emit in rows of 4
    for i in range(0, 64, 4):
        L.append(f'    {vals_re[i]}, {vals_re[i+1]}, {vals_re[i+2]}, {vals_re[i+3]},')
    L.append('};')
    L.append('static const double __attribute__((aligned(8))) iw_im[64] = {')
    for i in range(0, 64, 4):
        L.append(f'    {vals_im[i]}, {vals_im[i+1]}, {vals_im[i+2]}, {vals_im[i+3]},')
    L.append('};')
    L.append('')
    return L


def emit_itw_apply(em, xr, xi, e, fwd):
    """Apply internal W64^e twiddle to (xr, xi) in-place.
    Uses static arrays → compiler emits {1to8} broadcast memory operands."""
    e = e % 64
    cls = classify_itw(e)

    if cls == 'one':
        return

    # [OPT 4] XOR negation for trivial twiddles
    if cls == 'neg_j':
        if fwd:
            em.o(f'{{ {T} tr = {xr}; {xr} = {xi}; {xi} = {neg("tr")}; }}')
        else:
            em.o(f'{{ {T} tr = {xr}; {xr} = {neg(xi)}; {xi} = tr; }}')
        return

    if cls == 'neg_one':
        em.o(f'{xr} = {neg(xr)};')
        em.o(f'{xi} = {neg(xi)};')
        return

    if cls == 'pos_j':
        if fwd:
            em.o(f'{{ {T} tr = {xr}; {xr} = {neg(xi)}; {xi} = tr; }}')
        else:
            em.o(f'{{ {T} tr = {xr}; {xr} = {xi}; {xi} = {neg("tr")}; }}')
        return

    if cls == 'w8':
        if e == 8:
            if fwd:
                em.o(f'{{ {T} tr = _mm512_mul_pd(vc,_mm512_add_pd({xr},{xi}));')
                em.o(f'  {xi} = _mm512_mul_pd(vc,_mm512_sub_pd({xi},{xr})); {xr} = tr; }}')
            else:
                em.o(f'{{ {T} tr = _mm512_mul_pd(vc,_mm512_sub_pd({xr},{xi}));')
                em.o(f'  {xi} = _mm512_mul_pd(vc,_mm512_add_pd({xr},{xi})); {xr} = tr; }}')
        elif e == 24:
            if fwd:
                em.o(f'{{ {T} tr = _mm512_mul_pd(vnc,_mm512_sub_pd({xr},{xi}));')
                em.o(f'  {xi} = _mm512_mul_pd(vnc,_mm512_add_pd({xr},{xi})); {xr} = tr; }}')
            else:
                em.o(f'{{ {T} tr = _mm512_mul_pd(vnc,_mm512_add_pd({xr},{xi}));')
                em.o(f'  {xi} = _mm512_mul_pd(vnc,_mm512_sub_pd({xi},{xr})); {xr} = tr; }}')
        return

    # [OPT 3] General case: load from static array → compiler uses {1to8}
    wr = f'_mm512_set1_pd(iw_re[{e}])'
    wi = f'_mm512_set1_pd(iw_im[{e}])'
    em.o(f'{{ {T} tr = {xr};')
    if fwd:
        em.o(f'  {xr} = _mm512_fmsub_pd({xr},{wr},_mm512_mul_pd({xi},{wi}));')
        em.o(f'  {xi} = _mm512_fmadd_pd(tr,{wi},_mm512_mul_pd({xi},{wr})); }}')
    else:
        em.o(f'  {xr} = _mm512_fmadd_pd({xr},{wr},_mm512_mul_pd({xi},{wi}));')
        em.o(f'  {xi} = _mm512_fnmadd_pd(tr,{wi},_mm512_mul_pd({xi},{wr})); }}')


# ═══════════════════════════════════════
# [OPT 1] Flattened log3 derivation tree
# ═══════════════════════════════════════

def emit_log3_derivation(em):
    """Emit log3 twiddle derivation with depth-3 tree for maximum ILP."""
    em.c('Log3: load W^1, W^8 from table, derive via depth-3 tree')
    em.o(f'const {T} ew1r = _mm512_load_pd(&tw_re[0*K+k]), ew1i = _mm512_load_pd(&tw_im[0*K+k]);')
    em.o(f'const {T} ew8r = _mm512_load_pd(&tw_re[7*K+k]), ew8i = _mm512_load_pd(&tw_im[7*K+k]);')
    em.b()

    # [OPT 1] Column bases: depth-3 tree (was depth-4)
    em.c('Column bases: W^2..W^7 (depth-3 tree)')
    cmul_split(em, 'ew2r','ew2i', 'ew1r','ew1i', 'ew1r','ew1i')  # depth 1
    cmul_split(em, 'ew3r','ew3i', 'ew1r','ew1i', 'ew2r','ew2i')  # depth 2
    cmul_split(em, 'ew4r','ew4i', 'ew2r','ew2i', 'ew2r','ew2i')  # depth 2 (parallel with ew3)
    cmul_split(em, 'ew5r','ew5i', 'ew1r','ew1i', 'ew4r','ew4i')  # depth 3
    cmul_split(em, 'ew6r','ew6i', 'ew3r','ew3i', 'ew3r','ew3i')  # depth 3 (parallel with ew5)
    cmul_split(em, 'ew7r','ew7i', 'ew3r','ew3i', 'ew4r','ew4i')  # depth 3 (parallel with ew5,ew6)
    em.b()

    # [OPT 1] Row bases: depth-3 tree (was depth-4)
    em.c('Row bases: W^16..W^56 (depth-3 tree)')
    cmul_split(em, 'ew16r','ew16i', 'ew8r','ew8i', 'ew8r','ew8i')    # depth 1
    cmul_split(em, 'ew24r','ew24i', 'ew8r','ew8i', 'ew16r','ew16i')   # depth 2
    cmul_split(em, 'ew32r','ew32i', 'ew16r','ew16i', 'ew16r','ew16i') # depth 2 (parallel with ew24)
    cmul_split(em, 'ew40r','ew40i', 'ew8r','ew8i', 'ew32r','ew32i')   # depth 3
    cmul_split(em, 'ew48r','ew48i', 'ew24r','ew24i', 'ew24r','ew24i') # depth 3 (parallel)
    cmul_split(em, 'ew56r','ew56i', 'ew24r','ew24i', 'ew32r','ew32i') # depth 3 (parallel)
    em.b()


# ═══════════════════════════════════════
# [OPT 2] Sequential cross twiddle application
# ═══════════════════════════════════════

ROW_BASES = {0: None, 8: ('ew8r','ew8i'), 16: ('ew16r','ew16i'),
             24: ('ew24r','ew24i'), 32: ('ew32r','ew32i'),
             40: ('ew40r','ew40i'), 48: ('ew48r','ew48i'), 56: ('ew56r','ew56i')}
COL_BASES = {0: None, 1: ('ew1r','ew1i'), 2: ('ew2r','ew2i'),
             3: ('ew3r','ew3i'), 4: ('ew4r','ew4i'), 5: ('ew5r','ew5i'),
             6: ('ew6r','ew6i'), 7: ('ew7r','ew7i')}


def emit_ext_twiddle(em, xr, xi, n, fwd):
    """Apply external twiddle W^n to (xr, xi) in-place.
    [OPT 2] Cross twiddles applied sequentially: data×col×row."""
    col = n % 8
    row8 = n - col

    if n == 0:
        return
    elif row8 == 0:
        # Pure column
        cr, ci = COL_BASES[col]
        cmul_inplace_split(em, xr, xi, cr, ci, fwd)
    elif col == 0:
        # Pure row
        rr, ri = ROW_BASES[row8]
        cmul_inplace_split(em, xr, xi, rr, ri, fwd)
    else:
        # [OPT 2] Sequential: data × col_base × row_base
        cr, ci = COL_BASES[col]
        rr, ri = ROW_BASES[row8]
        cmul_inplace_split(em, xr, xi, cr, ci, fwd)
        cmul_inplace_split(em, xr, xi, rr, ri, fwd)


# ═══════════════════════════════════════
# DIT generator
# ═══════════════════════════════════════

def gen_dit_tw(direction):
    fwd = direction == 'fwd'
    em = Emitter()

    em.L.append(ATTR)
    em.L.append(f'static void')
    em.L.append(f'radix64_tw_flat_dit_kernel_{direction}_avx512(')
    em.L.append(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
    em.L.append(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    em.L.append(f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,')
    em.L.append(f'    size_t K)')
    em.L.append(f'{{')
    em.ind = 1

    # [OPT 4] Sign mask for XOR negation
    em.o(f'const __m512i sign_mask = _mm512_set1_epi64(0x8000000000000000ULL);')
    em.o(f'const {T} vc = _mm512_set1_pd(0.707106781186547524400844362104849039284835938);')
    em.o(f'const {T} vnc = _mm512_set1_pd(-0.707106781186547524400844362104849039284835938);')
    # [OPT 3] No per-function constants — using static arrays

    em.o(f'__attribute__((aligned(64))) double sp_re[64*{C}], sp_im[64*{C}];')
    em.o(f'{T} x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i;')
    em.b()

    em.o(f'for (size_t k = 0; k < K; k += {C}) {{')
    em.ind += 1

    # [OPT 1] Flattened log3
    emit_log3_derivation(em)

    xr = [f'x{i}r' for i in range(8)]
    xi = [f'x{i}i' for i in range(8)]

    # PASS 1: 8 row sub-FFTs
    for n2 in range(8):
        em.c(f'Sub-FFT n2={n2}')
        for n1 in range(8):
            n = 8*n1 + n2
            em.o(f'{xr[n1]} = _mm512_load_pd(&in_re[{n}*K+k]); {xi[n1]} = _mm512_load_pd(&in_im[{n}*K+k]);')
            # [OPT 2] External twiddle applied sequentially
            emit_ext_twiddle(em, xr[n1], xi[n1], n, fwd)
        em.b()
        emit_r8_butterfly(em, fwd, xr, xi, xr, xi)
        em.b()
        for k1 in range(8):
            slot = n2 * 8 + k1
            em.o(f'_mm512_store_pd(&sp_re[{slot}*{C}], {xr[k1]}); _mm512_store_pd(&sp_im[{slot}*{C}], {xi[k1]});')
        em.b()

    # PASS 2: 8 column FFTs
    for k1 in range(8):
        em.c(f'Column k1={k1}')
        for n2 in range(8):
            slot = n2 * 8 + k1
            em.o(f'{xr[n2]} = _mm512_load_pd(&sp_re[{slot}*{C}]); {xi[n2]} = _mm512_load_pd(&sp_im[{slot}*{C}]);')
        # [OPT 3] Internal W64 twiddles via static array broadcast
        if k1 > 0:
            for n2 in range(1, 8):
                e = (n2 * k1) % 64
                emit_itw_apply(em, xr[n2], xi[n2], e, fwd)
        em.b()
        emit_r8_butterfly(em, fwd, xr, xi, xr, xi)
        em.b()
        for k2 in range(8):
            m = k1 + 8 * k2
            em.o(f'_mm512_store_pd(&out_re[{m}*K+k], {xr[k2]}); _mm512_store_pd(&out_im[{m}*K+k], {xi[k2]});')
        em.b()

    em.ind -= 1
    em.o('}')
    em.L.append('}')
    em.L.append('')
    return em.L


# ═══════════════════════════════════════
# DIF generator
# ═══════════════════════════════════════

def gen_dif_tw(direction):
    fwd = direction == 'fwd'
    em = Emitter()

    em.L.append(ATTR)
    em.L.append(f'static void')
    em.L.append(f'radix64_tw_flat_dif_kernel_{direction}_avx512(')
    em.L.append(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
    em.L.append(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    em.L.append(f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,')
    em.L.append(f'    size_t K)')
    em.L.append(f'{{')
    em.ind = 1

    em.o(f'const __m512i sign_mask = _mm512_set1_epi64(0x8000000000000000ULL);')
    em.o(f'const {T} vc = _mm512_set1_pd(0.707106781186547524400844362104849039284835938);')
    em.o(f'const {T} vnc = _mm512_set1_pd(-0.707106781186547524400844362104849039284835938);')

    em.o(f'__attribute__((aligned(64))) double sp_re[64*{C}], sp_im[64*{C}];')
    em.o(f'{T} x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i;')
    em.b()

    em.o(f'for (size_t k = 0; k < K; k += {C}) {{')
    em.ind += 1

    emit_log3_derivation(em)

    xr = [f'x{i}r' for i in range(8)]
    xi = [f'x{i}i' for i in range(8)]

    # PASS 1: Load (no twiddle) → R=8 butterfly → spill
    for n2 in range(8):
        em.c(f'Sub-FFT n2={n2} (DIF: no external twiddle on inputs)')
        for n1 in range(8):
            n = 8*n1 + n2
            em.o(f'{xr[n1]} = _mm512_load_pd(&in_re[{n}*K+k]); {xi[n1]} = _mm512_load_pd(&in_im[{n}*K+k]);')
        em.b()
        emit_r8_butterfly(em, fwd, xr, xi, xr, xi)
        em.b()
        for k1 in range(8):
            slot = n2 * 8 + k1
            em.o(f'_mm512_store_pd(&sp_re[{slot}*{C}], {xr[k1]}); _mm512_store_pd(&sp_im[{slot}*{C}], {xi[k1]});')
        em.b()

    # PASS 2: Reload → internal W64 → R=8 butterfly → external tw → store
    for k1 in range(8):
        em.c(f'Column k1={k1}')
        for n2 in range(8):
            slot = n2 * 8 + k1
            em.o(f'{xr[n2]} = _mm512_load_pd(&sp_re[{slot}*{C}]); {xi[n2]} = _mm512_load_pd(&sp_im[{slot}*{C}]);')
        if k1 > 0:
            for n2 in range(1, 8):
                e = (n2 * k1) % 64
                emit_itw_apply(em, xr[n2], xi[n2], e, fwd)
        em.b()
        emit_r8_butterfly(em, fwd, xr, xi, xr, xi)
        em.b()
        # DIF: apply external twiddle to outputs
        for k2 in range(8):
            m = k1 + 8 * k2
            emit_ext_twiddle(em, xr[k2], xi[k2], m, fwd)
            em.o(f'_mm512_store_pd(&out_re[{m}*K+k], {xr[k2]}); _mm512_store_pd(&out_im[{m}*K+k], {xi[k2]});')
        em.b()

    em.ind -= 1
    em.o('}')
    em.L.append('}')
    em.L.append('')
    return em.L


# ═══════════════════════════════════════
# File generator
# ═══════════════════════════════════════

def gen_file():
    guard = 'FFT_RADIX64_AVX512_TW_H'
    L = ['/**',
         ' * @file fft_radix64_avx512_tw.h',
         ' * @brief DFT-64 AVX-512 split — 8×8 CT + log3 twiddles (v2)',
         ' * Optimizations: depth-3 log3 tree, sequential cross tw,',
         ' * static W64 arrays ({1to8} broadcast), XOR negation.',
         ' * Generated by gen_r64_avx512_tw.py',
         ' */', '',
         f'#ifndef {guard}', f'#define {guard}',
         '#include <stddef.h>', '#include <immintrin.h>', '']

    # [OPT 3] File-scope static arrays
    L.extend(emit_itw_static_arrays())

    for d in ('fwd', 'bwd'):
        L.extend(gen_dit_tw(d))
        L.append('')
    for d in ('fwd', 'bwd'):
        L.extend(gen_dif_tw(d))
        L.append('')

    L.append(f'#endif /* {guard} */')
    return L


if __name__ == '__main__':
    print('\n'.join(gen_file()))
