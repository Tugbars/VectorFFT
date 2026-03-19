#!/usr/bin/env python3
"""
gen_r25_native_il.py — DFT-25 (5×5 CT) native interleaved codelets.

Native IL arithmetic (fmaddsub/addsub) + log3 external twiddles.
Internal W25 twiddles as pre-interleaved broadcast constants.

Pass 1: 5 sub-FFTs (radix-5), results spilled to L1 buffer
Pass 2: 5 column FFTs with internal W25 twiddles + radix-5

External twiddles: log3 from W^1 and W^5 (2 base loads, 22 VZMUL derivations).
Internal twiddles: 9 unique W25 constants, pre-broadcast outside k-loop.

Usage: gen_r25_native_il.py <avx2|avx512> <dit|dif>
"""
import math, sys

N, N1, N2 = 25, 5, 5

# ════════════════════════════════════════
# Internal W25 twiddle data
# ════════════════════════════════════════

def w25(e):
    e = e % 25
    a = 2.0 * math.pi * e / 25
    return (math.cos(a), -math.sin(a))

def collect_internal_twiddles():
    exps = set()
    for n2 in range(1, N2):
        for k1 in range(1, N1):
            e = (n2 * k1) % N
            if e != 0: exps.add(e)
    return sorted(exps)

ITW_EXPS = collect_internal_twiddles()  # [1,2,3,4,6,8,9,12,16]

# ════════════════════════════════════════
# ISA abstraction
# ════════════════════════════════════════

def isa_avx2():
    T = '__m256d'; C = 2; W = 4
    attr = '__attribute__((target("avx2,fma")))'
    def load(v, n):  return [f'{v} = _mm256_load_pd(&in[2*({n}*K+k)]);']
    def store(v, n): return [f'_mm256_store_pd(&out[2*({n}*K+k)], {v});']
    def tw_load(v, idx):
        return [
            f'{{ __m128d _tr = _mm_load_pd(&tw_re[{idx}*K+k]);',
            f'  __m128d _ti = _mm_load_pd(&tw_im[{idx}*K+k]);',
            f'  {v} = _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_unpacklo_pd(_tr,_ti)), _mm_unpackhi_pd(_tr,_ti), 1); }}']
    def spill(v, slot):  return [f'_mm256_store_pd(&sp[{slot}*{W}], {v});']
    def reload(v, slot): return [f'{v} = _mm256_load_pd(&sp[{slot}*{W}]);']
    def set1_c(vr, vi):
        """Create interleaved constant [vr,vi,vr,vi]"""
        return f'_mm256_set_pd({vi},{vr},{vi},{vr})'
    add  = lambda a,b: f'_mm256_add_pd({a},{b})'
    sub  = lambda a,b: f'_mm256_sub_pd({a},{b})'
    mul  = lambda a,b: f'_mm256_mul_pd({a},{b})'
    fma  = lambda a,b,c: f'_mm256_fmadd_pd({a},{b},{c})'
    fnma = lambda a,b,c: f'_mm256_fnmadd_pd({a},{b},{c})'
    set1 = lambda v: f'_mm256_set1_pd({v})'
    flip = lambda x: f'_mm256_permute_pd({x},0x5)'
    dupl = lambda x: f'_mm256_movedup_pd({x})'
    duph = lambda x: f'_mm256_permute_pd({x},0xF)'
    vfmai  = lambda b,c: f'_mm256_addsub_pd({c},{flip(b)})'
    vfnmsi = lambda b,c: f'_mm256_sub_pd({c},_mm256_addsub_pd(_mm256_setzero_pd(),{flip(b)}))'
    vzmul  = lambda tw,x: f'_mm256_fmaddsub_pd({x},{dupl(tw)},{mul(flip(x),duph(tw))})'
    vzmulj = lambda tw,x: f'_mm256_fmsubadd_pd({x},{dupl(tw)},{mul(flip(x),duph(tw))})'
    return dict(T=T, C=C, W=W, attr=attr, load=load, store=store, tw_load=tw_load,
                spill=spill, reload=reload, set1_c=set1_c,
                add=add, sub=sub, mul=mul, fma=fma, fnma=fnma, set1=set1,
                flip=flip, dupl=dupl, duph=duph,
                vfmai=vfmai, vfnmsi=vfnmsi, vzmul=vzmul, vzmulj=vzmulj,
                align=32)

def isa_avx512():
    T = '__m512d'; C = 4; W = 8
    attr = '__attribute__((target("avx512f,avx512dq,fma")))'
    def load(v, n):  return [f'{v} = _mm512_load_pd(&in[2*({n}*K+k)]);']
    def store(v, n): return [f'_mm512_store_pd(&out[2*({n}*K+k)], {v});']
    def tw_load(v, idx):
        return [
            f'{{ __m256d _tr = _mm256_load_pd(&tw_re[{idx}*K+k]);',
            f'  __m256d _ti = _mm256_load_pd(&tw_im[{idx}*K+k]);',
            f'  {v} = _mm512_permutex2var_pd(_mm512_castpd256_pd512(_tr),',
            f'    _mm512_set_epi64(11,3,10,2,9,1,8,0), _mm512_castpd256_pd512(_ti)); }}']
    def spill(v, slot):  return [f'_mm512_store_pd(&sp[{slot}*{W}], {v});']
    def reload(v, slot): return [f'{v} = _mm512_load_pd(&sp[{slot}*{W}]);']
    def set1_c(vr, vi):
        return f'_mm512_set_pd({vi},{vr},{vi},{vr},{vi},{vr},{vi},{vr})'
    add  = lambda a,b: f'_mm512_add_pd({a},{b})'
    sub  = lambda a,b: f'_mm512_sub_pd({a},{b})'
    mul  = lambda a,b: f'_mm512_mul_pd({a},{b})'
    fma  = lambda a,b,c: f'_mm512_fmadd_pd({a},{b},{c})'
    fnma = lambda a,b,c: f'_mm512_fnmadd_pd({a},{b},{c})'
    set1 = lambda v: f'_mm512_set1_pd({v})'
    flip = lambda x: f'_mm512_permute_pd({x},0x55)'
    dupl = lambda x: f'_mm512_movedup_pd({x})'
    duph = lambda x: f'_mm512_permute_pd({x},0xFF)'
    vfmai  = lambda b,c: f'_mm512_fmaddsub_pd(_mm512_set1_pd(1.0),{c},{flip(b)})'
    vfnmsi = lambda b,c: f'_mm512_sub_pd({c},_mm512_fmaddsub_pd(_mm512_set1_pd(1.0),_mm512_setzero_pd(),{flip(b)}))'
    vzmul  = lambda tw,x: f'_mm512_fmaddsub_pd({x},{dupl(tw)},{mul(flip(x),duph(tw))})'
    vzmulj = lambda tw,x: f'_mm512_fmsubadd_pd({x},{dupl(tw)},{mul(flip(x),duph(tw))})'
    return dict(T=T, C=C, W=W, attr=attr, load=load, store=store, tw_load=tw_load,
                spill=spill, reload=reload, set1_c=set1_c,
                add=add, sub=sub, mul=mul, fma=fma, fnma=fnma, set1=set1,
                flip=flip, dupl=dupl, duph=duph,
                vfmai=vfmai, vfnmsi=vfnmsi, vzmul=vzmul, vzmulj=vzmulj,
                align=64)


class Emitter:
    def __init__(self): self.L=[]; self.ind=0
    def o(self, s=''): self.L.append('    '*self.ind + s)
    def c(self, s): self.o(f'/* {s} */')
    def b(self): self.L.append('')
    def lines(self, ll):
        for l in ll: self.o(l)


# ════════════════════════════════════════
# Radix-5 butterfly (native IL, inline)
# ════════════════════════════════════════

def emit_r5_butterfly(em, isa, fwd, xvars, yvars):
    """Native IL radix-5 butterfly. xvars[0..4] → yvars[0..4]."""
    T = isa['T']
    add, sub, mul = isa['add'], isa['sub'], isa['mul']
    fma, fnma = isa['fma'], isa['fnma']
    vfmai, vfnmsi = isa['vfmai'], isa['vfnmsi']
    x = xvars; y = yvars

    em.o(f'{{ const {T} s1={add(x[1],x[4])}, s2={add(x[2],x[3])};')
    em.o(f'  const {T} d1={sub(x[1],x[4])}, d2={sub(x[2],x[3])};')
    em.o(f'  {y[0]} = {add(x[0], add("s1","s2"))};')
    em.o(f'  const {T} t0={fnma("cK3",add("s1","s2"),x[0])};')
    em.o(f'  const {T} t1={mul("cK0",sub("s1","s2"))};')
    em.o(f'  const {T} p1={add("t0","t1")}, p2={sub("t0","t1")};')
    em.o(f'  const {T} U={fma("cK2","d2",mul("cK1","d1"))};')
    em.o(f'  const {T} V={fnma("cK2","d1",mul("cK1","d2"))};')
    em.o(f'  const {T} jU={isa["vbyi"]("U")}, jV={isa["vbyi"]("V")};')
    if fwd:
        em.o(f'  {y[1]}={sub("p1","jU")}; {y[4]}={add("p1","jU")};')
        em.o(f'  {y[2]}={add("p2","jV")}; {y[3]}={sub("p2","jV")}; }}')
    else:
        em.o(f'  {y[1]}={add("p1","jU")}; {y[4]}={sub("p1","jU")};')
        em.o(f'  {y[2]}={sub("p2","jV")}; {y[3]}={add("p2","jV")}; }}')


def add_vbyi(isa):
    """Ensure isa has vbyi."""
    if 'vbyi' not in isa:
        flip = isa['flip']
        if isa['C'] == 2:  # AVX2
            isa['vbyi'] = lambda x: f'_mm256_addsub_pd(_mm256_setzero_pd(),{flip(x)})'
        else:  # AVX-512
            isa['vbyi'] = lambda x: f'_mm512_fmaddsub_pd(_mm512_set1_pd(1.0),_mm512_setzero_pd(),{flip(x)})'


# ════════════════════════════════════════
# Log3 external twiddle derivation
# ════════════════════════════════════════

def emit_log3_ext_derive(em, isa, fwd):
    """Load W^1 and W^5 from split table, derive all 24 external twiddles.
    Column bases: W^2=W^1², W^3=W^1·W^2, W^4=W^1·W^3
    Row bases: W^10=W^5², W^15=W^5·W^10, W^20=W^5·W^15
    Cross: W^(5j+c) = W^c · W^(5j) for each (j,c) pair
    Total: 2 loads + 22 VZMUL."""
    T = isa['T']
    zmul = isa['vzmul']

    em.c('Log3: load W^1, W^5 from split table')
    em.lines(isa['tw_load']('ew1', 0))    # tw index 0 = W^1
    em.lines(isa['tw_load']('ew5', 4))    # tw index 4 = W^5
    em.b()

    em.c('Derive column bases: W^2, W^3, W^4')
    em.o(f'const {T} ew2 = {zmul("ew1","ew1")};')
    em.o(f'const {T} ew3 = {zmul("ew1","ew2")};')
    em.o(f'const {T} ew4 = {zmul("ew1","ew3")};')
    em.b()

    em.c('Derive row bases: W^10, W^15, W^20')
    em.o(f'const {T} ew10 = {zmul("ew5","ew5")};')
    em.o(f'const {T} ew15 = {zmul("ew5","ew10")};')
    em.o(f'const {T} ew20 = {zmul("ew5","ew15")};')
    em.b()


def get_ext_tw_name(n):
    """Get external twiddle variable name for index n (1..24)."""
    if n == 0: return None
    # Base twiddles already derived
    bases = {1:'ew1', 2:'ew2', 3:'ew3', 4:'ew4',
             5:'ew5', 10:'ew10', 15:'ew15', 20:'ew20'}
    if n in bases: return bases[n]
    return f'ew{n}'


def emit_ext_tw_for_n(em, isa, n):
    """Emit derivation for W^n if not already a base. Uses cross-product."""
    T = isa['T']
    zmul = isa['vzmul']
    bases = {1,2,3,4,5,10,15,20}
    if n in bases: return  # Already available

    # Decompose n = 5*row + col
    col = n % 5
    row5 = n - col  # multiple of 5
    row_var = {0: None, 5:'ew5', 10:'ew10', 15:'ew15', 20:'ew20'}
    col_var = {0: None, 1:'ew1', 2:'ew2', 3:'ew3', 4:'ew4'}

    if col == 0:
        return  # row bases already derived
    if row5 == 0:
        return  # column bases already derived

    em.o(f'const {T} ew{n} = {zmul(col_var[col], row_var[row5])};')


# ════════════════════════════════════════
# Kernel generators
# ════════════════════════════════════════

def gen_dit_tw(isa_cfg, direction):
    """DIT: twiddle inputs → sub-FFTs → internal tw → column FFTs → store."""
    isa = isa_cfg
    add_vbyi(isa)
    T, C, W = isa['T'], isa['C'], isa['W']
    fwd = direction == 'fwd'
    zmul = isa['vzmul'] if fwd else isa['vzmulj']
    isa_name = 'avx2' if C == 2 else 'avx512'

    em = Emitter()
    if isa['attr']: em.L.append(isa['attr'])
    em.L.append(f'static void')
    em.L.append(f'radix25_tw_flat_dit_kernel_{direction}_il_{isa_name}(')
    em.L.append(f'    const double * __restrict__ in,')
    em.L.append(f'    double * __restrict__ out,')
    em.L.append(f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,')
    em.L.append(f'    size_t K)')
    em.L.append(f'{{')
    em.ind = 1

    # Radix-5 butterfly constants
    em.o(f'const {T} cK0 = {isa["set1"]("0.559016994374947424102293417182819058860154590")};')
    em.o(f'const {T} cK1 = {isa["set1"]("0.951056516295153572116439333379382143405698634")};')
    em.o(f'const {T} cK2 = {isa["set1"]("0.587785252292473129168705954639072768597652438")};')
    em.o(f'const {T} cK3 = {isa["set1"]("0.250000000000000000000000000000000000000000000")};')
    em.b()

    # Internal W25 broadcast constants (pre-interleaved)
    em.c('Internal W25 twiddle constants (pre-interleaved)')
    for e in ITW_EXPS:
        wr, wi = w25(e)
        em.o(f'const {T} iw{e} = {isa["set1_c"](f"{wr:.20e}", f"{wi:.20e}")};')
    em.b()

    # Spill buffer
    em.o(f'__attribute__((aligned({isa["align"]}))) double sp[{25*W}];')
    em.b()

    em.o(f'for (size_t k = 0; k < K; k += {C}) {{')
    em.ind += 1

    # Working variables (declared once, reused per sub-FFT)
    em.o(f'{T} x0,x1,x2,x3,x4, y0,y1,y2,y3,y4;')
    em.b()

    # External twiddle derivation
    em.o(f'{T} ew1, ew5;')
    emit_log3_ext_derive(em, isa, fwd)

    # ── PASS 1: 5 sub-FFTs by n2 ──
    xv = [f'x{i}' for i in range(5)]
    yv = [f'y{i}' for i in range(5)]

    for n2 in range(N2):
        em.c(f'Sub-FFT n2={n2}')

        # Load + external twiddle
        for n1 in range(N1):
            n = N2*n1 + n2  # input index
            em.lines(isa['load'](xv[n1], n))
            if n > 0:
                tw_name = get_ext_tw_name(n)
                if tw_name is None:
                    pass  # n=0, no twiddle
                else:
                    # May need to derive cross-product twiddle
                    emit_ext_tw_for_n(em, isa, n)
                    em.o(f'{xv[n1]} = {zmul(tw_name, xv[n1])};')

        # Radix-5 butterfly
        emit_r5_butterfly(em, isa, fwd, xv, yv)

        # Spill results
        for k1 in range(N1):
            em.lines(isa['spill'](yv[k1], n2*N1 + k1))
        em.b()

    # ── PASS 2: column FFTs with internal W25 twiddles ──
    em.c('PASS 2: column FFTs')
    em.b()
    em.o(f'{T} c0,c1,c2,c3,c4, r0,r1,r2,r3,r4;')

    for k1 in range(N1):
        em.c(f'Column k1={k1}')
        cv = [f'c{i}' for i in range(N2)]
        for n2 in range(N2):
            em.lines(isa['reload'](cv[n2], n2*N1 + k1))

        # Apply internal W25 twiddles
        if k1 > 0:
            for n2 in range(1, N2):
                e = (n2 * k1) % N
                if fwd:
                    em.o(f'{cv[n2]} = {isa["vzmul"](f"iw{e}", cv[n2])};')
                else:
                    em.o(f'{cv[n2]} = {isa["vzmulj"](f"iw{e}", cv[n2])};')
        em.b()

        # Radix-5 butterfly
        rv = [f'r{i}' for i in range(N2)]
        emit_r5_butterfly(em, isa, fwd, cv, rv)

        # Store outputs
        for k2 in range(N2):
            m = k1 + N1*k2  # output index
            em.lines(isa['store'](rv[k2], m))
        em.b()

    em.ind -= 1
    em.o('}')
    em.L.append('}'); em.L.append('')
    return em.L


def gen_dif_tw(isa_cfg, direction):
    """DIF: load → sub-FFTs → internal tw → column FFTs → external tw → store."""
    isa = isa_cfg
    add_vbyi(isa)
    T, C, W = isa['T'], isa['C'], isa['W']
    fwd = direction == 'fwd'
    zmul = isa['vzmul'] if fwd else isa['vzmulj']
    isa_name = 'avx2' if C == 2 else 'avx512'

    em = Emitter()
    if isa['attr']: em.L.append(isa['attr'])
    em.L.append(f'static void')
    em.L.append(f'radix25_tw_flat_dif_kernel_{direction}_il_{isa_name}(')
    em.L.append(f'    const double * __restrict__ in,')
    em.L.append(f'    double * __restrict__ out,')
    em.L.append(f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,')
    em.L.append(f'    size_t K)')
    em.L.append(f'{{')
    em.ind = 1

    em.o(f'const {T} cK0 = {isa["set1"]("0.559016994374947424102293417182819058860154590")};')
    em.o(f'const {T} cK1 = {isa["set1"]("0.951056516295153572116439333379382143405698634")};')
    em.o(f'const {T} cK2 = {isa["set1"]("0.587785252292473129168705954639072768597652438")};')
    em.o(f'const {T} cK3 = {isa["set1"]("0.250000000000000000000000000000000000000000000")};')
    em.b()

    em.c('Internal W25 twiddle constants (pre-interleaved)')
    for e in ITW_EXPS:
        wr, wi = w25(e)
        em.o(f'const {T} iw{e} = {isa["set1_c"](f"{wr:.20e}", f"{wi:.20e}")};')
    em.b()

    em.o(f'__attribute__((aligned({isa["align"]}))) double sp[{25*W}];')
    em.b()

    em.o(f'for (size_t k = 0; k < K; k += {C}) {{')
    em.ind += 1

    # Working variables
    em.o(f'{T} x0,x1,x2,x3,x4, y0,y1,y2,y3,y4;')
    em.b()

    em.o(f'{T} ew1, ew5;')
    emit_log3_ext_derive(em, isa, fwd)

    # ── PASS 1: Load (no external tw) → radix-5 → spill ──
    xv = [f'x{i}' for i in range(5)]
    yv = [f'y{i}' for i in range(5)]

    for n2 in range(N2):
        em.c(f'Sub-FFT n2={n2}')
        for n1 in range(N1):
            n = N2*n1 + n2
            em.lines(isa['load'](xv[n1], n))

        emit_r5_butterfly(em, isa, fwd, xv, yv)

        for k1 in range(N1):
            em.lines(isa['spill'](yv[k1], n2*N1 + k1))
        em.b()

    # ── PASS 2: reload → internal W25 → radix-5 → external tw → store ──
    em.c('PASS 2: column FFTs + external twiddle')
    em.b()
    em.o(f'{T} c0,c1,c2,c3,c4, r0,r1,r2,r3,r4;')

    for k1 in range(N1):
        em.c(f'Column k1={k1}')
        cv = [f'c{i}' for i in range(N2)]
        for n2 in range(N2):
            em.lines(isa['reload'](cv[n2], n2*N1 + k1))

        if k1 > 0:
            for n2 in range(1, N2):
                e = (n2 * k1) % N
                if fwd:
                    em.o(f'{cv[n2]} = {isa["vzmul"](f"iw{e}", cv[n2])};')
                else:
                    em.o(f'{cv[n2]} = {isa["vzmulj"](f"iw{e}", cv[n2])};')
        em.b()

        rv = [f'r{i}' for i in range(N2)]
        emit_r5_butterfly(em, isa, fwd, cv, rv)

        # External twiddle on outputs
        for k2 in range(N2):
            m = k1 + N1*k2
            if m > 0:
                tw_name = get_ext_tw_name(m)
                emit_ext_tw_for_n(em, isa, m)
                em.o(f'{rv[k2]} = {zmul(tw_name, rv[k2])};')
            em.lines(isa['store'](rv[k2], m))
        em.b()

    em.ind -= 1
    em.o('}')
    em.L.append('}'); em.L.append('')
    return em.L


def gen_file(isa_name, mode):
    isa = isa_avx2() if isa_name == 'avx2' else isa_avx512()
    ISA = isa_name.upper()
    suffix = '_dif_tw' if mode == 'dif' else '_tw'
    guard = f'FFT_RADIX25_{ISA}_IL{"_DIF_TW" if mode == "dif" else "_TW"}_H'

    L = [f'/**',
         f' * @file fft_radix25_{isa_name}_il{suffix}.h',
         f' * @brief DFT-25 {ISA} — native interleaved 5x5 CT + log3',
         f' * addsub/fmaddsub arithmetic, zero shuffle overhead.',
         f' * Generated by gen_r25_native_il.py',
         f' */', f'',
         f'#ifndef {guard}', f'#define {guard}',
         f'#include <stddef.h>', f'#include <immintrin.h>', f'']

    gen = gen_dit_tw if mode == 'dit' else gen_dif_tw
    for d in ('fwd', 'bwd'):
        L.extend(gen(isa, d))
        L.append('')

    L.append(f'#endif /* {guard} */')
    return L


if __name__ == '__main__':
    if len(sys.argv) < 3 or sys.argv[1] not in ('avx2','avx512') or sys.argv[2] not in ('dit','dif'):
        print("Usage: gen_r25_native_il.py <avx2|avx512> <dit|dif>", file=sys.stderr)
        sys.exit(1)
    print('\n'.join(gen_file(sys.argv[1], sys.argv[2])))
