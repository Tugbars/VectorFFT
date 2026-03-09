#!/usr/bin/env python3
"""
Generate R=8 twiddle-log3 codelets for all ISAs.
Load W^1, W^2, W^4 — derive W^3, W^5, W^6, W^7 in registers.
Outputs: DIT tw (fwd/bwd) and DIF tw (fwd/bwd) for scalar/avx2/avx512.
"""
import sys

def isa_config(isa):
    if isa == 'scalar':
        return dict(T='double', W=1, P='', attr='', guard='',
                    ld=lambda p: f'*({p})', st=lambda p,v: f'*({p}) = {v};')
    elif isa == 'avx2':
        return dict(T='__m256d', W=4, P='_mm256', 
                    attr='__attribute__((target("avx2,fma")))',
                    guard='__AVX2__',
                    ld=lambda p: f'_mm256_load_pd({p})',
                    st=lambda p,v: f'_mm256_store_pd({p},{v});')
    else:  # avx512
        return dict(T='__m512d', W=8, P='_mm512',
                    attr='__attribute__((target("avx512f,avx512dq,fma")))',
                    guard='__AVX512F__',
                    ld=lambda p: f'_mm512_load_pd({p})',
                    st=lambda p,v: f'_mm512_store_pd({p},{v});')

def ops(isa):
    if isa == 'scalar':
        return dict(
            add=lambda a,b: f'({a}+{b})', sub=lambda a,b: f'({a}-{b})',
            mul=lambda a,b: f'({a}*{b})',
            fma=lambda a,b,c: f'({a}*{b}+{c})', fms=lambda a,b,c: f'({a}*{b}-{c})',
            fnma=lambda a,b,c: f'({c}-{a}*{b})',
            set1=lambda v: v)
    P = '_mm256' if isa == 'avx2' else '_mm512'
    return dict(
        add=lambda a,b: f'{P}_add_pd({a},{b})', sub=lambda a,b: f'{P}_sub_pd({a},{b})',
        mul=lambda a,b: f'{P}_mul_pd({a},{b})',
        fma=lambda a,b,c: f'{P}_fmadd_pd({a},{b},{c})',
        fms=lambda a,b,c: f'{P}_fmsub_pd({a},{b},{c})',
        fnma=lambda a,b,c: f'{P}_fnmadd_pd({a},{b},{c})',
        set1=lambda v: f'{P}_set1_pd({v})')

def cmul_re(o, ar, ai, br, bi): return o['fms'](ar, br, o['mul'](ai, bi))
def cmul_im(o, ar, ai, br, bi): return o['fma'](ar, bi, o['mul'](ai, br))
def cmulc_re(o, ar, ai, br, bi): return o['fma'](ar, br, o['mul'](ai, bi))
def cmulc_im(o, ar, ai, br, bi): return o['fnma'](ar, bi, o['mul'](ai, br))

# ════════════════════════════════════════
# IL load/store helpers
# ════════════════════════════════════════

def il_load(L, cfg, isa, v, idx):
    """Load one radix slot from interleaved layout into v_re, v_im."""
    T = cfg['T']
    if isa == 'scalar':
        L.append(f'        const {T} {v}r = in[2*({idx}*K+k)], {v}i = in[2*({idx}*K+k)+1];')
    elif isa == 'avx2':
        L.append(f'        {{ __m256d _lo = _mm256_load_pd(&in[2*({idx}*K+k)]);')
        L.append(f'          __m256d _hi = _mm256_load_pd(&in[2*({idx}*K+k+2)]);')
        L.append(f'          {v}r = _mm256_permute4x64_pd(_mm256_shuffle_pd(_lo,_hi,0x0), 0xD8);')
        L.append(f'          {v}i = _mm256_permute4x64_pd(_mm256_shuffle_pd(_lo,_hi,0xF), 0xD8); }}')
    else:  # avx512
        L.append(f'        {{ __m512d _lo = _mm512_load_pd(&in[2*({idx}*K+k)]);')
        L.append(f'          __m512d _hi = _mm512_load_pd(&in[2*({idx}*K+k+4)]);')
        L.append(f'          {v}r = _mm512_permutexvar_pd(_mm512_set_epi64(7,5,3,1,6,4,2,0), _mm512_unpacklo_pd(_lo,_hi));')
        L.append(f'          {v}i = _mm512_permutexvar_pd(_mm512_set_epi64(7,5,3,1,6,4,2,0), _mm512_unpackhi_pd(_lo,_hi)); }}')

def il_store(L, cfg, isa, vr, vi, idx):
    """Store v_re, v_im to interleaved layout."""
    if isa == 'scalar':
        L.append(f'        out[2*({idx}*K+k)] = {vr}; out[2*({idx}*K+k)+1] = {vi};')
    elif isa == 'avx2':
        L.append(f'        {{ __m256d _rp = _mm256_permute4x64_pd({vr}, 0xD8);')
        L.append(f'          __m256d _ip = _mm256_permute4x64_pd({vi}, 0xD8);')
        L.append(f'          _mm256_store_pd(&out[2*({idx}*K+k)], _mm256_shuffle_pd(_rp,_ip,0x0));')
        L.append(f'          _mm256_store_pd(&out[2*({idx}*K+k+2)], _mm256_shuffle_pd(_rp,_ip,0xF)); }}')
    else:  # avx512
        L.append(f'        {{ __m512d _rp = _mm512_permutexvar_pd(_mm512_set_epi64(7,3,6,2,5,1,4,0), {vr});')
        L.append(f'          __m512d _ip = _mm512_permutexvar_pd(_mm512_set_epi64(7,3,6,2,5,1,4,0), {vi});')
        L.append(f'          _mm512_store_pd(&out[2*({idx}*K+k)], _mm512_unpacklo_pd(_rp,_ip));')
        L.append(f'          _mm512_store_pd(&out[2*({idx}*K+k+4)], _mm512_unpackhi_pd(_rp,_ip)); }}')

def gen_derive_twiddles(L, I, o, cfg):
    """Emit W^1,W^2,W^4 loads + W^3,W^5,W^6,W^7 derivation."""
    T = cfg['T']
    L.append(f'        /* Load 3 base twiddles: W^1, W^2, W^4 */')
    L.append(f'        const {T} tw1r = {cfg["ld"]("&tw_re[0*K+k]")}, tw1i = {cfg["ld"]("&tw_im[0*K+k]")};')
    L.append(f'        const {T} tw2r = {cfg["ld"]("&tw_re[1*K+k]")}, tw2i = {cfg["ld"]("&tw_im[1*K+k]")};')
    L.append(f'        const {T} tw4r = {cfg["ld"]("&tw_re[3*K+k]")}, tw4i = {cfg["ld"]("&tw_im[3*K+k]")};')
    L.append(f'        /* Derive W^3 = W^1 × W^2 */')
    L.append(f'        const {T} tw3r = {cmul_re(o,"tw1r","tw1i","tw2r","tw2i")};')
    L.append(f'        const {T} tw3i = {cmul_im(o,"tw1r","tw1i","tw2r","tw2i")};')
    L.append(f'        /* Derive W^5 = W^1 × W^4 */')
    L.append(f'        const {T} tw5r = {cmul_re(o,"tw1r","tw1i","tw4r","tw4i")};')
    L.append(f'        const {T} tw5i = {cmul_im(o,"tw1r","tw1i","tw4r","tw4i")};')
    L.append(f'        /* Derive W^6 = W^2 × W^4 */')
    L.append(f'        const {T} tw6r = {cmul_re(o,"tw2r","tw2i","tw4r","tw4i")};')
    L.append(f'        const {T} tw6i = {cmul_im(o,"tw2r","tw2i","tw4r","tw4i")};')
    L.append(f'        /* Derive W^7 = W^3 × W^4 */')
    L.append(f'        const {T} tw7r = {cmul_re(o,"tw3r","tw3i","tw4r","tw4i")};')
    L.append(f'        const {T} tw7i = {cmul_im(o,"tw3r","tw3i","tw4r","tw4i")};')

def gen_dit_tw(isa, direction, il=False):
    """DIT: twiddle inputs → butterfly → store."""
    cfg = isa_config(isa)
    o = ops(isa)
    T, W = cfg['T'], cfg['W']
    fwd = direction == 'fwd'
    cmul = (cmul_re, cmul_im) if fwd else (cmulc_re, cmulc_im)
    
    il_tag = '_il' if il else ''
    name = f'radix8_tw_dit_kernel_{direction}{il_tag}_{isa}'
    L = []
    if cfg['attr']: L.append(cfg['attr'])
    L.append(f'static inline void')
    L.append(f'{name}(')
    if il:
        L.append(f'    const double * __restrict__ in,')
        L.append(f'    double * __restrict__ out,')
    else:
        L.append(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
        L.append(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    L.append(f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,')
    L.append(f'    size_t K)')
    L.append(f'{{')
    L.append(f'    const {T} vc = {o["set1"]("0.707106781186547572737310929369414225220680236816")};')
    L.append(f'    const {T} vnc = {o["set1"]("-0.707106781186547572737310929369414225220680236816")};')
    L.append(f'    for (size_t k = 0; k < K; k += {W}) {{')
    
    gen_derive_twiddles(L, isa, o, cfg)
    
    # Load + twiddle inputs
    if il:
        L.append(f'        {T} x0r, x0i;')
        il_load(L, cfg, isa, 'x0', 0)
        for n in range(1, 8):
            L.append(f'        {T} r{n}r, r{n}i;')
            il_load(L, cfg, isa, f'r{n}', n)
            L.append(f'        const {T} x{n}r = {cmul[0](o,f"r{n}r",f"r{n}i",f"tw{n}r",f"tw{n}i")};')
            L.append(f'        const {T} x{n}i = {cmul[1](o,f"r{n}r",f"r{n}i",f"tw{n}r",f"tw{n}i")};')
    else:
        L.append(f'        const {T} x0r = {cfg["ld"]("&in_re[k]")}, x0i = {cfg["ld"]("&in_im[k]")};')
        for n in range(1, 8):
            L.append(f'        const {T} r{n}r = {cfg["ld"](f"&in_re[{n}*K+k]")}, r{n}i = {cfg["ld"](f"&in_im[{n}*K+k]")};')
            L.append(f'        const {T} x{n}r = {cmul[0](o,f"r{n}r",f"r{n}i",f"tw{n}r",f"tw{n}i")};')
            L.append(f'        const {T} x{n}i = {cmul[1](o,f"r{n}r",f"r{n}i",f"tw{n}r",f"tw{n}i")};')
    
    # DFT-8 butterfly
    emit_butterfly(L, o, cfg, fwd, il=il, isa=isa)
    
    L.append(f'    }}')
    L.append(f'}}')
    return L

def gen_dif_tw(isa, direction, il=False):
    """DIF: butterfly → twiddle outputs → store."""
    cfg = isa_config(isa)
    o = ops(isa)
    T, W = cfg['T'], cfg['W']
    fwd = direction == 'fwd'
    cmul = (cmul_re, cmul_im) if fwd else (cmulc_re, cmulc_im)
    
    il_tag = '_il' if il else ''
    name = f'radix8_tw_dif_kernel_{direction}{il_tag}_{isa}'
    L = []
    if cfg['attr']: L.append(cfg['attr'])
    L.append(f'static inline void')
    L.append(f'{name}(')
    if il:
        L.append(f'    const double * __restrict__ in,')
        L.append(f'    double * __restrict__ out,')
    else:
        L.append(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
        L.append(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    L.append(f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,')
    L.append(f'    size_t K)')
    L.append(f'{{')
    L.append(f'    const {T} vc = {o["set1"]("0.707106781186547572737310929369414225220680236816")};')
    L.append(f'    const {T} vnc = {o["set1"]("-0.707106781186547572737310929369414225220680236816")};')
    L.append(f'    for (size_t k = 0; k < K; k += {W}) {{')
    
    gen_derive_twiddles(L, isa, o, cfg)
    
    # Load inputs
    if il:
        for n in range(8):
            L.append(f'        {T} x{n}r, x{n}i;')
            il_load(L, cfg, isa, f'x{n}', n)
    else:
        for n in range(8):
            L.append(f'        const {T} x{n}r = {cfg["ld"](f"&in_re[{n}*K+k]")}, x{n}i = {cfg["ld"](f"&in_im[{n}*K+k]")};')
    
    # DFT-8 butterfly into Y vars
    emit_butterfly_to_Y(L, o, cfg, fwd)
    
    # Twiddle outputs and store
    if il:
        il_store(L, cfg, isa, 'Y0r', 'Y0i', 0)
        for n in range(1, 8):
            L.append(f'        {{ const {T} tr = {cmul[0](o,f"Y{n}r",f"Y{n}i",f"tw{n}r",f"tw{n}i")};')
            L.append(f'          const {T} ti = {cmul[1](o,f"Y{n}r",f"Y{n}i",f"tw{n}r",f"tw{n}i")};')
            il_store(L, cfg, isa, 'tr', 'ti', n)
            L.append(f'        }}')
    else:
        L.append(f'        {cfg["st"]("&out_re[0*K+k]","Y0r")} {cfg["st"]("&out_im[0*K+k]","Y0i")}')
        for n in range(1, 8):
            L.append(f'        {{ const {T} tr = {cmul[0](o,f"Y{n}r",f"Y{n}i",f"tw{n}r",f"tw{n}i")};')
            L.append(f'          const {T} ti = {cmul[1](o,f"Y{n}r",f"Y{n}i",f"tw{n}r",f"tw{n}i")};')
            L.append(f'          {cfg["st"](f"&out_re[{n}*K+k]","tr")} {cfg["st"](f"&out_im[{n}*K+k]","ti")} }}')
    
    L.append(f'    }}')
    L.append(f'}}')
    return L

def emit_butterfly(L, o, cfg, fwd, il=False, isa='scalar'):
    """DFT-8 butterfly: x0..x7 → output stores."""
    T = cfg['T']
    add, sub, mul = o['add'], o['sub'], o['mul']

    def ST(idx, vr, vi):
        if il:
            il_store(L, cfg, isa, vr, vi, idx)
        else:
            L.append(f'        {cfg["st"](f"&out_re[{idx}*K+k]",vr)} {cfg["st"](f"&out_im[{idx}*K+k]",vi)}')
    
    L.append(f'        /* DFT-4 evens */')
    L.append(f'        const {T} epr = {add("x0r","x4r")}, epi = {add("x0i","x4i")};')
    L.append(f'        const {T} eqr = {sub("x0r","x4r")}, eqi = {sub("x0i","x4i")};')
    L.append(f'        const {T} err = {add("x2r","x6r")}, eri = {add("x2i","x6i")};')
    L.append(f'        const {T} esr = {sub("x2r","x6r")}, esi = {sub("x2i","x6i")};')
    L.append(f'        const {T} A0r = {add("epr","err")}, A0i = {add("epi","eri")};')
    L.append(f'        const {T} A2r = {sub("epr","err")}, A2i = {sub("epi","eri")};')
    if fwd:
        L.append(f'        const {T} A1r = {add("eqr","esi")}, A1i = {sub("eqi","esr")};')
        L.append(f'        const {T} A3r = {sub("eqr","esi")}, A3i = {add("eqi","esr")};')
    else:
        L.append(f'        const {T} A1r = {sub("eqr","esi")}, A1i = {add("eqi","esr")};')
        L.append(f'        const {T} A3r = {add("eqr","esi")}, A3i = {sub("eqi","esr")};')
    
    L.append(f'        /* DFT-4 odds */')
    L.append(f'        const {T} opr = {add("x1r","x5r")}, opi = {add("x1i","x5i")};')
    L.append(f'        const {T} oqr = {sub("x1r","x5r")}, oqi = {sub("x1i","x5i")};')
    L.append(f'        const {T} orr = {add("x3r","x7r")}, ori = {add("x3i","x7i")};')
    L.append(f'        const {T} osr = {sub("x3r","x7r")}, osi = {sub("x3i","x7i")};')
    L.append(f'        const {T} B0r = {add("opr","orr")}, B0i = {add("opi","ori")};')
    L.append(f'        const {T} B2r = {sub("opr","orr")}, B2i = {sub("opi","ori")};')
    if fwd:
        L.append(f'        const {T} B1r = {add("oqr","osi")}, B1i = {sub("oqi","osr")};')
        L.append(f'        const {T} B3r = {sub("oqr","osi")}, B3i = {add("oqi","osr")};')
    else:
        L.append(f'        const {T} B1r = {sub("oqr","osi")}, B1i = {add("oqi","osr")};')
        L.append(f'        const {T} B3r = {add("oqr","osi")}, B3i = {sub("oqi","osr")};')
    
    L.append(f'        /* W8 combine */')
    ST(0, add("A0r","B0r"), add("A0i","B0i"))
    ST(4, sub("A0r","B0r"), sub("A0i","B0i"))
    if fwd:
        L.append(f'        const {T} t1r = {mul("vc",add("B1r","B1i"))}, t1i = {mul("vc",sub("B1i","B1r"))};')
    else:
        L.append(f'        const {T} t1r = {mul("vc",sub("B1r","B1i"))}, t1i = {mul("vc",add("B1r","B1i"))};')
    ST(1, add("A1r","t1r"), add("A1i","t1i"))
    ST(5, sub("A1r","t1r"), sub("A1i","t1i"))
    if fwd:
        ST(2, add("A2r","B2i"), sub("A2i","B2r"))
        ST(6, sub("A2r","B2i"), add("A2i","B2r"))
    else:
        ST(2, sub("A2r","B2i"), add("A2i","B2r"))
        ST(6, add("A2r","B2i"), sub("A2i","B2r"))
    if fwd:
        L.append(f'        const {T} t3r = {mul("vnc",sub("B3r","B3i"))}, t3i = {mul("vnc",add("B3r","B3i"))};')
    else:
        L.append(f'        const {T} t3r = {mul("vnc",add("B3r","B3i"))}, t3i = {mul("vc",sub("B3r","B3i"))};')
    ST(3, add("A3r","t3r"), add("A3i","t3i"))
    ST(7, sub("A3r","t3r"), sub("A3i","t3i"))

def emit_butterfly_to_Y(L, o, cfg, fwd):
    """DFT-8 butterfly: x0..x7 → Y0..Y7 variables (for DIF twiddle-after)."""
    T = cfg['T']
    add, sub, mul = o['add'], o['sub'], o['mul']
    
    L.append(f'        /* DFT-4 evens */')
    L.append(f'        const {T} epr = {add("x0r","x4r")}, epi = {add("x0i","x4i")};')
    L.append(f'        const {T} eqr = {sub("x0r","x4r")}, eqi = {sub("x0i","x4i")};')
    L.append(f'        const {T} err = {add("x2r","x6r")}, eri = {add("x2i","x6i")};')
    L.append(f'        const {T} esr = {sub("x2r","x6r")}, esi = {sub("x2i","x6i")};')
    L.append(f'        const {T} A0r = {add("epr","err")}, A0i = {add("epi","eri")};')
    L.append(f'        const {T} A2r = {sub("epr","err")}, A2i = {sub("epi","eri")};')
    if fwd:
        L.append(f'        const {T} A1r = {add("eqr","esi")}, A1i = {sub("eqi","esr")};')
        L.append(f'        const {T} A3r = {sub("eqr","esi")}, A3i = {add("eqi","esr")};')
    else:
        L.append(f'        const {T} A1r = {sub("eqr","esi")}, A1i = {add("eqi","esr")};')
        L.append(f'        const {T} A3r = {add("eqr","esi")}, A3i = {sub("eqi","esr")};')
    
    L.append(f'        /* DFT-4 odds */')
    L.append(f'        const {T} opr = {add("x1r","x5r")}, opi = {add("x1i","x5i")};')
    L.append(f'        const {T} oqr = {sub("x1r","x5r")}, oqi = {sub("x1i","x5i")};')
    L.append(f'        const {T} orr = {add("x3r","x7r")}, ori = {add("x3i","x7i")};')
    L.append(f'        const {T} osr = {sub("x3r","x7r")}, osi = {sub("x3i","x7i")};')
    L.append(f'        const {T} B0r = {add("opr","orr")}, B0i = {add("opi","ori")};')
    L.append(f'        const {T} B2r = {sub("opr","orr")}, B2i = {sub("opi","ori")};')
    if fwd:
        L.append(f'        const {T} B1r = {add("oqr","osi")}, B1i = {sub("oqi","osr")};')
        L.append(f'        const {T} B3r = {sub("oqr","osi")}, B3i = {add("oqi","osr")};')
    else:
        L.append(f'        const {T} B1r = {sub("oqr","osi")}, B1i = {add("oqi","osr")};')
        L.append(f'        const {T} B3r = {add("oqr","osi")}, B3i = {sub("oqi","osr")};')
    
    L.append(f'        /* W8 combine into Y vars */')
    L.append(f'        const {T} Y0r = {add("A0r","B0r")}, Y0i = {add("A0i","B0i")};')
    L.append(f'        const {T} Y4r = {sub("A0r","B0r")}, Y4i = {sub("A0i","B0i")};')
    if fwd:
        L.append(f'        const {T} t1r = {mul("vc",add("B1r","B1i"))}, t1i = {mul("vc",sub("B1i","B1r"))};')
    else:
        L.append(f'        const {T} t1r = {mul("vc",sub("B1r","B1i"))}, t1i = {mul("vc",add("B1r","B1i"))};')
    L.append(f'        const {T} Y1r = {add("A1r","t1r")}, Y1i = {add("A1i","t1i")};')
    L.append(f'        const {T} Y5r = {sub("A1r","t1r")}, Y5i = {sub("A1i","t1i")};')
    if fwd:
        L.append(f'        const {T} Y2r = {add("A2r","B2i")}, Y2i = {sub("A2i","B2r")};')
        L.append(f'        const {T} Y6r = {sub("A2r","B2i")}, Y6i = {add("A2i","B2r")};')
    else:
        L.append(f'        const {T} Y2r = {sub("A2r","B2i")}, Y2i = {add("A2i","B2r")};')
        L.append(f'        const {T} Y6r = {add("A2r","B2i")}, Y6i = {sub("A2i","B2r")};')
    if fwd:
        L.append(f'        const {T} t3r = {mul("vnc",sub("B3r","B3i"))}, t3i = {mul("vnc",add("B3r","B3i"))};')
    else:
        L.append(f'        const {T} t3r = {mul("vnc",add("B3r","B3i"))}, t3i = {mul("vc",sub("B3r","B3i"))};')
    L.append(f'        const {T} Y3r = {add("A3r","t3r")}, Y3i = {add("A3i","t3i")};')
    L.append(f'        const {T} Y7r = {sub("A3r","t3r")}, Y7i = {sub("A3i","t3i")};')

def gen_notw(isa, direction, il=False):
    """Generate notw butterfly."""
    cfg = isa_config(isa)
    o = ops(isa)
    T, W = cfg['T'], cfg['W']
    fwd = direction == 'fwd'
    
    il_tag = '_il' if il else ''
    name = f'radix8_notw_dit_kernel_{direction}{il_tag}_{isa}'
    L = []
    if cfg['attr']: L.append(cfg['attr'])
    L.append(f'static inline void')
    L.append(f'{name}(')
    if il:
        L.append(f'    const double * __restrict__ in,')
        L.append(f'    double * __restrict__ out,')
    else:
        L.append(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
        L.append(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    L.append(f'    size_t K)')
    L.append(f'{{')
    L.append(f'    const {T} vc = {o["set1"]("0.707106781186547572737310929369414225220680236816")};')
    L.append(f'    const {T} vnc = {o["set1"]("-0.707106781186547572737310929369414225220680236816")};')
    L.append(f'    for (size_t k = 0; k < K; k += {W}) {{')
    if il:
        for n in range(8):
            L.append(f'        {T} x{n}r, x{n}i;')
            il_load(L, cfg, isa, f'x{n}', n)
    else:
        for n in range(8):
            L.append(f'        const {T} x{n}r = {cfg["ld"](f"&in_re[{n}*K+k]")}, x{n}i = {cfg["ld"](f"&in_im[{n}*K+k]")};')
    emit_butterfly(L, o, cfg, fwd, il=il, isa=isa)
    L.append(f'    }}')
    L.append(f'}}')
    return L

def gen_dit_file(isa, il=False):
    """Generate complete DIT file."""
    ISA = isa.upper()
    il_tag = '_il' if il else ''
    guard = f'FFT_RADIX8_{ISA}{il_tag.upper()}_H'
    layout = 'interleaved' if il else 'split re/im'
    
    L = []
    L.append(f'/**')
    L.append(f' * @file fft_radix8_{isa}{il_tag}.h')
    L.append(f' * @brief DFT-8 {ISA} codelets — {layout} + log3 twiddles')
    L.append(f' * Generated by gen_r8_log3.py')
    L.append(f' */')
    L.append(f'#ifndef {guard}')
    L.append(f'#define {guard}')
    L.append(f'#include <stddef.h>')
    if isa != 'scalar':
        L.append(f'#include <immintrin.h>')
    L.append(f'')
    
    for direction in ['fwd', 'bwd']:
        L.extend(gen_notw(isa, direction, il=il))
        L.append('')
    
    for direction in ['fwd', 'bwd']:
        L.extend(gen_dit_tw(isa, direction, il=il))
        L.append('')
    
    L.append(f'#endif /* {guard} */')
    return L

def gen_dif_file(isa, il=False):
    """Generate DIF tw file."""
    ISA = isa.upper()
    il_tag = '_il' if il else ''
    guard = f'FFT_RADIX8_{ISA}{il_tag.upper()}_DIF_TW_H'
    layout = 'interleaved' if il else 'split re/im'
    
    L = []
    L.append(f'/**')
    L.append(f' * @file fft_radix8_{isa}{il_tag}_dif_tw.h')
    L.append(f' * @brief DFT-8 {ISA} DIF tw — {layout} + log3')
    L.append(f' * Generated by gen_r8_log3.py')
    L.append(f' */')
    L.append(f'#ifndef {guard}')
    L.append(f'#define {guard}')
    L.append(f'#include <stddef.h>')
    if isa != 'scalar':
        L.append(f'#include <immintrin.h>')
    L.append(f'')
    
    for direction in ['fwd', 'bwd']:
        L.extend(gen_dif_tw(isa, direction, il=il))
        L.append('')
    
    L.append(f'#endif /* {guard} */')
    return L

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: gen_r8_log3.py <scalar|avx2|avx512> <dit|dif> [il]", file=sys.stderr)
        sys.exit(1)
    isa, mode = sys.argv[1], sys.argv[2]
    il = len(sys.argv) > 3 and sys.argv[3] == 'il'
    if mode == 'dit':
        lines = gen_dit_file(isa, il=il)
    else:
        lines = gen_dif_file(isa, il=il)
    print('\n'.join(lines))
