#!/usr/bin/env python3
"""
gen_radix32_avx512_tw_ladder_v2.py — Improved radix-32 AVX-512 codelet generator

Improvements over v1:
  #2. Fused last sub-FFT: keep first 4 of n2=3 results in saved regs,
      skip 4 spills + 4 reloads per k-step (saves 16 mem ops)
  #3. U2 spill reuse: A-complete then B-complete with shared 32-slot
      spill buffer instead of 64-slot (halves spill footprint)
"""

import math, sys

N, N1, N2 = 32, 8, 4
NFUSE = 4  # how many last-sub-FFT results to keep live in registers

def wN(e, tN):
    e = e % tN
    a = 2.0 * math.pi * e / tN
    return (math.cos(a), -math.sin(a))

def wN_label(e, tN):
    return f"W{tN}_{e % tN}"

def twiddle_is_trivial(e, tN):
    e = e % tN
    if e == 0: return True, 'one'
    if (8*e) % tN == 0:
        o = (8*e)//tN
        t = ['one','w8_1','neg_j','w8_3','neg_one','neg_w8_1','pos_j','neg_w8_3']
        return True, t[o%8]
    return False, 'cmul'

def collect_internal_twiddles():
    tw = set()
    for n2 in range(1,N2):
        for k1 in range(1,N1):
            e=(n2*k1)%N
            _,t=twiddle_is_trivial(e,N)
            if t=='cmul': tw.add((e,N))
    return tw

class E:
    def __init__(self):
        self.L=[]; self.ind=1; self.spill_c=0; self.reload_c=0
    def o(s,t=""): s.L.append("    "*s.ind+t)
    def c(s,t): s.o(f"/* {t} */")
    def b(s): s.L.append("")
    def add(s,a,b): return f"_mm512_add_pd({a},{b})"
    def sub(s,a,b): return f"_mm512_sub_pd({a},{b})"
    def mul(s,a,b): return f"_mm512_mul_pd({a},{b})"
    def neg(s,a):   return f"_mm512_xor_pd({a},sign_flip)"
    def fma(s,a,b,c): return f"_mm512_fmadd_pd({a},{b},{c})"
    def fms(s,a,b,c): return f"_mm512_fmsub_pd({a},{b},{c})"

    def emit_load(s,v,n,k_expr="k"):
        s.o(f"{v}_re = LD(&in_re[{n}*K+{k_expr}]);")
        s.o(f"{v}_im = LD(&in_im[{n}*K+{k_expr}]);")
    def emit_store(s,v,m,k_expr="k"):
        s.o(f"ST(&out_re[{m}*K+{k_expr}],{v}_re);")
        s.o(f"ST(&out_im[{m}*K+{k_expr}],{v}_im);")
    def emit_spill(s,v,slot):
        s.o(f"_mm512_store_pd(&spill_re[{slot}*8],{v}_re);")
        s.o(f"_mm512_store_pd(&spill_im[{slot}*8],{v}_im);")
        s.spill_c+=1
    def emit_reload(s,v,slot):
        s.o(f"{v}_re = _mm512_load_pd(&spill_re[{slot}*8]);")
        s.o(f"{v}_im = _mm512_load_pd(&spill_im[{slot}*8]);")
        s.reload_c+=1
    def emit_cmul(s, dst_r, dst_i, ar, ai, br, bi, d):
        fwd = (d=='fwd')
        if fwd:
            s.o(f"{dst_r} = {s.fms(ar,br,s.mul(ai,bi))};")
            s.o(f"{dst_i} = {s.fma(ar,bi,s.mul(ai,br))};")
        else:
            s.o(f"{dst_r} = {s.fma(ar,br,s.mul(ai,bi))};")
            s.o(f"{dst_i} = {s.fms(ai,br,s.mul(ar,bi))};")
    def emit_cmul_inplace(s, v, wr, wi, d):
        fwd = (d=='fwd')
        s.o(f"{{ __m512d tr={v}_re;")
        if fwd:
            s.o(f"  {v}_re={s.fms(f'{v}_re',wr,s.mul(f'{v}_im',wi))};")
            s.o(f"  {v}_im={s.fma('tr',wi,s.mul(f'{v}_im',wr))}; }}")
        else:
            s.o(f"  {v}_re={s.fma(f'{v}_re',wr,s.mul(f'{v}_im',wi))};")
            s.o(f"  {v}_im={s.fms(f'{v}_im',wr,s.mul('tr',wi))}; }}")

    def emit_radix8(s,v,d,label=""):
        fwd=(d=='fwd')
        if label: s.c(f"{label} [{d}]")
        s.o(f"{{ __m512d e0r,e0i,e1r,e1i,e2r,e2i,e3r,e3i;")
        s.o(f"  __m512d t0r,t0i,t1r,t1i,t2r,t2i,t3r,t3i;")
        s.o(f"  t0r={s.add(f'{v[0]}_re',f'{v[4]}_re')}; t0i={s.add(f'{v[0]}_im',f'{v[4]}_im')};")
        s.o(f"  t1r={s.sub(f'{v[0]}_re',f'{v[4]}_re')}; t1i={s.sub(f'{v[0]}_im',f'{v[4]}_im')};")
        s.o(f"  t2r={s.add(f'{v[2]}_re',f'{v[6]}_re')}; t2i={s.add(f'{v[2]}_im',f'{v[6]}_im')};")
        s.o(f"  t3r={s.sub(f'{v[2]}_re',f'{v[6]}_re')}; t3i={s.sub(f'{v[2]}_im',f'{v[6]}_im')};")
        s.o(f"  e0r={s.add('t0r','t2r')}; e0i={s.add('t0i','t2i')};")
        s.o(f"  e2r={s.sub('t0r','t2r')}; e2i={s.sub('t0i','t2i')};")
        j_add, j_sub = ('add','sub') if fwd else ('sub','add')
        s.o(f"  e1r={getattr(s,j_add)('t1r','t3i')}; e1i={getattr(s,j_sub)('t1i','t3r')};")
        s.o(f"  e3r={getattr(s,j_sub)('t1r','t3i')}; e3i={getattr(s,j_add)('t1i','t3r')};")
        s.o(f"  __m512d o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;")
        s.o(f"  t0r={s.add(f'{v[1]}_re',f'{v[5]}_re')}; t0i={s.add(f'{v[1]}_im',f'{v[5]}_im')};")
        s.o(f"  t1r={s.sub(f'{v[1]}_re',f'{v[5]}_re')}; t1i={s.sub(f'{v[1]}_im',f'{v[5]}_im')};")
        s.o(f"  t2r={s.add(f'{v[3]}_re',f'{v[7]}_re')}; t2i={s.add(f'{v[3]}_im',f'{v[7]}_im')};")
        s.o(f"  t3r={s.sub(f'{v[3]}_re',f'{v[7]}_re')}; t3i={s.sub(f'{v[3]}_im',f'{v[7]}_im')};")
        s.o(f"  o0r={s.add('t0r','t2r')}; o0i={s.add('t0i','t2i')};")
        s.o(f"  o2r={s.sub('t0r','t2r')}; o2i={s.sub('t0i','t2i')};")
        s.o(f"  o1r={getattr(s,j_add)('t1r','t3i')}; o1i={getattr(s,j_sub)('t1i','t3r')};")
        s.o(f"  o3r={getattr(s,j_sub)('t1r','t3i')}; o3i={getattr(s,j_add)('t1i','t3r')};")
        if fwd:
            s.o(f"  t0r={s.mul(s.add('o1r','o1i'),'sqrt2_inv')};")
            s.o(f"  t0i={s.mul(s.sub('o1i','o1r'),'sqrt2_inv')};")
            s.o(f"  o1r=t0r; o1i=t0i;")
            s.o(f"  t0r=o2i; t0i={s.neg('o2r')};")
            s.o(f"  o2r=t0r; o2i=t0i;")
            s.o(f"  t0r={s.mul(s.sub('o3i','o3r'),'sqrt2_inv')};")
            s.o(f"  t0i={s.neg(s.mul(s.add('o3r','o3i'),'sqrt2_inv'))};")
            s.o(f"  o3r=t0r; o3i=t0i;")
        else:
            s.o(f"  t0r={s.mul(s.sub('o1r','o1i'),'sqrt2_inv')};")
            s.o(f"  t0i={s.mul(s.add('o1r','o1i'),'sqrt2_inv')};")
            s.o(f"  o1r=t0r; o1i=t0i;")
            s.o(f"  t0r={s.neg('o2i')}; t0i=o2r;")
            s.o(f"  o2r=t0r; o2i=t0i;")
            s.o(f"  t0r={s.neg(s.mul(s.add('o3r','o3i'),'sqrt2_inv'))};")
            s.o(f"  t0i={s.mul(s.sub('o3r','o3i'),'sqrt2_inv')};")
            s.o(f"  o3r=t0r; o3i=t0i;")
        for i,(j) in [(0,4),(1,5),(2,6),(3,7)]:
            s.o(f"  {v[i]}_re={s.add(f'e{i}r',f'o{i}r')}; {v[i]}_im={s.add(f'e{i}i',f'o{i}i')};")
            s.o(f"  {v[j]}_re={s.sub(f'e{i}r',f'o{i}r')}; {v[j]}_im={s.sub(f'e{i}i',f'o{i}i')};")
        s.o(f"}}")

    def emit_radix4(s,v,d,label=""):
        fwd=(d=='fwd')
        if label: s.c(f"{label} [{d}]")
        a,b,c,dd=v[0],v[1],v[2],v[3]
        s.o(f"{{ __m512d t0r,t0i,t1r,t1i,t2r,t2i,t3r,t3i;")
        s.o(f"  t0r={s.add(f'{a}_re',f'{c}_re')}; t0i={s.add(f'{a}_im',f'{c}_im')};")
        s.o(f"  t1r={s.sub(f'{a}_re',f'{c}_re')}; t1i={s.sub(f'{a}_im',f'{c}_im')};")
        s.o(f"  t2r={s.add(f'{b}_re',f'{dd}_re')}; t2i={s.add(f'{b}_im',f'{dd}_im')};")
        s.o(f"  t3r={s.sub(f'{b}_re',f'{dd}_re')}; t3i={s.sub(f'{b}_im',f'{dd}_im')};")
        s.o(f"  {a}_re={s.add('t0r','t2r')}; {a}_im={s.add('t0i','t2i')};")
        s.o(f"  {c}_re={s.sub('t0r','t2r')}; {c}_im={s.sub('t0i','t2i')};")
        if fwd:
            s.o(f"  {b}_re={s.add('t1r','t3i')}; {b}_im={s.sub('t1i','t3r')};")
            s.o(f"  {dd}_re={s.sub('t1r','t3i')}; {dd}_im={s.add('t1i','t3r')};")
        else:
            s.o(f"  {b}_re={s.sub('t1r','t3i')}; {b}_im={s.add('t1i','t3r')};")
            s.o(f"  {dd}_re={s.add('t1r','t3i')}; {dd}_im={s.sub('t1i','t3r')};")
        s.o(f"}}")

    def emit_twiddle(s,dst,src,e,tN,d):
        _,typ=twiddle_is_trivial(e,tN)
        fwd=(d=='fwd')
        if typ=='one':
            if dst!=src: s.o(f"{dst}_re={src}_re; {dst}_im={src}_im;")
        elif typ=='neg_one':
            s.o(f"{dst}_re={s.neg(f'{src}_re')}; {dst}_im={s.neg(f'{src}_im')};")
        elif typ=='neg_j':
            if fwd: s.o(f"{{ __m512d t={src}_re; {dst}_re={src}_im; {dst}_im={s.neg('t')}; }}")
            else:   s.o(f"{{ __m512d t={src}_re; {dst}_re={s.neg(f'{src}_im')}; {dst}_im=t; }}")
        elif typ=='pos_j':
            if fwd: s.o(f"{{ __m512d t={src}_re; {dst}_re={s.neg(f'{src}_im')}; {dst}_im=t; }}")
            else:   s.o(f"{{ __m512d t={src}_re; {dst}_re={src}_im; {dst}_im={s.neg('t')}; }}")
        elif typ=='w8_1':
            s.o(f"{{ __m512d tr={src}_re,ti={src}_im;")
            if fwd: s.o(f"  {dst}_re={s.mul(s.add('tr','ti'),'sqrt2_inv')}; {dst}_im={s.mul(s.sub('ti','tr'),'sqrt2_inv')}; }}")
            else:   s.o(f"  {dst}_re={s.mul(s.sub('tr','ti'),'sqrt2_inv')}; {dst}_im={s.mul(s.add('tr','ti'),'sqrt2_inv')}; }}")
        elif typ=='w8_3':
            s.o(f"{{ __m512d tr={src}_re,ti={src}_im;")
            if fwd: s.o(f"  {dst}_re={s.mul(s.sub('ti','tr'),'sqrt2_inv')}; {dst}_im={s.neg(s.mul(s.add('tr','ti'),'sqrt2_inv'))}; }}")
            else:   s.o(f"  {dst}_re={s.neg(s.mul(s.add('tr','ti'),'sqrt2_inv'))}; {dst}_im={s.mul(s.sub('tr','ti'),'sqrt2_inv')}; }}")
        elif typ=='neg_w8_1':
            s.o(f"{{ __m512d tr={src}_re,ti={src}_im;")
            if fwd: s.o(f"  {dst}_re={s.neg(s.mul(s.add('tr','ti'),'sqrt2_inv'))}; {dst}_im={s.mul(s.sub('tr','ti'),'sqrt2_inv')}; }}")
            else:   s.o(f"  {dst}_re={s.mul(s.sub('ti','tr'),'sqrt2_inv')}; {dst}_im={s.neg(s.mul(s.add('tr','ti'),'sqrt2_inv'))}; }}")
        elif typ=='neg_w8_3':
            s.o(f"{{ __m512d tr={src}_re,ti={src}_im;")
            if fwd: s.o(f"  {dst}_re={s.mul(s.sub('tr','ti'),'sqrt2_inv')}; {dst}_im={s.mul(s.add('tr','ti'),'sqrt2_inv')}; }}")
            else:   s.o(f"  {dst}_re={s.mul(s.add('tr','ti'),'sqrt2_inv')}; {dst}_im={s.mul(s.sub('ti','tr'),'sqrt2_inv')}; }}")
        else:
            label=wN_label(e,tN)
            s.o(f"{{ __m512d tr={src}_re;")
            if fwd:
                s.o(f"  {dst}_re={s.fms(f'{src}_re',f'tw_{label}_re',s.mul(f'{src}_im',f'tw_{label}_im'))};")
                s.o(f"  {dst}_im={s.fma('tr',f'tw_{label}_im',s.mul(f'{src}_im',f'tw_{label}_re'))}; }}")
            else:
                s.o(f"  {dst}_re={s.fma(f'{src}_re',f'tw_{label}_re',s.mul(f'{src}_im',f'tw_{label}_im'))};")
                s.o(f"  {dst}_im={s.fms(f'{src}_im',f'tw_{label}_re',s.mul('tr',f'tw_{label}_im'))}; }}")


ROW_CHAIN = [
    (0, None, False), (1, None, False), (2, None, False),
    (3, ('b4','b8'), True), (4, None, False),
    (5, ('b4','b16'), False), (6, ('b8','b16'), False), (7, ('b12','b16'), False),
]
BASE_MAP = {'b1':0, 'b2':1, 'b4':2, 'b8':3, 'b16':4}
COL_TW = {0: None, 1: 'b1', 2: 'b2', 3: 'b3'}


def emit_ladder_kernel(em, d, itw_set, u):
    suffix = f"_{d}_avx512_u{u}"
    k_step = 8 * u
    spill_slots = 32  # IMPROVEMENT #3: shared for U2

    em.L.append(f"static __attribute__((target(\"avx512f,avx512dq,fma\"))) void")
    em.L.append(f"radix32_tw_ladder_dit_kernel{suffix}(")
    em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
    em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
    em.L.append(f"    const double * __restrict__ base_tw_re, const double * __restrict__ base_tw_im,")
    em.L.append(f"    size_t K)")
    em.L.append(f"{{")
    em.ind = 1; em.spill_c = 0; em.reload_c = 0

    em.o("const __m512d sign_flip = _mm512_set1_pd(-0.0);")
    em.o("const __m512d sqrt2_inv = _mm512_set1_pd(0.70710678118654752440);")
    em.b()
    em.o(f"__attribute__((aligned(64))) double spill_re[{spill_slots*8}];")
    em.o(f"__attribute__((aligned(64))) double spill_im[{spill_slots*8}];")
    em.b()
    em.o("__m512d x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
    em.o("__m512d x4_re,x4_im,x5_re,x5_im,x6_re,x6_im,x7_re,x7_im;")
    em.o("__m512d s0_re,s0_im,s1_re,s1_im,s2_re,s2_im,s3_re,s3_im;")
    em.b()

    if itw_set:
        em.c(f"Hoisted internal W32 broadcasts [{d}]")
        for (e,tN) in sorted(itw_set):
            label=wN_label(e,tN)
            em.o(f"const __m512d tw_{label}_re = _mm512_set1_pd({label}_re);")
            em.o(f"const __m512d tw_{label}_im = _mm512_set1_pd({label}_im);")
        em.b()

    em.o(f"for (size_t k = 0; k < K; k += {k_step}) {{")
    em.ind += 1

    xv8 = [f"x{i}" for i in range(8)]
    xv4 = [f"x{i}" for i in range(4)]

    def emit_one_pipeline(em, pipe, k_expr, spill_base, d):
        em.c(f"{pipe}Load 5 base twiddle pairs")
        for name, idx in sorted(BASE_MAP.items(), key=lambda x:x[1]):
            em.o(f"const __m512d {pipe}{name}_re = _mm512_load_pd(&base_tw_re[{idx}*K+{k_expr}]);")
            em.o(f"const __m512d {pipe}{name}_im = _mm512_load_pd(&base_tw_im[{idx}*K+{k_expr}]);")
        em.b()
        em.c(f"{pipe}Derive b3 = b1*b2")
        em.o(f"__m512d {pipe}b3_re, {pipe}b3_im;")
        em.emit_cmul(f"{pipe}b3_re", f"{pipe}b3_im",
                     f"{pipe}b1_re", f"{pipe}b1_im",
                     f"{pipe}b2_re", f"{pipe}b2_im", d)
        em.b()
        em.c(f"{pipe}Derived row twiddle scratch")
        em.o(f"__m512d {pipe}r3_re,{pipe}r3_im, {pipe}r5_re,{pipe}r5_im, {pipe}r6_re,{pipe}r6_im, {pipe}r7_re,{pipe}r7_im;")
        em.o(f"__m512d {pipe}b12_re, {pipe}b12_im;")
        em.b()

        for n2 in range(N2):
            is_last = (n2 == N2 - 1)
            em.c(f"{pipe}sub-FFT n2={n2}")
            for n1, derive, keep in ROW_CHAIN:
                n = N2*n1 + n2
                em.emit_load(f"x{n1}", n, k_expr)
                if n1 > 0:
                    if derive is None:
                        base_name = {1:'b4', 2:'b8', 4:'b16'}[n1]
                        wr, wi = f"{pipe}{base_name}_re", f"{pipe}{base_name}_im"
                    else:
                        a_name, b_name = derive
                        tw_name = f"r{n1}"
                        em.emit_cmul(f"{pipe}{tw_name}_re", f"{pipe}{tw_name}_im",
                                     f"{pipe}{a_name}_re", f"{pipe}{a_name}_im",
                                     f"{pipe}{b_name}_re", f"{pipe}{b_name}_im", d)
                        wr, wi = f"{pipe}{tw_name}_re", f"{pipe}{tw_name}_im"
                        if keep:
                            em.o(f"{pipe}b12_re = {wr}; {pipe}b12_im = {wi};")
                    em.emit_cmul_inplace(f"x{n1}", wr, wi, d)
            em.b()
            em.emit_radix8(xv8, d, f"{pipe}radix-8 n2={n2}")
            em.b()
            col = COL_TW[n2]
            if col is not None:
                em.c(f"{pipe}col twiddle {col}")
                wr, wi = f"{pipe}{col}_re", f"{pipe}{col}_im"
                for k1 in range(N1):
                    em.emit_cmul_inplace(f"x{k1}", wr, wi, d)
                em.b()

            if is_last:
                em.c(f"{pipe}FUSED: save x0..x{NFUSE-1} in s-regs, spill x{NFUSE}..x{N1-1}")
                for k1 in range(NFUSE):
                    em.o(f"s{k1}_re = x{k1}_re; s{k1}_im = x{k1}_im;")
                for k1 in range(NFUSE, N1):
                    em.emit_spill(f"x{k1}", spill_base + n2*N1 + k1)
            else:
                for k1 in range(N1):
                    em.emit_spill(f"x{k1}", spill_base + n2*N1 + k1)
            em.b()

    def emit_pass2(em, pipe, k_expr, spill_base, d):
        last_n2 = N2 - 1
        for k1 in range(N1):
            em.c(f"{pipe}column k1={k1}")
            if k1 < NFUSE:
                for n2 in range(last_n2):
                    em.emit_reload(f"x{n2}", spill_base + n2*N1 + k1)
                em.o(f"x{last_n2}_re = s{k1}_re; x{last_n2}_im = s{k1}_im;")
            else:
                for n2 in range(N2):
                    em.emit_reload(f"x{n2}", spill_base + n2*N1 + k1)
            em.b()
            if k1 > 0:
                for n2 in range(1,N2):
                    e=(n2*k1)%N
                    em.emit_twiddle(f"x{n2}",f"x{n2}",e,N,d)
                em.b()
            em.emit_radix4(xv4,d,f"{pipe}radix-4 k1={k1}")
            em.b()
            for k2 in range(N2):
                em.emit_store(f"x{k2}", k1+N1*k2, k_expr)
            em.b()

    if u == 1:
        emit_one_pipeline(em, "", "k", 0, d)
        em.c(f"PASS 2 [{d}]"); em.b()
        emit_pass2(em, "", "k", 0, d)
    else:
        # IMPROVEMENT #3: A-complete then B-complete, shared spill buffer
        em.c(f"Pipeline A [{d}]")
        emit_one_pipeline(em, "A_", "k", 0, d)
        emit_pass2(em, "A_", "k", 0, d)
        em.c(f"Pipeline B [{d}]")
        emit_one_pipeline(em, "B_", "k+8", 0, d)  # reuse slot 0
        emit_pass2(em, "B_", "k+8", 0, d)

    em.ind -= 1
    em.o("}")
    em.L.append("}"); em.L.append("")
    return em.spill_c, em.reload_c


def emit_flat_kernel(em, d, itw_set):
    em.L.append(f"static __attribute__((target(\"avx512f,avx512dq,fma\"))) void")
    em.L.append(f"radix32_tw_flat_dit_kernel_{d}_avx512(")
    em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
    em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
    em.L.append(f"    const double * __restrict__ tw_re, const double * __restrict__ tw_im,")
    em.L.append(f"    size_t K)")
    em.L.append(f"{{")
    em.ind = 1; em.spill_c = 0; em.reload_c = 0

    em.o("const __m512d sign_flip = _mm512_set1_pd(-0.0);")
    em.o("const __m512d sqrt2_inv = _mm512_set1_pd(0.70710678118654752440);")
    em.b()
    em.o("__attribute__((aligned(64))) double spill_re[256];")
    em.o("__attribute__((aligned(64))) double spill_im[256];")
    em.b()
    em.o("__m512d x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
    em.o("__m512d x4_re,x4_im,x5_re,x5_im,x6_re,x6_im,x7_re,x7_im;")
    em.o("__m512d s0_re,s0_im,s1_re,s1_im,s2_re,s2_im,s3_re,s3_im;")
    em.b()

    if itw_set:
        em.c(f"Hoisted internal W32 broadcasts")
        for (e,tN) in sorted(itw_set):
            label=wN_label(e,tN)
            em.o(f"const __m512d tw_{label}_re = _mm512_set1_pd({label}_re);")
            em.o(f"const __m512d tw_{label}_im = _mm512_set1_pd({label}_im);")
        em.b()

    em.o("for (size_t k = 0; k < K; k += 8) {")
    em.ind += 1
    xv8 = [f"x{i}" for i in range(8)]
    xv4 = [f"x{i}" for i in range(4)]
    fwd = (d == 'fwd')
    last_n2 = N2 - 1

    for n2 in range(N2):
        is_last = (n2 == last_n2)
        em.c(f"sub-FFT n2={n2}")
        for n1 in range(N1):
            n = N2*n1 + n2
            em.emit_load(f"x{n1}", n)
            if n > 0:
                em.o(f"{{ __m512d wr = _mm512_load_pd(&tw_re[{n-1}*K+k]);")
                em.o(f"  __m512d wi = _mm512_load_pd(&tw_im[{n-1}*K+k]);")
                em.o(f"  __m512d tr = x{n1}_re;")
                if fwd:
                    em.o(f"  x{n1}_re = {em.fms(f'x{n1}_re','wr',em.mul(f'x{n1}_im','wi'))};")
                    em.o(f"  x{n1}_im = {em.fma('tr','wi',em.mul(f'x{n1}_im','wr'))}; }}")
                else:
                    em.o(f"  x{n1}_re = {em.fma(f'x{n1}_re','wr',em.mul(f'x{n1}_im','wi'))};")
                    em.o(f"  x{n1}_im = {em.fms(f'x{n1}_im','wr',em.mul('tr','wi'))}; }}")
        em.b()
        em.emit_radix8(xv8, d, f"radix-8 n2={n2}")
        em.b()
        if is_last:
            em.c(f"FUSED: save x0..x{NFUSE-1} in s-regs, spill x{NFUSE}..x{N1-1}")
            for k1 in range(NFUSE):
                em.o(f"s{k1}_re = x{k1}_re; s{k1}_im = x{k1}_im;")
            for k1 in range(NFUSE, N1):
                em.emit_spill(f"x{k1}", n2*N1+k1)
        else:
            for k1 in range(N1):
                em.emit_spill(f"x{k1}", n2*N1+k1)
        em.b()

    em.c(f"PASS 2"); em.b()
    for k1 in range(N1):
        em.c(f"column k1={k1}")
        if k1 < NFUSE:
            for n2 in range(last_n2):
                em.emit_reload(f"x{n2}", n2*N1+k1)
            em.o(f"x{last_n2}_re = s{k1}_re; x{last_n2}_im = s{k1}_im;")
        else:
            for n2 in range(N2):
                em.emit_reload(f"x{n2}", n2*N1+k1)
        em.b()
        if k1 > 0:
            for n2 in range(1,N2):
                e=(n2*k1)%N
                em.emit_twiddle(f"x{n2}",f"x{n2}",e,N,d)
            em.b()
        em.emit_radix4(xv4,d,f"radix-4 k1={k1}")
        em.b()
        for k2 in range(N2):
            em.emit_store(f"x{k2}", k1+N1*k2)
        em.b()

    em.ind -= 1
    em.o("}")
    em.L.append("}"); em.L.append("")
    return em.spill_c, em.reload_c


def main():
    itw_set = collect_internal_twiddles()
    em = E()
    em.L.append("/**"); em.L.append(" * @file fft_radix32_avx512_tw_ladder.h")
    em.L.append(" * @brief DFT-32 AVX-512 twiddled v2: fused spill + U2 reuse")
    em.L.append(" * Generated by gen_radix32_v2.py"); em.L.append(" */")
    em.L.append(""); em.L.append("#ifndef FFT_RADIX32_AVX512_TW_LADDER_H")
    em.L.append("#define FFT_RADIX32_AVX512_TW_LADDER_H")
    em.L.append(""); em.L.append("#include <immintrin.h>"); em.L.append("")

    by_tN = {}
    for (e,tN) in sorted(itw_set): by_tN.setdefault(tN,[]).append(e)
    for tN in sorted(by_tN):
        g = f"FFT_W{tN}_TWIDDLES_DEFINED"
        em.L.append(f"#ifndef {g}"); em.L.append(f"#define {g}")
        for e in sorted(by_tN[tN]):
            wr,wi=wN(e,tN); l=wN_label(e,tN)
            em.L.append(f"static const double {l}_re = {wr:.20e};")
            em.L.append(f"static const double {l}_im = {wi:.20e};")
        em.L.append(f"#endif"); em.L.append("")

    em.L.append("#ifndef R32L_LD"); em.L.append("#define R32L_LD(p) _mm512_loadu_pd(p)"); em.L.append("#endif")
    em.L.append("#ifndef R32L_ST"); em.L.append("#define R32L_ST(p,v) _mm512_storeu_pd((p),(v))"); em.L.append("#endif")
    em.L.append("#define LD R32L_LD"); em.L.append("#define ST R32L_ST"); em.L.append("")

    em.L.append("/* === FLAT U=1 === */"); em.L.append("")
    ff = emit_flat_kernel(em, 'fwd', itw_set)
    fb = emit_flat_kernel(em, 'bwd', itw_set)
    em.L.append("/* === LADDER U=1 === */"); em.L.append("")
    l1f = emit_ladder_kernel(em, 'fwd', itw_set, 1)
    l1b = emit_ladder_kernel(em, 'bwd', itw_set, 1)
    em.L.append("/* === LADDER U=2 === */"); em.L.append("")
    l2f = emit_ladder_kernel(em, 'fwd', itw_set, 2)
    l2b = emit_ladder_kernel(em, 'bwd', itw_set, 2)

    em.L.append("#undef LD"); em.L.append("#undef ST"); em.L.append("")
    em.L.append("#endif /* FFT_RADIX32_AVX512_TW_LADDER_H */")
    print("\n".join(em.L))

    v1_spill = 32+32  # old: 32sp + 32rl per kernel
    v2_spill_flat = ff[0]+ff[1]
    print(f"\n=== v2 STATS ===", file=sys.stderr)
    print(f"  NFUSE={NFUSE}: flat {ff[0]}sp+{ff[1]}rl={v2_spill_flat} (was {v1_spill}), saved {v1_spill - v2_spill_flat}", file=sys.stderr)
    print(f"  U1 ladder: {l1f[0]}sp+{l1f[1]}rl={l1f[0]+l1f[1]}", file=sys.stderr)
    print(f"  U2 ladder: {l2f[0]}sp+{l2f[1]}rl={l2f[0]+l2f[1]} (was {64+64}), spill buf=32 slots (was 64)", file=sys.stderr)
    print(f"  Lines: {len(em.L)}", file=sys.stderr)

if __name__ == '__main__':
    main()
