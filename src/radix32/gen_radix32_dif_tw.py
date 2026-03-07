#!/usr/bin/env python3
"""
gen_radix32_dif_tw.py — DIF twiddled DFT-32 codelet generator

Generates radix32_tw_flat_dif_kernel_{fwd,bwd}_{scalar,avx512}

DIF vs DIT:
  DIT: external twiddle on INPUT  (before butterfly)  — used in forward path
  DIF: external twiddle on OUTPUT (after butterfly)    — used in backward path

Same 8×4 decomposition, same internal W32 constants, same spill/fuse strategy.
Only the external twiddle placement differs.

Usage:
  python3 gen_radix32_dif_tw.py scalar  > fft_radix32_scalar_dif_tw.h
  python3 gen_radix32_dif_tw.py avx512  > fft_radix32_avx512_dif_tw.h
"""

import math, sys

N, N1, N2 = 32, 8, 4
NFUSE = 4

# ── Twiddle constants ──

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


# ═══════════════════════════════════════════════════════════════
# SCALAR EMITTER
# ═══════════════════════════════════════════════════════════════

class ScalarEmitter:
    def __init__(self):
        self.L=[]; self.ind=1; self.spill_c=0; self.reload_c=0
    def o(s,t=""): s.L.append("    "*s.ind+t)
    def c(s,t): s.o(f"/* {t} */")
    def b(s): s.L.append("")
    def add(s,a,b): return f"({a})+({b})"
    def sub(s,a,b): return f"({a})-({b})"
    def mul(s,a,b): return f"({a})*({b})"
    def neg(s,a):   return f"-({a})"
    def fma(s,a,b,c): return f"({a})*({b})+({c})"
    def fms(s,a,b,c): return f"({a})*({b})-({c})"

    def emit_load(s,v,n,k_expr="k"):
        s.o(f"{v}_re = in_re[{n}*K+{k_expr}];")
        s.o(f"{v}_im = in_im[{n}*K+{k_expr}];")
    def emit_store(s,v,m,k_expr="k"):
        s.o(f"out_re[{m}*K+{k_expr}] = {v}_re;")
        s.o(f"out_im[{m}*K+{k_expr}] = {v}_im;")
    def emit_spill(s,v,slot):
        s.o(f"spill_re[{slot}] = {v}_re;")
        s.o(f"spill_im[{slot}] = {v}_im;")
        s.spill_c+=1
    def emit_reload(s,v,slot):
        s.o(f"{v}_re = spill_re[{slot}];")
        s.o(f"{v}_im = spill_im[{slot}];")
        s.reload_c+=1

    def emit_ext_tw_inplace(s, v, tw_idx, d, k_expr="k"):
        """Apply external twiddle tw[(tw_idx)*K+k] to variable v in-place."""
        fwd = (d == 'fwd')
        s.o(f"{{ double wr = tw_re[{tw_idx}*K+{k_expr}], wi = tw_im[{tw_idx}*K+{k_expr}];")
        s.o(f"  double tr = {v}_re;")
        if fwd:
            s.o(f"  {v}_re = {v}_re*wr - {v}_im*wi;")
            s.o(f"  {v}_im = tr*wi + {v}_im*wr; }}")
        else:
            s.o(f"  {v}_re = {v}_re*wr + {v}_im*wi;")
            s.o(f"  {v}_im = {v}_im*wr - tr*wi; }}")

    def emit_cmul_inplace(s, v, wr, wi, d):
        fwd = (d=='fwd')
        s.o(f"{{ double tr={v}_re;")
        if fwd:
            s.o(f"  {v}_re = {s.fms(f'{v}_re',wr,s.mul(f'{v}_im',wi))};")
            s.o(f"  {v}_im = {s.fma('tr',wi,s.mul(f'{v}_im',wr))}; }}")
        else:
            s.o(f"  {v}_re = {s.fma(f'{v}_re',wr,s.mul(f'{v}_im',wi))};")
            s.o(f"  {v}_im = {s.fms(f'{v}_im',wr,s.mul('tr',wi))}; }}")

    def emit_twiddle(s,dst,src,e,tN,d):
        _,typ=twiddle_is_trivial(e,tN)
        fwd=(d=='fwd')
        if typ=='one':
            if dst!=src: s.o(f"{dst}_re={src}_re; {dst}_im={src}_im;")
        elif typ=='neg_one':
            s.o(f"{dst}_re=-{src}_re; {dst}_im=-{src}_im;")
        elif typ=='neg_j':
            if fwd: s.o(f"{{ double t={src}_re; {dst}_re={src}_im; {dst}_im=-t; }}")
            else:   s.o(f"{{ double t={src}_re; {dst}_re=-{src}_im; {dst}_im=t; }}")
        elif typ=='pos_j':
            if fwd: s.o(f"{{ double t={src}_re; {dst}_re=-{src}_im; {dst}_im=t; }}")
            else:   s.o(f"{{ double t={src}_re; {dst}_re={src}_im; {dst}_im=-t; }}")
        elif typ=='w8_1':
            s.o(f"{{ double tr={src}_re,ti={src}_im;")
            if fwd: s.o(f"  {dst}_re=(tr+ti)*SQRT2_INV; {dst}_im=(ti-tr)*SQRT2_INV; }}")
            else:   s.o(f"  {dst}_re=(tr-ti)*SQRT2_INV; {dst}_im=(tr+ti)*SQRT2_INV; }}")
        elif typ=='w8_3':
            s.o(f"{{ double tr={src}_re,ti={src}_im;")
            if fwd: s.o(f"  {dst}_re=(ti-tr)*SQRT2_INV; {dst}_im=-(tr+ti)*SQRT2_INV; }}")
            else:   s.o(f"  {dst}_re=-(tr+ti)*SQRT2_INV; {dst}_im=(tr-ti)*SQRT2_INV; }}")
        elif typ=='neg_w8_1':
            s.o(f"{{ double tr={src}_re,ti={src}_im;")
            if fwd: s.o(f"  {dst}_re=-(tr+ti)*SQRT2_INV; {dst}_im=(tr-ti)*SQRT2_INV; }}")
            else:   s.o(f"  {dst}_re=(ti-tr)*SQRT2_INV; {dst}_im=-(tr+ti)*SQRT2_INV; }}")
        elif typ=='neg_w8_3':
            s.o(f"{{ double tr={src}_re,ti={src}_im;")
            if fwd: s.o(f"  {dst}_re=(tr-ti)*SQRT2_INV; {dst}_im=(tr+ti)*SQRT2_INV; }}")
            else:   s.o(f"  {dst}_re=(tr+ti)*SQRT2_INV; {dst}_im=(ti-tr)*SQRT2_INV; }}")
        else:
            label=wN_label(e,tN)
            s.o(f"{{ double tr={src}_re;")
            if fwd:
                s.o(f"  {dst}_re={src}_re*{label}_re - {src}_im*{label}_im;")
                s.o(f"  {dst}_im=tr*{label}_im + {src}_im*{label}_re; }}")
            else:
                s.o(f"  {dst}_re={src}_re*{label}_re + {src}_im*{label}_im;")
                s.o(f"  {dst}_im={src}_im*{label}_re - tr*{label}_im; }}")

    def emit_radix8(s,v,d,label=""):
        fwd=(d=='fwd')
        if label: s.c(f"{label} [{d}]")
        s.o(f"{{ double e0r,e0i,e1r,e1i,e2r,e2i,e3r,e3i;")
        s.o(f"  double t0r,t0i,t1r,t1i,t2r,t2i,t3r,t3i;")
        s.o(f"  t0r={s.add(f'{v[0]}_re',f'{v[4]}_re')}; t0i={s.add(f'{v[0]}_im',f'{v[4]}_im')};")
        s.o(f"  t1r={s.sub(f'{v[0]}_re',f'{v[4]}_re')}; t1i={s.sub(f'{v[0]}_im',f'{v[4]}_im')};")
        s.o(f"  t2r={s.add(f'{v[2]}_re',f'{v[6]}_re')}; t2i={s.add(f'{v[2]}_im',f'{v[6]}_im')};")
        s.o(f"  t3r={s.sub(f'{v[2]}_re',f'{v[6]}_re')}; t3i={s.sub(f'{v[2]}_im',f'{v[6]}_im')};")
        s.o(f"  e0r={s.add('t0r','t2r')}; e0i={s.add('t0i','t2i')};")
        s.o(f"  e2r={s.sub('t0r','t2r')}; e2i={s.sub('t0i','t2i')};")
        ja,js = ('add','sub') if fwd else ('sub','add')
        s.o(f"  e1r={getattr(s,ja)('t1r','t3i')}; e1i={getattr(s,js)('t1i','t3r')};")
        s.o(f"  e3r={getattr(s,js)('t1r','t3i')}; e3i={getattr(s,ja)('t1i','t3r')};")
        s.o(f"  double o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;")
        s.o(f"  t0r={s.add(f'{v[1]}_re',f'{v[5]}_re')}; t0i={s.add(f'{v[1]}_im',f'{v[5]}_im')};")
        s.o(f"  t1r={s.sub(f'{v[1]}_re',f'{v[5]}_re')}; t1i={s.sub(f'{v[1]}_im',f'{v[5]}_im')};")
        s.o(f"  t2r={s.add(f'{v[3]}_re',f'{v[7]}_re')}; t2i={s.add(f'{v[3]}_im',f'{v[7]}_im')};")
        s.o(f"  t3r={s.sub(f'{v[3]}_re',f'{v[7]}_re')}; t3i={s.sub(f'{v[3]}_im',f'{v[7]}_im')};")
        s.o(f"  o0r={s.add('t0r','t2r')}; o0i={s.add('t0i','t2i')};")
        s.o(f"  o2r={s.sub('t0r','t2r')}; o2i={s.sub('t0i','t2i')};")
        s.o(f"  o1r={getattr(s,ja)('t1r','t3i')}; o1i={getattr(s,js)('t1i','t3r')};")
        s.o(f"  o3r={getattr(s,js)('t1r','t3i')}; o3i={getattr(s,ja)('t1i','t3r')};")
        if fwd:
            s.o(f"  t0r=(o1r+o1i)*SQRT2_INV; t0i=(o1i-o1r)*SQRT2_INV; o1r=t0r; o1i=t0i;")
            s.o(f"  t0r=o2i; t0i=-o2r; o2r=t0r; o2i=t0i;")
            s.o(f"  t0r=(o3i-o3r)*SQRT2_INV; t0i=-(o3r+o3i)*SQRT2_INV; o3r=t0r; o3i=t0i;")
        else:
            s.o(f"  t0r=(o1r-o1i)*SQRT2_INV; t0i=(o1r+o1i)*SQRT2_INV; o1r=t0r; o1i=t0i;")
            s.o(f"  t0r=-o2i; t0i=o2r; o2r=t0r; o2i=t0i;")
            s.o(f"  t0r=-(o3r+o3i)*SQRT2_INV; t0i=(o3r-o3i)*SQRT2_INV; o3r=t0r; o3i=t0i;")
        for i,j in [(0,4),(1,5),(2,6),(3,7)]:
            s.o(f"  {v[i]}_re=e{i}r+o{i}r; {v[i]}_im=e{i}i+o{i}i;")
            s.o(f"  {v[j]}_re=e{i}r-o{i}r; {v[j]}_im=e{i}i-o{i}i;")
        s.o(f"}}")

    def emit_radix4(s,v,d,label=""):
        fwd=(d=='fwd')
        if label: s.c(f"{label} [{d}]")
        a,b,c,dd=v[0],v[1],v[2],v[3]
        s.o(f"{{ double t0r,t0i,t1r,t1i,t2r,t2i,t3r,t3i;")
        s.o(f"  t0r={s.add(f'{a}_re',f'{c}_re')}; t0i={s.add(f'{a}_im',f'{c}_im')};")
        s.o(f"  t2r={s.sub(f'{a}_re',f'{c}_re')}; t2i={s.sub(f'{a}_im',f'{c}_im')};")
        s.o(f"  t1r={s.add(f'{b}_re',f'{dd}_re')}; t1i={s.add(f'{b}_im',f'{dd}_im')};")
        s.o(f"  t3r={s.sub(f'{b}_re',f'{dd}_re')}; t3i={s.sub(f'{b}_im',f'{dd}_im')};")
        s.o(f"  {a}_re=t0r+t1r; {a}_im=t0i+t1i;")
        s.o(f"  {c}_re=t0r-t1r; {c}_im=t0i-t1i;")
        if fwd:
            s.o(f"  {b}_re=t2r+t3i; {b}_im=t2i-t3r;")
            s.o(f"  {dd}_re=t2r-t3i; {dd}_im=t2i+t3r;")
        else:
            s.o(f"  {b}_re=t2r-t3i; {b}_im=t2i+t3r;")
            s.o(f"  {dd}_re=t2r+t3i; {dd}_im=t2i-t3r;")
        s.o(f"}}")


def emit_scalar_dif_kernel(em, d, itw_set):
    """DIF: load → butterfly → external twiddle on output → store"""
    em.L.append(f"static void")
    em.L.append(f"radix32_tw_flat_dif_kernel_{d}_scalar(")
    em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
    em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
    em.L.append(f"    const double * __restrict__ tw_re, const double * __restrict__ tw_im,")
    em.L.append(f"    size_t K)")
    em.L.append(f"{{")
    em.ind = 1; em.spill_c = 0; em.reload_c = 0

    em.o("double spill_re[32], spill_im[32];")
    em.o("double x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
    em.o("double x4_re,x4_im,x5_re,x5_im,x6_re,x6_im,x7_re,x7_im;")
    slist = ",".join(f"s{i}_re,s{i}_im" for i in range(NFUSE))
    em.o(f"double {slist};")
    em.b()

    em.o("for (size_t k = 0; k < K; k++) {")
    em.ind += 1

    xv8 = [f"x{i}" for i in range(8)]
    xv4 = [f"x{i}" for i in range(4)]
    fwd = (d == 'fwd')
    last_n2 = N2 - 1

    # ── PASS 1: Load (NO external twiddle) → radix-8 → spill ──
    for n2 in range(N2):
        is_last = (n2 == last_n2)
        em.c(f"sub-FFT n2={n2}")
        for n1 in range(N1):
            n = N2*n1 + n2
            em.emit_load(f"x{n1}", n)
            # DIF: NO external twiddle here (differs from DIT)
        em.b()
        em.emit_radix8(xv8, d, f"radix-8 n2={n2}")
        em.b()
        if is_last:
            em.c(f"FUSED: save x0..x{NFUSE-1} to locals, spill x{NFUSE}..x{N1-1}")
            for k1 in range(NFUSE):
                em.o(f"s{k1}_re = x{k1}_re; s{k1}_im = x{k1}_im;")
            for k1 in range(NFUSE, N1):
                em.emit_spill(f"x{k1}", n2*N1+k1)
        else:
            for k1 in range(N1):
                em.emit_spill(f"x{k1}", n2*N1+k1)
        em.b()

    # ── PASS 2: Reload → internal W32 twiddle → radix-4 → external twiddle → store ──
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
        # Internal W32 twiddles (same as DIT — part of 8×4 decomposition)
        if k1 > 0:
            for n2 in range(1,N2):
                e=(n2*k1)%N
                em.emit_twiddle(f"x{n2}",f"x{n2}",e,N,d)
            em.b()
        em.emit_radix4(xv4,d,f"radix-4 k1={k1}")
        em.b()

        # DIF: Apply external twiddle to outputs AFTER butterfly
        for k2 in range(N2):
            m = k1 + N1 * k2   # output index
            if m > 0:
                em.emit_ext_tw_inplace(f"x{k2}", m - 1, d)
        if k1 > 0 or True:  # at least one output got twiddled
            em.b()

        for k2 in range(N2):
            em.emit_store(f"x{k2}", k1+N1*k2)
        em.b()

    em.ind -= 1
    em.o("}")
    em.L.append("}"); em.L.append("")
    return em.spill_c, em.reload_c


# ═══════════════════════════════════════════════════════════════
# AVX2 EMITTER
# ═══════════════════════════════════════════════════════════════

class AVX2Emitter:
    NFUSE = 2  # AVX2: only 2 saved regs (16 YMM budget)
    def __init__(self):
        self.L=[]; self.ind=1; self.spill_c=0; self.reload_c=0
    def o(s,t=""): s.L.append("    "*s.ind+t)
    def c(s,t): s.o(f"/* {t} */")
    def b(s): s.L.append("")
    def add(s,a,b): return f"_mm256_add_pd({a},{b})"
    def sub(s,a,b): return f"_mm256_sub_pd({a},{b})"
    def mul(s,a,b): return f"_mm256_mul_pd({a},{b})"
    def neg(s,a):   return f"_mm256_xor_pd({a},sign_flip)"
    def fma(s,a,b,c): return f"_mm256_fmadd_pd({a},{b},{c})"
    def fms(s,a,b,c): return f"_mm256_fmsub_pd({a},{b},{c})"
    def fnma(s,a,b,c): return f"_mm256_fnmadd_pd({a},{b},{c})"

    def emit_load(s,v,n,k_expr="k"):
        s.o(f"{v}_re = LD(&in_re[{n}*K+{k_expr}]);")
        s.o(f"{v}_im = LD(&in_im[{n}*K+{k_expr}]);")
    def emit_store(s,v,m,k_expr="k"):
        s.o(f"ST(&out_re[{m}*K+{k_expr}],{v}_re);")
        s.o(f"ST(&out_im[{m}*K+{k_expr}],{v}_im);")
    def emit_spill(s,v,slot):
        s.o(f"_mm256_store_pd(&spill_re[{slot}*4],{v}_re);")
        s.o(f"_mm256_store_pd(&spill_im[{slot}*4],{v}_im);")
        s.spill_c+=1
    def emit_reload(s,v,slot):
        s.o(f"{v}_re = _mm256_load_pd(&spill_re[{slot}*4]);")
        s.o(f"{v}_im = _mm256_load_pd(&spill_im[{slot}*4]);")
        s.reload_c+=1

    def emit_ext_tw_inplace(s, v, tw_idx, d, k_expr="k"):
        fwd = (d == 'fwd')
        s.o(f"{{ __m256d wr = LD(&tw_re[{tw_idx}*K+{k_expr}]);")
        s.o(f"  __m256d wi = LD(&tw_im[{tw_idx}*K+{k_expr}]);")
        s.o(f"  __m256d tr = {v}_re;")
        if fwd:
            s.o(f"  {v}_re = {s.fms(f'{v}_re','wr',s.mul(f'{v}_im','wi'))};")
            s.o(f"  {v}_im = {s.fma('tr','wi',s.mul(f'{v}_im','wr'))}; }}")
        else:
            s.o(f"  {v}_re = {s.fma(f'{v}_re','wr',s.mul(f'{v}_im','wi'))};")
            s.o(f"  {v}_im = {s.fms(f'{v}_im','wr',s.mul('tr','wi'))}; }}")

    def emit_twiddle(s,dst,src,e,tN,d):
        _,typ=twiddle_is_trivial(e,tN)
        fwd=(d=='fwd')
        if typ=='one':
            if dst!=src: s.o(f"{dst}_re={src}_re; {dst}_im={src}_im;")
        elif typ=='neg_one':
            s.o(f"{dst}_re={s.neg(f'{src}_re')}; {dst}_im={s.neg(f'{src}_im')};")
        elif typ=='neg_j':
            if fwd: s.o(f"{{ __m256d t={src}_re; {dst}_re={src}_im; {dst}_im={s.neg('t')}; }}")
            else:   s.o(f"{{ __m256d t={src}_re; {dst}_re={s.neg(f'{src}_im')}; {dst}_im=t; }}")
        elif typ=='pos_j':
            if fwd: s.o(f"{{ __m256d t={src}_re; {dst}_re={s.neg(f'{src}_im')}; {dst}_im=t; }}")
            else:   s.o(f"{{ __m256d t={src}_re; {dst}_re={src}_im; {dst}_im={s.neg('t')}; }}")
        elif typ=='w8_1':
            s.o(f"{{ __m256d tr={src}_re,ti={src}_im;")
            if fwd: s.o(f"  {dst}_re={s.mul(s.add('tr','ti'),'sqrt2_inv')}; {dst}_im={s.mul(s.sub('ti','tr'),'sqrt2_inv')}; }}")
            else:   s.o(f"  {dst}_re={s.mul(s.sub('tr','ti'),'sqrt2_inv')}; {dst}_im={s.mul(s.add('tr','ti'),'sqrt2_inv')}; }}")
        elif typ=='w8_3':
            s.o(f"{{ __m256d tr={src}_re,ti={src}_im;")
            if fwd: s.o(f"  {dst}_re={s.mul(s.sub('ti','tr'),'sqrt2_inv')}; {dst}_im={s.neg(s.mul(s.add('tr','ti'),'sqrt2_inv'))}; }}")
            else:   s.o(f"  {dst}_re={s.neg(s.mul(s.add('tr','ti'),'sqrt2_inv'))}; {dst}_im={s.mul(s.sub('tr','ti'),'sqrt2_inv')}; }}")
        elif typ=='neg_w8_1':
            s.o(f"{{ __m256d tr={src}_re,ti={src}_im;")
            if fwd: s.o(f"  {dst}_re={s.neg(s.mul(s.add('tr','ti'),'sqrt2_inv'))}; {dst}_im={s.mul(s.sub('tr','ti'),'sqrt2_inv')}; }}")
            else:   s.o(f"  {dst}_re={s.mul(s.sub('ti','tr'),'sqrt2_inv')}; {dst}_im={s.neg(s.mul(s.add('tr','ti'),'sqrt2_inv'))}; }}")
        elif typ=='neg_w8_3':
            s.o(f"{{ __m256d tr={src}_re,ti={src}_im;")
            if fwd: s.o(f"  {dst}_re={s.mul(s.sub('tr','ti'),'sqrt2_inv')}; {dst}_im={s.mul(s.add('tr','ti'),'sqrt2_inv')}; }}")
            else:   s.o(f"  {dst}_re={s.mul(s.add('tr','ti'),'sqrt2_inv')}; {dst}_im={s.mul(s.sub('ti','tr'),'sqrt2_inv')}; }}")
        else:
            label=wN_label(e,tN)
            s.o(f"{{ __m256d tr={src}_re;")
            if fwd:
                s.o(f"  {dst}_re={s.fms(f'{src}_re',f'tw_{label}_re',s.mul(f'{src}_im',f'tw_{label}_im'))};")
                s.o(f"  {dst}_im={s.fma('tr',f'tw_{label}_im',s.mul(f'{src}_im',f'tw_{label}_re'))}; }}")
            else:
                s.o(f"  {dst}_re={s.fma(f'{src}_re',f'tw_{label}_re',s.mul(f'{src}_im',f'tw_{label}_im'))};")
                s.o(f"  {dst}_im={s.fms(f'{src}_im',f'tw_{label}_re',s.mul('tr',f'tw_{label}_im'))}; }}")

    def emit_radix8(s,v,d,label=""):
        fwd=(d=='fwd')
        if label: s.c(f"{label} [{d}]")
        s.o(f"{{ __m256d e0r,e0i,e1r,e1i,e2r,e2i,e3r,e3i;")
        s.o(f"  __m256d t0r,t0i,t1r,t1i,t2r,t2i,t3r,t3i;")
        s.o(f"  t0r={s.add(f'{v[0]}_re',f'{v[4]}_re')}; t0i={s.add(f'{v[0]}_im',f'{v[4]}_im')};")
        s.o(f"  t1r={s.sub(f'{v[0]}_re',f'{v[4]}_re')}; t1i={s.sub(f'{v[0]}_im',f'{v[4]}_im')};")
        s.o(f"  t2r={s.add(f'{v[2]}_re',f'{v[6]}_re')}; t2i={s.add(f'{v[2]}_im',f'{v[6]}_im')};")
        s.o(f"  t3r={s.sub(f'{v[2]}_re',f'{v[6]}_re')}; t3i={s.sub(f'{v[2]}_im',f'{v[6]}_im')};")
        s.o(f"  e0r={s.add('t0r','t2r')}; e0i={s.add('t0i','t2i')};")
        s.o(f"  e2r={s.sub('t0r','t2r')}; e2i={s.sub('t0i','t2i')};")
        ja,js = ('add','sub') if fwd else ('sub','add')
        s.o(f"  e1r={getattr(s,ja)('t1r','t3i')}; e1i={getattr(s,js)('t1i','t3r')};")
        s.o(f"  e3r={getattr(s,js)('t1r','t3i')}; e3i={getattr(s,ja)('t1i','t3r')};")
        s.o(f"  __m256d o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;")
        s.o(f"  t0r={s.add(f'{v[1]}_re',f'{v[5]}_re')}; t0i={s.add(f'{v[1]}_im',f'{v[5]}_im')};")
        s.o(f"  t1r={s.sub(f'{v[1]}_re',f'{v[5]}_re')}; t1i={s.sub(f'{v[1]}_im',f'{v[5]}_im')};")
        s.o(f"  t2r={s.add(f'{v[3]}_re',f'{v[7]}_re')}; t2i={s.add(f'{v[3]}_im',f'{v[7]}_im')};")
        s.o(f"  t3r={s.sub(f'{v[3]}_re',f'{v[7]}_re')}; t3i={s.sub(f'{v[3]}_im',f'{v[7]}_im')};")
        s.o(f"  o0r={s.add('t0r','t2r')}; o0i={s.add('t0i','t2i')};")
        s.o(f"  o2r={s.sub('t0r','t2r')}; o2i={s.sub('t0i','t2i')};")
        s.o(f"  o1r={getattr(s,ja)('t1r','t3i')}; o1i={getattr(s,js)('t1i','t3r')};")
        s.o(f"  o3r={getattr(s,js)('t1r','t3i')}; o3i={getattr(s,ja)('t1i','t3r')};")
        if fwd:
            s.o(f"  t0r={s.mul(s.add('o1r','o1i'),'sqrt2_inv')}; t0i={s.mul(s.sub('o1i','o1r'),'sqrt2_inv')};")
            s.o(f"  o1r=t0r; o1i=t0i;")
            s.o(f"  t0r=o2i; t0i={s.neg('o2r')}; o2r=t0r; o2i=t0i;")
            s.o(f"  t0r={s.mul(s.sub('o3i','o3r'),'sqrt2_inv')}; t0i={s.neg(s.mul(s.add('o3r','o3i'),'sqrt2_inv'))};")
            s.o(f"  o3r=t0r; o3i=t0i;")
        else:
            s.o(f"  t0r={s.mul(s.sub('o1r','o1i'),'sqrt2_inv')}; t0i={s.mul(s.add('o1r','o1i'),'sqrt2_inv')};")
            s.o(f"  o1r=t0r; o1i=t0i;")
            s.o(f"  t0r={s.neg('o2i')}; t0i=o2r; o2r=t0r; o2i=t0i;")
            s.o(f"  t0r={s.neg(s.mul(s.add('o3r','o3i'),'sqrt2_inv'))}; t0i={s.mul(s.sub('o3r','o3i'),'sqrt2_inv')};")
            s.o(f"  o3r=t0r; o3i=t0i;")
        for i,j in [(0,4),(1,5),(2,6),(3,7)]:
            s.o(f"  {v[i]}_re={s.add(f'e{i}r',f'o{i}r')}; {v[i]}_im={s.add(f'e{i}i',f'o{i}i')};")
            s.o(f"  {v[j]}_re={s.sub(f'e{i}r',f'o{i}r')}; {v[j]}_im={s.sub(f'e{i}i',f'o{i}i')};")
        s.o(f"}}")

    def emit_radix4(s,v,d,label=""):
        fwd=(d=='fwd')
        if label: s.c(f"{label} [{d}]")
        a,b,c,dd=v[0],v[1],v[2],v[3]
        s.o(f"{{ __m256d t0r,t0i,t1r,t1i,t2r,t2i,t3r,t3i;")
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


def emit_avx2_dif_kernel(em, d, itw_set):
    """AVX2 DIF: load → butterfly → external twiddle on output → store"""
    nfuse = AVX2Emitter.NFUSE
    em.L.append(f'__attribute__((target("avx2,fma")))')
    em.L.append(f"static void")
    em.L.append(f"radix32_tw_flat_dif_kernel_{d}_avx2(")
    em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
    em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
    em.L.append(f"    const double * __restrict__ tw_re, const double * __restrict__ tw_im,")
    em.L.append(f"    size_t K)")
    em.L.append(f"{{")
    em.ind = 1; em.spill_c = 0; em.reload_c = 0

    em.o("#define LD _mm256_load_pd")
    em.o("#define ST _mm256_store_pd")
    em.o("const __m256d sign_flip = _mm256_set1_pd(-0.0);")
    em.o("const __m256d sqrt2_inv = _mm256_set1_pd(0.70710678118654752440);")
    em.b()
    em.o("__attribute__((aligned(32))) double spill_re[128];")
    em.o("__attribute__((aligned(32))) double spill_im[128];")
    em.o("__m256d x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
    em.o("__m256d x4_re,x4_im,x5_re,x5_im,x6_re,x6_im,x7_re,x7_im;")
    slist = ",".join(f"s{i}_re,s{i}_im" for i in range(nfuse))
    em.o(f"__m256d {slist};")
    em.b()

    # Pre-load internal twiddle constants into YMM registers
    itw_labels = sorted(set(wN_label(e,tN) for e,tN in itw_set))
    if itw_labels:
        em.c("Pre-load internal W32 constants into registers")
        for label in itw_labels:
            em.o(f"const __m256d tw_{label}_re = _mm256_set1_pd({label}_re);")
            em.o(f"const __m256d tw_{label}_im = _mm256_set1_pd({label}_im);")
        em.b()

    em.o("for (size_t k = 0; k < K; k += 4) {")
    em.ind += 1
    xv8 = [f"x{i}" for i in range(8)]
    xv4 = [f"x{i}" for i in range(4)]
    fwd = (d == 'fwd')
    last_n2 = N2 - 1

    # ── PASS 1: Load (NO external twiddle) → radix-8 → spill ──
    for n2 in range(N2):
        is_last = (n2 == last_n2)
        em.c(f"sub-FFT n2={n2}")
        for n1 in range(N1):
            n = N2*n1 + n2
            em.emit_load(f"x{n1}", n)
        em.b()
        em.emit_radix8(xv8, d, f"radix-8 n2={n2}")
        em.b()
        if is_last:
            em.c(f"FUSED: save x0..x{nfuse-1} in s-regs, spill x{nfuse}..x{N1-1}")
            for k1 in range(nfuse):
                em.o(f"s{k1}_re = x{k1}_re; s{k1}_im = x{k1}_im;")
            for k1 in range(nfuse, N1):
                em.emit_spill(f"x{k1}", n2*N1+k1)
        else:
            for k1 in range(N1):
                em.emit_spill(f"x{k1}", n2*N1+k1)
        em.b()

    # ── PASS 2: Reload → internal W32 tw → radix-4 → external tw → store ──
    em.c(f"PASS 2"); em.b()
    for k1 in range(N1):
        em.c(f"column k1={k1}")
        if k1 < nfuse:
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

        # DIF: Apply external twiddle to outputs AFTER butterfly
        for k2 in range(N2):
            m = k1 + N1 * k2
            if m > 0:
                em.emit_ext_tw_inplace(f"x{k2}", m - 1, d)
        em.b()

        for k2 in range(N2):
            em.emit_store(f"x{k2}", k1+N1*k2)
        em.b()

    em.ind -= 1
    em.o("}")
    em.o("#undef LD")
    em.o("#undef ST")
    em.L.append("}"); em.L.append("")
    return em.spill_c, em.reload_c


# ═══════════════════════════════════════════════════════════════
# AVX-512 EMITTER
# ═══════════════════════════════════════════════════════════════

class AVX512Emitter:
    def __init__(self):
        self.L=[]; self.ind=1; self.spill_c=0; self.reload_c=0
    def o(s,t=""): s.L.append("    "*s.ind+t)
    def c(s,t): s.o(f"/* {t} */")
    def b(s): s.L.append("")
    def add(s,a,b): return f"_mm512_add_pd({a},{b})"
    def sub(s,a,b): return f"_mm512_sub_pd({a},{b})"
    def mul(s,a,b): return f"_mm512_mul_pd({a},{b})"
    def neg(s,a):   return f"_mm512_sub_pd(_mm512_setzero_pd(),{a})"
    def fma(s,a,b,c): return f"_mm512_fmadd_pd({a},{b},{c})"
    def fms(s,a,b,c): return f"_mm512_fmsub_pd({a},{b},{c})"
    def fnma(s,a,b,c): return f"_mm512_fnmadd_pd({a},{b},{c})"
    def set1(s,v): return f"_mm512_set1_pd({v})"

    def emit_load(s,v,n,k_expr="k"):
        s.o(f"{v}_re = _mm512_load_pd(&in_re[{n}*K+{k_expr}]);")
        s.o(f"{v}_im = _mm512_load_pd(&in_im[{n}*K+{k_expr}]);")
    def emit_store(s,v,m,k_expr="k"):
        s.o(f"_mm512_store_pd(&out_re[{m}*K+{k_expr}], {v}_re);")
        s.o(f"_mm512_store_pd(&out_im[{m}*K+{k_expr}], {v}_im);")
    def emit_spill(s,v,slot):
        s.o(f"_mm512_store_pd(&spill_re[{slot}*8], {v}_re);")
        s.o(f"_mm512_store_pd(&spill_im[{slot}*8], {v}_im);")
        s.spill_c+=1
    def emit_reload(s,v,slot):
        s.o(f"{v}_re = _mm512_load_pd(&spill_re[{slot}*8]);")
        s.o(f"{v}_im = _mm512_load_pd(&spill_im[{slot}*8]);")
        s.reload_c+=1

    def emit_ext_tw_inplace(s, v, tw_idx, d, k_expr="k"):
        """Apply external twiddle tw[(tw_idx)*K+k] to ZMM variable v in-place."""
        fwd = (d == 'fwd')
        s.o(f"{{ __m512d wr = _mm512_load_pd(&tw_re[{tw_idx}*K+{k_expr}]);")
        s.o(f"  __m512d wi = _mm512_load_pd(&tw_im[{tw_idx}*K+{k_expr}]);")
        s.o(f"  __m512d tr = {v}_re;")
        if fwd:
            s.o(f"  {v}_re = {s.fms(f'{v}_re','wr',s.mul(f'{v}_im','wi'))};")
            s.o(f"  {v}_im = {s.fma('tr','wi',s.mul(f'{v}_im','wr'))}; }}")
        else:
            s.o(f"  {v}_re = {s.fma(f'{v}_re','wr',s.mul(f'{v}_im','wi'))};")
            s.o(f"  {v}_im = {s.fms(f'{v}_im','wr',s.mul('tr','wi'))}; }}")

    def emit_cmul_inplace(s, v, wr, wi, d):
        fwd = (d=='fwd')
        s.o(f"{{ __m512d tr = {v}_re;")
        if fwd:
            s.o(f"  {v}_re = {s.fms(f'{v}_re',wr,s.mul(f'{v}_im',wi))};")
            s.o(f"  {v}_im = {s.fma('tr',wi,s.mul(f'{v}_im',wr))}; }}")
        else:
            s.o(f"  {v}_re = {s.fma(f'{v}_re',wr,s.mul(f'{v}_im',wi))};")
            s.o(f"  {v}_im = {s.fms(f'{v}_im',wr,s.mul('tr',wi))}; }}")

    def emit_twiddle(s,dst,src,e,tN,d):
        _,typ=twiddle_is_trivial(e,tN)
        fwd=(d=='fwd')
        sq = "_mm512_set1_pd(SQRT2_INV)"
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
            s.o(f"{{ __m512d sq={sq}, tr={src}_re, ti={src}_im;")
            if fwd: s.o(f"  {dst}_re={s.mul(s.add('tr','ti'),'sq')}; {dst}_im={s.mul(s.sub('ti','tr'),'sq')}; }}")
            else:   s.o(f"  {dst}_re={s.mul(s.sub('tr','ti'),'sq')}; {dst}_im={s.mul(s.add('tr','ti'),'sq')}; }}")
        elif typ=='w8_3':
            s.o(f"{{ __m512d sq={sq}, tr={src}_re, ti={src}_im;")
            if fwd: s.o(f"  {dst}_re={s.mul(s.sub('ti','tr'),'sq')}; {dst}_im={s.neg(s.mul(s.add('tr','ti'),'sq'))}; }}")
            else:   s.o(f"  {dst}_re={s.neg(s.mul(s.add('tr','ti'),'sq'))}; {dst}_im={s.mul(s.sub('tr','ti'),'sq')}; }}")
        elif typ=='neg_w8_1':
            s.o(f"{{ __m512d sq={sq}, tr={src}_re, ti={src}_im;")
            if fwd: s.o(f"  {dst}_re={s.neg(s.mul(s.add('tr','ti'),'sq'))}; {dst}_im={s.mul(s.sub('tr','ti'),'sq')}; }}")
            else:   s.o(f"  {dst}_re={s.mul(s.sub('ti','tr'),'sq')}; {dst}_im={s.neg(s.mul(s.add('tr','ti'),'sq'))}; }}")
        elif typ=='neg_w8_3':
            s.o(f"{{ __m512d sq={sq}, tr={src}_re, ti={src}_im;")
            if fwd: s.o(f"  {dst}_re={s.mul(s.sub('tr','ti'),'sq')}; {dst}_im={s.mul(s.add('tr','ti'),'sq')}; }}")
            else:   s.o(f"  {dst}_re={s.mul(s.add('tr','ti'),'sq')}; {dst}_im={s.mul(s.sub('ti','tr'),'sq')}; }}")
        else:
            label=wN_label(e,tN)
            s.o(f"{{ __m512d wr={s.set1(f'{label}_re')}, wi={s.set1(f'{label}_im')};")
            s.o(f"  __m512d tr={src}_re;")
            if fwd:
                s.o(f"  {dst}_re={s.fms(f'{src}_re','wr',s.mul(f'{src}_im','wi'))};")
                s.o(f"  {dst}_im={s.fma('tr','wi',s.mul(f'{src}_im','wr'))}; }}")
            else:
                s.o(f"  {dst}_re={s.fma(f'{src}_re','wr',s.mul(f'{src}_im','wi'))};")
                s.o(f"  {dst}_im={s.fms(f'{src}_im','wr',s.mul('tr','wi'))}; }}")

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
        ja,js = ('add','sub') if fwd else ('sub','add')
        s.o(f"  e1r={getattr(s,ja)('t1r','t3i')}; e1i={getattr(s,js)('t1i','t3r')};")
        s.o(f"  e3r={getattr(s,js)('t1r','t3i')}; e3i={getattr(s,ja)('t1i','t3r')};")
        s.o(f"  __m512d o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;")
        s.o(f"  t0r={s.add(f'{v[1]}_re',f'{v[5]}_re')}; t0i={s.add(f'{v[1]}_im',f'{v[5]}_im')};")
        s.o(f"  t1r={s.sub(f'{v[1]}_re',f'{v[5]}_re')}; t1i={s.sub(f'{v[1]}_im',f'{v[5]}_im')};")
        s.o(f"  t2r={s.add(f'{v[3]}_re',f'{v[7]}_re')}; t2i={s.add(f'{v[3]}_im',f'{v[7]}_im')};")
        s.o(f"  t3r={s.sub(f'{v[3]}_re',f'{v[7]}_re')}; t3i={s.sub(f'{v[3]}_im',f'{v[7]}_im')};")
        s.o(f"  o0r={s.add('t0r','t2r')}; o0i={s.add('t0i','t2i')};")
        s.o(f"  o2r={s.sub('t0r','t2r')}; o2i={s.sub('t0i','t2i')};")
        s.o(f"  o1r={getattr(s,ja)('t1r','t3i')}; o1i={getattr(s,js)('t1i','t3r')};")
        s.o(f"  o3r={getattr(s,js)('t1r','t3i')}; o3i={getattr(s,ja)('t1i','t3r')};")
        sq = "_mm512_set1_pd(SQRT2_INV)"
        if fwd:
            s.o(f"  {{ __m512d sq={sq};")
            s.o(f"    t0r={s.mul(s.add('o1r','o1i'),'sq')}; t0i={s.mul(s.sub('o1i','o1r'),'sq')}; o1r=t0r; o1i=t0i;")
            s.o(f"    t0r=o2i; t0i={s.neg('o2r')}; o2r=t0r; o2i=t0i;")
            s.o(f"    t0r={s.mul(s.sub('o3i','o3r'),'sq')}; t0i={s.neg(s.mul(s.add('o3r','o3i'),'sq'))}; o3r=t0r; o3i=t0i; }}")
        else:
            s.o(f"  {{ __m512d sq={sq};")
            s.o(f"    t0r={s.mul(s.sub('o1r','o1i'),'sq')}; t0i={s.mul(s.add('o1r','o1i'),'sq')}; o1r=t0r; o1i=t0i;")
            s.o(f"    t0r={s.neg('o2i')}; t0i=o2r; o2r=t0r; o2i=t0i;")
            s.o(f"    t0r={s.neg(s.mul(s.add('o3r','o3i'),'sq'))}; t0i={s.mul(s.sub('o3r','o3i'),'sq')}; o3r=t0r; o3i=t0i; }}")
        for i,j in [(0,4),(1,5),(2,6),(3,7)]:
            s.o(f"  {v[i]}_re={s.add(f'e{i}r',f'o{i}r')}; {v[i]}_im={s.add(f'e{i}i',f'o{i}i')};")
            s.o(f"  {v[j]}_re={s.sub(f'e{i}r',f'o{i}r')}; {v[j]}_im={s.sub(f'e{i}i',f'o{i}i')};")
        s.o(f"}}")

    def emit_radix4(s,v,d,label=""):
        fwd=(d=='fwd')
        if label: s.c(f"{label} [{d}]")
        a,b,c,dd=v[0],v[1],v[2],v[3]
        s.o(f"{{ __m512d t0r,t0i,t1r,t1i,t2r,t2i,t3r,t3i;")
        s.o(f"  t0r={s.add(f'{a}_re',f'{c}_re')}; t0i={s.add(f'{a}_im',f'{c}_im')};")
        s.o(f"  t2r={s.sub(f'{a}_re',f'{c}_re')}; t2i={s.sub(f'{a}_im',f'{c}_im')};")
        s.o(f"  t1r={s.add(f'{b}_re',f'{dd}_re')}; t1i={s.add(f'{b}_im',f'{dd}_im')};")
        s.o(f"  t3r={s.sub(f'{b}_re',f'{dd}_re')}; t3i={s.sub(f'{b}_im',f'{dd}_im')};")
        s.o(f"  {a}_re={s.add('t0r','t1r')}; {a}_im={s.add('t0i','t1i')};")
        s.o(f"  {c}_re={s.sub('t0r','t1r')}; {c}_im={s.sub('t0i','t1i')};")
        if fwd:
            s.o(f"  {b}_re={s.add('t2r','t3i')}; {b}_im={s.sub('t2i','t3r')};")
            s.o(f"  {dd}_re={s.sub('t2r','t3i')}; {dd}_im={s.add('t2i','t3r')};")
        else:
            s.o(f"  {b}_re={s.sub('t2r','t3i')}; {b}_im={s.add('t2i','t3r')};")
            s.o(f"  {dd}_re={s.add('t2r','t3i')}; {dd}_im={s.sub('t2i','t3r')};")
        s.o(f"}}")


def emit_avx512_dif_kernel(em, d, itw_set):
    """AVX-512 DIF: load → butterfly → external twiddle on output → store"""
    em.L.append(f"TARGET_AVX512")
    em.L.append(f"static void")
    em.L.append(f"radix32_tw_flat_dif_kernel_{d}_avx512(")
    em.L.append(f"    const double * RESTRICT in_re, const double * RESTRICT in_im,")
    em.L.append(f"    double * RESTRICT out_re, double * RESTRICT out_im,")
    em.L.append(f"    const double * RESTRICT tw_re, const double * RESTRICT tw_im,")
    em.L.append(f"    size_t K)")
    em.L.append(f"{{")
    em.ind = 1; em.spill_c = 0; em.reload_c = 0

    em.o("ALIGNAS_64 double spill_re[256], spill_im[256];")
    em.o("__m512d x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
    em.o("__m512d x4_re,x4_im,x5_re,x5_im,x6_re,x6_im,x7_re,x7_im;")
    slist = ",".join(f"s{i}_re,s{i}_im" for i in range(NFUSE))
    em.o(f"__m512d {slist};")
    em.b()

    em.o("for (size_t k = 0; k < K; k += 8) {")
    em.ind += 1
    xv8 = [f"x{i}" for i in range(8)]
    xv4 = [f"x{i}" for i in range(4)]
    fwd = (d == 'fwd')
    last_n2 = N2 - 1

    # ── PASS 1: Load (NO external twiddle) → radix-8 → spill ──
    for n2 in range(N2):
        is_last = (n2 == last_n2)
        em.c(f"sub-FFT n2={n2}")
        for n1 in range(N1):
            n = N2*n1 + n2
            em.emit_load(f"x{n1}", n)
            # DIF: NO external twiddle here
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

    # ── PASS 2: Reload → internal W32 tw → radix-4 → external tw → store ──
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
        # Internal W32 twiddles (same as DIT)
        if k1 > 0:
            for n2 in range(1,N2):
                e=(n2*k1)%N
                em.emit_twiddle(f"x{n2}",f"x{n2}",e,N,d)
            em.b()
        em.emit_radix4(xv4,d,f"radix-4 k1={k1}")
        em.b()

        # DIF: Apply external twiddle to outputs AFTER butterfly
        for k2 in range(N2):
            m = k1 + N1 * k2
            if m > 0:
                em.emit_ext_tw_inplace(f"x{k2}", m - 1, d)
        em.b()

        for k2 in range(N2):
            em.emit_store(f"x{k2}", k1+N1*k2)
        em.b()

    em.ind -= 1
    em.o("}")
    em.L.append("}"); em.L.append("")
    return em.spill_c, em.reload_c


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ('scalar', 'avx2', 'avx512'):
        print("Usage: gen_radix32_dif_tw.py scalar|avx2|avx512", file=sys.stderr)
        sys.exit(1)
    mode = sys.argv[1]

    itw_set = collect_internal_twiddles()

    if mode == 'scalar':
        em = ScalarEmitter()
        em.L.append("/**")
        em.L.append(" * @file fft_radix32_scalar_dif_tw.h")
        em.L.append(" * @brief Scalar DIF twiddled DFT-32 — twiddle AFTER butterfly")
        em.L.append(" *")
        em.L.append(" * For DIF backward path: natural input → stages outer→inner → output perm")
        em.L.append(" * Generated by gen_radix32_dif_tw.py")
        em.L.append(" */")
        em.L.append("")
        em.L.append("#ifndef FFT_RADIX32_SCALAR_DIF_TW_H")
        em.L.append("#define FFT_RADIX32_SCALAR_DIF_TW_H")
        em.L.append("")
        em.L.append("#ifndef SQRT2_INV")
        em.L.append("#define SQRT2_INV 0.70710678118654752440")
        em.L.append("#endif")
        em.L.append("")

        # Internal twiddle constants
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

        em.L.append("/* === SCALAR DIF TWIDDLED DFT-32 === */"); em.L.append("")
        ff = emit_scalar_dif_kernel(em, 'fwd', itw_set)
        fb = emit_scalar_dif_kernel(em, 'bwd', itw_set)
        em.L.append("#endif /* FFT_RADIX32_SCALAR_DIF_TW_H */")
        print("\n".join(em.L))
        print(f"\n=== SCALAR DIF TW ===", file=sys.stderr)
        print(f"  spills={ff[0]} reloads={ff[1]}", file=sys.stderr)
        print(f"  Lines: {len(em.L)}", file=sys.stderr)

    elif mode == 'avx2':
        em = AVX2Emitter()
        em.L.append("/**")
        em.L.append(" * @file fft_radix32_avx2_dif_tw.h")
        em.L.append(" * @brief AVX2 DIF twiddled DFT-32 — twiddle AFTER butterfly")
        em.L.append(" *")
        em.L.append(" * NFUSE=2 (16 YMM register budget). k-step=4.")
        em.L.append(" * Generated by gen_radix32_dif_tw.py")
        em.L.append(" */")
        em.L.append("")
        em.L.append("#ifndef FFT_RADIX32_AVX2_DIF_TW_H")
        em.L.append("#define FFT_RADIX32_AVX2_DIF_TW_H")
        em.L.append("")
        em.L.append("#include <immintrin.h>")
        em.L.append("")
        em.L.append("#ifndef SQRT2_INV")
        em.L.append("#define SQRT2_INV 0.70710678118654752440")
        em.L.append("#endif")
        em.L.append("")

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

        em.L.append("/* === AVX2 DIF TWIDDLED DFT-32 (NFUSE=2) === */"); em.L.append("")
        ff = emit_avx2_dif_kernel(em, 'fwd', itw_set)
        fb = emit_avx2_dif_kernel(em, 'bwd', itw_set)
        em.L.append("#endif /* FFT_RADIX32_AVX2_DIF_TW_H */")
        print("\n".join(em.L))
        print(f"\n=== AVX2 DIF TW (NFUSE=2) ===", file=sys.stderr)
        print(f"  spills={ff[0]} reloads={ff[1]}", file=sys.stderr)
        print(f"  Lines: {len(em.L)}", file=sys.stderr)

    elif mode == 'avx512':
        em = AVX512Emitter()
        em.L.append("/**")
        em.L.append(" * @file fft_radix32_avx512_dif_tw.h")
        em.L.append(" * @brief AVX-512 DIF twiddled DFT-32 — twiddle AFTER butterfly")
        em.L.append(" *")
        em.L.append(" * For DIF backward path: natural input → stages outer→inner → output perm")
        em.L.append(" * Generated by gen_radix32_dif_tw.py")
        em.L.append(" */")
        em.L.append("")
        em.L.append("#ifndef FFT_RADIX32_AVX512_DIF_TW_H")
        em.L.append("#define FFT_RADIX32_AVX512_DIF_TW_H")
        em.L.append("")
        em.L.append("#include <immintrin.h>")
        em.L.append("")
        em.L.append("#ifndef SQRT2_INV")
        em.L.append("#define SQRT2_INV 0.70710678118654752440")
        em.L.append("#endif")
        em.L.append("#ifndef TARGET_AVX512")
        em.L.append('#define TARGET_AVX512 __attribute__((target("avx512f,avx512dq,fma")))')
        em.L.append("#endif")
        em.L.append("#ifndef RESTRICT")
        em.L.append("#define RESTRICT __restrict__")
        em.L.append("#endif")
        em.L.append("#ifndef ALIGNAS_64")
        em.L.append('#define ALIGNAS_64 __attribute__((aligned(64)))')
        em.L.append("#endif")
        em.L.append("")

        # Internal twiddle constants
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

        em.L.append("/* === AVX-512 DIF TWIDDLED DFT-32 === */"); em.L.append("")
        ff = emit_avx512_dif_kernel(em, 'fwd', itw_set)
        fb = emit_avx512_dif_kernel(em, 'bwd', itw_set)
        em.L.append("#endif /* FFT_RADIX32_AVX512_DIF_TW_H */")
        print("\n".join(em.L))
        print(f"\n=== AVX-512 DIF TW ===", file=sys.stderr)
        print(f"  spills={ff[0]} reloads={ff[1]}", file=sys.stderr)
        print(f"  Lines: {len(em.L)}", file=sys.stderr)


if __name__ == '__main__':
    main()
