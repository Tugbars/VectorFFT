#!/usr/bin/env python3
"""
gen_radix32_avx512_n1_u2.py — DFT-32 AVX-512 N1 with U=1+prefetch and U=2+prefetch

Emits four kernels:
  radix32_n1_dit_kernel_fwd_avx512_u1   (K≥8,  k-step=8,  prefetch)
  radix32_n1_dit_kernel_bwd_avx512_u1
  radix32_n1_dit_kernel_fwd_avx512_u2   (K≥16, k-step=16, interleaved+prefetch)
  radix32_n1_dit_kernel_bwd_avx512_u2

U=2 interleaves two k-offsets (k, k+8) at sub-FFT granularity:
  Pass 1: for each n2: radix-8(A) → spill_A → radix-8(B) → spill_B
  Pass 2: for each k1: reload_A → twiddle → radix-4 → store_A → same for B
  
Prefetch strategy: before each load batch, prefetch the OTHER pipeline's
addresses so they arrive in L1 during current pipeline's compute.

8×4 decomposition: 4 radix-8 sub-FFTs + 8 radix-4 column combines.
"""

import math, sys

N, N1, N2 = 32, 8, 4

def wN(e, tN):
    e = e % tN
    angle = 2.0 * math.pi * e / tN
    return (math.cos(angle), -math.sin(angle))

def wN_label(e, tN):
    return f"W{tN}_{e % tN}"

def twiddle_is_trivial(e, tN):
    e = e % tN
    if e == 0: return True, 'one'
    if (8 * e) % tN == 0:
        octant = (8 * e) // tN
        types = ['one', 'w8_1', 'neg_j', 'w8_3',
                 'neg_one', 'neg_w8_1', 'pos_j', 'neg_w8_3']
        return True, types[octant % 8]
    return False, 'cmul'

def collect_twiddles():
    tw = set()
    for n2 in range(1, N2):
        for k1 in range(1, N1):
            e = (n2 * k1) % N
            _, typ = twiddle_is_trivial(e, N)
            if typ == 'cmul':
                tw.add((e, N))
    return tw

# ──────────────────────────────────────────────────────────────────
# Emitter with ISA ops
# ──────────────────────────────────────────────────────────────────

class E:
    def __init__(self):
        self.L = []
        self.ind = 1
        self.spill_c = 0
        self.reload_c = 0

    def o(self, s=""): self.L.append("    " * self.ind + s)
    def c(self, s): self.o(f"/* {s} */")
    def b(self): self.L.append("")

    def add(s, a, b): return f"_mm512_add_pd({a}, {b})"
    def sub(s, a, b): return f"_mm512_sub_pd({a}, {b})"
    def mul(s, a, b): return f"_mm512_mul_pd({a}, {b})"
    def neg(s, a):    return f"_mm512_xor_pd({a}, sign_flip)"
    def fma(s, a, b, c): return f"_mm512_fmadd_pd({a}, {b}, {c})"
    def fms(s, a, b, c): return f"_mm512_fmsub_pd({a}, {b}, {c})"

    # ── Load/store with k-offset parameter ──

    def emit_load(s, var, n, k_expr="k"):
        s.o(f"{var}_re = LD(&in_re[{n} * K + {k_expr}]);")
        s.o(f"{var}_im = LD(&in_im[{n} * K + {k_expr}]);")

    def emit_store(s, var, m, k_expr="k"):
        s.o(f"ST(&out_re[{m} * K + {k_expr}], {var}_re);")
        s.o(f"ST(&out_im[{m} * K + {k_expr}], {var}_im);")

    def emit_spill(s, var, slot):
        s.o(f"_mm512_store_pd(&spill_re[{slot} * 8], {var}_re);")
        s.o(f"_mm512_store_pd(&spill_im[{slot} * 8], {var}_im);")
        s.spill_c += 1

    def emit_reload(s, var, slot):
        s.o(f"{var}_re = _mm512_load_pd(&spill_re[{slot} * 8]);")
        s.o(f"{var}_im = _mm512_load_pd(&spill_im[{slot} * 8]);")
        s.reload_c += 1

    # ── Prefetch for 8 input addresses of a sub-FFT ──

    def emit_prefetch_subfft(s, n2, k_expr):
        """Prefetch 8 input re+im addresses for sub-FFT n2 at k_expr."""
        for n1 in range(N1):
            idx = N2 * n1 + n2
            s.o(f"_mm_prefetch((const char*)&in_re[{idx} * K + {k_expr}], _MM_HINT_T0);")
            s.o(f"_mm_prefetch((const char*)&in_im[{idx} * K + {k_expr}], _MM_HINT_T0);")

    # ── Radix-8 DIT butterfly ──

    def emit_radix8(s, v, d, label=""):
        fwd = (d == 'fwd')
        if label: s.c(f"{label} [{d}]")
        s.o(f"{{ __m512d e0r,e0i,e1r,e1i,e2r,e2i,e3r,e3i;")
        s.o(f"  __m512d t0r,t0i,t1r,t1i,t2r,t2i,t3r,t3i;")
        s.o(f"  t0r={s.add(f'{v[0]}_re',f'{v[4]}_re')}; t0i={s.add(f'{v[0]}_im',f'{v[4]}_im')};")
        s.o(f"  t1r={s.sub(f'{v[0]}_re',f'{v[4]}_re')}; t1i={s.sub(f'{v[0]}_im',f'{v[4]}_im')};")
        s.o(f"  t2r={s.add(f'{v[2]}_re',f'{v[6]}_re')}; t2i={s.add(f'{v[2]}_im',f'{v[6]}_im')};")
        s.o(f"  t3r={s.sub(f'{v[2]}_re',f'{v[6]}_re')}; t3i={s.sub(f'{v[2]}_im',f'{v[6]}_im')};")
        s.o(f"  e0r={s.add('t0r','t2r')}; e0i={s.add('t0i','t2i')};")
        s.o(f"  e2r={s.sub('t0r','t2r')}; e2i={s.sub('t0i','t2i')};")
        if fwd:
            s.o(f"  e1r={s.add('t1r','t3i')}; e1i={s.sub('t1i','t3r')};")
            s.o(f"  e3r={s.sub('t1r','t3i')}; e3i={s.add('t1i','t3r')};")
        else:
            s.o(f"  e1r={s.sub('t1r','t3i')}; e1i={s.add('t1i','t3r')};")
            s.o(f"  e3r={s.add('t1r','t3i')}; e3i={s.sub('t1i','t3r')};")
        s.o(f"  __m512d o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;")
        s.o(f"  t0r={s.add(f'{v[1]}_re',f'{v[5]}_re')}; t0i={s.add(f'{v[1]}_im',f'{v[5]}_im')};")
        s.o(f"  t1r={s.sub(f'{v[1]}_re',f'{v[5]}_re')}; t1i={s.sub(f'{v[1]}_im',f'{v[5]}_im')};")
        s.o(f"  t2r={s.add(f'{v[3]}_re',f'{v[7]}_re')}; t2i={s.add(f'{v[3]}_im',f'{v[7]}_im')};")
        s.o(f"  t3r={s.sub(f'{v[3]}_re',f'{v[7]}_re')}; t3i={s.sub(f'{v[3]}_im',f'{v[7]}_im')};")
        s.o(f"  o0r={s.add('t0r','t2r')}; o0i={s.add('t0i','t2i')};")
        s.o(f"  o2r={s.sub('t0r','t2r')}; o2i={s.sub('t0i','t2i')};")
        if fwd:
            s.o(f"  o1r={s.add('t1r','t3i')}; o1i={s.sub('t1i','t3r')};")
            s.o(f"  o3r={s.sub('t1r','t3i')}; o3i={s.add('t1i','t3r')};")
        else:
            s.o(f"  o1r={s.sub('t1r','t3i')}; o1i={s.add('t1i','t3r')};")
            s.o(f"  o3r={s.add('t1r','t3i')}; o3i={s.sub('t1i','t3r')};")
        # W₈ twiddles on odds
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
        # Combine
        s.o(f"  {v[0]}_re={s.add('e0r','o0r')}; {v[0]}_im={s.add('e0i','o0i')};")
        s.o(f"  {v[4]}_re={s.sub('e0r','o0r')}; {v[4]}_im={s.sub('e0i','o0i')};")
        s.o(f"  {v[1]}_re={s.add('e1r','o1r')}; {v[1]}_im={s.add('e1i','o1i')};")
        s.o(f"  {v[5]}_re={s.sub('e1r','o1r')}; {v[5]}_im={s.sub('e1i','o1i')};")
        s.o(f"  {v[2]}_re={s.add('e2r','o2r')}; {v[2]}_im={s.add('e2i','o2i')};")
        s.o(f"  {v[6]}_re={s.sub('e2r','o2r')}; {v[6]}_im={s.sub('e2i','o2i')};")
        s.o(f"  {v[3]}_re={s.add('e3r','o3r')}; {v[3]}_im={s.add('e3i','o3i')};")
        s.o(f"  {v[7]}_re={s.sub('e3r','o3r')}; {v[7]}_im={s.sub('e3i','o3i')};")
        s.o(f"}}")

    # ── Radix-4 butterfly ──

    def emit_radix4(s, v, d, label=""):
        fwd = (d == 'fwd')
        if label: s.c(f"{label} [{d}]")
        a, b, c, dd = v[0], v[1], v[2], v[3]
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

    # ── Twiddle application ──

    def emit_twiddle(s, dst, src, e, tN, d):
        _, typ = twiddle_is_trivial(e, tN)
        fwd = (d == 'fwd')
        if typ == 'one':
            if dst != src: s.o(f"{dst}_re={src}_re; {dst}_im={src}_im;")
        elif typ == 'neg_one':
            s.o(f"{dst}_re={s.neg(f'{src}_re')}; {dst}_im={s.neg(f'{src}_im')};")
        elif typ == 'neg_j':
            if fwd:
                s.o(f"{{ __m512d t={src}_re; {dst}_re={src}_im; {dst}_im={s.neg('t')}; }}")
            else:
                s.o(f"{{ __m512d t={src}_re; {dst}_re={s.neg(f'{src}_im')}; {dst}_im=t; }}")
        elif typ == 'pos_j':
            if fwd:
                s.o(f"{{ __m512d t={src}_re; {dst}_re={s.neg(f'{src}_im')}; {dst}_im=t; }}")
            else:
                s.o(f"{{ __m512d t={src}_re; {dst}_re={src}_im; {dst}_im={s.neg('t')}; }}")
        elif typ == 'w8_1':
            s.o(f"{{ __m512d tr={src}_re, ti={src}_im;")
            if fwd:
                s.o(f"  {dst}_re={s.mul(s.add('tr','ti'),'sqrt2_inv')};")
                s.o(f"  {dst}_im={s.mul(s.sub('ti','tr'),'sqrt2_inv')}; }}")
            else:
                s.o(f"  {dst}_re={s.mul(s.sub('tr','ti'),'sqrt2_inv')};")
                s.o(f"  {dst}_im={s.mul(s.add('tr','ti'),'sqrt2_inv')}; }}")
        elif typ == 'w8_3':
            s.o(f"{{ __m512d tr={src}_re, ti={src}_im;")
            if fwd:
                s.o(f"  {dst}_re={s.mul(s.sub('ti','tr'),'sqrt2_inv')};")
                s.o(f"  {dst}_im={s.neg(s.mul(s.add('tr','ti'),'sqrt2_inv'))}; }}")
            else:
                s.o(f"  {dst}_re={s.neg(s.mul(s.add('tr','ti'),'sqrt2_inv'))};")
                s.o(f"  {dst}_im={s.mul(s.sub('tr','ti'),'sqrt2_inv')}; }}")
        elif typ == 'neg_w8_1':
            s.o(f"{{ __m512d tr={src}_re, ti={src}_im;")
            if fwd:
                s.o(f"  {dst}_re={s.neg(s.mul(s.add('tr','ti'),'sqrt2_inv'))};")
                s.o(f"  {dst}_im={s.mul(s.sub('tr','ti'),'sqrt2_inv')}; }}")
            else:
                s.o(f"  {dst}_re={s.mul(s.sub('ti','tr'),'sqrt2_inv')};")
                s.o(f"  {dst}_im={s.neg(s.mul(s.add('tr','ti'),'sqrt2_inv'))}; }}")
        elif typ == 'neg_w8_3':
            s.o(f"{{ __m512d tr={src}_re, ti={src}_im;")
            if fwd:
                s.o(f"  {dst}_re={s.mul(s.sub('tr','ti'),'sqrt2_inv')};")
                s.o(f"  {dst}_im={s.mul(s.add('tr','ti'),'sqrt2_inv')}; }}")
            else:
                s.o(f"  {dst}_re={s.mul(s.add('tr','ti'),'sqrt2_inv')};")
                s.o(f"  {dst}_im={s.mul(s.sub('ti','tr'),'sqrt2_inv')}; }}")
        else:
            label = wN_label(e, tN)
            s.o(f"{{ __m512d tr={src}_re;")
            if fwd:
                s.o(f"  {dst}_re={s.fms(f'{src}_re',f'tw_{label}_re',s.mul(f'{src}_im',f'tw_{label}_im'))};")
                s.o(f"  {dst}_im={s.fma('tr',f'tw_{label}_im',s.mul(f'{src}_im',f'tw_{label}_re'))}; }}")
            else:
                s.o(f"  {dst}_re={s.fma(f'{src}_re',f'tw_{label}_re',s.mul(f'{src}_im',f'tw_{label}_im'))};")
                s.o(f"  {dst}_im={s.fms(f'{src}_im',f'tw_{label}_re',s.mul('tr',f'tw_{label}_im'))}; }}")

# ──────────────────────────────────────────────────────────────────
# Kernel preamble (shared between U=1 and U=2)
# ──────────────────────────────────────────────────────────────────

def emit_preamble(em, d, tw_set, u, spill_slots):
    em.o("const __m512d sign_flip = _mm512_set1_pd(-0.0);")
    em.o("const __m512d sqrt2_inv = _mm512_set1_pd(0.70710678118654752440);")
    em.b()
    em.o(f"__attribute__((aligned(64))) double spill_re[{spill_slots * 8}];")
    em.o(f"__attribute__((aligned(64))) double spill_im[{spill_slots * 8}];")
    em.b()
    em.o("__m512d x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
    em.o("__m512d x4_re,x4_im,x5_re,x5_im,x6_re,x6_im,x7_re,x7_im;")
    em.b()

    if tw_set:
        em.c(f"Hoisted W₃₂ broadcasts [{d}]")
        for (e, tN) in sorted(tw_set):
            label = wN_label(e, tN)
            em.o(f"const __m512d tw_{label}_re = _mm512_set1_pd({label}_re);")
            em.o(f"const __m512d tw_{label}_im = _mm512_set1_pd({label}_im);")
        em.b()

# ──────────────────────────────────────────────────────────────────
# U=1 kernel with prefetch
# ──────────────────────────────────────────────────────────────────

def emit_kernel_u1(em, d, tw_set):
    em.L.append(f"static __attribute__((target(\"avx512f,avx512dq,fma\"))) void")
    em.L.append(f"radix32_n1_dit_kernel_{d}_avx512_u1(")
    em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
    em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
    em.L.append(f"    size_t K)")
    em.L.append(f"{{")
    em.ind = 1
    em.spill_c = 0; em.reload_c = 0

    emit_preamble(em, d, tw_set, 1, 32)

    em.o("for (size_t k = 0; k < K; k += 8) {")
    em.ind += 1

    xv8 = [f"x{i}" for i in range(8)]
    xv4 = [f"x{i}" for i in range(4)]

    # PASS 1
    em.c(f"PASS 1: {N2} radix-{N1} sub-FFTs [{d}]")
    em.b()
    for n2 in range(N2):
        em.c(f"sub-FFT n₂={n2}")
        # Prefetch next sub-FFT (or next iteration's first sub-FFT)
        pf_n2 = (n2 + 1) % N2
        pf_k = "k + 8" if n2 == N2 - 1 else "k"
        em.c(f"prefetch n₂={pf_n2} at k_pf={pf_k}")
        em.emit_prefetch_subfft(pf_n2, pf_k)
        em.b()
        for n1 in range(N1):
            em.emit_load(f"x{n1}", N2 * n1 + n2)
        em.b()
        em.emit_radix8(xv8, d, f"radix-8 n₂={n2}")
        em.b()
        for k1 in range(N1):
            em.emit_spill(f"x{k1}", n2 * N1 + k1)
        em.b()

    # PASS 2
    em.c(f"PASS 2: {N1} radix-{N2} combines [{d}]")
    em.b()
    for k1 in range(N1):
        em.c(f"column k₁={k1}")
        for n2 in range(N2):
            em.emit_reload(f"x{n2}", n2 * N1 + k1)
        em.b()
        if k1 > 0:
            for n2 in range(1, N2):
                e = (n2 * k1) % N
                em.emit_twiddle(f"x{n2}", f"x{n2}", e, N, d)
            em.b()
        em.emit_radix4(xv4, d, f"radix-4 k₁={k1}")
        em.b()
        for k2 in range(N2):
            em.emit_store(f"x{k2}", k1 + N1 * k2)
        em.b()

    em.ind -= 1
    em.o("}")
    em.L.append("}")
    em.L.append("")
    return em.spill_c, em.reload_c

# ──────────────────────────────────────────────────────────────────
# U=2 kernel: interleaved dual pipeline + prefetch
# ──────────────────────────────────────────────────────────────────

def emit_kernel_u2(em, d, tw_set):
    em.L.append(f"static __attribute__((target(\"avx512f,avx512dq,fma\"))) void")
    em.L.append(f"radix32_n1_dit_kernel_{d}_avx512_u2(")
    em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
    em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
    em.L.append(f"    size_t K)")
    em.L.append(f"{{")
    em.ind = 1
    em.spill_c = 0; em.reload_c = 0

    # 64 spill slots: 0-31 for pipeline A, 32-63 for pipeline B
    emit_preamble(em, d, tw_set, 2, 64)

    em.o("for (size_t k = 0; k < K; k += 16) {")
    em.ind += 1

    xv8 = [f"x{i}" for i in range(8)]
    xv4 = [f"x{i}" for i in range(4)]
    SPILL_B_OFF = 32  # B region starts at slot 32

    # ═══ PASS 1: interleaved sub-FFTs ═══
    em.c(f"PASS 1: {N2} radix-{N1} sub-FFTs × 2 pipelines [{d}]")
    em.b()

    for n2 in range(N2):
        # ── Pipeline A at k ──
        em.c(f"═══ pipeline A, sub-FFT n₂={n2} ═══")
        # Prefetch B's inputs for this sub-FFT (arrive during A's compute)
        em.c(f"prefetch B n₂={n2}")
        em.emit_prefetch_subfft(n2, "k + 8")
        em.b()
        for n1 in range(N1):
            em.emit_load(f"x{n1}", N2 * n1 + n2, "k")
        em.b()
        em.emit_radix8(xv8, d, f"A: radix-8 n₂={n2}")
        em.b()
        for k1 in range(N1):
            em.emit_spill(f"x{k1}", n2 * N1 + k1)  # A region
        em.b()

        # ── Pipeline B at k+8 ──
        em.c(f"═══ pipeline B, sub-FFT n₂={n2} ═══")
        # Prefetch A's next sub-FFT (or next iteration)
        pf_n2 = (n2 + 1) % N2
        pf_k = "k + 16" if n2 == N2 - 1 else "k"
        em.c(f"prefetch A n₂={pf_n2} at {pf_k}")
        em.emit_prefetch_subfft(pf_n2, pf_k)
        em.b()
        for n1 in range(N1):
            em.emit_load(f"x{n1}", N2 * n1 + n2, "k + 8")
        em.b()
        em.emit_radix8(xv8, d, f"B: radix-8 n₂={n2}")
        em.b()
        for k1 in range(N1):
            em.emit_spill(f"x{k1}", SPILL_B_OFF + n2 * N1 + k1)  # B region
        em.b()

    # ═══ PASS 2: interleaved combines ═══
    em.c(f"PASS 2: {N1} radix-{N2} combines × 2 pipelines [{d}]")
    em.b()

    for k1 in range(N1):
        # ── Pipeline A ──
        em.c(f"═══ A, column k₁={k1} ═══")
        for n2 in range(N2):
            em.emit_reload(f"x{n2}", n2 * N1 + k1)
        em.b()
        if k1 > 0:
            for n2 in range(1, N2):
                e = (n2 * k1) % N
                em.emit_twiddle(f"x{n2}", f"x{n2}", e, N, d)
            em.b()
        em.emit_radix4(xv4, d, f"A: radix-4 k₁={k1}")
        em.b()
        for k2 in range(N2):
            em.emit_store(f"x{k2}", k1 + N1 * k2, "k")
        em.b()

        # ── Pipeline B ──
        em.c(f"═══ B, column k₁={k1} ═══")
        for n2 in range(N2):
            em.emit_reload(f"x{n2}", SPILL_B_OFF + n2 * N1 + k1)
        em.b()
        if k1 > 0:
            for n2 in range(1, N2):
                e = (n2 * k1) % N
                em.emit_twiddle(f"x{n2}", f"x{n2}", e, N, d)
            em.b()
        em.emit_radix4(xv4, d, f"B: radix-4 k₁={k1}")
        em.b()
        for k2 in range(N2):
            em.emit_store(f"x{k2}", k1 + N1 * k2, "k + 8")
        em.b()

    em.ind -= 1
    em.o("}")
    em.L.append("}")
    em.L.append("")
    return em.spill_c, em.reload_c

# ──────────────────────────────────────────────────────────────────
# File generation
# ──────────────────────────────────────────────────────────────────

def main():
    tw_set = collect_twiddles()
    em = E()

    em.L.append("/**")
    em.L.append(" * @file fft_radix32_avx512_n1_u2.h")
    em.L.append(" * @brief DFT-32 AVX-512 N1 — U=1 (prefetch) + U=2 (interleaved+prefetch)")
    em.L.append(" *")
    em.L.append(" * U=1: k-step=8,  32-slot spill (2KB), software prefetch")
    em.L.append(" * U=2: k-step=16, 64-slot spill (4KB), interleaved A/B pipelines + prefetch")
    em.L.append(" * 8×4: 4 radix-8 sub-FFTs + 8 radix-4 combines")
    em.L.append(" * Generated by gen_radix32_avx512_n1_u2.py")
    em.L.append(" */")
    em.L.append("")
    em.L.append("#ifndef FFT_RADIX32_AVX512_N1_U2_H")
    em.L.append("#define FFT_RADIX32_AVX512_N1_U2_H")
    em.L.append("")
    em.L.append("#include <immintrin.h>")
    em.L.append("")

    # Twiddle constants
    by_tN = {}
    for (e, tN) in sorted(tw_set):
        by_tN.setdefault(tN, []).append(e)
    for tN in sorted(by_tN):
        guard = f"FFT_W{tN}_TWIDDLES_DEFINED"
        em.L.append(f"#ifndef {guard}")
        em.L.append(f"#define {guard}")
        for e in sorted(by_tN[tN]):
            wr, wi = wN(e, tN)
            label = wN_label(e, tN)
            em.L.append(f"static const double {label}_re = {wr:.20e};")
            em.L.append(f"static const double {label}_im = {wi:.20e};")
        em.L.append(f"#endif")
        em.L.append("")

    # Load/store macros
    em.L.append("#ifndef R32U_LD")
    em.L.append("#define R32U_LD(p) _mm512_loadu_pd(p)")
    em.L.append("#endif")
    em.L.append("#ifndef R32U_ST")
    em.L.append("#define R32U_ST(p,v) _mm512_storeu_pd((p),(v))")
    em.L.append("#endif")
    em.L.append("")
    em.L.append("#define LD R32U_LD")
    em.L.append("#define ST R32U_ST")
    em.L.append("")

    # U=1 kernels
    em.L.append("/* ════════════════════════════════════════════════════════════ */")
    em.L.append("/*  U=1 KERNELS — k-step=8, prefetch, K≥8                    */")
    em.L.append("/* ════════════════════════════════════════════════════════════ */")
    em.L.append("")
    u1f_s, u1f_r = emit_kernel_u1(em, 'fwd', tw_set)
    u1b_s, u1b_r = emit_kernel_u1(em, 'bwd', tw_set)

    # U=2 kernels
    em.L.append("/* ════════════════════════════════════════════════════════════ */")
    em.L.append("/*  U=2 KERNELS — k-step=16, interleaved pipelines, K≥16     */")
    em.L.append("/* ════════════════════════════════════════════════════════════ */")
    em.L.append("")
    u2f_s, u2f_r = emit_kernel_u2(em, 'fwd', tw_set)
    u2b_s, u2b_r = emit_kernel_u2(em, 'bwd', tw_set)

    em.L.append("#undef LD")
    em.L.append("#undef ST")
    em.L.append("")
    em.L.append("#endif /* FFT_RADIX32_AVX512_N1_U2_H */")

    print("\n".join(em.L))

    print(f"\n=== DFT-32 AVX-512 U=1 ===", file=sys.stderr)
    print(f"  Forward:  {u1f_s} spills + {u1f_r} reloads = {u1f_s+u1f_r} L1 ops", file=sys.stderr)
    print(f"  Backward: {u1b_s} spills + {u1b_r} reloads = {u1b_s+u1b_r} L1 ops", file=sys.stderr)
    print(f"\n=== DFT-32 AVX-512 U=2 ===", file=sys.stderr)
    print(f"  Forward:  {u2f_s} spills + {u2f_r} reloads = {u2f_s+u2f_r} L1 ops", file=sys.stderr)
    print(f"  Backward: {u2b_s} spills + {u2b_r} reloads = {u2b_s+u2b_r} L1 ops", file=sys.stderr)
    print(f"  Twiddles: {len(tw_set)} generic W₃₂", file=sys.stderr)
    print(f"  Lines:    {len(em.L)}", file=sys.stderr)

if __name__ == '__main__':
    main()