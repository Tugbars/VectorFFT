#!/usr/bin/env python3
"""
gen_radix32_scalar.py — DFT-32 scalar codegen (N1 + twiddled)

Plain C, no intrinsics. k-step = 1.
8×4 decomposition: 4 radix-8 sub-FFTs + 8 radix-4 column combines.
Spill: 32 doubles re + 32 doubles im = 512 bytes stack.

Emits:
  radix32_n1_dit_kernel_{fwd,bwd}_scalar     (N1, no twiddles)
  radix32_tw_flat_dit_kernel_{fwd,bwd}_scalar (flat tw, 31 loads/k-step)
"""

import math, sys

N, N1, N2 = 32, 8, 4

def wN(e, tN):
    e = e % tN
    a = 2.0 * math.pi * e / tN
    return (math.cos(a), -math.sin(a))

def wN_label(e, tN):
    return f"W{tN}_{e % tN}"

def twiddle_is_trivial(e, tN):
    e = e % tN
    if e == 0: return True, 'one'
    if (8 * e) % tN == 0:
        o = (8 * e) // tN
        t = ['one','w8_1','neg_j','w8_3','neg_one','neg_w8_1','pos_j','neg_w8_3']
        return True, t[o % 8]
    return False, 'cmul'

def collect_twiddles():
    tw = set()
    for n2 in range(1, N2):
        for k1 in range(1, N1):
            e = (n2 * k1) % N
            _, typ = twiddle_is_trivial(e, N)
            if typ == 'cmul': tw.add((e, N))
    return tw


class E:
    """Scalar emitter — plain C doubles."""
    def __init__(self):
        self.L = []; self.ind = 1; self.spill_c = 0; self.reload_c = 0

    def o(s, t=""): s.L.append("    " * s.ind + t)
    def c(s, t): s.o(f"/* {t} */")
    def b(s): s.L.append("")

    def emit_load(s, v, n, k_expr="k"):
        s.o(f"double {v}_re = in_re[{n}*K+{k_expr}];")
        s.o(f"double {v}_im = in_im[{n}*K+{k_expr}];")

    def emit_load_nodef(s, v, n, k_expr="k"):
        s.o(f"{v}_re = in_re[{n}*K+{k_expr}];")
        s.o(f"{v}_im = in_im[{n}*K+{k_expr}];")

    def emit_store(s, v, m, k_expr="k"):
        s.o(f"out_re[{m}*K+{k_expr}] = {v}_re;")
        s.o(f"out_im[{m}*K+{k_expr}] = {v}_im;")

    def emit_spill(s, v, slot):
        s.o(f"spill_re[{slot}] = {v}_re;")
        s.o(f"spill_im[{slot}] = {v}_im;")
        s.spill_c += 1

    def emit_reload(s, v, slot):
        s.o(f"double {v}_re = spill_re[{slot}];")
        s.o(f"double {v}_im = spill_im[{slot}];")
        s.reload_c += 1

    def emit_reload_nodef(s, v, slot):
        s.o(f"{v}_re = spill_re[{slot}];")
        s.o(f"{v}_im = spill_im[{slot}];")
        s.reload_c += 1

    def emit_radix8(s, v, d, label=""):
        fwd = (d == 'fwd')
        if label: s.c(f"{label} [{d}]")
        s.o("{")
        s.ind += 1
        # Even: v0,v2,v4,v6 → e0,e1,e2,e3
        s.o(f"double t0r={v[0]}_re+{v[4]}_re, t0i={v[0]}_im+{v[4]}_im;")
        s.o(f"double t1r={v[0]}_re-{v[4]}_re, t1i={v[0]}_im-{v[4]}_im;")
        s.o(f"double t2r={v[2]}_re+{v[6]}_re, t2i={v[2]}_im+{v[6]}_im;")
        s.o(f"double t3r={v[2]}_re-{v[6]}_re, t3i={v[2]}_im-{v[6]}_im;")
        s.o(f"double e0r=t0r+t2r, e0i=t0i+t2i;")
        s.o(f"double e2r=t0r-t2r, e2i=t0i-t2i;")
        if fwd:
            s.o(f"double e1r=t1r+t3i, e1i=t1i-t3r;")
            s.o(f"double e3r=t1r-t3i, e3i=t1i+t3r;")
        else:
            s.o(f"double e1r=t1r-t3i, e1i=t1i+t3r;")
            s.o(f"double e3r=t1r+t3i, e3i=t1i-t3r;")
        # Odd: v1,v3,v5,v7 → o0,o1,o2,o3
        s.o(f"t0r={v[1]}_re+{v[5]}_re; t0i={v[1]}_im+{v[5]}_im;")
        s.o(f"t1r={v[1]}_re-{v[5]}_re; t1i={v[1]}_im-{v[5]}_im;")
        s.o(f"t2r={v[3]}_re+{v[7]}_re; t2i={v[3]}_im+{v[7]}_im;")
        s.o(f"t3r={v[3]}_re-{v[7]}_re; t3i={v[3]}_im-{v[7]}_im;")
        s.o(f"double o0r=t0r+t2r, o0i=t0i+t2i;")
        s.o(f"double o2r=t0r-t2r, o2i=t0i-t2i;")
        if fwd:
            s.o(f"double o1r=t1r+t3i, o1i=t1i-t3r;")
            s.o(f"double o3r=t1r-t3i, o3i=t1i+t3r;")
        else:
            s.o(f"double o1r=t1r-t3i, o1i=t1i+t3r;")
            s.o(f"double o3r=t1r+t3i, o3i=t1i-t3r;")
        # W8 rotations
        SQ = "0.70710678118654752440"
        if fwd:
            s.o(f"{{ double tr=(o1r+o1i)*{SQ}, ti=(o1i-o1r)*{SQ}; o1r=tr; o1i=ti; }}")
            s.o(f"{{ double tr=o2i, ti=-o2r; o2r=tr; o2i=ti; }}")
            s.o(f"{{ double tr=(o3i-o3r)*{SQ}, ti=-(o3r+o3i)*{SQ}; o3r=tr; o3i=ti; }}")
        else:
            s.o(f"{{ double tr=(o1r-o1i)*{SQ}, ti=(o1r+o1i)*{SQ}; o1r=tr; o1i=ti; }}")
            s.o(f"{{ double tr=-o2i, ti=o2r; o2r=tr; o2i=ti; }}")
            s.o(f"{{ double tr=-(o3r+o3i)*{SQ}, ti=(o3r-o3i)*{SQ}; o3r=tr; o3i=ti; }}")
        # Combine
        s.o(f"{v[0]}_re=e0r+o0r; {v[0]}_im=e0i+o0i;")
        s.o(f"{v[4]}_re=e0r-o0r; {v[4]}_im=e0i-o0i;")
        s.o(f"{v[1]}_re=e1r+o1r; {v[1]}_im=e1i+o1i;")
        s.o(f"{v[5]}_re=e1r-o1r; {v[5]}_im=e1i-o1i;")
        s.o(f"{v[2]}_re=e2r+o2r; {v[2]}_im=e2i+o2i;")
        s.o(f"{v[6]}_re=e2r-o2r; {v[6]}_im=e2i-o2i;")
        s.o(f"{v[3]}_re=e3r+o3r; {v[3]}_im=e3i+o3i;")
        s.o(f"{v[7]}_re=e3r-o3r; {v[7]}_im=e3i-o3i;")
        s.ind -= 1
        s.o("}")

    def emit_radix4(s, v, d, label=""):
        fwd = (d == 'fwd')
        if label: s.c(f"{label} [{d}]")
        a, b, c, dd = v[0], v[1], v[2], v[3]
        s.o("{")
        s.ind += 1
        s.o(f"double t0r={a}_re+{c}_re, t0i={a}_im+{c}_im;")
        s.o(f"double t1r={a}_re-{c}_re, t1i={a}_im-{c}_im;")
        s.o(f"double t2r={b}_re+{dd}_re, t2i={b}_im+{dd}_im;")
        s.o(f"double t3r={b}_re-{dd}_re, t3i={b}_im-{dd}_im;")
        s.o(f"{a}_re=t0r+t2r; {a}_im=t0i+t2i;")
        s.o(f"{c}_re=t0r-t2r; {c}_im=t0i-t2i;")
        if fwd:
            s.o(f"{b}_re=t1r+t3i; {b}_im=t1i-t3r;")
            s.o(f"{dd}_re=t1r-t3i; {dd}_im=t1i+t3r;")
        else:
            s.o(f"{b}_re=t1r-t3i; {b}_im=t1i+t3r;")
            s.o(f"{dd}_re=t1r+t3i; {dd}_im=t1i-t3r;")
        s.ind -= 1
        s.o("}")

    def emit_twiddle(s, dst, src, e, tN, d):
        _, typ = twiddle_is_trivial(e, tN)
        fwd = (d == 'fwd')
        if typ == 'one':
            if dst != src: s.o(f"{dst}_re={src}_re; {dst}_im={src}_im;")
        elif typ == 'neg_one':
            s.o(f"{dst}_re=-{src}_re; {dst}_im=-{src}_im;")
        elif typ == 'neg_j':
            if fwd: s.o(f"{{ double t={src}_re; {dst}_re={src}_im; {dst}_im=-t; }}")
            else:   s.o(f"{{ double t={src}_re; {dst}_re=-{src}_im; {dst}_im=t; }}")
        elif typ == 'pos_j':
            if fwd: s.o(f"{{ double t={src}_re; {dst}_re=-{src}_im; {dst}_im=t; }}")
            else:   s.o(f"{{ double t={src}_re; {dst}_re={src}_im; {dst}_im=-t; }}")
        elif typ in ('w8_1','w8_3','neg_w8_1','neg_w8_3'):
            SQ = "0.70710678118654752440"
            s.o(f"{{ double tr={src}_re, ti={src}_im;")
            if typ == 'w8_1':
                if fwd: s.o(f"  {dst}_re=(tr+ti)*{SQ}; {dst}_im=(ti-tr)*{SQ}; }}")
                else:   s.o(f"  {dst}_re=(tr-ti)*{SQ}; {dst}_im=(tr+ti)*{SQ}; }}")
            elif typ == 'w8_3':
                if fwd: s.o(f"  {dst}_re=(ti-tr)*{SQ}; {dst}_im=-(tr+ti)*{SQ}; }}")
                else:   s.o(f"  {dst}_re=-(tr+ti)*{SQ}; {dst}_im=(tr-ti)*{SQ}; }}")
            elif typ == 'neg_w8_1':
                if fwd: s.o(f"  {dst}_re=-(tr+ti)*{SQ}; {dst}_im=(tr-ti)*{SQ}; }}")
                else:   s.o(f"  {dst}_re=(ti-tr)*{SQ}; {dst}_im=-(tr+ti)*{SQ}; }}")
            elif typ == 'neg_w8_3':
                if fwd: s.o(f"  {dst}_re=(tr-ti)*{SQ}; {dst}_im=(tr+ti)*{SQ}; }}")
                else:   s.o(f"  {dst}_re=(tr+ti)*{SQ}; {dst}_im=(ti-tr)*{SQ}; }}")
        else:
            label = wN_label(e, tN)
            s.o(f"{{ double tr={src}_re;")
            if fwd:
                s.o(f"  {dst}_re={src}_re*{label}_re-{src}_im*{label}_im;")
                s.o(f"  {dst}_im=tr*{label}_im+{src}_im*{label}_re; }}")
            else:
                s.o(f"  {dst}_re={src}_re*{label}_re+{src}_im*{label}_im;")
                s.o(f"  {dst}_im={src}_im*{label}_re-tr*{label}_im; }}")

    def emit_cmul_inplace(s, v, wr, wi, d):
        fwd = (d == 'fwd')
        s.o(f"{{ double tr={v}_re;")
        if fwd:
            s.o(f"  {v}_re={v}_re*{wr}-{v}_im*{wi};")
            s.o(f"  {v}_im=tr*{wi}+{v}_im*{wr}; }}")
        else:
            s.o(f"  {v}_re={v}_re*{wr}+{v}_im*{wi};")
            s.o(f"  {v}_im={v}_im*{wr}-tr*{wi}; }}")


def emit_n1_kernel(em, d, tw_set):
    """N1 twiddle-less kernel."""
    em.L.append(f"static void radix32_n1_dit_kernel_{d}_scalar(")
    em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
    em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
    em.L.append(f"    size_t K)")
    em.L.append(f"{{")
    em.ind = 1
    em.spill_c = 0; em.reload_c = 0

    em.o("double spill_re[32], spill_im[32];")
    em.b()

    em.o("for (size_t k = 0; k < K; k++) {")
    em.ind += 1

    # Declare all working variables at loop top
    em.o("double x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
    em.o("double x4_re,x4_im,x5_re,x5_im,x6_re,x6_im,x7_re,x7_im;")
    em.b()

    xv8 = [f"x{i}" for i in range(8)]
    xv4 = [f"x{i}" for i in range(4)]

    # PASS 1
    em.c(f"PASS 1: 4 radix-8 sub-FFTs [{d}]")
    em.b()
    for n2 in range(N2):
        em.c(f"sub-FFT n₂={n2}")
        for n1 in range(N1):
            em.emit_load_nodef(f"x{n1}", N2 * n1 + n2)
        em.b()
        em.emit_radix8(xv8, d, f"radix-8 n₂={n2}")
        em.b()
        for k1 in range(N1):
            em.emit_spill(f"x{k1}", n2 * N1 + k1)
        em.b()

    # PASS 2
    em.c(f"PASS 2: 8 radix-4 combines [{d}]")
    em.b()
    for k1 in range(N1):
        em.c(f"column k₁={k1}")
        for n2 in range(N2):
            em.emit_reload_nodef(f"x{n2}", n2 * N1 + k1)
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


def emit_tw_kernel(em, d, tw_set):
    """Flat twiddled kernel."""
    em.L.append(f"static void radix32_tw_flat_dit_kernel_{d}_scalar(")
    em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
    em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
    em.L.append(f"    const double * __restrict__ tw_re, const double * __restrict__ tw_im,")
    em.L.append(f"    size_t K)")
    em.L.append(f"{{")
    em.ind = 1
    em.spill_c = 0; em.reload_c = 0

    em.o("double spill_re[32], spill_im[32];")
    em.b()

    fwd = (d == 'fwd')

    em.o("for (size_t k = 0; k < K; k++) {")
    em.ind += 1

    # Declare all working variables at loop top
    em.o("double x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
    em.o("double x4_re,x4_im,x5_re,x5_im,x6_re,x6_im,x7_re,x7_im;")
    em.b()

    xv8 = [f"x{i}" for i in range(8)]
    xv4 = [f"x{i}" for i in range(4)]

    em.c(f"PASS 1: 4 radix-8 sub-FFTs + external twiddles [{d}]")
    em.b()
    for n2 in range(N2):
        em.c(f"sub-FFT n₂={n2}")
        for n1 in range(N1):
            n = N2 * n1 + n2
            em.emit_load_nodef(f"x{n1}", n)
            if n > 0:
                em.emit_cmul_inplace(f"x{n1}",
                    f"tw_re[{n-1}*K+k]", f"tw_im[{n-1}*K+k]", d)
        em.b()
        em.emit_radix8(xv8, d, f"radix-8 n₂={n2}")
        em.b()
        for k1 in range(N1):
            em.emit_spill(f"x{k1}", n2 * N1 + k1)
        em.b()

    em.c(f"PASS 2: 8 radix-4 combines [{d}]")
    em.b()
    for k1 in range(N1):
        em.c(f"column k₁={k1}")
        for n2 in range(N2):
            em.emit_reload_nodef(f"x{n2}", n2 * N1 + k1)
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


def main():
    tw_set = collect_twiddles()
    em = E()

    em.L.append("/**")
    em.L.append(" * @file fft_radix32_scalar_gen.h")
    em.L.append(" * @brief DFT-32 scalar codelet — N1 + flat twiddled")
    em.L.append(" *")
    em.L.append(" * Plain C, no intrinsics. k-step=1.")
    em.L.append(" * 8×4 decomposition, 32-slot stack spill (512 bytes).")
    em.L.append(f" * Internal W₃₂: {len(tw_set)} constants")
    em.L.append(" * Generated by gen_radix32_scalar.py")
    em.L.append(" */")
    em.L.append("")
    em.L.append("#ifndef FFT_RADIX32_SCALAR_GEN_H")
    em.L.append("#define FFT_RADIX32_SCALAR_GEN_H")
    em.L.append("")
    em.L.append("#include <stddef.h>")
    em.L.append("")

    # Twiddle constants
    by_tN = {}
    for (e, tN) in sorted(tw_set):
        by_tN.setdefault(tN, []).append(e)
    for tN in sorted(by_tN):
        g = f"FFT_W{tN}_TWIDDLES_DEFINED"
        em.L.append(f"#ifndef {g}")
        em.L.append(f"#define {g}")
        for e in sorted(by_tN[tN]):
            wr, wi = wN(e, tN)
            l = wN_label(e, tN)
            em.L.append(f"static const double {l}_re = {wr:.20e};")
            em.L.append(f"static const double {l}_im = {wi:.20e};")
        em.L.append(f"#endif")
        em.L.append("")

    # N1 kernels
    em.L.append("/* ══════════ N1 (twiddle-less) ══════════ */")
    em.L.append("")
    n1f = emit_n1_kernel(em, 'fwd', tw_set)
    n1b = emit_n1_kernel(em, 'bwd', tw_set)

    # Twiddled kernels
    em.L.append("/* ══════════ FLAT TWIDDLED ══════════ */")
    em.L.append("")
    twf = emit_tw_kernel(em, 'fwd', tw_set)
    twb = emit_tw_kernel(em, 'bwd', tw_set)

    em.L.append("#endif /* FFT_RADIX32_SCALAR_GEN_H */")

    print("\n".join(em.L))

    print(f"\n=== DFT-32 SCALAR N1 ===", file=sys.stderr)
    print(f"  fwd: {n1f[0]}sp+{n1f[1]}rl={n1f[0]+n1f[1]} L1 ops", file=sys.stderr)
    print(f"=== DFT-32 SCALAR TW ===", file=sys.stderr)
    print(f"  fwd: {twf[0]}sp+{twf[1]}rl={twf[0]+twf[1]} L1 ops", file=sys.stderr)
    print(f"  W₃₂: {len(tw_set)} constants", file=sys.stderr)
    print(f"  Lines: {len(em.L)}", file=sys.stderr)

if __name__ == '__main__':
    main()
