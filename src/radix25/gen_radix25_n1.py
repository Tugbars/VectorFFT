#!/usr/bin/env python3
"""
gen_radix25_n1.py — DFT-25 N1 (notw) codelet generator

5×5 Cooley-Tukey: 5 radix-5 sub-FFTs + W₂₅ twiddles + 5 radix-5 combines.
No external twiddles — pure butterfly.

Usage:
  python3 gen_radix25_n1.py scalar > fft_radix25_scalar_n1.h
  python3 gen_radix25_n1.py avx512 > fft_radix25_avx512_n1.h
  python3 gen_radix25_n1.py avx2   > fft_radix25_avx2_n1.h
"""
import math, sys

N, N1, N2 = 25, 5, 5

def wN(e, tN):
    e = e % tN; a = 2.0*math.pi*e/tN
    return (math.cos(a), -math.sin(a))

def wN_label(e, tN): return f"W{tN}_{e%tN}"

def collect_internal_twiddles():
    tw = set()
    for n2 in range(1, N2):
        for k1 in range(1, N1):
            e = (n2*k1) % N
            if e != 0:
                tw.add((e, N))
    return tw

# ── Scalar ──────────────────────────────────────────────────

def gen_scalar_kernel(d):
    fwd = (d == 'fwd')
    lines = []
    L = lines.append
    L(f"static inline void")
    L(f"radix25_n1_dit_kernel_{d}_scalar(")
    L(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
    L(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
    L(f"    size_t K)")
    L(f"{{")
    L(f"    double sp_re[25], sp_im[25];")
    L(f"    double x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im,x4_re,x4_im;")
    L(f"")
    L(f"    for (size_t k = 0; k < K; k++) {{")

    xv = ["x0","x1","x2","x3","x4"]

    # Pass 1: 5 radix-5 sub-FFTs
    for n2 in range(N2):
        L(f"        /* sub-FFT n2={n2} */")
        for n1 in range(N1):
            n = N2*n1 + n2
            L(f"        x{n1}_re = in_re[{n}*K+k]; x{n1}_im = in_im[{n}*K+k];")
        # radix-5 butterfly
        a,b,c,dd,xe = xv
        L(f"        {{ double s1r={b}_re+{xe}_re, s1i={b}_im+{xe}_im;")
        L(f"          double s2r={c}_re+{dd}_re, s2i={c}_im+{dd}_im;")
        L(f"          double d1r={b}_re-{xe}_re, d1i={b}_im-{xe}_im;")
        L(f"          double d2r={c}_re-{dd}_re, d2i={c}_im-{dd}_im;")
        L(f"          double t0r={a}_re-0.25*(s1r+s2r), t0i={a}_im-0.25*(s1i+s2i);")
        L(f"          double t1r=0.559016994374947424102293417182819058860154590*(s1r-s2r);")
        L(f"          double t1i=0.559016994374947424102293417182819058860154590*(s1i-s2i);")
        L(f"          double p1r=t0r+t1r, p1i=t0i+t1i, p2r=t0r-t1r, p2i=t0i-t1i;")
        L(f"          double Ti=0.951056516295153572116439333379382143405698634*d1i+0.587785252292473129168705954639072768597652438*d2i;")
        L(f"          double Tu=0.951056516295153572116439333379382143405698634*d1r+0.587785252292473129168705954639072768597652438*d2r;")
        L(f"          double Tk=0.951056516295153572116439333379382143405698634*d2i-0.587785252292473129168705954639072768597652438*d1i;")
        L(f"          double Tv=0.951056516295153572116439333379382143405698634*d2r-0.587785252292473129168705954639072768597652438*d1r;")
        L(f"          {a}_re={a}_re+s1r+s2r; {a}_im={a}_im+s1i+s2i;")
        if fwd:
            L(f"          {b}_re=p1r+Ti; {b}_im=p1i-Tu; {xe}_re=p1r-Ti; {xe}_im=p1i+Tu;")
            L(f"          {c}_re=p2r-Tk; {c}_im=p2i+Tv; {dd}_re=p2r+Tk; {dd}_im=p2i-Tv; }}")
        else:
            L(f"          {b}_re=p1r-Ti; {b}_im=p1i+Tu; {xe}_re=p1r+Ti; {xe}_im=p1i-Tu;")
            L(f"          {c}_re=p2r+Tk; {c}_im=p2i-Tv; {dd}_re=p2r-Tk; {dd}_im=p2i+Tv; }}")
        for k1 in range(N1):
            L(f"        sp_re[{n2*N1+k1}] = x{k1}_re; sp_im[{n2*N1+k1}] = x{k1}_im;")
        L(f"")

    # Pass 2: internal twiddles + 5 radix-5 combines
    for k1 in range(N1):
        L(f"        /* column k1={k1} */")
        for n2 in range(N2):
            L(f"        x{n2}_re = sp_re[{n2*N1+k1}]; x{n2}_im = sp_im[{n2*N1+k1}];")
        if k1 > 0:
            for n2 in range(1, N2):
                e = (n2*k1) % N
                label = wN_label(e, N)
                if fwd:
                    L(f"        {{ double tr=x{n2}_re; x{n2}_re = tr*{label}_re - x{n2}_im*{label}_im; x{n2}_im = tr*{label}_im + x{n2}_im*{label}_re; }}")
                else:
                    L(f"        {{ double tr=x{n2}_re; x{n2}_re = tr*{label}_re + x{n2}_im*{label}_im; x{n2}_im = x{n2}_im*{label}_re - tr*{label}_im; }}")
        # radix-5 butterfly (same as pass 1)
        L(f"        {{ double s1r={b}_re+{xe}_re, s1i={b}_im+{xe}_im;")
        L(f"          double s2r={c}_re+{dd}_re, s2i={c}_im+{dd}_im;")
        L(f"          double d1r={b}_re-{xe}_re, d1i={b}_im-{xe}_im;")
        L(f"          double d2r={c}_re-{dd}_re, d2i={c}_im-{dd}_im;")
        L(f"          double t0r={a}_re-0.25*(s1r+s2r), t0i={a}_im-0.25*(s1i+s2i);")
        L(f"          double t1r=0.559016994374947424102293417182819058860154590*(s1r-s2r);")
        L(f"          double t1i=0.559016994374947424102293417182819058860154590*(s1i-s2i);")
        L(f"          double p1r=t0r+t1r, p1i=t0i+t1i, p2r=t0r-t1r, p2i=t0i-t1i;")
        L(f"          double Ti=0.951056516295153572116439333379382143405698634*d1i+0.587785252292473129168705954639072768597652438*d2i;")
        L(f"          double Tu=0.951056516295153572116439333379382143405698634*d1r+0.587785252292473129168705954639072768597652438*d2r;")
        L(f"          double Tk=0.951056516295153572116439333379382143405698634*d2i-0.587785252292473129168705954639072768597652438*d1i;")
        L(f"          double Tv=0.951056516295153572116439333379382143405698634*d2r-0.587785252292473129168705954639072768597652438*d1r;")
        L(f"          {a}_re={a}_re+s1r+s2r; {a}_im={a}_im+s1i+s2i;")
        if fwd:
            L(f"          {b}_re=p1r+Ti; {b}_im=p1i-Tu; {xe}_re=p1r-Ti; {xe}_im=p1i+Tu;")
            L(f"          {c}_re=p2r-Tk; {c}_im=p2i+Tv; {dd}_re=p2r+Tk; {dd}_im=p2i-Tv; }}")
        else:
            L(f"          {b}_re=p1r-Ti; {b}_im=p1i+Tu; {xe}_re=p1r+Ti; {xe}_im=p1i-Tu;")
            L(f"          {c}_re=p2r+Tk; {c}_im=p2i-Tv; {dd}_re=p2r-Tk; {dd}_im=p2i+Tv; }}")
        for k2 in range(N2):
            m = k1 + N1*k2
            L(f"        out_re[{m}*K+k] = x{k2}_re; out_im[{m}*K+k] = x{k2}_im;")
        L(f"")

    L(f"    }}")
    L(f"}}")
    L(f"")
    return lines

# ── SIMD (AVX-512 / AVX2) — reuse tw generator's emitter ──

def gen_simd_kernel(isa, d):
    """Generate N1 kernel by importing from gen_radix25_tw and stripping external twiddles."""
    # Import the SIMD emitter
    sys.path.insert(0, '.')
    from gen_radix25_tw import SimdEmitter, collect_internal_twiddles, N, N1, N2, wN_label
    itw_set = collect_internal_twiddles()

    em = SimdEmitter(isa)
    T = em.T; W = em.W
    suffix = isa

    em.L.append(f"{em.attr}")
    em.L.append(f"static void")
    em.L.append(f"radix25_n1_dit_kernel_{d}_{suffix}(")
    em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
    em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
    em.L.append(f"    size_t K)")
    em.L.append(f"{{")
    em.ind = 1

    align = 64 if isa == 'avx512' else 32
    em.o(f"__attribute__((aligned({align}))) double sp_re[{25*W}], sp_im[{25*W}];")
    em.o(f"{T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im,x4_re,x4_im;")
    em.b()
    em.o(f"const {T} c0 = {em._set1('0.559016994374947424102293417182819058860154590')};")
    em.o(f"const {T} c1 = {em._set1('0.951056516295153572116439333379382143405698634')};")
    em.o(f"const {T} c2 = {em._set1('0.587785252292473129168705954639072768597652438')};")
    em.o(f"const {T} q25 = {em._set1('0.25')};")
    em.b()

    if itw_set:
        em.c(f"Hoisted W25 broadcasts [{d}]")
        for (e, tN) in sorted(itw_set):
            label = wN_label(e, tN)
            em.o(f"const {T} tw_{label}_re = {em._set1(f'{label}_re')};")
            em.o(f"const {T} tw_{label}_im = {em._set1(f'{label}_im')};")
        em.b()

    xv = [f"x{i}" for i in range(5)]
    em.o(f"for (size_t k = 0; k < K; k += {W}) {{")
    em.ind += 1

    for n2 in range(N2):
        em.c(f"sub-FFT n2={n2}")
        for n1 in range(N1):
            n = N2*n1 + n2
            em.emit_load(f"x{n1}", n)
        em.b()
        em.emit_radix5(xv, d, f"radix-5 n2={n2}")
        em.b()
        for k1 in range(N1):
            em.emit_spill(f"x{k1}", n2*N1 + k1)
        em.b()

    for k1 in range(N1):
        em.c(f"column k1={k1}")
        for n2 in range(N2):
            em.emit_reload(f"x{n2}", n2*N1 + k1)
        em.b()
        if k1 > 0:
            for n2 in range(1, N2):
                e = (n2*k1) % N
                em.emit_int_tw(f"x{n2}", e, d)
            em.b()
        em.emit_radix5(xv, d, f"radix-5 k1={k1}")
        em.b()
        for k2 in range(N2):
            em.emit_store(f"x{k2}", k1 + N1*k2)
        em.b()

    em.ind -= 1
    em.o("}")
    em.L.append("}"); em.L.append("")
    return em.L

# ── File generation ──

def emit_twiddle_constants(lines, itw_set):
    by_tN = {}
    for (e, tN) in sorted(itw_set):
        by_tN.setdefault(tN, []).append(e)
    for tN in sorted(by_tN):
        g = f"FFT_W{tN}_TWIDDLES_DEFINED"
        lines.append(f"#ifndef {g}")
        lines.append(f"#define {g}")
        for e in sorted(by_tN[tN]):
            wr, wi = wN(e, tN)
            l = wN_label(e, tN)
            lines.append(f"static const double {l}_re = {wr:.20e};")
            lines.append(f"static const double {l}_im = {wi:.20e};")
        lines.append(f"#endif")
        lines.append("")

def main():
    if len(sys.argv) < 2:
        print("Usage: gen_radix25_n1.py scalar|avx512|avx2", file=sys.stderr)
        sys.exit(1)
    isa = sys.argv[1]
    itw_set = collect_internal_twiddles()
    lines = []

    if isa == 'scalar':
        guard = "FFT_RADIX25_SCALAR_N1_H"
        lines.append(f"/**")
        lines.append(f" * @file fft_radix25_scalar_n1.h")
        lines.append(f" * @brief Scalar N1 (notw) DFT-25 (5x5 CT)")
        lines.append(f" * Generated by gen_radix25_n1.py")
        lines.append(f" */")
        lines.append(f"#ifndef {guard}")
        lines.append(f"#define {guard}")
        lines.append(f"#include <stddef.h>")
        lines.append(f"")
        emit_twiddle_constants(lines, itw_set)
        lines.extend(gen_scalar_kernel('fwd'))
        lines.extend(gen_scalar_kernel('bwd'))
        lines.append(f"#endif /* {guard} */")

    elif isa in ('avx512', 'avx2'):
        ISA = isa.upper().replace('-','')
        guard = f"FFT_RADIX25_{ISA}_N1_H"
        lines.append(f"/**")
        lines.append(f" * @file fft_radix25_{isa}_n1.h")
        lines.append(f" * @brief {isa.upper()} N1 (notw) DFT-25 (5x5 CT)")
        lines.append(f" * Generated by gen_radix25_n1.py")
        lines.append(f" */")
        lines.append(f"#ifndef {guard}")
        lines.append(f"#define {guard}")
        lines.append(f"#include <immintrin.h>")
        lines.append(f"")
        emit_twiddle_constants(lines, itw_set)
        lines.extend(gen_simd_kernel(isa, 'fwd'))
        lines.extend(gen_simd_kernel(isa, 'bwd'))
        lines.append(f"#endif /* {guard} */")
    else:
        print(f"Unknown: {isa}", file=sys.stderr); sys.exit(1)

    print("\n".join(lines))
    print(f"Lines: {len(lines)}", file=sys.stderr)

if __name__ == '__main__':
    main()
