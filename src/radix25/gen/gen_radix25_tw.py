#!/usr/bin/env python3
"""
gen_radix25_tw.py — DFT-25 twiddled codelet generator

5×5 Cooley-Tukey: 5 radix-5 sub-FFTs + W₂₅ twiddles + 5 radix-5 combines.
External twiddles: 24 per k-step (flat table: tw[(n-1)*K+k], n=1..24).
Internal W₂₅: 9 unique broadcast constants (no trivials for N=25).

DIT: external twiddle on input (before butterfly)
DIF: external twiddle on output (after butterfly)

Usage:
  python3 gen_radix25_tw.py scalar > fft_radix25_scalar_tw.h
  python3 gen_radix25_tw.py avx512 > fft_radix25_avx512_tw.h
  python3 gen_radix25_tw.py avx2   > fft_radix25_avx2_tw.h
"""
import math, sys

N, N1, N2 = 25, 5, 5

def wN(e, tN):
    e = e % tN; a = 2.0*math.pi*e/tN
    return (math.cos(a), -math.sin(a))

def wN_label(e, tN): return f"W{tN}_{e%tN}"

def collect_internal_twiddles():
    """W₂₅^{n₂·k₁} for n₂=1..4, k₁=1..4, unique non-zero exponents."""
    tw = set()
    for n2 in range(1, N2):
        for k1 in range(1, N1):
            e = (n2*k1) % N
            if e != 0:
                tw.add((e, N))
    return tw

# ────────────────────────────────────────────────────────────────
# Scalar emitter
# ────────────────────────────────────────────────────────────────

class ScalarEmitter:
    K0 = "0.559016994374947424102293417182819058860154590"  # √5/4
    K1 = "0.951056516295153572116439333379382143405698634"  # sin(2π/5)
    K2 = "0.587785252292473129168705954639072768597652438"  # sin(4π/5)

    def __init__(self):
        self.L = []; self.ind = 1

    def o(self, t=""): self.L.append("    "*self.ind + t)
    def c(self, t): self.o(f"/* {t} */")
    def b(self): self.L.append("")

    def emit_load(self, v, n):
        self.o(f"{v}_re = in_re[{n}*K+k]; {v}_im = in_im[{n}*K+k];")

    def emit_store(self, v, m):
        self.o(f"out_re[{m}*K+k] = {v}_re; out_im[{m}*K+k] = {v}_im;")

    def emit_spill(self, v, slot):
        self.o(f"sp_re[{slot}] = {v}_re; sp_im[{slot}] = {v}_im;")

    def emit_reload(self, v, slot):
        self.o(f"{v}_re = sp_re[{slot}]; {v}_im = sp_im[{slot}];")

    def emit_ext_tw(self, v, tw_idx, d):
        fwd = (d == 'fwd')
        self.o(f"{{ double wr=tw_re[{tw_idx}*K+k], wi=tw_im[{tw_idx}*K+k], tr={v}_re;")
        if fwd:
            self.o(f"  {v}_re = tr*wr - {v}_im*wi; {v}_im = tr*wi + {v}_im*wr; }}")
        else:
            self.o(f"  {v}_re = tr*wr + {v}_im*wi; {v}_im = {v}_im*wr - tr*wi; }}")

    def emit_int_tw(self, v, e, d):
        """Apply broadcast W₂₅^e constant."""
        label = wN_label(e, N)
        fwd = (d == 'fwd')
        self.o(f"{{ double tr={v}_re;")
        if fwd:
            self.o(f"  {v}_re = tr*{label}_re - {v}_im*{label}_im;")
            self.o(f"  {v}_im = tr*{label}_im + {v}_im*{label}_re; }}")
        else:
            self.o(f"  {v}_re = tr*{label}_re + {v}_im*{label}_im;")
            self.o(f"  {v}_im = {v}_im*{label}_re - tr*{label}_im; }}")

    def emit_radix5(self, v, d, label=""):
        """In-place radix-5 butterfly on v[0..4]."""
        fwd = (d == 'fwd')
        if label: self.c(f"{label} [{d}]")
        a,b,c,dd,e = v[0],v[1],v[2],v[3],v[4]
        K0,K1,K2 = self.K0, self.K1, self.K2
        self.o(f"{{ double s1r={b}_re+{e}_re, s1i={b}_im+{e}_im;")
        self.o(f"  double s2r={c}_re+{dd}_re, s2i={c}_im+{dd}_im;")
        self.o(f"  double d1r={b}_re-{e}_re, d1i={b}_im-{e}_im;")
        self.o(f"  double d2r={c}_re-{dd}_re, d2i={c}_im-{dd}_im;")
        self.o(f"  double t0r={a}_re-0.25*(s1r+s2r), t0i={a}_im-0.25*(s1i+s2i);")
        self.o(f"  double t1r={K0}*(s1r-s2r), t1i={K0}*(s1i-s2i);")
        self.o(f"  double p1r=t0r+t1r, p1i=t0i+t1i;")
        self.o(f"  double p2r=t0r-t1r, p2i=t0i-t1i;")
        self.o(f"  double Ti={K1}*d1i+{K2}*d2i, Tu={K1}*d1r+{K2}*d2r;")
        self.o(f"  double Tk={K1}*d2i-{K2}*d1i, Tv={K1}*d2r-{K2}*d1r;")
        self.o(f"  {a}_re={a}_re+s1r+s2r; {a}_im={a}_im+s1i+s2i;")
        if fwd:
            self.o(f"  {b}_re=p1r+Ti; {b}_im=p1i-Tu;")
            self.o(f"  {e}_re=p1r-Ti; {e}_im=p1i+Tu;")
            self.o(f"  {c}_re=p2r-Tk; {c}_im=p2i+Tv;")
            self.o(f"  {dd}_re=p2r+Tk; {dd}_im=p2i-Tv; }}")
        else:
            self.o(f"  {b}_re=p1r-Ti; {b}_im=p1i+Tu;")
            self.o(f"  {e}_re=p1r+Ti; {e}_im=p1i-Tu;")
            self.o(f"  {c}_re=p2r+Tk; {c}_im=p2i-Tv;")
            self.o(f"  {dd}_re=p2r-Tk; {dd}_im=p2i+Tv; }}")


def emit_scalar_kernel(em, d, mode, itw_set):
    dit = (mode == 'dit')
    name = f"radix25_tw_flat_{'dit' if dit else 'dif'}_kernel_{d}_scalar"
    em.L.append(f"static inline void")
    em.L.append(f"{name}(")
    em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
    em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
    em.L.append(f"    const double * __restrict__ tw_re, const double * __restrict__ tw_im,")
    em.L.append(f"    size_t K)")
    em.L.append(f"{{")
    em.ind = 1
    em.o("double sp_re[25], sp_im[25];")
    em.o("double x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im,x4_re,x4_im;")
    em.b()
    xv = [f"x{i}" for i in range(5)]
    em.o("for (size_t k = 0; k < K; k++) {")
    em.ind += 1

    # Pass 1: 5 radix-5 sub-FFTs
    for n2 in range(N2):
        em.c(f"sub-FFT n2={n2}")
        for n1 in range(N1):
            n = N2*n1 + n2
            em.emit_load(f"x{n1}", n)
            if dit and n > 0:
                em.emit_ext_tw(f"x{n1}", n-1, d)
        em.b()
        em.emit_radix5(xv, d, f"radix-5 n2={n2}")
        em.b()
        for k1 in range(N1):
            em.emit_spill(f"x{k1}", n2*N1 + k1)
        em.b()

    # Pass 2: 5 radix-5 combines + internal W₂₅ twiddles
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
        if not dit:
            for k2 in range(N2):
                m = k1 + N1*k2
                if m > 0:
                    em.emit_ext_tw(f"x{k2}", m-1, d)
            em.b()
        for k2 in range(N2):
            em.emit_store(f"x{k2}", k1 + N1*k2)
        em.b()

    em.ind -= 1
    em.o("}")
    em.L.append("}"); em.L.append("")


# ────────────────────────────────────────────────────────────────
# SIMD emitter (AVX-512 / AVX2)
# ────────────────────────────────────────────────────────────────

class SimdEmitter:
    def __init__(self, isa):
        self.isa = isa
        self.L = []; self.ind = 1
        if isa == 'avx512':
            self.W = 8; self.T = '__m512d'; self.P = '_mm512'
            self.attr = '__attribute__((target("avx512f,avx512dq,fma")))'
        else:
            self.W = 4; self.T = '__m256d'; self.P = '_mm256'
            self.attr = '__attribute__((target("avx2,fma")))'

    def o(self, t=""): self.L.append("    "*self.ind + t)
    def c(self, t): self.o(f"/* {t} */")
    def b(self): self.L.append("")
    def _add(self,a,b): return f"{self.P}_add_pd({a},{b})"
    def _sub(self,a,b): return f"{self.P}_sub_pd({a},{b})"
    def _mul(self,a,b): return f"{self.P}_mul_pd({a},{b})"
    def _fma(self,a,b,c): return f"{self.P}_fmadd_pd({a},{b},{c})"
    def _fms(self,a,b,c): return f"{self.P}_fmsub_pd({a},{b},{c})"
    def _fnma(self,a,b,c): return f"{self.P}_fnmadd_pd({a},{b},{c})"
    def _set1(self,v): return f"{self.P}_set1_pd({v})"
    def _load(self,p): return f"{self.P}_load_pd({p})"
    def _store(self,p,v): return f"{self.P}_store_pd({p},{v})"

    def emit_load(self, v, n):
        self.o(f"{v}_re = {self._load(f'&in_re[{n}*K+k]')};")
        self.o(f"{v}_im = {self._load(f'&in_im[{n}*K+k]')};")

    def emit_store(self, v, m):
        self.o(f"{self._store(f'&out_re[{m}*K+k]', f'{v}_re')};")
        self.o(f"{self._store(f'&out_im[{m}*K+k]', f'{v}_im')};")

    def emit_spill(self, v, slot):
        self.o(f"{self._store(f'&sp_re[{slot}*{self.W}]', f'{v}_re')};")
        self.o(f"{self._store(f'&sp_im[{slot}*{self.W}]', f'{v}_im')};")

    def emit_reload(self, v, slot):
        self.o(f"{v}_re = {self._load(f'&sp_re[{slot}*{self.W}]')};")
        self.o(f"{v}_im = {self._load(f'&sp_im[{slot}*{self.W}]')};")

    def emit_ext_tw(self, v, tw_idx, d):
        fwd = (d == 'fwd')
        self.o(f"{{ {self.T} wr={self._load(f'&tw_re[{tw_idx}*K+k]')};")
        self.o(f"  {self.T} wi={self._load(f'&tw_im[{tw_idx}*K+k]')};")
        self.o(f"  {self.T} tr={v}_re;")
        if fwd:
            self.o(f"  {v}_re = {self._fms(f'{v}_re','wr',self._mul(f'{v}_im','wi'))};")
            self.o(f"  {v}_im = {self._fma('tr','wi',self._mul(f'{v}_im','wr'))}; }}")
        else:
            self.o(f"  {v}_re = {self._fma(f'{v}_re','wr',self._mul(f'{v}_im','wi'))};")
            self.o(f"  {v}_im = {self._fms(f'{v}_im','wr',self._mul('tr','wi'))}; }}")

    def emit_int_tw(self, v, e, d):
        label = wN_label(e, N)
        fwd = (d == 'fwd')
        self.o(f"{{ {self.T} tr={v}_re;")
        if fwd:
            self.o(f"  {v}_re = {self._fms(f'{v}_re',f'tw_{label}_re',self._mul(f'{v}_im',f'tw_{label}_im'))};")
            self.o(f"  {v}_im = {self._fma('tr',f'tw_{label}_im',self._mul(f'{v}_im',f'tw_{label}_re'))}; }}")
        else:
            self.o(f"  {v}_re = {self._fma(f'{v}_re',f'tw_{label}_re',self._mul(f'{v}_im',f'tw_{label}_im'))};")
            self.o(f"  {v}_im = {self._fms(f'{v}_im',f'tw_{label}_re',self._mul('tr',f'tw_{label}_im'))}; }}")

    def emit_radix5(self, v, d, label=""):
        fwd = (d == 'fwd')
        T = self.T
        if label: self.c(f"{label} [{d}]")
        a,b,c,dd,e = v[0],v[1],v[2],v[3],v[4]
        self.o(f"{{ {T} s1r={self._add(f'{b}_re',f'{e}_re')}, s1i={self._add(f'{b}_im',f'{e}_im')};")
        self.o(f"  {T} s2r={self._add(f'{c}_re',f'{dd}_re')}, s2i={self._add(f'{c}_im',f'{dd}_im')};")
        self.o(f"  {T} d1r={self._sub(f'{b}_re',f'{e}_re')}, d1i={self._sub(f'{b}_im',f'{e}_im')};")
        self.o(f"  {T} d2r={self._sub(f'{c}_re',f'{dd}_re')}, d2i={self._sub(f'{c}_im',f'{dd}_im')};")
        self.o(f"  {T} t0r={self._fnma('q25',self._add('s1r','s2r'),f'{a}_re')}, t0i={self._fnma('q25',self._add('s1i','s2i'),f'{a}_im')};")
        self.o(f"  {T} t1r={self._mul('c0',self._sub('s1r','s2r'))}, t1i={self._mul('c0',self._sub('s1i','s2i'))};")
        self.o(f"  {T} p1r={self._add('t0r','t1r')}, p1i={self._add('t0i','t1i')};")
        self.o(f"  {T} p2r={self._sub('t0r','t1r')}, p2i={self._sub('t0i','t1i')};")
        self.o(f"  {T} Ti={self._fma('c1','d1i',self._mul('c2','d2i'))}, Tu={self._fma('c1','d1r',self._mul('c2','d2r'))};")
        self.o(f"  {T} Tk={self._fnma('c2','d1i',self._mul('c1','d2i'))}, Tv={self._fnma('c2','d1r',self._mul('c1','d2r'))};")
        self.o(f"  {a}_re={self._add(f'{a}_re',self._add('s1r','s2r'))}; {a}_im={self._add(f'{a}_im',self._add('s1i','s2i'))};")
        if fwd:
            self.o(f"  {b}_re={self._add('p1r','Ti')}; {b}_im={self._sub('p1i','Tu')};")
            self.o(f"  {e}_re={self._sub('p1r','Ti')}; {e}_im={self._add('p1i','Tu')};")
            self.o(f"  {c}_re={self._sub('p2r','Tk')}; {c}_im={self._add('p2i','Tv')};")
            self.o(f"  {dd}_re={self._add('p2r','Tk')}; {dd}_im={self._sub('p2i','Tv')}; }}")
        else:
            self.o(f"  {b}_re={self._sub('p1r','Ti')}; {b}_im={self._add('p1i','Tu')};")
            self.o(f"  {e}_re={self._add('p1r','Ti')}; {e}_im={self._sub('p1i','Tu')};")
            self.o(f"  {c}_re={self._add('p2r','Tk')}; {c}_im={self._sub('p2i','Tv')};")
            self.o(f"  {dd}_re={self._sub('p2r','Tk')}; {dd}_im={self._add('p2i','Tv')}; }}")


def emit_simd_kernel(em, d, mode, itw_set):
    dit = (mode == 'dit')
    T = em.T; W = em.W; P = em.P
    suffix = em.isa
    name = f"radix25_tw_flat_{'dit' if dit else 'dif'}_kernel_{d}_{suffix}"
    em.L.append(f"{em.attr}")
    em.L.append(f"static void")
    em.L.append(f"{name}(")
    em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
    em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
    em.L.append(f"    const double * __restrict__ tw_re, const double * __restrict__ tw_im,")
    em.L.append(f"    size_t K)")
    em.L.append(f"{{")
    em.ind = 1

    align = 64 if em.isa == 'avx512' else 32
    em.o(f"__attribute__((aligned({align}))) double sp_re[{25*W}], sp_im[{25*W}];")
    em.o(f"{T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im,x4_re,x4_im;")
    em.b()

    # Radix-5 constants
    em.o(f"const {T} c0 = {em._set1('0.559016994374947424102293417182819058860154590')};")
    em.o(f"const {T} c1 = {em._set1('0.951056516295153572116439333379382143405698634')};")
    em.o(f"const {T} c2 = {em._set1('0.587785252292473129168705954639072768597652438')};")
    em.o(f"const {T} q25 = {em._set1('0.25')};")
    em.b()

    # Hoisted internal W₂₅ broadcasts
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

    # Pass 1
    for n2 in range(N2):
        em.c(f"sub-FFT n2={n2}")
        for n1 in range(N1):
            n = N2*n1 + n2
            em.emit_load(f"x{n1}", n)
            if dit and n > 0:
                em.emit_ext_tw(f"x{n1}", n-1, d)
        em.b()
        em.emit_radix5(xv, d, f"radix-5 n2={n2}")
        em.b()
        for k1 in range(N1):
            em.emit_spill(f"x{k1}", n2*N1 + k1)
        em.b()

    # Pass 2
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
        if not dit:
            for k2 in range(N2):
                m = k1 + N1*k2
                if m > 0:
                    em.emit_ext_tw(f"x{k2}", m-1, d)
            em.b()
        for k2 in range(N2):
            em.emit_store(f"x{k2}", k1 + N1*k2)
        em.b()

    em.ind -= 1
    em.o("}")
    em.L.append("}"); em.L.append("")


# ────────────────────────────────────────────────────────────────
# File generation
# ────────────────────────────────────────────────────────────────

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


def gen_scalar():
    itw_set = collect_internal_twiddles()
    em = ScalarEmitter()
    em.L.append("/**")
    em.L.append(" * @file fft_radix25_scalar_tw.h")
    em.L.append(" * @brief Scalar DIT + DIF twiddled DFT-25 (5x5 CT)")
    em.L.append(" * Generated by gen_radix25_tw.py")
    em.L.append(" */")
    em.L.append("")
    em.L.append("#ifndef FFT_RADIX25_SCALAR_TW_H")
    em.L.append("#define FFT_RADIX25_SCALAR_TW_H")
    em.L.append("#include <stddef.h>")
    em.L.append("")
    emit_twiddle_constants(em.L, itw_set)

    for mode in ('dit', 'dif'):
        for d in ('fwd', 'bwd'):
            emit_scalar_kernel(em, d, mode, itw_set)

    em.L.append("#endif /* FFT_RADIX25_SCALAR_TW_H */")
    return em.L


def gen_simd(isa):
    itw_set = collect_internal_twiddles()
    em = SimdEmitter(isa)
    ISA = isa.upper()
    em.L.append("/**")
    em.L.append(f" * @file fft_radix25_{isa}_tw.h")
    em.L.append(f" * @brief {ISA} DIT + DIF twiddled DFT-25 (5x5 CT)")
    em.L.append(f" * Generated by gen_radix25_tw.py")
    em.L.append(" */")
    em.L.append("")
    em.L.append(f"#ifndef FFT_RADIX25_{ISA.replace('-','')}_TW_H")
    em.L.append(f"#define FFT_RADIX25_{ISA.replace('-','')}_TW_H")
    em.L.append("#include <immintrin.h>")
    em.L.append("")
    emit_twiddle_constants(em.L, itw_set)

    for mode in ('dit', 'dif'):
        for d in ('fwd', 'bwd'):
            emit_simd_kernel(em, d, mode, itw_set)

    em.L.append(f"#endif /* FFT_RADIX25_{ISA.replace('-','')}_TW_H */")
    return em.L


def main():
    if len(sys.argv) < 2:
        print("Usage: gen_radix25_tw.py scalar|avx512|avx2", file=sys.stderr)
        sys.exit(1)
    isa = sys.argv[1]
    if isa == 'scalar':
        lines = gen_scalar()
    elif isa in ('avx512', 'avx2'):
        lines = gen_simd(isa)
    else:
        print(f"Unknown ISA: {isa}", file=sys.stderr)
        sys.exit(1)
    print("\n".join(lines))
    print(f"Lines: {len(lines)}", file=sys.stderr)

if __name__ == '__main__':
    main()
