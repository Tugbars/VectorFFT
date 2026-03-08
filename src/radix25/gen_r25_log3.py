#!/usr/bin/env python3
"""
gen_r25_log3.py — DFT-25 (5×5 CT) tw codelets with log3 twiddle derivation.

Load W^1 and W^5 only. Derive all 24 twiddles in registers:
  Column bases: W^2=W^1², W^3=W^1·W^2, W^4=W^1·W^3       (3 cmuls)
  Row powers:   W^10=W^5², W^15=W^5·W^10, W^20=W^5·W^15   (3 cmuls)
  Per-group:    W^(5j+n2) = W^n2 · W^(5j)                  (4 cmuls × 4 groups)

Total: 2 loads + 22 cmuls instead of 48 loads.

Internal W25 twiddles (pass 2) are broadcast constants — unchanged.

Usage:
  python3 gen_r25_log3.py <scalar|avx2|avx512> <dit|dif>
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

# ════════════════════════════════════════
# ISA abstraction
# ════════════════════════════════════════

def isa_config(isa):
    if isa == 'scalar':
        return dict(T='double', W=1, attr='',
                    ld=lambda p: f'*({p})', st=lambda p,v: f'*({p}) = {v};')
    elif isa == 'avx2':
        return dict(T='__m256d', W=4,
                    attr='__attribute__((target("avx2,fma")))',
                    ld=lambda p: f'_mm256_load_pd({p})',
                    st=lambda p,v: f'_mm256_store_pd({p},{v});')
    else:
        return dict(T='__m512d', W=8,
                    attr='__attribute__((target("avx512f,avx512dq,fma")))',
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
        add=lambda a,b: f'{P}_add_pd({a},{b})',
        sub=lambda a,b: f'{P}_sub_pd({a},{b})',
        mul=lambda a,b: f'{P}_mul_pd({a},{b})',
        fma=lambda a,b,c: f'{P}_fmadd_pd({a},{b},{c})',
        fms=lambda a,b,c: f'{P}_fmsub_pd({a},{b},{c})',
        fnma=lambda a,b,c: f'{P}_fnmadd_pd({a},{b},{c})',
        set1=lambda v: f'{P}_set1_pd({v})')

def cmul_re(o, ar,ai,br,bi): return o['fms'](ar,br, o['mul'](ai,bi))
def cmul_im(o, ar,ai,br,bi): return o['fma'](ar,bi, o['mul'](ai,br))
def cmulc_re(o, ar,ai,br,bi): return o['fma'](ar,br, o['mul'](ai,bi))
def cmulc_im(o, ar,ai,br,bi): return o['fnma'](ar,bi, o['mul'](ai,br))

# ════════════════════════════════════════
# Code emitter helpers
# ════════════════════════════════════════

class Emitter:
    def __init__(self, isa):
        self.isa = isa
        self.cfg = isa_config(isa)
        self.o_ops = ops(isa)
        self.T = self.cfg['T']
        self.W = self.cfg['W']
        self.L = []
        self.ind = 0

    def line(self, s=''): self.L.append('    ' * self.ind + s)
    def comment(self, s): self.line(f'/* {s} */')
    def blank(self): self.L.append('')

    def load(self, v, idx):
        self.line(f'{v}_re = {self.cfg["ld"](f"&in_re[{idx}*K+k]")};')
        self.line(f'{v}_im = {self.cfg["ld"](f"&in_im[{idx}*K+k]")};')

    def store(self, v, idx):
        self.line(f'{self.cfg["st"](f"&out_re[{idx}*K+k]", f"{v}_re")}')
        self.line(f'{self.cfg["st"](f"&out_im[{idx}*K+k]", f"{v}_im")}')

    def spill(self, v, slot):
        if self.isa == 'scalar':
            self.line(f'sp_re[{slot}] = {v}_re; sp_im[{slot}] = {v}_im;')
        else:
            self.line(f'{self.cfg["st"](f"&sp_re[{slot}*{self.W}]", f"{v}_re")}')
            self.line(f'{self.cfg["st"](f"&sp_im[{slot}*{self.W}]", f"{v}_im")}')

    def reload(self, v, slot):
        if self.isa == 'scalar':
            self.line(f'{v}_re = sp_re[{slot}]; {v}_im = sp_im[{slot}];')
        else:
            self.line(f'{v}_re = {self.cfg["ld"](f"&sp_re[{slot}*{self.W}]")};')
            self.line(f'{v}_im = {self.cfg["ld"](f"&sp_im[{slot}*{self.W}]")};')

    def cmul(self, dst, a, b):
        """dst = a × b (complex multiply)"""
        o = self.o_ops
        self.line(f'const {self.T} {dst}_re = {cmul_re(o, f"{a}_re",f"{a}_im",f"{b}_re",f"{b}_im")};')
        self.line(f'const {self.T} {dst}_im = {cmul_im(o, f"{a}_re",f"{a}_im",f"{b}_re",f"{b}_im")};')

    def cmul_apply(self, v, tw, fwd):
        """v = v × tw (fwd) or v × conj(tw) (bwd), in-place via temp"""
        o = self.o_ops
        cm = (cmul_re, cmul_im) if fwd else (cmulc_re, cmulc_im)
        self.line(f'{{ {self.T} tr = {v}_re;')
        self.line(f'  {v}_re = {cm[0](o, f"{v}_re", f"{v}_im", f"{tw}_re", f"{tw}_im")};')
        self.line(f'  {v}_im = {cm[1](o, "tr", f"{v}_im", f"{tw}_re", f"{tw}_im")};')
        self.line(f'}}')

    def cmul_into(self, dst, src, tw, fwd):
        """dst = src × tw (fwd) or src × conj(tw) (bwd)"""
        o = self.o_ops
        cm = (cmul_re, cmul_im) if fwd else (cmulc_re, cmulc_im)
        self.line(f'const {self.T} {dst}_re = {cm[0](o, f"{src}_re",f"{src}_im",f"{tw}_re",f"{tw}_im")};')
        self.line(f'const {self.T} {dst}_im = {cm[1](o, f"{src}_re",f"{src}_im",f"{tw}_re",f"{tw}_im")};')

    def int_tw_apply(self, v, exp, fwd):
        """Apply internal W25 constant (broadcast)."""
        label = wN_label(exp, N)
        o = self.o_ops
        cm = (cmul_re, cmul_im) if fwd else (cmulc_re, cmulc_im)
        self.line(f'{{ {self.T} tr = {v}_re;')
        self.line(f'  {v}_re = {cm[0](o, f"{v}_re", f"{v}_im", f"tw_{label}_re", f"tw_{label}_im")};')
        self.line(f'  {v}_im = {cm[1](o, "tr", f"{v}_im", f"tw_{label}_re", f"tw_{label}_im")};')
        self.line(f'}}')

    def radix5(self, v, fwd, label=''):
        """In-place radix-5 butterfly on v[0..4]."""
        o = self.o_ops
        add, sub, mul, fma, fnma = o['add'], o['sub'], o['mul'], o['fma'], o['fnma']
        if label: self.comment(f'{label} [{"fwd" if fwd else "bwd"}]')
        a,b,c,d,e = [f'{v[i]}' for i in range(5)]
        self.line(f'{{ {self.T} s1r={add(f"{b}_re",f"{e}_re")}, s1i={add(f"{b}_im",f"{e}_im")};')
        self.line(f'  {self.T} s2r={add(f"{c}_re",f"{d}_re")}, s2i={add(f"{c}_im",f"{d}_im")};')
        self.line(f'  {self.T} d1r={sub(f"{b}_re",f"{e}_re")}, d1i={sub(f"{b}_im",f"{e}_im")};')
        self.line(f'  {self.T} d2r={sub(f"{c}_re",f"{d}_re")}, d2i={sub(f"{c}_im",f"{d}_im")};')
        self.line(f'  {self.T} t0r={fnma("q25",add("s1r","s2r"),f"{a}_re")}, t0i={fnma("q25",add("s1i","s2i"),f"{a}_im")};')
        self.line(f'  {self.T} t1r={mul("c0",sub("s1r","s2r"))}, t1i={mul("c0",sub("s1i","s2i"))};')
        self.line(f'  {self.T} p1r={add("t0r","t1r")}, p1i={add("t0i","t1i")};')
        self.line(f'  {self.T} p2r={sub("t0r","t1r")}, p2i={sub("t0i","t1i")};')
        self.line(f'  {self.T} Ti={fma("c1","d1i",mul("c2","d2i"))}, Tu={fma("c1","d1r",mul("c2","d2r"))};')
        self.line(f'  {self.T} Tk={fnma("c2","d1i",mul("c1","d2i"))}, Tv={fnma("c2","d1r",mul("c1","d2r"))};')
        self.line(f'  {a}_re={add(f"{a}_re",add("s1r","s2r"))}; {a}_im={add(f"{a}_im",add("s1i","s2i"))};')
        if fwd:
            self.line(f'  {b}_re={add("p1r","Ti")}; {b}_im={sub("p1i","Tu")};')
            self.line(f'  {e}_re={sub("p1r","Ti")}; {e}_im={add("p1i","Tu")};')
            self.line(f'  {c}_re={sub("p2r","Tk")}; {c}_im={add("p2i","Tv")};')
            self.line(f'  {d}_re={add("p2r","Tk")}; {d}_im={sub("p2i","Tv")}; }}')
        else:
            self.line(f'  {b}_re={sub("p1r","Ti")}; {b}_im={add("p1i","Tu")};')
            self.line(f'  {e}_re={add("p1r","Ti")}; {e}_im={sub("p1i","Tu")};')
            self.line(f'  {c}_re={add("p2r","Tk")}; {c}_im={sub("p2i","Tv")};')
            self.line(f'  {d}_re={sub("p2r","Tk")}; {d}_im={add("p2i","Tv")}; }}')


def emit_twiddle_constants(lines, itw_set):
    lines.append('#ifndef FFT_W25_TWIDDLES_DEFINED')
    lines.append('#define FFT_W25_TWIDDLES_DEFINED')
    for (e, tN) in sorted(itw_set):
        wr, wi = wN(e, tN)
        l = wN_label(e, tN)
        lines.append(f'static const double {l}_re = {wr:.20e};')
        lines.append(f'static const double {l}_im = {wi:.20e};')
    lines.append('#endif')
    lines.append('')


def emit_log3_derive(em):
    """Emit W^1,W^5 loads + full derivation chain."""
    em.comment('Load 2 bases: W^1, W^5')
    em.line(f'const {em.T} ew1_re = {em.cfg["ld"]("&tw_re[0*K+k]")}, ew1_im = {em.cfg["ld"]("&tw_im[0*K+k]")};')
    em.line(f'const {em.T} ew5_re = {em.cfg["ld"]("&tw_re[4*K+k]")}, ew5_im = {em.cfg["ld"]("&tw_im[4*K+k]")};')
    em.comment('Column bases: W^2, W^3, W^4')
    em.cmul('ew2', 'ew1', 'ew1')
    em.cmul('ew3', 'ew1', 'ew2')
    em.cmul('ew4', 'ew1', 'ew3')
    em.comment('Row powers: W^10, W^15, W^20')
    em.cmul('ew10', 'ew5', 'ew5')
    em.cmul('ew15', 'ew5', 'ew10')
    em.cmul('ew20', 'ew5', 'ew15')
    em.blank()


def get_tw_var(n):
    """Return the variable name for external twiddle W^n.
    W^0 = 1 (no twiddle), W^1..W^4 = ew1..ew4, W^5 = ew5,
    W^10 = ew10, W^15 = ew15, W^20 = ew20,
    others = ew{n} (derived per-group)."""
    if n == 0: return None
    if n in (1,2,3,4,5,10,15,20): return f'ew{n}'
    return f'ew{n}'


def emit_dit_kernel(em, fwd, itw_set):
    """DIT: twiddle inputs → sub-FFTs → internal tw → column FFTs → store."""
    T = em.T
    xv = [f'x{i}' for i in range(5)]
    row_vars = {0: None, 5: 'ew5', 10: 'ew10', 15: 'ew15', 20: 'ew20'}

    # Pass 1: 5 sub-FFTs by n2
    for n2 in range(N2):
        em.comment(f'sub-FFT n2={n2}')
        # Derive per-group twiddles for this n2
        for n1 in range(N1):
            n = N2*n1 + n2
            em.load(f'x{n1}', n)
            if n > 0:
                # Need W^n = W^n2 * W^(5*n1)
                if n2 == 0:
                    # W^(5*n1) already derived as ew5/ew10/ew15/ew20
                    tw = row_vars[5*n1]
                    em.cmul_apply(f'x{n1}', tw, fwd)
                elif n1 == 0:
                    # W^n2 already derived as ew1..ew4
                    em.cmul_apply(f'x{n1}', f'ew{n2}', fwd)
                else:
                    # Derive W^n = W^n2 * W^(5*n1) inline
                    em.cmul_into(f'ew{n}', f'ew{n2}', row_vars[5*n1], True)
                    em.cmul_apply(f'x{n1}', f'ew{n}', fwd)
        em.blank()
        em.radix5(xv, fwd, f'radix-5 n2={n2}')
        em.blank()
        for k1 in range(N1):
            em.spill(f'x{k1}', n2*N1 + k1)
        em.blank()

    # Pass 2: column FFTs with internal W25 twiddles
    for k1 in range(N1):
        em.comment(f'column k1={k1}')
        for n2 in range(N2):
            em.reload(f'x{n2}', n2*N1 + k1)
        em.blank()
        if k1 > 0:
            for n2 in range(1, N2):
                e = (n2*k1) % N
                em.int_tw_apply(f'x{n2}', e, fwd)
            em.blank()
        em.radix5(xv, fwd, f'radix-5 k1={k1}')
        em.blank()
        for k2 in range(N2):
            em.store(f'x{k2}', k1 + N1*k2)
        em.blank()


def emit_dif_kernel(em, fwd, itw_set):
    """DIF: load → sub-FFTs → internal tw → column FFTs → twiddle outputs → store."""
    T = em.T
    xv = [f'x{i}' for i in range(5)]
    row_vars = {0: None, 5: 'ew5', 10: 'ew10', 15: 'ew15', 20: 'ew20'}

    # Pass 1: 5 sub-FFTs (no external twiddle)
    for n2 in range(N2):
        em.comment(f'sub-FFT n2={n2}')
        for n1 in range(N1):
            n = N2*n1 + n2
            em.load(f'x{n1}', n)
        em.blank()
        em.radix5(xv, fwd, f'radix-5 n2={n2}')
        em.blank()
        for k1 in range(N1):
            em.spill(f'x{k1}', n2*N1 + k1)
        em.blank()

    # Pass 2: column FFTs + internal W25 + external twiddle on output
    for k1 in range(N1):
        em.comment(f'column k1={k1}')
        for n2 in range(N2):
            em.reload(f'x{n2}', n2*N1 + k1)
        em.blank()
        if k1 > 0:
            for n2 in range(1, N2):
                e = (n2*k1) % N
                em.int_tw_apply(f'x{n2}', e, fwd)
            em.blank()
        em.radix5(xv, fwd, f'radix-5 k1={k1}')
        em.blank()
        # External twiddle on outputs
        for k2 in range(N2):
            m = k1 + N1*k2
            if m > 0:
                if k1 == 0:
                    tw = row_vars[5*k2]
                    em.cmul_apply(f'x{k2}', tw, fwd)
                elif k2 == 0:
                    em.cmul_apply(f'x{k2}', f'ew{k1}', fwd)
                else:
                    em.cmul_into(f'ew{m}', f'ew{k1}', row_vars[5*k2], True)
                    em.cmul_apply(f'x{k2}', f'ew{m}', fwd)
        em.blank()
        for k2 in range(N2):
            em.store(f'x{k2}', k1 + N1*k2)
        em.blank()


def gen_kernel(isa, direction, mode, itw_set):
    em = Emitter(isa)
    fwd = direction == 'fwd'
    dit = mode == 'dit'
    name = f'radix25_tw_flat_{"dit" if dit else "dif"}_kernel_{direction}_{isa}'

    em.L = []
    if em.cfg['attr']: em.L.append(em.cfg['attr'])
    em.L.append(f'static void')
    em.L.append(f'{name}(')
    em.L.append(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
    em.L.append(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    em.L.append(f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,')
    em.L.append(f'    size_t K)')
    em.L.append(f'{{')
    em.ind = 1

    align = 64 if isa == 'avx512' else (32 if isa == 'avx2' else 8)
    if isa != 'scalar':
        em.line(f'__attribute__((aligned({align}))) double sp_re[{25*em.W}], sp_im[{25*em.W}];')
    else:
        em.line(f'double sp_re[25], sp_im[25];')
    em.line(f'{em.T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im,x4_re,x4_im;')
    em.blank()

    # Radix-5 constants
    o = em.o_ops
    em.line(f'const {em.T} c0 = {o["set1"]("0.559016994374947424102293417182819058860154590")};')
    em.line(f'const {em.T} c1 = {o["set1"]("0.951056516295153572116439333379382143405698634")};')
    em.line(f'const {em.T} c2 = {o["set1"]("0.587785252292473129168705954639072768597652438")};')
    em.line(f'const {em.T} q25 = {o["set1"]("0.25")};')
    em.blank()

    # Hoisted internal W25 broadcasts
    if itw_set:
        em.comment(f'Internal W25 broadcasts')
        for (e, tN) in sorted(itw_set):
            label = wN_label(e, tN)
            em.line(f'const {em.T} tw_{label}_re = {o["set1"](f"{label}_re")};')
            em.line(f'const {em.T} tw_{label}_im = {o["set1"](f"{label}_im")};')
        em.blank()

    em.line(f'for (size_t k = 0; k < K; k += {em.W}) {{')
    em.ind += 1

    emit_log3_derive(em)

    if dit:
        emit_dit_kernel(em, fwd, itw_set)
    else:
        emit_dif_kernel(em, fwd, itw_set)

    em.ind -= 1
    em.line('}')
    em.L.append('}')
    em.L.append('')
    return em.L


def gen_file(isa, mode):
    itw_set = collect_internal_twiddles()
    ISA = isa.upper()
    guard = f'FFT_RADIX25_{ISA}_TW_H' if mode == 'dit' else f'FFT_RADIX25_{ISA}_DIF_TW_H'

    L = []
    L.append(f'/**')
    L.append(f' * @file fft_radix25_{isa}_tw.h' if mode == 'dit' else f' * @file fft_radix25_{isa}_dif_tw.h')
    L.append(f' * @brief {ISA} {"DIT" if mode == "dit" else "DIF"} twiddled DFT-25 (5×5 CT) with log3')
    L.append(f' * Load W^1,W^5 only — derive all 24 twiddles in registers.')
    L.append(f' * Generated by gen_r25_log3.py')
    L.append(f' */')
    L.append(f'')
    L.append(f'#ifndef {guard}')
    L.append(f'#define {guard}')
    L.append(f'#include <stddef.h>')
    if isa != 'scalar':
        L.append(f'#include <immintrin.h>')
    L.append(f'')

    emit_twiddle_constants(L, itw_set)

    for d in ('fwd', 'bwd'):
        L.extend(gen_kernel(isa, d, mode, itw_set))

    L.append(f'#endif /* {guard} */')
    return L


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: gen_r25_log3.py <scalar|avx2|avx512> <dit|dif>", file=sys.stderr)
        sys.exit(1)
    isa, mode = sys.argv[1], sys.argv[2]
    lines = gen_file(isa, mode)
    print('\n'.join(lines))
