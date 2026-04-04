#!/usr/bin/env python3
#
# Copyright (c) 2025 Tuğbars Heptaşkın
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
"""
gen_radix10.py — Unified DFT-10 codelet generator for VectorFFT

5×2 Cooley-Tukey: pass 1 = 2× DFT-5 sub-FFTs, pass 2 = 5× DFT-2 columns.
Internal W10 twiddles: 4 unique (W10^1..W10^4).  W10^5 = −1 (free negation).
10 spill + 10 reload per kernel.  DFT-2 = add/sub (zero constants).

DFT-5 butterfly (register-pressure-optimized, same as gen_radix5.py):
  4 constants: cA=sqrt(5)/4, cB=sin(2π/5), cC=sin(4π/5), cD=0.25
  AVX2: peak 16 YMM, zero spills   |   AVX-512: max-ILP, peak 22/32 ZMM

Register pressure per pass:
  Pass 1: DFT-5 on 5 inputs → peak 16 YMM (same as standalone R=5)
  Pass 2: reload 2, twiddle, DFT-2 → 4 regs + twiddle temp (trivial)
  Spill: 10 complex values = 80 bytes (fits L1)

Log3 twiddle derivation (R=10 external):
  Load W^1, derive W^2..W^9 via chain: w(n+1)=w1*w(n)  (8 cmuls)

Variants: notw, dit_tw, dif_tw, dit_tw_log3, dif_tw_log3,
          ct_n1, ct_t1_dit, ct_t1_dif.
Plus: SV (single-vector), U=2 (AVX-512), n1_ovs (SIMD transpose).

Usage:
  python3 gen_radix10.py --isa avx2 --variant all
  python3 gen_radix10.py --isa all  --variant ct_n1
  python3 gen_radix10.py --isa avx2 --variant ct_t1_dit
"""

import sys, math, argparse, re

R = 10
N, N1, N2 = 10, 5, 2   # 5×2 CT decomposition

# DFT-5 constants
cA_val = 0.559016994374947424102293417182819058860154590
cB_val = 0.951056516295153572116439333379382143405698634
cC_val = 0.587785252292473129168705954639072768597652438
cD_val = 0.250000000000000000000000000000000000000000000


# ═══════════════════════════════════════════════════════════════
# TWIDDLE ANALYSIS
# ═══════════════════════════════════════════════════════════════

def wN(e, tN):
    e = e % tN; a = 2.0 * math.pi * e / tN
    return (math.cos(a), -math.sin(a))

def wN_label(e, tN):
    return f"W{tN}_{e % tN}"

def collect_internal_twiddles():
    """Internal W10: n2=1, k1=1..4 → exponents {1,2,3,4}. All require cmul."""
    tw = set()
    for n2 in range(1, N2):
        for k1 in range(1, N1):
            e = (n2 * k1) % N
            if e != 0 and e != 5:   # 5 → W10^5=−1 (negation)
                tw.add((e, N))
    return tw


# ═══════════════════════════════════════════════════════════════
# ISA CONFIGURATION
# ═══════════════════════════════════════════════════════════════

class ISAConfig:
    def __init__(self, name, T, width, k_step, p, sm, target, align, ld_prefix):
        self.name = name; self.T = T; self.width = width
        self.k_step = k_step; self.p = p; self.sm = sm
        self.target = target; self.align = align; self.ld_prefix = ld_prefix

ISA_SCALAR = ISAConfig('scalar', 'double', 1, 1, '', 1, '', '', 'R10S')
ISA_AVX2   = ISAConfig('avx2', '__m256d', 4, 4, '_mm256', 4,
    '__attribute__((target("avx2,fma")))',
    '__attribute__((aligned(32)))', 'R10A')
ISA_AVX512 = ISAConfig('avx512', '__m512d', 8, 8, '_mm512', 8,
    '__attribute__((target("avx512f,avx512dq,fma")))',
    '__attribute__((aligned(64)))', 'R10L')

ALL_ISA = {'scalar': ISA_SCALAR, 'avx2': ISA_AVX2, 'avx512': ISA_AVX512}


# ═══════════════════════════════════════════════════════════════
# EMITTER
# ═══════════════════════════════════════════════════════════════

class Emitter:
    def __init__(self, isa):
        self.isa = isa; self.L = []; self.ind = 1
        self.spill_c = self.reload_c = 0
        self.n_add = self.n_sub = self.n_mul = self.n_neg = 0
        self.n_fma = self.n_fms = 0
        self.n_load = self.n_store = 0
        self.addr_mode = 'K'

    def reset(self):
        self.spill_c = self.reload_c = 0
        self.n_add = self.n_sub = self.n_mul = self.n_neg = 0
        self.n_fma = self.n_fms = 0
        self.n_load = self.n_store = 0

    def get_stats(self):
        ta = self.n_add + self.n_sub + self.n_mul + self.n_neg + self.n_fma + self.n_fms
        fl = self.n_add + self.n_sub + self.n_neg + self.n_mul + 2*(self.n_fma + self.n_fms)
        return {'add': self.n_add, 'sub': self.n_sub, 'mul': self.n_mul, 'neg': self.n_neg,
                'fma': self.n_fma, 'fms': self.n_fms, 'total_arith': ta, 'flops': fl,
                'load': self.n_load, 'store': self.n_store,
                'spill': self.spill_c, 'reload': self.reload_c,
                'total_mem': self.n_load + self.n_store + self.spill_c + self.reload_c}

    def o(self, t=""): self.L.append("    " * self.ind + t)
    def c(self, t): self.o(f"/* {t} */")
    def b(self): self.L.append("")

    # ── Arithmetic (counted) ──
    def add(self, a, b):
        self.n_add += 1
        if self.isa.name == 'scalar': return f"({a})+({b})"
        return f"{self.isa.p}_add_pd({a},{b})"

    def sub(self, a, b):
        self.n_sub += 1
        if self.isa.name == 'scalar': return f"({a})-({b})"
        return f"{self.isa.p}_sub_pd({a},{b})"

    def mul(self, a, b):
        self.n_mul += 1
        if self.isa.name == 'scalar': return f"({a})*({b})"
        return f"{self.isa.p}_mul_pd({a},{b})"

    def neg(self, a):
        self.n_neg += 1
        if self.isa.name == 'scalar': return f"-({a})"
        return f"{self.isa.p}_xor_pd({a},sign_flip)"

    def fma(self, a, b, c):
        self.n_fma += 1
        if self.isa.name == 'scalar': return f"({a})*({b})+({c})"
        return f"{self.isa.p}_fmadd_pd({a},{b},{c})"

    def fms(self, a, b, c):
        self.n_fms += 1
        if self.isa.name == 'scalar': return f"({a})*({b})-({c})"
        return f"{self.isa.p}_fmsub_pd({a},{b},{c})"

    def fnma(self, a, b, c):
        """c - a*b"""
        self.n_fma += 1
        if self.isa.name == 'scalar': return f"({c})-({a})*({b})"
        return f"{self.isa.p}_fnmadd_pd({a},{b},{c})"

    # ── Addressing ──
    def _in_addr(self, n, ke="k"):
        if self.addr_mode in ('n1', 'n1_ovs'): return f"{n}*is+{ke}"
        elif self.addr_mode == 't1':
            if self.isa.name == 'scalar': return f"m*ms+{n}*ios"
            return f"m+{n}*ios"
        return f"{n}*K+{ke}"

    def _out_addr(self, m, ke="k"):
        if self.addr_mode == 'n1': return f"{m}*os+{ke}"
        elif self.addr_mode == 'n1_ovs': return f"{m}*{self.isa.k_step}"
        elif self.addr_mode == 't1':
            if self.isa.name == 'scalar': return f"m*ms+{m}*ios"
            return f"m+{m}*ios"
        return f"{m}*K+{ke}"

    def _in_buf(self):
        return "rio_re" if self.addr_mode == 't1' else "in_re"
    def _in_buf_im(self):
        return "rio_im" if self.addr_mode == 't1' else "in_im"
    def _out_buf(self):
        if self.addr_mode == 't1': return "rio_re"
        if self.addr_mode == 'n1_ovs': return "tbuf_re"
        return "out_re"
    def _out_buf_im(self):
        if self.addr_mode == 't1': return "rio_im"
        if self.addr_mode == 'n1_ovs': return "tbuf_im"
        return "out_im"

    def emit_load(self, v, n, ke="k"):
        self.n_load += 2
        ib, ibi = self._in_buf(), self._in_buf_im()
        addr = self._in_addr(n, ke)
        if self.isa.name == 'scalar':
            self.o(f"{v}_re = {ib}[{addr}]; {v}_im = {ibi}[{addr}];")
        else:
            self.o(f"{v}_re = LD(&{ib}[{addr}]);")
            self.o(f"{v}_im = LD(&{ibi}[{addr}]);")

    def emit_store(self, v, m, ke="k"):
        self.n_store += 2
        ob, obi = self._out_buf(), self._out_buf_im()
        addr = self._out_addr(m, ke)
        if self.isa.name == 'scalar':
            self.o(f"{ob}[{addr}] = {v}_re; {obi}[{addr}] = {v}_im;")
        else:
            self.o(f"ST(&{ob}[{addr}], {v}_re);")
            self.o(f"ST(&{obi}[{addr}], {v}_im);")

    def emit_spill(self, v, slot):
        self.spill_c += 1
        sm = self.isa.sm
        if self.isa.name == 'scalar':
            self.o(f"spill_re[{slot}] = {v}_re; spill_im[{slot}] = {v}_im;")
        else:
            self.o(f"{self.isa.p}_store_pd(&spill_re[{slot}*{sm}], {v}_re);")
            self.o(f"{self.isa.p}_store_pd(&spill_im[{slot}*{sm}], {v}_im);")

    def emit_reload(self, v, slot):
        self.reload_c += 1
        sm = self.isa.sm
        if self.isa.name == 'scalar':
            self.o(f"{v}_re = spill_re[{slot}]; {v}_im = spill_im[{slot}];")
        else:
            self.o(f"{v}_re = {self.isa.p}_load_pd(&spill_re[{slot}*{sm}]);")
            self.o(f"{v}_im = {self.isa.p}_load_pd(&spill_im[{slot}*{sm}]);")

    # ── External twiddle (flat) ──
    def _tw_addr(self, tw_idx, ke="k"):
        if self.addr_mode == 't1': return f"{tw_idx}*me+m"
        return f"{tw_idx}*K+{ke}"
    def _tw_buf(self):
        return "W_re" if self.addr_mode == 't1' else "tw_re"
    def _tw_buf_im(self):
        return "W_im" if self.addr_mode == 't1' else "tw_im"

    def emit_ext_tw(self, v, tw_idx, d, ke="k"):
        fwd = (d == 'fwd'); T = self.isa.T
        tb, tbi = self._tw_buf(), self._tw_buf_im()
        ta = self._tw_addr(tw_idx, ke)
        self.n_load += 2
        if self.isa.name == 'scalar':
            self.o(f"{{ double wr = {tb}[{ta}], wi = {tbi}[{ta}], tr = {v}_re;")
            if fwd:
                self.o(f"  {v}_re = {v}_re*wr - {v}_im*wi;")
                self.o(f"  {v}_im = tr*wi + {v}_im*wr; }}")
            else:
                self.o(f"  {v}_re = {v}_re*wr + {v}_im*wi;")
                self.o(f"  {v}_im = {v}_im*wr - tr*wi; }}")
        else:
            self.o(f"{{ const {T} wr = LD(&{tb}[{ta}]);")
            self.o(f"  const {T} wi = LD(&{tbi}[{ta}]);")
            self.o(f"  const {T} tr = {v}_re;")
            if fwd:
                self.o(f"  {v}_re = {self.fms(f'{v}_re','wr',self.mul(f'{v}_im','wi'))};")
                self.o(f"  {v}_im = {self.fma('tr','wi',self.mul(f'{v}_im','wr'))}; }}")
            else:
                self.o(f"  {v}_re = {self.fma(f'{v}_re','wr',self.mul(f'{v}_im','wi'))};")
                self.o(f"  {v}_im = {self.fms(f'{v}_im','wr',self.mul('tr','wi'))}; }}")

    # ── Complex multiply helpers ──
    def emit_cmul(self, dst_r, dst_i, ar, ai, br, bi, d):
        fwd = (d == 'fwd')
        if fwd:
            self.o(f"{dst_r} = {self.fms(ar, br, self.mul(ai, bi))};")
            self.o(f"{dst_i} = {self.fma(ar, bi, self.mul(ai, br))};")
        else:
            self.o(f"{dst_r} = {self.fma(ar, br, self.mul(ai, bi))};")
            self.o(f"{dst_i} = {self.fms(ai, br, self.mul(ar, bi))};")

    def emit_cmul_inplace(self, v, wr, wi, d):
        fwd = (d == 'fwd'); T = self.isa.T
        self.o(f"{{ {T} tr = {v}_re;")
        if fwd:
            self.o(f"  {v}_re = {self.fms(f'{v}_re', wr, self.mul(f'{v}_im', wi))};")
            self.o(f"  {v}_im = {self.fma('tr', wi, self.mul(f'{v}_im', wr))}; }}")
        else:
            self.o(f"  {v}_re = {self.fma(f'{v}_re', wr, self.mul(f'{v}_im', wi))};")
            self.o(f"  {v}_im = {self.fms(f'{v}_im', wr, self.mul('tr', wi))}; }}")

    def emit_cmul_inplace_named(self, v, wr, wi, d):
        """Same as emit_cmul_inplace but wr/wi are named strings (not f-string refs)."""
        self.emit_cmul_inplace(v, wr, wi, d)

    # ── Internal W10 twiddle ──
    def emit_int_tw(self, v, e, d):
        """Apply internal W10^e twiddle. e=0: skip. e=5: negate. else: cmul."""
        e = e % N
        if e == 0: return
        if e == 5:  # W10^5 = −1
            self.o(f"{v}_re = {self.neg(f'{v}_re')}; {v}_im = {self.neg(f'{v}_im')};")
            return
        label = wN_label(e, N)
        self.emit_cmul_inplace_named(v, f"tw_{label}_re", f"tw_{label}_im", d)

    # ── DFT-5 butterfly ──
    def emit_radix5(self, v, d, label=""):
        """DFT-5 on v[0..4] in-place.
        AVX2/scalar: register-pressure-optimized (peak 16 YMM).
        AVX-512: max-ILP (peak 22/32 ZMM)."""
        fwd = (d == 'fwd'); T = self.isa.T
        if label: self.c(f"{label} [{d}]")
        x0, x1, x2, x3, x4 = v[0], v[1], v[2], v[3], v[4]
        self.o(f"{{")

        if self.isa.name == 'avx512':
            self.o(f"{T} s1r={self.add(f'{x1}_re',f'{x4}_re')}, s1i={self.add(f'{x1}_im',f'{x4}_im')};")
            self.o(f"{T} s2r={self.add(f'{x2}_re',f'{x3}_re')}, s2i={self.add(f'{x2}_im',f'{x3}_im')};")
            self.o(f"{T} d1r={self.sub(f'{x1}_re',f'{x4}_re')}, d1i={self.sub(f'{x1}_im',f'{x4}_im')};")
            self.o(f"{T} d2r={self.sub(f'{x2}_re',f'{x3}_re')}, d2i={self.sub(f'{x2}_im',f'{x3}_im')};")
            self.o(f"{T} ss_r={self.add('s1r','s2r')}, ss_i={self.add('s1i','s2i')};")
            self.o(f"{T} t0r={self.fnma('cD','ss_r',f'{x0}_re')}, t0i={self.fnma('cD','ss_i',f'{x0}_im')};")
            self.o(f"{x0}_re={self.add(f'{x0}_re','ss_r')}; {x0}_im={self.add(f'{x0}_im','ss_i')};")
            self.o(f"{T} p1r={self.fma('cA',self.sub('s1r','s2r'),'t0r')}, "
                   f"p1i={self.fma('cA',self.sub('s1i','s2i'),'t0i')};")
            self.o(f"{T} p2r={self.fnma('cA',self.sub('s1r','s2r'),'t0r')}, "
                   f"p2i={self.fnma('cA',self.sub('s1i','s2i'),'t0i')};")
        else:
            # AVX2/scalar: interleave s/d pairs, precompute sd
            self.o(f"{T} s1r={self.add(f'{x1}_re',f'{x4}_re')}, s1i={self.add(f'{x1}_im',f'{x4}_im')};")
            self.o(f"{T} d1r={self.sub(f'{x1}_re',f'{x4}_re')}, d1i={self.sub(f'{x1}_im',f'{x4}_im')};")
            self.o(f"{T} s2r={self.add(f'{x2}_re',f'{x3}_re')}, s2i={self.add(f'{x2}_im',f'{x3}_im')};")
            self.o(f"{T} d2r={self.sub(f'{x2}_re',f'{x3}_re')}, d2i={self.sub(f'{x2}_im',f'{x3}_im')};")
            self.o(f"{T} ss_r={self.add('s1r','s2r')}, ss_i={self.add('s1i','s2i')};")
            self.o(f"{T} sd_r={self.sub('s1r','s2r')}, sd_i={self.sub('s1i','s2i')};")
            self.o(f"{T} t0r={self.fnma('cD','ss_r',f'{x0}_re')}, t0i={self.fnma('cD','ss_i',f'{x0}_im')};")
            self.o(f"{x0}_re={self.add(f'{x0}_re','ss_r')}; {x0}_im={self.add(f'{x0}_im','ss_i')};")
            self.o(f"{T} p1r={self.fma('cA','sd_r','t0r')}, p1i={self.fma('cA','sd_i','t0i')};")
            self.o(f"{T} p2r={self.fnma('cA','sd_r','t0r')}, p2i={self.fnma('cA','sd_i','t0i')};")

        self.o(f"{T} qar={self.fma('cC','d2i',self.mul('cB','d1i'))}, "
               f"qai={self.fma('cC','d2r',self.mul('cB','d1r'))};")
        self.o(f"{T} qbr={self.fms('cB','d2i',self.mul('cC','d1i'))}, "
               f"qbi={self.fms('cB','d2r',self.mul('cC','d1r'))};")

        if fwd:
            self.o(f"{x1}_re={self.add('p1r','qar')}; {x1}_im={self.sub('p1i','qai')};")
            self.o(f"{x4}_re={self.sub('p1r','qar')}; {x4}_im={self.add('p1i','qai')};")
            self.o(f"{x2}_re={self.sub('p2r','qbr')}; {x2}_im={self.add('p2i','qbi')};")
            self.o(f"{x3}_re={self.add('p2r','qbr')}; {x3}_im={self.sub('p2i','qbi')};")
        else:
            self.o(f"{x1}_re={self.sub('p1r','qar')}; {x1}_im={self.add('p1i','qai')};")
            self.o(f"{x4}_re={self.add('p1r','qar')}; {x4}_im={self.sub('p1i','qai')};")
            self.o(f"{x2}_re={self.add('p2r','qbr')}; {x2}_im={self.sub('p2i','qbi')};")
            self.o(f"{x3}_re={self.sub('p2r','qbr')}; {x3}_im={self.add('p2i','qbi')};")
        self.o(f"}}")

    # ── DFT-2 butterfly (direction-independent) ──
    def emit_radix2(self, v, d, label=""):
        """DFT-2: y0=x0+x1, y1=x0−x1. Same fwd and bwd."""
        if label: self.c(f"{label} [{d}]")
        a, b = v[0], v[1]; T = self.isa.T
        self.o(f"{{ {T} tr={self.sub(f'{a}_re',f'{b}_re')}, ti={self.sub(f'{a}_im',f'{b}_im')};")
        self.o(f"  {a}_re={self.add(f'{a}_re',f'{b}_re')}; {a}_im={self.add(f'{a}_im',f'{b}_im')};")
        self.o(f"  {b}_re=tr; {b}_im=ti; }}")


# ═══════════════════════════════════════════════════════════════
# KERNEL BODY EMITTERS
# ═══════════════════════════════════════════════════════════════

def emit_kernel_body(em, d, itw_set, variant):
    """Emit inner loop body for notw, dit_tw, dif_tw.
    5×2 CT: Pass 1 = 2× DFT-5, Pass 2 = 5× DFT-2 with W10 twiddles."""
    xv5 = [f"x{i}" for i in range(N1)]
    xv2 = [f"x{i}" for i in range(N2)]

    # PASS 1: N2=2 radix-5 sub-FFTs
    for n2 in range(N2):
        em.c(f"sub-FFT n2={n2}")
        for n1 in range(N1):
            n = N2 * n1 + n2
            em.emit_load(f"x{n1}", n)
            if variant == 'dit_tw' and n > 0:
                em.emit_ext_tw(f"x{n1}", n - 1, d)
        em.b()
        em.emit_radix5(xv5, d, f"radix-5 n2={n2}")
        em.b()
        for k1 in range(N1):
            em.emit_spill(f"x{k1}", n2 * N1 + k1)
        em.b()

    # PASS 2: N1=5 radix-2 columns with internal W10 twiddles
    em.c("PASS 2: 5x radix-2 columns")
    em.b()
    for k1 in range(N1):
        em.c(f"column k1={k1}")
        for n2 in range(N2):
            em.emit_reload(f"x{n2}", n2 * N1 + k1)
        em.b()
        if k1 > 0:
            for n2 in range(1, N2):
                e = (n2 * k1) % N
                em.emit_int_tw(f"x{n2}", e, d)
            em.b()
        em.emit_radix2(xv2, d, f"radix-2 k1={k1}")
        em.b()
        if variant == 'dif_tw':
            for k2 in range(N2):
                m = k1 + N1 * k2
                if m > 0:
                    em.emit_ext_tw(f"x{k2}", m - 1, d)
            em.b()
        for k2 in range(N2):
            em.emit_store(f"x{k2}", k1 + N1 * k2)
        em.b()


def emit_kernel_body_log3(em, d, itw_set, variant):
    """Log3 kernel: load W^1, derive W^2..W^9 (8 cmuls), apply as twiddles."""
    is_dit = variant == 'dit_tw_log3'
    T = em.isa.T
    xv5 = [f"x{i}" for i in range(N1)]
    xv2 = [f"x{i}" for i in range(N2)]

    # Load base twiddle w1, derive w2..w9
    em.c("Log3: load W^1, derive W^2..W^9 (8 cmuls, depth-4 parallel chain)")
    em.c("  Chain: w2=w1*w1, w3=w1*w2, w4=w2*w2, w5=w1*w4,")
    em.c("         w6=w3*w3, w7=w3*w4, w8=w4*w4, w9=w4*w5")
    em.c("  Critical path: w1->w2->w4->w8 = depth 4 (vs 8 serial)")
    tb, tbi = em._tw_buf(), em._tw_buf_im()
    ta = em._tw_addr(0)
    if em.isa.name == 'scalar':
        em.o(f"const double w1r = {tb}[{ta}], w1i = {tbi}[{ta}];")
    else:
        ld = f"{em.isa.p}_load_pd"
        em.o(f"const {T} w1r = {ld}(&{tb}[{ta}]), w1i = {ld}(&{tbi}[{ta}]);")
    em.n_load += 2
    # Parallel derivation chain (depth 4):
    #   Level 1: w2 = w1*w1
    #   Level 2: w3 = w1*w2,  w4 = w2*w2
    #   Level 3: w5 = w1*w4,  w6 = w3*w3,  w7 = w3*w4,  w8 = w4*w4
    #   Level 4: w9 = w4*w5
    derivations = [
        ('w2', 'w1', 'w1'),   # w2 = w1^2
        ('w3', 'w1', 'w2'),   # w3 = w1*w2
        ('w4', 'w2', 'w2'),   # w4 = w2^2
        ('w5', 'w1', 'w4'),   # w5 = w1*w4
        ('w6', 'w3', 'w3'),   # w6 = w3^2
        ('w7', 'w3', 'w4'),   # w7 = w3*w4
        ('w8', 'w4', 'w4'),   # w8 = w4^2
        ('w9', 'w4', 'w5'),   # w9 = w4*w5
    ]
    for dst, a, b in derivations:
        em.o(f"{T} {dst}r, {dst}i;")
        em.emit_cmul(f"{dst}r", f"{dst}i", f"{a}r", f"{a}i", f"{b}r", f"{b}i", 'fwd')
    em.b()

    # PASS 1: 2× DFT-5 with log3 twiddles
    for n2 in range(N2):
        em.c(f"sub-FFT n2={n2}")
        for n1 in range(N1):
            n = N2 * n1 + n2
            em.emit_load(f"x{n1}", n)
            if is_dit and n > 0:
                em.emit_cmul_inplace(f"x{n1}", f"w{n}r", f"w{n}i", d)
        em.b()
        em.emit_radix5(xv5, d, f"radix-5 n2={n2}")
        em.b()
        for k1 in range(N1):
            em.emit_spill(f"x{k1}", n2 * N1 + k1)
        em.b()

    # PASS 2: 5× DFT-2 with internal W10 twiddles (hoisted broadcasts)
    em.c("PASS 2: 5x radix-2 columns")
    em.b()
    for k1 in range(N1):
        em.c(f"column k1={k1}")
        for n2 in range(N2):
            em.emit_reload(f"x{n2}", n2 * N1 + k1)
        em.b()
        if k1 > 0:
            for n2 in range(1, N2):
                e = (n2 * k1) % N
                em.emit_int_tw(f"x{n2}", e, d)
            em.b()
        em.emit_radix2(xv2, d, f"radix-2 k1={k1}")
        em.b()
        if not is_dit:
            for k2 in range(N2):
                m = k1 + N1 * k2
                if m > 0:
                    em.emit_cmul_inplace(f"x{k2}", f"w{m}r", f"w{m}i", d)
            em.b()
        for k2 in range(N2):
            em.emit_store(f"x{k2}", k1 + N1 * k2)
        em.b()


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def emit_twiddle_constants(L, itw_set):
    """Emit scalar W10 constant definitions (#ifndef guarded)."""
    by_tN = {}
    for (e, tN) in sorted(itw_set): by_tN.setdefault(tN, []).append(e)
    for tN in sorted(by_tN):
        g = f"FFT_W{tN}_TWIDDLES_DEFINED"
        L.append(f"#ifndef {g}"); L.append(f"#define {g}")
        for e in sorted(by_tN[tN]):
            wr, wi = wN(e, tN); l = wN_label(e, tN)
            L.append(f"static const double {l}_re = {wr:.20e};")
            L.append(f"static const double {l}_im = {wi:.20e};")
        L.append(f"#endif"); L.append("")


def emit_dft5_constants(em):
    """Emit 4 DFT-5 constants + sign_flip."""
    T = em.isa.T
    if em.isa.name == 'scalar':
        em.o(f"const double cA = {cA_val:.45f};")
        em.o(f"const double cB = {cB_val:.45f};")
        em.o(f"const double cC = {cC_val:.45f};")
        em.o(f"const double cD = {cD_val:.45f};")
    else:
        s = f"{em.isa.p}_set1_pd"
        em.o(f"const {T} sign_flip = {s}(-0.0);")
        em.o(f"const {T} cA = {s}({cA_val:.45f});")
        em.o(f"const {T} cB = {s}({cB_val:.45f});")
        em.o(f"const {T} cC = {s}({cC_val:.45f});")
        em.o(f"const {T} cD = {s}({cD_val:.45f});")
        em.o(f"(void)sign_flip;")
    em.b()


def emit_dft5_constants_raw(lines, isa, indent=1):
    """Append DFT-5 constant declarations directly to a line list."""
    pad = "    " * indent; T = isa.T
    if isa.name == 'scalar':
        lines.append(f"{pad}const double cA = {cA_val:.45f};")
        lines.append(f"{pad}const double cB = {cB_val:.45f};")
        lines.append(f"{pad}const double cC = {cC_val:.45f};")
        lines.append(f"{pad}const double cD = {cD_val:.45f};")
    else:
        s = f"{isa.p}_set1_pd"
        lines.append(f"{pad}const {T} sign_flip = {s}(-0.0);")
        lines.append(f"{pad}const {T} cA = {s}({cA_val:.45f});")
        lines.append(f"{pad}const {T} cB = {s}({cB_val:.45f});")
        lines.append(f"{pad}const {T} cC = {s}({cC_val:.45f});")
        lines.append(f"{pad}const {T} cD = {s}({cD_val:.45f});")
        lines.append(f"{pad}(void)sign_flip;")
    lines.append(f"")


def emit_hoisted_w10(L_or_em, isa, itw_set, indent=1, use_em=False):
    """Emit hoisted W10 broadcast constants."""
    if not itw_set: return
    T = isa.T
    if use_em:
        em = L_or_em
        if isa.name != 'scalar':
            s = f"{isa.p}_set1_pd"
            for (e, tN) in sorted(itw_set):
                label = wN_label(e, tN)
                em.o(f"const {T} tw_{label}_re = {s}({label}_re);")
                em.o(f"const {T} tw_{label}_im = {s}({label}_im);")
        else:
            for (e, tN) in sorted(itw_set):
                label = wN_label(e, tN)
                em.o(f"const double tw_{label}_re = {label}_re;")
                em.o(f"const double tw_{label}_im = {label}_im;")
        em.b()
    else:
        L = L_or_em; pad = "    " * indent
        if isa.name != 'scalar':
            s = f"{isa.p}_set1_pd"
            for (e, tN) in sorted(itw_set):
                label = wN_label(e, tN)
                L.append(f"{pad}const {T} tw_{label}_re = {s}({label}_re);")
                L.append(f"{pad}const {T} tw_{label}_im = {s}({label}_im);")
        else:
            for (e, tN) in sorted(itw_set):
                label = wN_label(e, tN)
                L.append(f"{pad}const double tw_{label}_re = {label}_re;")
                L.append(f"{pad}const double tw_{label}_im = {label}_im;")
        L.append(f"")


def insert_stats_into_header(lines, stats):
    table = [" *", " * -- Operation counts per k-step --", " *"]
    sep20 = '-' * 20; s5 = '-' * 5; s3 = '-' * 3; s4 = '-' * 4
    table.append(f" *   {'kernel':<20s} {'add':>5s} {'sub':>5s} {'mul':>5s} {'neg':>5s}"
                 f" {'fma':>5s} {'fms':>5s} | {'arith':>5s} {'flops':>5s}"
                 f" | {'ld':>3s} {'st':>3s} {'sp':>3s} {'rl':>3s} {'mem':>4s}")
    table.append(f" *   {sep20} {s5} {s5} {s5} {s5}"
                 f" {s5} {s5} + {s5} {s5}"
                 f" + {s3} {s3} {s3} {s3} {s4}")
    for k in sorted(stats.keys()):
        s = stats[k]
        sp = s.get('spill', 0); rl = s.get('reload', 0)
        table.append(f" *   {k:<20s} {s['add']:5d} {s['sub']:5d} {s['mul']:5d} {s['neg']:5d}"
                     f" {s['fma']:5d} {s['fms']:5d} | {s['total_arith']:5d} {s['flops']:5d}"
                     f" | {s['load']:3d} {s['store']:3d} {sp:3d} {rl:3d} {s['total_mem']:4d}")
    table.append(" *")
    for i, line in enumerate(lines):
        if line.strip() == '*/':
            for j, tl in enumerate(table):
                lines.insert(i + j, tl)
            return


# ═══════════════════════════════════════════════════════════════
# FILE EMITTER — standard variants
# ═══════════════════════════════════════════════════════════════

def emit_file(isa, itw_set, variant):
    """Emit complete header for one ISA and one variant."""
    em = Emitter(isa); T = isa.T
    is_log3 = variant.endswith('_log3')

    vmap = {
        'notw':         ('radix10_n1_dit_kernel',        None,   'N1 (no twiddle)'),
        'dit_tw':       ('radix10_tw_flat_dit_kernel',   'flat', 'DIT twiddled (flat)'),
        'dif_tw':       ('radix10_tw_flat_dif_kernel',   'flat', 'DIF twiddled (flat)'),
        'dit_tw_log3':  ('radix10_tw_log3_dit_kernel',   'flat', 'DIT twiddled (log3)'),
        'dif_tw_log3':  ('radix10_tw_log3_dif_kernel',   'flat', 'DIF twiddled (log3)'),
    }
    func_base, tw_params, vname = vmap[variant]
    guard = f"FFT_RADIX10_{isa.name.upper()}_{variant.upper()}_H"

    em.L.append(f"/**")
    em.L.append(f" * @file fft_radix10_{isa.name}_{variant}.h")
    em.L.append(f" * @brief DFT-10 {isa.name.upper()} {vname}")
    em.L.append(f" *")
    em.L.append(f" * 5x2 CT, k-step={isa.k_step}")
    em.L.append(f" * Generated by gen_radix10.py")
    em.L.append(f" */")
    em.L.append(f""); em.L.append(f"#ifndef {guard}"); em.L.append(f"#define {guard}")
    em.L.append(f"")

    if isa.name != 'scalar':
        em.L.append(f"#include <immintrin.h>"); em.L.append(f"")

    emit_twiddle_constants(em.L, itw_set)

    lp = isa.ld_prefix
    if isa.name == 'scalar':
        em.L.append(f"#ifndef {lp}_LD"); em.L.append(f"#define {lp}_LD(p) (*(p))")
        em.L.append(f"#define {lp}_ST(p,v) (*(p)=(v))"); em.L.append(f"#endif")
    elif isa.name == 'avx2':
        em.L.append(f"#ifndef {lp}_LD"); em.L.append(f"#define {lp}_LD(p) _mm256_load_pd(p)")
        em.L.append(f"#define {lp}_ST(p,v) _mm256_store_pd((p),(v))"); em.L.append(f"#endif")
    else:
        em.L.append(f"#ifndef {lp}_LD"); em.L.append(f"#define {lp}_LD(p) _mm512_load_pd(p)")
        em.L.append(f"#define {lp}_ST(p,v) _mm512_store_pd((p),(v))"); em.L.append(f"#endif")
    em.L.append(f"#define LD {lp}_LD"); em.L.append(f"#define ST {lp}_ST"); em.L.append(f"")

    stats = {}
    for d in ['fwd', 'bwd']:
        em.reset(); em.addr_mode = 'K'

        if isa.target: em.L.append(f"static {isa.target} void")
        else:          em.L.append(f"static void")
        em.L.append(f"{func_base}_{d}_{isa.name}(")
        em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
        em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
        if tw_params:
            em.L.append(f"    const double * __restrict__ tw_re, const double * __restrict__ tw_im,")
        em.L.append(f"    size_t K)")
        em.L.append(f"{{"); em.ind = 1

        emit_dft5_constants(em)

        spill_total = N * isa.sm
        if isa.name == 'scalar':
            em.o(f"double spill_re[{N}], spill_im[{N}];")
        else:
            em.o(f"{isa.align} double spill_re[{spill_total}];")
            em.o(f"{isa.align} double spill_im[{spill_total}];")
        em.b()

        em.o(f"{T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im,x4_re,x4_im;")
        em.b()

        emit_hoisted_w10(em, isa, itw_set, use_em=True)

        if isa.name == 'scalar': em.o(f"for (size_t k = 0; k < K; k++) {{")
        else:                    em.o(f"for (size_t k = 0; k < K; k += {isa.k_step}) {{")
        em.ind += 1
        if is_log3: emit_kernel_body_log3(em, d, itw_set, variant)
        else:       emit_kernel_body(em, d, itw_set, variant)
        em.ind -= 1
        em.o("}"); em.L.append("}"); em.L.append("")
        stats[d] = em.get_stats()

    # U=2 pipelining (AVX-512 only, non-log3)
    if isa.name == 'avx512' and not is_log3:
        VL = isa.k_step
        u2_name = func_base + '_u2'
        has_tw = tw_params is not None
        for d in ['fwd', 'bwd']:
            em_u1 = Emitter(isa); em_u1.addr_mode = 'K'; em_u1.ind = 2
            emit_kernel_body(em_u1, d, itw_set, variant)
            body_a = '\n'.join(em_u1.L)
            body_b = body_a.replace('+k]', f'+k+{VL}]').replace('+k)', f'+k+{VL})')
            if isa.target: em.L.append(f"static {isa.target} void")
            else:          em.L.append(f"static void")
            em.L.append(f"{u2_name}_{d}_{isa.name}(")
            em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
            em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
            if has_tw:
                em.L.append(f"    const double * __restrict__ tw_re, const double * __restrict__ tw_im,")
            em.L.append(f"    size_t K)")
            em.L.append(f"{{")
            em.L.append(f"    /* U=2: two independent k-groups per iteration.")
            em.L.append(f"     * Spill buffer is intentionally shared: pipeline A completes all stores")
            em.L.append(f"     * before pipeline B starts, so slots are reused, not shared simultaneously. */")
            emit_dft5_constants_raw(em.L, isa, indent=1)
            em.L.append(f"    {isa.align} double spill_re[{N*isa.sm}];")
            em.L.append(f"    {isa.align} double spill_im[{N*isa.sm}];")
            emit_hoisted_w10(em.L, isa, itw_set, indent=1)
            xdecl = f"        {T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im,x4_re,x4_im;"
            em.L.append(f"    for (size_t k = 0; k < K; k += {VL*2}) {{")
            em.L.append(f"        {{ /* Pipeline A: k */")
            em.L.append(xdecl); em.L.append(body_a); em.L.append(f"        }}")
            em.L.append(f"        {{ /* Pipeline B: k+{VL} */")
            em.L.append(xdecl); em.L.append(body_b); em.L.append(f"        }}")
            em.L.append(f"    }}"); em.L.append(f"}}"); em.L.append(f"")

    em.L.append(f"#undef LD"); em.L.append(f"#undef ST"); em.L.append(f"")
    em.L.append(f"#endif /* {guard} */")
    insert_stats_into_header(em.L, stats)
    return em.L, stats


# ═══════════════════════════════════════════════════════════════
# FILE EMITTER — CT variants
# ═══════════════════════════════════════════════════════════════

def emit_file_ct(isa, itw_set, ct_variant):
    """Emit FFTW-style n1 or t1 codelet."""
    em = Emitter(isa); T = isa.T

    is_n1     = ct_variant == 'ct_n1'
    is_t1_dit = ct_variant == 'ct_t1_dit'
    is_t1_dit_log3 = ct_variant == 'ct_t1_dit_log3'
    is_t1_dif = ct_variant == 'ct_t1_dif'
    em.addr_mode = 'n1' if is_n1 else 't1'

    if is_n1:
        func_base = "radix10_n1"; vname = "n1 (separate is/os)"
    elif is_t1_dif:
        func_base = "radix10_t1_dif"; vname = "t1 DIF (in-place twiddle)"
    elif is_t1_dit_log3:
        func_base = "radix10_t1_dit_log3"; vname = "t1 DIT log3 (in-place, derived twiddles)"
    else:
        func_base = "radix10_t1_dit"; vname = "t1 DIT (in-place twiddle)"

    guard = f"FFT_RADIX10_{isa.name.upper()}_CT_{ct_variant.upper()}_H"
    em.L.append(f"/**")
    em.L.append(f" * @file fft_radix10_{isa.name}_{ct_variant}.h")
    em.L.append(f" * @brief DFT-10 {isa.name.upper()} {vname}")
    em.L.append(f" *")
    em.L.append(f" * FFTW-style codelet for recursive CT executor.")
    em.L.append(f" * Generated by gen_radix10.py --variant {ct_variant}")
    em.L.append(f" */")
    em.L.append(f""); em.L.append(f"#ifndef {guard}"); em.L.append(f"#define {guard}")
    em.L.append(f"")

    if isa.name != 'scalar':
        em.L.append(f"#include <immintrin.h>"); em.L.append(f"")

    emit_twiddle_constants(em.L, itw_set)

    lp = isa.ld_prefix + "_CT"
    if isa.name == 'scalar':
        em.L.append(f"#ifndef {lp}_LD"); em.L.append(f"#define {lp}_LD(p) (*(p))")
        em.L.append(f"#define {lp}_ST(p,v) (*(p)=(v))"); em.L.append(f"#endif")
    elif isa.name == 'avx2':
        em.L.append(f"#ifndef {lp}_LD"); em.L.append(f"#define {lp}_LD(p) _mm256_loadu_pd(p)")
        em.L.append(f"#define {lp}_ST(p,v) _mm256_storeu_pd((p),(v))"); em.L.append(f"#endif")
    else:
        em.L.append(f"#ifndef {lp}_LD"); em.L.append(f"#define {lp}_LD(p) _mm512_loadu_pd(p)")
        em.L.append(f"#define {lp}_ST(p,v) _mm512_storeu_pd((p),(v))"); em.L.append(f"#endif")
    em.L.append(f"#define LD {lp}_LD"); em.L.append(f"#define ST {lp}_ST"); em.L.append(f"")

    for d in ['fwd', 'bwd']:
        em.reset(); em.addr_mode = 'n1' if is_n1 else 't1'

        if isa.target: em.L.append(f"static {isa.target} void")
        else:          em.L.append(f"static void")

        if is_n1:
            em.L.append(f"{func_base}_{d}_{isa.name}(")
            em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
            em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
            if isa.name == 'scalar':
                em.L.append(f"    size_t is, size_t os, size_t vl, size_t ivs, size_t ovs)")
            else:
                em.L.append(f"    size_t is, size_t os, size_t vl)")
        else:
            em.L.append(f"{func_base}_{d}_{isa.name}(")
            em.L.append(f"    double * __restrict__ rio_re, double * __restrict__ rio_im,")
            em.L.append(f"    const double * __restrict__ W_re, const double * __restrict__ W_im,")
            if isa.name == 'scalar':
                em.L.append(f"    size_t ios, size_t mb, size_t me, size_t ms)")
            else:
                em.L.append(f"    size_t ios, size_t me)")

        em.L.append(f"{{"); em.ind = 1

        emit_dft5_constants(em)

        spill_total = N * isa.sm
        if isa.name == 'scalar':
            em.o(f"double spill_re[{N}], spill_im[{N}];")
        else:
            em.o(f"{isa.align} double spill_re[{spill_total}];")
            em.o(f"{isa.align} double spill_im[{spill_total}];")
        em.b()

        em.o(f"{T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im,x4_re,x4_im;")
        em.b()

        emit_hoisted_w10(em, isa, itw_set, use_em=True)

        if is_n1:
            if isa.name == 'scalar': em.o(f"for (size_t k = 0; k < vl; k++) {{")
            else:                    em.o(f"for (size_t k = 0; k < vl; k += {isa.k_step}) {{")
        else:
            if isa.name == 'scalar': em.o(f"for (size_t m = mb; m < me; m++) {{")
            else:                    em.o(f"for (size_t m = 0; m < me; m += {isa.k_step}) {{")

        em.ind += 1
        if is_t1_dit_log3:
            emit_kernel_body_log3(em, d, itw_set, 'dit_tw_log3')
        else:
            kernel_variant = 'notw' if is_n1 else ('dif_tw' if is_t1_dif else 'dit_tw')
            emit_kernel_body(em, d, itw_set, kernel_variant)
        em.ind -= 1
        em.o("}"); em.L.append("}"); em.L.append("")

    # ── n1_ovs: butterfly → tbuf → 4×4 transpose ──
    if is_n1 and isa.name != 'scalar':
        VL = isa.k_step
        for d in ['fwd', 'bwd']:
            em.L.append(f"")
            if isa.target: em.L.append(f"static {isa.target} void")
            else:          em.L.append(f"static void")
            em.L.append(f"radix10_n1_ovs_{d}_{isa.name}(")
            em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
            em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
            em.L.append(f"    size_t is, size_t os, size_t vl, size_t ovs)")
            em.L.append(f"{{")
            em.L.append(f"    /* n1_ovs: butterfly -> tbuf, 2x 4x4 transpose (bins 0-3, 4-7) + scatter (bins 8,9) */")
            em.L.append(f"    {isa.align} double tbuf_re[{R * VL}];")
            em.L.append(f"    {isa.align} double tbuf_im[{R * VL}];")
            em.L.append(f"    {isa.align} double spill_re[{R * VL}];")
            em.L.append(f"    {isa.align} double spill_im[{R * VL}];")

            emit_dft5_constants_raw(em.L, isa, indent=1)
            emit_hoisted_w10(em.L, isa, itw_set, indent=1)

            em.L.append(f"    {T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im,x4_re,x4_im;")
            em.L.append(f"")
            em.L.append(f"    for (size_t k = 0; k < vl; k += {VL}) {{")

            em2 = Emitter(isa); em2.L = []; em2.ind = 2; em2.reset()
            em2.addr_mode = 'n1_ovs'
            emit_kernel_body(em2, d, itw_set, 'notw')
            em.L.extend(em2.L)

            # 4×4 transpose group 1: bins 0-3
            em.L.append(f"        /* 4x4 transpose: bins 0-3 */")
            for comp, arr in [('re', 'out_re'), ('im', 'out_im')]:
                bn = f"tbuf_{comp}"
                em.L.append(f"        {{ {T} a_=LD(&{bn}[0*{VL}]), b_=LD(&{bn}[1*{VL}]);")
                em.L.append(f"          {T} c_=LD(&{bn}[2*{VL}]), d_=LD(&{bn}[3*{VL}]);")
                if isa.name == 'avx2':
                    em.L.append(f"          {T} lo_ab=_mm256_unpacklo_pd(a_,b_), hi_ab=_mm256_unpackhi_pd(a_,b_);")
                    em.L.append(f"          {T} lo_cd=_mm256_unpacklo_pd(c_,d_), hi_cd=_mm256_unpackhi_pd(c_,d_);")
                    em.L.append(f"          _mm256_storeu_pd(&{arr}[(k+0)*ovs+os*0], _mm256_permute2f128_pd(lo_ab,lo_cd,0x20));")
                    em.L.append(f"          _mm256_storeu_pd(&{arr}[(k+1)*ovs+os*0], _mm256_permute2f128_pd(hi_ab,hi_cd,0x20));")
                    em.L.append(f"          _mm256_storeu_pd(&{arr}[(k+2)*ovs+os*0], _mm256_permute2f128_pd(lo_ab,lo_cd,0x31));")
                    em.L.append(f"          _mm256_storeu_pd(&{arr}[(k+3)*ovs+os*0], _mm256_permute2f128_pd(hi_ab,hi_cd,0x31));")
                else:
                    for j in range(VL):
                        for b in range(4):
                            em.L.append(f"          {arr}[(k+{j})*ovs+os*{b}] = {bn}[{b}*{VL}+{j}];")
                em.L.append(f"        }}")

            # 4×4 transpose group 2: bins 4-7
            em.L.append(f"        /* 4x4 transpose: bins 4-7 */")
            for comp, arr in [('re', 'out_re'), ('im', 'out_im')]:
                bn = f"tbuf_{comp}"
                em.L.append(f"        {{ {T} a_=LD(&{bn}[4*{VL}]), b_=LD(&{bn}[5*{VL}]);")
                em.L.append(f"          {T} c_=LD(&{bn}[6*{VL}]), d_=LD(&{bn}[7*{VL}]);")
                if isa.name == 'avx2':
                    em.L.append(f"          {T} lo_ab=_mm256_unpacklo_pd(a_,b_), hi_ab=_mm256_unpackhi_pd(a_,b_);")
                    em.L.append(f"          {T} lo_cd=_mm256_unpacklo_pd(c_,d_), hi_cd=_mm256_unpackhi_pd(c_,d_);")
                    em.L.append(f"          _mm256_storeu_pd(&{arr}[(k+0)*ovs+os*4], _mm256_permute2f128_pd(lo_ab,lo_cd,0x20));")
                    em.L.append(f"          _mm256_storeu_pd(&{arr}[(k+1)*ovs+os*4], _mm256_permute2f128_pd(hi_ab,hi_cd,0x20));")
                    em.L.append(f"          _mm256_storeu_pd(&{arr}[(k+2)*ovs+os*4], _mm256_permute2f128_pd(lo_ab,lo_cd,0x31));")
                    em.L.append(f"          _mm256_storeu_pd(&{arr}[(k+3)*ovs+os*4], _mm256_permute2f128_pd(hi_ab,hi_cd,0x31));")
                else:
                    for j in range(VL):
                        for b in range(4):
                            em.L.append(f"          {arr}[(k+{j})*ovs+os*{4+b}] = {bn}[{4+b}*{VL}+{j}];")
                em.L.append(f"        }}")

            # Scatter bins 8-9
            em.L.append(f"        /* Scatter: bins 8,9 */")
            for bin_idx in [8, 9]:
                if isa.name == 'avx2':
                    for comp, arr in [('re', 'out_re'), ('im', 'out_im')]:
                        bn = f"tbuf_{comp}"
                        em.L.append(f"        {{ {T} _v = LD(&{bn}[{bin_idx}*{VL}]);")
                        em.L.append(f"          __m128d lo = _mm256_castpd256_pd128(_v), hi = _mm256_extractf128_pd(_v, 1);")
                        em.L.append(f"          _mm_storel_pd(&{arr}[(k+0)*ovs+os*{bin_idx}], lo);")
                        em.L.append(f"          _mm_storeh_pd(&{arr}[(k+1)*ovs+os*{bin_idx}], lo);")
                        em.L.append(f"          _mm_storel_pd(&{arr}[(k+2)*ovs+os*{bin_idx}], hi);")
                        em.L.append(f"          _mm_storeh_pd(&{arr}[(k+3)*ovs+os*{bin_idx}], hi); }}")
                else:
                    for comp, arr in [('re', 'out_re'), ('im', 'out_im')]:
                        bn = f"tbuf_{comp}"
                        for j in range(VL):
                            em.L.append(f"        {arr}[(k+{j})*ovs+os*{bin_idx}] = {bn}[{bin_idx}*{VL}+{j}];")

            em.L.append(f"    }}")
            em.L.append(f"}}")

    em.L.append(f"")
    em.L.append(f"#undef LD"); em.L.append(f"#undef ST"); em.L.append(f"")
    em.L.append(f"#endif /* {guard} */")
    return em.L


# ═══════════════════════════════════════════════════════════════
# SV CODELET GENERATION — text transform from K-loop output
# ═══════════════════════════════════════════════════════════════

def _t2_to_sv(body):
    """Transform a t2 codelet body to sv: strip k-loop, K→vs."""
    lines = body.split('\n')
    out = []; in_loop = False; depth = 0
    for line in lines:
        stripped = line.strip()
        if not in_loop and 'for (size_t k' in stripped and 'k < K' in stripped:
            in_loop = True; depth = 1; continue
        if in_loop:
            for ch in stripped:
                if ch == '{': depth += 1
                elif ch == '}': depth -= 1
            if depth <= 0:
                in_loop = False
                if stripped == '}': continue
        line = re.sub(r'(\d+)\*K\+k', r'\1*vs', line)
        line = re.sub(r'\[k\]', '[0]', line)
        if line.startswith('        '): line = line[4:]
        out.append(line)
    return '\n'.join(out)


def emit_sv_variants(t2_lines, isa, variant):
    """Extract t2 functions, emit sv (single-vector) versions."""
    if isa.name == 'scalar': return []
    if variant.endswith('_log3'): return []

    text = '\n'.join(t2_lines)

    if variant == 'notw':
        t2_pattern = 'radix10_n1_dit_kernel'
        sv_name = 'radix10_n1sv_kernel'
    elif variant == 'dit_tw':
        t2_pattern = 'radix10_tw_flat_dit_kernel'
        sv_name = 'radix10_t1sv_dit_kernel'
    elif variant == 'dif_tw':
        t2_pattern = 'radix10_tw_flat_dif_kernel'
        sv_name = 'radix10_t1sv_dif_kernel'
    else:
        return []

    out = []; out.append('')
    for d in ('fwd', 'bwd'):
        pat = f'{t2_pattern}_{d}_{isa.name}'
        match = re.search(
            r'((?:static .+?\n)?(?:__attribute__.+?\n)?'
            r'void\s+' + re.escape(pat) + r'\(.*?\)\s*\{.*?\n\})',
            text, re.DOTALL)
        if not match: continue
        body = match.group(0)
        sv_body = _t2_to_sv(body)
        sv_body = sv_body.replace(pat, f'{sv_name}_{d}_{isa.name}')
        sv_body = re.sub(r'size_t K\)', 'size_t vs)', sv_body)
        out.append(sv_body); out.append('')
    return out


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

ALL_VARIANTS = [
    'notw', 'dit_tw', 'dif_tw', 'dit_tw_log3', 'dif_tw_log3',
    'ct_n1', 'ct_t1_dit', 'ct_t1_dit_log3', 'ct_t1_dif',
]

CT_VARIANTS = {'ct_n1', 'ct_t1_dit', 'ct_t1_dit_log3', 'ct_t1_dif'}


def main():
    parser = argparse.ArgumentParser(description='DFT-10 unified codelet generator')
    parser.add_argument('--isa', default='avx2',
                        help='scalar, avx2, avx512, or all')
    parser.add_argument('--variant', default='all',
                        help='notw, dit_tw, dif_tw, dit_tw_log3, dif_tw_log3, '
                             'ct_n1, ct_t1_dit, ct_t1_dit_log3, ct_t1_dif, or all')
    args = parser.parse_args()

    isa_list = list(ALL_ISA.values()) if args.isa == 'all' else [ALL_ISA[args.isa]]
    var_list = ALL_VARIANTS if args.variant == 'all' else [args.variant]

    itw_set = collect_internal_twiddles()

    for isa in isa_list:
        for variant in var_list:
            if variant in CT_VARIANTS:
                lines = emit_file_ct(isa, itw_set, variant)
            else:
                lines, stats = emit_file(isa, itw_set, variant)
                sv = emit_sv_variants(lines, isa, variant)
                if sv:
                    # Insert SV before #undef LD (so LD/ST macros are still active)
                    for i in range(len(lines)-1, -1, -1):
                        if '#undef LD' in lines[i]:
                            for j, sl in enumerate(sv):
                                lines.insert(i+j, sl)
                            break
            print('\n'.join(lines))
            print(f"/* gen_radix10.py: {isa.name}/{variant} — {len(lines)} lines */",
                  file=sys.stderr)


if __name__ == '__main__':
    main()
