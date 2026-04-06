#!/usr/bin/env python3
#
# Copyright (c) 2025 Tugbars Heptaskin
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
gen_radix12.py -- Unified DFT-12 codelet generator for VectorFFT

4x3 Cooley-Tukey: pass 1 = 3x DFT-4, pass 2 = 4x DFT-3 columns.
Internal W12 twiddles: 5 unique exponents, only 3 require cmul.
  W12^3 = -j (free j-rotation), W12^6 = -1 (free negation).
12 spill + 12 reload per kernel.

DFT-4 butterfly (zero constants):
  s=x0+x2, d=x0-x2, t=x1+x3, u=x1-x3
  y0=s+t, y2=s-t, y1=d+j*u, y3=d-j*u
  8 add/sub, zero multiplies.

DFT-3 butterfly (2 constants, FMA-fused S terms):
  KHALF = 0.5, KS = sqrt(3)/2
  y0 = x0 + x1 + x2
  R = x0 - 0.5*(x1+x2)        [fnma(KHALF, s, x0)]
  S = KS * (x1 - x2)           [cross re/im]
  y1 = R + j*S,  y2 = R - j*S  (fwd: y1_re = R_re + S_im, etc.)
  FMA-fused: y1_re = fma(KS, di, Rr),  y1_im = fnma(KS, dr, Ri)

Register pressure per pass:
  Pass 1: DFT-4 on 4 inputs -> peak 8 YMM
  Pass 2: DFT-3 on 3 inputs -> peak ~10 YMM (3 data + 2 const + temps)
  Spill: 12 complex values = 96 bytes (fits L1)

n1_ovs: 12 bins = 3 groups of 4 -> three clean 4x4 SIMD transposes, zero scatter.

Log3 twiddle derivation (R=12 external):
  Load W^1, derive W^2..W^11 via depth-4 parallel chain (10 cmuls)

Usage:
  python3 gen_radix12.py --isa avx2 --variant all
  python3 gen_radix12.py --isa all  --variant ct_n1
"""

import sys, math, argparse, re

R = 12
N, N1, N2 = 12, 4, 3   # 4x3 CT: Pass1=3xDFT-4, Pass2=4xDFT-3

# DFT-3 constants
KHALF_val = 0.500000000000000000000000000000000000000000000
KS_val    = 0.866025403784438646763723170752936183471402627

# ═══════════════════════════════════════════════════════════════
# TWIDDLE ANALYSIS
# ═══════════════════════════════════════════════════════════════

def wN(e, tN):
    e = e % tN; a = 2.0 * math.pi * e / tN
    return (math.cos(a), -math.sin(a))

def wN_label(e, tN):
    return f"W{tN}_{e % tN}"

def collect_internal_twiddles():
    """Internal W12: k1=1..3, n2=1..2 -> exponents {1,2,3,4,6}.
    W12^3=-j (free), W12^6=-1 (free). Only {1,2,4} require cmul."""
    tw = set()
    for k1 in range(1, N1):
        for n2 in range(1, N2):
            e = (k1 * n2) % N
            if e != 0:
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

ISA_SCALAR = ISAConfig('scalar', 'double', 1, 1, '', 1, '', '', 'R12S')
ISA_AVX2   = ISAConfig('avx2', '__m256d', 4, 4, '_mm256', 4,
    '__attribute__((target("avx2,fma")))',
    '__attribute__((aligned(32)))', 'R12A')
ISA_AVX512 = ISAConfig('avx512', '__m512d', 8, 8, '_mm512', 8,
    '__attribute__((target("avx512f,avx512dq,fma")))',
    '__attribute__((aligned(64)))', 'R12L')

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
        self.tw_hoisted = False

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
        self.n_fma += 1
        if self.isa.name == 'scalar': return f"({c})-({a})*({b})"
        return f"{self.isa.p}_fnmadd_pd({a},{b},{c})"

    # ── Addressing ──
    def _in_addr(self, n, ke="k"):
        if self.addr_mode in ('n1', 'n1_ovs'): return f"{n}*is+{ke}"
        elif self.addr_mode == 't1':
            if self.isa.name == 'scalar': return f"m*ms+{n}*ios"
            return f"m+{n}*ios"
        elif self.addr_mode == 't1_oop': return f"m+{n}*is"
        return f"{n}*K+{ke}"

    def _out_addr(self, m, ke="k"):
        if self.addr_mode == 'n1': return f"{m}*os+{ke}"
        elif self.addr_mode == 'n1_ovs': return f"{m}*{self.isa.k_step}"
        elif self.addr_mode == 't1':
            if self.isa.name == 'scalar': return f"m*ms+{m}*ios"
            return f"m+{m}*ios"
        elif self.addr_mode == 't1_oop': return f"m+{m}*os"
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
        self.spill_c += 1; sm = self.isa.sm
        if self.isa.name == 'scalar':
            self.o(f"spill_re[{slot}] = {v}_re; spill_im[{slot}] = {v}_im;")
        else:
            self.o(f"{self.isa.p}_store_pd(&spill_re[{slot}*{sm}], {v}_re);")
            self.o(f"{self.isa.p}_store_pd(&spill_im[{slot}*{sm}], {v}_im);")

    def emit_reload(self, v, slot):
        self.reload_c += 1; sm = self.isa.sm
        if self.isa.name == 'scalar':
            self.o(f"{v}_re = spill_re[{slot}]; {v}_im = spill_im[{slot}];")
        else:
            self.o(f"{v}_re = {self.isa.p}_load_pd(&spill_re[{slot}*{sm}]);")
            self.o(f"{v}_im = {self.isa.p}_load_pd(&spill_im[{slot}*{sm}]);")

    # ── External twiddle ──
    def _tw_addr(self, tw_idx, ke="k"):
        if self.addr_mode in ('t1', 't1_oop'): return f"{tw_idx}*me+m"
        return f"{tw_idx}*K+{ke}"
    def _tw_buf(self):
        return "W_re" if self.addr_mode in ('t1', 't1_oop') else "tw_re"
    def _tw_buf_im(self):
        return "W_im" if self.addr_mode in ('t1', 't1_oop') else "tw_im"

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
    def emit_hoist_all_tw_scalars(self, R):
        """Emit broadcast of twiddle scalars BEFORE the m-loop.
        Register-budget-aware: only hoists what fits in SIMD registers.
        AVX2: 5 twiddle pairs (10 of 16 YMM), AVX-512: 12 pairs (24 of 32 ZMM).
        Remaining twiddles are broadcast inline inside the loop."""
        T = self.isa.T
        n_tw = R - 1
        if self.isa.name == 'scalar':
            max_hoist = n_tw  # scalar has unlimited "registers"
        elif self.isa.name == 'avx2':
            max_hoist = 5     # 10 YMM for twiddles, 6 for data/temps
        else:  # avx512
            max_hoist = 12    # 24 ZMM for twiddles, 8 for data/temps
        n_hoist = min(n_tw, max_hoist)
        self.tw_hoisted_set = set(range(n_hoist))
        for i in range(n_hoist):
            if self.isa.name == 'scalar':
                self.o(f"const double tw{i}_re = W_re[{i}], tw{i}_im = W_im[{i}];")
            elif self.isa.name == 'avx2':
                self.o(f"const {T} tw{i}_re = _mm256_broadcast_sd(&W_re[{i}]);")
                self.o(f"const {T} tw{i}_im = _mm256_broadcast_sd(&W_im[{i}]);")
            else:  # avx512
                self.o(f"const {T} tw{i}_re = _mm512_set1_pd(W_re[{i}]);")
                self.o(f"const {T} tw{i}_im = _mm512_set1_pd(W_im[{i}]);")

    def emit_apply_hoisted_tw(self, v, tw_idx, d):
        """Apply pre-hoisted twiddle tw{idx} to variable v (no broadcast)."""
        fwd = (d == 'fwd')
        T = self.isa.T
        wr = f"tw{tw_idx}_re"
        wi = f"tw{tw_idx}_im"
        if self.isa.name == 'scalar':
            self.o(f"{{ double tr = {v}_re;")
            if fwd:
                self.o(f"  {v}_re = {v}_re*{wr} - {v}_im*{wi};")
                self.o(f"  {v}_im = tr*{wi} + {v}_im*{wr}; }}")
            else:
                self.o(f"  {v}_re = {v}_re*{wr} + {v}_im*{wi};")
                self.o(f"  {v}_im = {v}_im*{wr} - tr*{wi}; }}")
        else:
            self.o(f"{{ const {T} tr = {v}_re;")
            if fwd:
                self.o(f"  {v}_re = {self.fms(f'{v}_re',wr,self.mul(f'{v}_im',wi))};")
                self.o(f"  {v}_im = {self.fma('tr',wi,self.mul(f'{v}_im',wr))}; }}")
            else:
                self.o(f"  {v}_re = {self.fma(f'{v}_re',wr,self.mul(f'{v}_im',wi))};")
                self.o(f"  {v}_im = {self.fms(f'{v}_im',wr,self.mul('tr',wi))}; }}")

    def emit_ext_tw_scalar(self, v, tw_idx, d):
        """Emit twiddle multiply using scalar broadcast (t1s variant).
        W_re/W_im are (R-1) scalars, NOT (R-1)*me arrays.
        Broadcasts one double to SIMD width."""
        if self.tw_hoisted and tw_idx in self.tw_hoisted_set:
            return self.emit_apply_hoisted_tw(v, tw_idx, d)
        fwd = (d == 'fwd')
        T = self.isa.T
        self.n_load += 2
        if self.isa.name == 'scalar':
            self.o(f"{{ double wr = W_re[{tw_idx}], wi = W_im[{tw_idx}], tr = {v}_re;")
            if fwd:
                self.o(f"  {v}_re = {v}_re*wr - {v}_im*wi;")
                self.o(f"  {v}_im = tr*wi + {v}_im*wr; }}")
            else:
                self.o(f"  {v}_re = {v}_re*wr + {v}_im*wi;")
                self.o(f"  {v}_im = {v}_im*wr - tr*wi; }}")
        elif self.isa.name == 'avx2':
            self.o(f"{{ const {T} wr = _mm256_broadcast_sd(&W_re[{tw_idx}]);")
            self.o(f"  const {T} wi = _mm256_broadcast_sd(&W_im[{tw_idx}]);")
            self.o(f"  const {T} tr = {v}_re;")
            if fwd:
                self.o(f"  {v}_re = {self.fms(f'{v}_re','wr',self.mul(f'{v}_im','wi'))};")
                self.o(f"  {v}_im = {self.fma('tr','wi',self.mul(f'{v}_im','wr'))}; }}")
            else:
                self.o(f"  {v}_re = {self.fma(f'{v}_re','wr',self.mul(f'{v}_im','wi'))};")
                self.o(f"  {v}_im = {self.fms(f'{v}_im','wr',self.mul('tr','wi'))}; }}")
        else:  # avx512
            self.o(f"{{ const {T} wr = _mm512_set1_pd(W_re[{tw_idx}]);")
            self.o(f"  const {T} wi = _mm512_set1_pd(W_im[{tw_idx}]);")
            self.o(f"  const {T} tr = {v}_re;")
            if fwd:
                self.o(f"  {v}_re = {self.fms(f'{v}_re','wr',self.mul(f'{v}_im','wi'))};")
                self.o(f"  {v}_im = {self.fma('tr','wi',self.mul(f'{v}_im','wr'))}; }}")
            else:
                self.o(f"  {v}_re = {self.fma(f'{v}_re','wr',self.mul(f'{v}_im','wi'))};")
                self.o(f"  {v}_im = {self.fms(f'{v}_im','wr',self.mul('tr','wi'))}; }}")

    # -- Complex multiply helpers (for log3 derivation) --

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

    # ── Internal W12 twiddle with trivial cases ──
    def emit_int_tw(self, v, e, d):
        """Apply W12^e. e=0: skip. e=3: -j rotation. e=6: negate. e=9: +j rotation. else: cmul."""
        e = e % N
        if e == 0: return
        fwd = (d == 'fwd')
        T = self.isa.T
        if e == 6:
            # W12^6 = -1: negate both components
            self.o(f"{v}_re = {self.neg(f'{v}_re')}; {v}_im = {self.neg(f'{v}_im')};")
            return
        if e == 3:
            # W12^3 = -j: multiply by (0, -1)
            # fwd: (re,im) * (0,-1) = (im, -re) ... wait
            # W12^3 = cos(pi/2) - j*sin(pi/2) = 0 - j = -j
            # (a+jb)*(-j) = -ja + b = b - ja -> (b, -a) ... that's (im, -re)
            # But for bwd: (a+jb)*(+j) = ja - b = -b + ja -> (-im, re)
            if fwd:
                self.o(f"{{ {T} tr = {v}_re; {v}_re = {v}_im; {v}_im = {self.neg('tr')}; }}")
            else:
                self.o(f"{{ {T} tr = {v}_re; {v}_re = {self.neg(f'{v}_im')}; {v}_im = tr; }}")
            return
        if e == 9:
            # W12^9 = +j: conjugate of W12^3
            # (a+jb)*(+j) = ja - b -> (-b, a) = (-im, re)
            # bwd: (a+jb)*(-j) -> (b, -a) = (im, -re)
            if fwd:
                self.o(f"{{ {T} tr = {v}_re; {v}_re = {self.neg(f'{v}_im')}; {v}_im = tr; }}")
            else:
                self.o(f"{{ {T} tr = {v}_re; {v}_re = {v}_im; {v}_im = {self.neg('tr')}; }}")
            return
        # General cmul via hoisted broadcast constants
        label = wN_label(e, N)
        self.emit_cmul_inplace(v, f"tw_{label}_re", f"tw_{label}_im", d)

    # ── DFT-4 butterfly (zero constants) ──
    def emit_radix4(self, v, d, label=""):
        fwd = (d == 'fwd'); T = self.isa.T
        if label: self.c(f"{label} [{d}]")
        x0, x1, x2, x3 = v[0], v[1], v[2], v[3]
        self.o(f"{{")
        self.o(f"{T} sr={self.add(f'{x0}_re',f'{x2}_re')}, si={self.add(f'{x0}_im',f'{x2}_im')};")
        self.o(f"{T} dr={self.sub(f'{x0}_re',f'{x2}_re')}, di={self.sub(f'{x0}_im',f'{x2}_im')};")
        self.o(f"{T} tr={self.add(f'{x1}_re',f'{x3}_re')}, ti={self.add(f'{x1}_im',f'{x3}_im')};")
        self.o(f"{T} ur={self.sub(f'{x1}_re',f'{x3}_re')}, ui={self.sub(f'{x1}_im',f'{x3}_im')};")
        self.o(f"{x0}_re={self.add('sr','tr')}; {x0}_im={self.add('si','ti')};")
        self.o(f"{x2}_re={self.sub('sr','tr')}; {x2}_im={self.sub('si','ti')};")
        if fwd:
            self.o(f"{x1}_re={self.add('dr','ui')}; {x1}_im={self.sub('di','ur')};")
            self.o(f"{x3}_re={self.sub('dr','ui')}; {x3}_im={self.add('di','ur')};")
        else:
            self.o(f"{x1}_re={self.sub('dr','ui')}; {x1}_im={self.add('di','ur')};")
            self.o(f"{x3}_re={self.add('dr','ui')}; {x3}_im={self.sub('di','ur')};")
        self.o(f"}}")

    # ── DFT-3 butterfly (FMA-fused S terms, 2 constants) ──
    def emit_radix3(self, v, d, label=""):
        """DFT-3 on v[0..2] in-place.
        y0 = x0 + x1 + x2
        R = x0 - 0.5*(x1+x2)
        y1 = R + j*KS*(x1-x2),  y2 = R - j*KS*(x1-x2)
        FMA-fused: y1_re = fma(KS, di, Rr), y1_im = fnma(KS, dr, Ri) [fwd]"""
        fwd = (d == 'fwd'); T = self.isa.T
        if label: self.c(f"{label} [{d}]")
        x0, x1, x2 = v[0], v[1], v[2]
        self.o(f"{{")
        # s = x1+x2, d = x1-x2
        self.o(f"{T} sr={self.add(f'{x1}_re',f'{x2}_re')}, si={self.add(f'{x1}_im',f'{x2}_im')};")
        self.o(f"{T} dr={self.sub(f'{x1}_re',f'{x2}_re')}, di={self.sub(f'{x1}_im',f'{x2}_im')};")
        # R = x0 - 0.5*s  [fnma(KHALF, s, x0)]
        self.o(f"{T} Rr={self.fnma('KHALF','sr',f'{x0}_re')}, Ri={self.fnma('KHALF','si',f'{x0}_im')};")
        # y0 = x0 + s
        self.o(f"{x0}_re={self.add(f'{x0}_re','sr')}; {x0}_im={self.add(f'{x0}_im','si')};")
        # y1 = R + j*KS*d,  y2 = R - j*KS*d
        # fwd: j*KS*d = (+KS*di, -KS*dr)
        #   y1_re = Rr + KS*di = fma(KS, di, Rr)
        #   y1_im = Ri - KS*dr = fnma(KS, dr, Ri)
        #   y2_re = Rr - KS*di = fnma(KS, di, Rr)
        #   y2_im = Ri + KS*dr = fma(KS, dr, Ri)
        if fwd:
            self.o(f"{x1}_re={self.fma('KS','di','Rr')}; {x1}_im={self.fnma('KS','dr','Ri')};")
            self.o(f"{x2}_re={self.fnma('KS','di','Rr')}; {x2}_im={self.fma('KS','dr','Ri')};")
        else:
            self.o(f"{x1}_re={self.fnma('KS','di','Rr')}; {x1}_im={self.fma('KS','dr','Ri')};")
            self.o(f"{x2}_re={self.fma('KS','di','Rr')}; {x2}_im={self.fnma('KS','dr','Ri')};")
        self.o(f"}}")


# ═══════════════════════════════════════════════════════════════
# KERNEL BODY EMITTERS
# ═══════════════════════════════════════════════════════════════

def emit_kernel_body(em, d, itw_set, variant):
    """4x3 CT: Pass 1 = 3x DFT-4, Pass 2 = 4x DFT-3 with W12 twiddles."""
    xv4 = [f"x{i}" for i in range(N1)]
    xv3 = [f"x{i}" for i in range(N2)]

    # PASS 1: N2=3 radix-4 sub-FFTs
    for n2 in range(N2):
        em.c(f"sub-FFT n2={n2}")
        for n1 in range(N1):
            n = N2 * n1 + n2
            em.emit_load(f"x{n1}", n)
            if variant == 'dit_tw' and n > 0:
                em.emit_ext_tw(f"x{n1}", n - 1, d)
            elif variant == 'dit_tw_scalar' and n > 0:
                em.emit_ext_tw_scalar(f"x{n1}", n - 1, d)
        em.b()
        em.emit_radix4(xv4, d, f"radix-4 n2={n2}")
        em.b()
        for k1 in range(N1):
            em.emit_spill(f"x{k1}", n2 * N1 + k1)
        em.b()

    # PASS 2: N1=4 radix-3 columns with internal W12 twiddles
    em.c("PASS 2: 4x radix-3 columns")
    em.b()
    for k1 in range(N1):
        em.c(f"column k1={k1}")
        for n2 in range(N2):
            em.emit_reload(f"x{n2}", n2 * N1 + k1)
        em.b()
        if k1 > 0:
            for n2 in range(1, N2):
                e = (k1 * n2) % N
                em.emit_int_tw(f"x{n2}", e, d)
            em.b()
        em.emit_radix3(xv3, d, f"radix-3 k1={k1}")
        em.b()
        if variant == 'dif_tw':
            for k2 in range(N2):
                m = k1 + N1 * k2
                if m > 0:
                    em.emit_ext_tw(f"x{k2}", m - 1, d)
            em.b()
        elif variant == 'dif_tw_scalar':
            for k2 in range(N2):
                m = k1 + N1 * k2
                if m > 0:
                    em.emit_ext_tw_scalar(f"x{k2}", m - 1, d)
            em.b()
        for k2 in range(N2):
            em.emit_store(f"x{k2}", k1 + N1 * k2)
        em.b()


def emit_kernel_body_log3(em, d, itw_set, variant):
    """Log3 kernel: load W^1, derive W^2..W^11 (10 cmuls, depth-4 chain)."""
    is_dit = variant == 'dit_tw_log3'
    T = em.isa.T
    xv4 = [f"x{i}" for i in range(N1)]
    xv3 = [f"x{i}" for i in range(N2)]

    em.c("Log3: load W^1, derive W^2..W^11 (10 cmuls, depth-4 parallel chain)")
    tb, tbi = em._tw_buf(), em._tw_buf_im()
    ta = em._tw_addr(0)
    if em.isa.name == 'scalar':
        em.o(f"const double w1r = {tb}[{ta}], w1i = {tbi}[{ta}];")
    else:
        ld = f"{em.isa.p}_load_pd"
        em.o(f"const {T} w1r = {ld}(&{tb}[{ta}]), w1i = {ld}(&{tbi}[{ta}]);")
    em.n_load += 2
    derivations = [
        ('w2',  'w1',  'w1'),   # w2  = w1^2
        ('w3',  'w1',  'w2'),   # w3  = w1*w2
        ('w4',  'w2',  'w2'),   # w4  = w2^2
        ('w5',  'w1',  'w4'),   # w5  = w1*w4
        ('w6',  'w3',  'w3'),   # w6  = w3^2
        ('w7',  'w3',  'w4'),   # w7  = w3*w4
        ('w8',  'w4',  'w4'),   # w8  = w4^2
        ('w9',  'w4',  'w5'),   # w9  = w4*w5
        ('w10', 'w5',  'w5'),   # w10 = w5^2
        ('w11', 'w5',  'w6'),   # w11 = w5*w6
    ]
    for dst, a, b in derivations:
        em.o(f"{T} {dst}r, {dst}i;")
        em.emit_cmul(f"{dst}r", f"{dst}i", f"{a}r", f"{a}i", f"{b}r", f"{b}i", 'fwd')
    em.b()

    for n2 in range(N2):
        em.c(f"sub-FFT n2={n2}")
        for n1 in range(N1):
            n = N2 * n1 + n2
            em.emit_load(f"x{n1}", n)
            if is_dit and n > 0:
                em.emit_cmul_inplace(f"x{n1}", f"w{n}r", f"w{n}i", d)
        em.b()
        em.emit_radix4(xv4, d, f"radix-4 n2={n2}")
        em.b()
        for k1 in range(N1):
            em.emit_spill(f"x{k1}", n2 * N1 + k1)
        em.b()

    em.c("PASS 2: 4x radix-3 columns")
    em.b()
    for k1 in range(N1):
        em.c(f"column k1={k1}")
        for n2 in range(N2):
            em.emit_reload(f"x{n2}", n2 * N1 + k1)
        em.b()
        if k1 > 0:
            for n2 in range(1, N2):
                e = (k1 * n2) % N
                em.emit_int_tw(f"x{n2}", e, d)
            em.b()
        em.emit_radix3(xv3, d, f"radix-3 k1={k1}")
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


# ── AVX-512 single-pass: zero spill, all 12 values in registers ──

def emit_kernel_body_avx512_singlepass(em, d, itw_set, variant):
    """AVX-512 zero-spill: all 12 values live in ZMM registers.
    24 data ZMM + 2 const (KHALF,KS) + sign_flip = 27/32 ZMM.
    No spill buffer, no reload. Saves 24 memory ops per k-step."""
    groups = [[f"r{n2}{k1}" for k1 in range(N1)] for n2 in range(N2)]

    em.c("AVX-512 single-pass: all 12 values in registers, zero spill")

    # Load all 12 inputs
    for n2 in range(N2):
        em.c(f"Load group n2={n2}")
        for n1 in range(N1):
            n = N2 * n1 + n2
            em.emit_load(groups[n2][n1], n)
            if variant == 'dit_tw' and n > 0:
                em.emit_ext_tw(groups[n2][n1], n - 1, d)
            elif variant == 'dit_tw_scalar' and n > 0:
                em.emit_ext_tw_scalar(groups[n2][n1], n - 1, d)
    em.b()

    # 3x DFT-4 (each on one n2 group)
    for n2 in range(N2):
        em.emit_radix4(groups[n2], d, f"radix-4 n2={n2}")
    em.b()

    # Internal W12 twiddles
    em.c("Internal W12 twiddles (3 cmul + 1 j-rotation + 1 negate)")
    for k1 in range(1, N1):
        for n2 in range(1, N2):
            e = (k1 * n2) % N
            em.emit_int_tw(groups[n2][k1], e, d)
    em.b()

    # 4x DFT-3 (column-wise)
    for k1 in range(N1):
        col = [groups[n2][k1] for n2 in range(N2)]
        em.emit_radix3(col, d, f"radix-3 k1={k1}")
    em.b()

    # DIF twiddle
    if variant == 'dif_tw':
        for k1 in range(N1):
            for k2 in range(N2):
                m = k1 + N1 * k2
                if m > 0:
                    em.emit_ext_tw(groups[k2][k1], m - 1, d)
        em.b()
    elif variant == 'dif_tw_scalar':
        for k1 in range(N1):
            for k2 in range(N2):
                m = k1 + N1 * k2
                if m > 0:
                    em.emit_ext_tw_scalar(groups[k2][k1], m - 1, d)
        em.b()

    # Store all 12 outputs
    for k1 in range(N1):
        for k2 in range(N2):
            m = k1 + N1 * k2
            em.emit_store(groups[k2][k1], m)


# ── Fused last-group: skip spill for last DFT-4 group, save 8 mem ops ──

def emit_kernel_body_fused(em, d, itw_set, variant):
    """Fused: don't spill last DFT-4 group, keep x0..x3 live.
    Standard: 12 spill + 12 reload = 24 memory ops.
    Fused:     8 spill +  8 reload = 16 memory ops.  Saves 8 L1 round-trips.
    Pass 2 columns consume x{k1} from last group + 2 reloads into t0,t1."""
    xv4 = [f"x{i}" for i in range(N1)]
    n2_last = N2 - 1

    # Pass 1: first N2-1=2 groups with normal spill
    for n2 in range(N2 - 1):
        em.c(f"sub-FFT n2={n2}")
        for n1 in range(N1):
            n = N2 * n1 + n2
            em.emit_load(f"x{n1}", n)
            if variant == 'dit_tw' and n > 0:
                em.emit_ext_tw(f"x{n1}", n - 1, d)
            elif variant == 'dit_tw_scalar' and n > 0:
                em.emit_ext_tw_scalar(f"x{n1}", n - 1, d)
        em.b()
        em.emit_radix4(xv4, d, f"radix-4 n2={n2}")
        em.b()
        for k1 in range(N1):
            em.emit_spill(f"x{k1}", n2 * N1 + k1)
        em.b()

    # Last group: DFT-4, keep results in x0..x3 (NO spill)
    em.c(f"sub-FFT n2={n2_last} (fused: keep in x0..x3, no spill)")
    for n1 in range(N1):
        n = N2 * n1 + n2_last
        em.emit_load(f"x{n1}", n)
        if variant == 'dit_tw' and n > 0:
            em.emit_ext_tw(f"x{n1}", n - 1, d)
        elif variant == 'dit_tw_scalar' and n > 0:
            em.emit_ext_tw_scalar(f"x{n1}", n - 1, d)
    em.b()
    em.emit_radix4(xv4, d, f"radix-4 n2={n2_last}")
    em.b()

    # Pass 2: each column k1 reloads 2 from spill + uses x{k1} live
    em.c("PASS 2: fused radix-3 columns (2 reload + 1 live from x{k1})")
    em.b()
    for k1 in range(N1):
        em.c(f"column k1={k1}")
        for n2 in range(N2 - 1):
            em.emit_reload(f"t{n2}", n2 * N1 + k1)
        em.b()

        # Internal twiddles
        if k1 > 0:
            for n2 in range(1, N2 - 1):
                e = (k1 * n2) % N
                em.emit_int_tw(f"t{n2}", e, d)
            e_last = (k1 * n2_last) % N
            em.emit_int_tw(f"x{k1}", e_last, d)
            em.b()

        # DFT-3 on [t0, t1, x{k1}]
        dft3_vars = [f"t{n2}" for n2 in range(N2 - 1)] + [f"x{k1}"]
        em.emit_radix3(dft3_vars, d, f"radix-3 k1={k1}")
        em.b()

        # DIF twiddle
        if variant == 'dif_tw':
            for k2 in range(N2):
                m = k1 + N1 * k2
                if m > 0:
                    em.emit_ext_tw(dft3_vars[k2], m - 1, d)
            em.b()
        elif variant == 'dif_tw_scalar':
            for k2 in range(N2):
                m = k1 + N1 * k2
                if m > 0:
                    em.emit_ext_tw_scalar(dft3_vars[k2], m - 1, d)
            em.b()

        # Store
        for k2 in range(N2):
            m = k1 + N1 * k2
            em.emit_store(dft3_vars[k2], m)
        em.b()


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def emit_twiddle_constants(L, itw_set):
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


def emit_constants(em):
    """DFT-3: KHALF + KS. DFT-4: zero constants. Plus sign_flip for neg()."""
    T = em.isa.T
    if em.isa.name == 'scalar':
        em.o(f"const double KHALF = {KHALF_val:.45f};")
        em.o(f"const double KS = {KS_val:.45f};")
    else:
        s = f"{em.isa.p}_set1_pd"
        em.o(f"const {T} sign_flip = {s}(-0.0);")
        em.o(f"const {T} KHALF = {s}({KHALF_val:.45f});")
        em.o(f"const {T} KS = {s}({KS_val:.45f});")
    em.b()


def emit_constants_raw(lines, isa, indent=1):
    pad = "    " * indent; T = isa.T
    if isa.name == 'scalar':
        lines.append(f"{pad}const double KHALF = {KHALF_val:.45f};")
        lines.append(f"{pad}const double KS = {KS_val:.45f};")
    else:
        s = f"{isa.p}_set1_pd"
        lines.append(f"{pad}const {T} sign_flip = {s}(-0.0);")
        lines.append(f"{pad}const {T} KHALF = {s}({KHALF_val:.45f});")
        lines.append(f"{pad}const {T} KS = {s}({KS_val:.45f});")
    lines.append(f"")


def emit_hoisted_w12(L_or_em, isa, itw_set, indent=1, use_em=False):
    """Emit hoisted W12 broadcast constants (only for non-trivial exponents)."""
    # Filter out trivial exponents (3, 6, 9) that don't need broadcasts
    nontrivial = sorted([(e, tN) for (e, tN) in itw_set if e % 12 not in (0, 3, 6, 9)])
    if not nontrivial: return
    T = isa.T
    if use_em:
        em = L_or_em
        if isa.name != 'scalar':
            s = f"{isa.p}_set1_pd"
            for (e, tN) in nontrivial:
                label = wN_label(e, tN)
                em.o(f"const {T} tw_{label}_re = {s}({label}_re);")
                em.o(f"const {T} tw_{label}_im = {s}({label}_im);")
        else:
            for (e, tN) in nontrivial:
                label = wN_label(e, tN)
                em.o(f"const double tw_{label}_re = {label}_re;")
                em.o(f"const double tw_{label}_im = {label}_im;")
        em.b()
    else:
        L = L_or_em; pad = "    " * indent
        if isa.name != 'scalar':
            s = f"{isa.p}_set1_pd"
            for (e, tN) in nontrivial:
                label = wN_label(e, tN)
                L.append(f"{pad}const {T} tw_{label}_re = {s}({label}_re);")
                L.append(f"{pad}const {T} tw_{label}_im = {s}({label}_im);")
        else:
            for (e, tN) in nontrivial:
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
# FILE EMITTER -- standard variants
# ═══════════════════════════════════════════════════════════════

def emit_file(isa, itw_set, variant):
    em = Emitter(isa); T = isa.T
    is_log3 = variant.endswith('_log3')
    vmap = {
        'notw':         ('radix12_n1_dit_kernel',        None,   'N1 (no twiddle)'),
        'dit_tw':       ('radix12_tw_flat_dit_kernel',   'flat', 'DIT twiddled (flat)'),
        'dif_tw':       ('radix12_tw_flat_dif_kernel',   'flat', 'DIF twiddled (flat)'),
        'dit_tw_log3':  ('radix12_tw_log3_dit_kernel',   'flat', 'DIT twiddled (log3)'),
        'dif_tw_log3':  ('radix12_tw_log3_dif_kernel',   'flat', 'DIF twiddled (log3)'),
    }
    func_base, tw_params, vname = vmap[variant]
    guard = f"FFT_RADIX12_{isa.name.upper()}_{variant.upper()}_H"
    em.L.append(f"/**")
    em.L.append(f" * @file fft_radix12_{isa.name}_{variant}.h")
    em.L.append(f" * @brief DFT-12 {isa.name.upper()} {vname}")
    em.L.append(f" *")
    em.L.append(f" * 4x3 CT, k-step={isa.k_step}")
    em.L.append(f" * Generated by gen_radix12.py")
    em.L.append(f" */")
    em.L.append(f""); em.L.append(f"#ifndef {guard}"); em.L.append(f"#define {guard}"); em.L.append(f"")
    if isa.name != 'scalar': em.L.append(f"#include <immintrin.h>"); em.L.append(f"")
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
        emit_constants(em)

        use_singlepass = (isa.name == 'avx512' and not is_log3)
        use_fused = (not is_log3 and not use_singlepass)

        if use_singlepass:
            # AVX-512 zero-spill: all 12 in registers, no spill buffer
            for n2 in range(N2):
                vars_str = ", ".join([f"r{n2}{k1}_re,r{n2}{k1}_im" for k1 in range(N1)])
                em.o(f"{T} {vars_str};")
            em.b()
        elif use_fused:
            # Fused: reduced spill (8 entries), x0..x3 + t0,t1
            spill_count = (N2 - 1) * N1  # 2*4 = 8
            spill_total = spill_count * isa.sm
            if isa.name == 'scalar': em.o(f"double spill_re[{spill_count}], spill_im[{spill_count}];")
            else:
                em.o(f"{isa.align} double spill_re[{spill_total}];")
                em.o(f"{isa.align} double spill_im[{spill_total}];")
            em.b()
            em.o(f"{T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
            em.o(f"{T} t0_re,t0_im,t1_re,t1_im;")
            em.b()
        else:
            # Standard 2-pass (log3 path)
            spill_total = N * isa.sm
            if isa.name == 'scalar': em.o(f"double spill_re[{N}], spill_im[{N}];")
            else:
                em.o(f"{isa.align} double spill_re[{spill_total}];")
                em.o(f"{isa.align} double spill_im[{spill_total}];")
            em.b()
            em.o(f"{T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
            em.b()

        emit_hoisted_w12(em, isa, itw_set, use_em=True)
        if isa.name == 'scalar': em.o(f"for (size_t k = 0; k < K; k++) {{")
        else:                    em.o(f"for (size_t k = 0; k < K; k += {isa.k_step}) {{")
        em.ind += 1
        if is_log3:
            emit_kernel_body_log3(em, d, itw_set, variant)
        elif use_singlepass:
            emit_kernel_body_avx512_singlepass(em, d, itw_set, variant)
        elif use_fused:
            emit_kernel_body_fused(em, d, itw_set, variant)
        else:
            emit_kernel_body(em, d, itw_set, variant)
        em.ind -= 1
        em.o("}"); em.L.append("}"); em.L.append("")
        stats[d] = em.get_stats()

    # U=2 (AVX-512 only, non-log3)
    if isa.name == 'avx512' and not is_log3:
        VL = isa.k_step; u2_name = func_base + '_u2'; has_tw = tw_params is not None
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
            emit_constants_raw(em.L, isa, indent=1)
            em.L.append(f"    {isa.align} double spill_re[{N*isa.sm}];")
            em.L.append(f"    {isa.align} double spill_im[{N*isa.sm}];")
            emit_hoisted_w12(em.L, isa, itw_set, indent=1)
            xdecl = f"        {T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;"
            em.L.append(f"    for (size_t k = 0; k < K; k += {VL*2}) {{")
            em.L.append(f"        {{ /* Pipeline A: k */"); em.L.append(xdecl)
            em.L.append(body_a); em.L.append(f"        }}")
            em.L.append(f"        {{ /* Pipeline B: k+{VL} */"); em.L.append(xdecl)
            em.L.append(body_b); em.L.append(f"        }}")
            em.L.append(f"    }}"); em.L.append(f"}}"); em.L.append(f"")

    em.L.append(f"#undef LD"); em.L.append(f"#undef ST"); em.L.append(f"")
    em.L.append(f"#endif /* {guard} */")
    insert_stats_into_header(em.L, stats)
    return em.L, stats


# ═══════════════════════════════════════════════════════════════
# FILE EMITTER -- CT variants
# ═══════════════════════════════════════════════════════════════

def emit_file_ct(isa, itw_set, ct_variant):
    em = Emitter(isa); T = isa.T
    is_n1 = ct_variant == 'ct_n1'
    is_t1_dit = ct_variant == 'ct_t1_dit'
    is_t1s_dit = ct_variant == 'ct_t1s_dit'
    is_t1_dit_log3 = ct_variant == 'ct_t1_dit_log3'
    is_t1_dif = ct_variant == 'ct_t1_dif'
    is_t1_oop_dit = ct_variant == 'ct_t1_oop_dit'
    em.addr_mode = 'n1' if is_n1 else ('t1_oop' if is_t1_oop_dit else 't1')

    if is_n1:           func_base = "radix12_n1"; vname = "n1 (separate is/os)"
    elif is_t1_oop_dit: func_base = "radix12_t1_oop_dit"; vname = "t1_oop DIT (out-of-place, separate is/os, with twiddle)"
    elif is_t1s_dit:    func_base = "radix12_t1s_dit"; vname = "t1s DIT (in-place, scalar broadcast twiddle)"
    elif is_t1_dif:     func_base = "radix12_t1_dif"; vname = "t1 DIF (in-place twiddle)"
    elif is_t1_dit_log3: func_base = "radix12_t1_dit_log3"; vname = "t1 DIT log3 (in-place, derived twiddles)"
    else:               func_base = "radix12_t1_dit"; vname = "t1 DIT (in-place twiddle)"

    guard = f"FFT_RADIX12_{isa.name.upper()}_CT_{ct_variant.upper()}_H"
    em.L.append(f"/**")
    em.L.append(f" * @file fft_radix12_{isa.name}_{ct_variant}.h")
    em.L.append(f" * @brief DFT-12 {isa.name.upper()} {vname}")
    em.L.append(f" *")
    em.L.append(f" * FFTW-style codelet for recursive CT executor.")
    em.L.append(f" * Generated by gen_radix12.py --variant {ct_variant}")
    em.L.append(f" */")
    em.L.append(f""); em.L.append(f"#ifndef {guard}"); em.L.append(f"#define {guard}"); em.L.append(f"")
    if isa.name != 'scalar': em.L.append(f"#include <immintrin.h>"); em.L.append(f"")
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
        em.reset(); em.addr_mode = 'n1' if is_n1 else ('t1_oop' if is_t1_oop_dit else 't1')
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
        elif is_t1_oop_dit:
            em.L.append(f"{func_base}_{d}_{isa.name}(")
            em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
            em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
            em.L.append(f"    const double * __restrict__ W_re, const double * __restrict__ W_im,")
            em.L.append(f"    size_t is, size_t os, size_t me)")
        else:
            em.L.append(f"{func_base}_{d}_{isa.name}(")
            em.L.append(f"    double * __restrict__ rio_re, double * __restrict__ rio_im,")
            em.L.append(f"    const double * __restrict__ W_re, const double * __restrict__ W_im,")
            if isa.name == 'scalar':
                em.L.append(f"    size_t ios, size_t mb, size_t me, size_t ms)")
            else:
                em.L.append(f"    size_t ios, size_t me)")
        em.L.append(f"{{"); em.ind = 1
        emit_constants(em)
        spill_total = N * isa.sm
        if isa.name == 'scalar': em.o(f"double spill_re[{N}], spill_im[{N}];")
        else:
            em.o(f"{isa.align} double spill_re[{spill_total}];")
            em.o(f"{isa.align} double spill_im[{spill_total}];")
        em.b()
        em.o(f"{T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
        em.b()
        emit_hoisted_w12(em, isa, itw_set, use_em=True)

        # Hoist twiddle broadcasts before the loop (t1s only)
        if is_t1s_dit:
            em.tw_hoisted = True
            em.emit_hoist_all_tw_scalars(R)
            em.b()

        if is_n1:
            if isa.name == 'scalar': em.o(f"for (size_t k = 0; k < vl; k++) {{")
            else:                    em.o(f"for (size_t k = 0; k < vl; k += {isa.k_step}) {{")
        else:  # t1, t1_oop
            if isa.name == 'scalar' and not is_t1_oop_dit: em.o(f"for (size_t m = mb; m < me; m++) {{")
            else:                    em.o(f"for (size_t m = 0; m < me; m += {isa.k_step}) {{")
        em.ind += 1
        if is_t1s_dit:
            emit_kernel_body(em, d, itw_set, 'dit_tw_scalar')
        elif is_t1_dit_log3:
            emit_kernel_body_log3(em, d, itw_set, 'dit_tw_log3')
        else:
            kernel_variant = 'notw' if is_n1 else ('dif_tw' if is_t1_dif else 'dit_tw')
            if is_t1_oop_dit:
                em.addr_mode = 't1_oop'
                kernel_variant = 'dit_tw'
            emit_kernel_body(em, d, itw_set, kernel_variant)
        em.ind -= 1
        em.o("}"); em.L.append("}"); em.L.append("")

    # n1_ovs: 12 bins = 3 groups of 4, zero scatter
    if is_n1 and isa.name != 'scalar':
        VL = isa.k_step
        for d in ['fwd', 'bwd']:
            em.L.append(f"")
            if isa.target: em.L.append(f"static {isa.target} void")
            else:          em.L.append(f"static void")
            em.L.append(f"radix12_n1_ovs_{d}_{isa.name}(")
            em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
            em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
            em.L.append(f"    size_t is, size_t os, size_t vl, size_t ovs)")
            em.L.append(f"{{")
            em.L.append(f"    /* n1_ovs: butterfly -> tbuf, 3x 4x4 transpose (bins 0-11), zero scatter */")
            em.L.append(f"    {isa.align} double tbuf_re[{R * VL}];")
            em.L.append(f"    {isa.align} double tbuf_im[{R * VL}];")
            em.L.append(f"    {isa.align} double spill_re[{R * VL}];")
            em.L.append(f"    {isa.align} double spill_im[{R * VL}];")
            emit_constants_raw(em.L, isa, indent=1)
            emit_hoisted_w12(em.L, isa, itw_set, indent=1)
            em.L.append(f"    {T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
            em.L.append(f"")
            em.L.append(f"    for (size_t k = 0; k < vl; k += {VL}) {{")

            em2 = Emitter(isa); em2.L = []; em2.ind = 2; em2.reset()
            em2.addr_mode = 'n1_ovs'
            emit_kernel_body(em2, d, itw_set, 'notw')
            em.L.extend(em2.L)

            for grp in range(3):
                base_bin = grp * 4
                em.L.append(f"        /* 4x4 transpose: bins {base_bin}-{base_bin+3} */")
                for comp, arr in [('re', 'out_re'), ('im', 'out_im')]:
                    bn = f"tbuf_{comp}"
                    em.L.append(f"        {{ {T} a_=LD(&{bn}[{base_bin}*{VL}]), b_=LD(&{bn}[{base_bin+1}*{VL}]);")
                    em.L.append(f"          {T} c_=LD(&{bn}[{base_bin+2}*{VL}]), d_=LD(&{bn}[{base_bin+3}*{VL}]);")
                    if isa.name == 'avx2':
                        em.L.append(f"          {T} lo_ab=_mm256_unpacklo_pd(a_,b_), hi_ab=_mm256_unpackhi_pd(a_,b_);")
                        em.L.append(f"          {T} lo_cd=_mm256_unpacklo_pd(c_,d_), hi_cd=_mm256_unpackhi_pd(c_,d_);")
                        em.L.append(f"          _mm256_storeu_pd(&{arr}[(k+0)*ovs+os*{base_bin}], _mm256_permute2f128_pd(lo_ab,lo_cd,0x20));")
                        em.L.append(f"          _mm256_storeu_pd(&{arr}[(k+1)*ovs+os*{base_bin}], _mm256_permute2f128_pd(hi_ab,hi_cd,0x20));")
                        em.L.append(f"          _mm256_storeu_pd(&{arr}[(k+2)*ovs+os*{base_bin}], _mm256_permute2f128_pd(lo_ab,lo_cd,0x31));")
                        em.L.append(f"          _mm256_storeu_pd(&{arr}[(k+3)*ovs+os*{base_bin}], _mm256_permute2f128_pd(hi_ab,hi_cd,0x31));")
                    else:
                        for j in range(VL):
                            for b in range(4):
                                em.L.append(f"          {arr}[(k+{j})*ovs+os*{base_bin+b}] = {bn}[{base_bin+b}*{VL}+{j}];")
                    em.L.append(f"        }}")
            em.L.append(f"    }}")
            em.L.append(f"}}")

    em.L.append(f"")
    em.L.append(f"#undef LD"); em.L.append(f"#undef ST"); em.L.append(f"")
    em.L.append(f"#endif /* {guard} */")
    return em.L


# ═══════════════════════════════════════════════════════════════
# SV CODELET GENERATION
# ═══════════════════════════════════════════════════════════════

def _t2_to_sv(body):
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
    if isa.name == 'scalar': return []
    if variant.endswith('_log3'): return []
    text = '\n'.join(t2_lines)
    if variant == 'notw':
        t2_pattern = 'radix12_n1_dit_kernel'; sv_name = 'radix12_n1sv_kernel'
    elif variant == 'dit_tw':
        t2_pattern = 'radix12_tw_flat_dit_kernel'; sv_name = 'radix12_t1sv_dit_kernel'
    elif variant == 'dit_tw_scalar':
        t2_pattern = 'radix12_tw_flat_dit_kernel'; sv_name = 'radix12_t1sv_dit_kernel'
    elif variant == 'dif_tw':
        t2_pattern = 'radix12_tw_flat_dif_kernel'; sv_name = 'radix12_t1sv_dif_kernel'
    elif variant == 'dif_tw_scalar':
        t2_pattern = 'radix12_tw_flat_dif_kernel'; sv_name = 'radix12_t1sv_dif_kernel'
    else: return []
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
    'ct_n1', 'ct_t1_dit', 'ct_t1s_dit', 'ct_t1_dit_log3', 'ct_t1_dif', 'ct_t1_oop_dit',
]
CT_VARIANTS = {'ct_n1', 'ct_t1_dit', 'ct_t1s_dit', 'ct_t1_dit_log3', 'ct_t1_dif', 'ct_t1_oop_dit'}

def main():
    parser = argparse.ArgumentParser(description='DFT-12 unified codelet generator')
    parser.add_argument('--isa', default='avx2', help='scalar, avx2, avx512, or all')
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
                    for i in range(len(lines)-1, -1, -1):
                        if '#undef LD' in lines[i]:
                            for j, sl in enumerate(sv):
                                lines.insert(i+j, sl)
                            break
            print('\n'.join(lines))
            print(f"/* gen_radix12.py: {isa.name}/{variant} -- {len(lines)} lines */",
                  file=sys.stderr)

if __name__ == '__main__':
    main()
