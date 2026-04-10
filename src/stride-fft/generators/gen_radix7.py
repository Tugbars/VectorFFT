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
gen_radix7.py -- Unified DFT-7 codelet generator for VectorFFT

Monolithic DFT-7 butterfly with 6 constants (no sign_flip needed).
Sethi-Ullman scheduled for peak 16 YMMs on AVX2 with explicit spill buffer.

DFT-7 butterfly (forward):
  a=x1+x6, b=x2+x5, c=x3+x4
  d=x1-x6, e=x2-x5, f=x3-x4
  y0 = x0 + a + b + c
  R1 = fmadd(KP623, a, fnmadd(KP222, b, fnmadd(KP900, c, x0)))
  R2 = x0 - KP222*a - KP900*b + KP623*c  = fmadd(KP623, c, fnmadd(KP900, b, fnmadd(KP222, a, x0)))
  R3 = fnmadd(KP222, c, fmadd(KP623, b, fnmadd(KP900, a, x0)))
  S1_re = KP781*d_im + KP974*e_im + KP433*f_im   (sin(2pi/7)*d + sin(4pi/7)*e + sin(6pi/7)*f)
  S1_im = KP781*d_re + KP974*e_re + KP433*f_re
  S2_re = KP974*d_im - KP433*e_im - KP781*f_im   (sin(4pi/7)*d - sin(6pi/7)*e - sin(2pi/7)*f)
  S2_im = KP974*d_re - KP433*e_re - KP781*f_re
  S3_re = KP433*d_im - KP781*e_im + KP974*f_im   (sin(6pi/7)*d - sin(2pi/7)*e + sin(4pi/7)*f)
  S3_im = KP433*d_re - KP781*e_re + KP974*f_re
  y1 = R1 + (S1_re, -S1_im);  y6 = R1 - (S1_re, -S1_im)   [fwd]
  y2 = R2 + (S2_re, -S2_im);  y5 = R2 - (S2_re, -S2_im)   [fwd]
  y3 = R3 + (S3_re, -S3_im);  y4 = R3 - (S3_re, -S3_im)   [fwd]
  BWD: swap y_k and y_{7-k} roles (negate S imaginary path)

Constants (6):
  KP623 = cos(2pi/7)  = 0.623489801858733530525004884004239810632274731
  KP222 = |cos(4pi/7)|= 0.222520933956314404288902564496794759466355569
  KP900 = |cos(6pi/7)|= 0.900968867902419126236102319507445051165919162
  KP974 = sin(4pi/7)  = 0.974927912181823607018131682993931217232785801
  KP781 = sin(2pi/7)  = 0.781831482468029808708444526674057750232334519
  KP433 = sin(6pi/7)  = 0.433883739117558120475768332848358754609990728

Spill budget: d,e,f,R1,R2,R3 = 6 complex spills = 12 SIMD stores + 12 SIMD reloads.

Log3 twiddle derivation (R=7):
  Load w1 from table.  w2=w1*w1, w3=w1*w2, w4=w2*w2, w5=w2*w3, w6=w3*w3  (5 cmuls)

Usage:
  python3 gen_radix7.py --isa avx2 --variant all
  python3 gen_radix7.py --isa all --variant ct_n1
  python3 gen_radix7.py --isa avx2 --variant ct_t1_dit
"""

import sys, math, argparse, re

R = 7

# DFT-7 constants
KP623_val = 0.623489801858733530525004884004239810632274731
KP222_val = 0.222520933956314404288902564496794759466355569
KP900_val = 0.900968867902419126236102319507445051165919162
KP974_val = 0.974927912181823607018131682993931217232785801
KP781_val = 0.781831482468029808708444526674057750232334519
KP433_val = 0.433883739117558120475768332848358754609990728

# ================================================================
# ISA CONFIGURATION
# ================================================================

class ISAConfig:
    def __init__(self, name, T, width, k_step, p, sm, target, align, ld_prefix):
        self.name = name
        self.T = T
        self.width = width
        self.k_step = k_step
        self.p = p
        self.sm = sm
        self.target = target
        self.align = align
        self.ld_prefix = ld_prefix

ISA_SCALAR = ISAConfig('scalar', 'double', 1, 1, '', 1, '', '', 'R7S')
ISA_AVX2   = ISAConfig('avx2', '__m256d', 4, 4, '_mm256', 4,
    '__attribute__((target("avx2,fma")))',
    '__attribute__((aligned(32)))', 'R7A')
ISA_AVX512 = ISAConfig('avx512', '__m512d', 8, 8, '_mm512', 8,
    '__attribute__((target("avx512f,avx512dq,fma")))',
    '__attribute__((aligned(64)))', 'R7L')

ALL_ISA = {'scalar': ISA_SCALAR, 'avx2': ISA_AVX2, 'avx512': ISA_AVX512}

# ================================================================
# EMITTER
# ================================================================

class Emitter:
    def __init__(self, isa):
        self.isa = isa
        self.L = []
        self.ind = 1
        self.spill_c = 0
        self.reload_c = 0
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
        fl = self.n_add + self.n_sub + self.n_neg + self.n_mul + 2 * (self.n_fma + self.n_fms)
        return {
            'add': self.n_add, 'sub': self.n_sub, 'mul': self.n_mul, 'neg': self.n_neg,
            'fma': self.n_fma, 'fms': self.n_fms, 'total_arith': ta, 'flops': fl,
            'load': self.n_load, 'store': self.n_store,
            'spill': self.spill_c, 'reload': self.reload_c,
            'total_mem': self.n_load + self.n_store + self.spill_c + self.reload_c,
        }

    def o(self, t=""): self.L.append("    " * self.ind + t)
    def c(self, t): self.o(f"/* {t} */")
    def b(self): self.L.append("")

    # -- Arithmetic (counted) --
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

    # -- Spill / Reload to aligned stack buffer --
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

    # -- Addressing --
    def _in_addr(self, n, ke="k"):
        if self.addr_mode in ('n1', 'n1_ovs'):
            return f"{n}*is+{ke}"
        elif self.addr_mode == 't1':
            if self.isa.name == 'scalar': return f"m*ms+{n}*ios"
            return f"m+{n}*ios"
        elif self.addr_mode == 't1_oop':
            return f"m+{n}*is"
        return f"{n}*K+{ke}"

    def _out_addr(self, m, ke="k"):
        if self.addr_mode == 'n1':
            return f"{m}*os+{ke}"
        elif self.addr_mode == 'n1_ovs':
            return f"{m}*{self.isa.k_step}"
        elif self.addr_mode == 't1':
            if self.isa.name == 'scalar': return f"m*ms+{m}*ios"
            return f"m+{m}*ios"
        elif self.addr_mode == 't1_oop':
            return f"m+{m}*os"
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
            if getattr(self, 'store_scale', False):
                self.o(f"{ob}[{addr}] = scale * {v}_re; {obi}[{addr}] = scale * {v}_im;")
            else:
                self.o(f"{ob}[{addr}] = {v}_re; {obi}[{addr}] = {v}_im;")
        else:
            if getattr(self, 'store_scale', False):
                self.o(f"ST(&{ob}[{addr}], {self.isa.p}_mul_pd(vscale, {v}_re));")
                self.o(f"ST(&{obi}[{addr}], {self.isa.p}_mul_pd(vscale, {v}_im));")
            else:
                self.o(f"ST(&{ob}[{addr}], {v}_re);")
                self.o(f"ST(&{obi}[{addr}], {v}_im);")

    # -- External twiddle (flat) --
    def _tw_addr(self, tw_idx, ke="k"):
        if self.addr_mode in ('t1', 't1_oop'):
            return f"{tw_idx}*me+m"
        return f"{tw_idx}*K+{ke}"

    def _tw_buf(self):
        return "W_re" if self.addr_mode in ('t1', 't1_oop') else "tw_re"
    def _tw_buf_im(self):
        return "W_im" if self.addr_mode in ('t1', 't1_oop') else "tw_im"

    def emit_ext_tw(self, v, tw_idx, d, ke="k"):
        fwd = (d == 'fwd')
        T = self.isa.T
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

    # -- Complex multiply helpers (for log3 derivation) --
    def emit_cmul(self, dst_r, dst_i, ar, ai, br, bi, d):
        """Emit dst = a * b (fwd) or a * conj(b) (bwd)."""
        fwd = (d == 'fwd')
        if fwd:
            self.o(f"{dst_r} = {self.fms(ar, br, self.mul(ai, bi))};")
            self.o(f"{dst_i} = {self.fma(ar, bi, self.mul(ai, br))};")
        else:
            self.o(f"{dst_r} = {self.fma(ar, br, self.mul(ai, bi))};")
            self.o(f"{dst_i} = {self.fms(ai, br, self.mul(ar, bi))};")

    def emit_cmul_inplace(self, v, wr, wi, d):
        """Emit v *= (wr + j*wi) (fwd) or v *= conj(wr + j*wi) (bwd)."""
        fwd = (d == 'fwd')
        T = self.isa.T
        self.o(f"{{ {T} tr = {v}_re;")
        if fwd:
            self.o(f"  {v}_re = {self.fms(f'{v}_re', wr, self.mul(f'{v}_im', wi))};")
            self.o(f"  {v}_im = {self.fma('tr', wi, self.mul(f'{v}_im', wr))}; }}")
        else:
            self.o(f"  {v}_re = {self.fma(f'{v}_re', wr, self.mul(f'{v}_im', wi))};")
            self.o(f"  {v}_im = {self.fms(f'{v}_im', wr, self.mul('tr', wi))}; }}")

    # -- DFT-7 butterfly --
    def emit_radix7_butterfly(self, d, out_names=None):
        """Emit DFT-7 butterfly on x0..x6.
        AVX2/scalar: Sethi-Ullman 3-phase with explicit spill buffer (peak 16 YMM).
        AVX-512: single-pass, no spills (peak 24 of 32 ZMM).
        out_names: if provided, list of 7 output variable names.
                   Default: overwrite x0..x6."""
        fwd = (d == 'fwd')
        T = self.isa.T

        if out_names is None:
            out_names = [f'x{i}' for i in range(7)]

        if self.isa.name == 'avx512':
            return self._emit_radix7_butterfly_nospill(d, out_names)

        self.c(f"DFT-7 butterfly [{d}]")

        # ---- Phase 1: Load pairs, compute sums/diffs, spill diffs ----
        # x0..x6 are already declared and loaded by caller.
        # ar/ai..fr/fi, R1r/R1i..R3r/R3i are pre-declared at function scope.

        self.c("Phase 1 — symmetric/antisymmetric pairs")
        self.o(f"ar={self.add('x1_re','x6_re')}; ai={self.add('x1_im','x6_im')};")
        self.o(f"dr={self.sub('x1_re','x6_re')}; di={self.sub('x1_im','x6_im')};")
        self.o(f"br={self.add('x2_re','x5_re')}; bi={self.add('x2_im','x5_im')};")
        self.o(f"er={self.sub('x2_re','x5_re')}; ei={self.sub('x2_im','x5_im')};")
        self.o(f"cr={self.add('x3_re','x4_re')}; ci={self.add('x3_im','x4_im')};")
        self.o(f"fr={self.sub('x3_re','x4_re')}; fi={self.sub('x3_im','x4_im')};")

        # Spill diffs: d->slot0, e->slot1, f->slot2
        self.c("Spill diffs d,e,f")
        # Use temp variable names matching spill slots
        # We spill by storing d_re/d_im into spill_re[0]/spill_im[0], etc.
        # Use a small trick: emit_spill expects "v" where v_re and v_im exist.
        # We have ar/ai as separate vars, not "a_re"/"a_im", so we emit manually.
        sm = self.isa.sm
        if self.isa.name == 'scalar':
            self.o(f"spill_re[0] = dr; spill_im[0] = di;")
            self.o(f"spill_re[1] = er; spill_im[1] = ei;")
            self.o(f"spill_re[2] = fr; spill_im[2] = fi;")
        else:
            self.o(f"{self.isa.p}_store_pd(&spill_re[0*{sm}], dr);")
            self.o(f"{self.isa.p}_store_pd(&spill_im[0*{sm}], di);")
            self.o(f"{self.isa.p}_store_pd(&spill_re[1*{sm}], er);")
            self.o(f"{self.isa.p}_store_pd(&spill_im[1*{sm}], ei);")
            self.o(f"{self.isa.p}_store_pd(&spill_re[2*{sm}], fr);")
            self.o(f"{self.isa.p}_store_pd(&spill_im[2*{sm}], fi);")
        self.spill_c += 3

        # ---- Phase 2: y0 and cosine (R) terms using {x0,a,b,c} ----
        self.b()
        self.c("Phase 2 — R-term first level + y0 + R completion")

        # R-term depth reduction: compute 3 independent seeds from x0 first.
        # t1 = x0 - KP900*c,  t2 = x0 - KP222*a,  t3 = x0 - KP900*a
        # These are independent and issue on all FMA ports simultaneously.
        # Then R1/R2/R3 each need only depth 2 from their seed (not depth 3 from x0).
        self.c("3 independent R seeds from x0 (critical path: 1 FMA latency)")
        self.o(f"{{ {T} t1r={self.fnma('KP900','cr','x0_re')}, t1i={self.fnma('KP900','ci','x0_im')};")
        self.o(f"  {T} t2r={self.fnma('KP222','ar','x0_re')}, t2i={self.fnma('KP222','ai','x0_im')};")
        self.o(f"  {T} t3r={self.fnma('KP900','ar','x0_re')}, t3i={self.fnma('KP900','ai','x0_im')};")

        # R1 = KP623*a + (x0 - KP900*c - KP222*b) = fma(KP623, a, fnma(KP222, b, t1))
        self.o(f"  R1r={self.fma('KP623','ar',self.fnma('KP222','br','t1r'))};")
        self.o(f"  R1i={self.fma('KP623','ai',self.fnma('KP222','bi','t1i'))};")

        # R2 = KP623*c + (x0 - KP222*a - KP900*b) = fma(KP623, c, fnma(KP900, b, t2))
        self.o(f"  R2r={self.fma('KP623','cr',self.fnma('KP900','br','t2r'))};")
        self.o(f"  R2i={self.fma('KP623','ci',self.fnma('KP900','bi','t2i'))};")

        # R3 = KP623*b + (x0 - KP900*a - KP222*c) = fnma(KP222, c, fma(KP623, b, t3))
        self.o(f"  R3r={self.fnma('KP222','cr',self.fma('KP623','br','t3r'))};")
        self.o(f"  R3i={self.fnma('KP222','ci',self.fma('KP623','bi','t3i'))}; }}")

        # y0 = x0 + a + b + c  (overwrite x0 now that R terms are computed)
        y0 = out_names[0]
        self.o(f"{y0}_re={self.add('x0_re',self.add(self.add('ar','br'),'cr'))};")
        self.o(f"{y0}_im={self.add('x0_im',self.add(self.add('ai','bi'),'ci'))};")

        # Spill R1,R2,R3 -> slots 3,4,5; a,b,c,x0 are now dead
        self.c("Spill R1, R2, R3  (a,b,c,x0 dead after this)")
        if self.isa.name == 'scalar':
            self.o(f"spill_re[3] = R1r; spill_im[3] = R1i;")
            self.o(f"spill_re[4] = R2r; spill_im[4] = R2i;")
            self.o(f"spill_re[5] = R3r; spill_im[5] = R3i;")
        else:
            self.o(f"{self.isa.p}_store_pd(&spill_re[3*{sm}], R1r);")
            self.o(f"{self.isa.p}_store_pd(&spill_im[3*{sm}], R1i);")
            self.o(f"{self.isa.p}_store_pd(&spill_re[4*{sm}], R2r);")
            self.o(f"{self.isa.p}_store_pd(&spill_im[4*{sm}], R2i);")
            self.o(f"{self.isa.p}_store_pd(&spill_re[5*{sm}], R3r);")
            self.o(f"{self.isa.p}_store_pd(&spill_im[5*{sm}], R3i);")
        self.spill_c += 3

        # ---- Phase 3: Reload diffs, compute S terms, combine + store ----
        self.b()
        self.c("Phase 3 — sine S terms + combine")

        # Reload diffs
        if self.isa.name == 'scalar':
            self.o(f"dr = spill_re[0]; di = spill_im[0];")
            self.o(f"er = spill_re[1]; ei = spill_im[1];")
            self.o(f"fr = spill_re[2]; fi = spill_im[2];")
        else:
            self.o(f"dr = {self.isa.p}_load_pd(&spill_re[0*{sm}]);")
            self.o(f"di = {self.isa.p}_load_pd(&spill_im[0*{sm}]);")
            self.o(f"er = {self.isa.p}_load_pd(&spill_re[1*{sm}]);")
            self.o(f"ei = {self.isa.p}_load_pd(&spill_im[1*{sm}]);")
            self.o(f"fr = {self.isa.p}_load_pd(&spill_re[2*{sm}]);")
            self.o(f"fi = {self.isa.p}_load_pd(&spill_im[2*{sm}]);")
        self.reload_c += 3

        # Pair k=1 (y1, y6):
        # S1_re = KP974*d_im + KP781*e_im + KP433*f_im
        # S1_pos = KP974*d_re + KP781*e_re + KP433*f_re
        # fwd: y1 = R1 + (S1_re, -S1_pos);  y6 = R1 - (S1_re, -S1_pos)
        # i.e. y1_re = R1_re + S1_re;  y1_im = R1_im - S1_pos
        #      y6_re = R1_re - S1_re;  y6_im = R1_im + S1_pos
        # bwd: swap roles -> y1_re = R1_re - S1_re; y1_im = R1_im + S1_pos
        #                    y6_re = R1_re + S1_re; y6_im = R1_im - S1_pos
        self.b()
        self.c("Pair k=1 -> y1, y6")
        if self.isa.name == 'scalar':
            self.o(f"R1r = spill_re[3]; R1i = spill_im[3];")
        else:
            self.o(f"R1r = {self.isa.p}_load_pd(&spill_re[3*{sm}]);")
            self.o(f"R1i = {self.isa.p}_load_pd(&spill_im[3*{sm}]);")
        self.reload_c += 1
        # S1_re = KP974*d_im + KP781*e_im + KP433*f_im  (imag inputs -> real output cross)
        # S1_pos = KP974*d_re + KP781*e_re + KP433*f_re
        # S1 = KP781*d + KP974*e + KP433*f  (sin(2pi/7)*d + sin(4pi/7)*e + sin(6pi/7)*f)
        self.o(f"{{ {T} S1re={self.fma('KP433','fi',self.fma('KP974','ei',self.mul('KP781','di')))};")
        self.o(f"  {T} S1pos={self.fma('KP433','fr',self.fma('KP974','er',self.mul('KP781','dr')))};")
        y1, y6 = out_names[1], out_names[6]
        if fwd:
            self.o(f"  {y1}_re={self.add('R1r','S1re')}; {y1}_im={self.sub('R1i','S1pos')};")
            self.o(f"  {y6}_re={self.sub('R1r','S1re')}; {y6}_im={self.add('R1i','S1pos')}; }}")
        else:
            self.o(f"  {y6}_re={self.add('R1r','S1re')}; {y6}_im={self.sub('R1i','S1pos')};")
            self.o(f"  {y1}_re={self.sub('R1r','S1re')}; {y1}_im={self.add('R1i','S1pos')}; }}")

        # Pair k=2 (y2, y5):
        # S2_re = KP781*d_im - KP433*e_im - KP974*f_im
        # S2_pos = KP781*d_re - KP433*e_re - KP974*f_re
        self.b()
        self.c("Pair k=2 -> y2, y5")
        if self.isa.name == 'scalar':
            self.o(f"R2r = spill_re[4]; R2i = spill_im[4];")
        else:
            self.o(f"R2r = {self.isa.p}_load_pd(&spill_re[4*{sm}]);")
            self.o(f"R2i = {self.isa.p}_load_pd(&spill_im[4*{sm}]);")
        self.reload_c += 1
        # S2 = KP974*d - KP433*e - KP781*f  (sin(4pi/7)*d - sin(6pi/7)*e - sin(2pi/7)*f)
        self.o(f"{{ {T} S2re={self.fms('KP974','di',self.fma('KP433','ei',self.mul('KP781','fi')))};")
        self.o(f"  {T} S2pos={self.fms('KP974','dr',self.fma('KP433','er',self.mul('KP781','fr')))};")
        y2, y5 = out_names[2], out_names[5]
        if fwd:
            self.o(f"  {y2}_re={self.add('R2r','S2re')}; {y2}_im={self.sub('R2i','S2pos')};")
            self.o(f"  {y5}_re={self.sub('R2r','S2re')}; {y5}_im={self.add('R2i','S2pos')}; }}")
        else:
            self.o(f"  {y5}_re={self.add('R2r','S2re')}; {y5}_im={self.sub('R2i','S2pos')};")
            self.o(f"  {y2}_re={self.sub('R2r','S2re')}; {y2}_im={self.add('R2i','S2pos')}; }}")

        # Pair k=3 (y3, y4):
        # S3_re = KP433*d_im - KP974*e_im + KP781*f_im
        # S3_pos = KP433*d_re - KP974*e_re + KP781*f_re
        self.b()
        self.c("Pair k=3 -> y3, y4")
        if self.isa.name == 'scalar':
            self.o(f"R3r = spill_re[5]; R3i = spill_im[5];")
        else:
            self.o(f"R3r = {self.isa.p}_load_pd(&spill_re[5*{sm}]);")
            self.o(f"R3i = {self.isa.p}_load_pd(&spill_im[5*{sm}]);")
        self.reload_c += 1
        # S3 = KP433*d - KP781*e + KP974*f  (sin(6pi/7)*d - sin(2pi/7)*e + sin(4pi/7)*f)
        self.o(f"{{ {T} S3re={self.fma('KP974','fi',self.fnma('KP781','ei',self.mul('KP433','di')))};")
        self.o(f"  {T} S3pos={self.fma('KP974','fr',self.fnma('KP781','er',self.mul('KP433','dr')))};")
        y3, y4 = out_names[3], out_names[4]
        if fwd:
            self.o(f"  {y3}_re={self.add('R3r','S3re')}; {y3}_im={self.sub('R3i','S3pos')};")
            self.o(f"  {y4}_re={self.sub('R3r','S3re')}; {y4}_im={self.add('R3i','S3pos')}; }}")
        else:
            self.o(f"  {y4}_re={self.add('R3r','S3re')}; {y4}_im={self.sub('R3i','S3pos')};")
            self.o(f"  {y3}_re={self.sub('R3r','S3re')}; {y3}_im={self.add('R3i','S3pos')}; }}")

    def _emit_radix7_butterfly_nospill(self, d, out_names):
        """AVX-512 path: single-pass, no spills. Peak 24 of 32 ZMM registers.
        All sums a,b,c and diffs d,e,f stay live throughout."""
        fwd = (d == 'fwd')
        T = self.isa.T

        self.c(f"DFT-7 butterfly [{d}] (AVX-512 no-spill, peak 26/32 ZMM)")

        # Symmetric / antisymmetric pairs — all live simultaneously
        self.c("Sums and diffs (all live, no spill)")
        self.o(f"{T} ar={self.add('x1_re','x6_re')}, ai={self.add('x1_im','x6_im')};")
        self.o(f"{T} dr={self.sub('x1_re','x6_re')}, di={self.sub('x1_im','x6_im')};")
        self.o(f"{T} br={self.add('x2_re','x5_re')}, bi={self.add('x2_im','x5_im')};")
        self.o(f"{T} er={self.sub('x2_re','x5_re')}, ei={self.sub('x2_im','x5_im')};")
        self.o(f"{T} cr={self.add('x3_re','x4_re')}, ci={self.add('x3_im','x4_im')};")
        self.o(f"{T} fr={self.sub('x3_re','x4_re')}, fi={self.sub('x3_im','x4_im')};")

        # R-term first level: 3 independent fnma from x0 (depth reduction).
        # These issue on all FMA ports simultaneously, reducing critical path
        # from 3 to 2 FMA latencies per R term.
        self.b()
        self.c("R-term first level (3 independent seeds from x0)")
        self.o(f"{T} t1r={self.fnma('KP900','cr','x0_re')}, t1i={self.fnma('KP900','ci','x0_im')};")
        self.o(f"{T} t2r={self.fnma('KP222','ar','x0_re')}, t2i={self.fnma('KP222','ai','x0_im')};")
        self.o(f"{T} t3r={self.fnma('KP900','ar','x0_re')}, t3i={self.fnma('KP900','ai','x0_im')};")

        # Pair k=1 (y1, y6): R1 from t1, S1 from diffs
        self.b()
        self.c("Pair k=1 -> y1, y6")
        self.o(f"{{ {T} Rr={self.fma('KP623','ar',self.fnma('KP222','br','t1r'))};")
        self.o(f"  {T} Ri={self.fma('KP623','ai',self.fnma('KP222','bi','t1i'))};")
        self.o(f"  {T} Sre={self.fma('KP433','fi',self.fma('KP974','ei',self.mul('KP781','di')))};")
        self.o(f"  {T} Spos={self.fma('KP433','fr',self.fma('KP974','er',self.mul('KP781','dr')))};")
        y1, y6 = out_names[1], out_names[6]
        if fwd:
            self.o(f"  {y1}_re={self.add('Rr','Sre')}; {y1}_im={self.sub('Ri','Spos')};")
            self.o(f"  {y6}_re={self.sub('Rr','Sre')}; {y6}_im={self.add('Ri','Spos')}; }}")
        else:
            self.o(f"  {y6}_re={self.add('Rr','Sre')}; {y6}_im={self.sub('Ri','Spos')};")
            self.o(f"  {y1}_re={self.sub('Rr','Sre')}; {y1}_im={self.add('Ri','Spos')}; }}")

        # Pair k=2 (y2, y5): R2 from t2
        self.b()
        self.c("Pair k=2 -> y2, y5")
        self.o(f"{{ {T} Rr={self.fma('KP623','cr',self.fnma('KP900','br','t2r'))};")
        self.o(f"  {T} Ri={self.fma('KP623','ci',self.fnma('KP900','bi','t2i'))};")
        self.o(f"  {T} Sre={self.fms('KP974','di',self.fma('KP433','ei',self.mul('KP781','fi')))};")
        self.o(f"  {T} Spos={self.fms('KP974','dr',self.fma('KP433','er',self.mul('KP781','fr')))};")
        y2, y5 = out_names[2], out_names[5]
        if fwd:
            self.o(f"  {y2}_re={self.add('Rr','Sre')}; {y2}_im={self.sub('Ri','Spos')};")
            self.o(f"  {y5}_re={self.sub('Rr','Sre')}; {y5}_im={self.add('Ri','Spos')}; }}")
        else:
            self.o(f"  {y5}_re={self.add('Rr','Sre')}; {y5}_im={self.sub('Ri','Spos')};")
            self.o(f"  {y2}_re={self.sub('Rr','Sre')}; {y2}_im={self.add('Ri','Spos')}; }}")

        # Pair k=3 (y3, y4): R3 from t3
        self.b()
        self.c("Pair k=3 -> y3, y4")
        self.o(f"{{ {T} Rr={self.fnma('KP222','cr',self.fma('KP623','br','t3r'))};")
        self.o(f"  {T} Ri={self.fnma('KP222','ci',self.fma('KP623','bi','t3i'))};")
        self.o(f"  {T} Sre={self.fma('KP974','fi',self.fnma('KP781','ei',self.mul('KP433','di')))};")
        self.o(f"  {T} Spos={self.fma('KP974','fr',self.fnma('KP781','er',self.mul('KP433','dr')))};")
        y3, y4 = out_names[3], out_names[4]
        if fwd:
            self.o(f"  {y3}_re={self.add('Rr','Sre')}; {y3}_im={self.sub('Ri','Spos')};")
            self.o(f"  {y4}_re={self.sub('Rr','Sre')}; {y4}_im={self.add('Ri','Spos')}; }}")
        else:
            self.o(f"  {y4}_re={self.add('Rr','Sre')}; {y4}_im={self.sub('Ri','Spos')};")
            self.o(f"  {y3}_re={self.sub('Rr','Sre')}; {y3}_im={self.add('Ri','Spos')}; }}")

        # y0 = x0 + a + b + c — LAST, after all pairs that read x0
        self.b()
        y0 = out_names[0]
        self.o(f"{y0}_re={self.add('x0_re',self.add(self.add('ar','br'),'cr'))};")
        self.o(f"{y0}_im={self.add('x0_im',self.add(self.add('ai','bi'),'ci'))};")



# ================================================================
# HELPERS: constants
# ================================================================

def emit_dft7_constants(em):
    """Emit the 6 DFT-7 constants + sign_flip as SIMD broadcasts or scalars."""
    T = em.isa.T
    if em.isa.name == 'scalar':
        em.o(f"const double KP623 = {KP623_val:.45f};")
        em.o(f"const double KP222 = {KP222_val:.45f};")
        em.o(f"const double KP900 = {KP900_val:.45f};")
        em.o(f"const double KP974 = {KP974_val:.45f};")
        em.o(f"const double KP781 = {KP781_val:.45f};")
        em.o(f"const double KP433 = {KP433_val:.45f};")
    else:
        set1 = f"{em.isa.p}_set1_pd"
        em.o(f"const {T} sign_flip = {set1}(-0.0);")
        em.o(f"const {T} KP623 = {set1}({KP623_val:.45f});")
        em.o(f"const {T} KP222 = {set1}({KP222_val:.45f});")
        em.o(f"const {T} KP900 = {set1}({KP900_val:.45f});")
        em.o(f"const {T} KP974 = {set1}({KP974_val:.45f});")
        em.o(f"const {T} KP781 = {set1}({KP781_val:.45f});")
        em.o(f"const {T} KP433 = {set1}({KP433_val:.45f});")
        em.o(f"(void)sign_flip;  /* reserved for neg() */")
    em.b()


def emit_dft7_constants_raw(lines, isa, indent=1):
    """Append DFT-7 constant declarations + sign_flip to a line list."""
    pad = "    " * indent
    T = isa.T
    if isa.name == 'scalar':
        lines.append(f"{pad}const double KP623 = {KP623_val:.45f};")
        lines.append(f"{pad}const double KP222 = {KP222_val:.45f};")
        lines.append(f"{pad}const double KP900 = {KP900_val:.45f};")
        lines.append(f"{pad}const double KP974 = {KP974_val:.45f};")
        lines.append(f"{pad}const double KP781 = {KP781_val:.45f};")
        lines.append(f"{pad}const double KP433 = {KP433_val:.45f};")
    else:
        set1 = f"{isa.p}_set1_pd"
        lines.append(f"{pad}const {T} sign_flip = {set1}(-0.0);")
        lines.append(f"{pad}const {T} KP623 = {set1}({KP623_val:.45f});")
        lines.append(f"{pad}const {T} KP222 = {set1}({KP222_val:.45f});")
        lines.append(f"{pad}const {T} KP900 = {set1}({KP900_val:.45f});")
        lines.append(f"{pad}const {T} KP974 = {set1}({KP974_val:.45f});")
        lines.append(f"{pad}const {T} KP781 = {set1}({KP781_val:.45f});")
        lines.append(f"{pad}const {T} KP433 = {set1}({KP433_val:.45f});")
        lines.append(f"{pad}(void)sign_flip;")
    lines.append(f"")


def emit_spill_decl(em):
    """Emit the aligned spill buffer declaration (6 slots)."""
    sm = em.isa.sm
    T = em.isa.T
    N_SPILL = 6
    if em.isa.name == 'scalar':
        em.o(f"double spill_re[{N_SPILL}], spill_im[{N_SPILL}];")
    else:
        em.o(f"{em.isa.align} {em.isa.T} spill_re_buf[{N_SPILL}];")
        em.o(f"{em.isa.align} {em.isa.T} spill_im_buf[{N_SPILL}];")
        em.o(f"double * __restrict__ spill_re = (double*)spill_re_buf;")
        em.o(f"double * __restrict__ spill_im = (double*)spill_im_buf;")


def emit_spill_decl_raw(lines, isa, indent=1):
    """Append spill buffer declarations to a line list."""
    pad = "    " * indent
    N_SPILL = 6
    if isa.name == 'scalar':
        lines.append(f"{pad}double spill_re[{N_SPILL}], spill_im[{N_SPILL}];")
    else:
        lines.append(f"{pad}{isa.align} {isa.T} spill_re_buf[{N_SPILL}];")
        lines.append(f"{pad}{isa.align} {isa.T} spill_im_buf[{N_SPILL}];")
        lines.append(f"{pad}double * __restrict__ spill_re = (double*)spill_re_buf;")
        lines.append(f"{pad}double * __restrict__ spill_im = (double*)spill_im_buf;")


# ================================================================
# KERNEL BODY EMITTERS
# ================================================================

def emit_kernel_body(em, d, variant):
    """Emit the inner loop body for notw, dit_tw, dif_tw."""
    T = em.isa.T

    # Load inputs
    for n in range(R):
        em.emit_load(f"x{n}", n)

    # DIT: twiddle inputs 1..6 before butterfly
    if variant == 'dit_tw':
        for n in range(1, R):
            em.emit_ext_tw(f"x{n}", n - 1, d)
    elif variant == 'dit_tw_scalar':
        for n in range(1, R):
            em.emit_ext_tw_scalar(f"x{n}", n - 1, d)

    em.b()
    em.emit_radix7_butterfly(d)
    em.b()

    # DIF: twiddle outputs 1..6 after butterfly
    if variant == 'dif_tw':
        for m in range(1, R):
            em.emit_ext_tw(f"x{m}", m - 1, d)
    elif variant == 'dif_tw_scalar':
        for m in range(1, R):
            em.emit_ext_tw_scalar(f"x{m}", m - 1, d)

    # Store outputs
    for m in range(R):
        em.emit_store(f"x{m}", m)


def emit_kernel_body_log3(em, d, variant):
    """Emit the log3 variant: derive w2..w6 from w1."""
    T = em.isa.T
    is_dit = variant == 'dit_tw_log3'

    # Load base twiddle w1
    em.c("Load base twiddle w1, derive w2..w6 (5 cmuls)")
    tb, tbi = em._tw_buf(), em._tw_buf_im()
    ta = em._tw_addr(0)
    if em.isa.name == 'scalar':
        em.o(f"const double w1r = {tb}[{ta}], w1i = {tbi}[{ta}];")
    else:
        em.o(f"const {T} w1r = LD(&{tb}[{ta}]), w1i = LD(&{tbi}[{ta}]);")
    em.n_load += 2

    # w2 = w1 * w1
    em.o(f"{T} w2r, w2i;")
    em.emit_cmul("w2r", "w2i", "w1r", "w1i", "w1r", "w1i", 'fwd')

    # w3 = w1 * w2
    em.o(f"{T} w3r, w3i;")
    em.emit_cmul("w3r", "w3i", "w1r", "w1i", "w2r", "w2i", 'fwd')

    # w4 = w2 * w2
    em.o(f"{T} w4r, w4i;")
    em.emit_cmul("w4r", "w4i", "w2r", "w2i", "w2r", "w2i", 'fwd')

    # w5 = w2 * w3
    em.o(f"{T} w5r, w5i;")
    em.emit_cmul("w5r", "w5i", "w2r", "w2i", "w3r", "w3i", 'fwd')

    # w6 = w3 * w3
    em.o(f"{T} w6r, w6i;")
    em.emit_cmul("w6r", "w6i", "w3r", "w3i", "w3r", "w3i", 'fwd')
    em.b()

    # Load inputs
    for n in range(R):
        em.emit_load(f"x{n}", n)

    # DIT: apply twiddles to x1..x6
    if is_dit:
        for n in range(1, R):
            wname = f"w{n}"
            em.emit_cmul_inplace(f"x{n}", f"{wname}r", f"{wname}i", d)
    em.b()

    em.emit_radix7_butterfly(d)
    em.b()

    # DIF: apply twiddles to outputs x1..x6
    if not is_dit:
        for m in range(1, R):
            wname = f"w{m}"
            em.emit_cmul_inplace(f"x{m}", f"{wname}r", f"{wname}i", d)

    for m in range(R):
        em.emit_store(f"x{m}", m)


# ================================================================
# FILE EMITTER -- standard variants (notw, dit_tw, dif_tw, log3)
# ================================================================

def emit_file(isa, variant):
    """Emit complete header for one ISA and one variant."""
    em = Emitter(isa)
    T = isa.T
    is_log3 = variant.endswith('_log3')

    func_base = ''
    tw_params = None
    if variant == 'notw':
        func_base = 'radix7_n1_dit_kernel'
    elif variant == 'dit_tw':
        func_base = 'radix7_tw_flat_dit_kernel'
        tw_params = 'flat'
    elif variant == 'dit_tw_scalar':
        func_base = 'radix7_tw_flat_dit_kernel'
        tw_params = 'flat'
    elif variant == 'dif_tw':
        func_base = 'radix7_tw_flat_dif_kernel'
        tw_params = 'flat'
    elif variant == 'dif_tw_scalar':
        func_base = 'radix7_tw_flat_dif_kernel'
        tw_params = 'flat'
    elif variant == 'dit_tw_log3':
        func_base = 'radix7_tw_log3_dit_kernel'
        tw_params = 'flat'
    elif variant == 'dif_tw_log3':
        func_base = 'radix7_tw_log3_dif_kernel'
        tw_params = 'flat'

    vname = {
        'notw': 'N1 (no twiddle)', 'dit_tw': 'DIT twiddled (flat)',
        'dif_tw': 'DIF twiddled (flat)',
        'dit_tw_log3': 'DIT twiddled (log3 derived)',
        'dif_tw_log3': 'DIF twiddled (log3 derived)',
    }[variant]
    guard = f"FFT_RADIX7_{isa.name.upper()}_{variant.upper()}_H"

    em.L.append(f"/**")
    em.L.append(f" * @file fft_radix7_{isa.name}_{variant}.h")
    em.L.append(f" * @brief DFT-7 {isa.name.upper()} {vname}")
    em.L.append(f" *")
    em.L.append(f" * Monolithic DFT-7 butterfly, 6 constants. AVX2: peak 16 YMMs + spill. AVX-512: 24/32 ZMMs, no spill.")
    em.L.append(f" * k-step={isa.k_step}")
    em.L.append(f" * Generated by gen_radix7.py")
    em.L.append(f" */")
    em.L.append(f"")
    em.L.append(f"#ifndef {guard}")
    em.L.append(f"#define {guard}")
    em.L.append(f"")

    if isa.name != 'scalar':
        em.L.append(f"#include <immintrin.h>")
        em.L.append(f"")

    # LD/ST macros
    lp = isa.ld_prefix
    if isa.name == 'scalar':
        em.L.append(f"#ifndef {lp}_LD")
        em.L.append(f"#define {lp}_LD(p) (*(p))")
        em.L.append(f"#define {lp}_ST(p,v) (*(p)=(v))")
        em.L.append(f"#endif")
    elif isa.name == 'avx2':
        em.L.append(f"#ifndef {lp}_LD")
        em.L.append(f"#define {lp}_LD(p) _mm256_load_pd(p)")
        em.L.append(f"#define {lp}_ST(p,v) _mm256_store_pd((p),(v))")
        em.L.append(f"#endif")
    else:
        em.L.append(f"#ifndef {lp}_LD")
        em.L.append(f"#define {lp}_LD(p) _mm512_load_pd(p)")
        em.L.append(f"#define {lp}_ST(p,v) _mm512_store_pd((p),(v))")
        em.L.append(f"#endif")
    em.L.append(f"#define LD {lp}_LD")
    em.L.append(f"#define ST {lp}_ST")
    em.L.append(f"")

    stats = {}
    for d in ['fwd', 'bwd']:
        em.reset()
        em.addr_mode = 'K'

        if isa.target:
            em.L.append(f"static {isa.target} void")
        else:
            em.L.append(f"static void")
        em.L.append(f"{func_base}_{d}_{isa.name}(")
        em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
        em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
        if tw_params:
            em.L.append(f"    const double * __restrict__ tw_re, const double * __restrict__ tw_im,")
        em.L.append(f"    size_t K)")
        em.L.append(f"{{")
        em.ind = 1

        # Constants
        emit_dft7_constants(em)

        # Working registers
        em.o(f"{T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
        em.o(f"{T} x4_re,x4_im,x5_re,x5_im,x6_re,x6_im;")

        # Spill buffer + working temps (AVX2/scalar only, AVX-512 uses no-spill path)
        if isa.name != 'avx512':
            emit_spill_decl(em)
            em.b()
            em.o(f"{T} ar,ai,br,bi,cr,ci,dr,di,er,ei,fr,fi;")
            em.o(f"{T} R1r,R1i,R2r,R2i,R3r,R3i;")
            em.b()

        # K loop
        if isa.name == 'scalar':
            em.o(f"for (size_t k = 0; k < K; k++) {{")
        else:
            em.o(f"for (size_t k = 0; k < K; k += {isa.k_step}) {{")
        em.ind += 1
        if is_log3:
            emit_kernel_body_log3(em, d, variant)
        else:
            emit_kernel_body(em, d, variant)
        em.ind -= 1
        em.o("}")
        em.L.append("}")
        em.L.append("")

        stats[d] = em.get_stats()

    # U=2 pipelining (AVX-512 only, non-log3)
    # 2×14 working + 6 shared constants = 34 ZMM; with shared spill buffer this fits.
    if isa.name == 'avx512' and not is_log3 and func_base is not None:
        VL = isa.k_step
        u2_name = func_base + '_u2'
        has_tw = tw_params is not None

        for d in ['fwd', 'bwd']:
            em_u1 = Emitter(isa)
            em_u1.addr_mode = 'K'
            em_u1.ind = 2
            emit_kernel_body(em_u1, d, variant)
            body_a = '\n'.join(em_u1.L)
            body_b = body_a.replace('+k]', f'+k+{VL}]').replace('+k)', f'+k+{VL})')

            if isa.target:
                em.L.append(f"static {isa.target} void")
            else:
                em.L.append(f"static void")
            em.L.append(f"{u2_name}_{d}_{isa.name}(")
            em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
            em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
            if has_tw:
                em.L.append(f"    const double * __restrict__ tw_re, const double * __restrict__ tw_im,")
            em.L.append(f"    size_t K)")
            em.L.append(f"{{")
            em.L.append(f"    /* U=2: two independent k-groups per iteration */")

            emit_dft7_constants_raw(em.L, isa, indent=1)
            # AVX-512 U=2: no spill buffer needed (no-spill butterfly path)
            xdecl = (f"        {T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im,"
                     f"x4_re,x4_im,x5_re,x5_im,x6_re,x6_im;")
            em.L.append(f"    for (size_t k = 0; k < K; k += {VL*2}) {{")
            em.L.append(f"        {{ /* Pipeline A: k */")
            em.L.append(xdecl)
            em.L.append(body_a)
            em.L.append(f"        }}")
            em.L.append(f"        {{ /* Pipeline B: k+{VL} */")
            em.L.append(xdecl)
            em.L.append(body_b)
            em.L.append(f"        }}")
            em.L.append(f"    }}")
            em.L.append(f"}}")
            em.L.append(f"")

    em.L.append(f"#undef LD")
    em.L.append(f"#undef ST")
    em.L.append(f"")
    em.L.append(f"#endif /* {guard} */")

    insert_stats_into_header(em.L, stats)
    return em.L, stats


# ================================================================
# FILE EMITTER -- CT variants (ct_n1, ct_t1_dit, ct_t1_dif)
# ================================================================

def emit_file_ct(isa, ct_variant):
    """Emit FFTW-style n1 or t1 codelet."""
    em = Emitter(isa)
    T = isa.T

    is_n1 = ct_variant == 'ct_n1'
    is_n1_scaled = ct_variant == 'ct_n1_scaled'
    is_t1_dit = ct_variant == 'ct_t1_dit'
    is_t1s_dit = ct_variant == 'ct_t1s_dit'
    is_t1_dit_log3 = ct_variant == 'ct_t1_dit_log3'
    is_t1_dif = ct_variant == 'ct_t1_dif'
    is_t1_oop_dit = ct_variant == 'ct_t1_oop_dit'
    em.addr_mode = 'n1' if (is_n1 or is_n1_scaled) else ('t1_oop' if is_t1_oop_dit else 't1')

    if is_n1:
        func_base = "radix7_n1"
        vname = "n1 (separate is/os)"
    elif is_n1_scaled:
        func_base = "radix7_n1_scaled"
        vname = "n1_scaled (separate is/os, output *= scale)"
    elif is_t1_oop_dit:
        func_base = "radix7_t1_oop_dit"
        vname = "t1_oop DIT (out-of-place, separate is/os, with twiddle)"
    elif is_t1_dif:
        func_base = "radix7_t1_dif"
        vname = "t1 DIF (in-place twiddle)"
    elif is_t1s_dit:
        func_base = "radix7_t1s_dit"
        vname = "t1s DIT (in-place, scalar broadcast twiddle)"
    elif is_t1_dit_log3:
        func_base = "radix7_t1_dit_log3"
        vname = "t1 DIT log3 (in-place, derived twiddles)"
    else:
        func_base = "radix7_t1_dit"
        vname = "t1 DIT (in-place twiddle)"

    guard = f"FFT_RADIX7_{isa.name.upper()}_CT_{ct_variant.upper()}_H"

    em.L.append(f"/**")
    em.L.append(f" * @file fft_radix7_{isa.name}_{ct_variant}.h")
    em.L.append(f" * @brief DFT-7 {isa.name.upper()} {vname}")
    em.L.append(f" *")
    em.L.append(f" * FFTW-style codelet for recursive CT executor.")
    em.L.append(f" * Generated by gen_radix7.py --variant {ct_variant}")
    em.L.append(f" */")
    em.L.append(f"")
    em.L.append(f"#ifndef {guard}")
    em.L.append(f"#define {guard}")
    em.L.append(f"")

    if isa.name != 'scalar':
        em.L.append(f"#include <immintrin.h>")
        em.L.append(f"")

    # LD/ST macros (unaligned for CT variants)
    lp = isa.ld_prefix + "_CT"
    if isa.name == 'scalar':
        em.L.append(f"#ifndef {lp}_LD")
        em.L.append(f"#define {lp}_LD(p) (*(p))")
        em.L.append(f"#define {lp}_ST(p,v) (*(p)=(v))")
        em.L.append(f"#endif")
    elif isa.name == 'avx2':
        em.L.append(f"#ifndef {lp}_LD")
        em.L.append(f"#define {lp}_LD(p) _mm256_loadu_pd(p)")
        em.L.append(f"#define {lp}_ST(p,v) _mm256_storeu_pd((p),(v))")
        em.L.append(f"#endif")
    else:
        em.L.append(f"#ifndef {lp}_LD")
        em.L.append(f"#define {lp}_LD(p) _mm512_loadu_pd(p)")
        em.L.append(f"#define {lp}_ST(p,v) _mm512_storeu_pd((p),(v))")
        em.L.append(f"#endif")
    em.L.append(f"#define LD {lp}_LD")
    em.L.append(f"#define ST {lp}_ST")
    em.L.append(f"")

    for d in ['fwd', 'bwd']:
        em.reset()
        em.addr_mode = 'n1' if (is_n1 or is_n1_scaled) else ('t1_oop' if is_t1_oop_dit else 't1')
        em.store_scale = is_n1_scaled

        if isa.target:
            em.L.append(f"static {isa.target} void")
        else:
            em.L.append(f"static void")

        if is_n1_scaled:
            em.L.append(f"{func_base}_{d}_{isa.name}(")
            em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
            em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
            em.L.append(f"    size_t is, size_t os, size_t vl, double scale)")
        elif is_n1:
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
        else:  # t1
            em.L.append(f"{func_base}_{d}_{isa.name}(")
            em.L.append(f"    double * __restrict__ rio_re, double * __restrict__ rio_im,")
            em.L.append(f"    const double * __restrict__ W_re, const double * __restrict__ W_im,")
            if isa.name == 'scalar':
                em.L.append(f"    size_t ios, size_t mb, size_t me, size_t ms)")
            else:
                em.L.append(f"    size_t ios, size_t me)")

        em.L.append(f"{{")
        em.ind = 1

        emit_dft7_constants(em)

        em.o(f"{T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
        em.o(f"{T} x4_re,x4_im,x5_re,x5_im,x6_re,x6_im;")
        if isa.name != 'avx512':
            emit_spill_decl(em)
            em.o(f"{T} ar,ai,br,bi,cr,ci,dr,di,er,ei,fr,fi;")
            em.o(f"{T} R1r,R1i,R2r,R2i,R3r,R3i;")
        em.b()

        # Hoist twiddle broadcasts before the loop (t1s only)
        if is_t1s_dit:
            em.tw_hoisted = True
            em.emit_hoist_all_tw_scalars(R)
            em.b()

        # Broadcast scale factor before the loop (n1_scaled only)
        if is_n1_scaled and isa.name != 'scalar':
            set1 = f"{isa.p}_set1_pd"
            em.o(f"const {T} vscale = {set1}(scale);")
            em.b()

        # Loop
        if is_n1 or is_n1_scaled:
            if isa.name == 'scalar':
                em.o(f"for (size_t k = 0; k < vl; k++) {{")
            else:
                em.o(f"for (size_t k = 0; k < vl; k += {isa.k_step}) {{")
        else:  # t1, t1_oop
            if isa.name == 'scalar':
                em.o(f"for (size_t m = mb; m < me; m++) {{")
            else:
                em.o(f"for (size_t m = 0; m < me; m += {isa.k_step}) {{")

        em.ind += 1
        if is_t1s_dit:
            emit_kernel_body(em, d, 'dit_tw_scalar')
        elif is_t1_dit_log3:
            emit_kernel_body_log3(em, d, 'dit_tw_log3')
        else:
            kernel_variant = 'notw' if (is_n1 or is_n1_scaled) else ('dif_tw' if is_t1_dif else 'dit_tw')
            if is_t1_oop_dit:
                em.addr_mode = 't1_oop'
                kernel_variant = 'dit_tw'
            emit_kernel_body(em, d, kernel_variant)
        em.ind -= 1
        em.o("}")
        em.L.append("}")
        em.L.append("")

    # n1_ovs: butterfly with fused SIMD transpose stores
    if is_n1 and isa.name != 'scalar':
        VL = isa.k_step

        for d in ['fwd', 'bwd']:
            em2 = Emitter(isa)
            em2.addr_mode = 'n1_ovs'

            em.L.append(f"")
            if isa.target:
                em.L.append(f"static {isa.target} void")
            else:
                em.L.append(f"static void")
            em.L.append(f"radix7_n1_ovs_{d}_{isa.name}(")
            em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
            em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
            em.L.append(f"    size_t is, size_t os, size_t vl, size_t ovs)")
            em.L.append(f"{{")

            em.L.append(f"    /* n1_ovs: butterfly -> tbuf, then 4x4 transpose (bins 0-3) + scalar scatter (bins 4-6) */")
            em.L.append(f"    {isa.align} double tbuf_re[{R * VL}];")
            em.L.append(f"    {isa.align} double tbuf_im[{R * VL}];")

            emit_dft7_constants_raw(em.L, isa, indent=1)
            if isa.name != 'avx512':
                emit_spill_decl_raw(em.L, isa, indent=1)

            em.L.append(f"    {T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
            em.L.append(f"    {T} x4_re,x4_im,x5_re,x5_im,x6_re,x6_im;")
            if isa.name != 'avx512':
                em.L.append(f"    {T} ar,ai,br,bi,cr,ci,dr,di,er,ei,fr,fi;")
                em.L.append(f"    {T} R1r,R1i,R2r,R2i,R3r,R3i;")
            em.L.append(f"")

            em.L.append(f"    for (size_t k = 0; k < vl; k += {isa.k_step}) {{")

            # Use fresh Emitter for kernel body
            em2.L = []
            em2.ind = 2
            em2.reset()
            em2.addr_mode = 'n1_ovs'
            emit_kernel_body(em2, d, 'notw')
            em.L.extend(em2.L)

            # Transpose bins 0-3 (standard 4x4 AVX2 transpose)
            em.L.append(f"        /* 4x4 transpose: bins 0-3 -> output at stride ovs */")
            for comp, arr in [('re', 'out_re'), ('im', 'out_im')]:
                bname = f"tbuf_{comp}"
                em.L.append(f"        {{ {T} a_=LD(&{bname}[0*{VL}]), b_=LD(&{bname}[1*{VL}]);")
                em.L.append(f"          {T} c_=LD(&{bname}[2*{VL}]), d_=LD(&{bname}[3*{VL}]);")
                if isa.name == 'avx2':
                    em.L.append(f"          {T} lo_ab=_mm256_unpacklo_pd(a_,b_), hi_ab=_mm256_unpackhi_pd(a_,b_);")
                    em.L.append(f"          {T} lo_cd=_mm256_unpacklo_pd(c_,d_), hi_cd=_mm256_unpackhi_pd(c_,d_);")
                    em.L.append(f"          _mm256_storeu_pd(&{arr}[(k+0)*ovs+os*0], _mm256_permute2f128_pd(lo_ab,lo_cd,0x20));")
                    em.L.append(f"          _mm256_storeu_pd(&{arr}[(k+1)*ovs+os*0], _mm256_permute2f128_pd(hi_ab,hi_cd,0x20));")
                    em.L.append(f"          _mm256_storeu_pd(&{arr}[(k+2)*ovs+os*0], _mm256_permute2f128_pd(lo_ab,lo_cd,0x31));")
                    em.L.append(f"          _mm256_storeu_pd(&{arr}[(k+3)*ovs+os*0], _mm256_permute2f128_pd(hi_ab,hi_cd,0x31));")
                else:  # avx512
                    for j in range(isa.k_step):
                        for b in range(4):
                            em.L.append(f"          {arr}[(k+{j})*ovs+os*{b}] = {bname}[{b}*{VL}+{j}];")
                em.L.append(f"        }}")

            # Bins 4,5,6: extract from tbuf or YMM registers (still in tbuf from notw store)
            if isa.name == 'avx2':
                # Bins 4-6: extract from tbuf via YMM load + scalar scatter
                for bn in [4, 5, 6]:
                    em.L.append(f"        /* Bin {bn}: extract from tbuf -> scatter */")
                    for comp, arr in [('re', 'out_re'), ('im', 'out_im')]:
                        bname = f"tbuf_{comp}"
                        em.L.append(f"        {{ __m256d v=LD(&{bname}[{bn}*{VL}]);")
                        em.L.append(f"          __m128d lo=_mm256_castpd256_pd128(v), hi=_mm256_extractf128_pd(v,1);")
                        em.L.append(f"          _mm_storel_pd(&{arr}[(k+0)*ovs+os*{bn}], lo);")
                        em.L.append(f"          _mm_storeh_pd(&{arr}[(k+1)*ovs+os*{bn}], lo);")
                        em.L.append(f"          _mm_storel_pd(&{arr}[(k+2)*ovs+os*{bn}], hi);")
                        em.L.append(f"          _mm_storeh_pd(&{arr}[(k+3)*ovs+os*{bn}], hi); }}")
            else:
                # AVX-512 fallback: scalar scatter from tbuf
                for bn in [4, 5, 6]:
                    em.L.append(f"        /* Bin {bn}: scalar scatter */")
                    for comp, arr in [('re', 'out_re'), ('im', 'out_im')]:
                        bname = f"tbuf_{comp}"
                        for j in range(VL):
                            em.L.append(f"        {arr}[(k+{j})*ovs+os*{bn}] = {bname}[{bn}*{VL}+{j}];")

            em.L.append(f"    }}")
            em.L.append(f"}}")

    em.L.append(f"")
    em.L.append(f"#undef LD")
    em.L.append(f"#undef ST")
    em.L.append(f"")
    em.L.append(f"#endif /* {guard} */")

    return em.L


# ================================================================
# SV CODELET GENERATION — text transform from K-loop output
# ================================================================

def _t2_to_sv(body):
    """Transform a t2 codelet body to sv: strip k-loop, K->vs in addressing."""
    lines = body.split('\n')
    out = []
    in_loop = False
    depth = 0
    for line in lines:
        stripped = line.strip()
        if not in_loop and 'for (size_t k' in stripped and 'k < K' in stripped:
            in_loop = True
            depth = 1
            continue
        if in_loop:
            for ch in stripped:
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
            if depth <= 0:
                in_loop = False
                if stripped == '}':
                    continue
        line = re.sub(r'(\d+)\*K\+k', r'\1*vs', line)
        line = re.sub(r'\[k\]', '[0]', line)
        if line.startswith('        '):
            line = line[4:]
        out.append(line)
    return '\n'.join(out)


def emit_sv_variants(t2_lines, isa, variant):
    """Extract t2 functions from generated lines, emit sv versions."""
    if isa.name == 'scalar':
        return []
    if variant.endswith('_log3'):
        return []

    text = '\n'.join(t2_lines)

    if variant == 'notw':
        t2_pattern = 'radix7_n1_dit_kernel'
        sv_name = 'radix7_n1sv_kernel'
    elif variant == 'dit_tw':
        t2_pattern = 'radix7_tw_flat_dit_kernel'
        sv_name = 'radix7_t1sv_dit_kernel'
    elif variant == 'dit_tw_scalar':
        t2_pattern = 'radix7_tw_flat_dit_kernel'
        sv_name = 'radix7_t1sv_dit_kernel'
    elif variant == 'dif_tw':
        t2_pattern = 'radix7_tw_flat_dif_kernel'
        sv_name = 'radix7_t1sv_dif_kernel'
    elif variant == 'dif_tw_scalar':
        t2_pattern = 'radix7_tw_flat_dif_kernel'
        sv_name = 'radix7_t1sv_dif_kernel'
    else:
        return []

    out = []
    out.append('')
    out.append(f'/* === sv codelets: no loop, elements at stride vs === */')
    out.append(f'/* Executor calls K/{isa.k_step} times, advancing base pointers by {isa.k_step}. */')

    for d in ['fwd', 'bwd']:
        func_name = f'{t2_pattern}_{d}_{isa.name}'
        sv_func_name = f'{sv_name}_{d}_{isa.name}'

        func_start = text.find(f'{func_name}(')
        if func_start < 0:
            continue

        static_start = text.rfind('static', 0, func_start)
        if static_start < 0:
            continue

        brace_start = text.find('{', func_start)
        if brace_start < 0:
            continue

        depth = 0
        pos = brace_start
        while pos < len(text):
            if text[pos] == '{':
                depth += 1
            elif text[pos] == '}':
                depth -= 1
                if depth == 0:
                    break
            pos += 1

        func_body = text[brace_start + 1:pos]
        sv_body = _t2_to_sv(func_body)

        sig = text[static_start:brace_start]
        sig = sig.replace(func_name, sv_func_name)
        sig = sig.replace('size_t K)', 'size_t vs)')

        out.append(sig + '{')
        out.append(sv_body)
        out.append('}')
        out.append('')

    return out


def insert_stats_into_header(lines, stats):
    table = [" *", " * -- Operation counts per k-step --", " *"]
    s5 = '-' * 5; s3 = '-' * 3; s4 = '-' * 4; sep = '-' * 20
    table.append(f" *   {'kernel':<20s} {'add':>5s} {'sub':>5s} {'mul':>5s} {'neg':>5s}"
                 f" {'fma':>5s} {'fms':>5s} | {'arith':>5s} {'flops':>5s}"
                 f" | {'ld':>3s} {'st':>3s} {'mem':>4s}")
    table.append(f" *   {sep} {s5} {s5} {s5} {s5}"
                 f" {s5} {s5} + {s5} {s5}"
                 f" + {s3} {s3} {s4}")
    for k in sorted(stats.keys()):
        s = stats[k]
        table.append(f" *   {k:<20s} {s['add']:5d} {s['sub']:5d} {s['mul']:5d} {s['neg']:5d}"
                     f" {s['fma']:5d} {s['fms']:5d} | {s['total_arith']:5d} {s['flops']:5d}"
                     f" | {s['load']:3d} {s['store']:3d} {s['total_mem']:4d}")
    table.append(" *")
    for i, line in enumerate(lines):
        if line.strip() == '*/':
            for j, tl in enumerate(table):
                lines.insert(i + j, tl)
            return


def print_file(lines, label):
    """Print lines to stdout with a stderr label."""
    print("\n".join(lines))
    print(f"\n=== {label} ===", file=sys.stderr)


# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(description='Unified R=7 codelet generator')
    parser.add_argument('--isa', default='avx2',
                        choices=['scalar', 'avx2', 'avx512', 'all'])
    parser.add_argument('--variant', default='notw',
                        choices=['notw', 'dit_tw', 'dif_tw', 'dit_tw_log3', 'dif_tw_log3',
                                 'ct_n1', 'ct_n1_scaled', 'ct_t1_dit', 'ct_t1s_dit', 'ct_t1_dit_log3', 'ct_t1_dif', 'ct_t1_oop_dit', 'all'])
    args = parser.parse_args()

    if args.isa == 'all':
        targets = [ISA_SCALAR, ISA_AVX2, ISA_AVX512]
    else:
        targets = [ALL_ISA[args.isa]]

    std_variants = ['notw', 'dit_tw', 'dif_tw', 'dit_tw_log3', 'dif_tw_log3']
    ct_variants = ['ct_n1', 'ct_n1_scaled', 'ct_t1_dit', 'ct_t1s_dit', 'ct_t1_dit_log3', 'ct_t1_dif', 'ct_t1_oop_dit']

    if args.variant == 'all':
        variants = std_variants + ct_variants
    else:
        variants = [args.variant]

    for isa in targets:
        for variant in variants:
            if variant.startswith('ct_'):
                lines = emit_file_ct(isa, variant)
                print_file(lines, f"{isa.name.upper()} {variant}")
            else:
                lines, stats = emit_file(isa, variant)
                sv_lines = emit_sv_variants(lines, isa, variant)
                if sv_lines:
                    for i in range(len(lines)):
                        if lines[i].strip() == '#undef LD':
                            lines[i:i] = sv_lines
                            break
                print_file(lines, f"{isa.name.upper()} {variant}")
                for k in sorted(stats.keys()):
                    s = stats[k]
                    print(f"  {k}: arith={s['total_arith']} flops={s['flops']}"
                          f" mem={s['total_mem']}", file=sys.stderr)


if __name__ == '__main__':
    main()
