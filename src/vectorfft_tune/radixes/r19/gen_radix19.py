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
gen_radix19.py -- Unified DFT-19 codelet generator for VectorFFT

Monolithic DFT-19 butterfly (straight-line genfft DAG translation).
26 constants, 542 ops (314 add + 114 mul + 114 FMA).

Register pressure WAY exceeds both AVX2 (16) and AVX-512 (32):
  19 inputs(38 regs) + 26 constants = 64+
Both ISAs use the same butterfly (no separate no-spill path).
The genfft DAG interleaves loads with computation, so we emit the
entire DAG as one straight-line block and let the C compiler handle
register allocation / spills.  All butterfly temporaries are wrapped
in a single large { } scope block.

DFT-19 constants (26, from genfft / fft_radix19_genfft.h):
  R19_K00..R19_K25

Log3 twiddle for R=19 (18 cmuls from w1):
  w2=w1^2, w3=w1*w2, w4=w2^2, w5=w2*w3, w6=w3^2,
  w7=w3*w4, w8=w4^2, w9=w4*w5, w10=w5^2, w11=w5*w6,
  w12=w6^2, w13=w6*w7, w14=w7^2, w15=w7*w8, w16=w8^2,
  w17=w8*w9, w18=w9^2

No U=2 -- register pressure too high (same reason as R=13).

n1_ovs: 19 bins = 4 groups of 4 (bins 0-3, 4-7, 8-11, 12-15)
         + bins 16,17,18 leftover (extract+scatter)

Usage:
  python3 gen_radix19.py --isa avx2 --variant all
  python3 gen_radix19.py --isa all --variant ct_n1
  python3 gen_radix19.py --isa avx2 --variant ct_t1_dit
"""

import sys, math, argparse, re

R = 19

# ----------------------------------------------------------------
# DFT-19 constants (from fft_radix19_genfft.h lines 13-38)
# ----------------------------------------------------------------
KP179300334_val = +0.179300334119296584577610532085718061875823403
KP162767826_val = +0.162767826960215280668389833805086266639764554
KP241806419_val = +0.241806419514128370059838955399511567460135386
KP013100793_val = +0.013100793502659825656895236613605693837043816
KP185607687_val = +0.185607687596956708228094509753437184655916111
KP155537010_val = +0.155537010430162564075726579585293585038171935
KP158451106_val = +0.158451106083228390851271075100177128178845017
KP183126246_val = +0.183126246861675942937369045715821923165720834
KP074045235_val = +0.074045235744154800077778893070976440188821847
KP230562959_val = +0.230562959670963878045147682107728427313711553
KP241001675_val = +0.241001675574024313981278600823211969789894482
KP023667861_val = +0.023667861736006226270400486931784179505054668
KP055555555_val = +0.055555555555555555555555555555555555555555556
KP237443964_val = +0.237443964223181698316581063237697643204557853
KP047564053_val = +0.047564053261075059242632762200029095167430147
KP079217082_val = +0.079217082690559198579121284568817782276492249
KP228837560_val = +0.228837560550358693759630648470316792073020180
KP642787609_val = +0.642787609686539326322643409907263432907559884
KP766044443_val = +0.766044443118978035202392650555416673935832457
KP939692620_val = +0.939692620785908384054109277324731469936208134
KP342020143_val = +0.342020143325668733044099614682259580763083368
KP984807753_val = +0.984807753012208059366743024589523013670643252
KP173648177_val = +0.173648177666930348851716626769314796000375677
KP242161052_val = +0.242161052418926308457610110214423092174277996
KP500000000_val = +0.500000000000000000000000000000000000000000000
KP866025403_val = +0.866025403784438646763723170752936183471402627

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

ISA_SCALAR = ISAConfig('scalar', 'double', 1, 1, '', 1, '', '', 'R19S')
ISA_AVX2   = ISAConfig('avx2', '__m256d', 4, 4, '_mm256', 4,
    '__attribute__((target("avx2,fma")))',
    '__attribute__((aligned(32)))', 'R19A')
ISA_AVX512 = ISAConfig('avx512', '__m512d', 8, 8, '_mm512', 8,
    '__attribute__((target("avx512f,avx512dq,fma")))',
    '__attribute__((aligned(64)))', 'R19L')

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
        """a*b + c"""
        self.n_fma += 1
        if self.isa.name == 'scalar': return f"({a})*({b})+({c})"
        return f"{self.isa.p}_fmadd_pd({a},{b},{c})"

    def fms(self, a, b, c):
        """a*b - c"""
        self.n_fms += 1
        if self.isa.name == 'scalar': return f"({a})*({b})-({c})"
        return f"{self.isa.p}_fmsub_pd({a},{b},{c})"

    def fnma(self, a, b, c):
        """c - a*b"""
        self.n_fma += 1
        if self.isa.name == 'scalar': return f"({c})-({a})*({b})"
        return f"{self.isa.p}_fnmadd_pd({a},{b},{c})"

    def fnms(self, a, b, c):
        """-a*b - c  (fnmsub: -(a*b+c))"""
        self.n_fms += 1
        if self.isa.name == 'scalar': return f"-(({a})*({b})+({c}))"
        return f"{self.isa.p}_fnmsub_pd({a},{b},{c})"

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

    # ------------------------------------------------------------------
    # DFT-19 butterfly
    # ------------------------------------------------------------------

    def emit_radix19_butterfly(self, d):
        """Emit DFT-19 butterfly, translating genfft DAG exactly.

        R=19 register pressure exceeds both AVX2 and AVX-512.
        Both ISAs emit the DAG as one straight-line block inside a { } scope
        and let the C compiler handle register allocation / spills.

        The butterfly body is a direct port of R19_BUTTERFLY_BODY from
        fft_radix19_genfft.h, using the split-complex layout (x0_re/x0_im..
        x18_re/x18_im) and Emitter arithmetic methods.

        For the backward direction we swap real/imag inputs/outputs
        (same as the production genfft.h: bwd calls fwd(ii,ri,io,ro)).
        """
        fwd = (d == 'fwd')
        T = self.isa.T

        self.c(f"DFT-19 butterfly [{d}] (straight-line genfft DAG, compiler-managed regs)")
        self.o("{")
        self.ind += 1

        if fwd:
            self._emit_r19_fwd()
        else:
            self._emit_r19_bwd()

        self.ind -= 1
        self.o("}")

    # ----------------------------------------------------------------
    # Forward butterfly  (direct port of R19_BUTTERFLY_BODY)
    # ----------------------------------------------------------------

    def _emit_r19_fwd(self):
        T = self.isa.T

        def V(name, expr):
            self.o(f"const {T} {name} = {expr};")

        def STr(n, expr):
            self.o(f"x{n}_re = {expr};")

        def STi(n, expr):
            self.o(f"x{n}_im = {expr};")

        def xr(n): return f"x{n}_re"
        def xi(n): return f"x{n}_im"

        # ----------------------------------------------------------------
        # Flat translation of R19_BUTTERFLY_BODY from fft_radix19_genfft.h
        # lines 40-696. All {} scopes flattened. Order follows DAG exactly.
        # ----------------------------------------------------------------

        # T1 = LD(&ri[0*K+k]);  TB = LD(&ii[0*K+k]);
        V("T1",  xr(0))
        V("TB",  xi(0))

        # Pair (1,18) real
        V("T2",  xr(1))
        V("T3",  xr(18))
        V("T4",  self.add("T2", "T3"))
        V("T1u", self.sub("T2", "T3"))

        # Pair (11,8) and (7,12) real
        V("T8",  xr(11))
        V("T9",  xr(8))
        V("Ta",  self.add("T8", "T9"))
        V("T1w", self.sub("T8", "T9"))
        V("T5",  xr(7))
        V("T6",  xr(12))
        V("T7",  self.add("T5", "T6"))
        V("T1v", self.sub("T5", "T6"))

        V("T2s", self.mul("KP866025403", self.sub("T1v", "T1w")))
        V("T2T", self.mul("KP866025403", self.sub("T7", "Ta")))
        V("Tb",  self.add("T7", "Ta"))
        V("T2z", self.fnma("KP500000000", "Tb", "T4"))
        V("T1x", self.add("T1v", "T1w"))
        V("T28", self.fnma("KP500000000", "T1x", "T1u"))

        # Pair (1,18) imag, (11,8) imag, (7,12) imag
        V("TC",  xi(1))
        V("TD",  xi(18))
        V("TE",  self.add("TC", "TD"))
        V("T1c", self.sub("TC", "TD"))

        V("TI",  xi(11))
        V("TJ",  xi(8))
        V("TK",  self.add("TI", "TJ"))
        V("T1e", self.sub("TI", "TJ"))
        V("TF",  xi(7))
        V("TG",  xi(12))
        V("TH",  self.add("TF", "TG"))
        V("T1d", self.sub("TF", "TG"))

        V("T29", self.mul("KP866025403", self.sub("T1e", "T1d")))
        V("T2A", self.mul("KP866025403", self.sub("TK", "TH")))
        V("TL",  self.add("TH", "TK"))
        V("T2S", self.fnma("KP500000000", "TL", "TE"))
        V("T1f", self.add("T1d", "T1e"))
        V("T2r", self.fnma("KP500000000", "T1f", "T1c"))

        # Pair (4,15) real, (9,10) real, (6,13) real
        V("Td",  xr(4))
        V("Te",  xr(15))
        V("Tf",  self.add("Td", "Te"))
        V("T1z", self.sub("Td", "Te"))
        V("TN",  xi(4))
        V("TO",  xi(15))
        V("TP",  self.add("TN", "TO"))
        V("T1h", self.sub("TN", "TO"))

        V("Tg",  xr(9))
        V("Th",  xr(10))
        V("Ti",  self.add("Tg", "Th"))
        V("T1A", self.sub("Tg", "Th"))
        V("Tj",  xr(6))
        V("Tk",  xr(13))
        V("Tl",  self.add("Tj", "Tk"))
        V("T1B", self.sub("Tj", "Tk"))

        V("Tm",  self.add("Ti", "Tl"))
        V("T1C", self.add("T1A", "T1B"))

        V("TQ",  xi(9))
        V("TR",  xi(10))
        V("TS",  self.add("TQ", "TR"))
        V("T1i", self.sub("TQ", "TR"))
        V("TT",  xi(6))
        V("TU",  xi(13))
        V("TV",  self.add("TT", "TU"))
        V("T1j", self.sub("TT", "TU"))

        V("TW",  self.add("TS", "TV"))
        V("T1k", self.add("T1i", "T1j"))
        V("Tn",  self.add("Tf", "Tm"))
        V("T1l", self.add("T1h", "T1k"))
        V("T1D", self.add("T1z", "T1C"))
        V("TX",  self.add("TP", "TW"))

        V("T2b", self.fnma("KP500000000", "T1C", "T1z"))
        V("T2c", self.mul("KP866025403", self.sub("T1j", "T1i")))
        V("T2d", self.sub("T2b", "T2c"))
        V("T3J", self.add("T2b", "T2c"))
        V("T2C", self.fnma("KP500000000", "Tm", "Tf"))
        V("T2D", self.mul("KP866025403", self.sub("TV", "TS")))
        V("T2E", self.sub("T2C", "T2D"))
        V("T3q", self.add("T2C", "T2D"))

        V("T2F", self.fnma("KP500000000", "TW", "TP"))
        V("T2G", self.mul("KP866025403", self.sub("Ti", "Tl")))
        V("T2H", self.sub("T2F", "T2G"))
        V("T3r", self.add("T2G", "T2F"))
        V("T2e", self.fnma("KP500000000", "T1k", "T1h"))
        V("T2f", self.mul("KP866025403", self.sub("T1A", "T1B")))
        V("T2g", self.sub("T2e", "T2f"))
        V("T3K", self.add("T2f", "T2e"))

        # Pair (16,3) real, (17,2) real, (5,14) real
        V("To",  xr(16))
        V("Tp",  xr(3))
        V("Tq",  self.add("To", "Tp"))
        V("T1E", self.sub("To", "Tp"))
        V("TY",  xi(16))
        V("TZ",  xi(3))
        V("T10", self.add("TY", "TZ"))
        V("T1m", self.sub("TY", "TZ"))

        V("Tr",  xr(17))
        V("Ts",  xr(2))
        V("Tt",  self.add("Tr", "Ts"))
        V("T1F", self.sub("Tr", "Ts"))
        V("Tu",  xr(5))
        V("Tv",  xr(14))
        V("Tw",  self.add("Tu", "Tv"))
        V("T1G", self.sub("Tu", "Tv"))

        V("Tx",  self.add("Tt", "Tw"))
        V("T1H", self.add("T1F", "T1G"))

        V("T11", xi(17))
        V("T12", xi(2))
        V("T13", self.add("T11", "T12"))
        V("T1n", self.sub("T11", "T12"))
        V("T14", xi(5))
        V("T15", xi(14))
        V("T16", self.add("T14", "T15"))
        V("T1o", self.sub("T14", "T15"))

        V("T17", self.add("T13", "T16"))
        V("T1p", self.add("T1n", "T1o"))
        V("Ty",  self.add("Tq", "Tx"))
        V("T1q", self.add("T1m", "T1p"))
        V("T1I", self.add("T1E", "T1H"))
        V("T18", self.add("T10", "T17"))

        V("T2i", self.fnma("KP500000000", "T1p", "T1m"))
        V("T2j", self.mul("KP866025403", self.sub("T1F", "T1G")))
        V("T2k", self.sub("T2i", "T2j"))
        V("T3N", self.add("T2j", "T2i"))
        V("T2J", self.fnma("KP500000000", "T17", "T10"))
        V("T2K", self.mul("KP866025403", self.sub("Tt", "Tw")))
        V("T2L", self.sub("T2J", "T2K"))
        V("T3u", self.add("T2K", "T2J"))

        V("T2M", self.fnma("KP500000000", "Tx", "Tq"))
        V("T2N", self.mul("KP866025403", self.sub("T16", "T13")))
        V("T2O", self.sub("T2M", "T2N"))
        V("T3t", self.add("T2M", "T2N"))
        V("T2l", self.fnma("KP500000000", "T1H", "T1E"))
        V("T2m", self.mul("KP866025403", self.sub("T1o", "T1n")))
        V("T2n", self.sub("T2l", "T2m"))
        V("T3M", self.add("T2l", "T2m"))

        # Outer combination: TA, T23, T1W, T1s, T1Z, T1P, T1M
        V("T1V", self.mul("KP866025403", self.sub("T18", "TX")))
        V("Tc",  self.add("T4", "Tb"))
        V("Tz",  self.add("Tn", "Ty"))
        V("T1U", self.fnma("KP500000000", "Tz", "Tc"))
        V("TA",  self.add("Tc", "Tz"))
        V("T23", self.add("T1U", "T1V"))
        V("T1W", self.sub("T1U", "T1V"))

        V("T1O", self.mul("KP866025403", self.sub("T1D", "T1I")))
        V("T1g", self.add("T1c", "T1f"))
        V("T1r", self.add("T1l", "T1q"))
        V("T1N", self.fnma("KP500000000", "T1r", "T1g"))
        V("T1s", self.mul("KP242161052", self.add("T1g", "T1r")))
        V("T1Z", self.add("T1O", "T1N"))
        V("T1P", self.sub("T1N", "T1O"))

        V("T1L", self.mul("KP866025403", self.sub("T1q", "T1l")))
        V("T1y", self.add("T1u", "T1x"))
        V("T1J", self.add("T1D", "T1I"))
        V("T1K", self.fnma("KP500000000", "T1J", "T1y"))
        V("T1M", self.sub("T1K", "T1L"))
        V("T4Z", self.mul("KP242161052", self.add("T1y", "T1J")))
        V("T20", self.add("T1K", "T1L"))

        V("T1S", self.mul("KP866025403", self.sub("Tn", "Ty")))
        V("TM",  self.add("TE", "TL"))
        V("T19", self.add("TX", "T18"))
        V("T1R", self.fnma("KP500000000", "T19", "TM"))
        V("T1a", self.add("TM", "T19"))
        V("T22", self.add("T1S", "T1R"))
        V("T1T", self.sub("T1R", "T1S"))

        # Block: T2q, T2x, T33, T3h, T36, T3g, T2R, T2Y, T3a, T3j, T3d, T3k
        V("T2a", self.sub("T28", "T29"))
        V("T2t", self.sub("T2r", "T2s"))

        V("T2h", self.fnma("KP984807753", "T2g", self.mul("KP173648177", "T2d")))
        V("T2o", self.fma("KP342020143", "T2k", self.mul("KP939692620", "T2n")))
        V("T2p", self.sub("T2h", "T2o"))
        V("T32", self.mul("KP866025403", self.add("T2h", "T2o")))
        V("T2u", self.fma("KP173648177", "T2g", self.mul("KP984807753", "T2d")))
        V("T2v", self.fnma("KP939692620", "T2k", self.mul("KP342020143", "T2n")))
        V("T2w", self.add("T2u", "T2v"))
        V("T35", self.mul("KP866025403", self.sub("T2v", "T2u")))

        V("T2q", self.add("T2a", "T2p"))
        V("T2x", self.add("T2t", "T2w"))
        V("T31", self.fnma("KP500000000", "T2w", "T2t"))
        V("T33", self.sub("T31", "T32"))
        V("T3h", self.add("T32", "T31"))
        V("T34", self.fnma("KP500000000", "T2p", "T2a"))
        V("T36", self.sub("T34", "T35"))
        V("T3g", self.add("T34", "T35"))

        V("T2B", self.sub("T2z", "T2A"))
        V("T2U", self.sub("T2S", "T2T"))

        V("T2I", self.fnma("KP984807753", "T2H", self.mul("KP173648177", "T2E")))
        V("T2P", self.fma("KP342020143", "T2L", self.mul("KP939692620", "T2O")))
        V("T2Q", self.sub("T2I", "T2P"))
        V("T3c", self.mul("KP866025403", self.add("T2I", "T2P")))
        V("T2V", self.fma("KP173648177", "T2H", self.mul("KP984807753", "T2E")))
        V("T2W", self.fnma("KP939692620", "T2L", self.mul("KP342020143", "T2O")))
        V("T2X", self.add("T2V", "T2W"))
        V("T39", self.mul("KP866025403", self.sub("T2W", "T2V")))

        V("T2R", self.add("T2B", "T2Q"))
        V("T2Y", self.add("T2U", "T2X"))
        V("T38", self.fnma("KP500000000", "T2Q", "T2B"))
        V("T3a", self.sub("T38", "T39"))
        V("T3j", self.add("T38", "T39"))
        V("T3b", self.fnma("KP500000000", "T2X", "T2U"))
        V("T3d", self.sub("T3b", "T3c"))
        V("T3k", self.add("T3c", "T3b"))

        # Block: T3B, T45, T3G, T46, T4e, T4S
        V("T3p", self.add("T2z", "T2A"))
        V("T3D", self.add("T2T", "T2S"))

        V("T3s", self.fnma("KP642787609", "T3r", self.mul("KP766044443", "T3q")))
        V("T3v", self.fnma("KP984807753", "T3u", self.mul("KP173648177", "T3t")))
        V("T3w", self.add("T3s", "T3v"))
        V("T3C", self.mul("KP866025403", self.sub("T3s", "T3v")))
        V("T3y", self.fma("KP173648177", "T3u", self.mul("KP984807753", "T3t")))
        V("T3z", self.fma("KP766044443", "T3r", self.mul("KP642787609", "T3q")))
        V("T3A", self.mul("KP866025403", self.sub("T3y", "T3z")))
        V("T3E", self.add("T3z", "T3y"))

        V("T3x", self.fnma("KP500000000", "T3w", "T3p"))
        V("T3B", self.add("T3x", "T3A"))
        V("T45", self.sub("T3x", "T3A"))
        V("T3F", self.fnma("KP500000000", "T3E", "T3D"))
        V("T3G", self.add("T3C", "T3F"))
        V("T46", self.sub("T3F", "T3C"))
        V("T4c", self.add("T3p", "T3w"))
        V("T4d", self.add("T3D", "T3E"))
        V("T4e", self.fma("KP228837560", "T4c", self.mul("KP079217082", "T4d")))
        V("T4S", self.fnma("KP079217082", "T4c", self.mul("KP228837560", "T4d")))

        # Block: T3U, T43, T3Z, T42, T4b, T4R
        V("T3I", self.add("T28", "T29"))
        V("T3W", self.add("T2s", "T2r"))

        V("T3L", self.fnma("KP642787609", "T3K", self.mul("KP766044443", "T3J")))
        V("T3O", self.fnma("KP984807753", "T3N", self.mul("KP173648177", "T3M")))
        V("T3P", self.add("T3L", "T3O"))
        V("T3V", self.mul("KP866025403", self.sub("T3L", "T3O")))
        V("T3R", self.fma("KP173648177", "T3N", self.mul("KP984807753", "T3M")))
        V("T3S", self.fma("KP766044443", "T3K", self.mul("KP642787609", "T3J")))
        V("T3T", self.mul("KP866025403", self.sub("T3R", "T3S")))
        V("T3X", self.add("T3S", "T3R"))

        V("T3Q", self.fnma("KP500000000", "T3P", "T3I"))
        V("T3U", self.add("T3Q", "T3T"))
        V("T43", self.sub("T3Q", "T3T"))
        V("T3Y", self.fnma("KP500000000", "T3X", "T3W"))
        V("T3Z", self.add("T3V", "T3Y"))
        V("T42", self.sub("T3Y", "T3V"))
        V("T49", self.add("T3I", "T3P"))
        V("T4a", self.add("T3W", "T3X"))
        V("T4b", self.fma("KP047564053", "T49", self.mul("KP237443964", "T4a")))
        V("T4R", self.fnma("KP237443964", "T49", self.mul("KP047564053", "T4a")))

        # ST(&ro[0*K+k], ADD(T1, TA));  ST(&io[0*K+k], ADD(TB, T1a));
        STr(0, self.add("T1", "TA"))
        STi(0, self.add("TB", "T1a"))

        # Big output block -- T1t through all stores
        V("T1b", self.fnma("KP055555555", "TA", "T1"))
        V("T1t", self.add("T1b", "T1s"))
        V("T4j", self.sub("T1b", "T1s"))

        V("T21", self.fma("KP023667861", "T1Z", self.mul("KP241001675", "T20")))
        V("T24", self.fma("KP230562959", "T22", self.mul("KP074045235", "T23")))
        V("T25", self.add("T21", "T24"))
        V("T4l", self.sub("T21", "T24"))
        V("T1Q", self.fnma("KP023667861", "T1P", self.mul("KP241001675", "T1M")))
        V("T1X", self.fnma("KP074045235", "T1W", self.mul("KP230562959", "T1T")))
        V("T1Y", self.add("T1Q", "T1X"))
        V("T4k", self.sub("T1X", "T1Q"))

        V("T5u", self.mul("KP866025403", self.add("T1Y", "T25")))
        V("T6A", self.mul("KP866025403", self.sub("T4l", "T4k")))
        V("T26", self.sub("T1Y", "T25"))
        V("T5b", self.fnma("KP500000000", "T26", "T1t"))
        V("T4m", self.add("T4k", "T4l"))
        V("T6d", self.fnma("KP500000000", "T4m", "T4j"))

        V("T4Y", self.fnma("KP055555555", "T1a", "TB"))
        V("T50", self.sub("T4Y", "T4Z"))
        V("T65", self.add("T4Z", "T4Y"))

        V("T54", self.fnma("KP241001675", "T1Z", self.mul("KP023667861", "T20")))
        V("T55", self.fnma("KP074045235", "T22", self.mul("KP230562959", "T23")))
        V("T56", self.add("T54", "T55"))
        V("T67", self.sub("T55", "T54"))
        V("T51", self.fma("KP241001675", "T1P", self.mul("KP023667861", "T1M")))
        V("T52", self.fma("KP074045235", "T1T", self.mul("KP230562959", "T1W")))
        V("T53", self.sub("T51", "T52"))
        V("T66", self.add("T51", "T52"))

        V("T5c", self.mul("KP866025403", self.sub("T53", "T56")))
        V("T6e", self.mul("KP866025403", self.add("T66", "T67")))
        V("T57", self.add("T53", "T56"))
        V("T5v", self.fnma("KP500000000", "T57", "T50"))
        V("T68", self.sub("T66", "T67"))
        V("T6z", self.fma("KP500000000", "T68", "T65"))

        # Inner block: T41..T4x, T5n..T5I, T6p..T6K
        V("T3H", self.fma("KP183126246", "T3B", self.mul("KP158451106", "T3G")))
        V("T40", self.fma("KP155537010", "T3U", self.mul("KP185607687", "T3Z")))
        V("T41", self.sub("T3H", "T40"))
        V("T4t", self.add("T40", "T3H"))
        V("T4L", self.fnma("KP155537010", "T3Z", self.mul("KP185607687", "T3U")))
        V("T4M", self.fnma("KP158451106", "T3B", self.mul("KP183126246", "T3G")))
        V("T4N", self.add("T4L", "T4M"))
        V("T5Z", self.sub("T4M", "T4L"))

        V("T4f", self.add("T4b", "T4e"))
        V("T4v", self.sub("T4e", "T4b"))
        V("T44", self.fnma("KP241806419", "T43", self.mul("KP013100793", "T42")))
        V("T47", self.fnma("KP179300334", "T46", self.mul("KP162767826", "T45")))
        V("T48", self.add("T44", "T47"))
        V("T4u", self.sub("T47", "T44"))
        V("T4g", self.add("T48", "T4f"))
        V("T4w", self.add("T4u", "T4v"))
        V("T4T", self.add("T4R", "T4S"))
        V("T61", self.sub("T4S", "T4R"))
        V("T4O", self.fma("KP179300334", "T45", self.mul("KP162767826", "T46")))
        V("T4P", self.fma("KP013100793", "T43", self.mul("KP241806419", "T42")))
        V("T4Q", self.sub("T4O", "T4P"))
        V("T60", self.add("T4P", "T4O"))
        V("T4U", self.add("T4Q", "T4T"))
        V("T62", self.add("T60", "T61"))

        V("T4h", self.add("T41", "T4g"))
        V("T4V", self.add("T4N", "T4U"))
        V("T63", self.add("T5Z", "T62"))
        V("T4x", self.add("T4t", "T4w"))

        V("T5l", self.fnma("KP500000000", "T4U", "T4N"))
        V("T5m", self.mul("KP866025403", self.sub("T48", "T4f")))
        V("T5n", self.sub("T5l", "T5m"))
        V("T5H", self.add("T5m", "T5l"))
        V("T6n", self.fnma("KP500000000", "T62", "T5Z"))
        V("T6o", self.mul("KP866025403", self.sub("T4u", "T4v")))
        V("T6p", self.sub("T6n", "T6o"))
        V("T6J", self.add("T6o", "T6n"))

        V("T6q", self.fnma("KP500000000", "T4w", "T4t"))
        V("T6r", self.mul("KP866025403", self.sub("T61", "T60")))
        V("T6s", self.sub("T6q", "T6r"))
        V("T6K", self.add("T6q", "T6r"))
        V("T5o", self.fnma("KP500000000", "T4g", "T41"))
        V("T5p", self.mul("KP866025403", self.sub("T4T", "T4Q")))
        V("T5q", self.sub("T5o", "T5p"))
        V("T5I", self.add("T5o", "T5p"))

        # Inner block: T30..T4K, T5Y, T4s, T5g..T5E, T6i..T6G
        V("T2y", self.fma("KP241806419", "T2q", self.mul("KP013100793", "T2x")))
        V("T2Z", self.fma("KP162767826", "T2R", self.mul("KP179300334", "T2Y")))
        V("T30", self.add("T2y", "T2Z"))
        V("T4o", self.sub("T2Z", "T2y"))
        V("T4A", self.fnma("KP013100793", "T2q", self.mul("KP241806419", "T2x")))
        V("T4B", self.fnma("KP179300334", "T2R", self.mul("KP162767826", "T2Y")))
        V("T4C", self.add("T4A", "T4B"))
        V("T5U", self.sub("T4B", "T4A"))

        V("T37", self.fnma("KP047564053", "T36", self.mul("KP237443964", "T33")))
        V("T3e", self.fnma("KP079217082", "T3d", self.mul("KP228837560", "T3a")))
        V("T3f", self.add("T37", "T3e"))
        V("T4p", self.sub("T3e", "T37"))
        V("T3i", self.fnma("KP185607687", "T3h", self.mul("KP155537010", "T3g")))
        V("T3l", self.fnma("KP158451106", "T3k", self.mul("KP183126246", "T3j")))
        V("T3m", self.add("T3i", "T3l"))
        V("T4q", self.sub("T3l", "T3i"))

        V("T3n", self.add("T3f", "T3m"))
        V("T4r", self.add("T4p", "T4q"))

        V("T4D", self.fma("KP228837560", "T3d", self.mul("KP079217082", "T3a")))
        V("T4E", self.fma("KP047564053", "T33", self.mul("KP237443964", "T36")))
        V("T4F", self.sub("T4D", "T4E"))
        V("T5V", self.add("T4E", "T4D"))
        V("T4G", self.fma("KP155537010", "T3h", self.mul("KP185607687", "T3g")))
        V("T4H", self.fma("KP183126246", "T3k", self.mul("KP158451106", "T3j")))
        V("T4I", self.add("T4G", "T4H"))
        V("T5W", self.sub("T4H", "T4G"))

        V("T4J", self.add("T4F", "T4I"))
        V("T5X", self.add("T5V", "T5W"))
        V("T3o", self.add("T30", "T3n"))
        V("T4K", self.add("T4C", "T4J"))
        V("T5Y", self.add("T5U", "T5X"))
        V("T4s", self.add("T4o", "T4r"))

        V("T5e", self.fnma("KP500000000", "T3n", "T30"))
        V("T5f", self.mul("KP866025403", self.sub("T4F", "T4I")))
        V("T5g", self.sub("T5e", "T5f"))
        V("T5F", self.add("T5e", "T5f"))
        V("T6g", self.fnma("KP500000000", "T4r", "T4o"))
        V("T6h", self.mul("KP866025403", self.sub("T5V", "T5W")))
        V("T6i", self.sub("T6g", "T6h"))
        V("T6H", self.add("T6g", "T6h"))

        V("T6j", self.fnma("KP500000000", "T5X", "T5U"))
        V("T6k", self.mul("KP866025403", self.sub("T4q", "T4p")))
        V("T6l", self.sub("T6j", "T6k"))
        V("T6G", self.add("T6k", "T6j"))
        V("T5h", self.fnma("KP500000000", "T4J", "T4C"))
        V("T5i", self.mul("KP866025403", self.sub("T3m", "T3f")))
        V("T5j", self.sub("T5h", "T5i"))
        V("T5E", self.add("T5i", "T5h"))

        # Stores: bin 1 group
        V("T4W", self.mul("KP866025403", self.sub("T4K", "T4V")))
        V("T27", self.add("T1t", "T26"))
        V("T4i", self.add("T3o", "T4h"))
        V("T4z", self.fnma("KP500000000", "T4i", "T27"))
        STr(1,  self.add("T27", "T4i"))
        STr(7,  self.add("T4z", "T4W"))
        STr(11, self.sub("T4z", "T4W"))

        V("T4X", self.mul("KP866025403", self.sub("T4h", "T3o")))
        V("T58", self.add("T50", "T57"))
        V("T59", self.add("T4K", "T4V"))
        V("T5a", self.fnma("KP500000000", "T59", "T58"))
        STi(7,  self.add("T4X", "T5a"))
        STi(1,  self.add("T58", "T59"))
        STi(11, self.sub("T5a", "T4X"))

        V("T64", self.mul("KP866025403", self.sub("T5Y", "T63")))
        V("T4n", self.add("T4j", "T4m"))
        V("T4y", self.add("T4s", "T4x"))
        V("T5T", self.fnma("KP500000000", "T4y", "T4n"))
        STr(18, self.add("T4n", "T4y"))
        STr(12, self.add("T5T", "T64"))
        STr(8,  self.sub("T5T", "T64"))

        V("T6c", self.mul("KP866025403", self.sub("T4x", "T4s")))
        V("T69", self.sub("T65", "T68"))
        V("T6a", self.add("T5Y", "T63"))
        V("T6b", self.fnma("KP500000000", "T6a", "T69"))
        STi(8,  self.sub("T6b", "T6c"))
        STi(18, self.add("T69", "T6a"))
        STi(12, self.add("T6c", "T6b"))

        # Stores: bin 5 group
        V("T5d", self.sub("T5b", "T5c"))
        V("T5w", self.add("T5u", "T5v"))

        V("T5k", self.fma("KP173648177", "T5g", self.mul("KP984807753", "T5j")))
        V("T5r", self.fnma("KP939692620", "T5q", self.mul("KP342020143", "T5n")))
        V("T5s", self.add("T5k", "T5r"))
        V("T5t", self.mul("KP866025403", self.sub("T5r", "T5k")))
        V("T5x", self.fnma("KP984807753", "T5g", self.mul("KP173648177", "T5j")))
        V("T5y", self.fma("KP939692620", "T5n", self.mul("KP342020143", "T5q")))
        V("T5z", self.sub("T5x", "T5y"))
        V("T5C", self.mul("KP866025403", self.add("T5x", "T5y")))

        STr(5,  self.add("T5d", "T5s"))
        STi(5,  self.add("T5w", "T5z"))
        V("T5A", self.fnma("KP500000000", "T5z", "T5w"))
        STi(16, self.add("T5t", "T5A"))
        STi(17, self.sub("T5A", "T5t"))
        V("T5B", self.fnma("KP500000000", "T5s", "T5d"))
        STr(17, self.sub("T5B", "T5C"))
        STr(16, self.add("T5B", "T5C"))

        # Stores: bin 14 group
        V("T6f", self.add("T6d", "T6e"))
        V("T6B", self.sub("T6z", "T6A"))

        V("T6m", self.fma("KP173648177", "T6i", self.mul("KP984807753", "T6l")))
        V("T6t", self.fnma("KP939692620", "T6s", self.mul("KP342020143", "T6p")))
        V("T6u", self.add("T6m", "T6t"))
        V("T6E", self.mul("KP866025403", self.sub("T6t", "T6m")))
        V("T6w", self.fnma("KP984807753", "T6i", self.mul("KP173648177", "T6l")))
        V("T6x", self.fma("KP939692620", "T6p", self.mul("KP342020143", "T6s")))
        V("T6y", self.mul("KP866025403", self.add("T6w", "T6x")))
        V("T6C", self.sub("T6x", "T6w"))

        STr(14, self.add("T6f", "T6u"))
        STi(14, self.sub("T6B", "T6C"))
        V("T6v", self.fnma("KP500000000", "T6u", "T6f"))
        STr(2,  self.sub("T6v", "T6y"))
        STr(3,  self.add("T6v", "T6y"))
        V("T6D", self.fma("KP500000000", "T6C", "T6B"))
        STi(2,  self.sub("T6D", "T6E"))
        STi(3,  self.add("T6E", "T6D"))

        # Stores: bin 10 group
        V("T6F", self.add("T6A", "T6z"))
        V("T6O", self.sub("T6d", "T6e"))

        V("T6I", self.fnma("KP642787609", "T6H", self.mul("KP766044443", "T6G")))
        V("T6L", self.fnma("KP984807753", "T6K", self.mul("KP173648177", "T6J")))
        V("T6M", self.add("T6I", "T6L"))
        V("T6N", self.mul("KP866025403", self.sub("T6I", "T6L")))
        V("T6P", self.fma("KP766044443", "T6H", self.mul("KP642787609", "T6G")))
        V("T6Q", self.fma("KP984807753", "T6J", self.mul("KP173648177", "T6K")))
        V("T6R", self.add("T6P", "T6Q"))
        V("T6U", self.mul("KP866025403", self.sub("T6Q", "T6P")))

        STi(10, self.add("T6F", "T6M"))
        STr(10, self.add("T6O", "T6R"))
        V("T6S", self.fnma("KP500000000", "T6R", "T6O"))
        STr(13, self.add("T6N", "T6S"))
        STr(15, self.sub("T6S", "T6N"))
        V("T6T", self.fnma("KP500000000", "T6M", "T6F"))
        STi(15, self.sub("T6T", "T6U"))
        STi(13, self.add("T6T", "T6U"))

        # Stores: bin 9 group
        V("T5D", self.sub("T5v", "T5u"))
        V("T5L", self.add("T5b", "T5c"))

        V("T5G", self.fnma("KP642787609", "T5F", self.mul("KP766044443", "T5E")))
        V("T5J", self.fnma("KP984807753", "T5I", self.mul("KP173648177", "T5H")))
        V("T5K", self.add("T5G", "T5J"))
        V("T5Q", self.mul("KP866025403", self.sub("T5G", "T5J")))
        V("T5M", self.fma("KP766044443", "T5F", self.mul("KP642787609", "T5E")))
        V("T5N", self.fma("KP984807753", "T5H", self.mul("KP173648177", "T5I")))
        V("T5O", self.add("T5M", "T5N"))
        V("T5S", self.mul("KP866025403", self.sub("T5N", "T5M")))

        STi(9,  self.add("T5D", "T5K"))
        STr(9,  self.add("T5L", "T5O"))
        V("T5P", self.fnma("KP500000000", "T5O", "T5L"))
        STr(4,  self.sub("T5P", "T5Q"))
        STr(6,  self.add("T5Q", "T5P"))
        V("T5R", self.fnma("KP500000000", "T5K", "T5D"))
        STi(4,  self.sub("T5R", "T5S"))
        STi(6,  self.add("T5R", "T5S"))

    def _emit_r19_bwd(self):
        """Backward DFT-19: bwd(ri,ii,ro,io) = fwd(ii,ri,io,ro).

        The inputs x{n}_re and x{n}_im were loaded by emit_kernel_body
        from in_re and in_im respectively.  For backward we treat:
          - x{n}_im as the "real" input (ii->ri)
          - x{n}_re as the "imaginary" input (ri->ii)
        and swap the real/imaginary output buffers.

        We do this by re-emitting the forward butterfly with swapped
        variable name references and swapped store targets.
        """
        T = self.isa.T

        def V(name, expr):
            self.o(f"const {T} {name} = {expr};")

        def STr(n, expr):
            self.o(f"x{n}_im = {expr};")

        def STi(n, expr):
            self.o(f"x{n}_re = {expr};")

        def xr(n): return f"x{n}_im"
        def xi(n): return f"x{n}_re"

        # ----------------------------------------------------------------
        # Forward butterfly DAG with swapped xr/xi and swapped STr/STi.
        # ----------------------------------------------------------------

        V("T1",  xr(0))
        V("TB",  xi(0))

        V("T2",  xr(1))
        V("T3",  xr(18))
        V("T4",  self.add("T2", "T3"))
        V("T1u", self.sub("T2", "T3"))

        V("T8",  xr(11))
        V("T9",  xr(8))
        V("Ta",  self.add("T8", "T9"))
        V("T1w", self.sub("T8", "T9"))
        V("T5",  xr(7))
        V("T6",  xr(12))
        V("T7",  self.add("T5", "T6"))
        V("T1v", self.sub("T5", "T6"))

        V("T2s", self.mul("KP866025403", self.sub("T1v", "T1w")))
        V("T2T", self.mul("KP866025403", self.sub("T7", "Ta")))
        V("Tb",  self.add("T7", "Ta"))
        V("T2z", self.fnma("KP500000000", "Tb", "T4"))
        V("T1x", self.add("T1v", "T1w"))
        V("T28", self.fnma("KP500000000", "T1x", "T1u"))

        V("TC",  xi(1))
        V("TD",  xi(18))
        V("TE",  self.add("TC", "TD"))
        V("T1c", self.sub("TC", "TD"))

        V("TI",  xi(11))
        V("TJ",  xi(8))
        V("TK",  self.add("TI", "TJ"))
        V("T1e", self.sub("TI", "TJ"))
        V("TF",  xi(7))
        V("TG",  xi(12))
        V("TH",  self.add("TF", "TG"))
        V("T1d", self.sub("TF", "TG"))

        V("T29", self.mul("KP866025403", self.sub("T1e", "T1d")))
        V("T2A", self.mul("KP866025403", self.sub("TK", "TH")))
        V("TL",  self.add("TH", "TK"))
        V("T2S", self.fnma("KP500000000", "TL", "TE"))
        V("T1f", self.add("T1d", "T1e"))
        V("T2r", self.fnma("KP500000000", "T1f", "T1c"))

        V("Td",  xr(4))
        V("Te",  xr(15))
        V("Tf",  self.add("Td", "Te"))
        V("T1z", self.sub("Td", "Te"))
        V("TN",  xi(4))
        V("TO",  xi(15))
        V("TP",  self.add("TN", "TO"))
        V("T1h", self.sub("TN", "TO"))

        V("Tg",  xr(9))
        V("Th",  xr(10))
        V("Ti",  self.add("Tg", "Th"))
        V("T1A", self.sub("Tg", "Th"))
        V("Tj",  xr(6))
        V("Tk",  xr(13))
        V("Tl",  self.add("Tj", "Tk"))
        V("T1B", self.sub("Tj", "Tk"))

        V("Tm",  self.add("Ti", "Tl"))
        V("T1C", self.add("T1A", "T1B"))

        V("TQ",  xi(9))
        V("TR",  xi(10))
        V("TS",  self.add("TQ", "TR"))
        V("T1i", self.sub("TQ", "TR"))
        V("TT",  xi(6))
        V("TU",  xi(13))
        V("TV",  self.add("TT", "TU"))
        V("T1j", self.sub("TT", "TU"))

        V("TW",  self.add("TS", "TV"))
        V("T1k", self.add("T1i", "T1j"))
        V("Tn",  self.add("Tf", "Tm"))
        V("T1l", self.add("T1h", "T1k"))
        V("T1D", self.add("T1z", "T1C"))
        V("TX",  self.add("TP", "TW"))

        V("T2b", self.fnma("KP500000000", "T1C", "T1z"))
        V("T2c", self.mul("KP866025403", self.sub("T1j", "T1i")))
        V("T2d", self.sub("T2b", "T2c"))
        V("T3J", self.add("T2b", "T2c"))
        V("T2C", self.fnma("KP500000000", "Tm", "Tf"))
        V("T2D", self.mul("KP866025403", self.sub("TV", "TS")))
        V("T2E", self.sub("T2C", "T2D"))
        V("T3q", self.add("T2C", "T2D"))

        V("T2F", self.fnma("KP500000000", "TW", "TP"))
        V("T2G", self.mul("KP866025403", self.sub("Ti", "Tl")))
        V("T2H", self.sub("T2F", "T2G"))
        V("T3r", self.add("T2G", "T2F"))
        V("T2e", self.fnma("KP500000000", "T1k", "T1h"))
        V("T2f", self.mul("KP866025403", self.sub("T1A", "T1B")))
        V("T2g", self.sub("T2e", "T2f"))
        V("T3K", self.add("T2f", "T2e"))

        V("To",  xr(16))
        V("Tp",  xr(3))
        V("Tq",  self.add("To", "Tp"))
        V("T1E", self.sub("To", "Tp"))
        V("TY",  xi(16))
        V("TZ",  xi(3))
        V("T10", self.add("TY", "TZ"))
        V("T1m", self.sub("TY", "TZ"))

        V("Tr",  xr(17))
        V("Ts",  xr(2))
        V("Tt",  self.add("Tr", "Ts"))
        V("T1F", self.sub("Tr", "Ts"))
        V("Tu",  xr(5))
        V("Tv",  xr(14))
        V("Tw",  self.add("Tu", "Tv"))
        V("T1G", self.sub("Tu", "Tv"))

        V("Tx",  self.add("Tt", "Tw"))
        V("T1H", self.add("T1F", "T1G"))

        V("T11", xi(17))
        V("T12", xi(2))
        V("T13", self.add("T11", "T12"))
        V("T1n", self.sub("T11", "T12"))
        V("T14", xi(5))
        V("T15", xi(14))
        V("T16", self.add("T14", "T15"))
        V("T1o", self.sub("T14", "T15"))

        V("T17", self.add("T13", "T16"))
        V("T1p", self.add("T1n", "T1o"))
        V("Ty",  self.add("Tq", "Tx"))
        V("T1q", self.add("T1m", "T1p"))
        V("T1I", self.add("T1E", "T1H"))
        V("T18", self.add("T10", "T17"))

        V("T2i", self.fnma("KP500000000", "T1p", "T1m"))
        V("T2j", self.mul("KP866025403", self.sub("T1F", "T1G")))
        V("T2k", self.sub("T2i", "T2j"))
        V("T3N", self.add("T2j", "T2i"))
        V("T2J", self.fnma("KP500000000", "T17", "T10"))
        V("T2K", self.mul("KP866025403", self.sub("Tt", "Tw")))
        V("T2L", self.sub("T2J", "T2K"))
        V("T3u", self.add("T2K", "T2J"))

        V("T2M", self.fnma("KP500000000", "Tx", "Tq"))
        V("T2N", self.mul("KP866025403", self.sub("T16", "T13")))
        V("T2O", self.sub("T2M", "T2N"))
        V("T3t", self.add("T2M", "T2N"))
        V("T2l", self.fnma("KP500000000", "T1H", "T1E"))
        V("T2m", self.mul("KP866025403", self.sub("T1o", "T1n")))
        V("T2n", self.sub("T2l", "T2m"))
        V("T3M", self.add("T2l", "T2m"))

        V("T1V", self.mul("KP866025403", self.sub("T18", "TX")))
        V("Tc",  self.add("T4", "Tb"))
        V("Tz",  self.add("Tn", "Ty"))
        V("T1U", self.fnma("KP500000000", "Tz", "Tc"))
        V("TA",  self.add("Tc", "Tz"))
        V("T23", self.add("T1U", "T1V"))
        V("T1W", self.sub("T1U", "T1V"))

        V("T1O", self.mul("KP866025403", self.sub("T1D", "T1I")))
        V("T1g", self.add("T1c", "T1f"))
        V("T1r", self.add("T1l", "T1q"))
        V("T1N", self.fnma("KP500000000", "T1r", "T1g"))
        V("T1s", self.mul("KP242161052", self.add("T1g", "T1r")))
        V("T1Z", self.add("T1O", "T1N"))
        V("T1P", self.sub("T1N", "T1O"))

        V("T1L", self.mul("KP866025403", self.sub("T1q", "T1l")))
        V("T1y", self.add("T1u", "T1x"))
        V("T1J", self.add("T1D", "T1I"))
        V("T1K", self.fnma("KP500000000", "T1J", "T1y"))
        V("T1M", self.sub("T1K", "T1L"))
        V("T4Z", self.mul("KP242161052", self.add("T1y", "T1J")))
        V("T20", self.add("T1K", "T1L"))

        V("T1S", self.mul("KP866025403", self.sub("Tn", "Ty")))
        V("TM",  self.add("TE", "TL"))
        V("T19", self.add("TX", "T18"))
        V("T1R", self.fnma("KP500000000", "T19", "TM"))
        V("T1a", self.add("TM", "T19"))
        V("T22", self.add("T1S", "T1R"))
        V("T1T", self.sub("T1R", "T1S"))

        V("T2a", self.sub("T28", "T29"))
        V("T2t", self.sub("T2r", "T2s"))

        V("T2h", self.fnma("KP984807753", "T2g", self.mul("KP173648177", "T2d")))
        V("T2o", self.fma("KP342020143", "T2k", self.mul("KP939692620", "T2n")))
        V("T2p", self.sub("T2h", "T2o"))
        V("T32", self.mul("KP866025403", self.add("T2h", "T2o")))
        V("T2u", self.fma("KP173648177", "T2g", self.mul("KP984807753", "T2d")))
        V("T2v", self.fnma("KP939692620", "T2k", self.mul("KP342020143", "T2n")))
        V("T2w", self.add("T2u", "T2v"))
        V("T35", self.mul("KP866025403", self.sub("T2v", "T2u")))

        V("T2q", self.add("T2a", "T2p"))
        V("T2x", self.add("T2t", "T2w"))
        V("T31", self.fnma("KP500000000", "T2w", "T2t"))
        V("T33", self.sub("T31", "T32"))
        V("T3h", self.add("T32", "T31"))
        V("T34", self.fnma("KP500000000", "T2p", "T2a"))
        V("T36", self.sub("T34", "T35"))
        V("T3g", self.add("T34", "T35"))

        V("T2B", self.sub("T2z", "T2A"))
        V("T2U", self.sub("T2S", "T2T"))

        V("T2I", self.fnma("KP984807753", "T2H", self.mul("KP173648177", "T2E")))
        V("T2P", self.fma("KP342020143", "T2L", self.mul("KP939692620", "T2O")))
        V("T2Q", self.sub("T2I", "T2P"))
        V("T3c", self.mul("KP866025403", self.add("T2I", "T2P")))
        V("T2V", self.fma("KP173648177", "T2H", self.mul("KP984807753", "T2E")))
        V("T2W", self.fnma("KP939692620", "T2L", self.mul("KP342020143", "T2O")))
        V("T2X", self.add("T2V", "T2W"))
        V("T39", self.mul("KP866025403", self.sub("T2W", "T2V")))

        V("T2R", self.add("T2B", "T2Q"))
        V("T2Y", self.add("T2U", "T2X"))
        V("T38", self.fnma("KP500000000", "T2Q", "T2B"))
        V("T3a", self.sub("T38", "T39"))
        V("T3j", self.add("T38", "T39"))
        V("T3b", self.fnma("KP500000000", "T2X", "T2U"))
        V("T3d", self.sub("T3b", "T3c"))
        V("T3k", self.add("T3c", "T3b"))

        V("T3p", self.add("T2z", "T2A"))
        V("T3D", self.add("T2T", "T2S"))

        V("T3s", self.fnma("KP642787609", "T3r", self.mul("KP766044443", "T3q")))
        V("T3v", self.fnma("KP984807753", "T3u", self.mul("KP173648177", "T3t")))
        V("T3w", self.add("T3s", "T3v"))
        V("T3C", self.mul("KP866025403", self.sub("T3s", "T3v")))
        V("T3y", self.fma("KP173648177", "T3u", self.mul("KP984807753", "T3t")))
        V("T3z", self.fma("KP766044443", "T3r", self.mul("KP642787609", "T3q")))
        V("T3A", self.mul("KP866025403", self.sub("T3y", "T3z")))
        V("T3E", self.add("T3z", "T3y"))

        V("T3x", self.fnma("KP500000000", "T3w", "T3p"))
        V("T3B", self.add("T3x", "T3A"))
        V("T45", self.sub("T3x", "T3A"))
        V("T3F", self.fnma("KP500000000", "T3E", "T3D"))
        V("T3G", self.add("T3C", "T3F"))
        V("T46", self.sub("T3F", "T3C"))
        V("T4c", self.add("T3p", "T3w"))
        V("T4d", self.add("T3D", "T3E"))
        V("T4e", self.fma("KP228837560", "T4c", self.mul("KP079217082", "T4d")))
        V("T4S", self.fnma("KP079217082", "T4c", self.mul("KP228837560", "T4d")))

        V("T3I", self.add("T28", "T29"))
        V("T3W", self.add("T2s", "T2r"))

        V("T3L", self.fnma("KP642787609", "T3K", self.mul("KP766044443", "T3J")))
        V("T3O", self.fnma("KP984807753", "T3N", self.mul("KP173648177", "T3M")))
        V("T3P", self.add("T3L", "T3O"))
        V("T3V", self.mul("KP866025403", self.sub("T3L", "T3O")))
        V("T3R", self.fma("KP173648177", "T3N", self.mul("KP984807753", "T3M")))
        V("T3S", self.fma("KP766044443", "T3K", self.mul("KP642787609", "T3J")))
        V("T3T", self.mul("KP866025403", self.sub("T3R", "T3S")))
        V("T3X", self.add("T3S", "T3R"))

        V("T3Q", self.fnma("KP500000000", "T3P", "T3I"))
        V("T3U", self.add("T3Q", "T3T"))
        V("T43", self.sub("T3Q", "T3T"))
        V("T3Y", self.fnma("KP500000000", "T3X", "T3W"))
        V("T3Z", self.add("T3V", "T3Y"))
        V("T42", self.sub("T3Y", "T3V"))
        V("T49", self.add("T3I", "T3P"))
        V("T4a", self.add("T3W", "T3X"))
        V("T4b", self.fma("KP047564053", "T49", self.mul("KP237443964", "T4a")))
        V("T4R", self.fnma("KP237443964", "T49", self.mul("KP047564053", "T4a")))

        STr(0, self.add("T1", "TA"))
        STi(0, self.add("TB", "T1a"))

        V("T1b", self.fnma("KP055555555", "TA", "T1"))
        V("T1t", self.add("T1b", "T1s"))
        V("T4j", self.sub("T1b", "T1s"))

        V("T21", self.fma("KP023667861", "T1Z", self.mul("KP241001675", "T20")))
        V("T24", self.fma("KP230562959", "T22", self.mul("KP074045235", "T23")))
        V("T25", self.add("T21", "T24"))
        V("T4l", self.sub("T21", "T24"))
        V("T1Q", self.fnma("KP023667861", "T1P", self.mul("KP241001675", "T1M")))
        V("T1X", self.fnma("KP074045235", "T1W", self.mul("KP230562959", "T1T")))
        V("T1Y", self.add("T1Q", "T1X"))
        V("T4k", self.sub("T1X", "T1Q"))

        V("T5u", self.mul("KP866025403", self.add("T1Y", "T25")))
        V("T6A", self.mul("KP866025403", self.sub("T4l", "T4k")))
        V("T26", self.sub("T1Y", "T25"))
        V("T5b", self.fnma("KP500000000", "T26", "T1t"))
        V("T4m", self.add("T4k", "T4l"))
        V("T6d", self.fnma("KP500000000", "T4m", "T4j"))

        V("T4Y", self.fnma("KP055555555", "T1a", "TB"))
        V("T50", self.sub("T4Y", "T4Z"))
        V("T65", self.add("T4Z", "T4Y"))

        V("T54", self.fnma("KP241001675", "T1Z", self.mul("KP023667861", "T20")))
        V("T55", self.fnma("KP074045235", "T22", self.mul("KP230562959", "T23")))
        V("T56", self.add("T54", "T55"))
        V("T67", self.sub("T55", "T54"))
        V("T51", self.fma("KP241001675", "T1P", self.mul("KP023667861", "T1M")))
        V("T52", self.fma("KP074045235", "T1T", self.mul("KP230562959", "T1W")))
        V("T53", self.sub("T51", "T52"))
        V("T66", self.add("T51", "T52"))

        V("T5c", self.mul("KP866025403", self.sub("T53", "T56")))
        V("T6e", self.mul("KP866025403", self.add("T66", "T67")))
        V("T57", self.add("T53", "T56"))
        V("T5v", self.fnma("KP500000000", "T57", "T50"))
        V("T68", self.sub("T66", "T67"))
        V("T6z", self.fma("KP500000000", "T68", "T65"))

        V("T3H", self.fma("KP183126246", "T3B", self.mul("KP158451106", "T3G")))
        V("T40", self.fma("KP155537010", "T3U", self.mul("KP185607687", "T3Z")))
        V("T41", self.sub("T3H", "T40"))
        V("T4t", self.add("T40", "T3H"))
        V("T4L", self.fnma("KP155537010", "T3Z", self.mul("KP185607687", "T3U")))
        V("T4M", self.fnma("KP158451106", "T3B", self.mul("KP183126246", "T3G")))
        V("T4N", self.add("T4L", "T4M"))
        V("T5Z", self.sub("T4M", "T4L"))

        V("T4f", self.add("T4b", "T4e"))
        V("T4v", self.sub("T4e", "T4b"))
        V("T44", self.fnma("KP241806419", "T43", self.mul("KP013100793", "T42")))
        V("T47", self.fnma("KP179300334", "T46", self.mul("KP162767826", "T45")))
        V("T48", self.add("T44", "T47"))
        V("T4u", self.sub("T47", "T44"))
        V("T4g", self.add("T48", "T4f"))
        V("T4w", self.add("T4u", "T4v"))
        V("T4T", self.add("T4R", "T4S"))
        V("T61", self.sub("T4S", "T4R"))
        V("T4O", self.fma("KP179300334", "T45", self.mul("KP162767826", "T46")))
        V("T4P", self.fma("KP013100793", "T43", self.mul("KP241806419", "T42")))
        V("T4Q", self.sub("T4O", "T4P"))
        V("T60", self.add("T4P", "T4O"))
        V("T4U", self.add("T4Q", "T4T"))
        V("T62", self.add("T60", "T61"))

        V("T4h", self.add("T41", "T4g"))
        V("T4V", self.add("T4N", "T4U"))
        V("T63", self.add("T5Z", "T62"))
        V("T4x", self.add("T4t", "T4w"))

        V("T5l", self.fnma("KP500000000", "T4U", "T4N"))
        V("T5m", self.mul("KP866025403", self.sub("T48", "T4f")))
        V("T5n", self.sub("T5l", "T5m"))
        V("T5H", self.add("T5m", "T5l"))
        V("T6n", self.fnma("KP500000000", "T62", "T5Z"))
        V("T6o", self.mul("KP866025403", self.sub("T4u", "T4v")))
        V("T6p", self.sub("T6n", "T6o"))
        V("T6J", self.add("T6o", "T6n"))

        V("T6q", self.fnma("KP500000000", "T4w", "T4t"))
        V("T6r", self.mul("KP866025403", self.sub("T61", "T60")))
        V("T6s", self.sub("T6q", "T6r"))
        V("T6K", self.add("T6q", "T6r"))
        V("T5o", self.fnma("KP500000000", "T4g", "T41"))
        V("T5p", self.mul("KP866025403", self.sub("T4T", "T4Q")))
        V("T5q", self.sub("T5o", "T5p"))
        V("T5I", self.add("T5o", "T5p"))

        V("T2y", self.fma("KP241806419", "T2q", self.mul("KP013100793", "T2x")))
        V("T2Z", self.fma("KP162767826", "T2R", self.mul("KP179300334", "T2Y")))
        V("T30", self.add("T2y", "T2Z"))
        V("T4o", self.sub("T2Z", "T2y"))
        V("T4A", self.fnma("KP013100793", "T2q", self.mul("KP241806419", "T2x")))
        V("T4B", self.fnma("KP179300334", "T2R", self.mul("KP162767826", "T2Y")))
        V("T4C", self.add("T4A", "T4B"))
        V("T5U", self.sub("T4B", "T4A"))

        V("T37", self.fnma("KP047564053", "T36", self.mul("KP237443964", "T33")))
        V("T3e", self.fnma("KP079217082", "T3d", self.mul("KP228837560", "T3a")))
        V("T3f", self.add("T37", "T3e"))
        V("T4p", self.sub("T3e", "T37"))
        V("T3i", self.fnma("KP185607687", "T3h", self.mul("KP155537010", "T3g")))
        V("T3l", self.fnma("KP158451106", "T3k", self.mul("KP183126246", "T3j")))
        V("T3m", self.add("T3i", "T3l"))
        V("T4q", self.sub("T3l", "T3i"))

        V("T3n", self.add("T3f", "T3m"))
        V("T4r", self.add("T4p", "T4q"))

        V("T4D", self.fma("KP228837560", "T3d", self.mul("KP079217082", "T3a")))
        V("T4E", self.fma("KP047564053", "T33", self.mul("KP237443964", "T36")))
        V("T4F", self.sub("T4D", "T4E"))
        V("T5V", self.add("T4E", "T4D"))
        V("T4G", self.fma("KP155537010", "T3h", self.mul("KP185607687", "T3g")))
        V("T4H", self.fma("KP183126246", "T3k", self.mul("KP158451106", "T3j")))
        V("T4I", self.add("T4G", "T4H"))
        V("T5W", self.sub("T4H", "T4G"))

        V("T4J", self.add("T4F", "T4I"))
        V("T5X", self.add("T5V", "T5W"))
        V("T3o", self.add("T30", "T3n"))
        V("T4K", self.add("T4C", "T4J"))
        V("T5Y", self.add("T5U", "T5X"))
        V("T4s", self.add("T4o", "T4r"))

        V("T5e", self.fnma("KP500000000", "T3n", "T30"))
        V("T5f", self.mul("KP866025403", self.sub("T4F", "T4I")))
        V("T5g", self.sub("T5e", "T5f"))
        V("T5F", self.add("T5e", "T5f"))
        V("T6g", self.fnma("KP500000000", "T4r", "T4o"))
        V("T6h", self.mul("KP866025403", self.sub("T5V", "T5W")))
        V("T6i", self.sub("T6g", "T6h"))
        V("T6H", self.add("T6g", "T6h"))

        V("T6j", self.fnma("KP500000000", "T5X", "T5U"))
        V("T6k", self.mul("KP866025403", self.sub("T4q", "T4p")))
        V("T6l", self.sub("T6j", "T6k"))
        V("T6G", self.add("T6k", "T6j"))
        V("T5h", self.fnma("KP500000000", "T4J", "T4C"))
        V("T5i", self.mul("KP866025403", self.sub("T3m", "T3f")))
        V("T5j", self.sub("T5h", "T5i"))
        V("T5E", self.add("T5i", "T5h"))

        V("T4W", self.mul("KP866025403", self.sub("T4K", "T4V")))
        V("T27", self.add("T1t", "T26"))
        V("T4i", self.add("T3o", "T4h"))
        V("T4z", self.fnma("KP500000000", "T4i", "T27"))
        STr(1,  self.add("T27", "T4i"))
        STr(7,  self.add("T4z", "T4W"))
        STr(11, self.sub("T4z", "T4W"))

        V("T4X", self.mul("KP866025403", self.sub("T4h", "T3o")))
        V("T58", self.add("T50", "T57"))
        V("T59", self.add("T4K", "T4V"))
        V("T5a", self.fnma("KP500000000", "T59", "T58"))
        STi(7,  self.add("T4X", "T5a"))
        STi(1,  self.add("T58", "T59"))
        STi(11, self.sub("T5a", "T4X"))

        V("T64", self.mul("KP866025403", self.sub("T5Y", "T63")))
        V("T4n", self.add("T4j", "T4m"))
        V("T4y", self.add("T4s", "T4x"))
        V("T5T", self.fnma("KP500000000", "T4y", "T4n"))
        STr(18, self.add("T4n", "T4y"))
        STr(12, self.add("T5T", "T64"))
        STr(8,  self.sub("T5T", "T64"))

        V("T6c", self.mul("KP866025403", self.sub("T4x", "T4s")))
        V("T69", self.sub("T65", "T68"))
        V("T6a", self.add("T5Y", "T63"))
        V("T6b", self.fnma("KP500000000", "T6a", "T69"))
        STi(8,  self.sub("T6b", "T6c"))
        STi(18, self.add("T69", "T6a"))
        STi(12, self.add("T6c", "T6b"))

        V("T5d", self.sub("T5b", "T5c"))
        V("T5w", self.add("T5u", "T5v"))

        V("T5k", self.fma("KP173648177", "T5g", self.mul("KP984807753", "T5j")))
        V("T5r", self.fnma("KP939692620", "T5q", self.mul("KP342020143", "T5n")))
        V("T5s", self.add("T5k", "T5r"))
        V("T5t", self.mul("KP866025403", self.sub("T5r", "T5k")))
        V("T5x", self.fnma("KP984807753", "T5g", self.mul("KP173648177", "T5j")))
        V("T5y", self.fma("KP939692620", "T5n", self.mul("KP342020143", "T5q")))
        V("T5z", self.sub("T5x", "T5y"))
        V("T5C", self.mul("KP866025403", self.add("T5x", "T5y")))

        STr(5,  self.add("T5d", "T5s"))
        STi(5,  self.add("T5w", "T5z"))
        V("T5A", self.fnma("KP500000000", "T5z", "T5w"))
        STi(16, self.add("T5t", "T5A"))
        STi(17, self.sub("T5A", "T5t"))
        V("T5B", self.fnma("KP500000000", "T5s", "T5d"))
        STr(17, self.sub("T5B", "T5C"))
        STr(16, self.add("T5B", "T5C"))

        V("T6f", self.add("T6d", "T6e"))
        V("T6B", self.sub("T6z", "T6A"))

        V("T6m", self.fma("KP173648177", "T6i", self.mul("KP984807753", "T6l")))
        V("T6t", self.fnma("KP939692620", "T6s", self.mul("KP342020143", "T6p")))
        V("T6u", self.add("T6m", "T6t"))
        V("T6E", self.mul("KP866025403", self.sub("T6t", "T6m")))
        V("T6w", self.fnma("KP984807753", "T6i", self.mul("KP173648177", "T6l")))
        V("T6x", self.fma("KP939692620", "T6p", self.mul("KP342020143", "T6s")))
        V("T6y", self.mul("KP866025403", self.add("T6w", "T6x")))
        V("T6C", self.sub("T6x", "T6w"))

        STr(14, self.add("T6f", "T6u"))
        STi(14, self.sub("T6B", "T6C"))
        V("T6v", self.fnma("KP500000000", "T6u", "T6f"))
        STr(2,  self.sub("T6v", "T6y"))
        STr(3,  self.add("T6v", "T6y"))
        V("T6D", self.fma("KP500000000", "T6C", "T6B"))
        STi(2,  self.sub("T6D", "T6E"))
        STi(3,  self.add("T6E", "T6D"))

        V("T6F", self.add("T6A", "T6z"))
        V("T6O", self.sub("T6d", "T6e"))

        V("T6I", self.fnma("KP642787609", "T6H", self.mul("KP766044443", "T6G")))
        V("T6L", self.fnma("KP984807753", "T6K", self.mul("KP173648177", "T6J")))
        V("T6M", self.add("T6I", "T6L"))
        V("T6N", self.mul("KP866025403", self.sub("T6I", "T6L")))
        V("T6P", self.fma("KP766044443", "T6H", self.mul("KP642787609", "T6G")))
        V("T6Q", self.fma("KP984807753", "T6J", self.mul("KP173648177", "T6K")))
        V("T6R", self.add("T6P", "T6Q"))
        V("T6U", self.mul("KP866025403", self.sub("T6Q", "T6P")))

        STi(10, self.add("T6F", "T6M"))
        STr(10, self.add("T6O", "T6R"))
        V("T6S", self.fnma("KP500000000", "T6R", "T6O"))
        STr(13, self.add("T6N", "T6S"))
        STr(15, self.sub("T6S", "T6N"))
        V("T6T", self.fnma("KP500000000", "T6M", "T6F"))
        STi(15, self.sub("T6T", "T6U"))
        STi(13, self.add("T6T", "T6U"))

        V("T5D", self.sub("T5v", "T5u"))
        V("T5L", self.add("T5b", "T5c"))

        V("T5G", self.fnma("KP642787609", "T5F", self.mul("KP766044443", "T5E")))
        V("T5J", self.fnma("KP984807753", "T5I", self.mul("KP173648177", "T5H")))
        V("T5K", self.add("T5G", "T5J"))
        V("T5Q", self.mul("KP866025403", self.sub("T5G", "T5J")))
        V("T5M", self.fma("KP766044443", "T5F", self.mul("KP642787609", "T5E")))
        V("T5N", self.fma("KP984807753", "T5H", self.mul("KP173648177", "T5I")))
        V("T5O", self.add("T5M", "T5N"))
        V("T5S", self.mul("KP866025403", self.sub("T5N", "T5M")))

        STi(9,  self.add("T5D", "T5K"))
        STr(9,  self.add("T5L", "T5O"))
        V("T5P", self.fnma("KP500000000", "T5O", "T5L"))
        STr(4,  self.sub("T5P", "T5Q"))
        STr(6,  self.add("T5Q", "T5P"))
        V("T5R", self.fnma("KP500000000", "T5K", "T5D"))
        STi(4,  self.sub("T5R", "T5S"))
        STi(6,  self.add("T5R", "T5S"))


# ================================================================
# HELPERS: constants
# ================================================================

DFT19_CONSTS = [
    ('KP179300334',   KP179300334_val),
    ('KP162767826',   KP162767826_val),
    ('KP241806419',   KP241806419_val),
    ('KP013100793',   KP013100793_val),
    ('KP185607687',   KP185607687_val),
    ('KP155537010',   KP155537010_val),
    ('KP158451106',   KP158451106_val),
    ('KP183126246',   KP183126246_val),
    ('KP074045235',   KP074045235_val),
    ('KP230562959',   KP230562959_val),
    ('KP241001675',   KP241001675_val),
    ('KP023667861',   KP023667861_val),
    ('KP055555555',   KP055555555_val),
    ('KP237443964',   KP237443964_val),
    ('KP047564053',   KP047564053_val),
    ('KP079217082',   KP079217082_val),
    ('KP228837560',   KP228837560_val),
    ('KP642787609',   KP642787609_val),
    ('KP766044443',   KP766044443_val),
    ('KP939692620',   KP939692620_val),
    ('KP342020143',   KP342020143_val),
    ('KP984807753',   KP984807753_val),
    ('KP173648177',   KP173648177_val),
    ('KP242161052',   KP242161052_val),
    ('KP500000000',   KP500000000_val),
    ('KP866025403',   KP866025403_val),
]


def emit_dft19_constants(em):
    """Emit the 26 DFT-19 constants + sign_flip as SIMD broadcasts or scalars."""
    T = em.isa.T
    if em.isa.name == 'scalar':
        for name, val in DFT19_CONSTS:
            em.o(f"const double {name} = {val:+.45f};")
    else:
        set1 = f"{em.isa.p}_set1_pd"
        em.o(f"const {T} sign_flip = {set1}(-0.0);")
        for name, val in DFT19_CONSTS:
            em.o(f"const {T} {name} = {set1}({val:+.45f});")
        em.o(f"(void)sign_flip;  /* used by neg() */")
    em.b()


def emit_dft19_constants_raw(lines, isa, indent=1):
    """Append DFT-19 constant declarations to a line list."""
    pad = "    " * indent
    T = isa.T
    if isa.name == 'scalar':
        for name, val in DFT19_CONSTS:
            lines.append(f"{pad}const double {name} = {val:+.45f};")
    else:
        set1 = f"{isa.p}_set1_pd"
        lines.append(f"{pad}const {T} sign_flip = {set1}(-0.0);")
        for name, val in DFT19_CONSTS:
            lines.append(f"{pad}const {T} {name} = {set1}({val:+.45f});")
        lines.append(f"{pad}(void)sign_flip;")


# ================================================================
# KERNEL BODY EMITTERS
# ================================================================

def emit_kernel_body(em, d, variant):
    """Emit the inner loop body for notw, dit_tw, dif_tw."""
    T = em.isa.T

    # Load all 19 inputs
    for n in range(R):
        em.emit_load(f"x{n}", n)

    # DIT: twiddle inputs 1..18 before butterfly
    if variant == 'dit_tw':
        for n in range(1, R):
            em.emit_ext_tw(f"x{n}", n - 1, d)
    elif variant == 'dit_tw_scalar':
        for n in range(1, R):
            em.emit_ext_tw_scalar(f"x{n}", n - 1, d)

    em.b()
    em.emit_radix19_butterfly(d)
    em.b()

    # DIF: twiddle outputs 1..18 after butterfly
    if variant == 'dif_tw':
        for m in range(1, R):
            em.emit_ext_tw(f"x{m}", m - 1, d)
    elif variant == 'dif_tw_scalar':
        for m in range(1, R):
            em.emit_ext_tw_scalar(f"x{m}", m - 1, d)

    # Store all 19 outputs
    for m in range(R):
        em.emit_store(f"x{m}", m)


def emit_kernel_body_log3(em, d, variant):
    """Emit the log3 variant: derive w2..w18 from w1 (18 cmuls)."""
    T = em.isa.T
    is_dit = variant == 'dit_tw_log3'

    em.c("Load base twiddle w1, derive w2..w18 (18 cmuls)")
    tb, tbi = em._tw_buf(), em._tw_buf_im()
    ta = em._tw_addr(0)
    if em.isa.name == 'scalar':
        em.o(f"const double w1r = {tb}[{ta}], w1i = {tbi}[{ta}];")
    else:
        em.o(f"const {T} w1r = LD(&{tb}[{ta}]), w1i = LD(&{tbi}[{ta}]);")
    em.n_load += 2

    # Chain: w2=w1^2, w3=w1*w2, w4=w2^2, w5=w2*w3, w6=w3^2,
    #        w7=w3*w4, w8=w4^2, w9=w4*w5, w10=w5^2, w11=w5*w6,
    #        w12=w6^2, w13=w6*w7, w14=w7^2, w15=w7*w8, w16=w8^2,
    #        w17=w8*w9, w18=w9^2
    chain = [
        (2,  "w1r","w1i",  "w1r","w1i"),   # w2 = w1^2
        (3,  "w1r","w1i",  "w2r","w2i"),   # w3 = w1*w2
        (4,  "w2r","w2i",  "w2r","w2i"),   # w4 = w2^2
        (5,  "w2r","w2i",  "w3r","w3i"),   # w5 = w2*w3
        (6,  "w3r","w3i",  "w3r","w3i"),   # w6 = w3^2
        (7,  "w3r","w3i",  "w4r","w4i"),   # w7 = w3*w4
        (8,  "w4r","w4i",  "w4r","w4i"),   # w8 = w4^2
        (9,  "w4r","w4i",  "w5r","w5i"),   # w9 = w4*w5
        (10, "w5r","w5i",  "w5r","w5i"),   # w10= w5^2
        (11, "w5r","w5i",  "w6r","w6i"),   # w11= w5*w6
        (12, "w6r","w6i",  "w6r","w6i"),   # w12= w6^2
        (13, "w6r","w6i",  "w7r","w7i"),   # w13= w6*w7
        (14, "w7r","w7i",  "w7r","w7i"),   # w14= w7^2
        (15, "w7r","w7i",  "w8r","w8i"),   # w15= w7*w8
        (16, "w8r","w8i",  "w8r","w8i"),   # w16= w8^2
        (17, "w8r","w8i",  "w9r","w9i"),   # w17= w8*w9
        (18, "w9r","w9i",  "w9r","w9i"),   # w18= w9^2
    ]
    for (wn, ar, ai, br, bi) in chain:
        em.o(f"{T} w{wn}r, w{wn}i;")
        em.emit_cmul(f"w{wn}r", f"w{wn}i", ar, ai, br, bi, 'fwd')
    em.b()

    # Load inputs
    for n in range(R):
        em.emit_load(f"x{n}", n)

    # DIT: apply twiddles to x1..x18
    if is_dit:
        for n in range(1, R):
            em.emit_cmul_inplace(f"x{n}", f"w{n}r", f"w{n}i", d)
    em.b()

    em.emit_radix19_butterfly(d)
    em.b()

    # DIF: apply twiddles to outputs x1..x18
    if not is_dit:
        for m in range(1, R):
            em.emit_cmul_inplace(f"x{m}", f"w{m}r", f"w{m}i", d)

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
        func_base = 'radix19_n1_dit_kernel'
    elif variant == 'dit_tw':
        func_base = 'radix19_tw_flat_dit_kernel'
        tw_params = 'flat'
    elif variant == 'dit_tw_scalar':
        func_base = 'radix19_tw_flat_dit_kernel'
        tw_params = 'flat'
    elif variant == 'dif_tw':
        func_base = 'radix19_tw_flat_dif_kernel'
        tw_params = 'flat'
    elif variant == 'dif_tw_scalar':
        func_base = 'radix19_tw_flat_dif_kernel'
        tw_params = 'flat'
    elif variant == 'dit_tw_log3':
        func_base = 'radix19_tw_log3_dit_kernel'
        tw_params = 'flat'
    elif variant == 'dif_tw_log3':
        func_base = 'radix19_tw_log3_dif_kernel'
        tw_params = 'flat'

    vname = {
        'notw':        'N1 (no twiddle)',
        'dit_tw':      'DIT twiddled (flat)',
        'dif_tw':      'DIF twiddled (flat)',
        'dit_tw_log3': 'DIT twiddled (log3 derived)',
        'dif_tw_log3': 'DIF twiddled (log3 derived)',
    }[variant]
    guard = f"FFT_RADIX19_{isa.name.upper()}_{variant.upper()}_H"

    em.L.append(f"/**")
    em.L.append(f" * @file fft_radix19_{isa.name}_{variant}.h")
    em.L.append(f" * @brief DFT-19 {isa.name.upper()} {vname}")
    em.L.append(f" *")
    em.L.append(f" * Monolithic DFT-19 butterfly (genfft DAG), 26 constants.")
    em.L.append(f" * Register pressure exceeds both AVX2 and AVX-512;")
    em.L.append(f" * compiler handles register allocation / spills.")
    em.L.append(f" * k-step={isa.k_step}")
    em.L.append(f" * Generated by gen_radix19.py")
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
        emit_dft19_constants(em)

        # Working registers for all 19 inputs/outputs
        em.o(f"{T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
        em.o(f"{T} x4_re,x4_im,x5_re,x5_im,x6_re,x6_im,x7_re,x7_im;")
        em.o(f"{T} x8_re,x8_im,x9_re,x9_im,x10_re,x10_im,x11_re,x11_im;")
        em.o(f"{T} x12_re,x12_im,x13_re,x13_im,x14_re,x14_im,x15_re,x15_im;")
        em.o(f"{T} x16_re,x16_im,x17_re,x17_im,x18_re,x18_im;")
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

    # No U=2 for R=19: register pressure exceeds AVX-512 (32 ZMM).

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

    is_n1     = ct_variant == 'ct_n1'
    is_n1_scaled = ct_variant == 'ct_n1_scaled'
    is_t1_dit = ct_variant == 'ct_t1_dit'
    is_t1s_dit = ct_variant == 'ct_t1s_dit'
    is_t1_dit_log3 = ct_variant == 'ct_t1_dit_log3'
    is_t1_dif = ct_variant == 'ct_t1_dif'
    is_t1_oop_dit = ct_variant == 'ct_t1_oop_dit'
    em.addr_mode = 'n1' if (is_n1 or is_n1_scaled) else ('t1_oop' if is_t1_oop_dit else 't1')

    if is_n1:
        func_base = "radix19_n1"
        vname = "n1 (separate is/os)"
    elif is_n1_scaled:
        func_base = "radix19_n1_scaled"
        vname = "n1_scaled (separate is/os, output *= scale)"
    elif is_t1_oop_dit:
        func_base = "radix19_t1_oop_dit"
        vname = "t1_oop DIT (out-of-place, separate is/os, with twiddle)"
    elif is_t1_dif:
        func_base = "radix19_t1_dif"
        vname = "t1 DIF (in-place twiddle)"
    elif is_t1s_dit:
        func_base = "radix19_t1s_dit"
        vname = "t1s DIT (in-place, scalar broadcast twiddle)"
    elif is_t1_dit_log3:
        func_base = "radix19_t1_dit_log3"
        vname = "t1 DIT log3 (in-place, derived twiddles)"
    else:
        func_base = "radix19_t1_dit"
        vname = "t1 DIT (in-place twiddle)"

    guard = f"FFT_RADIX19_{isa.name.upper()}_CT_{ct_variant.upper()}_H"

    em.L.append(f"/**")
    em.L.append(f" * @file fft_radix19_{isa.name}_{ct_variant}.h")
    em.L.append(f" * @brief DFT-19 {isa.name.upper()} {vname}")
    em.L.append(f" *")
    em.L.append(f" * FFTW-style codelet for recursive CT executor.")
    em.L.append(f" * Generated by gen_radix19.py --variant {ct_variant}")
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

        emit_dft19_constants(em)

        em.o(f"{T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
        em.o(f"{T} x4_re,x4_im,x5_re,x5_im,x6_re,x6_im,x7_re,x7_im;")
        em.o(f"{T} x8_re,x8_im,x9_re,x9_im,x10_re,x10_im,x11_re,x11_im;")
        em.o(f"{T} x12_re,x12_im,x13_re,x13_im,x14_re,x14_im,x15_re,x15_im;")
        em.o(f"{T} x16_re,x16_im,x17_re,x17_im,x18_re,x18_im;")
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

    # n1_ovs: butterfly with fused SIMD transpose stores (for R=19)
    # 19 bins = 4 groups of 4 (bins 0-3, 4-7, 8-11, 12-15) + bins 16,17,18 leftover
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
            em.L.append(f"radix19_n1_ovs_{d}_{isa.name}(")
            em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
            em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
            em.L.append(f"    size_t is, size_t os, size_t vl, size_t ovs)")
            em.L.append(f"{{")

            em.L.append(f"    /* n1_ovs: butterfly -> tbuf, then 4x4 transpose (bins 0-3..12-15) + scatter (bins 16,17,18) */")
            em.L.append(f"    {isa.align} double tbuf_re[{R * VL}];")
            em.L.append(f"    {isa.align} double tbuf_im[{R * VL}];")

            emit_dft19_constants_raw(em.L, isa, indent=1)

            em.L.append(f"    {T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
            em.L.append(f"    {T} x4_re,x4_im,x5_re,x5_im,x6_re,x6_im,x7_re,x7_im;")
            em.L.append(f"    {T} x8_re,x8_im,x9_re,x9_im,x10_re,x10_im,x11_re,x11_im;")
            em.L.append(f"    {T} x12_re,x12_im,x13_re,x13_im,x14_re,x14_im,x15_re,x15_im;")
            em.L.append(f"    {T} x16_re,x16_im,x17_re,x17_im,x18_re,x18_im;")
            em.L.append(f"")

            em.L.append(f"    for (size_t k = 0; k < vl; k += {isa.k_step}) {{")

            # Use fresh Emitter for kernel body
            em2.L = []
            em2.ind = 2
            em2.reset()
            em2.addr_mode = 'n1_ovs'
            emit_kernel_body(em2, d, 'notw')
            em.L.extend(em2.L)

            # 4 groups of 4 bins
            for grp in range(4):
                base = grp * 4
                em.L.append(f"        /* 4x4 transpose: bins {base}-{base+3} -> output at stride ovs */")
                for comp, arr in [('re', 'out_re'), ('im', 'out_im')]:
                    bname = f"tbuf_{comp}"
                    em.L.append(f"        {{ {T} a_=LD(&{bname}[{base+0}*{VL}]), b_=LD(&{bname}[{base+1}*{VL}]);")
                    em.L.append(f"          {T} c_=LD(&{bname}[{base+2}*{VL}]), d_=LD(&{bname}[{base+3}*{VL}]);")
                    if isa.name == 'avx2':
                        em.L.append(f"          {T} lo_ab=_mm256_unpacklo_pd(a_,b_), hi_ab=_mm256_unpackhi_pd(a_,b_);")
                        em.L.append(f"          {T} lo_cd=_mm256_unpacklo_pd(c_,d_), hi_cd=_mm256_unpackhi_pd(c_,d_);")
                        em.L.append(f"          _mm256_storeu_pd(&{arr}[(k+0)*ovs+os*{base+0}], _mm256_permute2f128_pd(lo_ab,lo_cd,0x20));")
                        em.L.append(f"          _mm256_storeu_pd(&{arr}[(k+1)*ovs+os*{base+0}], _mm256_permute2f128_pd(hi_ab,hi_cd,0x20));")
                        em.L.append(f"          _mm256_storeu_pd(&{arr}[(k+2)*ovs+os*{base+0}], _mm256_permute2f128_pd(lo_ab,lo_cd,0x31));")
                        em.L.append(f"          _mm256_storeu_pd(&{arr}[(k+3)*ovs+os*{base+0}], _mm256_permute2f128_pd(hi_ab,hi_cd,0x31));")
                    else:  # avx512
                        for j in range(VL):
                            for b in range(4):
                                em.L.append(f"          {arr}[(k+{j})*ovs+os*{base+b}] = {bname}[{base+b}*{VL}+{j}];")
                    em.L.append(f"        }}")

            # Bins 16, 17, 18: extract from tbuf -> scatter
            for sbin in [16, 17, 18]:
                em.L.append(f"        /* Bin {sbin}: extract from tbuf -> scatter */")
                if isa.name == 'avx2':
                    for comp, arr in [('re', 'out_re'), ('im', 'out_im')]:
                        bname = f"tbuf_{comp}"
                        em.L.append(f"        {{ __m256d v=LD(&{bname}[{sbin}*{VL}]);")
                        em.L.append(f"          __m128d lo=_mm256_castpd256_pd128(v), hi=_mm256_extractf128_pd(v,1);")
                        em.L.append(f"          _mm_storel_pd(&{arr}[(k+0)*ovs+os*{sbin}], lo);")
                        em.L.append(f"          _mm_storeh_pd(&{arr}[(k+1)*ovs+os*{sbin}], lo);")
                        em.L.append(f"          _mm_storel_pd(&{arr}[(k+2)*ovs+os*{sbin}], hi);")
                        em.L.append(f"          _mm_storeh_pd(&{arr}[(k+3)*ovs+os*{sbin}], hi); }}")
                else:  # avx512
                    for comp, arr in [('re', 'out_re'), ('im', 'out_im')]:
                        bname = f"tbuf_{comp}"
                        for j in range(VL):
                            em.L.append(f"        {arr}[(k+{j})*ovs+os*{sbin}] = {bname}[{sbin}*{VL}+{j}];")

            em.L.append(f"    }}")
            em.L.append(f"}}")

    em.L.append(f"")
    em.L.append(f"#undef LD")
    em.L.append(f"#undef ST")
    em.L.append(f"")
    em.L.append(f"#endif /* {guard} */")

    return em.L


# ================================================================
# SV CODELET GENERATION -- text transform from K-loop output
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
        t2_pattern = 'radix19_n1_dit_kernel'
        sv_name = 'radix19_n1sv_kernel'
    elif variant == 'dit_tw':
        t2_pattern = 'radix19_tw_flat_dit_kernel'
        sv_name = 'radix19_t1sv_dit_kernel'
    elif variant == 'dit_tw_scalar':
        t2_pattern = 'radix19_tw_flat_dit_kernel'
        sv_name = 'radix19_t1sv_dit_kernel'
    elif variant == 'dif_tw':
        t2_pattern = 'radix19_tw_flat_dif_kernel'
        sv_name = 'radix19_t1sv_dif_kernel'
    elif variant == 'dif_tw_scalar':
        t2_pattern = 'radix19_tw_flat_dif_kernel'
        sv_name = 'radix19_t1sv_dif_kernel'
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
    sep = '-' * 20
    s5 = '-' * 5; s3 = '-' * 3; s4 = '-' * 4
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
    parser = argparse.ArgumentParser(description='Unified R=19 codelet generator')
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
    ct_variants  = ['ct_n1', 'ct_n1_scaled', 'ct_t1_dit', 'ct_t1s_dit', 'ct_t1_dit_log3', 'ct_t1_dif', 'ct_t1_oop_dit']

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
    pass  # see framework block below


# ═══════════════════════════════════════════════════════════════
# Framework integration (autotuner-visible surface)
# ═══════════════════════════════════════════════════════════════
#
# R=19 is a prime radix with a monolithic 26-constant butterfly
# (genfft DAG, 542 ops = 314 add + 114 mul + 114 FMA). Register pressure
# grossly exceeds both AVX2 (16) and AVX-512 (32) at U=1: 19 inputs × 2
# re/im + 26 constants + sign_flip ≈ 64+ registers. U=N unrolling
# categorically infeasible; compiler handles spill via Sethi-Ullman
# scheduling and store-to-load forwarding.
#
# Heaviest prime butterfly in the portfolio.

VARIANTS = {
    'ct_t1_dit':      ('radix19_t1_dit',      'flat', 't1_dit'),
    'ct_t1s_dit':     ('radix19_t1s_dit',     't1s',  't1s_dit'),
    'ct_t1_dit_log3': ('radix19_t1_dit_log3', 'log3', 't1_dit_log3'),
}


def function_name(variant_id, isa, direction):
    if variant_id not in VARIANTS:
        raise KeyError(f"unknown variant {variant_id}; known: {list(VARIANTS)}")
    base, _, _ = VARIANTS[variant_id]
    return f'{base}_{direction}_{isa}'

def protocol(variant_id):
    return VARIANTS[variant_id][1]

def dispatcher(variant_id):
    return VARIANTS[variant_id][2]


def _emit_all_variants(isa_name):
    """Emit a single header containing all R=19 variants for the given ISA."""
    isa = ALL_ISA[isa_name]
    guard = f'FFT_RADIX19_{isa_name.upper()}_H'
    out = []
    out.append(f'/**')
    out.append(f' * @file fft_radix19_{isa_name}.h')
    out.append(f' * @brief DFT-19 {isa_name.upper()} codelets (flat, t1s, log3 × U=1)')
    out.append(f' *')
    out.append(f' * Prime-direct monolithic butterfly (26 constants, 542 ops from')
    out.append(f' * genfft DAG: 314 add + 114 mul + 114 FMA). Register pressure')
    out.append(f' * grossly exceeds both AVX2 (16 YMM) and AVX-512 (32 ZMM) at U=1.')
    out.append(f' * Heaviest prime butterfly in the portfolio.')
    out.append(f' * Compiler handles via Sethi-Ullman scheduling; U=N infeasible.')
    out.append(f' */')
    out.append(f'#ifndef {guard}')
    out.append(f'#define {guard}')
    out.append('')
    if isa_name != 'scalar':
        out.append('#include <immintrin.h>')
    out.append('#include <stddef.h>')
    out.append('')

    def _append_cleaned(lines):
        for L in lines:
            s = L.strip()
            if s.startswith('#ifndef FFT_RADIX19_'): continue
            if s.startswith('#define FFT_RADIX19_'): continue
            if s.startswith('#endif /* FFT_RADIX19_'): continue
            if s.startswith('#include <immintrin.h>'): continue
            if s.startswith('#include <stddef.h>'): continue
            out.append(L)
        out.append('')

    for v in ['ct_t1_dit', 'ct_t1s_dit', 'ct_t1_dit_log3']:
        lines = emit_file_ct(isa, v)
        _append_cleaned(lines)

    out.append(f'#endif /* {guard} */')
    return '\n'.join(out)


if __name__ == '__main__':
    import argparse as _argparse
    _ap = _argparse.ArgumentParser(add_help=False)
    _ap.add_argument('--isa', choices=list(ALL_ISA), default=None)
    _ap.add_argument('--variant', default=None)
    _ap.add_argument('--list-variants', action='store_true')
    _ap.add_argument('--list-name', action='store_true')
    _ap.add_argument('--list-protocol', action='store_true')
    _ap.add_argument('--direction', choices=['fwd', 'bwd'], default='fwd')
    _args, _unknown = _ap.parse_known_args()

    _bench_mode = (
        _args.list_variants or _args.list_name or _args.list_protocol or
        (_args.isa is not None and _args.variant is None and not _unknown)
    )

    if _args.list_variants:
        for v in VARIANTS:
            print(v)
        sys.exit(0)
    if _args.list_protocol:
        if not _args.variant:
            print('--list-protocol requires --variant', file=sys.stderr); sys.exit(2)
        print(protocol(_args.variant)); sys.exit(0)
    if _args.list_name:
        if not _args.variant or not _args.isa:
            print('--list-name requires --variant and --isa', file=sys.stderr); sys.exit(2)
        print(function_name(_args.variant, _args.isa, _args.direction)); sys.exit(0)
    if _bench_mode and _args.isa is not None:
        print(_emit_all_variants(_args.isa))
        sys.exit(0)
    main()
