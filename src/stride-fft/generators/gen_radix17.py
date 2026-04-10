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
gen_radix17.py -- Unified DFT-17 codelet generator for VectorFFT

Monolithic DFT-17 butterfly (straight-line genfft DAG translation).
23 constants, 366 ops (250 add + 70 mul + 46 FMA).

Register pressure WAY exceeds both AVX2 (16) and AVX-512 (32):
  17 inputs(34 regs) + 23 constants = 57+
Both ISAs use the same butterfly (no separate no-spill path).
The genfft DAG interleaves loads with computation, so we emit the
entire DAG as one straight-line block and let the C compiler handle
register allocation / spills.  All butterfly temporaries are wrapped
in a single large { } scope block.

DFT-17 constants (23, from genfft / fft_radix17_genfft.h):
  R17_K00..R17_K22

Log3 twiddle for R=17 (15 cmuls from w1):
  w2=w1^2, w3=w1*w2, w4=w2^2, w5=w2*w3, w6=w3^2,
  w7=w3*w4, w8=w4^2, w9=w4*w5, w10=w5^2, w11=w5*w6,
  w12=w6^2, w13=w6*w7, w14=w7^2, w15=w7*w8, w16=w8^2

No U=2 -- register pressure too high (same reason as R=13).

n1_ovs: 17 bins = 4 groups of 4 (bins 0-3, 4-7, 8-11, 12-15)
         + bin 16 leftover (extract+scatter)

Usage:
  python3 gen_radix17.py --isa avx2 --variant all
  python3 gen_radix17.py --isa all --variant ct_n1
  python3 gen_radix17.py --isa avx2 --variant ct_t1_dit
"""

import sys, math, argparse, re

R = 17

# ----------------------------------------------------------------
# DFT-17 constants (from fft_radix17_genfft.h lines 16-38)
# ----------------------------------------------------------------
KP382683432_val = +0.382683432365089771728459984030398866761344562
KP923879532_val = +0.923879532511286756128183189396788286822416626
KP039070284_val = +0.039070284615346466562699756048062282446982333
KP254715062_val = +0.254715062098957589711663291776992207183525060
KP251015702_val = +0.251015702549784655763030347279630555416044016
KP058286937_val = +0.058286937416869254028445865957004944101575445
KP212806139_val = +0.212806139341024484918843854987067623430668802
KP145326518_val = +0.145326518773307409751646882717193568490332259
KP035884665_val = +0.035884665796620944495119501392579193335899154
KP255183347_val = +0.255183347341994130592803084730480096912472273
KP062500000_val = +0.062500000000000000000000000000000000000000000
KP2_000000000_val = +2.000000000000000000000000000000000000000000000
KP1_414213562_val = +1.414213562373095048801688724209698078569671875
KP173544463_val = +0.173544463881753015525503524818483071996551637
KP190495588_val = +0.190495588022386815242151713446857544100600462
KP256951040_val = +0.256951040344387208026117263863478633468108486
KP019555379_val = +0.019555379462876637091214482719906682374292162
KP406231784_val = +0.406231784453331486292527922969716525973112504
KP317176192_val = +0.317176192832725115542545383489655119241102579
KP191341716_val = +0.191341716182544885864229992015199433380672281
KP461939766_val = +0.461939766255643378064091594698394143411208313
KP257694101_val = +0.257694101601103784363838115998379814071699952
KP707106781_val = +0.707106781186547524400844362104849039284835938

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

ISA_SCALAR = ISAConfig('scalar', 'double', 1, 1, '', 1, '', '', 'R17S')
ISA_AVX2   = ISAConfig('avx2', '__m256d', 4, 4, '_mm256', 4,
    '__attribute__((target("avx2,fma")))',
    '__attribute__((aligned(32)))', 'R17A')
ISA_AVX512 = ISAConfig('avx512', '__m512d', 8, 8, '_mm512', 8,
    '__attribute__((target("avx512f,avx512dq,fma")))',
    '__attribute__((aligned(64)))', 'R17L')

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
    # DFT-17 butterfly
    # ------------------------------------------------------------------

    def emit_radix17_butterfly(self, d):
        """Emit DFT-17 butterfly, translating genfft DAG exactly.

        R=17 register pressure exceeds both AVX2 and AVX-512.
        Both ISAs emit the DAG as one straight-line block inside a { } scope
        and let the C compiler handle register allocation / spills.

        The butterfly body is a direct port of R17_BUTTERFLY_BODY from
        fft_radix17_genfft.h, using the split-complex layout (x0_re/x0_im..
        x16_re/x16_im) and Emitter arithmetic methods.

        For the backward direction we swap real/imag inputs/outputs
        (same as the production genfft.h: bwd calls fwd(ii,ri,io,ro)).
        """
        fwd = (d == 'fwd')
        T = self.isa.T

        self.c(f"DFT-17 butterfly [{d}] (straight-line genfft DAG, compiler-managed regs)")
        self.o("{")
        self.ind += 1

        if fwd:
            self._emit_r17_fwd()
        else:
            self._emit_r17_bwd()

        self.ind -= 1
        self.o("}")

    # ----------------------------------------------------------------
    # Forward butterfly  (direct port of R17_BUTTERFLY_BODY)
    # The macro body from fft_radix17_genfft.h is the single source of
    # truth.  Each "V Txx;" becomes a local T variable; each macro op
    # becomes the corresponding Emitter call.
    #
    # Naming convention: macro var T1 -> Python var T1 (prefix-free),
    # emitted as C variable names T1, T49, TQ, ... directly (no _re/_im
    # suffix since the macro treats real and imaginary separately via
    # distinct LD/ST calls).  We use the same letter names but prefix with
    # ri/ii (real input / imag input) tracking:
    #
    #   LD(&ri[n*K+k]) -> x{n}_re  (from caller's already-loaded vars)
    #   LD(&ii[n*K+k]) -> x{n}_im
    #   ST(&ro[n*K+k], v) -> emit_store for re component n
    #   ST(&io[n*K+k], v) -> emit_store for im component n
    #
    # All intermediate variables are emitted as "const T name = expr;"
    # inside a single {} scope block.
    # ----------------------------------------------------------------

    def _emit_r17_fwd(self):
        T = self.isa.T

        # Helper: emit "const T name = expr;" (flat, no inner {})
        def V(name, expr):
            self.o(f"const {T} {name} = {expr};")

        # Store outputs back to x{n}_re / x{n}_im (caller does emit_store later)
        def STr(n, expr):
            self.o(f"x{n}_re = {expr};")

        def STi(n, expr):
            self.o(f"x{n}_im = {expr};")

        # Shorthand for already-loaded inputs
        def xr(n): return f"x{n}_re"
        def xi(n): return f"x{n}_im"

        # ----------------------------------------------------------------
        # Flat translation of R17_BUTTERFLY_BODY (no inner {} scopes).
        # All const T variables live in the enclosing butterfly {} scope.
        # Order follows the genfft DAG exactly so every var is defined
        # before it is used.
        # ----------------------------------------------------------------

        # DC inputs
        V("T1",  xr(0))
        V("T49", xi(0))

        # Conjugate pair (1,16) and (13,4): sums/diffs
        V("T2",  xr(1));  V("T3",  xr(16))
        V("T4",  self.add("T2","T3"))
        V("T5",  xr(13)); V("T6",  xr(4))
        V("T7",  self.add("T5","T6"))
        V("T8",  self.add("T4","T7"))
        V("T2G", self.sub("T2","T3"))
        V("T2B", self.sub("T5","T6"))

        # Imag inputs for same pairs
        V("T1l", xi(13)); V("T1m", xi(4))
        V("T3Q", self.add("T1l","T1m"))
        V("T1x", xi(1));  V("T1y", xi(16))
        V("T3P", self.add("T1x","T1y"))
        V("T1n", self.sub("T1l","T1m"))
        V("T41", self.add("T3P","T3Q"))
        V("T1z", self.sub("T1x","T1y"))

        # Pair (9,2) and (8,15): real
        V("T9",  xr(9));  V("Ta",  xr(2))
        V("Tb",  self.add("T9","Ta"))
        V("TA",  self.sub("T9","Ta"))
        V("Tc",  xr(8));  V("Td",  xr(15))
        V("Te",  self.add("Tc","Td"))
        V("TB",  self.sub("Tc","Td"))

        V("Tf",  self.add("Tb","Te"))
        V("T2H", self.mul("KP707106781", self.sub("Tb","Te")))
        V("T2C", self.mul("KP707106781", self.sub("TA","TB")))

        # Pair (9,2) and (8,15): imag
        V("T1o", xi(9));  V("T1p", xi(2))
        V("T1q", self.sub("T1o","T1p"))
        V("T1A", self.add("T1o","T1p"))
        V("T1r", xi(8));  V("T1s", xi(15))
        V("T1t", self.sub("T1r","T1s"))
        V("T1B", self.add("T1r","T1s"))

        V("T1u", self.mul("KP707106781", self.sub("T1q","T1t")))
        V("T42", self.add("T1A","T1B"))
        V("T1C", self.mul("KP707106781", self.sub("T1A","T1B")))

        # Pair (5,7): real
        V("To",  xr(5));  V("Tp",  xr(7))
        V("Tq",  self.add("To","Tp"))
        V("TG",  self.sub("To","Tp"))

        # Pair (14,11): imag
        V("T1b", xi(14)); V("T1c", xi(11))
        V("T1d", self.sub("T1b","T1c"))
        V("T1G", self.add("T1b","T1c"))

        # Pair (12,10): real
        V("Tr",  xr(12)); V("Ts",  xr(10))
        V("Tt",  self.add("Tr","Ts"))
        V("TH",  self.sub("Tr","Ts"))

        # Pair (3,6): imag
        V("T18", xi(3));  V("T19", xi(6))
        V("T1a", self.sub("T18","T19"))
        V("T1F", self.add("T18","T19"))

        V("Tu",  self.add("Tq","Tt"))
        V("TI",  self.add("TG","TH"))
        V("T3S", self.add("T1F","T1G"))
        V("T3K", self.add("T1a","T1d"))

        V("T1E", self.sub("Tq","Tt"))
        V("T1H", self.sub("T1F","T1G"))
        V("T1I", self.add("T1E","T1H"))
        V("T1P", self.sub("T1H","T1E"))
        V("T17", self.sub("TG","TH"))
        V("T1e", self.sub("T1a","T1d"))
        V("T1f", self.add("T17","T1e"))
        V("T1h", self.sub("T17","T1e"))

        # Pair (3,6): real
        V("Th",  xr(3));  V("Ti",  xr(6))
        V("Tj",  self.add("Th","Ti"))
        V("TD",  self.sub("Th","Ti"))

        # Pair (12,10): imag
        V("T12", xi(12)); V("T13", xi(10))
        V("T14", self.sub("T12","T13"))
        V("T1L", self.add("T12","T13"))

        # Pair (14,11): real
        V("Tk",  xr(14)); V("Tl",  xr(11))
        V("Tm",  self.add("Tk","Tl"))
        V("TE",  self.sub("Tk","Tl"))

        # Pair (5,7): imag
        V("TZ",  xi(5));  V("T10", xi(7))
        V("T11", self.sub("TZ","T10"))
        V("T1K", self.add("TZ","T10"))

        V("Tn",  self.add("Tj","Tm"))
        V("TF",  self.add("TD","TE"))
        V("T3T", self.add("T1K","T1L"))
        V("T3L", self.add("T11","T14"))

        V("T1J", self.sub("Tj","Tm"))
        V("T1M", self.sub("T1K","T1L"))
        V("T1N", self.sub("T1J","T1M"))
        V("T1Q", self.add("T1J","T1M"))
        V("TY",  self.sub("TD","TE"))
        V("T15", self.sub("T11","T14"))
        V("T16", self.sub("TY","T15"))
        V("T1i", self.add("TY","T15"))

        # Outer-scope combination
        V("T46", self.add("T41","T42"))
        V("T47", self.add("T3S","T3T"))
        V("T48", self.mul("KP257694101", self.sub("T46","T47")))
        V("T4a", self.add("T46","T47"))

        V("TQ",  self.sub("T8","Tf"))
        V("T44", self.add("T3K","T3L"))
        V("T43", self.sub("T41","T42"))
        V("TR",  self.add("TF","TI"))

        V("Tg",  self.add("T8","Tf"))
        V("Tv",  self.add("Tn","Tu"))
        V("Tw",  self.add("Tg","Tv"))
        V("TT",  self.mul("KP257694101", self.sub("Tg","Tv")))
        V("Tx",  self.sub("T4","T7"))
        V("Ty",  self.mul("KP707106781", self.sub("Tn","Tu")))
        V("Tz",  self.sub("Tx","Ty"))
        V("TM",  self.add("Tx","Ty"))

        V("T3R", self.sub("T3P","T3Q"))
        V("T3U", self.mul("KP707106781", self.sub("T3S","T3T")))
        V("T3V", self.sub("T3R","T3U"))
        V("T3X", self.add("T3R","T3U"))

        V("TC",  self.add("TA","TB"))
        V("TJ",  self.mul("KP707106781", self.sub("TF","TI")))
        V("TK",  self.sub("TC","TJ"))
        V("TN",  self.add("TC","TJ"))
        V("T3M", self.mul("KP707106781", self.sub("T3K","T3L")))
        V("T3N", self.add("T1q","T1t"))
        V("T3O", self.sub("T3M","T3N"))
        V("T3Y", self.add("T3N","T3M"))

        V("T1Z", self.sub("T1u","T1n"))
        V("T2S", self.add("T2B","T2C"))
        V("T20", self.fma("KP461939766","T1h",self.mul("KP191341716","T1i")))
        V("T21", self.fma("KP461939766","T1f",self.mul("KP191341716","T16")))
        V("T22", self.sub("T20","T21"))
        V("T2T", self.add("T21","T20"))
        V("T23", self.sub("T1Z","T22"))
        V("T2Z", self.sub("T2S","T2T"))
        V("T2a", self.add("T1Z","T22"))
        V("T2U", self.add("T2S","T2T"))

        V("T24", self.sub("T1z","T1C"))
        V("T2V", self.add("T2G","T2H"))
        V("T25", self.fnma("KP191341716","T1I",self.mul("KP461939766","T1N")))
        V("T26", self.fma("KP191341716","T1P",self.mul("KP461939766","T1Q")))
        V("T27", self.sub("T25","T26"))
        V("T2W", self.add("T25","T26"))
        V("T28", self.sub("T24","T27"))
        V("T30", self.sub("T2V","T2W"))
        V("T2b", self.add("T24","T27"))
        V("T2X", self.add("T2V","T2W"))

        V("T1v", self.add("T1n","T1u"))
        V("T2D", self.sub("T2B","T2C"))
        V("T1g", self.fnma("KP191341716","T1f",self.mul("KP461939766","T16")))
        V("T1j", self.fnma("KP461939766","T1i",self.mul("KP191341716","T1h")))
        V("T1k", self.add("T1g","T1j"))
        V("T2E", self.sub("T1g","T1j"))
        V("T1w", self.sub("T1k","T1v"))
        V("T2M", self.sub("T2D","T2E"))
        V("T1V", self.add("T1v","T1k"))
        V("T2F", self.add("T2D","T2E"))

        V("T1D", self.add("T1z","T1C"))
        V("T2I", self.sub("T2G","T2H"))
        V("T1O", self.fma("KP461939766","T1I",self.mul("KP191341716","T1N")))
        V("T1R", self.fnma("KP191341716","T1Q",self.mul("KP461939766","T1P")))
        V("T1S", self.add("T1O","T1R"))
        V("T2J", self.sub("T1R","T1O"))
        V("T1T", self.add("T1D","T1S"))
        V("T2N", self.sub("T2I","T2J"))
        V("T1W", self.sub("T1D","T1S"))
        V("T2K", self.add("T2I","T2J"))

        # Store y0
        STr(0, self.add("T1","Tw"))
        STi(0, self.add("T49","T4a"))

        # Output section -- flat, no inner {}
        V("T3p", self.fnma("KP406231784","TQ",self.mul("KP317176192","TR")))
        V("T2i", self.fma("KP019555379","TK",self.mul("KP256951040","Tz")))
        V("T2j", self.fma("KP190495588","TN",self.mul("KP173544463","TM")))
        V("T3q", self.mul("KP1_414213562", self.sub("T2j","T2i")))
        V("T2k", self.mul("KP2_000000000", self.add("T2i","T2j")))
        V("T3A", self.sub("T3q","T3p"))
        V("T3r", self.add("T3p","T3q"))

        V("T4u", self.fnma("KP317176192","T44",self.mul("KP406231784","T43")))
        V("T4n", self.fnma("KP256951040","T3V",self.mul("KP019555379","T3O")))
        V("T4o", self.fma("KP190495588","T3Y",self.mul("KP173544463","T3X")))
        V("T4v", self.mul("KP1_414213562", self.add("T4n","T4o")))
        V("T4p", self.mul("KP2_000000000", self.sub("T4n","T4o")))
        V("T4G", self.add("T4u","T4v"))
        V("T4w", self.sub("T4u","T4v"))

        V("TS",  self.fma("KP317176192","TQ",self.mul("KP406231784","TR")))
        V("TU",  self.fnma("KP062500000","Tw","T1"))
        V("TV",  self.add("TT","TU"))
        V("T2y", self.sub("TU","TT"))
        V("TL",  self.fnma("KP256951040","TK",self.mul("KP019555379","Tz")))
        V("TO",  self.fnma("KP173544463","TN",self.mul("KP190495588","TM")))
        V("T2x", self.mul("KP1_414213562", self.sub("TO","TL")))
        V("TP",  self.mul("KP2_000000000", self.add("TL","TO")))
        V("T3v", self.sub("T2y","T2x"))
        V("TW",  self.add("TS","TV"))
        V("T2f", self.sub("TV","TS"))
        V("T2z", self.add("T2x","T2y"))

        V("T45", self.fma("KP317176192","T43",self.mul("KP406231784","T44")))
        V("T4b", self.fnma("KP062500000","T4a","T49"))
        V("T4c", self.add("T48","T4b"))
        V("T4A", self.sub("T4b","T48"))
        V("T3W", self.fma("KP256951040","T3O",self.mul("KP019555379","T3V")))
        V("T3Z", self.fnma("KP173544463","T3Y",self.mul("KP190495588","T3X")))
        V("T4z", self.mul("KP1_414213562", self.sub("T3Z","T3W")))
        V("T40", self.mul("KP2_000000000", self.add("T3W","T3Z")))
        V("T4J", self.sub("T4A","T4z"))
        V("T4d", self.add("T45","T4c"))
        V("T4k", self.sub("T4c","T45"))
        V("T4B", self.add("T4z","T4A"))

        V("T2l", self.fnma("KP035884665","T1V",self.mul("KP255183347","T1W")))
        V("T2m", self.fnma("KP212806139","T1w",self.mul("KP145326518","T1T")))
        V("T2n", self.add("T2l","T2m"))
        V("T3f", self.sub("T2m","T2l"))
        V("T3g", self.fnma("KP251015702","T2F",self.mul("KP058286937","T2K")))
        V("T3h", self.fma("KP254715062","T2M",self.mul("KP039070284","T2N")))
        V("T3i", self.sub("T3g","T3h"))
        V("T4g", self.add("T3h","T3g"))

        V("T2o", self.fnma("KP254715062","T23",self.mul("KP039070284","T28")))
        V("T2p", self.fma("KP251015702","T2a",self.mul("KP058286937","T2b")))
        V("T2q", self.add("T2o","T2p"))
        V("T3d", self.sub("T2p","T2o"))
        V("T3a", self.fma("KP212806139","T2U",self.mul("KP145326518","T2X")))
        V("T3b", self.fnma("KP035884665","T2Z",self.mul("KP255183347","T30")))
        V("T3c", self.sub("T3a","T3b"))
        V("T4h", self.add("T3b","T3a"))

        V("T2r", self.mul("KP1_414213562", self.sub("T2n","T2q")))
        V("T4m", self.mul("KP1_414213562", self.sub("T4h","T4g")))

        V("T3e", self.sub("T3c","T3d"))
        V("T3j", self.add("T3f","T3i"))
        V("T3k", self.fnma("KP382683432","T3j",self.mul("KP923879532","T3e")))
        V("T3C", self.fma("KP923879532","T3j",self.mul("KP382683432","T3e")))
        V("T3l", self.sub("T3i","T3f"))
        V("T3m", self.add("T3d","T3c"))
        V("T3n", self.fnma("KP923879532","T3m",self.mul("KP382683432","T3l")))
        V("T3B", self.fma("KP923879532","T3l",self.mul("KP382683432","T3m")))

        V("T1U", self.fma("KP145326518","T1w",self.mul("KP212806139","T1T")))
        V("T1X", self.fma("KP255183347","T1V",self.mul("KP035884665","T1W")))
        V("T1Y", self.sub("T1U","T1X"))
        V("T2A", self.add("T1X","T1U"))
        V("T2Y", self.fnma("KP212806139","T2X",self.mul("KP145326518","T2U")))
        V("T31", self.fma("KP255183347","T2Z",self.mul("KP035884665","T30")))
        V("T32", self.sub("T2Y","T31"))
        V("T3I", self.add("T31","T2Y"))

        V("T29", self.fma("KP039070284","T23",self.mul("KP254715062","T28")))
        V("T2c", self.fnma("KP251015702","T2b",self.mul("KP058286937","T2a")))
        V("T2d", self.add("T29","T2c"))
        V("T2R", self.sub("T2c","T29"))
        V("T2L", self.fma("KP058286937","T2F",self.mul("KP251015702","T2K")))
        V("T2O", self.fnma("KP254715062","T2N",self.mul("KP039070284","T2M")))
        V("T2P", self.sub("T2L","T2O"))
        V("T3H", self.add("T2O","T2L"))

        V("T2g", self.mul("KP1_414213562", self.sub("T1Y","T2d")))
        V("T4j", self.mul("KP1_414213562", self.sub("T3I","T3H")))

        V("T2Q", self.sub("T2A","T2P"))
        V("T33", self.add("T2R","T32"))
        V("T34", self.fnma("KP382683432","T33",self.mul("KP923879532","T2Q")))
        V("T3x", self.fma("KP382683432","T2Q",self.mul("KP923879532","T33")))
        V("T35", self.add("T2A","T2P"))
        V("T36", self.sub("T32","T2R"))
        V("T37", self.fma("KP923879532","T35",self.mul("KP382683432","T36")))
        V("T3w", self.fnma("KP382683432","T35",self.mul("KP923879532","T36")))

        # Stores: outputs 1..16
        V("TX",  self.add("TP","TW"))
        V("T2e", self.mul("KP2_000000000", self.add("T1Y","T2d")))
        STr(16, self.sub("TX","T2e"))
        STr(1,  self.add("TX","T2e"))

        V("T3J", self.mul("KP2_000000000", self.add("T3H","T3I")))
        V("T4e", self.add("T40","T4d"))
        STi(1,  self.add("T3J","T4e"))
        STi(16, self.sub("T4e","T3J"))

        V("T2h", self.add("T2f","T2g"))
        V("T2s", self.add("T2k","T2r"))
        STr(9,  self.sub("T2h","T2s"))
        STr(2,  self.add("T2h","T2s"))

        V("T4l", self.add("T4j","T4k"))
        V("T4q", self.add("T4m","T4p"))
        STi(2,  self.sub("T4l","T4q"))
        STi(9,  self.add("T4q","T4l"))

        V("T4r", self.sub("T4p","T4m"))
        V("T4s", self.sub("T4k","T4j"))
        STi(8,  self.add("T4r","T4s"))
        STi(15, self.sub("T4s","T4r"))

        V("T2t", self.sub("T2f","T2g"))
        V("T2u", self.sub("T2r","T2k"))
        STr(15, self.sub("T2t","T2u"))
        STr(8,  self.add("T2t","T2u"))

        V("T2v", self.sub("TW","TP"))
        V("T2w", self.mul("KP2_000000000", self.add("T2n","T2q")))
        STr(13, self.sub("T2v","T2w"))
        STr(4,  self.add("T2v","T2w"))

        V("T4f", self.sub("T4d","T40"))
        V("T4i", self.mul("KP2_000000000", self.add("T4g","T4h")))
        STi(4,  self.sub("T4f","T4i"))
        STi(13, self.add("T4i","T4f"))

        V("T38", self.add("T34","T37"))
        V("T39", self.add("T2z","T38"))
        V("T3t", self.sub("T2z","T38"))
        V("T3o", self.add("T3k","T3n"))
        V("T3s", self.sub("T3o","T3r"))
        V("T3u", self.add("T3r","T3o"))
        STr(6,  self.sub("T39","T3s"))
        STr(11, self.add("T3t","T3u"))
        STr(3,  self.add("T39","T3s"))
        STr(14, self.sub("T3t","T3u"))

        V("T4t", self.add("T3C","T3B"))
        V("T4x", self.add("T4t","T4w"))
        V("T4D", self.sub("T4t","T4w"))
        V("T4y", self.add("T3x","T3w"))
        V("T4C", self.add("T4y","T4B"))
        V("T4E", self.sub("T4B","T4y"))
        STi(3,  self.add("T4x","T4C"))
        STi(14, self.sub("T4E","T4D"))
        STi(6,  self.sub("T4C","T4x"))
        STi(11, self.add("T4D","T4E"))

        V("T4F", self.sub("T3k","T3n"))
        V("T4H", self.add("T4F","T4G"))
        V("T4L", self.sub("T4F","T4G"))
        V("T4I", self.sub("T34","T37"))
        V("T4K", self.add("T4I","T4J"))
        V("T4M", self.sub("T4J","T4I"))
        STi(5,  self.add("T4H","T4K"))
        STi(12, self.sub("T4M","T4L"))
        STi(7,  self.sub("T4K","T4H"))
        STi(10, self.add("T4L","T4M"))

        V("T3y", self.sub("T3w","T3x"))
        V("T3z", self.add("T3v","T3y"))
        V("T3F", self.sub("T3v","T3y"))
        V("T3D", self.sub("T3B","T3C"))
        V("T3E", self.add("T3A","T3D"))
        V("T3G", self.sub("T3D","T3A"))
        STr(7,  self.sub("T3z","T3E"))
        STr(10, self.add("T3F","T3G"))
        STr(5,  self.add("T3z","T3E"))
        STr(12, self.sub("T3F","T3G"))

    def _emit_r17_bwd(self):
        """Backward DFT-17: bwd(ri,ii,ro,io) = fwd(ii,ri,io,ro).

        The inputs x{n}_re and x{n}_im were loaded by emit_kernel_body
        from in_re and in_im respectively.  For backward we treat:
          - x{n}_im as the "real" input (ii->ri)
          - x{n}_re as the "imaginary" input (ri->ii)
        and swap the real/imaginary output buffers.

        We do this by re-emitting the forward butterfly with swapped
        variable name references and swapped store targets.
        """
        T = self.isa.T

        # Helper: emit "const T name = expr;"
        def V(name, expr):
            self.o(f"const {T} {name} = {expr};")

        # Backward: STr assigns to x{n}_im (bwd "real" out = im component)
        #           STi assigns to x{n}_re (bwd "imag" out = re component)
        # emit_kernel_body will then call emit_store(x{n}, n) which stores
        # x{n}_re -> out_re and x{n}_im -> out_im.
        # Net result: bwd real->out_im, bwd imag->out_re (correct swap).
        def STr(n, expr):
            self.o(f"x{n}_im = {expr};")

        def STi(n, expr):
            self.o(f"x{n}_re = {expr};")

        # Swap real/imaginary input variable references.
        # bwd DFT input: real part = x{n}_im (loaded from in_im)
        #                imag part = x{n}_re (loaded from in_re)
        def xr(n): return f"x{n}_im"   # bwd: "real" DFT input = imaginary component
        def xi(n): return f"x{n}_re"   # bwd: "imag" DFT input = real component

        # ----------------------------------------------------------------
        # Forward butterfly DAG with swapped xr/xi and swapped STr/STi.
        # This is identical to _emit_r17_fwd except for the two swaps above.
        # ----------------------------------------------------------------

        V("T1",  xr(0))
        V("T49", xi(0))

        V("T2",  xr(1));  V("T3",  xr(16))
        V("T4",  self.add("T2","T3"))
        V("T5",  xr(13)); V("T6",  xr(4))
        V("T7",  self.add("T5","T6"))
        V("T8",  self.add("T4","T7"))
        V("T2G", self.sub("T2","T3"))
        V("T2B", self.sub("T5","T6"))

        V("T1l", xi(13)); V("T1m", xi(4))
        V("T3Q", self.add("T1l","T1m"))
        V("T1x", xi(1));  V("T1y", xi(16))
        V("T3P", self.add("T1x","T1y"))
        V("T1n", self.sub("T1l","T1m"))
        V("T41", self.add("T3P","T3Q"))
        V("T1z", self.sub("T1x","T1y"))

        V("T9",  xr(9));  V("Ta",  xr(2))
        V("Tb",  self.add("T9","Ta"))
        V("TA",  self.sub("T9","Ta"))
        V("Tc",  xr(8));  V("Td",  xr(15))
        V("Te",  self.add("Tc","Td"))
        V("TB",  self.sub("Tc","Td"))

        V("Tf",  self.add("Tb","Te"))
        V("T2H", self.mul("KP707106781", self.sub("Tb","Te")))
        V("T2C", self.mul("KP707106781", self.sub("TA","TB")))

        V("T1o", xi(9));  V("T1p", xi(2))
        V("T1q", self.sub("T1o","T1p"))
        V("T1A", self.add("T1o","T1p"))
        V("T1r", xi(8));  V("T1s", xi(15))
        V("T1t", self.sub("T1r","T1s"))
        V("T1B", self.add("T1r","T1s"))

        V("T1u", self.mul("KP707106781", self.sub("T1q","T1t")))
        V("T42", self.add("T1A","T1B"))
        V("T1C", self.mul("KP707106781", self.sub("T1A","T1B")))

        V("To",  xr(5));  V("Tp",  xr(7))
        V("Tq",  self.add("To","Tp"))
        V("TG",  self.sub("To","Tp"))

        V("T1b", xi(14)); V("T1c", xi(11))
        V("T1d", self.sub("T1b","T1c"))
        V("T1G", self.add("T1b","T1c"))

        V("Tr",  xr(12)); V("Ts",  xr(10))
        V("Tt",  self.add("Tr","Ts"))
        V("TH",  self.sub("Tr","Ts"))

        V("T18", xi(3));  V("T19", xi(6))
        V("T1a", self.sub("T18","T19"))
        V("T1F", self.add("T18","T19"))

        V("Tu",  self.add("Tq","Tt"))
        V("TI",  self.add("TG","TH"))
        V("T3S", self.add("T1F","T1G"))
        V("T3K", self.add("T1a","T1d"))

        V("T1E", self.sub("Tq","Tt"))
        V("T1H", self.sub("T1F","T1G"))
        V("T1I", self.add("T1E","T1H"))
        V("T1P", self.sub("T1H","T1E"))
        V("T17", self.sub("TG","TH"))
        V("T1e", self.sub("T1a","T1d"))
        V("T1f", self.add("T17","T1e"))
        V("T1h", self.sub("T17","T1e"))

        V("Th",  xr(3));  V("Ti",  xr(6))
        V("Tj",  self.add("Th","Ti"))
        V("TD",  self.sub("Th","Ti"))

        V("T12", xi(12)); V("T13", xi(10))
        V("T14", self.sub("T12","T13"))
        V("T1L", self.add("T12","T13"))

        V("Tk",  xr(14)); V("Tl",  xr(11))
        V("Tm",  self.add("Tk","Tl"))
        V("TE",  self.sub("Tk","Tl"))

        V("TZ",  xi(5));  V("T10", xi(7))
        V("T11", self.sub("TZ","T10"))
        V("T1K", self.add("TZ","T10"))

        V("Tn",  self.add("Tj","Tm"))
        V("TF",  self.add("TD","TE"))
        V("T3T", self.add("T1K","T1L"))
        V("T3L", self.add("T11","T14"))

        V("T1J", self.sub("Tj","Tm"))
        V("T1M", self.sub("T1K","T1L"))
        V("T1N", self.sub("T1J","T1M"))
        V("T1Q", self.add("T1J","T1M"))
        V("TY",  self.sub("TD","TE"))
        V("T15", self.sub("T11","T14"))
        V("T16", self.sub("TY","T15"))
        V("T1i", self.add("TY","T15"))

        V("T46", self.add("T41","T42"))
        V("T47", self.add("T3S","T3T"))
        V("T48", self.mul("KP257694101", self.sub("T46","T47")))
        V("T4a", self.add("T46","T47"))

        V("TQ",  self.sub("T8","Tf"))
        V("T44", self.add("T3K","T3L"))
        V("T43", self.sub("T41","T42"))
        V("TR",  self.add("TF","TI"))

        V("Tg",  self.add("T8","Tf"))
        V("Tv",  self.add("Tn","Tu"))
        V("Tw",  self.add("Tg","Tv"))
        V("TT",  self.mul("KP257694101", self.sub("Tg","Tv")))
        V("Tx",  self.sub("T4","T7"))
        V("Ty",  self.mul("KP707106781", self.sub("Tn","Tu")))
        V("Tz",  self.sub("Tx","Ty"))
        V("TM",  self.add("Tx","Ty"))

        V("T3R", self.sub("T3P","T3Q"))
        V("T3U", self.mul("KP707106781", self.sub("T3S","T3T")))
        V("T3V", self.sub("T3R","T3U"))
        V("T3X", self.add("T3R","T3U"))

        V("TC",  self.add("TA","TB"))
        V("TJ",  self.mul("KP707106781", self.sub("TF","TI")))
        V("TK",  self.sub("TC","TJ"))
        V("TN",  self.add("TC","TJ"))
        V("T3M", self.mul("KP707106781", self.sub("T3K","T3L")))
        V("T3N", self.add("T1q","T1t"))
        V("T3O", self.sub("T3M","T3N"))
        V("T3Y", self.add("T3N","T3M"))

        V("T1Z", self.sub("T1u","T1n"))
        V("T2S", self.add("T2B","T2C"))
        V("T20", self.fma("KP461939766","T1h",self.mul("KP191341716","T1i")))
        V("T21", self.fma("KP461939766","T1f",self.mul("KP191341716","T16")))
        V("T22", self.sub("T20","T21"))
        V("T2T", self.add("T21","T20"))
        V("T23", self.sub("T1Z","T22"))
        V("T2Z", self.sub("T2S","T2T"))
        V("T2a", self.add("T1Z","T22"))
        V("T2U", self.add("T2S","T2T"))

        V("T24", self.sub("T1z","T1C"))
        V("T2V", self.add("T2G","T2H"))
        V("T25", self.fnma("KP191341716","T1I",self.mul("KP461939766","T1N")))
        V("T26", self.fma("KP191341716","T1P",self.mul("KP461939766","T1Q")))
        V("T27", self.sub("T25","T26"))
        V("T2W", self.add("T25","T26"))
        V("T28", self.sub("T24","T27"))
        V("T30", self.sub("T2V","T2W"))
        V("T2b", self.add("T24","T27"))
        V("T2X", self.add("T2V","T2W"))

        V("T1v", self.add("T1n","T1u"))
        V("T2D", self.sub("T2B","T2C"))
        V("T1g", self.fnma("KP191341716","T1f",self.mul("KP461939766","T16")))
        V("T1j", self.fnma("KP461939766","T1i",self.mul("KP191341716","T1h")))
        V("T1k", self.add("T1g","T1j"))
        V("T2E", self.sub("T1g","T1j"))
        V("T1w", self.sub("T1k","T1v"))
        V("T2M", self.sub("T2D","T2E"))
        V("T1V", self.add("T1v","T1k"))
        V("T2F", self.add("T2D","T2E"))

        V("T1D", self.add("T1z","T1C"))
        V("T2I", self.sub("T2G","T2H"))
        V("T1O", self.fma("KP461939766","T1I",self.mul("KP191341716","T1N")))
        V("T1R", self.fnma("KP191341716","T1Q",self.mul("KP461939766","T1P")))
        V("T1S", self.add("T1O","T1R"))
        V("T2J", self.sub("T1R","T1O"))
        V("T1T", self.add("T1D","T1S"))
        V("T2N", self.sub("T2I","T2J"))
        V("T1W", self.sub("T1D","T1S"))
        V("T2K", self.add("T2I","T2J"))

        # Store y0 (bwd: real result goes to io, imag to ro)
        STr(0, self.add("T1","Tw"))
        STi(0, self.add("T49","T4a"))

        V("T3p", self.fnma("KP406231784","TQ",self.mul("KP317176192","TR")))
        V("T2i", self.fma("KP019555379","TK",self.mul("KP256951040","Tz")))
        V("T2j", self.fma("KP190495588","TN",self.mul("KP173544463","TM")))
        V("T3q", self.mul("KP1_414213562", self.sub("T2j","T2i")))
        V("T2k", self.mul("KP2_000000000", self.add("T2i","T2j")))
        V("T3A", self.sub("T3q","T3p"))
        V("T3r", self.add("T3p","T3q"))

        V("T4u", self.fnma("KP317176192","T44",self.mul("KP406231784","T43")))
        V("T4n", self.fnma("KP256951040","T3V",self.mul("KP019555379","T3O")))
        V("T4o", self.fma("KP190495588","T3Y",self.mul("KP173544463","T3X")))
        V("T4v", self.mul("KP1_414213562", self.add("T4n","T4o")))
        V("T4p", self.mul("KP2_000000000", self.sub("T4n","T4o")))
        V("T4G", self.add("T4u","T4v"))
        V("T4w", self.sub("T4u","T4v"))

        V("TS",  self.fma("KP317176192","TQ",self.mul("KP406231784","TR")))
        V("TU",  self.fnma("KP062500000","Tw","T1"))
        V("TV",  self.add("TT","TU"))
        V("T2y", self.sub("TU","TT"))
        V("TL",  self.fnma("KP256951040","TK",self.mul("KP019555379","Tz")))
        V("TO",  self.fnma("KP173544463","TN",self.mul("KP190495588","TM")))
        V("T2x", self.mul("KP1_414213562", self.sub("TO","TL")))
        V("TP",  self.mul("KP2_000000000", self.add("TL","TO")))
        V("T3v", self.sub("T2y","T2x"))
        V("TW",  self.add("TS","TV"))
        V("T2f", self.sub("TV","TS"))
        V("T2z", self.add("T2x","T2y"))

        V("T45", self.fma("KP317176192","T43",self.mul("KP406231784","T44")))
        V("T4b", self.fnma("KP062500000","T4a","T49"))
        V("T4c", self.add("T48","T4b"))
        V("T4A", self.sub("T4b","T48"))
        V("T3W", self.fma("KP256951040","T3O",self.mul("KP019555379","T3V")))
        V("T3Z", self.fnma("KP173544463","T3Y",self.mul("KP190495588","T3X")))
        V("T4z", self.mul("KP1_414213562", self.sub("T3Z","T3W")))
        V("T40", self.mul("KP2_000000000", self.add("T3W","T3Z")))
        V("T4J", self.sub("T4A","T4z"))
        V("T4d", self.add("T45","T4c"))
        V("T4k", self.sub("T4c","T45"))
        V("T4B", self.add("T4z","T4A"))

        V("T2l", self.fnma("KP035884665","T1V",self.mul("KP255183347","T1W")))
        V("T2m", self.fnma("KP212806139","T1w",self.mul("KP145326518","T1T")))
        V("T2n", self.add("T2l","T2m"))
        V("T3f", self.sub("T2m","T2l"))
        V("T3g", self.fnma("KP251015702","T2F",self.mul("KP058286937","T2K")))
        V("T3h", self.fma("KP254715062","T2M",self.mul("KP039070284","T2N")))
        V("T3i", self.sub("T3g","T3h"))
        V("T4g", self.add("T3h","T3g"))

        V("T2o", self.fnma("KP254715062","T23",self.mul("KP039070284","T28")))
        V("T2p", self.fma("KP251015702","T2a",self.mul("KP058286937","T2b")))
        V("T2q", self.add("T2o","T2p"))
        V("T3d", self.sub("T2p","T2o"))
        V("T3a", self.fma("KP212806139","T2U",self.mul("KP145326518","T2X")))
        V("T3b", self.fnma("KP035884665","T2Z",self.mul("KP255183347","T30")))
        V("T3c", self.sub("T3a","T3b"))
        V("T4h", self.add("T3b","T3a"))

        V("T2r", self.mul("KP1_414213562", self.sub("T2n","T2q")))
        V("T4m", self.mul("KP1_414213562", self.sub("T4h","T4g")))

        V("T3e", self.sub("T3c","T3d"))
        V("T3j", self.add("T3f","T3i"))
        V("T3k", self.fnma("KP382683432","T3j",self.mul("KP923879532","T3e")))
        V("T3C", self.fma("KP923879532","T3j",self.mul("KP382683432","T3e")))
        V("T3l", self.sub("T3i","T3f"))
        V("T3m", self.add("T3d","T3c"))
        V("T3n", self.fnma("KP923879532","T3m",self.mul("KP382683432","T3l")))
        V("T3B", self.fma("KP923879532","T3l",self.mul("KP382683432","T3m")))

        V("T1U", self.fma("KP145326518","T1w",self.mul("KP212806139","T1T")))
        V("T1X", self.fma("KP255183347","T1V",self.mul("KP035884665","T1W")))
        V("T1Y", self.sub("T1U","T1X"))
        V("T2A", self.add("T1X","T1U"))
        V("T2Y", self.fnma("KP212806139","T2X",self.mul("KP145326518","T2U")))
        V("T31", self.fma("KP255183347","T2Z",self.mul("KP035884665","T30")))
        V("T32", self.sub("T2Y","T31"))
        V("T3I", self.add("T31","T2Y"))

        V("T29", self.fma("KP039070284","T23",self.mul("KP254715062","T28")))
        V("T2c", self.fnma("KP251015702","T2b",self.mul("KP058286937","T2a")))
        V("T2d", self.add("T29","T2c"))
        V("T2R", self.sub("T2c","T29"))
        V("T2L", self.fma("KP058286937","T2F",self.mul("KP251015702","T2K")))
        V("T2O", self.fnma("KP254715062","T2N",self.mul("KP039070284","T2M")))
        V("T2P", self.sub("T2L","T2O"))
        V("T3H", self.add("T2O","T2L"))

        V("T2g", self.mul("KP1_414213562", self.sub("T1Y","T2d")))
        V("T4j", self.mul("KP1_414213562", self.sub("T3I","T3H")))

        V("T2Q", self.sub("T2A","T2P"))
        V("T33", self.add("T2R","T32"))
        V("T34", self.fnma("KP382683432","T33",self.mul("KP923879532","T2Q")))
        V("T3x", self.fma("KP382683432","T2Q",self.mul("KP923879532","T33")))
        V("T35", self.add("T2A","T2P"))
        V("T36", self.sub("T32","T2R"))
        V("T37", self.fma("KP923879532","T35",self.mul("KP382683432","T36")))
        V("T3w", self.fnma("KP382683432","T35",self.mul("KP923879532","T36")))

        V("TX",  self.add("TP","TW"))
        V("T2e", self.mul("KP2_000000000", self.add("T1Y","T2d")))
        STr(16, self.sub("TX","T2e"))
        STr(1,  self.add("TX","T2e"))

        V("T3J", self.mul("KP2_000000000", self.add("T3H","T3I")))
        V("T4e", self.add("T40","T4d"))
        STi(1,  self.add("T3J","T4e"))
        STi(16, self.sub("T4e","T3J"))

        V("T2h", self.add("T2f","T2g"))
        V("T2s", self.add("T2k","T2r"))
        STr(9,  self.sub("T2h","T2s"))
        STr(2,  self.add("T2h","T2s"))

        V("T4l", self.add("T4j","T4k"))
        V("T4q", self.add("T4m","T4p"))
        STi(2,  self.sub("T4l","T4q"))
        STi(9,  self.add("T4q","T4l"))

        V("T4r", self.sub("T4p","T4m"))
        V("T4s", self.sub("T4k","T4j"))
        STi(8,  self.add("T4r","T4s"))
        STi(15, self.sub("T4s","T4r"))

        V("T2t", self.sub("T2f","T2g"))
        V("T2u", self.sub("T2r","T2k"))
        STr(15, self.sub("T2t","T2u"))
        STr(8,  self.add("T2t","T2u"))

        V("T2v", self.sub("TW","TP"))
        V("T2w", self.mul("KP2_000000000", self.add("T2n","T2q")))
        STr(13, self.sub("T2v","T2w"))
        STr(4,  self.add("T2v","T2w"))

        V("T4f", self.sub("T4d","T40"))
        V("T4i", self.mul("KP2_000000000", self.add("T4g","T4h")))
        STi(4,  self.sub("T4f","T4i"))
        STi(13, self.add("T4i","T4f"))

        V("T38", self.add("T34","T37"))
        V("T39", self.add("T2z","T38"))
        V("T3t", self.sub("T2z","T38"))
        V("T3o", self.add("T3k","T3n"))
        V("T3s", self.sub("T3o","T3r"))
        V("T3u", self.add("T3r","T3o"))
        STr(6,  self.sub("T39","T3s"))
        STr(11, self.add("T3t","T3u"))
        STr(3,  self.add("T39","T3s"))
        STr(14, self.sub("T3t","T3u"))

        V("T4t", self.add("T3C","T3B"))
        V("T4x", self.add("T4t","T4w"))
        V("T4D", self.sub("T4t","T4w"))
        V("T4y", self.add("T3x","T3w"))
        V("T4C", self.add("T4y","T4B"))
        V("T4E", self.sub("T4B","T4y"))
        STi(3,  self.add("T4x","T4C"))
        STi(14, self.sub("T4E","T4D"))
        STi(6,  self.sub("T4C","T4x"))
        STi(11, self.add("T4D","T4E"))

        V("T4F", self.sub("T3k","T3n"))
        V("T4H", self.add("T4F","T4G"))
        V("T4L", self.sub("T4F","T4G"))
        V("T4I", self.sub("T34","T37"))
        V("T4K", self.add("T4I","T4J"))
        V("T4M", self.sub("T4J","T4I"))
        STi(5,  self.add("T4H","T4K"))
        STi(12, self.sub("T4M","T4L"))
        STi(7,  self.sub("T4K","T4H"))
        STi(10, self.add("T4L","T4M"))

        V("T3y", self.sub("T3w","T3x"))
        V("T3z", self.add("T3v","T3y"))
        V("T3F", self.sub("T3v","T3y"))
        V("T3D", self.sub("T3B","T3C"))
        V("T3E", self.add("T3A","T3D"))
        V("T3G", self.sub("T3D","T3A"))
        STr(7,  self.sub("T3z","T3E"))
        STr(10, self.add("T3F","T3G"))
        STr(5,  self.add("T3z","T3E"))
        STr(12, self.sub("T3F","T3G"))


# ================================================================
# HELPERS: constants
# ================================================================

DFT17_CONSTS = [
    ('KP382683432',   KP382683432_val),
    ('KP923879532',   KP923879532_val),
    ('KP039070284',   KP039070284_val),
    ('KP254715062',   KP254715062_val),
    ('KP251015702',   KP251015702_val),
    ('KP058286937',   KP058286937_val),
    ('KP212806139',   KP212806139_val),
    ('KP145326518',   KP145326518_val),
    ('KP035884665',   KP035884665_val),
    ('KP255183347',   KP255183347_val),
    ('KP062500000',   KP062500000_val),
    ('KP2_000000000', KP2_000000000_val),
    ('KP1_414213562', KP1_414213562_val),
    ('KP173544463',   KP173544463_val),
    ('KP190495588',   KP190495588_val),
    ('KP256951040',   KP256951040_val),
    ('KP019555379',   KP019555379_val),
    ('KP406231784',   KP406231784_val),
    ('KP317176192',   KP317176192_val),
    ('KP191341716',   KP191341716_val),
    ('KP461939766',   KP461939766_val),
    ('KP257694101',   KP257694101_val),
    ('KP707106781',   KP707106781_val),
]


def emit_dft17_constants(em):
    """Emit the 23 DFT-17 constants + sign_flip as SIMD broadcasts or scalars."""
    T = em.isa.T
    if em.isa.name == 'scalar':
        for name, val in DFT17_CONSTS:
            em.o(f"const double {name} = {val:+.45f};")
    else:
        set1 = f"{em.isa.p}_set1_pd"
        em.o(f"const {T} sign_flip = {set1}(-0.0);")
        for name, val in DFT17_CONSTS:
            em.o(f"const {T} {name} = {set1}({val:+.45f});")
        em.o(f"(void)sign_flip;  /* used by neg() */")
    em.b()


def emit_dft17_constants_raw(lines, isa, indent=1):
    """Append DFT-17 constant declarations to a line list."""
    pad = "    " * indent
    T = isa.T
    if isa.name == 'scalar':
        for name, val in DFT17_CONSTS:
            lines.append(f"{pad}const double {name} = {val:+.45f};")
    else:
        set1 = f"{isa.p}_set1_pd"
        lines.append(f"{pad}const {T} sign_flip = {set1}(-0.0);")
        for name, val in DFT17_CONSTS:
            lines.append(f"{pad}const {T} {name} = {set1}({val:+.45f});")
        lines.append(f"{pad}(void)sign_flip;")


# ================================================================
# KERNEL BODY EMITTERS
# ================================================================

def emit_kernel_body(em, d, variant):
    """Emit the inner loop body for notw, dit_tw, dif_tw."""
    T = em.isa.T

    # Load all 17 inputs
    for n in range(R):
        em.emit_load(f"x{n}", n)

    # DIT: twiddle inputs 1..16 before butterfly
    if variant == 'dit_tw':
        for n in range(1, R):
            em.emit_ext_tw(f"x{n}", n - 1, d)
    elif variant == 'dit_tw_scalar':
        for n in range(1, R):
            em.emit_ext_tw_scalar(f"x{n}", n - 1, d)

    em.b()
    em.emit_radix17_butterfly(d)
    em.b()

    # DIF: twiddle outputs 1..16 after butterfly
    if variant == 'dif_tw':
        for m in range(1, R):
            em.emit_ext_tw(f"x{m}", m - 1, d)
    elif variant == 'dif_tw_scalar':
        for m in range(1, R):
            em.emit_ext_tw_scalar(f"x{m}", m - 1, d)

    # Store all 17 outputs
    for m in range(R):
        em.emit_store(f"x{m}", m)


def emit_kernel_body_log3(em, d, variant):
    """Emit the log3 variant: derive w2..w16 from w1 (15 cmuls)."""
    T = em.isa.T
    is_dit = variant == 'dit_tw_log3'

    em.c("Load base twiddle w1, derive w2..w16 (15 cmuls)")
    tb, tbi = em._tw_buf(), em._tw_buf_im()
    ta = em._tw_addr(0)
    if em.isa.name == 'scalar':
        em.o(f"const double w1r = {tb}[{ta}], w1i = {tbi}[{ta}];")
    else:
        em.o(f"const {T} w1r = LD(&{tb}[{ta}]), w1i = LD(&{tbi}[{ta}]);")
    em.n_load += 2

    # Chain: w2=w1^2, w3=w1*w2, w4=w2^2, w5=w2*w3, w6=w3^2,
    #        w7=w3*w4, w8=w4^2, w9=w4*w5, w10=w5^2, w11=w5*w6,
    #        w12=w6^2, w13=w6*w7, w14=w7^2, w15=w7*w8, w16=w8^2
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
    ]
    for (wn, ar, ai, br, bi) in chain:
        em.o(f"{T} w{wn}r, w{wn}i;")
        em.emit_cmul(f"w{wn}r", f"w{wn}i", ar, ai, br, bi, 'fwd')
    em.b()

    # Load inputs
    for n in range(R):
        em.emit_load(f"x{n}", n)

    # DIT: apply twiddles to x1..x16
    if is_dit:
        for n in range(1, R):
            em.emit_cmul_inplace(f"x{n}", f"w{n}r", f"w{n}i", d)
    em.b()

    em.emit_radix17_butterfly(d)
    em.b()

    # DIF: apply twiddles to outputs x1..x16
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
        func_base = 'radix17_n1_dit_kernel'
    elif variant == 'dit_tw':
        func_base = 'radix17_tw_flat_dit_kernel'
        tw_params = 'flat'
    elif variant == 'dit_tw_scalar':
        func_base = 'radix17_tw_flat_dit_kernel'
        tw_params = 'flat'
    elif variant == 'dif_tw':
        func_base = 'radix17_tw_flat_dif_kernel'
        tw_params = 'flat'
    elif variant == 'dif_tw_scalar':
        func_base = 'radix17_tw_flat_dif_kernel'
        tw_params = 'flat'
    elif variant == 'dit_tw_log3':
        func_base = 'radix17_tw_log3_dit_kernel'
        tw_params = 'flat'
    elif variant == 'dif_tw_log3':
        func_base = 'radix17_tw_log3_dif_kernel'
        tw_params = 'flat'

    vname = {
        'notw':        'N1 (no twiddle)',
        'dit_tw':      'DIT twiddled (flat)',
        'dif_tw':      'DIF twiddled (flat)',
        'dit_tw_log3': 'DIT twiddled (log3 derived)',
        'dif_tw_log3': 'DIF twiddled (log3 derived)',
    }[variant]
    guard = f"FFT_RADIX17_{isa.name.upper()}_{variant.upper()}_H"

    em.L.append(f"/**")
    em.L.append(f" * @file fft_radix17_{isa.name}_{variant}.h")
    em.L.append(f" * @brief DFT-17 {isa.name.upper()} {vname}")
    em.L.append(f" *")
    em.L.append(f" * Monolithic DFT-17 butterfly (genfft DAG), 23 constants.")
    em.L.append(f" * Register pressure exceeds both AVX2 and AVX-512;")
    em.L.append(f" * compiler handles register allocation / spills.")
    em.L.append(f" * k-step={isa.k_step}")
    em.L.append(f" * Generated by gen_radix17.py")
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
        emit_dft17_constants(em)

        # Working registers for all 17 inputs/outputs
        em.o(f"{T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
        em.o(f"{T} x4_re,x4_im,x5_re,x5_im,x6_re,x6_im,x7_re,x7_im;")
        em.o(f"{T} x8_re,x8_im,x9_re,x9_im,x10_re,x10_im,x11_re,x11_im;")
        em.o(f"{T} x12_re,x12_im,x13_re,x13_im,x14_re,x14_im,x15_re,x15_im;")
        em.o(f"{T} x16_re,x16_im;")
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

    # No U=2 for R=17: register pressure exceeds AVX-512 (32 ZMM).

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
        func_base = "radix17_n1"
        vname = "n1 (separate is/os)"
    elif is_n1_scaled:
        func_base = "radix17_n1_scaled"
        vname = "n1_scaled (separate is/os, output *= scale)"
    elif is_t1_oop_dit:
        func_base = "radix17_t1_oop_dit"
        vname = "t1_oop DIT (out-of-place, separate is/os, with twiddle)"
    elif is_t1_dif:
        func_base = "radix17_t1_dif"
        vname = "t1 DIF (in-place twiddle)"
    elif is_t1s_dit:
        func_base = "radix17_t1s_dit"
        vname = "t1s DIT (in-place, scalar broadcast twiddle)"
    elif is_t1_dit_log3:
        func_base = "radix17_t1_dit_log3"
        vname = "t1 DIT log3 (in-place, derived twiddles)"
    else:
        func_base = "radix17_t1_dit"
        vname = "t1 DIT (in-place twiddle)"

    guard = f"FFT_RADIX17_{isa.name.upper()}_CT_{ct_variant.upper()}_H"

    em.L.append(f"/**")
    em.L.append(f" * @file fft_radix17_{isa.name}_{ct_variant}.h")
    em.L.append(f" * @brief DFT-17 {isa.name.upper()} {vname}")
    em.L.append(f" *")
    em.L.append(f" * FFTW-style codelet for recursive CT executor.")
    em.L.append(f" * Generated by gen_radix17.py --variant {ct_variant}")
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

        emit_dft17_constants(em)

        em.o(f"{T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
        em.o(f"{T} x4_re,x4_im,x5_re,x5_im,x6_re,x6_im,x7_re,x7_im;")
        em.o(f"{T} x8_re,x8_im,x9_re,x9_im,x10_re,x10_im,x11_re,x11_im;")
        em.o(f"{T} x12_re,x12_im,x13_re,x13_im,x14_re,x14_im,x15_re,x15_im;")
        em.o(f"{T} x16_re,x16_im;")
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

    # n1_ovs: butterfly with fused SIMD transpose stores (for R=17)
    # 17 bins = 4 groups of 4 (bins 0-3, 4-7, 8-11, 12-15) + bin 16 leftover
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
            em.L.append(f"radix17_n1_ovs_{d}_{isa.name}(")
            em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
            em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
            em.L.append(f"    size_t is, size_t os, size_t vl, size_t ovs)")
            em.L.append(f"{{")

            em.L.append(f"    /* n1_ovs: butterfly -> tbuf, then 4x4 transpose (bins 0-3..12-15) + scatter (bin 16) */")
            em.L.append(f"    {isa.align} double tbuf_re[{R * VL}];")
            em.L.append(f"    {isa.align} double tbuf_im[{R * VL}];")

            emit_dft17_constants_raw(em.L, isa, indent=1)

            em.L.append(f"    {T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
            em.L.append(f"    {T} x4_re,x4_im,x5_re,x5_im,x6_re,x6_im,x7_re,x7_im;")
            em.L.append(f"    {T} x8_re,x8_im,x9_re,x9_im,x10_re,x10_im,x11_re,x11_im;")
            em.L.append(f"    {T} x12_re,x12_im,x13_re,x13_im,x14_re,x14_im,x15_re,x15_im;")
            em.L.append(f"    {T} x16_re,x16_im;")
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

            # Bin 16: extract from tbuf -> scatter
            em.L.append(f"        /* Bin 16: extract from tbuf -> scatter */")
            if isa.name == 'avx2':
                for comp, arr in [('re', 'out_re'), ('im', 'out_im')]:
                    bname = f"tbuf_{comp}"
                    em.L.append(f"        {{ __m256d v=LD(&{bname}[16*{VL}]);")
                    em.L.append(f"          __m128d lo=_mm256_castpd256_pd128(v), hi=_mm256_extractf128_pd(v,1);")
                    em.L.append(f"          _mm_storel_pd(&{arr}[(k+0)*ovs+os*16], lo);")
                    em.L.append(f"          _mm_storeh_pd(&{arr}[(k+1)*ovs+os*16], lo);")
                    em.L.append(f"          _mm_storel_pd(&{arr}[(k+2)*ovs+os*16], hi);")
                    em.L.append(f"          _mm_storeh_pd(&{arr}[(k+3)*ovs+os*16], hi); }}")
            else:  # avx512
                for comp, arr in [('re', 'out_re'), ('im', 'out_im')]:
                    bname = f"tbuf_{comp}"
                    for j in range(VL):
                        em.L.append(f"        {arr}[(k+{j})*ovs+os*16] = {bname}[16*{VL}+{j}];")

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
        t2_pattern = 'radix17_n1_dit_kernel'
        sv_name = 'radix17_n1sv_kernel'
    elif variant == 'dit_tw':
        t2_pattern = 'radix17_tw_flat_dit_kernel'
        sv_name = 'radix17_t1sv_dit_kernel'
    elif variant == 'dit_tw_scalar':
        t2_pattern = 'radix17_tw_flat_dit_kernel'
        sv_name = 'radix17_t1sv_dit_kernel'
    elif variant == 'dif_tw':
        t2_pattern = 'radix17_tw_flat_dif_kernel'
        sv_name = 'radix17_t1sv_dif_kernel'
    elif variant == 'dif_tw_scalar':
        t2_pattern = 'radix17_tw_flat_dif_kernel'
        sv_name = 'radix17_t1sv_dif_kernel'
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
    parser = argparse.ArgumentParser(description='Unified R=17 codelet generator')
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
    main()
