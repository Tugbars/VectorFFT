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
gen_radix13.py -- Unified DFT-13 codelet generator for VectorFFT

Monolithic DFT-13 butterfly (straight-line genfft DAG translation).
21 constants, NO clean Rader decomposition -- mix of adds, subs, muls,
and scaled operations (multiply by 0.5, sqrt(3)/2, 2.0, etc.).

Register pressure exceeds BOTH AVX2 (16) and AVX-512 (32):
  13 inputs(26 regs) + 21 constants = 47
Both ISAs use the same butterfly (no separate no-spill path for AVX-512).
The genfft DAG interleaves loads with computation, so explicit Sethi-Ullman
phases are NOT needed -- we emit the DAG as one straight-line block and
let the C compiler handle register allocation.  All butterfly temporaries
are wrapped in a single large { } scope block.

DFT-13 constants (21, from genfft):
  KP500000000, KP866025403, KP300462606, KP387390585, KP265966249,
  KP113854479, KP503537032, KP575140729, KP174138601, KP256247671,
  KP156891391, KP011599105, KP300238635, KP1_732050807, KP258260390,
  KP132983124, KP251768516, KP075902986, KP083333333, KP2_000000000,
  (sign_flip for SIMD)

Log3 twiddle for R=13 (11 cmuls from w1):
  w2=w1^2, w3=w1*w2, w4=w2^2, w5=w2*w3, w6=w3^2,
  w7=w3*w4, w8=w4^2, w9=w4*w5, w10=w5^2, w11=w5*w6, w12=w6^2

Usage:
  python3 gen_radix13.py --isa avx2 --variant all
  python3 gen_radix13.py --isa all --variant ct_n1
  python3 gen_radix13.py --isa avx2 --variant ct_t1_dit
"""

import sys, math, argparse, re

R = 13

# DFT-13 constants (from genfft)
KP500000000_val = +0.500000000000000000000000000000000000000000000
KP866025403_val = +0.866025403784438646763723170752936183471402627
KP300462606_val = +0.300462606288665774426601772289207995520941381
KP387390585_val = +0.387390585467617292130675966426762851778775217
KP265966249_val = +0.265966249214837287587521063842185948798330267
KP113854479_val = +0.113854479055790798974654345867655310534642560
KP503537032_val = +0.503537032863766627246873853868466977093348562
KP575140729_val = +0.575140729474003121368385547455453388461001608
KP174138601_val = +0.174138601152135905005660794929264742616964676
KP256247671_val = +0.256247671582936600958684654061725059144125175
KP156891391_val = +0.156891391051584611046832726756003269660212636
KP011599105_val = +0.011599105605768290721655456654083252189827041
KP300238635_val = +0.300238635966332641462884626667381504676006424
KP1_732050807_val = +1.732050807568877293527446341505872366942805254
KP258260390_val = +0.258260390311744861420450644284508567852516811
KP132983124_val = +0.132983124607418643793760531921092974399165133
KP251768516_val = +0.251768516431883313623436926934233488546674281
KP075902986_val = +0.075902986037193865983102897245103540356428373
KP083333333_val = +0.083333333333333333333333333333333333333333333
KP2_000000000_val = +2.000000000000000000000000000000000000000000000

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

ISA_SCALAR = ISAConfig('scalar', 'double', 1, 1, '', 1, '', '', 'R13S')
ISA_AVX2   = ISAConfig('avx2', '__m256d', 4, 4, '_mm256', 4,
    '__attribute__((target("avx2,fma")))',
    '__attribute__((aligned(32)))', 'R13A')
ISA_AVX512 = ISAConfig('avx512', '__m512d', 8, 8, '_mm512', 8,
    '__attribute__((target("avx512f,avx512dq,fma")))',
    '__attribute__((aligned(64)))', 'R13L')

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
        return f"{n}*K+{ke}"

    def _out_addr(self, m, ke="k"):
        if self.addr_mode == 'n1':
            return f"{m}*os+{ke}"
        elif self.addr_mode == 'n1_ovs':
            return f"{m}*{self.isa.k_step}"
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

    # -- External twiddle (flat) --
    def _tw_addr(self, tw_idx, ke="k"):
        if self.addr_mode == 't1':
            return f"{tw_idx}*me+m"
        return f"{tw_idx}*K+{ke}"

    def _tw_buf(self):
        return "W_re" if self.addr_mode == 't1' else "tw_re"
    def _tw_buf_im(self):
        return "W_im" if self.addr_mode == 't1' else "tw_im"

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
        """Emit broadcast of all (R-1) twiddle scalars BEFORE the m-loop.
        This hoists loop-invariant broadcasts out of the inner loop,
        letting the compiler keep small R in registers and spill large R
        to L1-hot stack (aligned loads, not broadcasts per iteration)."""
        T = self.isa.T
        for i in range(R - 1):
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
        if self.tw_hoisted:
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

    # -- DFT-13 butterfly --
    def emit_radix13_butterfly(self, d, out_names=None):
        """Emit DFT-13 butterfly, translating genfft DAG exactly.

        R=13 register pressure exceeds both AVX2 (16) and AVX-512 (32).
        Both ISAs emit the DAG as one straight-line block inside a { } scope
        and let the C compiler handle register allocation / spills.
        out_names: if given, list of 13 output variable names.
                   Default: overwrite x0..x12.
        """
        fwd = (d == 'fwd')
        T = self.isa.T

        if out_names is None:
            out_names = [f'x{i}' for i in range(13)]

        self.c(f"DFT-13 butterfly [{d}] (straight-line genfft DAG, compiler-managed regs)")
        self.o("{")
        self.ind += 1

        if fwd:
            self._emit_radix13_butterfly_fwd(out_names)
        else:
            self._emit_radix13_butterfly_bwd(out_names)

        self.ind -= 1
        self.o("}")

    def _emit_radix13_butterfly_fwd(self, out_names):
        """Forward DFT-13 butterfly -- exact translation of genfft fwd kernel."""
        T = self.isa.T

        # === Load x0 (TW) ===
        self.c("TW = x0 (preserved, used later for y0 and TY)")
        self.o(f"{T} TWr = x0_re, TWi = x0_im;")
        self.b()

        # === Block 1: inputs x8, x5 -> T3, TH ===
        self.c("Block 1: x8, x5 -> T3 (diff), TH (sum)")
        self.o(f"{T} T3r = {self.sub('x8_re','x5_re')}, T3i = {self.sub('x8_im','x5_im')};")
        self.o(f"{T} THr = {self.add('x8_re','x5_re')}, THi = {self.add('x8_im','x5_im')};")
        self.b()

        # === Block 2: x12, x10, x4 -> Tl, Tw, Tp ===
        self.c("Block 2: x12, x10, x4 -> Tk, Tl, Tw, Tp")
        self.o(f"{T} Tkr = {self.add('x10_re','x4_re')}, Tki = {self.add('x10_im','x4_im')};")
        self.o(f"{T} Tlr = {self.add('x12_re','Tkr')}, Tli = {self.add('x12_im','Tki')};")
        self.o(f"{T} Twr = {self.sub('x10_re','x4_re')}, Twi = {self.sub('x10_im','x4_im')};")
        self.o(f"{T} Tpr = {self.fnma('KP500000000','Tkr','x12_re')}, Tpi = {self.fnma('KP500000000','Tki','x12_im')};")
        self.b()

        # === Block 3: x1, x3, x9 -> Tf, Tg, Tv, To ===
        self.c("Block 3: x1, x3, x9 -> Tf, Tg, Tv, To")
        self.o(f"{T} Tfr = {self.add('x3_re','x9_re')}, Tfi = {self.add('x3_im','x9_im')};")
        self.o(f"{T} Tgr = {self.add('x1_re','Tfr')}, Tgi = {self.add('x1_im','Tfi')};")
        self.o(f"{T} Tvr = {self.sub('x3_re','x9_re')}, Tvi = {self.sub('x3_im','x9_im')};")
        self.o(f"{T} Tor = {self.fnma('KP500000000','Tfr','x1_re')}, Toi = {self.fnma('KP500000000','Tfi','x1_im')};")
        self.b()

        # === Block 4: x11, x6 -> T6, Tr; x7, x2 -> T9, Ts ===
        self.c("Block 4: x11,x6 -> T6,Tr; x7,x2 -> T9,Ts; then Ta,TI")
        self.o(f"{T} T6r = {self.sub('x11_re','x6_re')}, T6i = {self.sub('x11_im','x6_im')};")
        self.o(f"{T} Trr = {self.add('x11_re','x6_re')}, Tri = {self.add('x11_im','x6_im')};")
        self.o(f"{T} T9r = {self.sub('x7_re','x2_re')}, T9i = {self.sub('x7_im','x2_im')};")
        self.o(f"{T} Tsr = {self.add('x7_re','x2_re')}, Tsi = {self.add('x7_im','x2_im')};")
        self.o(f"{T} Tar = {self.add('T6r','T9r')}, Tai = {self.add('T6i','T9i')};")
        self.o(f"{T} TIr = {self.add('Trr','Tsr')}, TIi = {self.add('Tri','Tsi')};")
        self.b()

        # === Intermediate combinations ===
        self.c("Intermediate combinations: Tb, Tm, Tq, Tt, Tu, TC")
        self.o(f"{T} Tbr = {self.add('T3r','Tar')}, Tbi = {self.add('T3i','Tai')};")
        self.o(f"{T} Tmr = {self.sub('Tgr','Tlr')}, Tmi = {self.sub('Tgi','Tli')};")
        self.o(f"{T} Tqr = {self.sub('Tor','Tpr')}, Tqi = {self.sub('Toi','Tpi')};")
        # Tt = KP866025403 * (Tr - Ts)
        self.o(f"{T} Ttr = {self.mul('KP866025403',self.sub('Trr','Tsr'))}, Tti = {self.mul('KP866025403',self.sub('Tri','Tsi'))};")
        self.o(f"{T} Tur = {self.add('Tqr','Ttr')}, Tui = {self.add('Tqi','Tti')};")
        self.o(f"{T} TCr = {self.sub('Tqr','Ttr')}, TCi = {self.sub('Tqi','Tti')};")
        self.b()

        # === TP, TQ -> TR, TX; TG, TJ -> TK, TU ===
        self.c("TP, TQ -> TR, TX; TG, TJ -> TK, TU")
        self.o(f"{T} TPr = {self.add('Tgr','Tlr')}, TPi = {self.add('Tgi','Tli')};")
        self.o(f"{T} TQr = {self.add('THr','TIr')}, TQi = {self.add('THi','TIi')};")
        self.o(f"{T} TRr = {self.mul('KP300462606',self.sub('TPr','TQr'))}, TRi = {self.mul('KP300462606',self.sub('TPi','TQi'))};")
        self.o(f"{T} TXr = {self.add('TPr','TQr')}, TXi = {self.add('TPi','TQi')};")
        self.o(f"{T} TGr = {self.add('Tor','Tpr')}, TGi = {self.add('Toi','Tpi')};")
        self.o(f"{T} TJr = {self.fnma('KP500000000','TIr','THr')}, TJi = {self.fnma('KP500000000','TIi','THi')};")
        self.o(f"{T} TKr = {self.sub('TGr','TJr')}, TKi = {self.sub('TGi','TJi')};")
        self.o(f"{T} TUr = {self.add('TGr','TJr')}, TUi = {self.add('TGi','TJi')};")
        self.b()

        # === Tx, Ty, Tz, TB, TL, TM, TN, TT ===
        self.c("Tx, Ty, Tz, TB; TL, TM, TN, TT")
        # Tx = KP866025403 * (Tv - Tw)
        self.o(f"{T} Txr = {self.mul('KP866025403',self.sub('Tvr','Twr'))}, Txi = {self.mul('KP866025403',self.sub('Tvi','Twi'))};")
        # Ty = T3 - KP500000000 * Ta
        self.o(f"{T} Tyr = {self.fnma('KP500000000','Tar','T3r')}, Tyi = {self.fnma('KP500000000','Tai','T3i')};")
        self.o(f"{T} Tzr = {self.sub('Txr','Tyr')}, Tzi = {self.sub('Txi','Tyi')};")
        self.o(f"{T} TBr = {self.add('Txr','Tyr')}, TBi = {self.add('Txi','Tyi')};")
        self.o(f"{T} TLr = {self.add('Tvr','Twr')}, TLi = {self.add('Tvi','Twi')};")
        self.o(f"{T} TMr = {self.sub('T6r','T9r')}, TMi = {self.sub('T6i','T9i')};")
        self.o(f"{T} TNr = {self.sub('TLr','TMr')}, TNi = {self.sub('TLi','TMi')};")
        self.o(f"{T} TTr = {self.add('TLr','TMr')}, TTi = {self.add('TLi','TMi')};")
        self.b()

        # === Store y0 = TW + TX ===
        y0 = out_names[0]
        self.c("y0 = TW + TX")
        self.o(f"{y0}_re = {self.add('TWr','TXr')}; {y0}_im = {self.add('TWi','TXi')};")
        self.b()

        # === Second half: output computation ===
        # T17, T18 -> T19, T1n
        self.c("T17, T18 -> T19, T1n")
        self.o(f"{T} T17r = {self.fma('KP387390585','TNr',self.mul('KP265966249','TKr'))};")
        self.o(f"{T} T17i = {self.fma('KP387390585','TNi',self.mul('KP265966249','TKi'))};")
        self.o(f"{T} T18r = {self.fms('KP113854479','TTr',self.mul('KP503537032','TUr'))};")
        self.o(f"{T} T18i = {self.fms('KP113854479','TTi',self.mul('KP503537032','TUi'))};")
        self.o(f"{T} T19r = {self.sub('T17r','T18r')}, T19i = {self.sub('T17i','T18i')};")
        self.o(f"{T} T1nr = {self.add('T17r','T18r')}, T1ni = {self.add('T17i','T18i')};")
        self.b()

        # T14 block: T14, T11, T12, T13, T1f, T1k
        self.c("T14 block: T11, T12, T13 -> T14, T1f, T1k")
        self.o(f"{T} T14r = {self.fma('KP575140729','Tmr',self.mul('KP174138601','Tbr'))};")
        self.o(f"{T} T14i = {self.fma('KP575140729','Tmi',self.mul('KP174138601','Tbi'))};")
        self.o(f"{T} T11r = {self.fms('KP256247671','TCr',self.mul('KP156891391','TBr'))};")
        self.o(f"{T} T11i = {self.fms('KP256247671','TCi',self.mul('KP156891391','TBi'))};")
        self.o(f"{T} T12r = {self.fma('KP011599105','Tzr',self.mul('KP300238635','Tur'))};")
        self.o(f"{T} T12i = {self.fma('KP011599105','Tzi',self.mul('KP300238635','Tui'))};")
        self.o(f"{T} T13r = {self.sub('T11r','T12r')}, T13i = {self.sub('T11i','T12i')};")
        self.o(f"{T} T1fr = {self.add('T14r','T13r')}, T1fi = {self.add('T14i','T13i')};")
        self.o(f"{T} T1kr = {self.mul('KP1_732050807',self.add('T11r','T12r'))};")
        self.o(f"{T} T1ki = {self.mul('KP1_732050807',self.add('T11i','T12i'))};")
        self.b()

        # Tn, TE block: Tn, TA, TD -> TE, T1e, T1j
        self.c("Tn, TE block: Tn, TA, TD -> TE, T1e, T1j")
        self.o(f"{T} Tnr = {self.fms('KP575140729','Tbr',self.mul('KP174138601','Tmr'))};")
        self.o(f"{T} Tni = {self.fms('KP575140729','Tbi',self.mul('KP174138601','Tmi'))};")
        self.o(f"{T} TAr = {self.fms('KP011599105','Tur',self.mul('KP300238635','Tzr'))};")
        self.o(f"{T} TAi = {self.fms('KP011599105','Tui',self.mul('KP300238635','Tzi'))};")
        self.o(f"{T} TDr = {self.fma('KP256247671','TBr',self.mul('KP156891391','TCr'))};")
        self.o(f"{T} TDi = {self.fma('KP256247671','TBi',self.mul('KP156891391','TCi'))};")
        self.o(f"{T} TEr = {self.sub('TAr','TDr')}, TEi = {self.sub('TAi','TDi')};")
        self.o(f"{T} T1er = {self.mul('KP1_732050807',self.add('TDr','TAr'))};")
        self.o(f"{T} T1ei = {self.mul('KP1_732050807',self.add('TDi','TAi'))};")
        self.o(f"{T} T1jr = {self.sub('Tnr','TEr')}, T1ji = {self.sub('Tni','TEi')};")
        self.b()

        # TO, T1b, TV, TY, T1a -> TS, T1m, TZ, T1c
        self.c("TO, TV, TY -> TS, T1m, TZ, T1c, T1a, T1b")
        self.o(f"{T} TOr = {self.fms('KP258260390','TKr',self.mul('KP132983124','TNr'))};")
        self.o(f"{T} TOi = {self.fms('KP258260390','TKi',self.mul('KP132983124','TNi'))};")
        self.o(f"{T} T1br = {self.sub('TRr','TOr')}, T1bi = {self.sub('TRi','TOi')};")
        self.o(f"{T} TVr = {self.fma('KP251768516','TTr',self.mul('KP075902986','TUr'))};")
        self.o(f"{T} TVi = {self.fma('KP251768516','TTi',self.mul('KP075902986','TUi'))};")
        self.o(f"{T} TYr = {self.fnma('KP083333333','TXr','TWr')}, TYi = {self.fnma('KP083333333','TXi','TWi')};")
        self.o(f"{T} T1ar = {self.sub('TYr','TVr')}, T1ai = {self.sub('TYi','TVi')};")
        self.o(f"{T} TSr = {self.fma('KP2_000000000','TOr','TRr')}, TSi = {self.fma('KP2_000000000','TOi','TRi')};")
        self.o(f"{T} T1mr = {self.add('T1br','T1ar')}, T1mi = {self.add('T1bi','T1ai')};")
        self.o(f"{T} TZr = {self.fma('KP2_000000000','TVr','TYr')}, TZi = {self.fma('KP2_000000000','TVi','TYi')};")
        self.o(f"{T} T1cr = {self.sub('T1ar','T1br')}, T1ci = {self.sub('T1ai','T1bi')};")
        self.b()

        # === Output pair (1, 12): TF + T10 / T10 - TF ===
        self.c("Output pair (1, 12)")
        # TF_re = -(2*TE_im + Tn_im),  TF_im = 2*TE_re + Tn_re
        self.o(f"{T} TFr = {self.neg(self.fma('KP2_000000000','TEi','Tni'))};")
        self.o(f"{T} TFi = {self.fma('KP2_000000000','TEr','Tnr')};")
        self.o(f"{T} T10r = {self.add('TSr','TZr')}, T10i = {self.add('TSi','TZi')};")
        y1, y12 = out_names[1], out_names[12]
        self.o(f"{y1}_re = {self.add('TFr','T10r')}; {y1}_im = {self.add('TFi','T10i')};")
        self.o(f"{y12}_re = {self.sub('T10r','TFr')}; {y12}_im = {self.sub('T10i','TFi')};")
        self.b()

        # === Output pair (5, 8): T15 + T16 / T16 - T15 ===
        self.c("Output pair (5, 8)")
        # fwd: T15_re = -(2*T13_im - T14_im),  T15_im = 2*T13_re - T14_re
        self.o(f"{T} T15r = {self.neg(self.fms('KP2_000000000','T13i','T14i'))};")
        self.o(f"{T} T15i = {self.fms('KP2_000000000','T13r','T14r')};")
        self.o(f"{T} T16r = {self.sub('TZr','TSr')}, T16i = {self.sub('TZi','TSi')};")
        y5, y8 = out_names[5], out_names[8]
        self.o(f"{y5}_re = {self.add('T15r','T16r')}; {y5}_im = {self.add('T15i','T16i')};")
        self.o(f"{y8}_re = {self.sub('T16r','T15r')}; {y8}_im = {self.sub('T16i','T15i')};")
        self.b()

        # === Output pair (4, 9): T1p +/- T1q ===
        self.c("Output pair (4, 9)")
        self.o(f"{T} T1pr = {self.add('T1nr','T1mr')}, T1pi = {self.add('T1ni','T1mi')};")
        # T1q_re = -(T1j_im + T1k_im),  T1q_im = T1j_re + T1k_re
        self.o(f"{T} T1qr = {self.neg(self.add('T1ji','T1ki'))};")
        self.o(f"{T} T1qi = {self.add('T1jr','T1kr')};")
        y4, y9 = out_names[4], out_names[9]
        self.o(f"{y4}_re = {self.sub('T1pr','T1qr')}; {y4}_im = {self.sub('T1pi','T1qi')};")
        self.o(f"{y9}_re = {self.add('T1qr','T1pr')}; {y9}_im = {self.add('T1qi','T1pi')};")
        self.b()

        # === Output pair (3, 10): T1l + T1o / T1o - T1l ===
        self.c("Output pair (3, 10)")
        # T1l_re = -(T1j_im - T1k_im),  T1l_im = T1j_re - T1k_re
        self.o(f"{T} T1lr = {self.neg(self.sub('T1ji','T1ki'))};")
        self.o(f"{T} T1li = {self.sub('T1jr','T1kr')};")
        self.o(f"{T} T1or = {self.sub('T1mr','T1nr')}, T1oi = {self.sub('T1mi','T1ni')};")
        y3, y10 = out_names[3], out_names[10]
        self.o(f"{y3}_re = {self.add('T1lr','T1or')}; {y3}_im = {self.add('T1li','T1oi')};")
        self.o(f"{y10}_re = {self.sub('T1or','T1lr')}; {y10}_im = {self.sub('T1oi','T1li')};")
        self.b()

        # === Output pair (6, 7): T1h + T1i / T1i - T1h ===
        self.c("Output pair (6, 7)")
        # T1h_re = -(T1e_im - T1f_im),  T1h_im = T1e_re - T1f_re
        self.o(f"{T} T1hr = {self.neg(self.sub('T1ei','T1fi'))};")
        self.o(f"{T} T1hi = {self.sub('T1er','T1fr')};")
        self.o(f"{T} T1ir = {self.sub('T1cr','T19r')}, T1ii = {self.sub('T1ci','T19i')};")
        y6, y7 = out_names[6], out_names[7]
        self.o(f"{y6}_re = {self.add('T1hr','T1ir')}; {y6}_im = {self.add('T1hi','T1ii')};")
        self.o(f"{y7}_re = {self.sub('T1ir','T1hr')}; {y7}_im = {self.sub('T1ii','T1hi')};")
        self.b()

        # === Output pair (2, 11): T1d +/- T1g ===
        self.c("Output pair (2, 11)")
        self.o(f"{T} T1dr = {self.add('T19r','T1cr')}, T1di = {self.add('T19i','T1ci')};")
        # T1g_re = -(T1e_im + T1f_im),  T1g_im = T1e_re + T1f_re
        self.o(f"{T} T1gr = {self.neg(self.add('T1ei','T1fi'))};")
        self.o(f"{T} T1gi = {self.add('T1er','T1fr')};")
        y2, y11 = out_names[2], out_names[11]
        self.o(f"{y2}_re = {self.sub('T1dr','T1gr')}; {y2}_im = {self.sub('T1di','T1gi')};")
        self.o(f"{y11}_re = {self.add('T1gr','T1dr')}; {y11}_im = {self.add('T1gi','T1di')};")

    def _emit_radix13_butterfly_bwd(self, out_names):
        """Backward DFT-13 butterfly -- exact translation of genfft bwd kernel.

        The bwd kernel uses different variable names in the input-loading section
        but the SAME structure for the output cross-terms. Key differences:
        - Input pairs are loaded in a different order / naming
        - Some intermediate expressions swap coefficient positions
        - The cross real/imag terms have identical negation patterns
        """
        T = self.isa.T

        # === Load x0 (TW) ===
        self.c("TW = x0 (preserved)")
        self.o(f"{T} TWr = x0_re, TWi = x0_im;")
        self.b()

        # === Block 1: x8, x5 -> Te, TH ===
        self.c("Block 1: x8, x5 -> Te (diff), TH (sum)")
        self.o(f"{T} Ter = {self.sub('x8_re','x5_re')}, Tei = {self.sub('x8_im','x5_im')};")
        self.o(f"{T} THr = {self.add('x8_re','x5_re')}, THi = {self.add('x8_im','x5_im')};")
        self.b()

        # === Block 2: x12, x10, x4 -> Ta, Tu, Tp ===
        self.c("Block 2: x12, x10, x4 -> T9, Ta, Tp, Tu")
        self.o(f"{T} T9r = {self.add('x10_re','x4_re')}, T9i = {self.add('x10_im','x4_im')};")
        self.o(f"{T} Tar = {self.add('x12_re','T9r')}, Tai = {self.add('x12_im','T9i')};")
        self.o(f"{T} Tur = {self.fnma('KP500000000','T9r','x12_re')}, Tui = {self.fnma('KP500000000','T9i','x12_im')};")
        self.o(f"{T} Tpr = {self.sub('x10_re','x4_re')}, Tpi = {self.sub('x10_im','x4_im')};")
        self.b()

        # === Block 3: x1, x3, x9 -> T4, T5, Tt, To ===
        self.c("Block 3: x1, x3, x9 -> T4, T5, Tt, To")
        self.o(f"{T} T4r = {self.add('x3_re','x9_re')}, T4i = {self.add('x3_im','x9_im')};")
        self.o(f"{T} T5r = {self.add('x1_re','T4r')}, T5i = {self.add('x1_im','T4i')};")
        self.o(f"{T} Ttr = {self.fnma('KP500000000','T4r','x1_re')}, Tti = {self.fnma('KP500000000','T4i','x1_im')};")
        self.o(f"{T} Tor = {self.sub('x3_re','x9_re')}, Toi = {self.sub('x3_im','x9_im')};")
        self.b()

        # === Block 4: x11, x6 -> Th, Tw; x7, x2 -> Tk, Tx ===
        self.c("Block 4: x11,x6 -> Th,Tw; x7,x2 -> Tk,Tx; then Tl,TI")
        self.o(f"{T} Thr = {self.sub('x11_re','x6_re')}, Thi = {self.sub('x11_im','x6_im')};")
        self.o(f"{T} Twr = {self.add('x11_re','x6_re')}, Twi = {self.add('x11_im','x6_im')};")
        self.o(f"{T} Tkr = {self.sub('x7_re','x2_re')}, Tki = {self.sub('x7_im','x2_im')};")
        self.o(f"{T} Txr = {self.add('x7_re','x2_re')}, Txi = {self.add('x7_im','x2_im')};")
        self.o(f"{T} Tlr = {self.add('Thr','Tkr')}, Tli = {self.add('Thi','Tki')};")
        self.o(f"{T} TIr = {self.add('Twr','Txr')}, TIi = {self.add('Twi','Txi')};")
        self.b()

        # === Intermediate combinations (bwd) ===
        self.c("Intermediate: Tb, Tm, Tq, Tr, Ts, TB")
        self.o(f"{T} Tbr = {self.sub('T5r','Tar')}, Tbi = {self.sub('T5i','Tai')};")
        self.o(f"{T} Tmr = {self.add('Ter','Tlr')}, Tmi = {self.add('Tei','Tli')};")
        # Tq = KP866025403 * (To - Tp)
        self.o(f"{T} Tqr = {self.mul('KP866025403',self.sub('Tor','Tpr'))}, Tqi = {self.mul('KP866025403',self.sub('Toi','Tpi'))};")
        # Tr = Te - KP500000000 * Tl
        self.o(f"{T} Trr = {self.fnma('KP500000000','Tlr','Ter')}, Tri = {self.fnma('KP500000000','Tli','Tei')};")
        self.o(f"{T} Tsr = {self.add('Tqr','Trr')}, Tsi = {self.add('Tqi','Tri')};")
        self.o(f"{T} TBr = {self.sub('Tqr','Trr')}, TBi = {self.sub('Tqi','Tri')};")
        self.b()

        # === TP, TQ -> TR, TX; TG, TJ -> TK, TU ===
        self.c("TP, TQ -> TR, TX; TG, TJ -> TK, TU")
        self.o(f"{T} TPr = {self.add('T5r','Tar')}, TPi = {self.add('T5i','Tai')};")
        self.o(f"{T} TQr = {self.add('THr','TIr')}, TQi = {self.add('THi','TIi')};")
        self.o(f"{T} TRr = {self.mul('KP300462606',self.sub('TPr','TQr'))}, TRi = {self.mul('KP300462606',self.sub('TPi','TQi'))};")
        self.o(f"{T} TXr = {self.add('TPr','TQr')}, TXi = {self.add('TPi','TQi')};")
        self.o(f"{T} TGr = {self.add('Ttr','Tur')}, TGi = {self.add('Tti','Tui')};")
        self.o(f"{T} TJr = {self.fnma('KP500000000','TIr','THr')}, TJi = {self.fnma('KP500000000','TIi','THi')};")
        self.o(f"{T} TKr = {self.sub('TGr','TJr')}, TKi = {self.sub('TGi','TJi')};")
        self.o(f"{T} TUr = {self.add('TGr','TJr')}, TUi = {self.add('TGi','TJi')};")
        self.b()

        # === Tv, Ty, Tz, TC; TL, TM, TN, TT ===
        self.c("Tv, Ty, Tz, TC; TL, TM, TN, TT")
        self.o(f"{T} Tvr = {self.sub('Ttr','Tur')}, Tvi = {self.sub('Tti','Tui')};")
        # Ty = KP866025403 * (Tw - Tx)
        self.o(f"{T} Tyr = {self.mul('KP866025403',self.sub('Twr','Txr'))}, Tyi = {self.mul('KP866025403',self.sub('Twi','Txi'))};")
        self.o(f"{T} Tzr = {self.sub('Tvr','Tyr')}, Tzi = {self.sub('Tvi','Tyi')};")
        self.o(f"{T} TCr = {self.add('Tvr','Tyr')}, TCi = {self.add('Tvi','Tyi')};")
        self.o(f"{T} TLr = {self.add('Tor','Tpr')}, TLi = {self.add('Toi','Tpi')};")
        self.o(f"{T} TMr = {self.sub('Thr','Tkr')}, TMi = {self.sub('Thi','Tki')};")
        self.o(f"{T} TNr = {self.sub('TLr','TMr')}, TNi = {self.sub('TLi','TMi')};")
        self.o(f"{T} TTr = {self.add('TLr','TMr')}, TTi = {self.add('TLi','TMi')};")
        self.b()

        # === Store y0 = TW + TX ===
        y0 = out_names[0]
        self.c("y0 = TW + TX")
        self.o(f"{y0}_re = {self.add('TWr','TXr')}; {y0}_im = {self.add('TWi','TXi')};")
        self.b()

        # === Second half: output computation (bwd) ===
        # T1a, T1b -> T1c, T1n
        self.c("T1a, T1b -> T1c, T1n")
        self.o(f"{T} T1ar = {self.fma('KP387390585','TNr',self.mul('KP265966249','TKr'))};")
        self.o(f"{T} T1ai = {self.fma('KP387390585','TNi',self.mul('KP265966249','TKi'))};")
        self.o(f"{T} T1br = {self.fms('KP113854479','TTr',self.mul('KP503537032','TUr'))};")
        self.o(f"{T} T1bi = {self.fms('KP113854479','TTi',self.mul('KP503537032','TUi'))};")
        self.o(f"{T} T1cr = {self.sub('T1ar','T1br')}, T1ci = {self.sub('T1ai','T1bi')};")
        self.o(f"{T} T1nr = {self.add('T1ar','T1br')}, T1ni = {self.add('T1ai','T1bi')};")
        self.b()

        # T11, T12, T13 -> T14, T17, T1k (bwd differences from fwd)
        self.c("T11, T12, T13 -> T14, T17, T1k (bwd)")
        self.o(f"{T} T11r = {self.fma('KP575140729','Tbr',self.mul('KP174138601','Tmr'))};")
        self.o(f"{T} T11i = {self.fma('KP575140729','Tbi',self.mul('KP174138601','Tmi'))};")
        # bwd T12 = KP156891391*Ts - KP256247671*Tz  (swapped from fwd)
        self.o(f"{T} T12r = {self.fms('KP156891391','Tsr',self.mul('KP256247671','Tzr'))};")
        self.o(f"{T} T12i = {self.fms('KP156891391','Tsi',self.mul('KP256247671','Tzi'))};")
        # bwd T13 = KP011599105*TB + KP300238635*TC  (swapped from fwd)
        self.o(f"{T} T13r = {self.fma('KP011599105','TBr',self.mul('KP300238635','TCr'))};")
        self.o(f"{T} T13i = {self.fma('KP011599105','TBi',self.mul('KP300238635','TCi'))};")
        self.o(f"{T} T14r = {self.add('T12r','T13r')}, T14i = {self.add('T12i','T13i')};")
        self.o(f"{T} T17r = {self.sub('T11r','T14r')}, T17i = {self.sub('T11i','T14i')};")
        # bwd T1k = KP1_732050807 * (T12 - T13)  (swapped sign from fwd)
        self.o(f"{T} T1kr = {self.mul('KP1_732050807',self.sub('T12r','T13r'))};")
        self.o(f"{T} T1ki = {self.mul('KP1_732050807',self.sub('T12i','T13i'))};")
        self.b()

        # Tn, TA, TD -> TE, T18, T1j (bwd)
        self.c("Tn, TA, TD -> TE, T18, T1j (bwd)")
        # bwd Tn = KP174138601*Tb - KP575140729*Tm  (swapped from fwd)
        self.o(f"{T} Tnr = {self.fms('KP174138601','Tbr',self.mul('KP575140729','Tmr'))};")
        self.o(f"{T} Tni = {self.fms('KP174138601','Tbi',self.mul('KP575140729','Tmi'))};")
        # bwd TA = KP256247671*Ts + KP156891391*Tz  (swapped from fwd)
        self.o(f"{T} TAr = {self.fma('KP256247671','Tsr',self.mul('KP156891391','Tzr'))};")
        self.o(f"{T} TAi = {self.fma('KP256247671','Tsi',self.mul('KP156891391','Tzi'))};")
        # bwd TD = KP300238635*TB - KP011599105*TC  (swapped from fwd)
        self.o(f"{T} TDr = {self.fms('KP300238635','TBr',self.mul('KP011599105','TCr'))};")
        self.o(f"{T} TDi = {self.fms('KP300238635','TBi',self.mul('KP011599105','TCi'))};")
        self.o(f"{T} TEr = {self.add('TAr','TDr')}, TEi = {self.add('TAi','TDi')};")
        # bwd T18 = KP1_732050807 * (TD - TA)  (swapped sign from fwd)
        self.o(f"{T} T18r = {self.mul('KP1_732050807',self.sub('TDr','TAr'))};")
        self.o(f"{T} T18i = {self.mul('KP1_732050807',self.sub('TDi','TAi'))};")
        self.o(f"{T} T1jr = {self.sub('Tnr','TEr')}, T1ji = {self.sub('Tni','TEi')};")
        self.b()

        # TO, T1e, TV, TY, T1d -> TS, T1m, TZ, T1f
        self.c("TO, TV, TY -> TS, T1m, TZ, T1f, T1d, T1e")
        self.o(f"{T} TOr = {self.fms('KP258260390','TKr',self.mul('KP132983124','TNr'))};")
        self.o(f"{T} TOi = {self.fms('KP258260390','TKi',self.mul('KP132983124','TNi'))};")
        self.o(f"{T} T1er = {self.sub('TRr','TOr')}, T1ei = {self.sub('TRi','TOi')};")
        self.o(f"{T} TVr = {self.fma('KP251768516','TTr',self.mul('KP075902986','TUr'))};")
        self.o(f"{T} TVi = {self.fma('KP251768516','TTi',self.mul('KP075902986','TUi'))};")
        self.o(f"{T} TYr = {self.fnma('KP083333333','TXr','TWr')}, TYi = {self.fnma('KP083333333','TXi','TWi')};")
        self.o(f"{T} T1dr = {self.sub('TYr','TVr')}, T1di = {self.sub('TYi','TVi')};")
        self.o(f"{T} TSr = {self.fma('KP2_000000000','TOr','TRr')}, TSi = {self.fma('KP2_000000000','TOi','TRi')};")
        self.o(f"{T} T1mr = {self.add('T1er','T1dr')}, T1mi = {self.add('T1ei','T1di')};")
        self.o(f"{T} TZr = {self.fma('KP2_000000000','TVr','TYr')}, TZi = {self.fma('KP2_000000000','TVi','TYi')};")
        self.o(f"{T} T1fr = {self.sub('T1dr','T1er')}, T1fi = {self.sub('T1di','T1ei')};")
        self.b()

        # === Output pair (1, 12): TF + T10 / T10 - TF ===
        self.c("Output pair (1, 12)")
        # bwd: same cross-term structure as fwd
        self.o(f"{T} TFr = {self.neg(self.fma('KP2_000000000','TEi','Tni'))};")
        self.o(f"{T} TFi = {self.fma('KP2_000000000','TEr','Tnr')};")
        self.o(f"{T} T10r = {self.add('TSr','TZr')}, T10i = {self.add('TSi','TZi')};")
        y1, y12 = out_names[1], out_names[12]
        self.o(f"{y1}_re = {self.add('TFr','T10r')}; {y1}_im = {self.add('TFi','T10i')};")
        self.o(f"{y12}_re = {self.sub('T10r','TFr')}; {y12}_im = {self.sub('T10i','TFi')};")
        self.b()

        # === Output pair (5, 8) ===
        self.c("Output pair (5, 8)")
        # bwd: T15_re = -(2*T14_im + T11_im),  T15_im = 2*T14_re + T11_re
        self.o(f"{T} T15r = {self.neg(self.fma('KP2_000000000','T14i','T11i'))};")
        self.o(f"{T} T15i = {self.fma('KP2_000000000','T14r','T11r')};")
        self.o(f"{T} T16r = {self.sub('TZr','TSr')}, T16i = {self.sub('TZi','TSi')};")
        y5, y8 = out_names[5], out_names[8]
        self.o(f"{y5}_re = {self.add('T15r','T16r')}; {y5}_im = {self.add('T15i','T16i')};")
        self.o(f"{y8}_re = {self.sub('T16r','T15r')}; {y8}_im = {self.sub('T16i','T15i')};")
        self.b()

        # === Output pair (4, 9) ===
        self.c("Output pair (4, 9)")
        self.o(f"{T} T1pr = {self.add('T1nr','T1mr')}, T1pi = {self.add('T1ni','T1mi')};")
        # T1q_re = -(T1j_im + T1k_im),  T1q_im = T1j_re + T1k_re
        self.o(f"{T} T1qr = {self.neg(self.add('T1ji','T1ki'))};")
        self.o(f"{T} T1qi = {self.add('T1jr','T1kr')};")
        y4, y9 = out_names[4], out_names[9]
        self.o(f"{y4}_re = {self.sub('T1pr','T1qr')}; {y4}_im = {self.sub('T1pi','T1qi')};")
        self.o(f"{y9}_re = {self.add('T1qr','T1pr')}; {y9}_im = {self.add('T1qi','T1pi')};")
        self.b()

        # === Output pair (3, 10) ===
        self.c("Output pair (3, 10)")
        # T1l_re = -(T1j_im - T1k_im),  T1l_im = T1j_re - T1k_re
        self.o(f"{T} T1lr = {self.neg(self.sub('T1ji','T1ki'))};")
        self.o(f"{T} T1li = {self.sub('T1jr','T1kr')};")
        self.o(f"{T} T1or = {self.sub('T1mr','T1nr')}, T1oi = {self.sub('T1mi','T1ni')};")
        y3, y10 = out_names[3], out_names[10]
        self.o(f"{y3}_re = {self.add('T1lr','T1or')}; {y3}_im = {self.add('T1li','T1oi')};")
        self.o(f"{y10}_re = {self.sub('T1or','T1lr')}; {y10}_im = {self.sub('T1oi','T1li')};")
        self.b()

        # === Output pair (6, 7) ===
        self.c("Output pair (6, 7)")
        # bwd: T1h_re = -(T18_im + T17_im),  T1h_im = T18_re + T17_re
        self.o(f"{T} T1hr = {self.neg(self.add('T18i','T17i'))};")
        self.o(f"{T} T1hi = {self.add('T18r','T17r')};")
        self.o(f"{T} T1ir = {self.sub('T1fr','T1cr')}, T1ii = {self.sub('T1fi','T1ci')};")
        y6, y7 = out_names[6], out_names[7]
        self.o(f"{y6}_re = {self.add('T1hr','T1ir')}; {y6}_im = {self.add('T1hi','T1ii')};")
        self.o(f"{y7}_re = {self.sub('T1ir','T1hr')}; {y7}_im = {self.sub('T1ii','T1hi')};")
        self.b()

        # === Output pair (2, 11) ===
        self.c("Output pair (2, 11)")
        # bwd: T19_re = -(T17_im - T18_im),  T19_im = T17_re - T18_re
        self.o(f"{T} T19r = {self.neg(self.sub('T17i','T18i'))};")
        self.o(f"{T} T19i = {self.sub('T17r','T18r')};")
        self.o(f"{T} T1gr = {self.add('T1cr','T1fr')}, T1gi = {self.add('T1ci','T1fi')};")
        y2, y11 = out_names[2], out_names[11]
        self.o(f"{y2}_re = {self.add('T19r','T1gr')}; {y2}_im = {self.add('T19i','T1gi')};")
        self.o(f"{y11}_re = {self.sub('T1gr','T19r')}; {y11}_im = {self.sub('T1gi','T19i')};")


# ================================================================
# HELPERS: constants
# ================================================================

def emit_dft13_constants(em):
    """Emit the 21 DFT-13 constants + sign_flip as SIMD broadcasts or scalars."""
    T = em.isa.T
    vals = [
        ('KP500000000', KP500000000_val),
        ('KP866025403', KP866025403_val),
        ('KP300462606', KP300462606_val),
        ('KP387390585', KP387390585_val),
        ('KP265966249', KP265966249_val),
        ('KP113854479', KP113854479_val),
        ('KP503537032', KP503537032_val),
        ('KP575140729', KP575140729_val),
        ('KP174138601', KP174138601_val),
        ('KP256247671', KP256247671_val),
        ('KP156891391', KP156891391_val),
        ('KP011599105', KP011599105_val),
        ('KP300238635', KP300238635_val),
        ('KP1_732050807', KP1_732050807_val),
        ('KP258260390', KP258260390_val),
        ('KP132983124', KP132983124_val),
        ('KP251768516', KP251768516_val),
        ('KP075902986', KP075902986_val),
        ('KP083333333', KP083333333_val),
        ('KP2_000000000', KP2_000000000_val),
    ]
    if em.isa.name == 'scalar':
        for name, val in vals:
            em.o(f"const double {name} = {val:+.45f};")
    else:
        set1 = f"{em.isa.p}_set1_pd"
        em.o(f"const {T} sign_flip = {set1}(-0.0);")
        for name, val in vals:
            em.o(f"const {T} {name} = {set1}({val:+.45f});")
        em.o(f"(void)sign_flip;  /* used by neg() */")
    em.b()


def emit_dft13_constants_raw(lines, isa, indent=1):
    """Append DFT-13 constant declarations to a line list."""
    pad = "    " * indent
    T = isa.T
    vals = [
        ('KP500000000', KP500000000_val),
        ('KP866025403', KP866025403_val),
        ('KP300462606', KP300462606_val),
        ('KP387390585', KP387390585_val),
        ('KP265966249', KP265966249_val),
        ('KP113854479', KP113854479_val),
        ('KP503537032', KP503537032_val),
        ('KP575140729', KP575140729_val),
        ('KP174138601', KP174138601_val),
        ('KP256247671', KP256247671_val),
        ('KP156891391', KP156891391_val),
        ('KP011599105', KP011599105_val),
        ('KP300238635', KP300238635_val),
        ('KP1_732050807', KP1_732050807_val),
        ('KP258260390', KP258260390_val),
        ('KP132983124', KP132983124_val),
        ('KP251768516', KP251768516_val),
        ('KP075902986', KP075902986_val),
        ('KP083333333', KP083333333_val),
        ('KP2_000000000', KP2_000000000_val),
    ]
    if isa.name == 'scalar':
        for name, val in vals:
            lines.append(f"{pad}const double {name} = {val:+.45f};")
    else:
        set1 = f"{isa.p}_set1_pd"
        lines.append(f"{pad}const {T} sign_flip = {set1}(-0.0);")
        for name, val in vals:
            lines.append(f"{pad}const {T} {name} = {set1}({val:+.45f});")
        lines.append(f"{pad}(void)sign_flip;")


# ================================================================
# KERNEL BODY EMITTERS
# ================================================================

def emit_kernel_body(em, d, variant):
    """Emit the inner loop body for notw, dit_tw, dif_tw."""
    T = em.isa.T

    # Load all 13 inputs
    for n in range(R):
        em.emit_load(f"x{n}", n)

    # DIT: twiddle inputs 1..12 before butterfly
    if variant == 'dit_tw':
        for n in range(1, R):
            em.emit_ext_tw(f"x{n}", n - 1, d)
    elif variant == 'dit_tw_scalar':
        for n in range(1, R):
            em.emit_ext_tw_scalar(f"x{n}", n - 1, d)

    em.b()
    em.emit_radix13_butterfly(d)
    em.b()

    # DIF: twiddle outputs 1..12 after butterfly
    if variant == 'dif_tw':
        for m in range(1, R):
            em.emit_ext_tw(f"x{m}", m - 1, d)
    elif variant == 'dif_tw_scalar':
        for m in range(1, R):
            em.emit_ext_tw_scalar(f"x{m}", m - 1, d)

    # Store all 13 outputs
    for m in range(R):
        em.emit_store(f"x{m}", m)


def emit_kernel_body_log3(em, d, variant):
    """Emit the log3 variant: derive w2..w12 from w1 (11 cmuls)."""
    T = em.isa.T
    is_dit = variant == 'dit_tw_log3'

    em.c("Load base twiddle w1, derive w2..w12 (11 cmuls)")
    tb, tbi = em._tw_buf(), em._tw_buf_im()
    ta = em._tw_addr(0)
    if em.isa.name == 'scalar':
        em.o(f"const double w1r = {tb}[{ta}], w1i = {tbi}[{ta}];")
    else:
        em.o(f"const {T} w1r = LD(&{tb}[{ta}]), w1i = LD(&{tbi}[{ta}]);")
    em.n_load += 2

    # w2 = w1^2
    em.o(f"{T} w2r, w2i;")
    em.emit_cmul("w2r", "w2i", "w1r", "w1i", "w1r", "w1i", 'fwd')

    # w3 = w1 * w2
    em.o(f"{T} w3r, w3i;")
    em.emit_cmul("w3r", "w3i", "w1r", "w1i", "w2r", "w2i", 'fwd')

    # w4 = w2^2
    em.o(f"{T} w4r, w4i;")
    em.emit_cmul("w4r", "w4i", "w2r", "w2i", "w2r", "w2i", 'fwd')

    # w5 = w2 * w3
    em.o(f"{T} w5r, w5i;")
    em.emit_cmul("w5r", "w5i", "w2r", "w2i", "w3r", "w3i", 'fwd')

    # w6 = w3^2
    em.o(f"{T} w6r, w6i;")
    em.emit_cmul("w6r", "w6i", "w3r", "w3i", "w3r", "w3i", 'fwd')

    # w7 = w3 * w4
    em.o(f"{T} w7r, w7i;")
    em.emit_cmul("w7r", "w7i", "w3r", "w3i", "w4r", "w4i", 'fwd')

    # w8 = w4^2
    em.o(f"{T} w8r, w8i;")
    em.emit_cmul("w8r", "w8i", "w4r", "w4i", "w4r", "w4i", 'fwd')

    # w9 = w4 * w5
    em.o(f"{T} w9r, w9i;")
    em.emit_cmul("w9r", "w9i", "w4r", "w4i", "w5r", "w5i", 'fwd')

    # w10 = w5^2
    em.o(f"{T} w10r, w10i;")
    em.emit_cmul("w10r", "w10i", "w5r", "w5i", "w5r", "w5i", 'fwd')

    # w11 = w5 * w6
    em.o(f"{T} w11r, w11i;")
    em.emit_cmul("w11r", "w11i", "w5r", "w5i", "w6r", "w6i", 'fwd')

    # w12 = w6^2
    em.o(f"{T} w12r, w12i;")
    em.emit_cmul("w12r", "w12i", "w6r", "w6i", "w6r", "w6i", 'fwd')
    em.b()

    # Load inputs
    for n in range(R):
        em.emit_load(f"x{n}", n)

    # DIT: apply twiddles to x1..x12
    if is_dit:
        for n in range(1, R):
            em.emit_cmul_inplace(f"x{n}", f"w{n}r", f"w{n}i", d)
    em.b()

    em.emit_radix13_butterfly(d)
    em.b()

    # DIF: apply twiddles to outputs x1..x12
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
        func_base = 'radix13_n1_dit_kernel'
    elif variant == 'dit_tw':
        func_base = 'radix13_tw_flat_dit_kernel'
        tw_params = 'flat'
    elif variant == 'dit_tw_scalar':
        func_base = 'radix13_tw_flat_dit_kernel'
        tw_params = 'flat'
    elif variant == 'dif_tw':
        func_base = 'radix13_tw_flat_dif_kernel'
        tw_params = 'flat'
    elif variant == 'dif_tw_scalar':
        func_base = 'radix13_tw_flat_dif_kernel'
        tw_params = 'flat'
    elif variant == 'dit_tw_log3':
        func_base = 'radix13_tw_log3_dit_kernel'
        tw_params = 'flat'
    elif variant == 'dif_tw_log3':
        func_base = 'radix13_tw_log3_dif_kernel'
        tw_params = 'flat'

    vname = {
        'notw': 'N1 (no twiddle)',
        'dit_tw': 'DIT twiddled (flat)',
        'dif_tw': 'DIF twiddled (flat)',
        'dit_tw_log3': 'DIT twiddled (log3 derived)',
        'dif_tw_log3': 'DIF twiddled (log3 derived)',
    }[variant]
    guard = f"FFT_RADIX13_{isa.name.upper()}_{variant.upper()}_H"

    em.L.append(f"/**")
    em.L.append(f" * @file fft_radix13_{isa.name}_{variant}.h")
    em.L.append(f" * @brief DFT-13 {isa.name.upper()} {vname}")
    em.L.append(f" *")
    em.L.append(f" * Monolithic DFT-13 butterfly (genfft DAG), 21 constants.")
    em.L.append(f" * Register pressure exceeds both AVX2 and AVX-512;")
    em.L.append(f" * compiler handles register allocation / spills.")
    em.L.append(f" * k-step={isa.k_step}")
    em.L.append(f" * Generated by gen_radix13.py")
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
        emit_dft13_constants(em)

        # Working registers for all 13 inputs/outputs
        em.o(f"{T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
        em.o(f"{T} x4_re,x4_im,x5_re,x5_im,x6_re,x6_im,x7_re,x7_im;")
        em.o(f"{T} x8_re,x8_im,x9_re,x9_im,x10_re,x10_im,x11_re,x11_im;")
        em.o(f"{T} x12_re,x12_im;")
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

    # No U=2 for R=13: register pressure (48+ regs per pipeline) exceeds AVX-512 (32 ZMM).
    # U=2 would cause 96+ regs across both pipelines — 3x capacity. The compiler would
    # generate worse spill code for U=2 than the already-heavy U=1 spill pattern.

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
    is_t1_dit = ct_variant == 'ct_t1_dit'
    is_t1s_dit = ct_variant == 'ct_t1s_dit'
    is_t1_dit_log3 = ct_variant == 'ct_t1_dit_log3'
    is_t1_dif = ct_variant == 'ct_t1_dif'
    em.addr_mode = 'n1' if is_n1 else 't1'

    if is_n1:
        func_base = "radix13_n1"
        vname = "n1 (separate is/os)"
    elif is_t1_dif:
        func_base = "radix13_t1_dif"
        vname = "t1 DIF (in-place twiddle)"
    elif is_t1s_dit:
        func_base = "radix13_t1s_dit"
        vname = "t1s DIT (in-place, scalar broadcast twiddle)"
    elif is_t1_dit_log3:
        func_base = "radix13_t1_dit_log3"
        vname = "t1 DIT log3 (in-place, derived twiddles)"
    else:
        func_base = "radix13_t1_dit"
        vname = "t1 DIT (in-place twiddle)"

    guard = f"FFT_RADIX13_{isa.name.upper()}_CT_{ct_variant.upper()}_H"

    em.L.append(f"/**")
    em.L.append(f" * @file fft_radix13_{isa.name}_{ct_variant}.h")
    em.L.append(f" * @brief DFT-13 {isa.name.upper()} {vname}")
    em.L.append(f" *")
    em.L.append(f" * FFTW-style codelet for recursive CT executor.")
    em.L.append(f" * Generated by gen_radix13.py --variant {ct_variant}")
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
        em.addr_mode = 'n1' if is_n1 else 't1'

        if isa.target:
            em.L.append(f"static {isa.target} void")
        else:
            em.L.append(f"static void")

        if is_n1:
            em.L.append(f"{func_base}_{d}_{isa.name}(")
            em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
            em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
            if isa.name == 'scalar':
                em.L.append(f"    size_t is, size_t os, size_t vl, size_t ivs, size_t ovs)")
            else:
                em.L.append(f"    size_t is, size_t os, size_t vl)")
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

        emit_dft13_constants(em)

        em.o(f"{T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
        em.o(f"{T} x4_re,x4_im,x5_re,x5_im,x6_re,x6_im,x7_re,x7_im;")
        em.o(f"{T} x8_re,x8_im,x9_re,x9_im,x10_re,x10_im,x11_re,x11_im;")
        em.o(f"{T} x12_re,x12_im;")
        em.b()

        # Hoist twiddle broadcasts before the loop (t1s only)
        if is_t1s_dit:
            em.tw_hoisted = True
            em.emit_hoist_all_tw_scalars(R)
            em.b()

        # Loop
        if is_n1:
            if isa.name == 'scalar':
                em.o(f"for (size_t k = 0; k < vl; k++) {{")
            else:
                em.o(f"for (size_t k = 0; k < vl; k += {isa.k_step}) {{")
        else:  # t1
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
            kernel_variant = 'notw' if is_n1 else ('dif_tw' if is_t1_dif else 'dit_tw')
            emit_kernel_body(em, d, kernel_variant)
        em.ind -= 1
        em.o("}")
        em.L.append("}")
        em.L.append("")

    # n1_ovs: butterfly with fused SIMD transpose stores (for R=13)
    # 13 bins = 3 groups of 4 (bins 0-3, 4-7, 8-11) + bin 12 leftover
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
            em.L.append(f"radix13_n1_ovs_{d}_{isa.name}(")
            em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
            em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
            em.L.append(f"    size_t is, size_t os, size_t vl, size_t ovs)")
            em.L.append(f"{{")

            em.L.append(f"    /* n1_ovs: butterfly -> tbuf, then 4x4 transpose (bins 0-3, 4-7, 8-11) + scatter (bin 12) */")
            em.L.append(f"    {isa.align} double tbuf_re[{R * VL}];")
            em.L.append(f"    {isa.align} double tbuf_im[{R * VL}];")

            emit_dft13_constants_raw(em.L, isa, indent=1)

            em.L.append(f"    {T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
            em.L.append(f"    {T} x4_re,x4_im,x5_re,x5_im,x6_re,x6_im,x7_re,x7_im;")
            em.L.append(f"    {T} x8_re,x8_im,x9_re,x9_im,x10_re,x10_im,x11_re,x11_im;")
            em.L.append(f"    {T} x12_re,x12_im;")
            em.L.append(f"")

            em.L.append(f"    for (size_t k = 0; k < vl; k += {isa.k_step}) {{")

            # Use fresh Emitter for kernel body
            em2.L = []
            em2.ind = 2
            em2.reset()
            em2.addr_mode = 'n1_ovs'
            emit_kernel_body(em2, d, 'notw')
            em.L.extend(em2.L)

            # 4x4 transpose: bins 0-3
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
                    for j in range(VL):
                        for b in range(4):
                            em.L.append(f"          {arr}[(k+{j})*ovs+os*{b}] = {bname}[{b}*{VL}+{j}];")
                em.L.append(f"        }}")

            # 4x4 transpose: bins 4-7
            em.L.append(f"        /* 4x4 transpose: bins 4-7 -> output at stride ovs */")
            for comp, arr in [('re', 'out_re'), ('im', 'out_im')]:
                bname = f"tbuf_{comp}"
                em.L.append(f"        {{ {T} a_=LD(&{bname}[4*{VL}]), b_=LD(&{bname}[5*{VL}]);")
                em.L.append(f"          {T} c_=LD(&{bname}[6*{VL}]), d_=LD(&{bname}[7*{VL}]);")
                if isa.name == 'avx2':
                    em.L.append(f"          {T} lo_ab=_mm256_unpacklo_pd(a_,b_), hi_ab=_mm256_unpackhi_pd(a_,b_);")
                    em.L.append(f"          {T} lo_cd=_mm256_unpacklo_pd(c_,d_), hi_cd=_mm256_unpackhi_pd(c_,d_);")
                    em.L.append(f"          _mm256_storeu_pd(&{arr}[(k+0)*ovs+os*4], _mm256_permute2f128_pd(lo_ab,lo_cd,0x20));")
                    em.L.append(f"          _mm256_storeu_pd(&{arr}[(k+1)*ovs+os*4], _mm256_permute2f128_pd(hi_ab,hi_cd,0x20));")
                    em.L.append(f"          _mm256_storeu_pd(&{arr}[(k+2)*ovs+os*4], _mm256_permute2f128_pd(lo_ab,lo_cd,0x31));")
                    em.L.append(f"          _mm256_storeu_pd(&{arr}[(k+3)*ovs+os*4], _mm256_permute2f128_pd(hi_ab,hi_cd,0x31));")
                else:  # avx512
                    for j in range(VL):
                        for b in range(4):
                            em.L.append(f"          {arr}[(k+{j})*ovs+os*{b+4}] = {bname}[{b+4}*{VL}+{j}];")
                em.L.append(f"        }}")

            # 4x4 transpose: bins 8-11
            em.L.append(f"        /* 4x4 transpose: bins 8-11 -> output at stride ovs */")
            for comp, arr in [('re', 'out_re'), ('im', 'out_im')]:
                bname = f"tbuf_{comp}"
                em.L.append(f"        {{ {T} a_=LD(&{bname}[8*{VL}]), b_=LD(&{bname}[9*{VL}]);")
                em.L.append(f"          {T} c_=LD(&{bname}[10*{VL}]), d_=LD(&{bname}[11*{VL}]);")
                if isa.name == 'avx2':
                    em.L.append(f"          {T} lo_ab=_mm256_unpacklo_pd(a_,b_), hi_ab=_mm256_unpackhi_pd(a_,b_);")
                    em.L.append(f"          {T} lo_cd=_mm256_unpacklo_pd(c_,d_), hi_cd=_mm256_unpackhi_pd(c_,d_);")
                    em.L.append(f"          _mm256_storeu_pd(&{arr}[(k+0)*ovs+os*8], _mm256_permute2f128_pd(lo_ab,lo_cd,0x20));")
                    em.L.append(f"          _mm256_storeu_pd(&{arr}[(k+1)*ovs+os*8], _mm256_permute2f128_pd(hi_ab,hi_cd,0x20));")
                    em.L.append(f"          _mm256_storeu_pd(&{arr}[(k+2)*ovs+os*8], _mm256_permute2f128_pd(lo_ab,lo_cd,0x31));")
                    em.L.append(f"          _mm256_storeu_pd(&{arr}[(k+3)*ovs+os*8], _mm256_permute2f128_pd(hi_ab,hi_cd,0x31));")
                else:  # avx512
                    for j in range(VL):
                        for b in range(4):
                            em.L.append(f"          {arr}[(k+{j})*ovs+os*{b+8}] = {bname}[{b+8}*{VL}+{j}];")
                em.L.append(f"        }}")

            # Bin 12: extract from tbuf via YMM load + scalar scatter
            em.L.append(f"        /* Bin 12: extract from tbuf -> scatter */")
            if isa.name == 'avx2':
                for comp, arr in [('re', 'out_re'), ('im', 'out_im')]:
                    bname = f"tbuf_{comp}"
                    em.L.append(f"        {{ __m256d v=LD(&{bname}[12*{VL}]);")
                    em.L.append(f"          __m128d lo=_mm256_castpd256_pd128(v), hi=_mm256_extractf128_pd(v,1);")
                    em.L.append(f"          _mm_storel_pd(&{arr}[(k+0)*ovs+os*12], lo);")
                    em.L.append(f"          _mm_storeh_pd(&{arr}[(k+1)*ovs+os*12], lo);")
                    em.L.append(f"          _mm_storel_pd(&{arr}[(k+2)*ovs+os*12], hi);")
                    em.L.append(f"          _mm_storeh_pd(&{arr}[(k+3)*ovs+os*12], hi); }}")
            else:  # avx512
                for comp, arr in [('re', 'out_re'), ('im', 'out_im')]:
                    bname = f"tbuf_{comp}"
                    for j in range(VL):
                        em.L.append(f"        {arr}[(k+{j})*ovs+os*12] = {bname}[12*{VL}+{j}];")

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
        t2_pattern = 'radix13_n1_dit_kernel'
        sv_name = 'radix13_n1sv_kernel'
    elif variant == 'dit_tw':
        t2_pattern = 'radix13_tw_flat_dit_kernel'
        sv_name = 'radix13_t1sv_dit_kernel'
    elif variant == 'dit_tw_scalar':
        t2_pattern = 'radix13_tw_flat_dit_kernel'
        sv_name = 'radix13_t1sv_dit_kernel'
    elif variant == 'dif_tw':
        t2_pattern = 'radix13_tw_flat_dif_kernel'
        sv_name = 'radix13_t1sv_dif_kernel'
    elif variant == 'dif_tw_scalar':
        t2_pattern = 'radix13_tw_flat_dif_kernel'
        sv_name = 'radix13_t1sv_dif_kernel'
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
    parser = argparse.ArgumentParser(description='Unified R=13 codelet generator')
    parser.add_argument('--isa', default='avx2',
                        choices=['scalar', 'avx2', 'avx512', 'all'])
    parser.add_argument('--variant', default='notw',
                        choices=['notw', 'dit_tw', 'dif_tw', 'dit_tw_log3', 'dif_tw_log3',
                                 'ct_n1', 'ct_t1_dit', 'ct_t1s_dit', 'ct_t1_dit_log3', 'ct_t1_dif', 'all'])
    args = parser.parse_args()

    if args.isa == 'all':
        targets = [ISA_SCALAR, ISA_AVX2, ISA_AVX512]
    else:
        targets = [ALL_ISA[args.isa]]

    std_variants = ['notw', 'dit_tw', 'dif_tw', 'dit_tw_log3', 'dif_tw_log3']
    ct_variants  = ['ct_n1', 'ct_t1_dit', 'ct_t1s_dit', 'ct_t1_dit_log3', 'ct_t1_dif']

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
