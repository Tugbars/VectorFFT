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
gen_radix25.py — Unified DFT-25 codelet generator

5x5 CT: pass 1 = N2 radix-5 sub-FFTs, pass 2 = N1 radix-5 column combines.
Internal W25 twiddles between passes.
ALL 9 unique exponents require cmul (none trivial).
25 spill + 25 reload per kernel.

Usage:
  python3 gen_radix25.py --isa avx2 --variant all
  python3 gen_radix25.py --isa avx512 --variant dit_tw
  python3 gen_radix25.py --isa scalar --variant notw
  python3 gen_radix25.py --isa all --variant all
"""
import math, sys, argparse, re

N, N1, N2 = 25, 5, 5

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
    """Collect unique internal W25 twiddle exponents.
    For R=25: n2=1..4, k1=1..4, exponent = (n2*k1) % 25.
    Unique: {1,2,3,4,6,8,9,12,16}. ALL require cmul."""
    tw = set()
    for n2 in range(1, N2):
        for k1 in range(1, N1):
            e = (n2 * k1) % N
            tw.add((e, N))
    return tw

# ═══════════════════════════════════════════════════════════════
# ISA CONFIGURATION
# ═══════════════════════════════════════════════════════════════

class ISAConfig:
    def __init__(self, name, T, width, k_step, p, sm, target, align, ld_prefix):
        self.name = name
        self.T = T              # register type
        self.width = width
        self.k_step = k_step
        self.p = p              # intrinsic prefix
        self.sm = sm            # spill multiplier
        self.target = target    # __attribute__
        self.align = align      # alignment attribute
        self.ld_prefix = ld_prefix  # LD/ST macro prefix

ISA_SCALAR = ISAConfig('scalar', 'double', 1, 1, '', 1, '', '', 'R25S')
ISA_AVX2   = ISAConfig('avx2', '__m256d', 4, 4, '_mm256', 4,
    '__attribute__((target("avx2,fma")))',
    '__attribute__((aligned(32)))', 'R25A')
ISA_AVX512 = ISAConfig('avx512', '__m512d', 8, 8, '_mm512', 8,
    '__attribute__((target("avx512f,avx512dq,fma")))',
    '__attribute__((aligned(64)))', 'R25L')

ALL_ISA = {'scalar': ISA_SCALAR, 'avx2': ISA_AVX2, 'avx512': ISA_AVX512}

# ═══════════════════════════════════════════════════════════════
# EMITTER
# ═══════════════════════════════════════════════════════════════

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

    # ── Load / Store (counted) ──
    # addr_mode: 'K' (default), 'n1' (separate is/os), 'n1_ovs', 't1' (in-place ios/ms)
    addr_mode = 'K'
    tw_hoisted = False

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
        if self.addr_mode == 't1': return "rio_re"
        return "in_re"

    def _in_buf_im(self):
        if self.addr_mode == 't1': return "rio_im"
        return "in_im"

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

    # ── External twiddle ──
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

    # ── Internal W25 twiddle (ALL exponents require cmul) ──
    def emit_twiddle(self, dst, src, e, tN, d):
        """Emit W25 internal twiddle via hoisted broadcast constants."""
        label = wN_label(e, tN)
        T = self.isa.T
        # Use precomputed tw_{label}_re / tw_{label}_im broadcasts
        self.emit_cmul_inplace_named(dst, f"tw_{label}_re", f"tw_{label}_im", d)

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

    def emit_cmul_inplace_named(self, v, wr, wi, d):
        """Emit v *= (wr + j*wi) (fwd) or v *= conj(wr+j*wi) (bwd) using named vars."""
        fwd = (d == 'fwd')
        T = self.isa.T
        self.o(f"{{ {T} tr = {v}_re;")
        if fwd:
            self.o(f"  {v}_re = {self.fms(f'{v}_re', wr, self.mul(f'{v}_im', wi))};")
            self.o(f"  {v}_im = {self.fma('tr', wi, self.mul(f'{v}_im', wr))}; }}")
        else:
            self.o(f"  {v}_re = {self.fma(f'{v}_re', wr, self.mul(f'{v}_im', wi))};")
            self.o(f"  {v}_im = {self.fms(f'{v}_im', wr, self.mul('tr', wi))}; }}")

    # ── Complex multiply helpers (log3 derivation) ──
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

    # ── Radix-5 butterfly ──
    def emit_radix5(self, v, d, label=""):
        """Emit DFT-5 butterfly on v[0..4] in-place.
        v: list of 5 variable name prefixes (each has _re, _im).
        Follows gen_radix5.py register-pressure-optimized path for avx2/scalar,
        AVX-512 uses max-ILP path."""
        fwd = (d == 'fwd')
        T = self.isa.T

        if label:
            self.c(f"{label} [{d}]")

        x0, x1, x2, x3, x4 = v[0], v[1], v[2], v[3], v[4]

        # Open scoped block so butterfly temporaries don't clash across multiple calls
        self.o(f"{{")

        if self.isa.name == 'avx512':
            # AVX-512: max-ILP path
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
            self.o(f"{T} qar={self.fma('cC','d2i',self.mul('cB','d1i'))}, "
                   f"qai={self.fma('cC','d2r',self.mul('cB','d1r'))};")
            self.o(f"{T} qbr={self.fms('cB','d2i',self.mul('cC','d1i'))}, "
                   f"qbi={self.fms('cB','d2r',self.mul('cC','d1r'))};")
        else:
            # AVX2/scalar: register-pressure-optimized
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

        # Close scoped block
        self.o(f"}}")


# ═══════════════════════════════════════════════════════════════
# LOG3 TWIDDLE DERIVATION TABLE
#
# R=25 external twiddle table has 24 rows (W^1 .. W^24).
# Bases loaded from table: W^1 (index 0) and W^5 (index 4).
# Derivation:
#   From W^1: w2=w1*w1, w3=w1*w2, w4=w1*w3    (3 cmuls)
#   Row n2=0 (k1=1..4): w1, w2, w3, w4         (indices 0..3)
#   Row n2=1 (k1=1..4): w5, w10=w5*w5, w15=w5*w10, w20=w10*w10
#   Row n2=2 (k1=1..4): w5*w1, w5*w2, w5*w3, w5*w4
#     = derive: w6=w5*w1, w7=w5*w2, w8=w5*w3, w9=w5*w4
#   Row n2=3 (k1=1..4): w10*w1, w10*w2, w10*w3, w10*w4
#     = derive: w11=w10*w1, w12=w10*w2, w13=w10*w3, w14=w10*w4
#   Row n2=4 (k1=1..4): w15*w1, w15*w2, w15*w3, w15*w4
#     = derive: w16=w15*w1, w17=w15*w2, w18=w15*w3, w19=w15*w4
#
# Note: The external twiddle for CT is W^(n2*k1), where:
#   - n2 indexes the input sub-FFT group (row of the input matrix)
#   - k1 indexes the position within the group
# Flat table layout: tw[n2][k1] for n2=0..3 (rows 0-based for
# n2=1..4 in the decomposition), k1=1..4, total 16 entries.
# ═══════════════════════════════════════════════════════════════

def _log3_subfft_chain(n2):
    """Return [(k1, tw_action), ...] for sub-FFT n2 (0-indexed from 0..4).
    tw_action = None | ('base', name) | ('derive', wname, a, b)
    For n2=0 (first sub-FFT), no external twiddles needed.
    For n2=1..4, external twiddles are W^(n2*k1) for k1=1..4.
    Variable names: b1=W^1, b5=W^5, w2..w4, w6..w24"""
    if n2 == 0:
        return [
            (0, None),
            (1, None),
            (2, None),
            (3, None),
            (4, None),
        ]
    elif n2 == 1:
        # W^1, W^2, W^3, W^4
        return [
            (0, None),
            (1, ('base', 'b1')),               # W^1 base
            (2, ('derive', 'w2', 'b1', 'b1')), # W^2=W^1*W^1
            (3, ('derive', 'w3', 'b1', 'w2')), # W^3=W^1*W^2
            (4, ('derive', 'w4', 'b1', 'w3')), # W^4=W^1*W^3
        ]
    elif n2 == 2:
        # W^2, W^4, W^6, W^8
        return [
            (0, None),
            (1, ('derive', 'w2', 'b1', 'b1')), # W^2=W^1*W^1
            (2, ('derive', 'w4', 'w2', 'w2')), # W^4=W^2*W^2
            (3, ('derive', 'w6', 'b5', 'b1')), # W^6=W^5*W^1
            (4, ('derive', 'w8', 'b5', 'w3_')), # W^8=W^5*W^3 but w3 may not be live
        ]
    elif n2 == 3:
        # W^3, W^6, W^9, W^12
        return [
            (0, None),
            (1, ('derive', 'w3', 'b1', 'w2')), # W^3=W^1*W^2 (need w2 first)
            (2, ('derive', 'w6', 'b5', 'b1')), # W^6=W^5*W^1
            (3, ('derive', 'w9', 'b5', 'w4')), # W^9=W^5*W^4 (need w4)
            (4, ('derive', 'w12','b5', 'w7_')), # W^12=W^5*W^7
        ]
    else:  # n2 == 4
        # W^4, W^8, W^12, W^16
        return [
            (0, None),
            (1, ('derive', 'w4', 'w2', 'w2')),  # W^4=W^2*W^2 (need w2)
            (2, ('derive', 'w8', 'b5', 'w3')),   # W^8=W^5*W^3 (need w3)
            (3, ('derive', 'w12','b5', 'w7')),    # W^12=W^5*W^7
            (4, ('derive', 'w16','b5', 'w11')),   # W^16=W^5*W^11
        ]


# ═══════════════════════════════════════════════════════════════
# KERNEL BODY EMITTERS
# ═══════════════════════════════════════════════════════════════

def emit_kernel_body(em, d, itw_set, variant):
    """Emit the inner loop body for notw, dit_tw, dif_tw."""
    xv5 = [f"x{i}" for i in range(N1)]

    # PASS 1: N2 radix-5 sub-FFTs
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
        em.emit_radix5(xv5, d, f"radix-5 n2={n2}")
        em.b()
        for k1 in range(N1):
            em.emit_spill(f"x{k1}", n2 * N1 + k1)
        em.b()

    # PASS 2: N1 radix-5 column combines
    em.c(f"PASS 2")
    em.b()
    for k1 in range(N1):
        em.c(f"column k1={k1}")
        for n2 in range(N2):
            em.emit_reload(f"x{n2}", n2 * N1 + k1)
        em.b()
        if k1 > 0:
            for n2 in range(1, N2):
                e = (n2 * k1) % N
                em.emit_twiddle(f"x{n2}", f"x{n2}", e, N, d)
            em.b()
        em.emit_radix5(xv5, d, f"radix-5 k1={k1}")
        em.b()
        # DIF: external twiddle on outputs
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
    """Emit DIT or DIF log3 kernel body.
    R=25 log3: load W^1 and W^5 as bases, derive all 24 twiddles."""
    assert variant in ('dit_tw_log3', 'dif_tw_log3')
    is_dit = variant == 'dit_tw_log3'
    T = em.isa.T
    xv5 = [f"x{i}" for i in range(N1)]

    if em.isa.name == 'scalar':
        ld = None
    else:
        ld = f"{em.isa.p}_load_pd"

    # Load 2 base twiddles: W^1 (index 0) and W^5 (index 4)
    em.c("Load 2 log3 base twiddles: W^1 and W^5")
    tb, tbi = em._tw_buf(), em._tw_buf_im()
    ta0 = em._tw_addr(0)
    ta4 = em._tw_addr(4)
    if em.isa.name == 'scalar':
        em.o(f"const double b1_re = {tb}[{ta0}], b1_im = {tbi}[{ta0}];")
        em.o(f"const double b5_re = {tb}[{ta4}], b5_im = {tbi}[{ta4}];")
    else:
        em.o(f"const {T} b1_re = {ld}(&{tb}[{ta0}]);")
        em.o(f"const {T} b1_im = {ld}(&{tbi}[{ta0}]);")
        em.o(f"const {T} b5_re = {ld}(&{tb}[{ta4}]);")
        em.o(f"const {T} b5_im = {ld}(&{tbi}[{ta4}]);")
    em.n_load += 4
    em.b()

    # Derive W^2 = W^1*W^1, W^3 = W^1*W^2, W^4 = W^1*W^3
    em.c("Derive W^2, W^3, W^4 from W^1")
    em.o(f"{T} w2_re, w2_im;")
    em.emit_cmul("w2_re", "w2_im", "b1_re", "b1_im", "b1_re", "b1_im", 'fwd')
    em.o(f"{T} w3_re, w3_im;")
    em.emit_cmul("w3_re", "w3_im", "b1_re", "b1_im", "w2_re", "w2_im", 'fwd')
    em.o(f"{T} w4_re, w4_im;")
    em.emit_cmul("w4_re", "w4_im", "b1_re", "b1_im", "w3_re", "w3_im", 'fwd')
    em.b()

    # Derive W^10 = W^5*W^5, W^15 = W^5*W^10, W^20 = W^10*W^10
    em.c("Derive W^10, W^15, W^20 from W^5")
    em.o(f"{T} w10_re, w10_im;")
    em.emit_cmul("w10_re", "w10_im", "b5_re", "b5_im", "b5_re", "b5_im", 'fwd')
    em.o(f"{T} w15_re, w15_im;")
    em.emit_cmul("w15_re", "w15_im", "b5_re", "b5_im", "w10_re", "w10_im", 'fwd')
    em.o(f"{T} w20_re, w20_im;")
    em.emit_cmul("w20_re", "w20_im", "w10_re", "w10_im", "w10_re", "w10_im", 'fwd')
    em.b()

    # PASS 1: N2 radix-5 sub-FFTs with log3 twiddles applied
    for n2 in range(N2):
        em.c(f"sub-FFT n2={n2}")
        for n1 in range(N1):
            n = N2 * n1 + n2
            em.emit_load(f"x{n1}", n)

            if is_dit and n > 0:
                # Twiddle W^(n2 * n1 * ... wait: flat table uses W^(output_idx-1)
                # In DIT: twiddle x[n1] by W^(row_twiddle)
                # Row twiddle index in flat table: n-1 (for n>0)
                tw_exp = n  # = N2*n1 + n2, but flat table index is n-1
                # Map to log3 variable: twiddle W^(n2_input * k1_input)
                # For sub-FFT n2, elements are at positions n2, n2+N2, n2+2*N2, ...
                # Input n = N2*n1 + n2, so twiddle = W^n
                # But actual CT twiddle is W^(n2 * k1) where n2 is row and k1 is column
                # For the flat DIT layout: twiddle[n-1] is W^n
                _apply_log3_twiddle_dit(em, f"x{n1}", n, d, T)

        em.b()
        em.emit_radix5(xv5, d, f"radix-5 n2={n2}")
        em.b()
        for k1 in range(N1):
            em.emit_spill(f"x{k1}", n2 * N1 + k1)
        em.b()

    # PASS 2: N1 radix-5 column combines with internal W25 twiddles
    em.c("PASS 2")
    em.b()
    for k1 in range(N1):
        em.c(f"column k1={k1}")
        for n2 in range(N2):
            em.emit_reload(f"x{n2}", n2 * N1 + k1)
        em.b()
        if k1 > 0:
            for n2 in range(1, N2):
                e = (n2 * k1) % N
                em.emit_twiddle(f"x{n2}", f"x{n2}", e, N, d)
            em.b()
        em.emit_radix5(xv5, d, f"radix-5 k1={k1}")
        em.b()

        if not is_dit:
            # DIF: post-twiddle outputs
            for k2 in range(N2):
                m = k1 + N1 * k2
                if m > 0:
                    _apply_log3_twiddle_dif(em, f"x{k2}", m, d, T)
            em.b()

        for k2 in range(N2):
            em.emit_store(f"x{k2}", k1 + N1 * k2)
        em.b()


def _apply_log3_twiddle_dit(em, vname, exp, d, T):
    """Apply W^exp twiddle to vname using pre-derived log3 variables.
    exp is the absolute DFT index (1-based flat table position + 1).
    Available vars: b1, b5, w2..w4, w10, w15, w20 (as _re/_i pairs)."""
    # Map exp to available variable
    _twiddle_by_exp_log3(em, vname, exp, d, T)


def _apply_log3_twiddle_dif(em, vname, exp, d, T):
    """Apply W^exp twiddle to vname for DIF output. Same as DIT but different direction."""
    _twiddle_by_exp_log3(em, vname, exp, d, T)


def _twiddle_by_exp_log3(em, vname, exp, d, T):
    """Apply W^exp twiddle to vname using log3-derived variables.
    exp is in range 1..24 (mod 25).
    Pre-derived: b1=W^1, b5=W^5, w2=W^2, w3=W^3, w4=W^4,
                 w10=W^10, w15=W^15, w20=W^20."""
    exp = exp % N
    if exp == 0:
        return  # trivial W^0

    # Map to available base/derived variables
    # W^5n for n=1,2,3,4: b5, w10, w15, w20
    # W^(5n+r) = W^(5n) * W^r: combine
    q, r = divmod(exp, 5)
    if r == 0:
        # Pure multiple of 5
        bases = {1: 'b5', 2: 'w10', 3: 'w15', 4: 'w20'}
        wname = bases[q]
        em.emit_cmul_inplace(vname, f"{wname}_re", f"{wname}_im", d)
    elif q == 0:
        # Small exponent 1..4
        small = {1: 'b1', 2: 'w2', 3: 'w3', 4: 'w4'}
        wname = small[r]
        em.emit_cmul_inplace(vname, f"{wname}_re", f"{wname}_im", d)
    else:
        # Need to combine: W^(5q) * W^r
        # Derive combined on the fly
        q_bases = {1: 'b5', 2: 'w10', 3: 'w15', 4: 'w20'}
        r_bases = {1: 'b1', 2: 'w2', 3: 'w3', 4: 'w4'}
        qname = q_bases[q]
        rname = r_bases[r]
        em.o(f"{{ {T} cwr, cwi;")
        em.emit_cmul("cwr", "cwi", f"{qname}_re", f"{qname}_im",
                     f"{rname}_re", f"{rname}_im", 'fwd')
        em.emit_cmul_inplace(vname, "cwr", "cwi", d)
        em.o(f"}}")


# ═══════════════════════════════════════════════════════════════
# FILE EMITTER HELPERS
# ═══════════════════════════════════════════════════════════════

def emit_twiddle_constants(L, itw_set):
    by_tN = {}
    for (e, tN) in sorted(itw_set):
        by_tN.setdefault(tN, []).append(e)
    for tN in sorted(by_tN):
        g = f"FFT_W{tN}_TWIDDLES_DEFINED"
        L.append(f"#ifndef {g}")
        L.append(f"#define {g}")
        for e in sorted(by_tN[tN]):
            wr, wi = wN(e, tN)
            l = wN_label(e, tN)
            L.append(f"static const double {l}_re = {wr:.20e};")
            L.append(f"static const double {l}_im = {wi:.20e};")
        L.append(f"#endif")
        L.append("")


def emit_dft5_constants(em):
    """Emit the 4 DFT-5 constants + sign_flip as SIMD broadcasts or scalars."""
    T = em.isa.T
    if em.isa.name == 'scalar':
        em.o(f"const double cA = {cA_val:.45f};")
        em.o(f"const double cB = {cB_val:.45f};")
        em.o(f"const double cC = {cC_val:.45f};")
        em.o(f"const double cD = {cD_val:.45f};")
    else:
        set1 = f"{em.isa.p}_set1_pd"
        em.o(f"const {T} sign_flip = {set1}(-0.0);")
        em.o(f"const {T} cA = {set1}({cA_val:.45f});")
        em.o(f"const {T} cB = {set1}({cB_val:.45f});")
        em.o(f"const {T} cC = {set1}({cC_val:.45f});")
        em.o(f"const {T} cD = {set1}({cD_val:.45f});")
        em.o(f"(void)sign_flip;  /* reserved for neg() */")
    em.b()


def emit_dft5_constants_raw(lines, isa, indent=1):
    """Append DFT-5 constant declarations directly to a line list."""
    pad = "    " * indent
    T = isa.T
    if isa.name == 'scalar':
        lines.append(f"{pad}const double cA = {cA_val:.45f};")
        lines.append(f"{pad}const double cB = {cB_val:.45f};")
        lines.append(f"{pad}const double cC = {cC_val:.45f};")
        lines.append(f"{pad}const double cD = {cD_val:.45f};")
    else:
        set1 = f"{isa.p}_set1_pd"
        lines.append(f"{pad}const {T} sign_flip = {set1}(-0.0);")
        lines.append(f"{pad}const {T} cA = {set1}({cA_val:.45f});")
        lines.append(f"{pad}const {T} cB = {set1}({cB_val:.45f});")
        lines.append(f"{pad}const {T} cC = {set1}({cC_val:.45f});")
        lines.append(f"{pad}const {T} cD = {set1}({cD_val:.45f});")
        lines.append(f"{pad}(void)sign_flip;")
    lines.append(f"")


def insert_stats_into_header(lines, stats):
    table = [" *", " * -- Operation counts per k-step --", " *"]
    s5 = '-' * 5; s3 = '-' * 3; s4 = '-' * 4; sep = '-' * 20
    table.append(f" *   {'kernel':<20s} {'add':>5s} {'sub':>5s} {'mul':>5s} {'neg':>5s}"
                 f" {'fma':>5s} {'fms':>5s} | {'arith':>5s} {'flops':>5s}"
                 f" | {'ld':>3s} {'st':>3s} {'sp':>3s} {'rl':>3s} {'mem':>4s}")
    table.append(f" *   {sep} {s5} {s5} {s5} {s5}"
                 f" {s5} {s5} + {s5} {s5}"
                 f" + {s3} {s3} {s3} {s3} {s4}")
    for k in sorted(stats.keys()):
        s = stats[k]
        table.append(f" *   {k:<20s} {s['add']:5d} {s['sub']:5d} {s['mul']:5d} {s['neg']:5d}"
                     f" {s['fma']:5d} {s['fms']:5d} | {s['total_arith']:5d} {s['flops']:5d}"
                     f" | {s['load']:3d} {s['store']:3d} {s['spill']:3d} {s['reload']:3d} {s['total_mem']:4d}")
    table.append(" *")
    for i, line in enumerate(lines):
        if line.strip() == '*/':
            for j, tl in enumerate(table):
                lines.insert(i + j, tl)
            return


# ═══════════════════════════════════════════════════════════════
# FILE EMITTER — standard variants (notw, dit_tw, dif_tw, log3)
# ═══════════════════════════════════════════════════════════════

def emit_file(isa, itw_set, variant):
    """Emit complete header for one ISA and one variant."""
    em = Emitter(isa)
    T = isa.T
    is_log3 = variant.endswith('_log3')

    if variant == 'notw':
        func_base = 'radix25_n1_dit_kernel'
        tw_params = None
    elif variant == 'dit_tw':
        func_base = 'radix25_tw_flat_dit_kernel'
        tw_params = 'flat'
    elif variant == 'dit_tw_scalar':
        func_base = 'radix25_tw_scalar_dit_kernel'
        tw_params = 'flat'
    elif variant == 'dif_tw':
        func_base = 'radix25_tw_flat_dif_kernel'
        tw_params = 'flat'
    elif variant == 'dif_tw_scalar':
        func_base = 'radix25_tw_scalar_dif_kernel'
        tw_params = 'flat'
    elif variant == 'dit_tw_log3':
        func_base = 'radix25_tw_log3_dit_kernel'
        tw_params = 'flat'
    elif variant == 'dif_tw_log3':
        func_base = 'radix25_tw_log3_dif_kernel'
        tw_params = 'flat'
    else:
        raise ValueError(f"Unknown variant: {variant}")

    vname = {
        'notw': 'N1 (no twiddle)', 'dit_tw': 'DIT twiddled (flat)',
        'dif_tw': 'DIF twiddled (flat)',
        'dit_tw_log3': 'DIT twiddled (log3 derived)',
        'dif_tw_log3': 'DIF twiddled (log3 derived)',
    }[variant]
    guard = f"FFT_RADIX25_{isa.name.upper()}_{variant.upper()}_H"

    em.L.append(f"/**")
    em.L.append(f" * @file fft_radix25_{isa.name}_{variant}.h")
    em.L.append(f" * @brief DFT-25 {isa.name.upper()} {vname}")
    em.L.append(f" *")
    em.L.append(f" * 5x5 CT, k-step={isa.k_step}")
    em.L.append(f" * Generated by gen_radix25.py")
    em.L.append(f" */")
    em.L.append(f"")
    em.L.append(f"#ifndef {guard}")
    em.L.append(f"#define {guard}")
    em.L.append(f"")

    if isa.name != 'scalar':
        em.L.append(f"#include <immintrin.h>")
        em.L.append(f"")

    # Twiddle constants (W25 internal twiddles)
    emit_twiddle_constants(em.L, itw_set)

    # LD/ST macros
    lp = isa.ld_prefix
    if isa.name == 'scalar':
        em.L.append(f"#ifndef {lp}_LD")
        em.L.append(f"#define {lp}_LD(p) (*(p))")
        em.L.append(f"#endif")
        em.L.append(f"#ifndef {lp}_ST")
        em.L.append(f"#define {lp}_ST(p,v) (*(p)=(v))")
        em.L.append(f"#endif")
    elif isa.name == 'avx2':
        em.L.append(f"#ifndef {lp}_LD")
        em.L.append(f"#define {lp}_LD(p) _mm256_load_pd(p)")
        em.L.append(f"#endif")
        em.L.append(f"#ifndef {lp}_ST")
        em.L.append(f"#define {lp}_ST(p,v) _mm256_store_pd((p),(v))")
        em.L.append(f"#endif")
    else:
        em.L.append(f"#ifndef {lp}_LD")
        em.L.append(f"#define {lp}_LD(p) _mm512_load_pd(p)")
        em.L.append(f"#endif")
        em.L.append(f"#ifndef {lp}_ST")
        em.L.append(f"#define {lp}_ST(p,v) _mm512_store_pd((p),(v))")
        em.L.append(f"#endif")
    em.L.append(f"#define LD {lp}_LD")
    em.L.append(f"#define ST {lp}_ST")
    em.L.append(f"")

    stats = {}
    for d in ['fwd', 'bwd']:
        em.reset()

        # Function signature
        if isa.target:
            em.L.append(f"static {isa.target} void")
        else:
            em.L.append(f"static void")
        em.L.append(f"{func_base}_{d}_{isa.name}(")
        em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
        em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
        if tw_params == 'flat':
            em.L.append(f"    const double * __restrict__ tw_re, const double * __restrict__ tw_im,")
        em.L.append(f"    size_t K)")
        em.L.append(f"{{")
        em.ind = 1

        # Constants
        emit_dft5_constants(em)

        # Spill buffer: 25 slots
        spill_total = N * isa.sm
        if isa.name == 'scalar':
            em.o(f"double spill_re[{N}], spill_im[{N}];")
        else:
            em.o(f"{isa.align} double spill_re[{spill_total}];")
            em.o(f"{isa.align} double spill_im[{spill_total}];")
        em.b()

        # Working registers: x0..x4
        em.o(f"{T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im,x4_re,x4_im;")
        em.b()

        # Hoisted internal W25 twiddle broadcasts (all require cmul)
        if itw_set:
            if isa.name != 'scalar':
                em.c(f"Hoisted internal W25 broadcasts [{d}]")
                set1 = f"{isa.p}_set1_pd"
                for (e, tN) in sorted(itw_set):
                    label = wN_label(e, tN)
                    em.o(f"const {T} tw_{label}_re = {set1}({label}_re);")
                    em.o(f"const {T} tw_{label}_im = {set1}({label}_im);")
            else:
                em.c(f"Hoisted W25 [{d}]")
                for (e, tN) in sorted(itw_set):
                    label = wN_label(e, tN)
                    em.o(f"const double tw_{label}_re = {label}_re;")
                    em.o(f"const double tw_{label}_im = {label}_im;")
            em.b()

        # K loop
        if isa.name == 'scalar':
            em.o(f"for (size_t k = 0; k < K; k++) {{")
        else:
            em.o(f"for (size_t k = 0; k < K; k += {isa.k_step}) {{")
        em.ind += 1
        if is_log3:
            emit_kernel_body_log3(em, d, itw_set, variant)
        else:
            emit_kernel_body(em, d, itw_set, variant)
        em.ind -= 1
        em.o("}")
        em.L.append("}")
        em.L.append("")

        stats[d] = em.get_stats()

    # U=2 pipelining (AVX-512 only, non-log3)
    if isa.name == 'avx512' and not is_log3 and func_base is not None:
        VL = isa.k_step
        u2_name = func_base + '_u2'
        has_tw = tw_params is not None

        for d in ['fwd', 'bwd']:
            em_u1 = Emitter(isa)
            em_u1.addr_mode = 'K'
            em_u1.ind = 2
            emit_kernel_body(em_u1, d, itw_set, variant)
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

            emit_dft5_constants_raw(em.L, isa, indent=1)

            spill_total = N * isa.sm
            em.L.append(f"    {isa.align} double spill_re[{spill_total}];")
            em.L.append(f"    {isa.align} double spill_im[{spill_total}];")

            if itw_set:
                set1 = f"{isa.p}_set1_pd"
                for (e, tN) in sorted(itw_set):
                    label = wN_label(e, tN)
                    em.L.append(f"    const {T} tw_{label}_re = {set1}({label}_re);")
                    em.L.append(f"    const {T} tw_{label}_im = {set1}({label}_im);")
                em.L.append(f"")

            xdecl = f"        {T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im,x4_re,x4_im;"
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


# ═══════════════════════════════════════════════════════════════
# SV CODELET GENERATION — text transform from t2 output
# ═══════════════════════════════════════════════════════════════

def _t2_to_sv(body):
    """Transform a t2 codelet body to sv: strip k-loop, K→vs in addressing."""
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
        t2_pattern = 'radix25_n1_dit_kernel'
        sv_name = 'radix25_n1sv_kernel'
    elif variant == 'dit_tw':
        t2_pattern = 'radix25_tw_flat_dit_kernel'
        sv_name = 'radix25_t1sv_dit_kernel'
    elif variant == 'dit_tw_scalar':
        t2_pattern = 'radix25_tw_scalar_dit_kernel'
        sv_name = 'radix25_t1ssv_dit_kernel'
    elif variant == 'dif_tw':
        t2_pattern = 'radix25_tw_flat_dif_kernel'
        sv_name = 'radix25_t1sv_dif_kernel'
    elif variant == 'dif_tw_scalar':
        t2_pattern = 'radix25_tw_scalar_dif_kernel'
        sv_name = 'radix25_t1ssv_dif_kernel'
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


# ═══════════════════════════════════════════════════════════════
# FILE EMITTER — CT variants (ct_n1, ct_t1_dit, ct_t1_dif)
# ═══════════════════════════════════════════════════════════════

def emit_file_ct(isa, itw_set, ct_variant):
    """Emit FFTW-style n1 or t1_dit codelet."""
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
        func_base = "radix25_n1"
        vname = "n1 (separate is/os)"
    elif is_n1_scaled:
        func_base = "radix25_n1_scaled"
        vname = "n1_scaled (separate is/os, output *= scale)"
    elif is_t1_oop_dit:
        func_base = "radix25_t1_oop_dit"
        vname = "t1_oop DIT (out-of-place, separate is/os, with twiddle)"
    elif is_t1_dif:
        func_base = "radix25_t1_dif"
        vname = "t1 DIF (in-place twiddle)"
    elif is_t1s_dit:
        func_base = "radix25_t1s_dit"
        vname = "t1s DIT (in-place, scalar broadcast twiddle)"
    elif is_t1_dit_log3:
        func_base = "radix25_t1_dit_log3"
        vname = "t1 DIT log3 (in-place, derived twiddles)"
    else:
        func_base = "radix25_t1_dit"
        vname = "t1 DIT (in-place twiddle)"
    guard = f"FFT_RADIX25_{isa.name.upper()}_CT_{ct_variant.upper()}_H"

    em.L.append(f"/**")
    em.L.append(f" * @file fft_radix25_{isa.name}_{ct_variant}.h")
    em.L.append(f" * @brief DFT-25 {isa.name.upper()} {vname}")
    em.L.append(f" *")
    em.L.append(f" * FFTW-style codelet for recursive CT executor.")
    em.L.append(f" * Generated by gen_radix25.py --variant {ct_variant}")
    em.L.append(f" */")
    em.L.append(f"")
    em.L.append(f"#ifndef {guard}")
    em.L.append(f"#define {guard}")
    em.L.append(f"")

    if isa.name != 'scalar':
        em.L.append(f"#include <immintrin.h>")
        em.L.append(f"")

    emit_twiddle_constants(em.L, itw_set)

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

        # Constants
        emit_dft5_constants(em)

        # Spill buffer: 25 slots
        spill_total = N * isa.sm
        if isa.name == 'scalar':
            em.o(f"double spill_re[{N}], spill_im[{N}];")
        else:
            em.o(f"{isa.align} double spill_re[{spill_total}];")
            em.o(f"{isa.align} double spill_im[{spill_total}];")
        em.b()

        em.o(f"{T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im,x4_re,x4_im;")
        em.b()

        # Hoisted internal W25 twiddle broadcasts
        if itw_set:
            if isa.name != 'scalar':
                set1 = f"{isa.p}_set1_pd"
                for (e, tN) in sorted(itw_set):
                    label = wN_label(e, tN)
                    em.o(f"const {T} tw_{label}_re = {set1}({label}_re);")
                    em.o(f"const {T} tw_{label}_im = {set1}({label}_im);")
            else:
                for (e, tN) in sorted(itw_set):
                    label = wN_label(e, tN)
                    em.o(f"const double tw_{label}_re = {label}_re;")
                    em.o(f"const double tw_{label}_im = {label}_im;")
            em.b()

        # Hoist twiddle broadcasts before the loop (t1s only)
        if is_t1s_dit:
            em.tw_hoisted = True
            em.emit_hoist_all_tw_scalars(N)
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
            if isa.name == 'scalar' and not is_t1_oop_dit:
                em.o(f"for (size_t m = mb; m < me; m++) {{")
            else:
                em.o(f"for (size_t m = 0; m < me; m += {isa.k_step}) {{")

        em.ind += 1
        if is_t1s_dit:
            emit_kernel_body(em, d, itw_set, 'dit_tw_scalar')
        elif is_t1_dit_log3:
            emit_kernel_body_log3(em, d, itw_set, 'dit_tw_log3')
        else:
            kernel_variant = 'notw' if (is_n1 or is_n1_scaled) else ('dif_tw' if is_t1_dif else 'dit_tw')
            if is_t1_oop_dit:
                em.addr_mode = 't1_oop'
                kernel_variant = 'dit_tw'
            emit_kernel_body(em, d, itw_set, kernel_variant)
        em.ind -= 1
        em.o("}")
        em.L.append("}")
        em.L.append("")

    # n1_ovs: butterfly with fused SIMD transpose stores
    # 25 bins = 6 groups of 4 (4x4 transpose) + 1 leftover bin 24
    if is_n1 and isa.name != 'scalar':
        VL = isa.k_step
        n_full_groups = N // 4   # 6 groups of 4
        leftover_bin = N - 1     # bin 24 (index from 0)
        # Actually: bins 0..23 = 6 groups of 4, bin 24 = leftover
        n_groups = (N - 1) // 4  # 6
        leftover = (N - 1) % 4 != 0  # True: bin 24 is leftover

        for d in ['fwd', 'bwd']:
            em2 = Emitter(isa)
            em2.addr_mode = 'n1_ovs'

            em.L.append(f"")
            if isa.target:
                em.L.append(f"static {isa.target} void")
            else:
                em.L.append(f"static void")
            em.L.append(f"radix25_n1_ovs_{d}_{isa.name}(")
            em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
            em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
            em.L.append(f"    size_t is, size_t os, size_t vl, size_t ovs)")
            em.L.append(f"{{")
            em.L.append(f"    /* n1_ovs: butterfly -> tbuf, then 4x4 transpose (bins 0-23) + scatter (bin 24) */")
            em.L.append(f"    {isa.align} double tbuf_re[{N * VL}];")
            em.L.append(f"    {isa.align} double tbuf_im[{N * VL}];")

            emit_dft5_constants_raw(em.L, isa, indent=1)

            spill_total = N * isa.sm
            if isa.name == 'scalar':
                em.L.append(f"    double spill_re[{N}], spill_im[{N}];")
            else:
                em.L.append(f"    {isa.align} double spill_re[{spill_total}];")
                em.L.append(f"    {isa.align} double spill_im[{spill_total}];")

            em.L.append(f"    {T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im,x4_re,x4_im;")
            em.L.append(f"")

            if itw_set:
                if isa.name != 'scalar':
                    set1 = f"{isa.p}_set1_pd"
                    for (e, tN) in sorted(itw_set):
                        label = wN_label(e, tN)
                        em.L.append(f"    const {T} tw_{label}_re = {set1}({label}_re);")
                        em.L.append(f"    const {T} tw_{label}_im = {set1}({label}_im);")
                else:
                    for (e, tN) in sorted(itw_set):
                        label = wN_label(e, tN)
                        em.L.append(f"    const double tw_{label}_re = {label}_re;")
                        em.L.append(f"    const double tw_{label}_im = {label}_im;")
                em.L.append(f"")

            em.L.append(f"    for (size_t k = 0; k < vl; k += {VL}) {{")

            em2.L = []
            em2.ind = 2
            em2.reset()
            em2.addr_mode = 'n1_ovs'
            emit_kernel_body(em2, d, itw_set, 'notw')
            em.L.extend(em2.L)

            # 4x4 transpose for groups of 4 bins (bins 0..23 = 6 groups)
            em.L.append(f"        /* 4x4 transpose: bins 0-23 -> output at stride ovs */")
            for g in range(6):  # 6 complete groups of 4
                b = g * 4
                for comp, arr in [('re', 'out_re'), ('im', 'out_im')]:
                    bname = f"tbuf_{comp}"
                    if isa.name == 'avx2':
                        em.L.append(f"        {{ __m256d a_=_mm256_load_pd(&{bname}[{b+0}*{VL}]), b_=_mm256_load_pd(&{bname}[{b+1}*{VL}]);")
                        em.L.append(f"          __m256d c_=_mm256_load_pd(&{bname}[{b+2}*{VL}]), d_=_mm256_load_pd(&{bname}[{b+3}*{VL}]);")
                        em.L.append(f"          __m256d lo_ab=_mm256_unpacklo_pd(a_,b_), hi_ab=_mm256_unpackhi_pd(a_,b_);")
                        em.L.append(f"          __m256d lo_cd=_mm256_unpacklo_pd(c_,d_), hi_cd=_mm256_unpackhi_pd(c_,d_);")
                        em.L.append(f"          _mm256_storeu_pd(&{arr}[(k+0)*ovs+os*{b}], _mm256_permute2f128_pd(lo_ab,lo_cd,0x20));")
                        em.L.append(f"          _mm256_storeu_pd(&{arr}[(k+1)*ovs+os*{b}], _mm256_permute2f128_pd(hi_ab,hi_cd,0x20));")
                        em.L.append(f"          _mm256_storeu_pd(&{arr}[(k+2)*ovs+os*{b}], _mm256_permute2f128_pd(lo_ab,lo_cd,0x31));")
                        em.L.append(f"          _mm256_storeu_pd(&{arr}[(k+3)*ovs+os*{b}], _mm256_permute2f128_pd(hi_ab,hi_cd,0x31));")
                        em.L.append(f"        }}")
                    else:  # avx512: scalar scatter fallback
                        for j in range(VL):
                            for bi in range(4):
                                em.L.append(f"          {arr}[(k+{j})*ovs+os*{b+bi}] = {bname}[{b+bi}*{VL}+{j}];")

            # Bin 24: leftover — extract+scatter on AVX2, scalar scatter on AVX-512
            if isa.name == 'avx2':
                em.L.append(f"        /* Bin 24: extract from x4 YMM (still live after last sub-FFT spill) -> scatter */")
                # x4 is x4 from the last column butterfly (k1=4 => the last pass 2 iteration)
                # But at this point x4 has already been stored to tbuf[24*VL].
                # Use tbuf-based scatter for safety (x4 may be overwritten).
                em.L.append(f"        /* Bin 24: scatter from tbuf */")
                for comp, arr in [('re', 'out_re'), ('im', 'out_im')]:
                    bname = f"tbuf_{comp}"
                    em.L.append(f"        {{ __m256d v24=_mm256_load_pd(&{bname}[24*{VL}]);")
                    em.L.append(f"          __m128d lo=_mm256_castpd256_pd128(v24), hi=_mm256_extractf128_pd(v24,1);")
                    em.L.append(f"          _mm_storel_pd(&{arr}[(k+0)*ovs+os*24], lo);")
                    em.L.append(f"          _mm_storeh_pd(&{arr}[(k+1)*ovs+os*24], lo);")
                    em.L.append(f"          _mm_storel_pd(&{arr}[(k+2)*ovs+os*24], hi);")
                    em.L.append(f"          _mm_storeh_pd(&{arr}[(k+3)*ovs+os*24], hi); }}")
            else:
                em.L.append(f"        /* Bin 24: scalar scatter */")
                for comp, arr in [('re', 'out_re'), ('im', 'out_im')]:
                    bname = f"tbuf_{comp}"
                    for j in range(VL):
                        em.L.append(f"        {arr}[(k+{j})*ovs+os*24] = {bname}[24*{VL}+{j}];")

            em.L.append(f"    }}")
            em.L.append(f"}}")

    em.L.append(f"")
    em.L.append(f"#undef LD")
    em.L.append(f"#undef ST")
    em.L.append(f"")
    em.L.append(f"#endif /* {guard} */")

    return em.L


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def print_file(lines, label):
    """Print lines to stdout with a stderr label."""
    print("\n".join(lines))
    print(f"\n=== {label} ===", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description='Unified R=25 codelet generator')
    parser.add_argument('--isa', default='avx2',
                        choices=['scalar', 'avx2', 'avx512', 'all'])
    parser.add_argument('--variant', default='notw',
                        choices=['notw', 'dit_tw', 'dif_tw', 'dit_tw_log3', 'dif_tw_log3',
                                 'ct_n1', 'ct_n1_scaled', 'ct_t1_dit', 'ct_t1s_dit', 'ct_t1_dit_log3', 'ct_t1_dif', 'ct_t1_oop_dit', 'all'])
    args = parser.parse_args()

    itw_set = collect_internal_twiddles()

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
                lines = emit_file_ct(isa, itw_set, variant)
                print_file(lines, f"{isa.name.upper()} {variant}")
            else:
                lines, stats = emit_file(isa, itw_set, variant)
                sv_lines = emit_sv_variants(lines, isa, variant)
                if sv_lines:
                    for i in range(len(lines)):
                        if lines[i].strip() == '#undef LD':
                            lines[i:i] = sv_lines
                            break
                print_file(lines, f"{isa.name.upper()} {variant}")
                for k in sorted(stats.keys()):
                    s = stats[k]
                    print(f"  {k}: {s['spill']}sp+{s['reload']}rl={s['spill']+s['reload']}"
                          f" arith={s['total_arith']} flops={s['flops']}"
                          f" mem={s['total_mem']}", file=sys.stderr)


# ═══════════════════════════════════════════════════════════════
# Bench-compatible variant metadata and CLI wrapper
#
# Tuning scope for R=25 (first composite):
#   - Twiddle strategies: flat / t1s / log3 (3 variants)
#   - Unroll factor: U=1 only
#     R=25 uses a 25-slot spill buffer for the internal 5×5 grid between
#     passes; the working set is already large. U=2 would double it.
#   - Skip: DIF (dominated), oop_dit, ct_n1 family
#
#   Note on t1s at R=25: external twiddle count is (R-1)=24 scalars, so
#   t1s can only partially hoist (5/24 on AVX2, 12/24 on AVX-512) due to
#   register budget. Remaining twiddles fall back to inline broadcast
#   loads. This is the same partial-hoist regime as R=32/R=64, and we
#   expect similar mixed results — t1s may win at very small me where
#   broadcast amortization across iterations dominates, or lose where
#   the inline-load tail matters.
#
#   VTune on R=25 AVX2 t1_dit at K=256 shows the codelet is near-ceiling
#   (50% retiring, CPI 0.328, FMA-dep-chain bound with emerging DTLB
#   pressure). Variant discrimination margins expected to be small.
# ═══════════════════════════════════════════════════════════════

VARIANTS = {
    'ct_t1_dit':      ('radix25_t1_dit',      'flat', 't1_dit'),
    'ct_t1s_dit':     ('radix25_t1s_dit',     't1s',  't1s_dit'),
    'ct_t1_dit_log3': ('radix25_t1_dit_log3', 'log3', 't1_dit_log3'),
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
    """Emit a single header containing all variants for the given ISA."""
    isa = ALL_ISA[isa_name]
    guard = f'FFT_RADIX25_{isa_name.upper()}_H'
    out = []
    out.append(f'/**')
    out.append(f' * @file fft_radix25_{isa_name}.h')
    out.append(f' * @brief DFT-25 {isa_name.upper()} codelets (flat, t1s, log3 × U=1)')
    out.append(f' *')
    out.append(f' * Composite radix: 5×5 Cooley-Tukey with fused internal twiddles.')
    out.append(f' * Auto-generated by gen_radix25.py (bench wrapper).')
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
            if s.startswith('#ifndef FFT_RADIX25_'): continue
            if s.startswith('#define FFT_RADIX25_'): continue
            if s.startswith('#endif /* FFT_RADIX25_'): continue
            if s.startswith('#include <immintrin.h>'): continue
            if s.startswith('#include <stddef.h>'): continue
            out.append(L)
        out.append('')

    itw_set = collect_internal_twiddles()
    for v in ['ct_t1_dit', 'ct_t1s_dit', 'ct_t1_dit_log3']:
        lines = emit_file_ct(isa, itw_set, v)
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
