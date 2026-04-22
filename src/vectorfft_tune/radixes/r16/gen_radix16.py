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
gen_radix16.py — Unified DFT-16 codelet generator

4×4 CT: pass 1 = N2 radix-4 sub-FFTs, pass 2 = N1 radix-4 column combines.
Internal W16 twiddles between passes. Only 3 non-trivial constants.
16 spill + 16 reload per kernel. No NFUSE (radix-4 uses all 4 working pairs).

Usage:
  python3 gen_radix16.py --isa avx512 --variant dit_tw
  python3 gen_radix16.py --isa avx2 --variant dif_tw
  python3 gen_radix16.py --isa scalar --variant notw
  python3 gen_radix16.py --isa all --variant all
"""
import math, sys, argparse, re

N, N1, N2 = 16, 4, 4

# ═══════════════════════════════════════════════════════════════
# TWIDDLE ANALYSIS
# ═══════════════════════════════════════════════════════════════

def wN(e, tN):
    e = e % tN; a = 2.0 * math.pi * e / tN
    return (math.cos(a), -math.sin(a))

def wN_label(e, tN):
    return f"W{tN}_{e % tN}"

def twiddle_is_trivial(e, tN):
    e = e % tN
    if e == 0: return True, 'one'
    if (8 * e) % tN == 0:
        o = (8 * e) // tN
        t = ['one','w8_1','neg_j','w8_3','neg_one','neg_w8_1','pos_j','neg_w8_3']
        return True, t[o % 8]
    return False, 'cmul'

def collect_internal_twiddles():
    tw = set()
    for n2 in range(1, N2):
        for k1 in range(1, N1):
            e = (n2 * k1) % N
            _, typ = twiddle_is_trivial(e, N)
            if typ == 'cmul':
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

ISA_SCALAR = ISAConfig('scalar', 'double', 1, 1, '', 1, '', '', 'R16S')
ISA_AVX2   = ISAConfig('avx2', '__m256d', 4, 4, '_mm256', 4,
    '__attribute__((target("avx2,fma")))',
    '__attribute__((aligned(32)))', 'R16A')
ISA_AVX512 = ISAConfig('avx512', '__m512d', 8, 8, '_mm512', 8,
    '__attribute__((target("avx512f,avx512dq,fma")))',
    '__attribute__((aligned(64)))', 'R16L')

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

    # ── Load / Store (counted) ──
    # addr_mode: 'K' (default), 'n1' (separate is/os), 't1' (in-place ios/ms)
    addr_mode = 'K'
    tw_hoisted = False

    def _in_addr(self, n, ke="k"):
        if self.addr_mode in ('n1', 'n1_ovs'):
            return f"{n}*is+{ke}"
        elif self.addr_mode == 't1':
            if self.isa.name == 'scalar': return f"m*ms+{n}*ios"
            return f"m+{n}*ios"
        elif self.addr_mode == 't1_buf':
            # t1_buf reads same as t1: rio[m + n*ios]
            return f"m+{n}*ios"
        elif self.addr_mode == 't1_oop':
            return f"m+{n}*is"
        return f"{n}*K+{ke}"

    def _out_addr(self, m, ke="k"):
        if self.addr_mode == 'n1':
            return f"{m}*os+{ke}"
        elif self.addr_mode == 'n1_ovs':
            # Write to local transposition buffer at stride VL
            return f"{m}*{self.isa.k_step}"
        elif self.addr_mode == 't1':
            if self.isa.name == 'scalar': return f"m*ms+{m}*ios"
            return f"m+{m}*ios"
        elif self.addr_mode == 't1_oop':
            return f"m+{m}*os"
        # 't1_buf' addr_mode: Pass-2 writes to tile output buffer, sequential
        if self.addr_mode == 't1_buf':
            # Writes to outbuf[m*TILE + tile_k], where tile_k is the within-tile k offset
            return f"{m}*TILE+kk"
        return f"{m}*K+{ke}"

    def _in_buf(self):
        if self.addr_mode in ('t1', 't1_buf'): return "rio_re"
        return "in_re"

    def _in_buf_im(self):
        if self.addr_mode in ('t1', 't1_buf'): return "rio_im"
        return "in_im"

    def _out_buf(self):
        if self.addr_mode == 't1': return "rio_re"
        if self.addr_mode == 'n1_ovs': return "tbuf_re"
        if self.addr_mode == 't1_buf': return "outbuf_re"
        return "out_re"

    def _out_buf_im(self):
        if self.addr_mode == 't1': return "rio_im"
        if self.addr_mode == 'n1_ovs': return "tbuf_im"
        if self.addr_mode == 't1_buf': return "outbuf_im"
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

    # ── Radix-4 butterfly ──
    def emit_radix4(self, v, d, label=""):
        fwd = (d == 'fwd')
        T = self.isa.T
        if label:
            self.c(f"{label} [{d}]")
        a, b, c, dd = v[0], v[1], v[2], v[3]
        self.o(f"{{ {T} t0r,t0i,t1r,t1i,t2r,t2i,t3r,t3i;")
        self.o(f"  t0r={self.add(f'{a}_re',f'{c}_re')}; t0i={self.add(f'{a}_im',f'{c}_im')};")
        self.o(f"  t1r={self.sub(f'{a}_re',f'{c}_re')}; t1i={self.sub(f'{a}_im',f'{c}_im')};")
        self.o(f"  t2r={self.add(f'{b}_re',f'{dd}_re')}; t2i={self.add(f'{b}_im',f'{dd}_im')};")
        self.o(f"  t3r={self.sub(f'{b}_re',f'{dd}_re')}; t3i={self.sub(f'{b}_im',f'{dd}_im')};")
        self.o(f"  {a}_re={self.add('t0r','t2r')}; {a}_im={self.add('t0i','t2i')};")
        self.o(f"  {c}_re={self.sub('t0r','t2r')}; {c}_im={self.sub('t0i','t2i')};")
        if fwd:
            self.o(f"  {b}_re={self.add('t1r','t3i')}; {b}_im={self.sub('t1i','t3r')};")
            self.o(f"  {dd}_re={self.sub('t1r','t3i')}; {dd}_im={self.add('t1i','t3r')};")
        else:
            self.o(f"  {b}_re={self.sub('t1r','t3i')}; {b}_im={self.add('t1i','t3r')};")
            self.o(f"  {dd}_re={self.add('t1r','t3i')}; {dd}_im={self.sub('t1i','t3r')};")
        self.o(f"}}")

    # ── Internal W16 twiddle ──
    def emit_twiddle(self, dst, src, e, tN, d):
        _, typ = twiddle_is_trivial(e, tN)
        fwd = (d == 'fwd')
        T = self.isa.T

        if typ == 'one':
            if dst != src:
                self.o(f"{dst}_re={src}_re; {dst}_im={src}_im;")
        elif typ == 'neg_one':
            self.o(f"{dst}_re={self.neg(f'{src}_re')}; {dst}_im={self.neg(f'{src}_im')};")
        elif typ == 'neg_j':
            if fwd:
                self.o(f"{{ {T} t={src}_re; {dst}_re={src}_im; {dst}_im={self.neg('t')}; }}")
            else:
                self.o(f"{{ {T} t={src}_re; {dst}_re={self.neg(f'{src}_im')}; {dst}_im=t; }}")
        elif typ == 'pos_j':
            if fwd:
                self.o(f"{{ {T} t={src}_re; {dst}_re={self.neg(f'{src}_im')}; {dst}_im=t; }}")
            else:
                self.o(f"{{ {T} t={src}_re; {dst}_re={src}_im; {dst}_im={self.neg('t')}; }}")
        elif typ == 'w8_1':
            self.o(f"{{ {T} tr={src}_re,ti={src}_im;")
            if fwd:
                self.o(f"  {dst}_re={self.mul(self.add('tr','ti'),'sqrt2_inv')}; {dst}_im={self.mul(self.sub('ti','tr'),'sqrt2_inv')}; }}")
            else:
                self.o(f"  {dst}_re={self.mul(self.sub('tr','ti'),'sqrt2_inv')}; {dst}_im={self.mul(self.add('tr','ti'),'sqrt2_inv')}; }}")
        elif typ == 'w8_3':
            self.o(f"{{ {T} tr={src}_re,ti={src}_im;")
            if fwd:
                self.o(f"  {dst}_re={self.mul(self.sub('ti','tr'),'sqrt2_inv')}; {dst}_im={self.neg(self.mul(self.add('tr','ti'),'sqrt2_inv'))}; }}")
            else:
                self.o(f"  {dst}_re={self.neg(self.mul(self.add('tr','ti'),'sqrt2_inv'))}; {dst}_im={self.mul(self.sub('tr','ti'),'sqrt2_inv')}; }}")
        elif typ == 'neg_w8_1':
            self.o(f"{{ {T} tr={src}_re,ti={src}_im;")
            if fwd:
                self.o(f"  {dst}_re={self.neg(self.mul(self.add('tr','ti'),'sqrt2_inv'))}; {dst}_im={self.mul(self.sub('tr','ti'),'sqrt2_inv')}; }}")
            else:
                self.o(f"  {dst}_re={self.mul(self.sub('ti','tr'),'sqrt2_inv')}; {dst}_im={self.neg(self.mul(self.add('tr','ti'),'sqrt2_inv'))}; }}")
        elif typ == 'neg_w8_3':
            self.o(f"{{ {T} tr={src}_re,ti={src}_im;")
            if fwd:
                self.o(f"  {dst}_re={self.mul(self.sub('tr','ti'),'sqrt2_inv')}; {dst}_im={self.mul(self.add('tr','ti'),'sqrt2_inv')}; }}")
            else:
                self.o(f"  {dst}_re={self.mul(self.add('tr','ti'),'sqrt2_inv')}; {dst}_im={self.mul(self.sub('ti','tr'),'sqrt2_inv')}; }}")
        else:
            label = wN_label(e, tN)
            if self.isa.name == 'scalar':
                self.o(f"{{ double tr={src}_re;")
                if fwd:
                    self.o(f"  {dst}_re={src}_re*{label}_re - {src}_im*{label}_im;")
                    self.o(f"  {dst}_im=tr*{label}_im + {src}_im*{label}_re; }}")
                else:
                    self.o(f"  {dst}_re={src}_re*{label}_re + {src}_im*{label}_im;")
                    self.o(f"  {dst}_im={src}_im*{label}_re - tr*{label}_im; }}")
            else:
                self.o(f"{{ {T} tr={src}_re;")
                if fwd:
                    self.o(f"  {dst}_re={self.fms(f'{src}_re',f'tw_{label}_re',self.mul(f'{src}_im',f'tw_{label}_im'))};")
                    self.o(f"  {dst}_im={self.fma('tr',f'tw_{label}_im',self.mul(f'{src}_im',f'tw_{label}_re'))}; }}")
                else:
                    self.o(f"  {dst}_re={self.fma(f'{src}_re',f'tw_{label}_re',self.mul(f'{src}_im',f'tw_{label}_im'))};")
                    self.o(f"  {dst}_im={self.fms(f'{src}_im',f'tw_{label}_re',self.mul('tr',f'tw_{label}_im'))}; }}")

    # ── External twiddle ──
    def _tw_addr(self, tw_idx, ke="k"):
        if self.addr_mode in ('t1', 't1_oop', 't1_buf'):
            return f"{tw_idx}*me+m"
        return f"{tw_idx}*K+{ke}"

    def _tw_buf(self):
        return "W_re" if self.addr_mode in ('t1', 't1_oop', 't1_buf') else "tw_re"

    def _tw_buf_im(self):
        return "W_im" if self.addr_mode in ('t1', 't1_oop', 't1_buf') else "tw_im"

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

    # ── Complex multiply (log3 derivation) ──
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


# ═══════════════════════════════════════════════════════════════
# LOG3 TWIDDLE DERIVATION TABLE
# ═══════════════════════════════════════════════════════════════

LOG3_BASE_MAP = {'b1': 0, 'b2': 1, 'b4': 3, 'b8': 7}

def _log3_subfft_chain(n2):
    if n2 == 0:
        return [(0, None), (1, ('base', 'b4')), (2, ('base', 'b8')),
                (3, ('derive', 'w12', 'b4', 'b8'))]
    elif n2 == 1:
        return [(0, ('base', 'b1')), (1, ('derive', 'w5', 'b1', 'b4')),
                (2, ('derive', 'w9', 'b1', 'b8')),
                (3, ('derive', 'w13', 'w5', 'b8'))]
    elif n2 == 2:
        return [(0, ('base', 'b2')), (1, ('derive', 'w6', 'b2', 'b4')),
                (2, ('derive', 'w10', 'b2', 'b8')),
                (3, ('derive', 'w14', 'w6', 'b8'))]
    else:
        return [(0, ('derive', 'w3', 'b1', 'b2')),
                (1, ('derive', 'w7', 'w3', 'b4')),
                (2, ('derive', 'w11', 'w3', 'b8')),
                (3, ('derive', 'w15', 'w7', 'b8'))]


def emit_kernel_body_log3(em, d, itw_set, variant):
    """Emit DIT or DIF log3 kernel body. AVX-512 only."""
    assert variant in ('dit_tw_log3', 'dif_tw_log3')
    is_dit = variant == 'dit_tw_log3'
    T = em.isa.T
    xv4 = [f"x{i}" for i in range(N1)]
    ld = f"{em.isa.p}_load_pd"

    em.c("Load 4 log3 base twiddles: W^1, W^2, W^4, W^8")
    tb, tbi = em._tw_buf(), em._tw_buf_im()
    for name, idx in sorted(LOG3_BASE_MAP.items(), key=lambda x: x[1]):
        ta = em._tw_addr(idx)
        em.o(f"const {T} {name}_re = {ld}(&{tb}[{ta}]);")
        em.o(f"const {T} {name}_im = {ld}(&{tbi}[{ta}]);")
    em.n_load += 8
    em.b()

    em.o(f"{T} tw_dr, tw_di;")
    em.b()

    for n2 in range(N2):
        chain = _log3_subfft_chain(n2)
        em.c(f"sub-FFT n2={n2}")

        derived_live = {}

        for n1, tw_action in chain:
            n = N2 * n1 + n2
            em.emit_load(f"x{n1}", n)

            if is_dit and tw_action is not None:
                if tw_action[0] == 'base':
                    bname = tw_action[1]
                    em.emit_cmul_inplace(f"x{n1}", f"{bname}_re", f"{bname}_im", d)
                elif tw_action[0] == 'derive':
                    wname, a, b = tw_action[1], tw_action[2], tw_action[3]
                    em.o(f"{T} {wname}_re, {wname}_im;")
                    em.emit_cmul(f"{wname}_re", f"{wname}_im",
                                 f"{a}_re", f"{a}_im", f"{b}_re", f"{b}_im", 'fwd')
                    em.emit_cmul_inplace(f"x{n1}", f"{wname}_re", f"{wname}_im", d)
                    derived_live[wname] = True

        em.b()
        em.emit_radix4(xv4, d, f"radix-4 n2={n2}")
        em.b()
        for k1 in range(N1):
            em.emit_spill(f"x{k1}", n2 * N1 + k1)
        em.b()

    em.c("PASS 2")
    em.b()
    for k1 in range(N1):
        em.c(f"column k1={k1}")
        for n2_ in range(N2):
            em.emit_reload(f"x{n2_}", n2_ * N1 + k1)
        em.b()
        if k1 > 0:
            for n2_ in range(1, N2):
                e = (n2_ * k1) % N
                em.emit_twiddle(f"x{n2_}", f"x{n2_}", e, N, d)
            em.b()
        em.emit_radix4(xv4, d, f"radix-4 k1={k1}")
        em.b()

        if not is_dit:
            # DIF log3 twiddle application, PASS 2 — apply to x{k2} AFTER butterfly.
            #
            # Chain-sharing structure: within a given k1 column, the 4 output
            # legs have indices m = k1 + 4*k2 for k2 in 0..3. These form exactly
            # the same set of m-values as DIT's sub-FFT n2=k1 handles in PASS 1
            # (since {N2*n1 + n2 : n1 in 0..3} = {k1 + N1*k2 : k2 in 0..3} when
            # n2 = k1 and N1 = N2 = 4). So DIT's _log3_subfft_chain(n2=k1)
            # applies verbatim here — same derivations, same live-set, same
            # cmul count.
            #
            # This saves ~19% of per-iteration cmuls vs the naive "recompute
            # the product of bases for each leg" approach: the DIT-style chain
            # uses 11 derivations for 15 non-DC legs; naive recompute uses 17.
            #
            # Mapping: the chain returns (n1, action) where n1 corresponds to
            # the k2 output index here. The apply target is x{k2} (=x{n1} in
            # chain terms). Derived twiddle names (w3, w5, w7, ...) match DIT's
            # naming since they refer to the same mathematical W^n.
            chain = _log3_subfft_chain(k1)
            for n1, tw_action in chain:
                k2 = n1  # rename for clarity: n1 in the chain is k2 here
                m = k1 + N1 * k2
                if tw_action is None:
                    assert m == 0, f"None action only for DC leg; got m={m}"
                    continue
                if tw_action[0] == 'base':
                    bname = tw_action[1]
                    em.emit_cmul_inplace(f"x{k2}", f"{bname}_re", f"{bname}_im", d)
                elif tw_action[0] == 'derive':
                    wname, a, b = tw_action[1], tw_action[2], tw_action[3]
                    # Only declare if we haven't already this pass 2 iteration.
                    # The derived_live dict in the DIT branch lives inside the
                    # sub-FFT loop; here we scope to this k1 iteration so each
                    # pass emits its own fresh declarations. Collisions across
                    # k1 iterations are prevented by the enclosing `{ ... }`
                    # block structure emit_radix4 uses.
                    em.o(f"{T} {wname}_re, {wname}_im;")
                    em.emit_cmul(f"{wname}_re", f"{wname}_im",
                                 f"{a}_re", f"{a}_im", f"{b}_re", f"{b}_im", 'fwd')
                    em.emit_cmul_inplace(f"x{k2}", f"{wname}_re", f"{wname}_im", d)
            em.b()

        for k2 in range(N2):
            em.emit_store(f"x{k2}", k1 + N1 * k2)
        em.b()


# ═══════════════════════════════════════════════════════════════
# LOG3 ISUB2: paired sub-FFT scheduling for AVX-512
# ═══════════════════════════════════════════════════════════════
# Same log3 twiddle protocol (W_re has me doubles = w1 only; codelet loads
# the 4 bases b1,b2,b4,b8 = W^1, W^2, W^4, W^8 per m-iteration, derives the
# other 11 twiddles).
#
# Where log3 runs each sub-FFT's derive+apply+radix-4+spill sequentially,
# isub2 pairs (n2=0, n2=1) and (n2=2, n2=3) and interleaves their derivation
# work to hide the longer chain behind the shorter one. The critical
# critical-path chain in log3 for R=16 is w15 = w7 * b8 = (w3*b4) * b8 =
# ((b1*b2)*b4) * b8 — three serial FMAs = ~12 cycles FMA latency on
# Golden Cove / Skylake-server class cores. In sequential log3, n2=3
# blocks on this chain with nothing useful to overlap with. In isub2,
# n2=2's derive+apply+radix-4 runs concurrently so the FMA pipe stays fed.
#
# Register budget: AVX-512 (32 ZMM). Peak observed live set in scheduler
# simulation: 28 ZMM for pair (2,3), 26 for pair (0,1). See discussion
# below for the staggered-load approach that made these fit.
#
# Not provided for AVX2 or scalar: AVX2 has only 16 YMM — the
# pair-live working set alone is 16 YMM before any twiddle derivation,
# which forces spilling and defeats the optimization. Registered in
# VARIANTS with supported_isas={'avx512'} and filtered by candidates.py.

# Per-sub-FFT twiddle action table reused from log3.
# Format: list of (n1, tw_action) where tw_action is one of:
#   None                                    — no twiddle (DC leg)
#   ('base', bname)                         — apply a loaded base twiddle
#   ('derive', wname, src_a, src_b)         — derive wname = src_a * src_b, then apply wname
#
# For isub2, we additionally need to know each derived twiddle's remaining
# uses so we can kill it at the right apply point. The table below is a
# straight per-sub-FFT definition; scheduling logic computes the use-graph
# and orders the derivation/apply operations to minimize peak liveness.

def _isub2_pair_plan(nA, nB):
    """Build the interleaved schedule for one (nA, nB) pair.

    Returns a list of operation dicts, each one of:
      {'op': 'load',    'sub': nA|nB, 'n1': 0..3}
      {'op': 'derive',  'name': wname, 'a': src_a, 'b': src_b}
      {'op': 'apply',   'sub': nA|nB, 'n1': 0..3, 'tw': tw_name,
       'kill': tw_name_or_None}
      {'op': 'radix4',  'sub': nA|nB}
      {'op': 'spill',   'sub': nA|nB, 'n1': 0..3}

    The 'kill' field tells the emitter that this was a derived twiddle's
    last use and the register can be reused (conceptually — in practice we
    rely on the C compiler's liveness analysis, but by emitting operations
    in the schedule order below we give it the shortest possible live
    ranges to work with).

    The schedule follows the staggered-load pattern (validated in
    scheduler simulation to peak at ~28 ZMM):
      - load nA's data
      - derive the deep-chain inputs early (both sub-FFTs)
      - apply nA's twiddles, interleaved with remaining derivations
      - radix-4 and spill nA
      - finish final deep derivation (e.g. w15)
      - load nB's data
      - apply nB's twiddles
      - radix-4 and spill nB
    """
    chain_A = _log3_subfft_chain(nA)
    chain_B = _log3_subfft_chain(nB)

    # Collect all derivations from both sub-FFTs, with use-counts.
    # A use is: (consumer_type, consumer_id) where consumer_type ∈
    # {'derive_into', 'apply_to'} and consumer_id identifies which
    # derivation or which x-leg uses this twiddle.
    derivations = {}   # name -> (src_a, src_b)
    uses = {}          # name -> list of consumer tags
    leg_tw = {}        # (sub, n1) -> tw_name or ('base', bname) or None

    for sub_id, chain in [(nA, chain_A), (nB, chain_B)]:
        for n1, action in chain:
            if action is None:
                leg_tw[(sub_id, n1)] = None
            elif action[0] == 'base':
                leg_tw[(sub_id, n1)] = ('base', action[1])
            elif action[0] == 'derive':
                _, wname, a, b = action
                derivations[wname] = (a, b)
                leg_tw[(sub_id, n1)] = ('derived', wname)
                uses.setdefault(wname, []).append(('apply_to', (sub_id, n1)))
                # Record derive-dependencies on other derived twiddles
                for src in (a, b):
                    if src.startswith('w'):
                        uses.setdefault(src, []).append(('derive_into', wname))

    # ---------- Step-by-step schedule construction ----------
    ops = []

    # 1. Load nA data
    for n1 in range(4):
        ops.append({'op': 'load', 'sub': nA, 'n1': n1})

    # 2. Derive deep-chain roots from both sub-FFTs first.
    # A "root" is a derivation whose inputs are bases only (not derived).
    # Topological order: level-1 derivations first.
    def depth(wname, seen=None):
        if seen is None: seen = set()
        if wname in seen: return 0
        seen.add(wname)
        a, b = derivations[wname]
        d = 0
        if a.startswith('w'): d = max(d, depth(a, seen))
        if b.startswith('w'): d = max(d, depth(b, seen))
        return d + 1

    all_derivs = list(derivations.keys())
    depths = {w: depth(w) for w in all_derivs}

    # Ordering strategy:
    #   - Level-1 derivations (both sub-FFTs) first, in a specific order
    #     that lets us kick off the deeper chains ASAP.
    #   - Then level-2 derivations from B's chain before A's apply work,
    #     so B's deep chain runs in parallel with A's applies.
    #   - Final level-3 derivation (w15 in pair (2,3)) just before
    #     loading B's data.
    # We compute this via topo sort + a priority for B's chain.

    # Track which twiddles are "ready" (all inputs have been derived or
    # are base/permanent), "derived" (emitted), "applied" (consumed by
    # the leg that uses them).
    derived = set()
    applied = set()  # derived twiddles whose last use has been emitted

    def ready(w):
        a, b = derivations[w]
        for src in (a, b):
            if src.startswith('w') and src not in derived:
                return False
        return True

    # Priority: derivations in B's chain first (longer), then A's.
    # Within the same priority, level-1 before level-2 before level-3.
    def prio(w):
        # Which sub-FFT is this derivation's ultimate consumer?
        # It's the sub-FFT where w gets applied. We find it via uses[w].
        consumer_sub = None
        for kind, tag in uses.get(w, []):
            if kind == 'apply_to':
                consumer_sub = tag[0]
                break
        # B's derivations get priority 0, A's priority 1.
        p_sub = 0 if consumer_sub == nB else 1
        return (p_sub, depths[w])

    # Issue level-1 derivations from BOTH sub-FFTs first (they can all run
    # in parallel after the bases are loaded).
    level1 = sorted([w for w in all_derivs if depths[w] == 1], key=prio)
    for w in level1:
        a, b = derivations[w]
        ops.append({'op': 'derive', 'name': w, 'a': a, 'b': b})
        derived.add(w)

    # Issue level-2 derivations interleaved with nA applies.
    # Simple heuristic: emit the deepest-chain B derivation first (e.g. w7
    # in pair 2,3), then apply nA's shallow legs, interleaving remaining
    # level-2s as their inputs become dead.
    level2 = sorted([w for w in all_derivs if depths[w] == 2], key=prio)

    # Which nA legs are ready to apply? Legs whose twiddle is either base
    # or already derived.
    def leg_applyable(sub, n1):
        tw = leg_tw[(sub, n1)]
        if tw is None: return True  # no twiddle
        if tw[0] == 'base': return True
        # derived: need the derived twiddle to have been emitted
        return tw[1] in derived and tw[1] not in applied

    def last_use(wname):
        """True if all other uses of wname have been resolved (derived or
        applied), so applying/deriving with it now kills it."""
        for kind, tag in uses[wname]:
            if kind == 'derive_into':
                if tag not in derived:
                    return False
            elif kind == 'apply_to':
                # If it's not THIS application we're considering, it must be done.
                # For now we conservatively return False; the actual caller passes
                # the current operation and checks "is this the last open use?".
                pass
        return True

    def emit_apply(sub, n1):
        tw = leg_tw[(sub, n1)]
        if tw is None:
            # No-op (DC leg). Skip — the kernel-body emitter also skips these.
            return
        if tw[0] == 'base':
            ops.append({'op': 'apply', 'sub': sub, 'n1': n1,
                        'tw': ('base', tw[1]), 'kill': None})
            return
        wname = tw[1]
        # Is this the last remaining use of wname?
        remaining = []
        for kind, tag in uses[wname]:
            if kind == 'derive_into':
                if tag not in derived:
                    remaining.append((kind, tag))
            elif kind == 'apply_to':
                if tag != (sub, n1) and tag not in applied_legs:
                    remaining.append((kind, tag))
        kill = wname if len(remaining) == 0 else None
        ops.append({'op': 'apply', 'sub': sub, 'n1': n1,
                    'tw': ('derived', wname), 'kill': kill})
        applied_legs.add((sub, n1))
        if kill:
            applied.add(wname)

    applied_legs = set()

    # Emit derivations (level2, then level3) and nA applies in a greedy
    # interleave: whenever a level-k derivation's inputs are ready AND
    # applying that derivation's input-twiddles first would save a register
    # (via last-use), do it.

    # Simplest scheme that respects dependencies: for each level-2 deriv,
    # emit the derivation; then if any nA leg became applyable AND applying
    # it kills a no-longer-needed derived twiddle, apply it.

    # In practice the greedy choice for pair (2,3) is hand-validated (see
    # the module-header scheduler simulation), and for pair (0,1) is even
    # easier. We emit a canonical order:

    # Step A: derive level-2s in priority order, interleaved with nA applies.
    pending_nA_applies = [(nA, n1) for n1 in range(4)]

    for w in level2:
        a, b = derivations[w]
        ops.append({'op': 'derive', 'name': w, 'a': a, 'b': b})
        derived.add(w)
        # After deriving w, check: is any of w's deriv-inputs now
        # exhausted for derive-uses? (i.e., all derive_into uses resolved)
        # If so, and that input has an apply-use on nA, fire the apply.
        for src in (a, b):
            if not src.startswith('w'): continue
            # Is src still needed for any more derivations?
            still_needed_for_derive = any(
                kind == 'derive_into' and tag not in derived
                for kind, tag in uses[src]
            )
            if not still_needed_for_derive:
                # Fire any apply-uses of src that target nA.
                for kind, tag in list(uses[src]):
                    if (kind == 'apply_to' and tag[0] == nA and
                            tag not in applied_legs):
                        emit_apply(*tag)
        # Also fire any nA applies that became possible and are base-only
        # (they're always possible; emit them early to clear the pending list).
        for (sub, n1) in list(pending_nA_applies):
            if (sub, n1) in applied_legs:
                pending_nA_applies.remove((sub, n1))
                continue
            tw = leg_tw[(sub, n1)]
            if tw is None or (tw[0] == 'base'):
                emit_apply(sub, n1)
                pending_nA_applies.remove((sub, n1))

    # Step B: fire remaining nA applies (whatever's still pending).
    for (sub, n1) in list(pending_nA_applies):
        if (sub, n1) in applied_legs:
            continue
        tw = leg_tw[(sub, n1)]
        # Derived twiddles that nA needs but haven't been applied yet.
        if tw is None or tw[0] == 'base':
            emit_apply(sub, n1)
        else:
            wname = tw[1]
            if wname in derived:
                emit_apply(sub, n1)

    # Step C: radix-4 + spill nA.
    ops.append({'op': 'radix4', 'sub': nA})
    for n1 in range(4):
        ops.append({'op': 'spill', 'sub': nA, 'n1': n1})

    # Step D: emit level-3 derivations (e.g. w15 for pair 2,3).
    level3 = sorted([w for w in all_derivs if depths[w] >= 3], key=prio)
    for w in level3:
        a, b = derivations[w]
        ops.append({'op': 'derive', 'name': w, 'a': a, 'b': b})
        derived.add(w)

    # Step E: load nB data.
    for n1 in range(4):
        ops.append({'op': 'load', 'sub': nB, 'n1': n1})

    # Step F: apply nB's twiddles.
    for n1 in range(4):
        emit_apply(nB, n1)

    # Step G: radix-4 + spill nB.
    ops.append({'op': 'radix4', 'sub': nB})
    for n1 in range(4):
        ops.append({'op': 'spill', 'sub': nB, 'n1': n1})

    return ops


def emit_kernel_body_log3_isub2(em, d, itw_set, variant):
    """Emit the DIT log3_isub2 kernel body. AVX-512 only.

    Layout: same as log3 (PASS 1 = N2 radix-4 sub-FFTs with twiddle derives;
    PASS 2 = N1 radix-4 column combines with internal W16 twiddles). The
    difference is PASS 1's scheduling: sub-FFTs (0,1) and (2,3) are paired
    and their derivation work is interleaved.

    The DIF variant is not provided — the isub2 technique has no benefit
    for DIF (the twiddle is applied after the radix-4, so the critical
    dependency chain runs through different arithmetic). Only 'dit_tw_log3'
    is accepted.
    """
    assert variant == 'dit_tw_log3', \
        "log3_isub2 is DIT-only; DIF has different chain structure"
    assert em.isa.name == 'avx512', \
        "log3_isub2 requires AVX-512 for register budget"

    T = em.isa.T
    xv4 = [f"x{i}" for i in range(N1)]
    ld = f"{em.isa.p}_load_pd"

    em.c("Load 4 log3 base twiddles: W^1, W^2, W^4, W^8")
    tb, tbi = em._tw_buf(), em._tw_buf_im()
    for name, idx in sorted(LOG3_BASE_MAP.items(), key=lambda x: x[1]):
        ta = em._tw_addr(idx)
        em.o(f"const {T} {name}_re = {ld}(&{tb}[{ta}]);")
        em.o(f"const {T} {name}_im = {ld}(&{tbi}[{ta}]);")
    em.n_load += 8
    em.b()

    # We need two sets of working-data names: one for nA, one for nB of a
    # pair. Scheme: rename x0..x3 → x0a..x3a for nA, x0b..x3b for nB. Only
    # one set is live at a time (staggered loads), but their names must
    # differ so the emitted C declarations don't collide.
    em.o(f"{T} x0a_re, x0a_im, x1a_re, x1a_im, "
         f"x2a_re, x2a_im, x3a_re, x3a_im;")
    em.o(f"{T} x0b_re, x0b_im, x1b_re, x1b_im, "
         f"x2b_re, x2b_im, x3b_re, x3b_im;")
    em.b()

    # Helpers bound to the right variable-name suffix.
    def _xname(sub, n1, nA, nB):
        suffix = 'a' if sub == nA else 'b'
        return f"x{n1}{suffix}"

    # Emit the two pairs.
    for (nA, nB) in [(0, 1), (2, 3)]:
        em.c(f"─── pair (n2={nA}, n2={nB}) ───")
        plan = _isub2_pair_plan(nA, nB)
        for op in plan:
            k = op['op']
            if k == 'load':
                # Standard emit_load but using xNa/xNb name instead of xN.
                # emit_load writes to `{v}_re`/`{v}_im`, so pass the right v.
                xv = _xname(op['sub'], op['n1'], nA, nB)
                n_global = N2 * op['n1'] + op['sub']
                em.emit_load(xv, n_global)
            elif k == 'derive':
                # Declare and compute the derived twiddle.
                w, a, b = op['name'], op['a'], op['b']
                em.o(f"{T} {w}_re, {w}_im;")
                em.emit_cmul(f"{w}_re", f"{w}_im",
                             f"{a}_re", f"{a}_im",
                             f"{b}_re", f"{b}_im", 'fwd')
            elif k == 'apply':
                xv = _xname(op['sub'], op['n1'], nA, nB)
                tw = op['tw']
                if tw is None:
                    continue  # shouldn't reach here but safe
                tw_kind, tw_name = tw
                em.emit_cmul_inplace(xv, f"{tw_name}_re", f"{tw_name}_im", d)
            elif k == 'radix4':
                # Butterfly the 4 legs of this sub-FFT.
                suffix = 'a' if op['sub'] == nA else 'b'
                local_xv4 = [f"x{i}{suffix}" for i in range(N1)]
                em.emit_radix4(local_xv4, d, f"radix-4 n2={op['sub']}")
            elif k == 'spill':
                xv = _xname(op['sub'], op['n1'], nA, nB)
                em.emit_spill(xv, op['sub'] * N1 + op['n1'])
            else:
                raise RuntimeError(f"unknown op kind in plan: {k}")
        em.b()

    # PASS 2: identical to log3 (4 column combines, internal W16 twiddles).
    em.c("PASS 2 — identical to log3")
    em.b()
    for k1 in range(N1):
        em.c(f"column k1={k1}")
        for n2_ in range(N2):
            em.emit_reload(f"x{n2_}", n2_ * N1 + k1)
        em.b()
        if k1 > 0:
            for n2_ in range(1, N2):
                e = (n2_ * k1) % N
                em.emit_twiddle(f"x{n2_}", f"x{n2_}", e, N, d)
            em.b()
        em.emit_radix4(xv4, d, f"radix-4 k1={k1}")
        em.b()
        for k2 in range(N2):
            em.emit_store(f"x{k2}", k1 + N1 * k2)
        em.b()


# ═══════════════════════════════════════════════════════════════
# LOG_HALF TWIDDLE DERIVATION TABLE
# ═══════════════════════════════════════════════════════════════
# Loads 8 base twiddles W^1..W^8; derives W^9..W^15 each as one depth-1
# cmul of the form W^(8+k) = W^k * W^8 for k ∈ {1..7}.
#
# Rationale: fills the gap between log3 (4 loads, 11 derivations with up
# to depth-3 chains) and flat (15 loads, 0 derivations). Wins the regime
# where log3's derivation chains dominate but flat's bandwidth is still
# affordable — specifically AVX-512 me=64 cells on Intel server cores
# measured at this portfolio (flat beats log3 by 12-16% there).
#
# All 7 derivations are mutually independent (each uses b8 and one unique
# b_k) so they issue in parallel with no dependency bubbles. The scheduler
# can overlap derivation with load and with application of the eight base
# twiddles, which already have no derivation step.
#
# Protocol: reuses 'log3' protocol (the harness allocates (R-1)*me for it
# even though log3 only reads 4 slots, per the sparse-flat convention in
# harness.c twiddle_doubles()). log_half reads 8 of those 15 slots. No
# allocator change needed.
#
# Protocol note: the twiddle buffer is treated as flat-indexed
# W_re[tw_idx*me + m] where tw_idx is 1-based (tw_idx=1 → W^1, ..., 15 → W^15).
# So the base loads read from W_re[(k-1)*me + m] for k in 1..8, i.e.
# tw_idx = 0, 1, 2, 3, 4, 5, 6, 7 for W^1..W^8.

# Base map: bname -> tw_idx (0-based position in the flat twiddle buffer).
# Matches the indexing used in emit_ext_tw where twiddle j of a leg is at
# tw_idx = j-1 (since leg 0 is DC and has no twiddle).
LOG_HALF_LOADED = [1, 2, 3, 4, 5, 6, 7, 8]
LOG_HALF_BASE_MAP = {f'b{k}': k - 1 for k in LOG_HALF_LOADED}  # b1→0, b2→1, ..., b8→7


def _log_half_subfft_chain(n2):
    """Per-sub-FFT twiddle-action table in the log_half scheme.

    Each element is (n1, action) where action is:
      None                              — DC leg, no twiddle apply
      ('base', bname)                   — apply a loaded base (b1..b8)
      ('derive_once', wname, a, 'b8')   — derive wname = a * b8, then apply
                                          a ∈ {b1..b7}, b is always b8

    This is a strict subset of the general "derive tw" action shape used
    by log3, specialized to the known-flat structure of log_half.
    """
    out = []
    for n1 in range(N1):
        n = N2 * n1 + n2
        if n == 0:
            out.append((n1, None))
        elif n in LOG_HALF_LOADED:
            out.append((n1, ('base', f'b{n}')))
        else:
            # n ∈ {9..15}: derive as W^(n-8) * W^8.
            k = n - 8
            out.append((n1, ('derive_once', f'w{n}', f'b{k}', 'b8')))
    return out


def emit_kernel_body_log_half(em, d, itw_set, variant):
    """Emit DIT log_half kernel body. AVX-512 only (8-base register footprint
    exceeds AVX2's 16-YMM budget once combined with working data + derived
    twiddles + radix-4 temps).

    Structure:
      1. Load 8 bases (b1..b8) from W_re/W_im at offsets (k-1)*me+m.
      2. For each sub-FFT n2 in 0..3:
         a. Load x0..x3 working data.
         b. Apply twiddles per _log_half_subfft_chain table.
         c. Radix-4 butterfly.
         d. Spill results.
      3. PASS 2 identical to log3: 4 column combines with internal W16
         twiddles, then store.

    The DIF variant is not provided — derivation pattern is DIT-oriented
    (derivations interleave with data application inside PASS 1). A DIF
    analog would apply twiddles after the radix-4 in PASS 2 and has a
    different scheduling profile that isn't tackled here.
    """
    assert variant == 'dit_tw_log_half', \
        "log_half is DIT-only for now; DIF variant not yet supported"
    assert em.isa.name == 'avx512', \
        "log_half requires AVX-512 for register budget"

    T = em.isa.T
    xv4 = [f"x{i}" for i in range(N1)]
    ld = f"{em.isa.p}_load_pd"

    em.c("Load 8 log_half base twiddles: W^1..W^8")
    tb, tbi = em._tw_buf(), em._tw_buf_im()
    for name, idx in sorted(LOG_HALF_BASE_MAP.items(), key=lambda x: x[1]):
        ta = em._tw_addr(idx)
        em.o(f"const {T} {name}_re = {ld}(&{tb}[{ta}]);")
        em.o(f"const {T} {name}_im = {ld}(&{tbi}[{ta}]);")
    em.n_load += 16   # 8 base twiddles × (re + im)
    em.b()

    # Sub-FFTs: PASS 1 identical structure to log3, just with a simpler chain.
    for n2 in range(N2):
        chain = _log_half_subfft_chain(n2)
        em.c(f"sub-FFT n2={n2}")

        for n1, tw_action in chain:
            n = N2 * n1 + n2
            em.emit_load(f"x{n1}", n)

            if tw_action is None:
                # DC leg — no twiddle
                continue
            if tw_action[0] == 'base':
                bname = tw_action[1]
                em.emit_cmul_inplace(f"x{n1}", f"{bname}_re", f"{bname}_im", d)
            elif tw_action[0] == 'derive_once':
                _, wname, a, b = tw_action
                em.o(f"{T} {wname}_re, {wname}_im;")
                em.emit_cmul(f"{wname}_re", f"{wname}_im",
                             f"{a}_re", f"{a}_im",
                             f"{b}_re", f"{b}_im", 'fwd')
                em.emit_cmul_inplace(f"x{n1}", f"{wname}_re", f"{wname}_im", d)
            else:
                raise RuntimeError(f"unknown action {tw_action}")

        em.b()
        em.emit_radix4(xv4, d, f"radix-4 n2={n2}")
        em.b()
        for k1 in range(N1):
            em.emit_spill(f"x{k1}", n2 * N1 + k1)
        em.b()

    # PASS 2: identical to log3.
    em.c("PASS 2 — identical to log3")
    em.b()
    for k1 in range(N1):
        em.c(f"column k1={k1}")
        for n2_ in range(N2):
            em.emit_reload(f"x{n2_}", n2_ * N1 + k1)
        em.b()
        if k1 > 0:
            for n2_ in range(1, N2):
                e = (n2_ * k1) % N
                em.emit_twiddle(f"x{n2_}", f"x{n2_}", e, N, d)
            em.b()
        em.emit_radix4(xv4, d, f"radix-4 k1={k1}")
        em.b()
        for k2 in range(N2):
            em.emit_store(f"x{k2}", k1 + N1 * k2)
        em.b()


# ═══════════════════════════════════════════════════════════════
# SHARED KERNEL BODY — supports buffered variants via addr_mode
# ═══════════════════════════════════════════════════════════════

def emit_kernel_body(em, d, itw_set, variant):
    """Emit the inner loop body. variant: 'notw', 'dit_tw', 'dif_tw'
    (optionally '_scalar' suffix for t1s). Addr mode determines store target."""
    xv4 = [f"x{i}" for i in range(N1)]

    # PASS 1: N2 radix-4 sub-FFTs
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

        # Twiddle prefetch (only for t1/t1_buf, NOT for t2 with 'dit_tw')
        if variant in ('dit_tw', 'dit_tw_scalar') and n2 < N2 - 1 and em.addr_mode in ('t1', 't1_buf', 't1_oop'):
            next_n2 = n2 + 1
            pf_indices = []
            for n1_pf in range(N1):
                n_pf = N2 * n1_pf + next_n2
                if n_pf > 0:
                    pf_indices.append(n_pf - 1)
            pf_i = 0
            for k1 in range(N1):
                if pf_i < len(pf_indices):
                    tw_idx = pf_indices[pf_i]
                    em.o(f"_mm_prefetch((const char*)&W_re[{tw_idx}*me+m], _MM_HINT_T0);")
                    em.o(f"_mm_prefetch((const char*)&W_im[{tw_idx}*me+m], _MM_HINT_T0);")
                    pf_i += 1
                em.emit_spill(f"x{k1}", n2 * N1 + k1)
        else:
            for k1 in range(N1):
                em.emit_spill(f"x{k1}", n2 * N1 + k1)
        em.b()

    # PASS 2: N1 radix-4 column combines
    if getattr(em, '_defer_consts', False):
        isa = em.isa
        T = isa.T
        em.c("PASS 2 — deferred constants (free regs during PASS 1)")
        if isa.name != 'scalar':
            em.o(f"const {T} sign_flip = {isa.p}_set1_pd(-0.0);")
            em.o(f"const {T} sqrt2_inv = {isa.p}_set1_pd(0.70710678118654752440);")
        else:
            em.o(f"const double sqrt2_inv = 0.70710678118654752440;")
        itw_set_l = getattr(em, '_itw_set', set())
        if itw_set_l:
            if isa.name != 'scalar':
                set1 = f"{isa.p}_set1_pd"
                for (e, tN) in sorted(itw_set_l):
                    label = wN_label(e, tN)
                    em.o(f"const {T} tw_{label}_re = {set1}({label}_re);")
                    em.o(f"const {T} tw_{label}_im = {set1}({label}_im);")
            else:
                for (e, tN) in sorted(itw_set_l):
                    label = wN_label(e, tN)
                    em.o(f"const double tw_{label}_re = {label}_re;")
                    em.o(f"const double tw_{label}_im = {label}_im;")
    else:
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
        em.emit_radix4(xv4, d, f"radix-4 k1={k1}")
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


# ═══════════════════════════════════════════════════════════════
# FILE EMITTER — t2 variants (notw/dit_tw/dif_tw)
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


def insert_stats_into_header(lines, stats):
    table = [" *", " * ── Operation counts per k-step ──", " *"]
    s5 = '─' * 5; s3 = '─' * 3; s4 = '─' * 4; sep = '─' * 20
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


def emit_file(isa, itw_set, variant):
    """Emit complete header for one ISA and one variant (t2: notw/dit_tw/dif_tw)."""
    em = Emitter(isa)
    T = isa.T
    is_log3 = variant.endswith('_log3')

    if variant == 'notw':
        func_base = 'radix16_n1_dit_kernel'; tw_params = None
    elif variant == 'dit_tw':
        func_base = 'radix16_tw_flat_dit_kernel'; tw_params = 'flat'
    elif variant == 'dit_tw_log3':
        func_base = 'radix16_tw_log3_dit_kernel'; tw_params = 'log3'
    elif variant == 'dif_tw':
        func_base = 'radix16_tw_flat_dif_kernel'; tw_params = 'flat'
    elif variant == 'dif_tw_log3':
        func_base = 'radix16_tw_log3_dif_kernel'; tw_params = 'log3'

    vname = {'notw': 'N1 (no twiddle)', 'dit_tw': 'DIT twiddled (flat)',
             'dif_tw': 'DIF twiddled (flat)',
             'dit_tw_log3': 'DIT twiddled (log3 derived)',
             'dif_tw_log3': 'DIF twiddled (log3 derived)'}[variant]
    guard = f"FFT_RADIX16_{isa.name.upper()}_{variant.upper()}_H"
    em.L.append(f"/**")
    em.L.append(f" * @file fft_radix16_{isa.name}_{variant}.h")
    em.L.append(f" * @brief DFT-16 {isa.name.upper()} {vname}")
    em.L.append(f" *")
    em.L.append(f" * 4x4 CT, k-step={isa.k_step}")
    em.L.append(f" * Generated by gen_radix16.py")
    em.L.append(f" */")
    em.L.append(f"")
    em.L.append(f"#ifndef {guard}")
    em.L.append(f"#define {guard}")
    em.L.append(f"")

    if isa.name != 'scalar':
        em.L.append(f"#include <immintrin.h>")
        em.L.append(f"")

    emit_twiddle_constants(em.L, itw_set)

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

        if isa.target:
            em.L.append(f"static {isa.target} void")
        else:
            em.L.append(f"static void")
        em.L.append(f"{func_base}_{d}_{isa.name}(")
        em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
        em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
        if tw_params == 'flat':
            em.L.append(f"    const double * __restrict__ tw_re, const double * __restrict__ tw_im,")
        elif tw_params == 'log3':
            em.L.append(f"    const double * __restrict__ tw_re, const double * __restrict__ tw_im,")
        em.L.append(f"    size_t K)")
        em.L.append(f"{{")
        em.ind = 1

        if isa.name != 'scalar':
            em.o(f"const {T} sign_flip = {isa.p}_set1_pd(-0.0);")
            em.o(f"const {T} sqrt2_inv = {isa.p}_set1_pd(0.70710678118654752440);")
        else:
            em.o(f"const double sqrt2_inv = 0.70710678118654752440;")
        em.b()

        spill_total = N * isa.sm
        if isa.name == 'scalar':
            em.o(f"double spill_re[{N}], spill_im[{N}];")
        else:
            em.o(f"{isa.align} double spill_re[{spill_total}];")
            em.o(f"{isa.align} double spill_im[{spill_total}];")
        em.b()

        em.o(f"{T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
        em.b()

        if isa.name != 'scalar' and itw_set:
            em.c(f"Hoisted internal W16 broadcasts [{d}]")
            set1 = f"{isa.p}_set1_pd"
            for (e, tN) in sorted(itw_set):
                label = wN_label(e, tN)
                em.o(f"const {T} tw_{label}_re = {set1}({label}_re);")
                em.o(f"const {T} tw_{label}_im = {set1}({label}_im);")
            em.b()

        if isa.name == 'scalar' and itw_set:
            em.c(f"Hoisted W16 [{d}]")
            for (e, tN) in sorted(itw_set):
                label = wN_label(e, tN)
                em.o(f"const double tw_{label}_re = {label}_re;")
                em.o(f"const double tw_{label}_im = {label}_im;")
            em.b()

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

    em.L.append(f"#undef LD")
    em.L.append(f"#undef ST")
    em.L.append(f"")
    em.L.append(f"#endif /* {guard} */")

    insert_stats_into_header(em.L, stats)
    return em.L, stats


# ═══════════════════════════════════════════════════════════════
# SV CODELET GENERATION
# ═══════════════════════════════════════════════════════════════

def _t2_to_sv(body):
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
    if isa.name == 'scalar':
        return []
    if variant.endswith('_log3'):
        return []

    text = '\n'.join(t2_lines)

    if variant == 'notw':
        t2_pattern = 'radix16_n1_dit_kernel'; sv_name = 'radix16_n1sv_kernel'
    elif variant == 'dit_tw':
        t2_pattern = 'radix16_tw_flat_dit_kernel'; sv_name = 'radix16_t1sv_dit_kernel'
    elif variant == 'dit_tw_scalar':
        t2_pattern = 'radix16_tw_flat_dit_kernel'; sv_name = 'radix16_t1sv_dit_kernel'
    elif variant == 'dif_tw':
        t2_pattern = 'radix16_tw_flat_dif_kernel'; sv_name = 'radix16_t1sv_dif_kernel'
    elif variant == 'dif_tw_scalar':
        t2_pattern = 'radix16_tw_flat_dif_kernel'; sv_name = 'radix16_t1sv_dif_kernel'
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
# CT CODELET EMITTER (n1 / t1 variants)
# ═══════════════════════════════════════════════════════════════

def emit_file_ct(isa, itw_set, ct_variant, tile=None, drain_mode='temporal'):
    """Emit FFTW-style n1 or t1_dit codelet.
    ct_variant: ct_n1, ct_n1_scaled, ct_t1_dit, ct_t1s_dit, ct_t1_dit_log3,
                ct_t1_dif, ct_t1_oop_dit,
                ct_t1_buf_dit (buffered flat),
                ct_t1_buf_dit_log3 (buffered log3).
    tile: tile size for buf variants (default 128 for AVX2, 64 for AVX-512).
    drain_mode: 'temporal' (normal stores, kept in cache) or 'stream'
                (non-temporal, bypasses cache, requires aligned destination).
                Ignored for non-buf variants.

    For buf variants, the generated function name encodes tile and drain_mode
    so the calibration harness can enumerate candidates:
      radix16_t1_buf_dit_tile{TILE}_{DRAIN}_{fwd|bwd}_{isa}
      radix16_t1_buf_dit_log3_tile{TILE}_{DRAIN}_{fwd|bwd}_{isa}
    """
    em = Emitter(isa)
    T = isa.T

    is_n1 = ct_variant == 'ct_n1'
    is_n1_scaled = ct_variant == 'ct_n1_scaled'
    is_t1_dit = ct_variant == 'ct_t1_dit'
    is_t1s_dit = ct_variant == 'ct_t1s_dit'
    is_t1_dit_log3 = ct_variant == 'ct_t1_dit_log3'
    is_t1_dif_log3 = ct_variant == 'ct_t1_dif_log3'
    is_t1_dit_log3_isub2 = ct_variant == 'ct_t1_dit_log3_isub2'
    is_t1_dit_log_half = ct_variant == 'ct_t1_dit_log_half'
    is_t1_dif = ct_variant == 'ct_t1_dif'
    is_t1_oop_dit = ct_variant == 'ct_t1_oop_dit'
    is_t1_buf_dit = ct_variant == 'ct_t1_buf_dit'
    is_t1_buf_dit_log3 = ct_variant == 'ct_t1_buf_dit_log3'
    is_buf = is_t1_buf_dit or is_t1_buf_dit_log3
    # Inner-body selector: log3 uses different kernel emitter
    buf_inner_is_log3 = is_t1_buf_dit_log3

    # log3_isub2 has an ISA requirement (register budget). Enforce early.
    if is_t1_dit_log3_isub2 and isa.name != 'avx512':
        raise ValueError(
            f"ct_t1_dit_log3_isub2 requires AVX-512 (current isa={isa.name}). "
            "Its scheduled working-data + twiddle peak (~30 ZMM) exceeds the "
            "16-YMM budget of AVX2 and the scalar emitter doesn't need it.")
    # log_half also has an ISA requirement — 8 bases + working data + derived
    # twiddles + radix-4 temps exceeds AVX2's 16 YMM.
    if is_t1_dit_log_half and isa.name != 'avx512':
        raise ValueError(
            f"ct_t1_dit_log_half requires AVX-512 (current isa={isa.name}). "
            "8 loaded bases occupy 16 ZMM before any working data or "
            "derivations, which already maxes out AVX2's YMM file.")

    # Default tile size (picked to match empirical sweet spot on Raptor Lake;
    # real platforms should bench-sweep)
    if tile is None:
        tile = 128 if isa.name == 'avx2' else 64
    if tile % isa.k_step != 0:
        raise ValueError(f"tile={tile} must be multiple of k_step={isa.k_step}")
    if drain_mode not in ('temporal', 'stream'):
        raise ValueError(f"drain_mode must be 'temporal' or 'stream'")
    if is_buf and isa.name == 'scalar' and drain_mode == 'stream':
        # Stream stores don't exist in scalar; silently demote
        drain_mode = 'temporal'

    # addr_mode:
    if is_n1 or is_n1_scaled:
        em.addr_mode = 'n1'
    elif is_t1_oop_dit:
        em.addr_mode = 't1_oop'
    elif is_buf:
        em.addr_mode = 't1_buf'
    else:
        em.addr_mode = 't1'

    # Function naming. Buf variants encode tile+drain in the name so the
    # calibration harness can discover all generated candidates by symbol name.
    if is_n1:
        func_base = "radix16_n1"; vname = "n1 (separate is/os)"
    elif is_n1_scaled:
        func_base = "radix16_n1_scaled"; vname = "n1_scaled"
    elif is_t1_oop_dit:
        func_base = "radix16_t1_oop_dit"; vname = "t1_oop DIT"
    elif is_t1_dif:
        func_base = "radix16_t1_dif"; vname = "t1 DIF"
    elif is_t1_dif_log3:
        func_base = "radix16_t1_dif_log3"; vname = "t1 DIF log3"
    elif is_t1s_dit:
        func_base = "radix16_t1s_dit"; vname = "t1s DIT (scalar broadcast twiddle)"
    elif is_t1_dit_log3:
        func_base = "radix16_t1_dit_log3"; vname = "t1 DIT log3"
    elif is_t1_dit_log3_isub2:
        func_base = "radix16_t1_dit_log3_isub2"
        vname = "t1 DIT log3 isub2 (paired sub-FFT scheduling, AVX-512)"
    elif is_t1_dit_log_half:
        func_base = "radix16_t1_dit_log_half"
        vname = "t1 DIT log_half (8 loaded bases, 7 depth-1 derivs, AVX-512)"
    elif is_t1_buf_dit:
        func_base = f"radix16_t1_buf_dit_tile{tile}_{drain_mode}"
        vname = f"t1 DIT buffered (TILE={tile}, drain={drain_mode})"
    elif is_t1_buf_dit_log3:
        func_base = f"radix16_t1_buf_dit_log3_tile{tile}_{drain_mode}"
        vname = f"t1 DIT log3 buffered (TILE={tile}, drain={drain_mode})"
    else:
        func_base = "radix16_t1_dit"; vname = "t1 DIT (in-place twiddle)"

    # Guard encodes full configuration so multiple buf variants with different
    # tile/drain can coexist in the same translation unit.
    if is_buf:
        guard = f"FFT_RADIX16_{isa.name.upper()}_{ct_variant.upper()}_TILE{tile}_{drain_mode.upper()}_H"
    else:
        guard = f"FFT_RADIX16_{isa.name.upper()}_CT_{ct_variant.upper()}_H"

    em.L.append(f"/**")
    em.L.append(f" * @file fft_radix16_{isa.name}_{ct_variant}.h")
    em.L.append(f" * @brief DFT-16 {isa.name.upper()} {vname}")
    em.L.append(f" *")
    em.L.append(f" * FFTW-style codelet for recursive CT executor.")
    em.L.append(f" * Generated by gen_radix16.py --variant {ct_variant}" +
                (f" --tile {tile} --drain {drain_mode}" if is_buf else ""))
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
        if is_n1 or is_n1_scaled:
            em.addr_mode = 'n1'
        elif is_t1_oop_dit:
            em.addr_mode = 't1_oop'
        elif is_buf:
            em.addr_mode = 't1_buf'
        else:
            em.addr_mode = 't1'
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
        else:  # t1, t1_buf
            em.L.append(f"{func_base}_{d}_{isa.name}(")
            em.L.append(f"    double * __restrict__ rio_re, double * __restrict__ rio_im,")
            em.L.append(f"    const double * __restrict__ W_re, const double * __restrict__ W_im,")
            if isa.name == 'scalar':
                em.L.append(f"    size_t ios, size_t mb, size_t me, size_t ms)")
            else:
                em.L.append(f"    size_t ios, size_t me)")

        em.L.append(f"{{")
        em.ind = 1

        # Constants — for DIT twiddle variants, defer sign_flip/sqrt2_inv/W16
        # to PASS 2. Exception: log3 variants use emit_kernel_body_log3 which does
        # not handle the deferred-emission protocol — they need constants up-front.
        # (The flat t1_buf variant still uses emit_kernel_body, so it can defer.)
        _defer_consts = (is_t1_dit or is_t1_oop_dit or is_t1s_dit or is_t1_buf_dit)
        if not _defer_consts:
            if isa.name != 'scalar':
                em.o(f"const {T} sign_flip = {isa.p}_set1_pd(-0.0);")
                em.o(f"const {T} sqrt2_inv = {isa.p}_set1_pd(0.70710678118654752440);")
            else:
                em.o(f"const double sqrt2_inv = 0.70710678118654752440;")
            em.b()

        # Spill buffer (same as baseline)
        spill_total = N * isa.sm
        if isa.name == 'scalar':
            em.o(f"double spill_re[{N}], spill_im[{N}];")
        else:
            em.o(f"{isa.align} double spill_re[{spill_total}];")
            em.o(f"{isa.align} double spill_im[{spill_total}];")
        em.b()

        # Tile output buffer for buf variants (flat or log3 inner body)
        if is_buf:
            em.c(f"Tile output buffer: {N} output streams × TILE={tile} k-positions")
            if isa.name != 'scalar':
                em.o(f"{isa.align} double outbuf_re[{N}*{tile}];")
                em.o(f"{isa.align} double outbuf_im[{N}*{tile}];")
            else:
                em.o(f"double outbuf_re[{N}*{tile}], outbuf_im[{N}*{tile}];")
            em.b()

        em.o(f"{T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
        em.b()

        if itw_set and not _defer_consts:
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

        em._defer_consts = _defer_consts
        em._itw_set = itw_set

        # Hoist twiddle broadcasts before the loop (t1s only)
        if is_t1s_dit:
            em.tw_hoisted = True
            em.emit_hoist_all_tw_scalars(N)
            em.b()

        if is_n1_scaled and isa.name != 'scalar':
            set1 = f"{isa.p}_set1_pd"
            em.o(f"const {T} vscale = {set1}(scale);")
            em.b()

        # Loop structure
        if is_buf:
            # Two-level loop: tile-outer + m-inner, plus drain, plus tail.
            # Inner body can be either flat dit_tw or log3 dit_tw (buf_inner_is_log3).
            em.c(f"Tile size — pass-2 stores go to outbuf, drained after each tile")
            em.o(f"#define TILE {tile}")
            em.o(f"const size_t n_full_tiles = me / TILE;")
            em.b()
            em.o(f"/* Full tiles */")
            em.o(f"for (size_t tile_base = 0; tile_base < n_full_tiles * TILE; tile_base += TILE) {{")
            em.ind += 1
            em.o(f"for (size_t kk = 0; kk < TILE; kk += {isa.k_step}) {{")
            em.ind += 1
            em.o(f"const size_t m = tile_base + kk;")
            em.b()
            if buf_inner_is_log3:
                # log3 body — shares the same emit_store routing via addr_mode='t1_buf'
                emit_kernel_body_log3(em, d, itw_set, 'dit_tw_log3')
            else:
                emit_kernel_body(em, d, itw_set, 'dit_tw')
            em.ind -= 1
            em.o("}")
            em.b()

            # Drain: 16 sequential streams, outbuf[m*TILE + kk] → rio[m*ios + tile_base + kk]
            em.c(f"Drain: {N} sequential streams, TILE doubles each (drain_mode={drain_mode})")
            em.o(f"for (size_t m_out = 0; m_out < {N}; m_out++) {{")
            em.ind += 1
            em.o(f"const double * __restrict__ src_re = &outbuf_re[m_out*TILE];")
            em.o(f"const double * __restrict__ src_im = &outbuf_im[m_out*TILE];")
            em.o(f"double * __restrict__ dst_re = &rio_re[m_out*ios + tile_base];")
            em.o(f"double * __restrict__ dst_im = &rio_im[m_out*ios + tile_base];")
            if isa.name == 'scalar':
                em.o(f"for (size_t j = 0; j < TILE; j++) {{")
                em.o(f"    dst_re[j] = src_re[j]; dst_im[j] = src_im[j];")
                em.o(f"}}")
            else:
                em.o(f"for (size_t j = 0; j < TILE; j += {isa.k_step}) {{")
                if drain_mode == 'stream':
                    # Stream (non-temporal) stores: bypass cache via write-combining.
                    # REQUIRES: rio_re[m_out*ios + tile_base + j] is k_step*8-byte aligned
                    # for all (m_out, tile_base, j) combinations. The calibration harness
                    # must ensure ios is a multiple of k_step (which is typical for FFTs).
                    em.o(f"    {isa.p}_stream_pd(&dst_re[j], LD(&src_re[j]));")
                    em.o(f"    {isa.p}_stream_pd(&dst_im[j], LD(&src_im[j]));")
                else:
                    em.o(f"    ST(&dst_re[j], LD(&src_re[j]));")
                    em.o(f"    ST(&dst_im[j], LD(&src_im[j]));")
                em.o(f"}}")
            em.ind -= 1
            em.o("}")
            em.ind -= 1
            em.o("}")
            # Stream stores need an sfence at the end to ensure visibility.
            # Without this, a downstream load could see stale data.
            if isa.name != 'scalar' and drain_mode == 'stream':
                em.o("_mm_sfence();")
            em.b()

            # Tail loop: me % TILE iterations use the standard stride-ios path.
            # For log3 inner body, we must switch addr_mode and emitter accordingly.
            em.c(f"Tail: me % TILE iterations, direct stride-ios stores")
            em.addr_mode = 't1'
            em.o(f"for (size_t m = n_full_tiles * TILE; m < me; m += {isa.k_step}) {{")
            em.ind += 1
            if buf_inner_is_log3:
                emit_kernel_body_log3(em, d, itw_set, 'dit_tw_log3')
            else:
                emit_kernel_body(em, d, itw_set, 'dit_tw')
            em.ind -= 1
            em.o("}")
            em.o("#undef TILE")
        elif is_n1 or is_n1_scaled:
            if isa.name == 'scalar':
                em.o(f"for (size_t k = 0; k < vl; k++) {{")
            else:
                em.o(f"for (size_t k = 0; k < vl; k += {isa.k_step}) {{")
            em.ind += 1
            if is_t1s_dit:
                emit_kernel_body(em, d, itw_set, 'dit_tw_scalar')
            elif is_t1_dit_log3:
                emit_kernel_body_log3(em, d, itw_set, 'dit_tw_log3')
            elif is_t1_dif_log3:
                raise ValueError(
                    "ct_t1_dif_log3 is a t1 variant; cannot emit in n1 loop")
            elif is_t1_dit_log3_isub2:
                # Shouldn't reach here — log3_isub2 is a t1 variant, not n1.
                # But if someone wires it wrong, this fails loudly at codegen
                # instead of silently falling through to plain dit_tw.
                raise ValueError(
                    "ct_t1_dit_log3_isub2 is a t1 variant; cannot emit in n1 loop")
            elif is_t1_dit_log_half:
                raise ValueError(
                    "ct_t1_dit_log_half is a t1 variant; cannot emit in n1 loop")
            else:
                kernel_variant = 'notw' if (is_n1 or is_n1_scaled) else ('dif_tw' if is_t1_dif else 'dit_tw')
                if is_t1_oop_dit:
                    em.addr_mode = 't1_oop'
                    kernel_variant = 'dit_tw'
                emit_kernel_body(em, d, itw_set, kernel_variant)
            em.ind -= 1
            em.o("}")
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
            elif is_t1_dif_log3:
                emit_kernel_body_log3(em, d, itw_set, 'dif_tw_log3')
            elif is_t1_dit_log3_isub2:
                emit_kernel_body_log3_isub2(em, d, itw_set, 'dit_tw_log3')
            elif is_t1_dit_log_half:
                emit_kernel_body_log_half(em, d, itw_set, 'dit_tw_log_half')
            else:
                kernel_variant = 'dif_tw' if is_t1_dif else 'dit_tw'
                if is_t1_oop_dit:
                    em.addr_mode = 't1_oop'
                    kernel_variant = 'dit_tw'
                emit_kernel_body(em, d, itw_set, kernel_variant)
            em.ind -= 1
            em.o("}")

        em.L.append("}")
        em.L.append("")

    em.L.append(f"#undef LD")
    em.L.append(f"#undef ST")
    em.L.append(f"")
    em.L.append(f"#endif /* {guard} */")

    return em.L


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Unified R=16 codelet generator')
    parser.add_argument('--isa', default='avx512',
                        choices=['scalar', 'avx2', 'avx512', 'all'])
    parser.add_argument('--variant', default='dit_tw',
                        choices=['notw', 'dit_tw', 'dif_tw', 'dit_tw_log3', 'dif_tw_log3',
                                 'ct_n1', 'ct_n1_scaled', 'ct_t1_dit', 'ct_t1s_dit',
                                 'ct_t1_dit_log3', 'ct_t1_dit_log3_isub2',
                                 'ct_t1_dit_log_half',
                                 'ct_t1_dif', 'ct_t1_dif_log3', 'ct_t1_oop_dit',
                                 'ct_t1_buf_dit',
                                 'all'])
    parser.add_argument('--tile', type=int, default=None,
                        help='Tile size for ct_t1_buf_dit{,_log3} (default: 128/AVX2, 64/AVX-512)')
    parser.add_argument('--drain', default='temporal',
                        choices=['temporal', 'stream'],
                        help="Drain store mode for ct_t1_buf_dit{,_log3}: "
                             "'temporal' (regular stores, output stays L1-hot) or "
                             "'stream' (non-temporal, bypasses cache)")
    parser.add_argument('--enumerate-buf-candidates', action='store_true',
                        help='Emit all (variant, tile, drain) combinations for buf variants, '
                             'one header per stdout run separated by marker lines. Intended '
                             'for calibration harness consumption.')
    args = parser.parse_args()

    itw_set = collect_internal_twiddles()

    if args.isa == 'all':
        targets = [ISA_SCALAR, ISA_AVX2, ISA_AVX512]
    else:
        targets = [ALL_ISA[args.isa]]

    if args.variant == 'all':
        variants = ['notw', 'dit_tw', 'dif_tw', 'dit_tw_log3', 'dif_tw_log3']
    else:
        variants = [args.variant]

    # --enumerate-buf-candidates: emit a matrix of buf variants for the harness
    if args.enumerate_buf_candidates:
        # Candidate matrix. Tile sizes chosen to keep outbuf in L1d:
        #   64  — outbuf 16 KiB; best L1 headroom alongside in-flight data
        #   128 — outbuf 32 KiB; fits 48 KiB L1d on Raptor Lake / Golden Cove
        # Drain mode: temporal only.
        #
        # Pruned (no measured win on any host, both Raptor Lake and Emerald Rapids):
        #   tile256 — outbuf 64 KiB exceeds 48 KiB L1d, evicts in-flight data;
        #             observed 44–58% slower than tile128 at every me ≥ 256 cell
        #             where it was applicable.
        #   _stream  — non-temporal drain bets against output reuse, which is
        #             wrong for a recursive FFT (the next stage reads it back
        #             immediately); observed 3–4× slower than temporal at every
        #             measured cell, cross-radix.
        # See _BUF_TILES / _BUF_DRAINS below for the canonical matrix.
        #
        # NOTE: ct_t1_buf_dit_log3 is not enumerated here — the underlying log3
        # kernel has a latent correctness bug at m>0 that must be resolved first.
        # The code path is still present in emit_file_ct for future use.
        buf_variants = ['ct_t1_buf_dit']
        tile_sizes = list(_BUF_TILES)
        drain_modes = list(_BUF_DRAINS)
        for isa in targets:
            for variant in buf_variants:
                for t in tile_sizes:
                    if t % isa.k_step != 0:
                        continue  # skip invalid (scalar handles any tile)
                    for dm in drain_modes:
                        if isa.name == 'scalar' and dm == 'stream':
                            continue  # no stream stores in scalar
                        tag = f"{isa.name}__{variant}__tile{t}__{dm}"
                        print(f"/* === BEGIN CANDIDATE {tag} === */")
                        lines = emit_file_ct(isa, itw_set, variant, tile=t, drain_mode=dm)
                        print("\n".join(lines))
                        print(f"/* === END CANDIDATE {tag} === */\n")
        return

    for isa in targets:
        for variant in variants:
            if variant.startswith('ct_'):
                lines = emit_file_ct(isa, itw_set, variant,
                                     tile=args.tile, drain_mode=args.drain)
                print("\n".join(lines))
            else:
                lines, stats = emit_file(isa, itw_set, variant)
                sv_lines = emit_sv_variants(lines, isa, variant)
                if sv_lines:
                    for i in range(len(lines)):
                        if lines[i].strip() == '#undef LD':
                            lines[i:i] = sv_lines
                            break
                print("\n".join(lines))

                print(f"\n=== {isa.name.upper()} {variant} ===", file=sys.stderr)
                for k in sorted(stats.keys()):
                    s = stats[k]
                    print(f"  {k}: {s['spill']}sp+{s['reload']}rl={s['spill']+s['reload']}"
                          f" arith={s['total_arith']} flops={s['flops']}"
                          f" mem={s['total_mem']}", file=sys.stderr)


# ═══════════════════════════════════════════════════════════════
# Variant metadata for bench / dispatcher infrastructure.
#
# Phase A matrix + Phase B buf matrix. Each buf variant has a unique
# function name (radix16_t1_buf_dit_tile{T}_{drain}_{fwd|bwd}_{isa}) so
# all coexist in one translation unit. All buf variants share dispatcher
# 't1_buf_dit' — the dispatcher picks the fastest (tile, drain) per
# (me, ios) based on the bench.
#
# Total: 4 base (Phase A) + 2 buf (Phase B) = 6 variants per ISA.
# ═══════════════════════════════════════════════════════════════

# Phase A (non-parameterized variants)
# Tuple layout: (func_base, protocol, dispatcher, supported_isas)
#   supported_isas: None  → all ISAs OK (scalar/avx2/avx512)
#                   set() → explicit restriction, e.g. {'avx512'} for AVX-512-only
VARIANTS = {
    'ct_t1_dit':               ('radix16_t1_dit',              'flat', 't1_dit',              None),
    'ct_t1_dif':               ('radix16_t1_dif',              'flat', 't1_dif',              None),
    'ct_t1s_dit':              ('radix16_t1s_dit',             't1s',  't1s_dit',             None),
    'ct_t1_dit_log3':          ('radix16_t1_dit_log3',         'log3', 't1_dit_log3',         None),
    # DIF-log3: mirror of DIT-log3 with twiddles applied in PASS 2 after the
    # column butterfly, rather than in PASS 1 before the sub-FFT butterfly.
    # Same 11 derivations, same live-set, same log3 protocol — just at the
    # opposite end of the codelet body. Competes in its own dispatcher
    # 't1_dif_log3' (distinct from the DIT-log3 family, since DIT and DIF
    # are called by different planner paths).
    'ct_t1_dif_log3':          ('radix16_t1_dif_log3',         'log3', 't1_dif_log3',         None),
    # log3_isub2: same log3 protocol, paired sub-FFT scheduling. AVX-512 only
    # because peak register pressure (~30 ZMM) exceeds AVX2's 16 YMM budget.
    # Shares dispatcher 't1_dit_log3' — the planner picks isub2 vs plain log3
    # per (me, ios) from bench results, same-protocol competition.
    'ct_t1_dit_log3_isub2':    ('radix16_t1_dit_log3_isub2',   'log3', 't1_dit_log3',         {'avx512'}),
    # log_half: 8 loaded bases + 7 depth-1 derivations. Reuses the same
    # sparse-log3 buffer allocation (harness.c allocates (R-1)*me for log3
    # already — see sparse-flat convention there). Targets me=64 AVX-512
    # cells where plain log3 is beaten by flat due to derivation-chain
    # latency. AVX-512 only: 16 ZMM for 8 bases alone exceeds AVX2.
    'ct_t1_dit_log_half':      ('radix16_t1_dit_log_half',     'log3', 't1_dit_log3',         {'avx512'}),
}

# Phase B buf variants: 2 tiles × 1 drain = 2 parameterized candidates.
# - tile sizes: 64 (outbuf 16 KiB), 128 (outbuf 32 KiB).
#   Both fit in 48 KiB L1d (Raptor Lake P-core, Golden Cove server).
# - drain modes: temporal only (regular stores; output stays L1-hot for the
#   next stage to read immediately).
#
# Pruned variants (R=16, both hosts measured to date):
#   tile256_*  — outbuf 64 KiB > 48 KiB L1d; 44–58% slower than tile128 at
#                every me ≥ 256 cell on Raptor Lake. The L1-resident outbuf
#                assumption breaks above 32 KiB on current Intel client/server.
#   *_stream   — non-temporal drain assumes output won't be reused; in a
#                recursive FFT the next stage reads it immediately, so the
#                cache-bypass hint is actively wrong. 3–4× slower than
#                temporal at every measured cell, cross-radix.
_BUF_TILES = [64, 128]
_BUF_DRAINS = ['temporal']
for _t in _BUF_TILES:
    for _d in _BUF_DRAINS:
        _vid = f'ct_t1_buf_dit_tile{_t}_{_d}'
        _fname = f'radix16_t1_buf_dit_tile{_t}_{_d}'
        VARIANTS[_vid] = (_fname, 'flat', 't1_buf_dit', None)

# Helper to decompose a buf variant ID back into (tile, drain)
def _parse_buf_variant(variant_id):
    """Given 'ct_t1_buf_dit_tile128_stream' -> (128, 'stream')."""
    assert variant_id.startswith('ct_t1_buf_dit_tile'), variant_id
    rest = variant_id[len('ct_t1_buf_dit_tile'):]
    tile_str, drain = rest.split('_', 1)
    return int(tile_str), drain

def is_buf_variant(variant_id):
    return variant_id.startswith('ct_t1_buf_dit_tile')


def function_name(variant_id, isa, direction):
    if variant_id not in VARIANTS:
        raise KeyError(f"unknown variant {variant_id}; known: {list(VARIANTS)}")
    base = VARIANTS[variant_id][0]
    return f'{base}_{direction}_{isa}'

def protocol(variant_id):
    return VARIANTS[variant_id][1]

def dispatcher(variant_id):
    return VARIANTS[variant_id][2]

def supported_isas(variant_id):
    """Return the set of ISA names this variant is designed for, or None if
    unrestricted. Used by candidates.py for enumeration-time filtering, so
    we never bench or emit a variant on an ISA it isn't designed for.

    Orthogonal to runtime AVX-512 host-capability detection: this answers
    'should this symbol even exist for this ISA?', while the runtime check
    answers 'does this machine support executing it?'"""
    return VARIANTS[variant_id][3]


def _emit_all_variants(isa_name):
    """Emit a single header containing all Phase-A + Phase-B variants for the
    given ISA. The bench's generate phase invokes this to get
    `fft_radix16_{isa}.h` in staging/.

    Phase A variants: emitted by emit_file_ct with tile=None.
    Phase B buf variants: emitted by emit_file_ct with each (tile, drain)
    combination; the generator encodes tile+drain in the function name
    so all six coexist without symbol collisions.
    """
    itw_set = collect_internal_twiddles()
    isa = ALL_ISA[isa_name]

    guard = f'FFT_RADIX16_{isa_name.upper()}_H'
    out = []
    out.append(f'/**')
    out.append(f' * @file fft_radix16_{isa_name}.h')
    out.append(f' * @brief DFT-16 {isa_name.upper()} codelets (Phase A + B)')
    out.append(f' *')
    out.append(f' * Auto-generated by gen_radix16.py.')
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
            # skip duplicated headers/guards/includes emitted by emit_file_ct
            if s.startswith('#ifndef FFT_RADIX16_'): continue
            if s.startswith('#define FFT_RADIX16_'): continue
            if s.startswith('#endif /* FFT_RADIX16_'): continue
            if s.startswith('#include <immintrin.h>'): continue
            if s.startswith('#include <stddef.h>'): continue
            out.append(L)
        out.append('')

    # Phase A variants (non-parameterized)
    for v in ['ct_t1_dit', 'ct_t1_dif', 'ct_t1s_dit',
              'ct_t1_dit_log3', 'ct_t1_dif_log3']:
        lines = emit_file_ct(isa, itw_set, v, tile=None, drain_mode='temporal')
        _append_cleaned(lines)

    # ISA-restricted variants: query supported_isas() to skip incompatible ISAs.
    # Source of truth for which ISAs a variant is designed for is VARIANTS[v][3].
    for v in ['ct_t1_dit_log3_isub2', 'ct_t1_dit_log_half']:
        allowed = supported_isas(v)
        if allowed is None or isa_name in allowed:
            lines = emit_file_ct(isa, itw_set, v, tile=None, drain_mode='temporal')
            _append_cleaned(lines)

    # Phase B buf variants (2 tiles × 1 drain = 2 combos per ISA)
    for t in _BUF_TILES:
        for d in _BUF_DRAINS:
            lines = emit_file_ct(isa, itw_set, 'ct_t1_buf_dit',
                                 tile=t, drain_mode=d)
            _append_cleaned(lines)

    out.append(f'#endif /* {guard} */')
    return '\n'.join(out)


if __name__ == '__main__':
    import argparse

    # If invoked bench-style with --isa only and no --variant, emit the
    # combined Phase-A header. Otherwise defer to the legacy main() for
    # backwards compatibility with standalone R=16 tooling.
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--isa', choices=list(ALL_ISA))
    ap.add_argument('--variant', default=None)
    ap.add_argument('--list-variants', action='store_true')
    ap.add_argument('--list-name', action='store_true')
    ap.add_argument('--list-protocol', action='store_true')
    ap.add_argument('--direction', choices=['fwd', 'bwd'], default='fwd')
    args, unknown = ap.parse_known_args()

    bench_mode = (
        args.list_variants or args.list_name or args.list_protocol or
        (args.isa is not None and args.variant is None and not unknown)
    )

    if args.list_variants:
        for v in VARIANTS:
            print(v)
        sys.exit(0)
    if args.list_protocol:
        if not args.variant:
            print('--list-protocol requires --variant', file=sys.stderr)
            sys.exit(2)
        print(protocol(args.variant))
        sys.exit(0)
    if args.list_name:
        if not args.variant or not args.isa:
            print('--list-name requires --variant and --isa', file=sys.stderr)
            sys.exit(2)
        print(function_name(args.variant, args.isa, args.direction))
        sys.exit(0)

    if bench_mode:
        print(_emit_all_variants(args.isa))
        sys.exit(0)

    # Fallback to the legacy main() for standalone/debug use.
    main()
