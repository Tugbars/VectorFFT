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
gen_radix6.py -- Unified DFT-6 codelet generator for VectorFFT

3x2 Cooley-Tukey: pass 1 = 2x DFT-3, pass 2 = 3x DFT-2 columns.
Internal W6 twiddles: 2 unique (W6^1, W6^2), both require cmul.
6 spill + 6 reload per kernel (fused: 3 spill + 3 reload saved).

DFT-3 butterfly (2 constants, FMA-fused S terms):
  KHALF=0.5, KS=sqrt(3)/2.  Same as gen_radix3.py/gen_radix12.py.

DFT-2 butterfly (zero constants): y0=x0+x1, y1=x0-x1.

Register pressure per pass:
  Pass 1: DFT-3 on 3 inputs -> peak ~10 YMM
  Pass 2: DFT-2 on 2 inputs -> 4 YMM (trivial)
  Spill: 6 complex = 48 bytes

AVX-512 single-pass: 12 data + 2 const + sign_flip + 2 tw = 17/32 ZMM.

n1_ovs: 6 bins = 1 group of 4 + 2 scatter (bins 4,5).

Usage:
  python3 gen_radix6.py --isa avx2 --variant all
  python3 gen_radix6.py --isa all  --variant ct_n1
"""

import sys, math, argparse, re

R = 6
N, N1, N2 = 6, 3, 2

KHALF_val = 0.500000000000000000000000000000000000000000000
KS_val    = 0.866025403784438646763723170752936183471402627


def wN(e, tN):
    e = e % tN; a = 2.0 * math.pi * e / tN
    return (math.cos(a), -math.sin(a))

def wN_label(e, tN):
    return f"W{tN}_{e % tN}"

def collect_internal_twiddles():
    """W6: k1=1..2, n2=1 -> exponents {1,2}. Both require cmul."""
    tw = set()
    for k1 in range(1, N1):
        for n2 in range(1, N2):
            e = (k1 * n2) % N
            if e != 0:
                tw.add((e, N))
    return tw


class ISAConfig:
    def __init__(self, name, T, width, k_step, p, sm, target, align, ld_prefix):
        self.name = name; self.T = T; self.width = width
        self.k_step = k_step; self.p = p; self.sm = sm
        self.target = target; self.align = align; self.ld_prefix = ld_prefix

ISA_SCALAR = ISAConfig('scalar', 'double', 1, 1, '', 1, '', '', 'R6S')
ISA_AVX2   = ISAConfig('avx2', '__m256d', 4, 4, '_mm256', 4,
    '__attribute__((target("avx2,fma")))',
    '__attribute__((aligned(32)))', 'R6A')
ISA_AVX512 = ISAConfig('avx512', '__m512d', 8, 8, '_mm512', 8,
    '__attribute__((target("avx512f,avx512dq,fma")))',
    '__attribute__((aligned(64)))', 'R6L')

ALL_ISA = {'scalar': ISA_SCALAR, 'avx2': ISA_AVX2, 'avx512': ISA_AVX512}


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

    def emit_int_tw(self, v, e, d):
        e = e % N
        if e == 0: return
        if e == 3:  # W6^3 = -1
            self.o(f"{v}_re = {self.neg(f'{v}_re')}; {v}_im = {self.neg(f'{v}_im')};")
            return
        label = wN_label(e, N)
        self.emit_cmul_inplace(v, f"tw_{label}_re", f"tw_{label}_im", d)

    def emit_radix3(self, v, d, label=""):
        fwd = (d == 'fwd'); T = self.isa.T
        if label: self.c(f"{label} [{d}]")
        x0, x1, x2 = v[0], v[1], v[2]
        self.o(f"{{")
        self.o(f"{T} sr={self.add(f'{x1}_re',f'{x2}_re')}, si={self.add(f'{x1}_im',f'{x2}_im')};")
        self.o(f"{T} dr={self.sub(f'{x1}_re',f'{x2}_re')}, di={self.sub(f'{x1}_im',f'{x2}_im')};")
        self.o(f"{T} Rr={self.fnma('KHALF','sr',f'{x0}_re')}, Ri={self.fnma('KHALF','si',f'{x0}_im')};")
        self.o(f"{x0}_re={self.add(f'{x0}_re','sr')}; {x0}_im={self.add(f'{x0}_im','si')};")
        if fwd:
            self.o(f"{x1}_re={self.fma('KS','di','Rr')}; {x1}_im={self.fnma('KS','dr','Ri')};")
            self.o(f"{x2}_re={self.fnma('KS','di','Rr')}; {x2}_im={self.fma('KS','dr','Ri')};")
        else:
            self.o(f"{x1}_re={self.fnma('KS','di','Rr')}; {x1}_im={self.fma('KS','dr','Ri')};")
            self.o(f"{x2}_re={self.fma('KS','di','Rr')}; {x2}_im={self.fnma('KS','dr','Ri')};")
        self.o(f"}}")

    def emit_radix2(self, v, d, label=""):
        if label: self.c(f"{label} [{d}]")
        a, b = v[0], v[1]; T = self.isa.T
        self.o(f"{{ {T} tr={self.sub(f'{a}_re',f'{b}_re')}, ti={self.sub(f'{a}_im',f'{b}_im')};")
        self.o(f"  {a}_re={self.add(f'{a}_re',f'{b}_re')}; {a}_im={self.add(f'{a}_im',f'{b}_im')};")
        self.o(f"  {b}_re=tr; {b}_im=ti; }}")


# ═══════════════════════════════════════════════════════════════
# KERNEL BODY
# ═══════════════════════════════════════════════════════════════

def emit_kernel_body_fused(em, d, itw_set, variant):
    """Fused 3x2 CT: don't spill last DFT-3 group, save 3 mem ops."""
    xv3 = [f"x{i}" for i in range(N1)]
    xv2 = [f"x{i}" for i in range(N2)]
    n2_last = N2 - 1  # = 1

    # Pass 1: first N2-1=1 group with spill
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
        em.emit_radix3(xv3, d, f"radix-3 n2={n2}")
        em.b()
        for k1 in range(N1):
            em.emit_spill(f"x{k1}", n2 * N1 + k1)
        em.b()

    # Last group: DFT-3, keep in x0..x2 (NO spill)
    em.c(f"sub-FFT n2={n2_last} (fused: keep in x0..x2, no spill)")
    for n1 in range(N1):
        n = N2 * n1 + n2_last
        em.emit_load(f"x{n1}", n)
        if variant == 'dit_tw' and n > 0:
            em.emit_ext_tw(f"x{n1}", n - 1, d)
        elif variant == 'dit_tw_scalar' and n > 0:
            em.emit_ext_tw_scalar(f"x{n1}", n - 1, d)
    em.b()
    em.emit_radix3(xv3, d, f"radix-3 n2={n2_last}")
    em.b()

    # Pass 2: fused columns (1 reload + 1 live per DFT-2)
    em.c("PASS 2: fused radix-2 columns (1 reload + 1 live)")
    em.b()
    for k1 in range(N1):
        em.c(f"column k1={k1}")
        # Reload from first group
        for n2 in range(N2 - 1):
            em.emit_reload(f"t{n2}", n2 * N1 + k1)
        em.b()
        if k1 > 0:
            for n2 in range(1, N2 - 1):
                e = (k1 * n2) % N
                em.emit_int_tw(f"t{n2}", e, d)
            e_last = (k1 * n2_last) % N
            em.emit_int_tw(f"x{k1}", e_last, d)
            em.b()
        dft2_vars = [f"t0", f"x{k1}"]
        em.emit_radix2(dft2_vars, d, f"radix-2 k1={k1}")
        em.b()
        if variant == 'dif_tw':
            for k2 in range(N2):
                m = k1 + N1 * k2
                if m > 0:
                    em.emit_ext_tw(dft2_vars[k2], m - 1, d)
            em.b()
        elif variant == 'dif_tw_scalar':
            for k2 in range(N2):
                m = k1 + N1 * k2
                if m > 0:
                    em.emit_ext_tw_scalar(dft2_vars[k2], m - 1, d)
            em.b()
        for k2 in range(N2):
            m = k1 + N1 * k2
            em.emit_store(dft2_vars[k2], m)
        em.b()


def emit_kernel_body(em, d, itw_set, variant):
    """Standard 3x2 CT (for log3, U=2, CT variants)."""
    xv3 = [f"x{i}" for i in range(N1)]
    xv2 = [f"x{i}" for i in range(N2)]
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
        em.emit_radix3(xv3, d, f"radix-3 n2={n2}")
        em.b()
        for k1 in range(N1):
            em.emit_spill(f"x{k1}", n2 * N1 + k1)
        em.b()
    em.c("PASS 2: 3x radix-2 columns")
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
        em.emit_radix2(xv2, d, f"radix-2 k1={k1}")
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
    is_dit = variant == 'dit_tw_log3'
    T = em.isa.T
    xv3 = [f"x{i}" for i in range(N1)]
    xv2 = [f"x{i}" for i in range(N2)]
    em.c("Log3: load W^1, derive W^2..W^5 (4 cmuls, depth-2)")
    tb, tbi = em._tw_buf(), em._tw_buf_im()
    ta = em._tw_addr(0)
    if em.isa.name == 'scalar':
        em.o(f"const double w1r = {tb}[{ta}], w1i = {tbi}[{ta}];")
    else:
        ld = f"{em.isa.p}_load_pd"
        em.o(f"const {T} w1r = {ld}(&{tb}[{ta}]), w1i = {ld}(&{tbi}[{ta}]);")
    em.n_load += 2
    derivations = [
        ('w2', 'w1', 'w1'),
        ('w3', 'w1', 'w2'),
        ('w4', 'w2', 'w2'),
        ('w5', 'w1', 'w4'),
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
        em.emit_radix3(xv3, d, f"radix-3 n2={n2}")
        em.b()
        for k1 in range(N1):
            em.emit_spill(f"x{k1}", n2 * N1 + k1)
        em.b()
    em.c("PASS 2: 3x radix-2 columns")
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

def emit_hoisted_w6(L_or_em, isa, itw_set, indent=1, use_em=False):
    nontrivial = sorted([(e, tN) for (e, tN) in itw_set if e % 6 not in (0, 3)])
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
    sep20 = '-'*20; s5='-'*5; s3='-'*3; s4='-'*4
    table.append(f" *   {'kernel':<20s} {'add':>5s} {'sub':>5s} {'mul':>5s} {'neg':>5s}"
                 f" {'fma':>5s} {'fms':>5s} | {'arith':>5s} {'flops':>5s}"
                 f" | {'ld':>3s} {'st':>3s} {'sp':>3s} {'rl':>3s} {'mem':>4s}")
    table.append(f" *   {sep20} {s5} {s5} {s5} {s5} {s5} {s5} + {s5} {s5} + {s3} {s3} {s3} {s3} {s4}")
    for k in sorted(stats.keys()):
        s = stats[k]; sp=s.get('spill',0); rl=s.get('reload',0)
        table.append(f" *   {k:<20s} {s['add']:5d} {s['sub']:5d} {s['mul']:5d} {s['neg']:5d}"
                     f" {s['fma']:5d} {s['fms']:5d} | {s['total_arith']:5d} {s['flops']:5d}"
                     f" | {s['load']:3d} {s['store']:3d} {sp:3d} {rl:3d} {s['total_mem']:4d}")
    table.append(" *")
    for i, line in enumerate(lines):
        if line.strip() == '*/':
            for j, tl in enumerate(table): lines.insert(i+j, tl)
            return


# ═══════════════════════════════════════════════════════════════
# FILE EMITTERS
# ═══════════════════════════════════════════════════════════════

def emit_file(isa, itw_set, variant):
    em = Emitter(isa); T = isa.T
    is_log3 = variant.endswith('_log3')
    vmap = {
        'notw':        ('radix6_n1_dit_kernel',       None,   'N1 (no twiddle)'),
        'dit_tw':      ('radix6_tw_flat_dit_kernel',  'flat', 'DIT twiddled (flat)'),
        'dif_tw':      ('radix6_tw_flat_dif_kernel',  'flat', 'DIF twiddled (flat)'),
        'dit_tw_log3': ('radix6_tw_log3_dit_kernel',  'flat', 'DIT twiddled (log3)'),
        'dif_tw_log3': ('radix6_tw_log3_dif_kernel',  'flat', 'DIF twiddled (log3)'),
    }
    func_base, tw_params, vname = vmap[variant]
    guard = f"FFT_RADIX6_{isa.name.upper()}_{variant.upper()}_H"
    em.L.append(f"/**"); em.L.append(f" * @file fft_radix6_{isa.name}_{variant}.h")
    em.L.append(f" * @brief DFT-6 {isa.name.upper()} {vname}")
    em.L.append(f" *"); em.L.append(f" * 3x2 CT, k-step={isa.k_step}")
    em.L.append(f" * Generated by gen_radix6.py"); em.L.append(f" */")
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

    use_fused = not is_log3
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
        if use_fused:
            spill_count = (N2 - 1) * N1  # 1*3 = 3
            spill_total = spill_count * isa.sm
            if isa.name == 'scalar': em.o(f"double spill_re[{spill_count}], spill_im[{spill_count}];")
            else:
                em.o(f"{isa.align} double spill_re[{spill_total}];")
                em.o(f"{isa.align} double spill_im[{spill_total}];")
            em.b()
            em.o(f"{T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im;")
            em.o(f"{T} t0_re,t0_im;")
        else:
            spill_total = N * isa.sm
            if isa.name == 'scalar': em.o(f"double spill_re[{N}], spill_im[{N}];")
            else:
                em.o(f"{isa.align} double spill_re[{spill_total}];")
                em.o(f"{isa.align} double spill_im[{spill_total}];")
            em.b()
            em.o(f"{T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im;")
        em.b()
        emit_hoisted_w6(em, isa, itw_set, use_em=True)
        if isa.name == 'scalar': em.o(f"for (size_t k = 0; k < K; k++) {{")
        else:                    em.o(f"for (size_t k = 0; k < K; k += {isa.k_step}) {{")
        em.ind += 1
        if is_log3:
            emit_kernel_body_log3(em, d, itw_set, variant)
        elif use_fused:
            emit_kernel_body_fused(em, d, itw_set, variant)
        else:
            emit_kernel_body(em, d, itw_set, variant)
        em.ind -= 1
        em.o("}"); em.L.append("}"); em.L.append("")
        stats[d] = em.get_stats()

    # U=2 pipelining (AVX-512 only, non-log3)
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
            emit_hoisted_w6(em.L, isa, itw_set, indent=1)
            xdecl = f"        {T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im;"
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
        t2_pattern = 'radix6_n1_dit_kernel'; sv_name = 'radix6_n1sv_kernel'
    elif variant == 'dit_tw':
        t2_pattern = 'radix6_tw_flat_dit_kernel'; sv_name = 'radix6_t1sv_dit_kernel'
    elif variant == 'dit_tw_scalar':
        t2_pattern = 'radix6_tw_flat_dit_kernel'; sv_name = 'radix6_t1sv_dit_kernel'
    elif variant == 'dif_tw':
        t2_pattern = 'radix6_tw_flat_dif_kernel'; sv_name = 'radix6_t1sv_dif_kernel'
    elif variant == 'dif_tw_scalar':
        t2_pattern = 'radix6_tw_flat_dif_kernel'; sv_name = 'radix6_t1sv_dif_kernel'
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


def emit_file_ct(isa, itw_set, ct_variant):
    em = Emitter(isa); T = isa.T
    is_n1 = ct_variant == 'ct_n1'
    is_t1_dit = ct_variant == 'ct_t1_dit'
    is_t1s_dit = ct_variant == 'ct_t1s_dit'
    is_t1_dit_log3 = ct_variant == 'ct_t1_dit_log3'
    is_t1_dif = ct_variant == 'ct_t1_dif'
    em.addr_mode = 'n1' if is_n1 else 't1'
    if is_n1:           func_base = "radix6_n1"; vname = "n1 (separate is/os)"
    elif is_t1s_dit:    func_base = "radix6_t1s_dit"; vname = "t1s DIT (in-place, scalar broadcast twiddle)"
    elif is_t1_dif:     func_base = "radix6_t1_dif"; vname = "t1 DIF"
    elif is_t1_dit_log3: func_base = "radix6_t1_dit_log3"; vname = "t1 DIT log3 (in-place, derived twiddles)"
    else:               func_base = "radix6_t1_dit"; vname = "t1 DIT"
    guard = f"FFT_RADIX6_{isa.name.upper()}_CT_{ct_variant.upper()}_H"
    em.L.append(f"/**"); em.L.append(f" * @file fft_radix6_{isa.name}_{ct_variant}.h")
    em.L.append(f" * @brief DFT-6 {isa.name.upper()} {vname}")
    em.L.append(f" *"); em.L.append(f" * Generated by gen_radix6.py"); em.L.append(f" */")
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
        emit_constants(em)
        spill_total = N * isa.sm
        if isa.name == 'scalar': em.o(f"double spill_re[{N}], spill_im[{N}];")
        else:
            em.o(f"{isa.align} double spill_re[{spill_total}];")
            em.o(f"{isa.align} double spill_im[{spill_total}];")
        em.b()
        em.o(f"{T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im;")
        em.b()
        emit_hoisted_w6(em, isa, itw_set, use_em=True)

        # Hoist twiddle broadcasts before the loop (t1s only)
        if is_t1s_dit:
            em.tw_hoisted = True
            em.emit_hoist_all_tw_scalars(R)
            em.b()

        if is_n1:
            if isa.name == 'scalar': em.o(f"for (size_t k = 0; k < vl; k++) {{")
            else:                    em.o(f"for (size_t k = 0; k < vl; k += {isa.k_step}) {{")
        else:
            if isa.name == 'scalar': em.o(f"for (size_t m = mb; m < me; m++) {{")
            else:                    em.o(f"for (size_t m = 0; m < me; m += {isa.k_step}) {{")
        em.ind += 1
        if is_t1s_dit:
            emit_kernel_body(em, d, itw_set, 'dit_tw_scalar')
        elif is_t1_dit_log3:
            emit_kernel_body_log3(em, d, itw_set, 'dit_tw_log3')
        else:
            kernel_variant = 'notw' if is_n1 else ('dif_tw' if is_t1_dif else 'dit_tw')
            emit_kernel_body(em, d, itw_set, kernel_variant)
        em.ind -= 1
        em.o("}"); em.L.append("}"); em.L.append("")

    # n1_ovs: 6 bins = 1 group of 4 (bins 0-3) + 2 scatter (bins 4,5)
    if is_n1 and isa.name != 'scalar':
        VL = isa.k_step
        for d in ['fwd', 'bwd']:
            em.L.append(f"")
            if isa.target: em.L.append(f"static {isa.target} void")
            else:          em.L.append(f"static void")
            em.L.append(f"radix6_n1_ovs_{d}_{isa.name}(")
            em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
            em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
            em.L.append(f"    size_t is, size_t os, size_t vl, size_t ovs)")
            em.L.append(f"{{")
            em.L.append(f"    /* n1_ovs: butterfly -> tbuf, 1x 4x4 transpose (bins 0-3) + scatter (bins 4,5) */")
            em.L.append(f"    {isa.align} double tbuf_re[{R * VL}];")
            em.L.append(f"    {isa.align} double tbuf_im[{R * VL}];")
            em.L.append(f"    {isa.align} double spill_re[{R * VL}];")
            em.L.append(f"    {isa.align} double spill_im[{R * VL}];")
            emit_constants_raw(em.L, isa, indent=1)
            emit_hoisted_w6(em.L, isa, itw_set, indent=1)
            em.L.append(f"    {T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im;")
            em.L.append(f"")
            em.L.append(f"    for (size_t k = 0; k < vl; k += {VL}) {{")

            em2 = Emitter(isa); em2.L = []; em2.ind = 2; em2.reset()
            em2.addr_mode = 'n1_ovs'
            emit_kernel_body(em2, d, itw_set, 'notw')
            em.L.extend(em2.L)

            # 4x4 transpose: bins 0-3
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

            # Scatter bins 4,5
            em.L.append(f"        /* Scatter: bins 4,5 */")
            for bin_idx in [4, 5]:
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

    em.L.append(f""); em.L.append(f"#undef LD"); em.L.append(f"#undef ST")
    em.L.append(f""); em.L.append(f"#endif /* {guard} */")
    return em.L


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

ALL_VARIANTS = ['notw', 'dit_tw', 'dif_tw', 'dit_tw_log3', 'dif_tw_log3',
                'ct_n1', 'ct_t1_dit', 'ct_t1s_dit', 'ct_t1_dit_log3', 'ct_t1_dif']
CT_VARIANTS = {'ct_n1', 'ct_t1_dit', 'ct_t1s_dit', 'ct_t1_dit_log3', 'ct_t1_dif'}

def main():
    parser = argparse.ArgumentParser(description='DFT-6 unified codelet generator')
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
            print(f"/* gen_radix6.py: {isa.name}/{variant} -- {len(lines)} lines */",
                  file=sys.stderr)

if __name__ == '__main__':
    main()
