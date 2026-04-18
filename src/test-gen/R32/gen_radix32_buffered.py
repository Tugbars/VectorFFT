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
gen_radix32.py — Unified radix-32 codelet generator

Emits all R=32 codelet variants from shared butterfly infrastructure
with per-ISA / per-variant strategy configurations.

8x4 decomposition: pass 1 = N2 radix-8 sub-FFTs, pass 2 = N1 radix-4 combines.
Internal twiddles W32^(n1*n2) between passes.

ISA-specific decisions preserved:
  Scalar:  NFUSE=4, k-step=1, no SIMD
  AVX2:    NFUSE=2, k-step=4, 16 YMM, flat only, U=1
  AVX-512: NFUSE=4, k-step=8, 32 ZMM, flat+ladder, U=1

Usage:
  python3 gen_radix32.py --isa avx2 --variant dit_tw
  python3 gen_radix32.py --isa avx512 --variant dit_tw
  python3 gen_radix32.py --isa scalar --variant dit_tw
  python3 gen_radix32.py --isa all --variant all

Variant selection notes (for planner / bench-time calibration)
──────────────────────────────────────────────────────────────

IMPORTANT: variant rankings depend on the target microarchitecture.
Numbers below are from a single data point (Sapphire Rapids, Xeon-class
server) and DO NOT generalize. Ranking determinants include:
  - L1 DTLB size (Raptor Lake ~16, SPR ~96, Zen4/5 ~72 entries)
  - L2 associativity (SPR 16-way, Zen4 8-way, Zen5 16-way)
  - L1 load bandwidth (Intel big-core 2 ports, Zen 3 ports)
  - FMA throughput per cycle
  - HW prefetcher aggressiveness
  - Cache replacement policy
Any of these can flip a variant's ranking. Always bench on the
target chip; do not extrapolate from the numbers here.

ct_t1_buf_dit — hardware-specific candidate
  Buffered variant targeting L1 DTLB-store-overhead bottleneck seen on
  Raptor Lake (13th/14th gen consumer Intel) at R=32 me=256, where
  VTune reported ~66% DTLB-store overhead with flat ct_t1_dit.
  On the Sapphire Rapids container (L1 DTLB ~96 entries vs Raptor
  Lake's ~16), buffered loses to log3 and flat at small K — no DTLB
  pressure to relieve.
  Untested on Raptor Lake as of this writing. Planner should not
  default to buf_dit; it should be a bench-time candidate whose
  inclusion in the selection table depends on the measured target.

ct_t1_dit_log3 vs ct_t1_ladder_dit — overlapping derived-twiddle variants
  Both read 5 base twiddle rows {0,1,3,7,15} from the flat W table
  and derive the other 26 via cmul chains. Relative performance
  depends on target µarch balance of FMA throughput vs L1/L2 load
  bandwidth:
    - More FMA-constrained chips: ladder/log3 lose to flat
    - More load-bandwidth-constrained chips: ladder/log3 win
  ladder AVX2 not yet ported (needs NFUSE=2 rework of the pipeline).
  Currently: log3 available for AVX2 + AVX-512; ladder for AVX-512 only.

  Three consolidation paths (decide after bench data from real targets):
    (a) Keep log3 for AVX2, ladder for AVX-512 — split ISA roles
    (b) Unify on log3 for both ISAs — simpler
    (c) Port ladder to AVX2, delete log3 — unified on ladder
  No decision taken; both variants coexist pending bench data.
"""

import math, sys, argparse, re, re

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

N, N1, N2 = 32, 8, 4

# ═══════════════════════════════════════════════════════════════
# TWIDDLE ANALYSIS — shared across all variants
# ═══════════════════════════════════════════════════════════════

def wN(e, tN):
    e = e % tN
    a = 2.0 * math.pi * e / tN
    return (math.cos(a), -math.sin(a))

def wN_label(e, tN):
    return f"W{tN}_{e % tN}"

def twiddle_is_trivial(e, tN):
    """Classify twiddle factor for zero-multiply fast paths.
    
    Returns (is_trivial, type_tag).
    8th-root-of-unity twiddles get special codegen (no mul needed).
    """
    e = e % tN
    if e == 0:
        return True, 'one'
    if (8 * e) % tN == 0:
        o = (8 * e) // tN
        t = ['one', 'w8_1', 'neg_j', 'w8_3',
             'neg_one', 'neg_w8_1', 'pos_j', 'neg_w8_3']
        return True, t[o % 8]
    return False, 'cmul'

def collect_internal_twiddles():
    """Find which W32 constants are needed for internal twiddles."""
    tw = set()
    for n2 in range(1, N2):
        for k1 in range(1, N1):
            e = (n2 * k1) % N
            _, t = twiddle_is_trivial(e, N)
            if t == 'cmul':
                tw.add((e, N))
    return tw


# ═══════════════════════════════════════════════════════════════
# ISA CONFIGURATION
# ═══════════════════════════════════════════════════════════════

class ISAConfig:
    """ISA-specific parameters. Each field matches the existing generator exactly."""
    def __init__(self, name, reg_type, width, k_step,
                 load_macro, store_macro, spill_mul,
                 target_attr, align_attr, align_bytes,
                 nfuse_tw, nfuse_notw):
        self.name = name
        self.reg_type = reg_type
        self.width = width          # doubles per register
        self.k_step = k_step
        self.load_macro = load_macro    # e.g. "LD" or raw intrinsic
        self.store_macro = store_macro
        self.spill_mul = spill_mul      # slot index multiplier (4 for avx2, 8 for avx512)
        self.target_attr = target_attr
        self.align_attr = align_attr
        self.align_bytes = align_bytes
        self.nfuse_tw = nfuse_tw
        self.nfuse_notw = nfuse_notw

ISA_SCALAR = ISAConfig(
    name='scalar', reg_type='double', width=1, k_step=1,
    load_macro=None, store_macro=None, spill_mul=1,
    target_attr='', align_attr='', align_bytes=0,
    nfuse_tw=4, nfuse_notw=4,
)

ISA_AVX2 = ISAConfig(
    name='avx2', reg_type='__m256d', width=4, k_step=4,
    load_macro='LD', store_macro='ST', spill_mul=4,
    target_attr='__attribute__((target("avx2,fma")))',
    align_attr='__attribute__((aligned(32)))',
    align_bytes=32,
    nfuse_tw=2, nfuse_notw=2,
)

ISA_AVX512 = ISAConfig(
    name='avx512', reg_type='__m512d', width=8, k_step=8,
    load_macro='LD', store_macro='ST', spill_mul=8,
    target_attr='__attribute__((target("avx512f,avx512dq,fma")))',
    align_attr='__attribute__((aligned(64)))',
    align_bytes=64,
    nfuse_tw=4, nfuse_notw=8,
)

ALL_ISA = {'scalar': ISA_SCALAR, 'avx2': ISA_AVX2, 'avx512': ISA_AVX512}

# ═══════════════════════════════════════════════════════════════
# EMITTER — shared code emission with ISA-parameterized operations
# ═══════════════════════════════════════════════════════════════

class Emitter:
    def __init__(self, isa):
        self.isa = isa
        self.L = []
        self.ind = 1
        self.spill_c = 0
        self.reload_c = 0
        # Arithmetic counters (per kernel, reset between fwd/bwd)
        self.n_add = 0
        self.n_sub = 0
        self.n_mul = 0
        self.n_neg = 0
        self.n_fma = 0
        self.n_fms = 0
        self.n_load = 0
        self.n_store = 0
        # Twiddle prefetch: distance ahead (in m) to prefetch next iteration's
        # twiddle cachelines. 0 = disabled. Target: hide L2-latency of twiddle
        # table that spills L1D at K >= 128 on SPR-class chips.
        self.twiddle_prefetch_distance = 0
        self.twiddle_prefetch_rows = 2  # how many rows to prefetch per k-step

    def reset_counters(self):
        self.spill_c = 0
        self.reload_c = 0
        self.n_add = 0
        self.n_sub = 0
        self.n_mul = 0
        self.n_neg = 0
        self.n_fma = 0
        self.n_fms = 0
        self.n_load = 0
        self.n_store = 0

    def get_stats(self):
        """Return a dict of all operation counts for the current kernel."""
        total_arith = self.n_add + self.n_sub + self.n_mul + self.n_neg + self.n_fma + self.n_fms
        total_mem = self.n_load + self.n_store + self.spill_c + self.reload_c
        # FMA counts as 2 FLOPs (mul + add/sub)
        flops = (self.n_add + self.n_sub + self.n_neg) + self.n_mul + 2 * (self.n_fma + self.n_fms)
        return {
            'add': self.n_add,
            'sub': self.n_sub,
            'mul': self.n_mul,
            'neg': self.n_neg,
            'fma': self.n_fma,
            'fms': self.n_fms,
            'total_arith': total_arith,
            'flops': flops,
            'load': self.n_load,
            'store': self.n_store,
            'spill': self.spill_c,
            'reload': self.reload_c,
            'total_mem': total_mem,
        }

    def o(self, t=""):
        self.L.append("    " * self.ind + t)

    def c(self, t):
        self.o(f"/* {t} */")

    def b(self):
        self.L.append("")

    # ── Arithmetic primitives (ISA-dispatched, counted) ──

    def add(self, a, b):
        self.n_add += 1
        if self.isa.name == 'scalar':
            return f"({a})+({b})"
        elif self.isa.name == 'avx2':
            return f"_mm256_add_pd({a},{b})"
        else:
            return f"_mm512_add_pd({a},{b})"

    def sub(self, a, b):
        self.n_sub += 1
        if self.isa.name == 'scalar':
            return f"({a})-({b})"
        elif self.isa.name == 'avx2':
            return f"_mm256_sub_pd({a},{b})"
        else:
            return f"_mm512_sub_pd({a},{b})"

    def mul(self, a, b):
        self.n_mul += 1
        if self.isa.name == 'scalar':
            return f"({a})*({b})"
        elif self.isa.name == 'avx2':
            return f"_mm256_mul_pd({a},{b})"
        else:
            return f"_mm512_mul_pd({a},{b})"

    def neg(self, a):
        self.n_neg += 1
        if self.isa.name == 'scalar':
            return f"-({a})"
        elif self.isa.name == 'avx2':
            return f"_mm256_xor_pd({a},sign_flip)"
        else:
            return f"_mm512_xor_pd({a},sign_flip)"

    def fma(self, a, b, c):
        self.n_fma += 1
        if self.isa.name == 'scalar':
            return f"({a})*({b})+({c})"
        elif self.isa.name == 'avx2':
            return f"_mm256_fmadd_pd({a},{b},{c})"
        else:
            return f"_mm512_fmadd_pd({a},{b},{c})"

    def fms(self, a, b, c):
        self.n_fms += 1
        if self.isa.name == 'scalar':
            return f"({a})*({b})-({c})"
        elif self.isa.name == 'avx2':
            return f"_mm256_fmsub_pd({a},{b},{c})"
        else:
            return f"_mm512_fmsub_pd({a},{b},{c})"

    # ── Load / Store / Spill / Reload ──

    # addr_mode: 'K' (default), 'n1' (separate is/os), 't1' (in-place ios/ms),
    #            't1_buf' (buffered output — reads like t1, stores to outbuf).
    addr_mode = 'K'

    def _in_addr(self, n, k_expr="k"):
        if self.addr_mode in ('n1', 'n1_ovs'): return f"{n}*is+{k_expr}"
        # t1s uses the same in-place data layout as t1 — only the TWIDDLE
        # delivery differs (scalar-broadcast vs vector-load).
        if self.addr_mode in ('t1', 't1_buf', 't1s'):
            if self.isa.name == 'scalar': return f"m*ms+{n}*ios"
            return f"m+{n}*ios"
        if self.addr_mode == 't1_oop': return f"m+{n}*is"
        return f"{n}*K+{k_expr}"

    def _out_addr(self, m, k_expr="k"):
        if self.addr_mode == 'n1': return f"{m}*os+{k_expr}"
        if self.addr_mode == 'n1_ovs': return f"{m}*{self.isa.k_step}"
        if self.addr_mode in ('t1', 't1s'):
            if self.isa.name == 'scalar': return f"m*ms+{m}*ios"
            return f"m+{m}*ios"
        if self.addr_mode == 't1_buf':
            # Stores go to outbuf[m_out*TILE + kk] — written contiguously per
            # output stream, drained to rio after each tile completes.
            return f"{m}*TILE+kk"
        if self.addr_mode == 't1_oop': return f"m+{m}*os"
        return f"{m}*K+{k_expr}"

    def _in_buf(self):
        return "rio_re" if self.addr_mode in ('t1', 't1_buf', 't1s') else "in_re"
    def _in_buf_im(self):
        return "rio_im" if self.addr_mode in ('t1', 't1_buf', 't1s') else "in_im"
    def _out_buf(self):
        if self.addr_mode in ('t1', 't1s'): return "rio_re"
        if self.addr_mode == 't1_buf': return "outbuf_re"
        if self.addr_mode == 'n1_ovs': return "tbuf_re"
        return "out_re"
    def _out_buf_im(self):
        if self.addr_mode in ('t1', 't1s'): return "rio_im"
        if self.addr_mode == 't1_buf': return "outbuf_im"
        if self.addr_mode == 'n1_ovs': return "tbuf_im"
        return "out_im"

    def _tw_addr(self, tw_idx, k_expr="k"):
        # t1s: W_re/W_im are (R-1) scalars (one per twiddle row), not arrays
        # indexed by m. Broadcast from W_re[tw_idx] happens in emit_ext_tw_scalar.
        if self.addr_mode == 't1s': return f"{tw_idx}"
        if self.addr_mode in ('t1', 't1_buf', 't1_oop'): return f"{tw_idx}*me+m"
        return f"{tw_idx}*K+{k_expr}"
    def _tw_buf(self):
        # t1s uses the same W_re/W_im names as t1, but reads scalar not vector
        if self.addr_mode in ('t1', 't1_buf', 't1_oop', 't1s'): return "W_re"
        return "tw_re"
    def _tw_buf_im(self):
        if self.addr_mode in ('t1', 't1_buf', 't1_oop', 't1s'): return "W_im"
        return "tw_im"

    def emit_load(self, v, n, k_expr="k"):
        self.n_load += 2  # re + im
        ib, ibi = self._in_buf(), self._in_buf_im()
        addr = self._in_addr(n, k_expr)
        if self.isa.name == 'scalar':
            self.o(f"{v}_re = {ib}[{addr}];")
            self.o(f"{v}_im = {ibi}[{addr}];")
        else:
            self.o(f"{v}_re = {self.isa.load_macro}(&{ib}[{addr}]);")
            self.o(f"{v}_im = {self.isa.load_macro}(&{ibi}[{addr}]);")

    def emit_store(self, v, m, k_expr="k"):
        self.n_store += 2  # re + im
        ob, obi = self._out_buf(), self._out_buf_im()
        addr = self._out_addr(m, k_expr)
        if self.isa.name == 'scalar':
            if getattr(self, 'store_scale', False):
                self.o(f"{ob}[{addr}] = scale * {v}_re;")
                self.o(f"{obi}[{addr}] = scale * {v}_im;")
            else:
                self.o(f"{ob}[{addr}] = {v}_re;")
                self.o(f"{obi}[{addr}] = {v}_im;")
        else:
            if getattr(self, 'store_scale', False):
                mul = '_mm256_mul_pd' if self.isa.name == 'avx2' else '_mm512_mul_pd'
                self.o(f"{self.isa.store_macro}(&{ob}[{addr}],{mul}(vscale,{v}_re));")
                self.o(f"{self.isa.store_macro}(&{obi}[{addr}],{mul}(vscale,{v}_im));")
            else:
                self.o(f"{self.isa.store_macro}(&{ob}[{addr}],{v}_re);")
                self.o(f"{self.isa.store_macro}(&{obi}[{addr}],{v}_im);")

    def emit_spill(self, v, slot):
        sm = self.isa.spill_mul
        if self.isa.name == 'scalar':
            self.o(f"spill_re[{slot}] = {v}_re;")
            self.o(f"spill_im[{slot}] = {v}_im;")
        elif self.isa.name == 'avx2':
            self.o(f"_mm256_store_pd(&spill_re[{slot}*{sm}],{v}_re);")
            self.o(f"_mm256_store_pd(&spill_im[{slot}*{sm}],{v}_im);")
        else:
            self.o(f"_mm512_store_pd(&spill_re[{slot}*{sm}],{v}_re);")
            self.o(f"_mm512_store_pd(&spill_im[{slot}*{sm}],{v}_im);")
        self.spill_c += 1

    def emit_reload(self, v, slot):
        sm = self.isa.spill_mul
        if self.isa.name == 'scalar':
            self.o(f"{v}_re = spill_re[{slot}];")
            self.o(f"{v}_im = spill_im[{slot}];")
        elif self.isa.name == 'avx2':
            self.o(f"{v}_re = _mm256_load_pd(&spill_re[{slot}*{sm}]);")
            self.o(f"{v}_im = _mm256_load_pd(&spill_im[{slot}*{sm}]);")
        else:
            self.o(f"{v}_re = _mm512_load_pd(&spill_re[{slot}*{sm}]);")
            self.o(f"{v}_im = _mm512_load_pd(&spill_im[{slot}*{sm}]);")
        self.reload_c += 1

    # ── Radix-8 butterfly (the core — identical math across all ISAs) ──

    def emit_radix8(self, v, d, label=""):
        """Emit split-radix DFT-8 on variables v[0]..v[7].
        
        Decomposed as two interleaved radix-4 (even/odd split) with
        W8 internal twiddles. This is the same algorithm across all
        generators — only the intrinsic names change.
        """
        fwd = (d == 'fwd')
        T = self.isa.reg_type
        if label:
            self.c(f"{label} [{d}]")
        self.o(f"{{ {T} e0r,e0i,e1r,e1i,e2r,e2i,e3r,e3i;")
        self.o(f"  {T} t0r,t0i,t1r,t1i,t2r,t2i,t3r,t3i;")
        # Even part: v[0], v[2], v[4], v[6]
        self.o(f"  t0r={self.add(f'{v[0]}_re',f'{v[4]}_re')}; t0i={self.add(f'{v[0]}_im',f'{v[4]}_im')};")
        self.o(f"  t1r={self.sub(f'{v[0]}_re',f'{v[4]}_re')}; t1i={self.sub(f'{v[0]}_im',f'{v[4]}_im')};")
        self.o(f"  t2r={self.add(f'{v[2]}_re',f'{v[6]}_re')}; t2i={self.add(f'{v[2]}_im',f'{v[6]}_im')};")
        self.o(f"  t3r={self.sub(f'{v[2]}_re',f'{v[6]}_re')}; t3i={self.sub(f'{v[2]}_im',f'{v[6]}_im')};")
        self.o(f"  e0r={self.add('t0r','t2r')}; e0i={self.add('t0i','t2i')};")
        self.o(f"  e2r={self.sub('t0r','t2r')}; e2i={self.sub('t0i','t2i')};")
        j_add, j_sub = ('add', 'sub') if fwd else ('sub', 'add')
        self.o(f"  e1r={getattr(self,j_add)('t1r','t3i')}; e1i={getattr(self,j_sub)('t1i','t3r')};")
        self.o(f"  e3r={getattr(self,j_sub)('t1r','t3i')}; e3i={getattr(self,j_add)('t1i','t3r')};")
        # Odd part: v[1], v[3], v[5], v[7]
        self.o(f"  {T} o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;")
        self.o(f"  t0r={self.add(f'{v[1]}_re',f'{v[5]}_re')}; t0i={self.add(f'{v[1]}_im',f'{v[5]}_im')};")
        self.o(f"  t1r={self.sub(f'{v[1]}_re',f'{v[5]}_re')}; t1i={self.sub(f'{v[1]}_im',f'{v[5]}_im')};")
        self.o(f"  t2r={self.add(f'{v[3]}_re',f'{v[7]}_re')}; t2i={self.add(f'{v[3]}_im',f'{v[7]}_im')};")
        self.o(f"  t3r={self.sub(f'{v[3]}_re',f'{v[7]}_re')}; t3i={self.sub(f'{v[3]}_im',f'{v[7]}_im')};")
        self.o(f"  o0r={self.add('t0r','t2r')}; o0i={self.add('t0i','t2i')};")
        self.o(f"  o2r={self.sub('t0r','t2r')}; o2i={self.sub('t0i','t2i')};")
        self.o(f"  o1r={getattr(self,j_add)('t1r','t3i')}; o1i={getattr(self,j_sub)('t1i','t3r')};")
        self.o(f"  o3r={getattr(self,j_sub)('t1r','t3i')}; o3i={getattr(self,j_add)('t1i','t3r')};")
        # W8 twiddles on odd outputs: o1 *= W8^1, o2 *= W8^2=-j, o3 *= W8^3
        if fwd:
            self.o(f"  t0r={self.mul(self.add('o1r','o1i'),'sqrt2_inv')};")
            self.o(f"  t0i={self.mul(self.sub('o1i','o1r'),'sqrt2_inv')};")
            self.o(f"  o1r=t0r; o1i=t0i;")
            self.o(f"  t0r=o2i; t0i={self.neg('o2r')};")
            self.o(f"  o2r=t0r; o2i=t0i;")
            self.o(f"  t0r={self.mul(self.sub('o3i','o3r'),'sqrt2_inv')};")
            self.o(f"  t0i={self.neg(self.mul(self.add('o3r','o3i'),'sqrt2_inv'))};")
            self.o(f"  o3r=t0r; o3i=t0i;")
        else:
            self.o(f"  t0r={self.mul(self.sub('o1r','o1i'),'sqrt2_inv')};")
            self.o(f"  t0i={self.mul(self.add('o1r','o1i'),'sqrt2_inv')};")
            self.o(f"  o1r=t0r; o1i=t0i;")
            self.o(f"  t0r={self.neg('o2i')}; t0i=o2r;")
            self.o(f"  o2r=t0r; o2i=t0i;")
            self.o(f"  t0r={self.neg(self.mul(self.add('o3r','o3i'),'sqrt2_inv'))};")
            self.o(f"  t0i={self.mul(self.sub('o3r','o3i'),'sqrt2_inv')};")
            self.o(f"  o3r=t0r; o3i=t0i;")
        # Combine: v[i] = e_i + o_i, v[i+4] = e_i - o_i
        for i, j in [(0, 4), (1, 5), (2, 6), (3, 7)]:
            self.o(f"  {v[i]}_re={self.add(f'e{i}r',f'o{i}r')}; {v[i]}_im={self.add(f'e{i}i',f'o{i}i')};")
            self.o(f"  {v[j]}_re={self.sub(f'e{i}r',f'o{i}r')}; {v[j]}_im={self.sub(f'e{i}i',f'o{i}i')};")
        self.o(f"}}")

    # ── Radix-4 butterfly ──

    def emit_radix4(self, v, d, label=""):
        fwd = (d == 'fwd')
        T = self.isa.reg_type
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

    # ── Internal twiddle W32^e (8 trivial specializations + cmul) ──

    def emit_twiddle(self, dst, src, e, tN, d):
        """Apply internal twiddle W_{tN}^e to variable src, store in dst.
        
        8 trivial cases avoid multiplications entirely.
        Non-trivial cases use pre-broadcast constant registers.
        """
        _, typ = twiddle_is_trivial(e, tN)
        fwd = (d == 'fwd')
        T = self.isa.reg_type

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
            # General cmul with pre-broadcast constant
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

    # ── External twiddle load + apply (DIT: pre-butterfly) ──

    def emit_ext_twiddle_dit(self, n1, n, d, k_expr="k"):
        """Load external twiddle for element n and apply to x{n1} (DIT: pre-multiply).

        In t1s addressing mode, the twiddle is a SCALAR (W_re[n-1]) that gets
        broadcast to the vector width, rather than a vector load from
        W_re[(n-1)*me + m]. The t1s_hoisted mechanism lets frequently-used
        twiddles be pre-broadcast once before the m-loop (amortizing the
        broadcast across the loop), while the rest are broadcast inline.
        """
        if n == 0:
            return  # first element has no twiddle
        # Dispatch to scalar-broadcast version if in t1s mode
        if self.addr_mode == 't1s':
            self.emit_ext_tw_scalar(f'x{n1}', n-1, d)
            return
        fwd = (d == 'fwd')
        T = self.isa.reg_type
        tb, tbi = self._tw_buf(), self._tw_buf_im()
        ta = self._tw_addr(n-1, k_expr)
        if self.isa.name == 'scalar':
            self.o(f"{{ double wr = {tb}[{ta}], wi = {tbi}[{ta}];")
            self.o(f"  double tr = x{n1}_re;")
            if fwd:
                self.o(f"  x{n1}_re = x{n1}_re*wr - x{n1}_im*wi;")
                self.o(f"  x{n1}_im = tr*wi + x{n1}_im*wr; }}")
            else:
                self.o(f"  x{n1}_re = x{n1}_re*wr + x{n1}_im*wi;")
                self.o(f"  x{n1}_im = x{n1}_im*wr - tr*wi; }}")
        else:
            load_fn = '_mm256_load_pd' if self.isa.name == 'avx2' else '_mm512_load_pd'
            self.o(f"{{ {T} wr = {load_fn}(&{tb}[{ta}]);")
            self.o(f"  {T} wi = {load_fn}(&{tbi}[{ta}]);")
            self.o(f"  {T} tr = x{n1}_re;")
            if fwd:
                self.o(f"  x{n1}_re = {self.fms(f'x{n1}_re','wr',self.mul(f'x{n1}_im','wi'))};")
                self.o(f"  x{n1}_im = {self.fma('tr','wi',self.mul(f'x{n1}_im','wr'))}; }}")
            else:
                self.o(f"  x{n1}_re = {self.fma(f'x{n1}_re','wr',self.mul(f'x{n1}_im','wi'))};")
                self.o(f"  x{n1}_im = {self.fms(f'x{n1}_im','wr',self.mul('tr','wi'))}; }}")

    # ── Scalar-broadcast twiddle (t1s variant) ──

    def emit_hoist_all_tw_scalars(self, R):
        """Emit broadcast of twiddle scalars BEFORE the m-loop.

        Register-budget-aware: only hoists what fits in SIMD registers without
        displacing data/temp registers needed inside the butterfly. Remaining
        twiddles are broadcast inline inside the loop.

        R=32 butterfly has R-1=31 twiddle rows. Register budget is tighter
        than R=16 because more x{n} vars are live inside the butterfly.

        Budgets (conservative, leave headroom for spills):
          AVX2  (16 YMM): hoist 4 twiddles → 8 YMM, leaves 8 for data+temps
          AVX-512 (32 ZMM): hoist 10 twiddles → 20 ZMM, leaves 12 for data+temps
        """
        T = self.isa.reg_type
        n_tw = R - 1
        if self.isa.name == 'scalar':
            max_hoist = n_tw  # scalar has unlimited 'registers'
        elif self.isa.name == 'avx2':
            max_hoist = 4     # 8 YMM for hoisted twiddles, 8 for butterfly state
        else:  # avx512
            max_hoist = 10    # 20 ZMM for hoisted twiddles, 12 for butterfly state
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
        """Apply a pre-hoisted twiddle tw{idx} to variable v (no broadcast)."""
        fwd = (d == 'fwd')
        T = self.isa.reg_type
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

        W_re/W_im are (R-1) scalars, NOT (R-1)*me arrays. A scalar is
        broadcast to SIMD width at each use (unless pre-hoisted).
        """
        if getattr(self, 'tw_hoisted', False) and tw_idx in getattr(self, 'tw_hoisted_set', set()):
            self.emit_apply_hoisted_tw(v, tw_idx, d)
            return
        fwd = (d == 'fwd')
        T = self.isa.reg_type
        self.n_load += 2  # scalar re + scalar im (but broadcast, not vector load)
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

    # ── Complex multiply (separate dst — needed by ladder) ──

    def emit_cmul(self, dst_r, dst_i, ar, ai, br, bi, d):
        """Emit dst = a * b (fwd) or a * conj(b) (bwd)."""
        fwd = (d == 'fwd')
        if fwd:
            self.o(f"{dst_r} = {self.fms(ar,br,self.mul(ai,bi))};")
            self.o(f"{dst_i} = {self.fma(ar,bi,self.mul(ai,br))};")
        else:
            self.o(f"{dst_r} = {self.fma(ar,br,self.mul(ai,bi))};")
            self.o(f"{dst_i} = {self.fms(ai,br,self.mul(ar,bi))};")

    # ── Complex multiply in-place (needed by ladder) ──

    def emit_cmul_inplace(self, v, wr, wi, d):
        """Emit v *= (wr + j*wi) (fwd) or v *= conj(wr + j*wi) (bwd)."""
        fwd = (d == 'fwd')
        T = self.isa.reg_type
        self.o(f"{{ {T} tr={v}_re;")
        if fwd:
            self.o(f"  {v}_re={self.fms(f'{v}_re',wr,self.mul(f'{v}_im',wi))};")
            self.o(f"  {v}_im={self.fma('tr',wi,self.mul(f'{v}_im',wr))}; }}")
        else:
            self.o(f"  {v}_re={self.fma(f'{v}_re',wr,self.mul(f'{v}_im',wi))};")
            self.o(f"  {v}_im={self.fms(f'{v}_im',wr,self.mul('tr',wi))}; }}")

    # ── External twiddle on OUTPUT (DIF: post-butterfly) ──

    def emit_ext_twiddle_dif(self, v, tw_idx, d, k_expr="k"):
        """Apply external twiddle to variable v in-place (DIF)."""
        fwd = (d == 'fwd')
        T = self.isa.reg_type
        tb, tbi = self._tw_buf(), self._tw_buf_im()
        ta = self._tw_addr(tw_idx, k_expr)
        if self.isa.name == 'scalar':
            self.o(f"{{ double wr = {tb}[{ta}], wi = {tbi}[{ta}];")
            self.o(f"  double tr = {v}_re;")
            if fwd:
                self.o(f"  {v}_re = {v}_re*wr - {v}_im*wi;")
                self.o(f"  {v}_im = tr*wi + {v}_im*wr; }}")
            else:
                self.o(f"  {v}_re = {v}_re*wr + {v}_im*wi;")
                self.o(f"  {v}_im = {v}_im*wr - tr*wi; }}")
        else:
            load_fn = '_mm256_load_pd' if self.isa.name == 'avx2' else '_mm512_load_pd'
            self.o(f"{{ {T} wr = {load_fn}(&{tb}[{ta}]);")
            self.o(f"  {T} wi = {load_fn}(&{tbi}[{ta}]);")
            self.o(f"  {T} tr = {v}_re;")
            if fwd:
                self.o(f"  {v}_re = {self.fms(f'{v}_re','wr',self.mul(f'{v}_im','wi'))};")
                self.o(f"  {v}_im = {self.fma('tr','wi',self.mul(f'{v}_im','wr'))}; }}")
            else:
                self.o(f"  {v}_re = {self.fma(f'{v}_re','wr',self.mul(f'{v}_im','wi'))};")
                self.o(f"  {v}_im = {self.fms(f'{v}_im','wr',self.mul('tr','wi'))}; }}")


# ═══════════════════════════════════════════════════════════════
# KERNEL EMITTER — DIT tw flat (shared structure, ISA-parameterized)
# ═══════════════════════════════════════════════════════════════

def emit_dit_tw_flat_kernel(em, d, nfuse, itw_set, k_expr="k"):
    """Emit one DIT twiddled flat kernel (fwd or bwd).
    
    This is the core kernel structure shared across scalar/AVX2/AVX-512 flat.
    PASS 1: N2 radix-8 sub-FFTs with external twiddle pre-multiply.
    PASS 2: N1 radix-4 column combines with internal twiddles.
    """
    isa = em.isa
    T = isa.reg_type
    fwd = (d == 'fwd')
    last_n2 = N2 - 1

    xv8 = [f"x{i}" for i in range(N1)]
    xv4 = [f"x{i}" for i in range(N2)]

    # Twiddle prefetch: warm L2 lines for the m-iteration `prefetch_distance`
    # ahead. Only active when em.twiddle_prefetch_distance > 0 and addr_mode
    # uses per-m twiddle stride (t1, t1_buf, t1_oop).
    if (em.twiddle_prefetch_distance > 0
            and em.addr_mode in ('t1', 't1_buf', 't1_oop')
            and isa.name != 'scalar'
            and k_expr == 'k' or k_expr == 'm'):
        # For 't1' modes the twiddle index expression is `n*me + m`. Prefetch
        # a few rows at offset `m + PREFETCH_DIST`. Each row is typically one
        # cacheline at VL=8 (AVX-512, 64 bytes) — prefetch only the first few
        # rows to keep front-end pressure low; HW prefetcher catches up.
        m_var = "m" if em.addr_mode in ('t1', 't1_buf', 't1_oop') else k_expr
        em.c(f"Twiddle prefetch: warm L2 lines {em.twiddle_prefetch_distance} m ahead "
             f"({em.twiddle_prefetch_rows} rows × re+im)")
        dist = em.twiddle_prefetch_distance
        for n in range(em.twiddle_prefetch_rows):
            # locality=2 (L2) for twiddle warming; rw=0 (read)
            em.o(f"__builtin_prefetch(&W_re[{n}*me + {m_var} + {dist}], 0, 2);")
            em.o(f"__builtin_prefetch(&W_im[{n}*me + {m_var} + {dist}], 0, 2);")
        em.b()

    # PASS 1: N2 sub-FFTs
    for n2 in range(N2):
        is_last = (n2 == last_n2)
        em.c(f"sub-FFT n2={n2}")
        for n1 in range(N1):
            n = N2 * n1 + n2
            em.emit_load(f"x{n1}", n, k_expr)
            em.emit_ext_twiddle_dit(n1, n, d, k_expr)
        em.b()
        em.emit_radix8(xv8, d, f"radix-8 n2={n2}")
        em.b()

        # Spill/fuse strategy
        if is_last:
            em.c(f"FUSED: save x0..x{nfuse-1} to s-regs, spill x{nfuse}..x{N1-1}")
            for k1 in range(nfuse):
                em.o(f"s{k1}_re = x{k1}_re; s{k1}_im = x{k1}_im;")
            for k1 in range(nfuse, N1):
                em.emit_spill(f"x{k1}", n2 * N1 + k1)
        else:
            for k1 in range(N1):
                em.emit_spill(f"x{k1}", n2 * N1 + k1)
        em.b()

    # PASS 2: N1 column combines
    em.c(f"PASS 2 — W32 twiddle broadcasts deferred to free regs during PASS 1")
    if isa.name != 'scalar' and itw_set:
        set1 = '_mm256_set1_pd' if isa.name == 'avx2' else '_mm512_set1_pd'
        for (e, tN) in sorted(itw_set):
            label = wN_label(e, tN)
            em.o(f"const {T} tw_{label}_re = {set1}({label}_re);")
            em.o(f"const {T} tw_{label}_im = {set1}({label}_im);")
    em.b()
    for k1 in range(N1):
        em.c(f"column k1={k1}")

        # Reload with fuse optimization
        if k1 < nfuse:
            for n2 in range(last_n2):
                em.emit_reload(f"x{n2}", n2 * N1 + k1)
            em.o(f"x{last_n2}_re = s{k1}_re; x{last_n2}_im = s{k1}_im;")
        else:
            for n2 in range(N2):
                em.emit_reload(f"x{n2}", n2 * N1 + k1)
        em.b()

        # Internal twiddles (skip k1=0 — all W32^0 = 1)
        if k1 > 0:
            for n2 in range(1, N2):
                e = (n2 * k1) % N
                em.emit_twiddle(f"x{n2}", f"x{n2}", e, N, d)
            em.b()

        em.emit_radix4(xv4, d, f"radix-4 k1={k1}")
        em.b()

        for k2 in range(N2):
            em.emit_store(f"x{k2}", k1 + N1 * k2, k_expr)
        em.b()


# ═══════════════════════════════════════════════════════════════
# KERNEL EMITTER — DIF tw flat
#
# Same 8×4 structure as DIT, but:
#   PASS 1: load (NO ext twiddle) → radix-8 → spill
#   PASS 2: reload → int twiddle → radix-4 → ext twiddle on OUTPUT → store
# ═══════════════════════════════════════════════════════════════

def emit_dit_tw_log3_kernel(em, d, nfuse, itw_set, k_expr="k"):
    """Emit one DIT twiddled log3 kernel (fwd or bwd).

    Loads 5 base twiddles: W^1(row0), W^2(row1), W^4(row3), W^8(row7), W^16(row15).
    Derives all 26 remaining twiddles via binary decomposition cmul chains.
    PASS 2 is identical to flat (internal twiddles are constants).
    """
    isa = em.isa
    T = isa.reg_type
    fwd = (d == 'fwd')
    last_n2 = N2 - 1

    xv8 = [f"x{i}" for i in range(N1)]
    xv4 = [f"x{i}" for i in range(N2)]

    # ── Load 5 base twiddles ──
    em.c("Log3: load 5 base twiddles W^1, W^2, W^4, W^8, W^16")
    tb, tbi = em._tw_buf(), em._tw_buf_im()
    bases = [('b1',0), ('b2',1), ('b4',3), ('b8',7), ('b16',15)]  # (name, row=j-1)
    for bname, row in bases:
        ta = em._tw_addr(row, k_expr)
        if isa.name == 'scalar':
            em.o(f"const double {bname}_re = {tb}[{ta}], {bname}_im = {tbi}[{ta}];")
        else:
            load_fn = '_mm256_load_pd' if isa.name == 'avx2' else '_mm512_load_pd'
            em.o(f"const {T} {bname}_re = {load_fn}(&{tb}[{ta}]);")
            em.o(f"const {T} {bname}_im = {load_fn}(&{tbi}[{ta}]);")
    em.b()

    # ── Derive column twiddles W^{4*n1} for n1=0..7 ──
    # W^0=1 (skip), W^4=b4 (base), W^8=b8 (base), W^16=b16 (base)
    # Derive: W^12=b4*b8, W^20=b4*b16, W^24=b8*b16, W^28=W^12*b16
    em.c("Derive column twiddles: W^12, W^20, W^24, W^28")
    em.o(f"{T} w12_re, w12_im;")
    em.emit_cmul("w12_re", "w12_im", "b4_re", "b4_im", "b8_re", "b8_im", 'fwd')
    em.o(f"{T} w20_re, w20_im;")
    em.emit_cmul("w20_re", "w20_im", "b4_re", "b4_im", "b16_re", "b16_im", 'fwd')
    em.o(f"{T} w24_re, w24_im;")
    em.emit_cmul("w24_re", "w24_im", "b8_re", "b8_im", "b16_re", "b16_im", 'fwd')
    em.o(f"{T} w28_re, w28_im;")
    em.emit_cmul("w28_re", "w28_im", "w12_re", "w12_im", "b16_re", "b16_im", 'fwd')
    em.b()

    # Also derive W^3 = W^1 * W^2 (needed for sub-FFT n2=3)
    em.o(f"{T} w3_re, w3_im;")
    em.emit_cmul("w3_re", "w3_im", "b1_re", "b1_im", "b2_re", "b2_im", 'fwd')
    em.b()

    # Column twiddle lookup: W^{4*n1} for n1=0..7
    # n1=0: 1, n1=1: b4, n1=2: b8, n1=3: w12, n1=4: b16, n1=5: w20, n1=6: w24, n1=7: w28
    col_tw = {0: None, 1: 'b4', 2: 'b8', 3: 'w12', 4: 'b16', 5: 'w20', 6: 'w24', 7: 'w28'}
    # Row twiddle: W^{n2} for n2=0..3
    # n2=0: 1, n2=1: b1, n2=2: b2, n2=3: w3
    row_tw = {0: None, 1: 'b1', 2: 'b2', 3: 'w3'}

    # ── PASS 1: N2 sub-FFTs with log3-derived twiddles ──
    for n2 in range(N2):
        is_last = (n2 == last_n2)
        em.c(f"sub-FFT n2={n2} (log3)")

        # For sub-FFT n2, each input n1 needs twiddle W^{4*n1+n2} = W^{4*n1} * W^{n2}
        for n1 in range(N1):
            n = N2 * n1 + n2
            em.emit_load(f"x{n1}", n, k_expr)

            if n == 0:
                pass  # no twiddle for element 0
            else:
                # Determine the twiddle: W^{4*n1+n2} = col_tw[n1] * row_tw[n2]
                ct = col_tw[n1]
                rt = row_tw[n2]

                if ct is None and rt is None:
                    pass  # W^0 = 1, no twiddle (shouldn't happen for n>0)
                elif ct is None:
                    # W^{n2} only (n1=0, n2>0)
                    em.emit_cmul_inplace(f"x{n1}", f"{rt}_re", f"{rt}_im", d)
                elif rt is None:
                    # W^{4*n1} only (n2=0, n1>0)
                    em.emit_cmul_inplace(f"x{n1}", f"{ct}_re", f"{ct}_im", d)
                else:
                    # W^{4*n1+n2} = col * row, derive in temp then apply
                    em.o(f"{{ {T} tw_r, tw_i;")
                    em.emit_cmul("tw_r", "tw_i", f"{ct}_re", f"{ct}_im", f"{rt}_re", f"{rt}_im", 'fwd')
                    em.emit_cmul_inplace(f"x{n1}", "tw_r", "tw_i", d)
                    em.o(f"}}")

        em.b()
        em.emit_radix8(xv8, d, f"radix-8 n2={n2}")
        em.b()

        # Spill/fuse strategy (identical to flat)
        if is_last:
            em.c(f"FUSED: save x0..x{nfuse-1} to s-regs, spill x{nfuse}..x{N1-1}")
            for k1 in range(nfuse):
                em.o(f"s{k1}_re = x{k1}_re; s{k1}_im = x{k1}_im;")
            for k1 in range(nfuse, N1):
                em.emit_spill(f"x{k1}", n2 * N1 + k1)
        else:
            for k1 in range(N1):
                em.emit_spill(f"x{k1}", n2 * N1 + k1)
        em.b()

    # ── PASS 2: identical to flat (internal twiddles are constants) ──
    em.c(f"PASS 2 — W32 twiddle broadcasts deferred to free regs during PASS 1")
    if isa.name != 'scalar' and itw_set:
        set1 = '_mm256_set1_pd' if isa.name == 'avx2' else '_mm512_set1_pd'
        for (e_tw, tN) in sorted(itw_set):
            label = wN_label(e_tw, tN)
            em.o(f"const {T} tw_{label}_re = {set1}({label}_re);")
            em.o(f"const {T} tw_{label}_im = {set1}({label}_im);")
    em.b()
    for k1 in range(N1):
        em.c(f"column k1={k1}")

        if k1 < nfuse:
            for n2 in range(last_n2):
                em.emit_reload(f"x{n2}", n2 * N1 + k1)
            em.o(f"x{last_n2}_re = s{k1}_re; x{last_n2}_im = s{k1}_im;")
        else:
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

        for k2 in range(N2):
            em.emit_store(f"x{k2}", k1 + N1 * k2, k_expr)
        em.b()


def emit_dif_tw_flat_kernel(em, d, nfuse, itw_set, k_expr="k"):
    """Emit one DIF twiddled flat kernel (fwd or bwd)."""
    fwd = (d == 'fwd')
    last_n2 = N2 - 1
    xv8 = [f"x{i}" for i in range(N1)]
    xv4 = [f"x{i}" for i in range(N2)]

    # PASS 1: load (no twiddle) → radix-8 → spill
    for n2 in range(N2):
        is_last = (n2 == last_n2)
        em.c(f"sub-FFT n2={n2}")
        for n1 in range(N1):
            n = N2 * n1 + n2
            em.emit_load(f"x{n1}", n, k_expr)
            # DIF: NO external twiddle on input
        em.b()
        em.emit_radix8(xv8, d, f"radix-8 n2={n2}")
        em.b()

        if is_last:
            em.c(f"FUSED: save x0..x{nfuse-1} to locals, spill x{nfuse}..x{N1-1}")
            for k1 in range(nfuse):
                em.o(f"s{k1}_re = x{k1}_re; s{k1}_im = x{k1}_im;")
            for k1 in range(nfuse, N1):
                em.emit_spill(f"x{k1}", n2 * N1 + k1)
        else:
            for k1 in range(N1):
                em.emit_spill(f"x{k1}", n2 * N1 + k1)
        em.b()

    # PASS 2: reload → internal twiddle → radix-4 → external twiddle → store
    em.c(f"PASS 2 — W32 twiddle broadcasts deferred to free regs during PASS 1")
    if em.isa.name != 'scalar' and itw_set:
        set1 = '_mm256_set1_pd' if em.isa.name == 'avx2' else '_mm512_set1_pd'
        T = em.isa.reg_type
        for (e, tN) in sorted(itw_set):
            label = wN_label(e, tN)
            em.o(f"const {T} tw_{label}_re = {set1}({label}_re);")
            em.o(f"const {T} tw_{label}_im = {set1}({label}_im);")
    em.b()
    for k1 in range(N1):
        em.c(f"column k1={k1}")
        if k1 < nfuse:
            for n2 in range(last_n2):
                em.emit_reload(f"x{n2}", n2 * N1 + k1)
            em.o(f"x{last_n2}_re = s{k1}_re; x{last_n2}_im = s{k1}_im;")
        else:
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

        # DIF: spill butterfly outputs so PASS 2b can reload and apply
        # external twiddles without holding all 8×2 butterfly regs live
        # through the ext_tw loads. Mirrors DIT's split-phase structure.
        for k2 in range(N2):
            em.emit_spill(f"x{k2}", k1 + N1 * k2)
        em.b()

    # PASS 2b: reload → ext_tw → store (separate phase, like DIT PASS 2)
    em.c(f"PASS 2b — ext_tw + store")
    for k1 in range(N1):
        for k2 in range(N2):
            em.emit_reload(f"x{k2}", k1 + N1 * k2)
        em.b()
        for k2 in range(N2):
            m = k1 + N1 * k2
            if m > 0:
                em.emit_ext_twiddle_dif(f"x{k2}", m - 1, d, k_expr)
        em.b()
        for k2 in range(N2):
            em.emit_store(f"x{k2}", k1 + N1 * k2, k_expr)
        em.b()


# ═══════════════════════════════════════════════════════════════
# KERNEL EMITTER — Notw (N1, no external twiddles)
# ═══════════════════════════════════════════════════════════════

def emit_notw_kernel(em, d, nfuse, itw_set, k_expr="k"):
    """Emit one N1 (notw) kernel — same as DIT but no external twiddles."""
    fwd = (d == 'fwd')
    last_n2 = N2 - 1
    xv8 = [f"x{i}" for i in range(N1)]
    xv4 = [f"x{i}" for i in range(N2)]

    # PASS 1: load (no twiddle) → radix-8 → spill
    for n2 in range(N2):
        is_last = (n2 == last_n2)
        em.c(f"sub-FFT n2={n2}")
        for n1 in range(N1):
            n = N2 * n1 + n2
            em.emit_load(f"x{n1}", n, k_expr)
        em.b()
        em.emit_radix8(xv8, d, f"radix-8 n2={n2}")
        em.b()

        if is_last:
            em.c(f"FUSED: save x0..x{nfuse-1} to s-regs, spill x{nfuse}..x{N1-1}")
            for k1 in range(nfuse):
                em.o(f"s{k1}_re = x{k1}_re; s{k1}_im = x{k1}_im;")
            for k1 in range(nfuse, N1):
                em.emit_spill(f"x{k1}", n2 * N1 + k1)
        else:
            for k1 in range(N1):
                em.emit_spill(f"x{k1}", n2 * N1 + k1)
        em.b()

    # PASS 2: reload → internal twiddle → radix-4 → store (no ext twiddle)
    em.c(f"PASS 2")
    em.b()
    for k1 in range(N1):
        em.c(f"column k1={k1}")
        if k1 < nfuse:
            for n2 in range(last_n2):
                em.emit_reload(f"x{n2}", n2 * N1 + k1)
            em.o(f"x{last_n2}_re = s{k1}_re; x{last_n2}_im = s{k1}_im;")
        else:
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

        for k2 in range(N2):
            em.emit_store(f"x{k2}", k1 + N1 * k2, k_expr)
        em.b()


# ═══════════════════════════════════════════════════════════════
# KERNEL EMITTER — Ladder twiddle (AVX-512 only)
#
# Instead of flat tw_re[31*K], uses 5 base twiddle pairs and
# derives the other 26 via complex multiply chains.
# Reduces memory loads from 31 to 5 per sub-FFT.
# ═══════════════════════════════════════════════════════════════

# Row twiddle derivation chain for 8×4 decomposition
# (n1, derive_from, keep_as_b12)
# n1=0: W^0 = 1 (no twiddle)
# n1=1: W^(n2) = b4^n2 (base)
# n1=2: W^(2*n2) = b8^n2 (base)
# n1=3: W^(3*n2) = b4*b8 → derive, keep as b12
# n1=4: W^(4*n2) = b16^n2 (base)
# n1=5: W^(5*n2) = b4*b16 → derive
# n1=6: W^(6*n2) = b8*b16 → derive
# n1=7: W^(7*n2) = b12*b16 → derive
ROW_CHAIN = [
    (0, None, False), (1, None, False), (2, None, False),
    (3, ('b4', 'b8'), True), (4, None, False),
    (5, ('b4', 'b16'), False), (6, ('b8', 'b16'), False), (7, ('b12', 'b16'), False),
]
BASE_MAP = {'b1': 0, 'b2': 1, 'b4': 2, 'b8': 3, 'b16': 4}
COL_TW = {0: None, 1: 'b1', 2: 'b2', 3: 'b3'}
NFUSE_LADDER = {'avx512': 4, 'avx2': 2, 'scalar': 4}


def emit_ladder_pipeline(em, pipe, k_expr, spill_base, d, itw_set, nfuse=None,
                         base_map_override=None, tw_buf_re='base_tw_re',
                         tw_buf_im='base_tw_im', tw_stride='K'):
    """Emit one pipeline of the ladder kernel (pass 1 + pass 2). ISA-agnostic.

    base_map_override: if provided, use these indices into the twiddle table
        instead of {'b1':0,'b2':1,'b4':2,'b8':3,'b16':4}. Use case: when the
        twiddle table layout is ct_t1_dit's (31 rows at W_re[n*me+k]), the
        powers of 2 live at rows {0, 1, 3, 7, 15}.
    tw_buf_re, tw_buf_im: name of the twiddle buffer pointers. Default is
        'base_tw_re'/'base_tw_im' (original ladder kernel); override to
        'W_re'/'W_im' for ct_t1_ladder variant.
    tw_stride: name of the twiddle-stride variable. Default 'K' (original);
        'me' for ct_t1_ladder variant.
    """
    xv8 = [f"x{i}" for i in range(N1)]
    xv4 = [f"x{i}" for i in range(N2)]
    if nfuse is None:
        nfuse = NFUSE_LADDER.get(em.isa.name, 4)
    last_n2 = N2 - 1
    T = em.isa.reg_type

    base_map = base_map_override if base_map_override is not None else BASE_MAP

    # Load 5 base twiddle pairs
    em.c(f"{pipe}Load 5 base twiddle pairs")
    for name, idx in sorted(base_map.items(), key=lambda x: x[1]):
        if em.isa.name == 'scalar':
            em.o(f"const {T} {pipe}{name}_re = {tw_buf_re}[{idx}*{tw_stride}+{k_expr}];")
            em.o(f"const {T} {pipe}{name}_im = {tw_buf_im}[{idx}*{tw_stride}+{k_expr}];")
        else:
            em.o(f"const {T} {pipe}{name}_re = LD(&{tw_buf_re}[{idx}*{tw_stride}+{k_expr}]);")
            em.o(f"const {T} {pipe}{name}_im = LD(&{tw_buf_im}[{idx}*{tw_stride}+{k_expr}]);")
    em.b()

    # Derive b3 = b1*b2
    em.c(f"{pipe}Derive b3 = b1*b2")
    em.o(f"{T} {pipe}b3_re, {pipe}b3_im;")
    em.emit_cmul(f"{pipe}b3_re", f"{pipe}b3_im",
                 f"{pipe}b1_re", f"{pipe}b1_im",
                 f"{pipe}b2_re", f"{pipe}b2_im", 'fwd')
    em.b()

    # Derived row twiddle scratch
    em.c(f"{pipe}Derived row twiddle scratch")
    em.o(f"{T} {pipe}r3_re,{pipe}r3_im, {pipe}r5_re,{pipe}r5_im, {pipe}r6_re,{pipe}r6_im, {pipe}r7_re,{pipe}r7_im;")
    em.o(f"{T} {pipe}b12_re, {pipe}b12_im;")
    em.b()

    # PASS 1: sub-FFTs with ladder twiddles
    for n2 in range(N2):
        is_last = (n2 == last_n2)
        em.c(f"{pipe}sub-FFT n2={n2}")
        for n1, derive, keep in ROW_CHAIN:
            n = N2 * n1 + n2
            em.emit_load(f"x{n1}", n, k_expr)
            if n1 > 0:
                if derive is None:
                    base_name = {1: 'b4', 2: 'b8', 4: 'b16'}[n1]
                    wr, wi = f"{pipe}{base_name}_re", f"{pipe}{base_name}_im"
                else:
                    a_name, b_name = derive
                    tw_name = f"r{n1}"
                    em.emit_cmul(f"{pipe}{tw_name}_re", f"{pipe}{tw_name}_im",
                                 f"{pipe}{a_name}_re", f"{pipe}{a_name}_im",
                                 f"{pipe}{b_name}_re", f"{pipe}{b_name}_im", 'fwd')
                    wr, wi = f"{pipe}{tw_name}_re", f"{pipe}{tw_name}_im"
                    if keep:
                        em.o(f"{pipe}b12_re = {wr}; {pipe}b12_im = {wi};")
                em.emit_cmul_inplace(f"x{n1}", wr, wi, d)
        em.b()
        em.emit_radix8(xv8, d, f"{pipe}radix-8 n2={n2}")
        em.b()

        # Column twiddle (applied to entire sub-FFT output)
        col = COL_TW[n2]
        if col is not None:
            em.c(f"{pipe}col twiddle {col}")
            wr, wi = f"{pipe}{col}_re", f"{pipe}{col}_im"
            for k1 in range(N1):
                em.emit_cmul_inplace(f"x{k1}", wr, wi, d)
            em.b()

        # Spill/fuse
        if is_last:
            em.c(f"{pipe}FUSED: save x0..x{nfuse-1} in s-regs, spill x{nfuse}..x{N1-1}")
            for k1 in range(nfuse):
                em.o(f"s{k1}_re = x{k1}_re; s{k1}_im = x{k1}_im;")
            for k1 in range(nfuse, N1):
                em.emit_spill(f"x{k1}", spill_base + n2 * N1 + k1)
        else:
            for k1 in range(N1):
                em.emit_spill(f"x{k1}", spill_base + n2 * N1 + k1)
        em.b()

    # PASS 2: same as flat DIT (internal twiddles + radix-4)
    em.c(f"{pipe}PASS 2 [{d}]")
    em.b()
    for k1 in range(N1):
        em.c(f"{pipe}column k1={k1}")
        if k1 < nfuse:
            for n2 in range(last_n2):
                em.emit_reload(f"x{n2}", spill_base + n2 * N1 + k1)
            em.o(f"x{last_n2}_re = s{k1}_re; x{last_n2}_im = s{k1}_im;")
        else:
            for n2 in range(N2):
                em.emit_reload(f"x{n2}", spill_base + n2 * N1 + k1)
        em.b()
        if k1 > 0:
            for n2 in range(1, N2):
                e = (n2 * k1) % N
                em.emit_twiddle(f"x{n2}", f"x{n2}", e, N, d)
            em.b()
        em.emit_radix4(xv4, d, f"{pipe}radix-4 k1={k1}")
        em.b()
        for k2 in range(N2):
            em.emit_store(f"x{k2}", k1 + N1 * k2, k_expr)
        em.b()


def emit_ladder_dif_pipeline(em, pipe, k_expr, spill_base, d, itw_set, nfuse=None,
                             base_map_override=None, tw_buf_re='base_tw_re',
                             tw_buf_im='base_tw_im', tw_stride='K'):
    """Emit DIF ladder pipeline: butterfly first, then ladder-derived twiddle on outputs."""
    xv8 = [f"x{i}" for i in range(N1)]
    xv4 = [f"x{i}" for i in range(N2)]
    if nfuse is None:
        nfuse = NFUSE_LADDER.get(em.isa.name, 4)
    last_n2 = N2 - 1
    T = em.isa.reg_type

    base_map = base_map_override if base_map_override is not None else BASE_MAP

    # Load 5 base twiddle pairs
    em.c(f"{pipe}Load 5 base twiddle pairs")
    for name, idx in sorted(base_map.items(), key=lambda x: x[1]):
        if em.isa.name == 'scalar':
            em.o(f"const {T} {pipe}{name}_re = {tw_buf_re}[{idx}*{tw_stride}+{k_expr}];")
            em.o(f"const {T} {pipe}{name}_im = {tw_buf_im}[{idx}*{tw_stride}+{k_expr}];")
        else:
            em.o(f"const {T} {pipe}{name}_re = LD(&{tw_buf_re}[{idx}*{tw_stride}+{k_expr}]);")
            em.o(f"const {T} {pipe}{name}_im = LD(&{tw_buf_im}[{idx}*{tw_stride}+{k_expr}]);")
    em.b()

    # Pre-derive all needed twiddles for output application
    # b3=b1*b2, w5=b1*b4, w6=b2*b4, w7=b3*b4, b24=b8*b16
    em.c(f"{pipe}Derive output twiddles")
    em.o(f"{T} {pipe}b3_re,{pipe}b3_im;")
    em.emit_cmul(f"{pipe}b3_re", f"{pipe}b3_im",
                 f"{pipe}b1_re", f"{pipe}b1_im",
                 f"{pipe}b2_re", f"{pipe}b2_im", 'fwd')
    em.o(f"{T} {pipe}w5_re,{pipe}w5_im;")
    em.emit_cmul(f"{pipe}w5_re", f"{pipe}w5_im",
                 f"{pipe}b1_re", f"{pipe}b1_im",
                 f"{pipe}b4_re", f"{pipe}b4_im", 'fwd')
    em.o(f"{T} {pipe}w6_re,{pipe}w6_im;")
    em.emit_cmul(f"{pipe}w6_re", f"{pipe}w6_im",
                 f"{pipe}b2_re", f"{pipe}b2_im",
                 f"{pipe}b4_re", f"{pipe}b4_im", 'fwd')
    em.o(f"{T} {pipe}w7_re,{pipe}w7_im;")
    em.emit_cmul(f"{pipe}w7_re", f"{pipe}w7_im",
                 f"{pipe}b3_re", f"{pipe}b3_im",
                 f"{pipe}b4_re", f"{pipe}b4_im", 'fwd')
    em.o(f"{T} {pipe}b24_re,{pipe}b24_im;")
    em.emit_cmul(f"{pipe}b24_re", f"{pipe}b24_im",
                 f"{pipe}b8_re", f"{pipe}b8_im",
                 f"{pipe}b16_re", f"{pipe}b16_im", 'fwd')
    em.o(f"{T} {pipe}tw_re,{pipe}tw_im;")  # scratch for combined twiddle
    em.b()

    # Row twiddle lookup: W^k1 for k1=0..7
    # k1=0: identity, k1=1: b1, k1=2: b2, k1=3: b3, k1=4: b4,
    # k1=5: w5, k1=6: w6, k1=7: w7
    row_tw = {0: None, 1: 'b1', 2: 'b2', 3: 'b3', 4: 'b4',
              5: 'w5', 6: 'w6', 7: 'w7'}
    # Column twiddle: W^(8*k2) for k2=0..3
    # k2=0: identity, k2=1: b8, k2=2: b16, k2=3: b24
    col_tw = {0: None, 1: 'b8', 2: 'b16', 3: 'b24'}

    # PASS 1: load raw (no external twiddle) → radix-8 → spill
    for n2 in range(N2):
        is_last = (n2 == last_n2)
        em.c(f"{pipe}sub-FFT n2={n2}")
        for n1 in range(N1):
            n = N2 * n1 + n2
            em.emit_load(f"x{n1}", n, k_expr)
        em.b()
        em.emit_radix8(xv8, d, f"{pipe}radix-8 n2={n2}")
        em.b()
        if is_last:
            em.c(f"{pipe}FUSED: save x0..x{nfuse-1} in s-regs, spill x{nfuse}..x{N1-1}")
            for k1 in range(nfuse):
                em.o(f"s{k1}_re = x{k1}_re; s{k1}_im = x{k1}_im;")
            for k1 in range(nfuse, N1):
                em.emit_spill(f"x{k1}", spill_base + n2 * N1 + k1)
        else:
            for k1 in range(N1):
                em.emit_spill(f"x{k1}", spill_base + n2 * N1 + k1)
        em.b()

    # PASS 2: reload → internal twiddle → radix-4 → output twiddle → store
    em.c(f"{pipe}PASS 2 [{d}]")
    em.b()
    for k1 in range(N1):
        em.c(f"{pipe}column k1={k1}")
        if k1 < nfuse:
            for n2 in range(last_n2):
                em.emit_reload(f"x{n2}", spill_base + n2 * N1 + k1)
            em.o(f"x{last_n2}_re = s{k1}_re; x{last_n2}_im = s{k1}_im;")
        else:
            for n2 in range(N2):
                em.emit_reload(f"x{n2}", spill_base + n2 * N1 + k1)
        em.b()
        if k1 > 0:
            for n2 in range(1, N2):
                e = (n2 * k1) % N
                em.emit_twiddle(f"x{n2}", f"x{n2}", e, N, d)
            em.b()
        em.emit_radix4(xv4, d, f"{pipe}radix-4 k1={k1}")
        em.b()

        # DIF: apply ladder-derived external twiddle to outputs
        # Output m = k1 + 8*k2. W^m = W^k1 * W^(8*k2)
        em.c(f"{pipe}DIF output twiddle (ladder)")
        rtw = row_tw[k1]  # W^k1 name or None
        for k2 in range(N2):
            m = k1 + N1 * k2
            if m == 0:
                continue
            ctw = col_tw[k2]  # W^(8*k2) name or None
            if rtw is None and ctw is None:
                continue  # m=0, already skipped
            elif rtw is not None and ctw is None:
                # Just row twiddle
                em.emit_cmul_inplace(f"x{k2}", f"{pipe}{rtw}_re", f"{pipe}{rtw}_im", d)
            elif rtw is None and ctw is not None:
                # Just column twiddle (k1=0, k2>0)
                em.emit_cmul_inplace(f"x{k2}", f"{pipe}{ctw}_re", f"{pipe}{ctw}_im", d)
            else:
                # Both: derive combined = rtw * ctw, then apply
                em.emit_cmul(f"{pipe}tw_re", f"{pipe}tw_im",
                             f"{pipe}{rtw}_re", f"{pipe}{rtw}_im",
                             f"{pipe}{ctw}_re", f"{pipe}{ctw}_im", 'fwd')
                em.emit_cmul_inplace(f"x{k2}", f"{pipe}tw_re", f"{pipe}tw_im", d)
        em.b()

        for k2 in range(N2):
            em.emit_store(f"x{k2}", k1 + N1 * k2, k_expr)
        em.b()


# ═══════════════════════════════════════════════════════════════
# FILE EMITTER — assembles header file for one ISA
# ═══════════════════════════════════════════════════════════════

def emit_twiddle_constants(em, itw_set):
    """Emit W32 constant definitions (guarded, shared across files)."""
    by_tN = {}
    for (e, tN) in sorted(itw_set):
        by_tN.setdefault(tN, []).append(e)
    for tN in sorted(by_tN):
        g = f"FFT_W{tN}_TWIDDLES_DEFINED"
        em.L.append(f"#ifndef {g}")
        em.L.append(f"#define {g}")
        for e in sorted(by_tN[tN]):
            wr, wi = wN(e, tN)
            l = wN_label(e, tN)
            em.L.append(f"static const double {l}_re = {wr:.20e};")
            em.L.append(f"static const double {l}_im = {wi:.20e};")
        em.L.append(f"#endif")
        em.L.append("")


def insert_stats_into_header(lines, stats):
    """Insert operation count table into the file header comment block.
    
    Finds the ' */' closing the header comment and inserts the stats
    table just before it.
    """
    # Build the stats table
    table = []
    table.append(" *")
    table.append(" * ── Operation counts per k-step ──")
    table.append(" *")
    table.append(f" *   {'kernel':<20s} {'add':>5s} {'sub':>5s} {'mul':>5s} {'neg':>5s}"
                 f" {'fma':>5s} {'fms':>5s} | {'arith':>5s} {'flops':>5s}"
                 f" | {'ld':>3s} {'st':>3s} {'sp':>3s} {'rl':>3s} {'mem':>4s}")
    sep = '─' * 20
    s5 = '─' * 5
    s3 = '─' * 3
    s4 = '─' * 4
    table.append(f" *   {sep} {s5} {s5} {s5} {s5}"
                 f" {s5} {s5} + {s5} {s5}"
                 f" + {s3} {s3} {s3} {s3} {s4}")
    for k in sorted(stats.keys()):
        s = stats[k]
        table.append(f" *   {k:<20s} {s['add']:5d} {s['sub']:5d} {s['mul']:5d} {s['neg']:5d}"
                     f" {s['fma']:5d} {s['fms']:5d} | {s['total_arith']:5d} {s['flops']:5d}"
                     f" | {s['load']:3d} {s['store']:3d} {s['spill']:3d} {s['reload']:3d} {s['total_mem']:4d}")
    table.append(" *")

    # Find the first ' */' that closes the header comment
    for i, line in enumerate(lines):
        if line.strip() == '*/':
            for j, tl in enumerate(table):
                lines.insert(i + j, tl)
            return
    # Fallback: couldn't find header end, prepend as standalone comment
    lines.insert(0, "/* " + " | ".join(f"{k}:{v}" for k, v in stats.items()) + " */")


def emit_dit_tw_flat_file(isa, itw_set):
    """Emit complete DIT tw flat header for one ISA. Returns (lines, stats)."""
    nfuse = isa.nfuse_tw
    T = isa.reg_type
    em = Emitter(isa)

    # File header
    em.L.append(f"/**")
    em.L.append(f" * @file fft_radix32_{isa.name}_tw.h")
    em.L.append(f" * @brief DFT-32 {isa.name.upper()} twiddled codelet — flat twiddles, NFUSE={nfuse}")
    em.L.append(f" *")
    em.L.append(f" * 8x4 decomposition, U=1, fwd + bwd")
    em.L.append(f" * Generated by gen_radix32.py")
    em.L.append(f" */")
    em.L.append(f"")

    guard = f"FFT_RADIX32_{isa.name.upper()}_TW_H"
    em.L.append(f"#ifndef {guard}")
    em.L.append(f"#define {guard}")
    em.L.append(f"")

    if isa.name != 'scalar':
        em.L.append(f"#include <immintrin.h>")
        em.L.append(f"")

    # Twiddle constants
    emit_twiddle_constants(em, itw_set)

    # LD/ST macros (SIMD only)
    if isa.name == 'avx2':
        em.L.append("#ifndef R32A_LD")
        em.L.append("#define R32A_LD(p) _mm256_loadu_pd(p)")
        em.L.append("#endif")
        em.L.append("#ifndef R32A_ST")
        em.L.append("#define R32A_ST(p,v) _mm256_storeu_pd((p),(v))")
        em.L.append("#endif")
        em.L.append("#define LD R32A_LD")
        em.L.append("#define ST R32A_ST")
        em.L.append("")
    elif isa.name == 'avx512':
        em.L.append("#ifndef R32L_LD")
        em.L.append("#define R32L_LD(p) _mm512_loadu_pd(p)")
        em.L.append("#endif")
        em.L.append("#ifndef R32L_ST")
        em.L.append("#define R32L_ST(p,v) _mm512_storeu_pd((p),(v))")
        em.L.append("#endif")
        em.L.append("#define LD R32L_LD")
        em.L.append("#define ST R32L_ST")
        em.L.append("")

    em.L.append(f"/* === FLAT U=1 ({isa.name.upper()}, NFUSE={nfuse}) === */")
    em.L.append("")

    stats = {}
    for d in ['fwd', 'bwd']:
        em.reset_counters()

        # Function signature
        if isa.target_attr:
            em.L.append(f"static {isa.target_attr} void")
        else:
            em.L.append(f"static void")
        em.L.append(f"radix32_tw_flat_dit_kernel_{d}_{isa.name}(")
        em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
        em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
        em.L.append(f"    const double * __restrict__ tw_re, const double * __restrict__ tw_im,")
        em.L.append(f"    size_t K)")
        em.L.append(f"{{")
        em.ind = 1

        # Constants
        if isa.name != 'scalar':
            em.o(f"const {T} sign_flip = {'_mm256_set1_pd' if isa.name == 'avx2' else '_mm512_set1_pd'}(-0.0);")
            em.o(f"const {T} sqrt2_inv = {'_mm256_set1_pd' if isa.name == 'avx2' else '_mm512_set1_pd'}(0.70710678118654752440);")
        else:
            pass  # scalar uses #define SQRT2_INV at file scope
        em.b()

        # Spill buffer
        spill_total = N * isa.spill_mul
        if isa.name == 'scalar':
            em.o(f"double spill_re[{N}], spill_im[{N}];")
        else:
            em.o(f"{isa.align_attr} double spill_re[{spill_total}];")
            em.o(f"{isa.align_attr} double spill_im[{spill_total}];")
        em.b()

        # Working registers
        em.o(f"{T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
        em.o(f"{T} x4_re,x4_im,x5_re,x5_im,x6_re,x6_im,x7_re,x7_im;")
        # Saved registers for NFUSE
        slist = ",".join(f"s{i}_re,s{i}_im" for i in range(nfuse))
        em.o(f"{T} {slist};")
        em.b()

        # Note: W32 twiddle broadcasts are deferred to PASS 2 (inside kernel)
        # to free registers during PASS 1 — gave -21.5% on R=32 t1_dit (AVX2)

        # Scalar needs SQRT2_INV define — handled at file scope
        if isa.name == 'scalar' and itw_set:
            em.c(f"Internal W32 constants are #defined at file scope")
            em.b()

        # K loop
        if isa.name == 'scalar':
            em.o(f"for (size_t k = 0; k < K; k++) {{")
        else:
            em.o(f"for (size_t k = 0; k < K; k += {isa.k_step}) {{")
        em.ind += 1

        # Core kernel
        emit_dit_tw_flat_kernel(em, d, nfuse, itw_set)

        em.ind -= 1
        em.o("}")
        em.L.append("}")
        em.L.append("")

        stats[d] = em.get_stats()

    # Cleanup macros
    if isa.name != 'scalar':
        em.L.append("#undef LD")
        em.L.append("#undef ST")
        em.L.append("")

    em.L.append(f"#endif /* {guard} */")
    insert_stats_into_header(em.L, stats)
    return em.L, stats


# ═══════════════════════════════════════════════════════════════
# FILE EMITTER — DIF tw flat
# ═══════════════════════════════════════════════════════════════

def emit_dif_tw_flat_file(isa, itw_set):
    """Emit complete DIF tw flat header for one ISA."""
    nfuse = isa.nfuse_tw
    T = isa.reg_type
    em = Emitter(isa)

    guard = f"FFT_RADIX32_{isa.name.upper()}_DIF_TW_H"
    em.L.append(f"/**")
    em.L.append(f" * @file fft_radix32_{isa.name}_dif_tw.h")
    em.L.append(f" * @brief DFT-32 {isa.name.upper()} DIF twiddled codelet — flat twiddles, NFUSE={nfuse}")
    em.L.append(f" *")
    em.L.append(f" * DIF: external twiddle on OUTPUT (after butterfly)")
    em.L.append(f" * 8x4 decomposition, fwd + bwd")
    em.L.append(f" * Generated by gen_radix32.py")
    em.L.append(f" */")
    em.L.append(f"")
    em.L.append(f"#ifndef {guard}")
    em.L.append(f"#define {guard}")
    em.L.append(f"")

    if isa.name != 'scalar':
        em.L.append(f"#include <immintrin.h>")
        em.L.append(f"")

    emit_twiddle_constants(em, itw_set)

    if isa.name == 'avx2':
        em.L.append("#ifndef R32DA_LD"); em.L.append("#define R32DA_LD(p) _mm256_loadu_pd(p)"); em.L.append("#endif")
        em.L.append("#ifndef R32DA_ST"); em.L.append("#define R32DA_ST(p,v) _mm256_storeu_pd((p),(v))"); em.L.append("#endif")
        em.L.append("#define LD R32DA_LD"); em.L.append("#define ST R32DA_ST"); em.L.append("")
    elif isa.name == 'avx512':
        em.L.append("#ifndef R32DL_LD"); em.L.append("#define R32DL_LD(p) _mm512_loadu_pd(p)"); em.L.append("#endif")
        em.L.append("#ifndef R32DL_ST"); em.L.append("#define R32DL_ST(p,v) _mm512_storeu_pd((p),(v))"); em.L.append("#endif")
        em.L.append("#define LD R32DL_LD"); em.L.append("#define ST R32DL_ST"); em.L.append("")

    em.L.append(f"/* === DIF FLAT ({isa.name.upper()}, NFUSE={nfuse}) === */")
    em.L.append("")

    stats = {}
    for d in ['fwd', 'bwd']:
        em.reset_counters()
        if isa.target_attr:
            em.L.append(f"static {isa.target_attr} void")
        else:
            em.L.append(f"static void")
        em.L.append(f"radix32_tw_flat_dif_kernel_{d}_{isa.name}(")
        em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
        em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
        em.L.append(f"    const double * __restrict__ tw_re, const double * __restrict__ tw_im,")
        em.L.append(f"    size_t K)")
        em.L.append(f"{{")
        em.ind = 1

        if isa.name != 'scalar':
            set1 = '_mm256_set1_pd' if isa.name == 'avx2' else '_mm512_set1_pd'
            em.o(f"const {T} sign_flip = {set1}(-0.0);")
            em.o(f"const {T} sqrt2_inv = {set1}(0.70710678118654752440);")
            em.b()
        else:
            em.o(f"const double sqrt2_inv = 0.70710678118654752440;")
            em.b()

        spill_total = N * isa.spill_mul
        if isa.name == 'scalar':
            em.o(f"double spill_re[{N}], spill_im[{N}];")
        else:
            em.o(f"{isa.align_attr} double spill_re[{spill_total}];")
            em.o(f"{isa.align_attr} double spill_im[{spill_total}];")
        em.b()
        em.o(f"{T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
        em.o(f"{T} x4_re,x4_im,x5_re,x5_im,x6_re,x6_im,x7_re,x7_im;")
        slist = ",".join(f"s{i}_re,s{i}_im" for i in range(nfuse))
        em.o(f"{T} {slist};")
        em.b()

        if isa.name != 'scalar' and itw_set:
            em.c(f"Hoisted internal W32 broadcasts")
            set1 = '_mm256_set1_pd' if isa.name == 'avx2' else '_mm512_set1_pd'
            for (e, tN) in sorted(itw_set):
                label = wN_label(e, tN)
                em.o(f"const {T} tw_{label}_re = {set1}({label}_re);")
                em.o(f"const {T} tw_{label}_im = {set1}({label}_im);")
            em.b()

        if isa.name == 'scalar' and itw_set:
            em.c(f"Internal W32 constants are #defined at file scope")
            em.b()

        if isa.name == 'scalar':
            em.o(f"for (size_t k = 0; k < K; k++) {{")
        else:
            em.o(f"for (size_t k = 0; k < K; k += {isa.k_step}) {{")
        em.ind += 1
        emit_dif_tw_flat_kernel(em, d, nfuse, itw_set)
        em.ind -= 1
        em.o("}")
        em.L.append("}")
        em.L.append("")
        stats[d] = em.get_stats()

    if isa.name != 'scalar':
        em.L.append("#undef LD"); em.L.append("#undef ST"); em.L.append("")
    em.L.append(f"#endif /* {guard} */")
    insert_stats_into_header(em.L, stats)
    return em.L, stats


# ═══════════════════════════════════════════════════════════════
# FILE EMITTER — Notw (N1)
# ═══════════════════════════════════════════════════════════════

def emit_notw_file(isa, itw_set):
    """Emit complete notw header for one ISA."""
    nfuse = isa.nfuse_notw
    T = isa.reg_type
    em = Emitter(isa)

    guard = f"FFT_RADIX32_{isa.name.upper()}_NOTW_H"
    em.L.append(f"/**")
    em.L.append(f" * @file fft_radix32_{isa.name}_notw.h")
    em.L.append(f" * @brief DFT-32 {isa.name.upper()} notw codelet — NFUSE={nfuse}")
    em.L.append(f" *")
    em.L.append(f" * No external twiddles. Pure DFT-32 butterfly.")
    em.L.append(f" * 8x4 decomposition, fwd + bwd")
    em.L.append(f" * Generated by gen_radix32.py")
    em.L.append(f" */")
    em.L.append(f"")
    em.L.append(f"#ifndef {guard}")
    em.L.append(f"#define {guard}")
    em.L.append(f"")

    if isa.name != 'scalar':
        em.L.append(f"#include <immintrin.h>")
        em.L.append(f"")

    emit_twiddle_constants(em, itw_set)

    # Notw-specific LD/ST macros
    if isa.name == 'avx2':
        em.L.append("#ifndef R32NA_LD"); em.L.append("#define R32NA_LD(p) _mm256_loadu_pd(p)"); em.L.append("#endif")
        em.L.append("#ifndef R32NA_ST"); em.L.append("#define R32NA_ST(p,v) _mm256_storeu_pd((p),(v))"); em.L.append("#endif")
        em.L.append("#define LD R32NA_LD"); em.L.append("#define ST R32NA_ST"); em.L.append("")
    elif isa.name == 'avx512':
        em.L.append("#ifndef R32N5_LD"); em.L.append("#define R32N5_LD(p) _mm512_loadu_pd(p)"); em.L.append("#endif")
        em.L.append("#ifndef R32N5_ST"); em.L.append("#define R32N5_ST(p,v) _mm512_storeu_pd((p),(v))"); em.L.append("#endif")
        em.L.append("#define LD R32N5_LD"); em.L.append("#define ST R32N5_ST"); em.L.append("")

    em.L.append(f"/* === NOTW ({isa.name.upper()}, NFUSE={nfuse}) === */")
    em.L.append("")

    stats = {}
    for d in ['fwd', 'bwd']:
        em.reset_counters()
        if isa.target_attr:
            em.L.append(f"static {isa.target_attr} void")
        else:
            em.L.append(f"static void")
        em.L.append(f"radix32_notw_dit_kernel_{d}_{isa.name}(")
        em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
        em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
        em.L.append(f"    size_t K)")
        em.L.append(f"{{")
        em.ind = 1

        if isa.name != 'scalar':
            set1 = '_mm256_set1_pd' if isa.name == 'avx2' else '_mm512_set1_pd'
            em.o(f"const {T} sign_flip = {set1}(-0.0);")
            em.o(f"const {T} sqrt2_inv = {set1}(0.70710678118654752440);")
            em.b()

        # Spill buffer — always needed (pass 1 spills for n2 < N2-1 regardless of NFUSE)
        spill_total = N * isa.spill_mul
        if isa.name == 'scalar':
            em.o(f"double spill_re[{N}], spill_im[{N}];")
        else:
            em.o(f"{isa.align_attr} double spill_re[{spill_total}];")
            em.o(f"{isa.align_attr} double spill_im[{spill_total}];")
        em.b()

        em.o(f"{T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
        em.o(f"{T} x4_re,x4_im,x5_re,x5_im,x6_re,x6_im,x7_re,x7_im;")
        slist = ",".join(f"s{i}_re,s{i}_im" for i in range(nfuse))
        em.o(f"{T} {slist};")
        em.b()

        if isa.name != 'scalar' and itw_set:
            em.c(f"Hoisted internal W32 broadcasts")
            set1 = '_mm256_set1_pd' if isa.name == 'avx2' else '_mm512_set1_pd'
            for (e, tN) in sorted(itw_set):
                label = wN_label(e, tN)
                em.o(f"const {T} tw_{label}_re = {set1}({label}_re);")
                em.o(f"const {T} tw_{label}_im = {set1}({label}_im);")
            em.b()

        if isa.name == 'scalar' and itw_set:
            em.c(f"Internal W32 constants are #defined at file scope")
            em.b()

        if isa.name == 'scalar':
            em.o(f"for (size_t k = 0; k < K; k++) {{")
        else:
            em.o(f"for (size_t k = 0; k < K; k += {isa.k_step}) {{")
        em.ind += 1
        emit_notw_kernel(em, d, nfuse, itw_set)
        em.ind -= 1
        em.o("}")
        em.L.append("}")
        em.L.append("")
        stats[d] = em.get_stats()

    if isa.name != 'scalar':
        em.L.append("#undef LD"); em.L.append("#undef ST"); em.L.append("")
    em.L.append(f"#endif /* {guard} */")
    insert_stats_into_header(em.L, stats)
    return em.L, stats


# ═══════════════════════════════════════════════════════════════
# FILE EMITTER — Ladder DIT (all ISAs)
# ═══════════════════════════════════════════════════════════════

def _emit_ladder_boilerplate(em, isa, T, itw_set):
    """Emit shared boilerplate for ladder kernels."""
    nfuse = NFUSE_LADDER.get(isa.name, 4)
    set1 = '_mm256_set1_pd' if isa.name == 'avx2' else '_mm512_set1_pd'

    if isa.name != 'scalar':
        em.o(f"const {T} sign_flip = {set1}(-0.0);")
        em.o(f"const {T} sqrt2_inv = {set1}(0.70710678118654752440);")
    else:
        em.o(f"const double sqrt2_inv = 0.70710678118654752440;")
    em.b()

    spill_total = N * isa.spill_mul
    if isa.name == 'scalar':
        em.o(f"double spill_re[{N}], spill_im[{N}];")
    else:
        em.o(f"{isa.align_attr} double spill_re[{spill_total}];")
        em.o(f"{isa.align_attr} double spill_im[{spill_total}];")
    em.b()

    em.o(f"{T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
    em.o(f"{T} x4_re,x4_im,x5_re,x5_im,x6_re,x6_im,x7_re,x7_im;")
    slist = ",".join(f"s{i}_re,s{i}_im" for i in range(nfuse))
    em.o(f"{T} {slist};")
    em.b()

    if isa.name != 'scalar' and itw_set:
        em.c(f"Hoisted internal W32 broadcasts")
        for (e, tN) in sorted(itw_set):
            label = wN_label(e, tN)
            em.o(f"const {T} tw_{label}_re = {set1}({label}_re);")
            em.o(f"const {T} tw_{label}_im = {set1}({label}_im);")
        em.b()
    elif isa.name == 'scalar' and itw_set:
        em.c(f"Internal W32 constants at file scope")
        em.b()


def emit_ladder_file(isa, itw_set):
    """Emit ladder DIT header for any ISA. Returns (lines, stats)."""
    nfuse = NFUSE_LADDER.get(isa.name, 4)
    T = isa.reg_type
    em = Emitter(isa)

    guard = f"FFT_RADIX32_{isa.name.upper()}_TW_LADDER_H"
    em.L.append(f"/**")
    em.L.append(f" * @file fft_radix32_{isa.name}_tw_ladder.h")
    em.L.append(f" * @brief DFT-32 {isa.name.upper()} ladder (log-derived) twiddles, NFUSE={nfuse}")
    em.L.append(f" *")
    em.L.append(f" * 5 base twiddles → 26 derived via cmul. Table 6.2x smaller than flat.")
    em.L.append(f" * Generated by gen_radix32.py")
    em.L.append(f" */")
    em.L.append(f"")
    em.L.append(f"#ifndef {guard}")
    em.L.append(f"#define {guard}")
    em.L.append(f"")

    if isa.name != 'scalar':
        em.L.append(f"#include <immintrin.h>")
        em.L.append(f"")
    else:
        em.L.append(f"#include <stddef.h>")
        em.L.append(f"")

    emit_twiddle_constants(em, itw_set)

    # LD/ST macros
    if isa.name == 'avx2':
        em.L.append("#ifndef R32LA_LD"); em.L.append("#define R32LA_LD(p) _mm256_loadu_pd(p)"); em.L.append("#endif")
        em.L.append("#ifndef R32LA_ST"); em.L.append("#define R32LA_ST(p,v) _mm256_storeu_pd((p),(v))"); em.L.append("#endif")
        em.L.append("#define LD R32LA_LD"); em.L.append("#define ST R32LA_ST"); em.L.append("")
    elif isa.name == 'avx512':
        em.L.append("#ifndef R32L_LD"); em.L.append("#define R32L_LD(p) _mm512_loadu_pd(p)"); em.L.append("#endif")
        em.L.append("#ifndef R32L_ST"); em.L.append("#define R32L_ST(p,v) _mm512_storeu_pd((p),(v))"); em.L.append("#endif")
        em.L.append("#define LD R32L_LD"); em.L.append("#define ST R32L_ST"); em.L.append("")

    if isa.name == 'scalar':
        em.L.append(f"#define SQRT2_INV 0.70710678118654752440")
        em.L.append(f"")

    stats = {}

    # === LADDER U=1 ===
    em.L.append(f"/* === LADDER U=1 ({isa.name.upper()}, NFUSE={nfuse}) === */"); em.L.append("")
    for d in ['fwd', 'bwd']:
        em.reset_counters()
        if isa.target_attr:
            em.L.append(f"static {isa.target_attr} void")
        else:
            em.L.append(f"static void")
        em.L.append(f"radix32_tw_ladder_dit_kernel_{d}_{isa.name}_u1(")
        em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
        em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
        em.L.append(f"    const double * __restrict__ base_tw_re, const double * __restrict__ base_tw_im,")
        em.L.append(f"    size_t K)")
        em.L.append(f"{{")
        em.ind = 1
        _emit_ladder_boilerplate(em, isa, T, itw_set)
        if isa.name == 'scalar':
            em.o(f"for (size_t k = 0; k < K; k++) {{")
        else:
            em.o(f"for (size_t k = 0; k < K; k += {isa.k_step}) {{")
        em.ind += 1
        emit_ladder_pipeline(em, "", "k", 0, d, itw_set)
        em.ind -= 1
        em.o("}")
        em.L.append("}"); em.L.append("")
        stats[f'ladder_u1_{d}'] = em.get_stats()

    # === DIF LADDER U=1 ===
    em.L.append(f"/* === DIF LADDER U=1 ({isa.name.upper()}, NFUSE={nfuse}) === */"); em.L.append("")
    for d in ['fwd', 'bwd']:
        em.reset_counters()
        if isa.target_attr:
            em.L.append(f"static {isa.target_attr} void")
        else:
            em.L.append(f"static void")
        em.L.append(f"radix32_tw_ladder_dif_kernel_{d}_{isa.name}_u1(")
        em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
        em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
        em.L.append(f"    const double * __restrict__ base_tw_re, const double * __restrict__ base_tw_im,")
        em.L.append(f"    size_t K)")
        em.L.append(f"{{")
        em.ind = 1
        _emit_ladder_boilerplate(em, isa, T, itw_set)
        if isa.name == 'scalar':
            em.o(f"for (size_t k = 0; k < K; k++) {{")
        else:
            em.o(f"for (size_t k = 0; k < K; k += {isa.k_step}) {{")
        em.ind += 1
        emit_ladder_dif_pipeline(em, "", "k", 0, d, itw_set)
        em.ind -= 1
        em.o("}")
        em.L.append("}"); em.L.append("")
        stats[f'ladder_dif_u1_{d}'] = em.get_stats()

    # Cleanup
    if isa.name != 'scalar':
        em.L.append("#undef LD"); em.L.append("#undef ST"); em.L.append("")
    else:
        em.L.append("#undef SQRT2_INV"); em.L.append("")

    em.L.append(f"#endif /* {guard} */")
    insert_stats_into_header(em.L, stats)
    return em.L, stats


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

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
    if 'ladder' in variant:
        return []

    text = '\n'.join(t2_lines)

    if variant == 'notw':
        t2_pattern = 'radix32_notw_dit_kernel'
        sv_name = 'radix32_n1sv_kernel'
    elif variant == 'dit_tw':
        t2_pattern = 'radix32_tw_flat_dit_kernel'
        sv_name = 'radix32_t1sv_dit_kernel'
    elif variant == 'dif_tw':
        t2_pattern = 'radix32_tw_flat_dif_kernel'
        sv_name = 'radix32_t1sv_dif_kernel'
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
    if 'ladder' in variant:
        return []

    text = '\n'.join(t2_lines)

    if variant == 'notw':
        t2_pattern = 'radix32_notw_dit_kernel'
        sv_name = 'radix32_n1sv_kernel'
    elif variant == 'dit_tw':
        t2_pattern = 'radix32_tw_flat_dit_kernel'
        sv_name = 'radix32_t1sv_dit_kernel'
    elif variant == 'dif_tw':
        t2_pattern = 'radix32_tw_flat_dif_kernel'
        sv_name = 'radix32_t1sv_dif_kernel'
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


def emit_ct_file(isa, itw_set, ct_variant, tile=None, drain_mode='temporal',
                 drain_prefetch=False, twiddle_prefetch_distance=0,
                 twiddle_prefetch_rows=2):
    """Emit FFTW-style n1 or t1_dit codelet for R=32.

    For ct_t1_buf_dit: tile is the number of k-positions per outbuf tile
    (default 64 for AVX2, 32 for AVX-512). drain_mode is 'temporal' (regular
    stores to rio, keeping output L1-hot) or 'stream' (non-temporal stores
    that bypass cache). Ignored for non-buf variants.

    drain_prefetch (buf variants only): if True, emit one `__builtin_prefetch`
    per drain stream (for re + im destinations) before the inner drain loop.

    twiddle_prefetch_distance (t1/t1_buf variants): if > 0, emit twiddle
    prefetches at the top of each k-step body targeting twiddles at m+distance.
    Intent: hide L2-latency for twiddle table accesses when table spills L1D
    (K >= 128 for R=32). Default 0 (disabled). Typical values: 8 (1 m-step
    ahead at VL=8) or 16 (2 ahead).

    twiddle_prefetch_rows: how many sub-FFT twiddle rows to prefetch per
    k-step. Each row is typically 1 cacheline. Default 2 keeps front-end
    pressure low; HW L2 prefetcher catches up for remaining rows.
    """
    is_n1 = ct_variant == 'ct_n1'
    is_n1_scaled = ct_variant == 'ct_n1_scaled'
    is_t1_dif = ct_variant == 'ct_t1_dif'
    is_t1_dit_log3 = ct_variant == 'ct_t1_dit_log3'
    is_t1_oop_dit = ct_variant == 'ct_t1_oop_dit'
    is_t1_buf_dit = ct_variant == 'ct_t1_buf_dit'
    is_t1_ladder_dit = ct_variant == 'ct_t1_ladder_dit'
    is_t1s_dit = ct_variant == 'ct_t1s_dit'
    nfuse = isa.nfuse_notw if (is_n1 or is_n1_scaled) else isa.nfuse_tw
    T = isa.reg_type
    em = Emitter(isa)
    em.twiddle_prefetch_distance = twiddle_prefetch_distance
    em.twiddle_prefetch_rows = twiddle_prefetch_rows

    # ct_t1_ladder_dit is AVX-512 only for now (ladder machinery assumes AVX-512
    # register budget; AVX2 variant would need NFUSE=2 rework)
    if is_t1_ladder_dit and isa.name != 'avx512':
        raise ValueError("ct_t1_ladder_dit currently supported on AVX-512 only "
                         "(ladder kernel design assumes 32-register budget)")

    # ct_t1s_dit (scalar-broadcast twiddle) disallows twiddle_prefetch — there
    # are only (R-1) scalar twiddles total, sized ~500 bytes, always L1-resident.
    # SW prefetch would be pure overhead.
    if is_t1s_dit and twiddle_prefetch_distance > 0:
        raise ValueError("ct_t1s_dit doesn't support twiddle_prefetch — "
                         "scalar twiddle table is always L1-resident")

    # Default tile size for buffered variant
    if is_t1_buf_dit:
        if tile is None:
            tile = 64 if isa.name == 'avx2' else 32
        if tile % isa.k_step != 0:
            raise ValueError(f"tile={tile} must be multiple of k_step={isa.k_step}")
        if drain_mode not in ('temporal', 'stream'):
            raise ValueError(f"drain_mode must be 'temporal' or 'stream'")
        if isa.name == 'scalar' and drain_mode == 'stream':
            drain_mode = 'temporal'  # no stream stores in scalar

    em.addr_mode = 'n1' if (is_n1 or is_n1_scaled) else \
                   ('t1_oop' if is_t1_oop_dit else \
                   ('t1_buf' if is_t1_buf_dit else \
                   ('t1s' if is_t1s_dit else 't1')))

    # Twiddle prefetch suffix applied to base names when active
    tpf_suffix = f"_tpf{twiddle_prefetch_distance}r{twiddle_prefetch_rows}" \
                 if twiddle_prefetch_distance > 0 else ""

    if is_n1:
        func_base = "radix32_n1"
        vname = "n1 (separate is/os)"
    elif is_n1_scaled:
        func_base = "radix32_n1_scaled"
        vname = "n1_scaled (separate is/os, output *= scale)"
    elif is_t1_oop_dit:
        func_base = f"radix32_t1_oop_dit{tpf_suffix}"
        vname = "t1_oop DIT (out-of-place, separate is/os, with twiddle)"
    elif is_t1_dif:
        func_base = f"radix32_t1_dif{tpf_suffix}"
        vname = "t1 DIF (in-place twiddle)"
    elif is_t1_dit_log3:
        func_base = f"radix32_t1_dit_log3{tpf_suffix}"
        vname = "t1 DIT log3 (in-place, derived twiddles)"
    elif is_t1s_dit:
        func_base = "radix32_t1s_dit"
        vname = "t1s DIT (in-place, scalar-broadcast twiddles)"
    elif is_t1_buf_dit:
        prefw_suffix = "_prefw" if drain_prefetch else ""
        func_base = f"radix32_t1_buf_dit_tile{tile}_{drain_mode}{prefw_suffix}{tpf_suffix}"
        vname = f"t1 DIT buffered (TILE={tile}, drain={drain_mode}, prefw={drain_prefetch}, tpf={twiddle_prefetch_distance})"
    elif is_t1_ladder_dit:
        func_base = f"radix32_t1_ladder_dit{tpf_suffix}"
        vname = "t1 DIT ladder (in-place, 5-base-twiddle log derivation, AVX-512)"
    else:
        func_base = f"radix32_t1_dit{tpf_suffix}"
        vname = "t1 DIT (in-place twiddle)"

    # Guard encodes full configuration so multiple variants can coexist.
    tpf_tag = f"_TPF{twiddle_prefetch_distance}R{twiddle_prefetch_rows}" \
              if twiddle_prefetch_distance > 0 else ""
    if is_t1_buf_dit:
        prefw_tag = "_PREFW" if drain_prefetch else ""
        guard = f"FFT_RADIX32_{isa.name.upper()}_{ct_variant.upper()}_TILE{tile}_{drain_mode.upper()}{prefw_tag}{tpf_tag}_H"
    else:
        guard = f"FFT_RADIX32_{isa.name.upper()}_CT_{ct_variant.upper()}{tpf_tag}_H"

    em.L.append(f"/**")
    em.L.append(f" * @file fft_radix32_{isa.name}_{ct_variant}.h")
    em.L.append(f" * @brief DFT-32 {isa.name.upper()} {vname} — NFUSE={nfuse}")
    em.L.append(f" *")
    em.L.append(f" * FFTW-style codelet for recursive CT executor.")
    em.L.append(f" * Generated by gen_radix32.py --variant {ct_variant}")
    em.L.append(f" */")
    em.L.append(f"")
    em.L.append(f"#ifndef {guard}")
    em.L.append(f"#define {guard}")
    em.L.append(f"")

    if isa.name != 'scalar':
        em.L.append(f"#include <immintrin.h>")
        em.L.append(f"")

    emit_twiddle_constants(em, itw_set)

    # LD/ST macros (unaligned for n1, aligned for t1)
    if isa.name == 'avx2':
        em.L.append("#ifndef R32CT_LD"); em.L.append("#define R32CT_LD(p) _mm256_loadu_pd(p)"); em.L.append("#endif")
        em.L.append("#ifndef R32CT_ST"); em.L.append("#define R32CT_ST(p,v) _mm256_storeu_pd((p),(v))"); em.L.append("#endif")
        em.L.append("#define LD R32CT_LD"); em.L.append("#define ST R32CT_ST"); em.L.append("")
    elif isa.name == 'avx512':
        em.L.append("#ifndef R32CT5_LD"); em.L.append("#define R32CT5_LD(p) _mm512_loadu_pd(p)"); em.L.append("#endif")
        em.L.append("#ifndef R32CT5_ST"); em.L.append("#define R32CT5_ST(p,v) _mm512_storeu_pd((p),(v))"); em.L.append("#endif")
        em.L.append("#define LD R32CT5_LD"); em.L.append("#define ST R32CT5_ST"); em.L.append("")

    for d in ['fwd', 'bwd']:
        em.reset_counters()
        # Reset hoist state — each direction's codelet gets its own hoist
        em.tw_hoisted = False
        em.tw_hoisted_set = set()
        em.addr_mode = 'n1' if (is_n1 or is_n1_scaled) else \
                       ('t1_oop' if is_t1_oop_dit else \
                       ('t1_buf' if is_t1_buf_dit else \
                       ('t1s' if is_t1s_dit else 't1')))
        em.store_scale = is_n1_scaled

        if isa.target_attr:
            em.L.append(f"static {isa.target_attr} void")
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
        else:
            # t1, t1_dif, t1_dit_log3, t1_buf_dit all share the same signature
            em.L.append(f"{func_base}_{d}_{isa.name}(")
            em.L.append(f"    double * __restrict__ rio_re, double * __restrict__ rio_im,")
            em.L.append(f"    const double * __restrict__ W_re, const double * __restrict__ W_im,")
            if isa.name == 'scalar':
                em.L.append(f"    size_t ios, size_t mb, size_t me, size_t ms)")
            else:
                em.L.append(f"    size_t ios, size_t me)")

        em.L.append(f"{{")
        em.ind = 1

        if isa.name != 'scalar':
            set1 = '_mm256_set1_pd' if isa.name == 'avx2' else '_mm512_set1_pd'
            em.o(f"const {T} sign_flip = {set1}(-0.0);")
            em.o(f"const {T} sqrt2_inv = {set1}(0.70710678118654752440);")
            em.b()
        else:
            em.o(f"const double sqrt2_inv = 0.70710678118654752440;")
            em.b()

        spill_total = N * isa.spill_mul
        if isa.name == 'scalar':
            em.o(f"double spill_re[{N}], spill_im[{N}];")
        else:
            em.o(f"{isa.align_attr} double spill_re[{spill_total}];")
            em.o(f"{isa.align_attr} double spill_im[{spill_total}];")
        em.b()

        # Tile output buffer for buffered variant: N=32 output streams × TILE k-positions
        if is_t1_buf_dit:
            em.c(f"Tile output buffer: {N} output streams × TILE={tile} k-positions")
            if isa.name != 'scalar':
                em.o(f"{isa.align_attr} double outbuf_re[{N}*{tile}];")
                em.o(f"{isa.align_attr} double outbuf_im[{N}*{tile}];")
            else:
                em.o(f"double outbuf_re[{N}*{tile}], outbuf_im[{N}*{tile}];")
            em.b()

        em.o(f"{T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
        em.o(f"{T} x4_re,x4_im,x5_re,x5_im,x6_re,x6_im,x7_re,x7_im;")
        slist = ",".join(f"s{i}_re,s{i}_im" for i in range(nfuse))
        em.o(f"{T} {slist};")
        em.b()

        # For DIT variants (t1_dit, t1_oop_dit, t1_dit_log3, t1_buf_dit), the
        # W32 broadcasts are deferred into PASS 2 inside the kernel — frees
        # registers during PASS 1. Gave -21.5% on R=32 t1_dit (AVX2). Other
        # variants still hoist.
        _defer_itw = ct_variant in ('ct_t1_dit', 'ct_t1_oop_dit', 'ct_t1_dit_log3',
                                    'ct_t1_dif', 'ct_t1_buf_dit')
        if isa.name != 'scalar' and itw_set and not _defer_itw:
            set1 = '_mm256_set1_pd' if isa.name == 'avx2' else '_mm512_set1_pd'
            for (e, tN) in sorted(itw_set):
                label = wN_label(e, tN)
                em.o(f"const {T} tw_{label}_re = {set1}({label}_re);")
                em.o(f"const {T} tw_{label}_im = {set1}({label}_im);")
            em.b()

        if isa.name == 'scalar' and itw_set:
            em.c(f"Internal W32 constants are #defined at file scope")
            em.b()

        # Broadcast scale factor before the loop (n1_scaled only)
        if is_n1_scaled and isa.name != 'scalar':
            set1 = '_mm256_set1_pd' if isa.name == 'avx2' else '_mm512_set1_pd'
            em.o(f"const {T} vscale = {set1}(scale);")
            em.b()

        # Hoist twiddle broadcasts before the m-loop (t1s only).
        # A register-budgeted subset of (R-1)=31 twiddles gets broadcast here
        # once; the rest are broadcast inline inside each butterfly (see
        # emit_ext_tw_scalar). The hoisted broadcasts are the amortization win:
        # one broadcast pays for me/k_step butterfly invocations.
        if is_t1s_dit:
            em.c(f"Hoist a register-budgeted subset of twiddle broadcasts")
            em.tw_hoisted = True
            em.emit_hoist_all_tw_scalars(N)
            em.b()

        # Loop
        if is_n1 or is_n1_scaled:
            if isa.name == 'scalar':
                em.o(f"for (size_t k = 0; k < vl; k++) {{")
            else:
                em.o(f"for (size_t k = 0; k < vl; k += {isa.k_step}) {{")
            em.ind += 1
            emit_notw_kernel(em, d, nfuse, itw_set)
            em.ind -= 1
            em.o("}")
        elif is_t1_buf_dit:
            # Buffered variant: tile-outer loop, kk-inner loop writes to outbuf,
            # drain to rio after each tile, tail falls back to stride-ios path.
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
            # Emit butterfly body in t1_buf mode (stores go to outbuf[m_out*TILE+kk])
            emit_dit_tw_flat_kernel(em, d, nfuse, itw_set)
            em.ind -= 1
            em.o("}")
            em.b()

            # Drain: 32 sequential streams, each TILE doubles
            em.c(f"Drain: {N} sequential streams, TILE doubles each (drain_mode={drain_mode}, prefw={drain_prefetch})")
            em.o(f"for (size_t m_out = 0; m_out < {N}; m_out++) {{")
            em.ind += 1
            em.o(f"const double * __restrict__ src_re = &outbuf_re[m_out*TILE];")
            em.o(f"const double * __restrict__ src_im = &outbuf_im[m_out*TILE];")
            em.o(f"double * __restrict__ dst_re = &rio_re[m_out*ios + tile_base];")
            em.o(f"double * __restrict__ dst_im = &rio_im[m_out*ios + tile_base];")
            # Minimal-aggressiveness prefetch: one per destination array per stream.
            # Targets the first cacheline of this stream's destination page, warming
            # L1 DTLB before the inner stores start. With -mprfchw this emits
            # PREFETCHW; without, it emits PREFETCHT1 (read-prefetch, still TLB-populating).
            if drain_prefetch and isa.name != 'scalar':
                em.o(f"__builtin_prefetch(dst_re, 1, 1);  /* write-intent, L2 locality */")
                em.o(f"__builtin_prefetch(dst_im, 1, 1);")
            if isa.name == 'scalar':
                em.o(f"for (size_t j = 0; j < TILE; j++) {{")
                em.o(f"    dst_re[j] = src_re[j]; dst_im[j] = src_im[j];")
                em.o(f"}}")
            else:
                em.o(f"for (size_t j = 0; j < TILE; j += {isa.k_step}) {{")
                if drain_mode == 'stream':
                    p = '_mm256' if isa.name == 'avx2' else '_mm512'
                    em.o(f"    {p}_stream_pd(&dst_re[j], LD(&src_re[j]));")
                    em.o(f"    {p}_stream_pd(&dst_im[j], LD(&src_im[j]));")
                else:
                    em.o(f"    ST(&dst_re[j], LD(&src_re[j]));")
                    em.o(f"    ST(&dst_im[j], LD(&src_im[j]));")
                em.o(f"}}")
            em.ind -= 1
            em.o("}")
            em.ind -= 1
            em.o("}")
            # Stream stores need sfence for visibility
            if isa.name != 'scalar' and drain_mode == 'stream':
                em.o("_mm_sfence();")
            em.b()

            # Tail: me % TILE iterations, direct stride-ios stores (standard t1 path)
            em.c(f"Tail: me % TILE iterations, direct stride-ios stores")
            em.addr_mode = 't1'
            em.o(f"for (size_t m = n_full_tiles * TILE; m < me; m += {isa.k_step}) {{")
            em.ind += 1
            emit_dit_tw_flat_kernel(em, d, nfuse, itw_set)
            em.ind -= 1
            em.o("}")
            em.o("#undef TILE")
        elif is_t1_ladder_dit:
            # Ladder DIT with ct_t1_dit-compatible signature.
            # Consumes flat W_re/W_im twiddle table but reads only rows
            # {0,1,3,7,15} (powers of 2 in flat layout).
            em.c(f"Ladder DIT — 5 base twiddles from flat W table at rows {{0,1,3,7,15}}")
            em.o(f"for (size_t m = 0; m < me; m += {isa.k_step}) {{")
            em.ind += 1
            # Use flat-W base map: rows 0, 1, 3, 7, 15 for b1, b2, b4, b8, b16
            flat_base_map = {'b1': 0, 'b2': 1, 'b4': 3, 'b8': 7, 'b16': 15}
            nfuse_lad = NFUSE_LADDER.get(isa.name, 4)
            emit_ladder_pipeline(em, "", "m", 0, d, itw_set,
                                 nfuse=nfuse_lad,
                                 base_map_override=flat_base_map,
                                 tw_buf_re='W_re',
                                 tw_buf_im='W_im',
                                 tw_stride='me')
            em.ind -= 1
            em.o("}")
        else:  # t1, t1_oop, t1_dif, t1_dit_log3, t1s
            if isa.name == 'scalar' and not is_t1_oop_dit:
                em.o(f"for (size_t m = mb; m < me; m++) {{")
            else:
                em.o(f"for (size_t m = 0; m < me; m += {isa.k_step}) {{")
            em.ind += 1
            if is_t1_dif:
                emit_dif_tw_flat_kernel(em, d, nfuse, itw_set)
            elif is_t1_dit_log3:
                emit_dit_tw_log3_kernel(em, d, nfuse, itw_set)
            else:
                # t1, t1_oop, t1s — same kernel, different addr_mode dispatch
                # for twiddle delivery (vector load vs scalar broadcast).
                if is_t1_oop_dit:
                    em.addr_mode = 't1_oop'
                emit_dit_tw_flat_kernel(em, d, nfuse, itw_set)
            em.ind -= 1
            em.o("}")
        em.L.append("}")
        em.L.append("")

    # n1_ovs: inline butterfly with fused SIMD transpose stores
    if is_n1 and isa.name != 'scalar':
        R = N  # 32
        VL = isa.k_step
        T = isa.reg_type
        n_groups = R // 4
        nfuse_ovs = isa.nfuse_notw

        for d in ['fwd', 'bwd']:
            em.L.append("")
            if isa.target_attr:
                em.L.append(f"static {isa.target_attr} void")
            else:
                em.L.append(f"static void")
            em.L.append(f"radix32_n1_ovs_{d}_{isa.name}(")
            em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
            em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
            em.L.append(f"    size_t is, size_t os, size_t vl, size_t ovs)")
            em.L.append(f"{{")

            # Boilerplate
            if isa.name != 'scalar':
                set1 = '_mm256_set1_pd' if isa.name == 'avx2' else '_mm512_set1_pd'
                em.L.append(f"    const {T} sign_flip = {set1}(-0.0);")
                em.L.append(f"    const {T} sqrt2_inv = {set1}(0.70710678118654752440);")
            em.L.append(f"    {isa.align_attr} double tbuf_re[{R*VL}];")
            em.L.append(f"    {isa.align_attr} double tbuf_im[{R*VL}];")
            spill_total = R * isa.spill_mul
            em.L.append(f"    {isa.align_attr} double spill_re[{spill_total}];")
            em.L.append(f"    {isa.align_attr} double spill_im[{spill_total}];")
            em.L.append(f"    {T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
            em.L.append(f"    {T} x4_re,x4_im,x5_re,x5_im,x6_re,x6_im,x7_re,x7_im;")
            slist = ",".join(f"s{i}_re,s{i}_im" for i in range(nfuse_ovs))
            em.L.append(f"    {T} {slist};")

            # Hoisted twiddle broadcasts
            if isa.name != 'scalar' and itw_set:
                set1 = '_mm256_set1_pd' if isa.name == 'avx2' else '_mm512_set1_pd'
                for (e, tN) in sorted(itw_set):
                    label = wN_label(e, tN)
                    em.L.append(f"    const {T} tw_{label}_re = {set1}({label}_re);")
                    em.L.append(f"    const {T} tw_{label}_im = {set1}({label}_im);")
            em.L.append(f"")

            # K loop
            em.L.append(f"    for (size_t k = 0; k < vl; k += {isa.k_step}) {{")

            # Emit butterfly body via Emitter with addr_mode='n1_ovs'
            em2 = Emitter(isa)
            em2.L = []
            em2.ind = 2
            em2.addr_mode = 'n1_ovs'
            em2.reset_counters()
            emit_notw_kernel(em2, d, nfuse_ovs, itw_set)
            em.L.extend(em2.L)

            # 4x4 transpose blocks: tbuf → output at stride ovs
            em.L.append(f"        /* 4x4 transpose: tbuf -> output at stride ovs */")
            for g in range(n_groups):
                b = g * 4
                for comp, arr in [('re', 'out_re'), ('im', 'out_im')]:
                    bname = f"tbuf_{comp}"
                    em.L.append(f"        {{ __m256d a_=_mm256_load_pd(&{bname}[{b+0}*{VL}]), b_=_mm256_load_pd(&{bname}[{b+1}*{VL}]);")
                    em.L.append(f"          __m256d c_=_mm256_load_pd(&{bname}[{b+2}*{VL}]), d_=_mm256_load_pd(&{bname}[{b+3}*{VL}]);")
                    em.L.append(f"          __m256d lo_ab=_mm256_unpacklo_pd(a_,b_), hi_ab=_mm256_unpackhi_pd(a_,b_);")
                    em.L.append(f"          __m256d lo_cd=_mm256_unpacklo_pd(c_,d_), hi_cd=_mm256_unpackhi_pd(c_,d_);")
                    em.L.append(f"          _mm256_storeu_pd(&{arr}[(k+0)*ovs+os*{b}], _mm256_permute2f128_pd(lo_ab,lo_cd,0x20));")
                    em.L.append(f"          _mm256_storeu_pd(&{arr}[(k+1)*ovs+os*{b}], _mm256_permute2f128_pd(hi_ab,hi_cd,0x20));")
                    em.L.append(f"          _mm256_storeu_pd(&{arr}[(k+2)*ovs+os*{b}], _mm256_permute2f128_pd(lo_ab,lo_cd,0x31));")
                    em.L.append(f"          _mm256_storeu_pd(&{arr}[(k+3)*ovs+os*{b}], _mm256_permute2f128_pd(hi_ab,hi_cd,0x31));")
                    em.L.append(f"        }}")

            em.L.append(f"    }}")
            em.L.append(f"}}")

    if isa.name != 'scalar':
        em.L.append("#undef LD"); em.L.append("#undef ST"); em.L.append("")
    em.L.append(f"#endif /* {guard} */")
    return em.L


def main():
    parser = argparse.ArgumentParser(description='Unified R=32 codelet generator')
    parser.add_argument('--isa', default='avx2',
                        choices=['scalar', 'avx2', 'avx512', 'all'])
    parser.add_argument('--variant', default='dit_tw',
                        choices=['dit_tw', 'dif_tw', 'notw', 'ladder',
                                 'ct_n1', 'ct_n1_scaled', 'ct_t1_dit', 'ct_t1_dit_log3',
                                 'ct_t1_dif', 'ct_t1_oop_dit', 'ct_t1_buf_dit',
                                 'ct_t1_ladder_dit', 'ct_t1s_dit', 'all'])
    parser.add_argument('--tile', type=int, default=None,
                        help='Tile size for ct_t1_buf_dit (default: 64/AVX2, 32/AVX-512)')
    parser.add_argument('--drain', default='temporal',
                        choices=['temporal', 'stream'],
                        help="Drain mode for ct_t1_buf_dit: 'temporal' or 'stream'")
    parser.add_argument('--drain-prefetch', action='store_true',
                        help='ct_t1_buf_dit only: emit PREFETCHW per drain stream '
                             '(warms L1 DTLB for rio pages). Compile with -mprfchw '
                             'for actual PREFETCHW; else gcc falls back to prefetcht1.')
    parser.add_argument('--twiddle-prefetch', type=int, default=0,
                        help='t1/t1_buf/t1_oop variants: issue __builtin_prefetch at '
                             'top of k-loop targeting twiddles at m+N. Intent: hide L2 '
                             'latency when twiddle table spills L1D (K>=128 for R=32). '
                             '0 = disabled (default). Typical values: 8, 16, 32.')
    parser.add_argument('--twiddle-prefetch-rows', type=int, default=2,
                        help='How many twiddle rows to prefetch per k-step. Default 2. '
                             'Each row = 1 cacheline (AVX-512) or 1-2 cachelines (AVX2). '
                             'HW prefetcher handles the rest after first few accesses.')
    parser.add_argument('--enumerate-buf-candidates', action='store_true',
                        help='Emit all (tile, drain, prefetch) combinations for the AVX2 buffered '
                             'variant as separately-named candidates. Rationale: R=32 AVX2 '
                             'ct_t1_dit is DTLB-store-bound (66.1% of clockticks per VTune) '
                             'and buffering is expected to attack this. AVX-512 baseline '
                             'is not equivalently bottlenecked, so buffering is AVX2-only.')
    args = parser.parse_args()

    itw_set = collect_internal_twiddles()

    if args.isa == 'all':
        targets = [ISA_SCALAR, ISA_AVX2, ISA_AVX512]
    else:
        targets = [ALL_ISA[args.isa]]

    def add_sqrt2_scalar(lines):
        idx = next(i for i, l in enumerate(lines) if 'TWIDDLES_DEFINED' in l or '#define FFT_RADIX32' in l)
        lines.insert(idx, "")
        lines.insert(idx, "#define SQRT2_INV 0.70710678118654752440")

    def print_file(lines, label, stats, isa_obj=None, variant_name=None):
        # Insert sv variants before #undef LD
        if isa_obj and variant_name:
            sv_lines = emit_sv_variants(lines, isa_obj, variant_name)
            if sv_lines:
                for i in range(len(lines)):
                    if lines[i].strip() == '#undef LD':
                        lines[i:i] = sv_lines
                        break
        print("\n".join(lines))
        print(f"\n{'='*72}", file=sys.stderr)
        print(f"  {label} — Operation Counts (per k-step)", file=sys.stderr)
        print(f"{'='*72}", file=sys.stderr)
        print(f"  {'kernel':<20s} {'add':>5s} {'sub':>5s} {'mul':>5s} {'neg':>5s}"
              f" {'fma':>5s} {'fms':>5s} │ {'arith':>6s} {'flops':>6s}"
              f" │ {'ld':>4s} {'st':>4s} {'sp':>4s} {'rl':>4s} {'mem':>5s}",
              file=sys.stderr)
        print(f"  {'─'*20} {'─'*5} {'─'*5} {'─'*5} {'─'*5}"
              f" {'─'*5} {'─'*5} ┼ {'─'*6} {'─'*6}"
              f" ┼ {'─'*4} {'─'*4} {'─'*4} {'─'*4} {'─'*5}",
              file=sys.stderr)
        for k in sorted(stats.keys()):
            s = stats[k]
            print(f"  {k:<20s} {s['add']:5d} {s['sub']:5d} {s['mul']:5d} {s['neg']:5d}"
                  f" {s['fma']:5d} {s['fms']:5d} │ {s['total_arith']:6d} {s['flops']:6d}"
                  f" │ {s['load']:4d} {s['store']:4d} {s['spill']:4d} {s['reload']:4d} {s['total_mem']:5d}",
                  file=sys.stderr)
        print(f"{'='*72}", file=sys.stderr)

    for isa in targets:
        if args.variant in ('dit_tw', 'all'):
            lines, stats = emit_dit_tw_flat_file(isa, itw_set)
            if isa.name == 'scalar':
                add_sqrt2_scalar(lines)
            print_file(lines, f"{isa.name.upper()} DIT TW", stats, isa, 'dit_tw')

        if args.variant in ('dif_tw', 'all'):
            lines, stats = emit_dif_tw_flat_file(isa, itw_set)
            if isa.name == 'scalar':
                add_sqrt2_scalar(lines)
            print_file(lines, f"{isa.name.upper()} DIF TW", stats, isa, 'dif_tw')

        if args.variant in ('notw', 'all'):
            lines, stats = emit_notw_file(isa, itw_set)
            if isa.name == 'scalar':
                add_sqrt2_scalar(lines)
            print_file(lines, f"{isa.name.upper()} NOTW", stats, isa, 'notw')

        if args.variant in ('ladder', 'all'):
            lines, stats = emit_ladder_file(isa, itw_set)
            if isa.name == 'scalar':
                add_sqrt2_scalar(lines)
            print_file(lines, f"{isa.name.upper()} LADDER", stats, isa, 'ladder')

        if args.variant in ('ct_n1',):
            lines = emit_ct_file(isa, itw_set, 'ct_n1')
            if isa.name == 'scalar':
                add_sqrt2_scalar(lines)
            print("\n".join(lines))

        if args.variant in ('ct_n1_scaled',):
            lines = emit_ct_file(isa, itw_set, 'ct_n1_scaled')
            if isa.name == 'scalar':
                add_sqrt2_scalar(lines)
            print("\n".join(lines))

        if args.variant in ('ct_t1_dit',):
            lines = emit_ct_file(isa, itw_set, 'ct_t1_dit',
                                 twiddle_prefetch_distance=args.twiddle_prefetch,
                                 twiddle_prefetch_rows=args.twiddle_prefetch_rows)
            if isa.name == 'scalar':
                add_sqrt2_scalar(lines)
            print("\n".join(lines))

        if args.variant in ('ct_t1s_dit',):
            # t1s doesn't accept twiddle_prefetch (validated inside emit_ct_file);
            # scalar twiddle table is always L1-resident.
            lines = emit_ct_file(isa, itw_set, 'ct_t1s_dit',
                                 twiddle_prefetch_distance=0,
                                 twiddle_prefetch_rows=1)
            if isa.name == 'scalar':
                add_sqrt2_scalar(lines)
            print("\n".join(lines))

        if args.variant in ('ct_t1_dit_log3',):
            lines = emit_ct_file(isa, itw_set, 'ct_t1_dit_log3',
                                 twiddle_prefetch_distance=args.twiddle_prefetch,
                                 twiddle_prefetch_rows=args.twiddle_prefetch_rows)
            if isa.name == 'scalar':
                add_sqrt2_scalar(lines)
            print("\n".join(lines))

        if args.variant in ('ct_t1_dif',):
            lines = emit_ct_file(isa, itw_set, 'ct_t1_dif',
                                 twiddle_prefetch_distance=args.twiddle_prefetch,
                                 twiddle_prefetch_rows=args.twiddle_prefetch_rows)
            if isa.name == 'scalar':
                add_sqrt2_scalar(lines)
            print("\n".join(lines))

        if args.variant in ('ct_t1_oop_dit',):
            lines = emit_ct_file(isa, itw_set, 'ct_t1_oop_dit',
                                 twiddle_prefetch_distance=args.twiddle_prefetch,
                                 twiddle_prefetch_rows=args.twiddle_prefetch_rows)
            if isa.name == 'scalar':
                add_sqrt2_scalar(lines)
            print("\n".join(lines))

        if args.variant in ('ct_t1_buf_dit',) and not args.enumerate_buf_candidates:
            lines = emit_ct_file(isa, itw_set, 'ct_t1_buf_dit',
                                 tile=args.tile, drain_mode=args.drain,
                                 drain_prefetch=args.drain_prefetch,
                                 twiddle_prefetch_distance=args.twiddle_prefetch,
                                 twiddle_prefetch_rows=args.twiddle_prefetch_rows)
            if isa.name == 'scalar':
                add_sqrt2_scalar(lines)
            print("\n".join(lines))

        if args.variant in ('ct_t1_ladder_dit',):
            # AVX-512 only check happens inside emit_ct_file
            if isa.name == 'avx512':
                lines = emit_ct_file(isa, itw_set, 'ct_t1_ladder_dit',
                                     twiddle_prefetch_distance=args.twiddle_prefetch,
                                     twiddle_prefetch_rows=args.twiddle_prefetch_rows)
                print("\n".join(lines))
            elif args.isa != 'all':
                raise SystemExit("ct_t1_ladder_dit requires --isa avx512")

    # --enumerate-buf-candidates: emit full (tile × drain × prefetch) matrix for AVX2.
    # 4 tiles × 2 drains × 2 prefetch = 16 candidates per ISA.
    # Only AVX2 — AVX-512 baseline isn't DTLB-bound the same way, and we
    # confirmed on R=16 that AVX-512 buffering regresses. If future profiling
    # on AVX-512 hardware reveals a different bottleneck buffering could help,
    # revisit.
    if args.enumerate_buf_candidates:
        tile_sizes = [32, 64, 128, 256]
        drain_modes = ['temporal', 'stream']
        prefetch_states = [False, True]
        for isa in targets:
            if isa.name != 'avx2':
                continue
            for t in tile_sizes:
                if t % isa.k_step != 0:
                    continue
                for dm in drain_modes:
                    for prefw in prefetch_states:
                        prefw_tag = "__prefw" if prefw else ""
                        tag = f"{isa.name}__ct_t1_buf_dit__tile{t}__{dm}{prefw_tag}"
                        print(f"/* === BEGIN CANDIDATE {tag} === */")
                        lines = emit_ct_file(isa, itw_set, 'ct_t1_buf_dit',
                                             tile=t, drain_mode=dm,
                                             drain_prefetch=prefw)
                        print("\n".join(lines))
                        print(f"/* === END CANDIDATE {tag} === */\n")
        return


if __name__ == '__main__':
    main()