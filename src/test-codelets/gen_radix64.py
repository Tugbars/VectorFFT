#!/usr/bin/env python3
"""
gen_radix64.py — Unified radix-64 codelet generator

Emits all R=64 codelet variants from shared butterfly infrastructure
with per-ISA / per-variant strategy configurations.

8x8 decomposition: pass 1 = N2 radix-8 sub-FFTs, pass 2 = N1 radix-8 combines.
Internal twiddles W64^(n1*n2) between passes (static arrays, broadcast inline).

ISA-specific decisions preserved:
  Scalar:  NFUSE=2, k-step=1, no SIMD
  AVX2:    NFUSE=0, k-step=4, 16 YMM, split butterfly
  AVX-512: NFUSE=2, k-step=8, 32 ZMM, monolithic butterfly

Usage:
  python3 gen_radix64.py --isa avx2 --variant notw
  python3 gen_radix64.py --isa avx512 --variant dit_tw
  python3 gen_radix64.py --isa scalar --variant notw
  python3 gen_radix64.py --isa all --variant all
"""

import math, sys, argparse, re

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

N, N1, N2 = 64, 8, 8

# ═══════════════════════════════════════════════════════════════
# TWIDDLE ANALYSIS — shared across all variants
# ═══════════════════════════════════════════════════════════════

def wN(e, tN):
    e = e % tN
    a = 2.0 * math.pi * e / tN
    return (math.cos(a), -math.sin(a))

def wN_label(e, tN):
    return f"W{tN}_{e % tN}"

def classify_itw(e):
    """Classify W64 internal twiddle for special-case codegen."""
    e = e % 64
    if e == 0:  return 'one'
    if e == 16: return 'neg_j'
    if e == 32: return 'neg_one'
    if e == 48: return 'pos_j'
    if e == 8 or e == 24: return 'w8'
    return 'general'

def collect_internal_twiddles():
    """Find which W64 exponents are needed for internal twiddles."""
    exps = set()
    for k1 in range(1, N1):
        for n2 in range(1, N2):
            exps.add((n2 * k1) % 64)
    return sorted(exps)

ITW_EXPS = collect_internal_twiddles()


# ═══════════════════════════════════════════════════════════════
# STATIC W64 ARRAYS
# ═══════════════════════════════════════════════════════════════

def emit_itw_static_arrays():
    """Emit iw_re[64]/iw_im[64] static arrays for internal twiddle broadcast."""
    L = []
    L.append('#ifndef FFT_W64_STATIC_ARRAYS_DEFINED')
    L.append('#define FFT_W64_STATIC_ARRAYS_DEFINED')
    L.append('/* Internal W64 twiddle constants -- static arrays for broadcast */')
    L.append('static const double __attribute__((aligned(8))) iw_re[64] = {')
    for i in range(0, 64, 4):
        vals = [f'{wN(j,64)[0]:.20e}' for j in range(i, i+4)]
        L.append(f'    {", ".join(vals)},')
    L.append('};')
    L.append('static const double __attribute__((aligned(8))) iw_im[64] = {')
    for i in range(0, 64, 4):
        vals = [f'{wN(j,64)[1]:.20e}' for j in range(i, i+4)]
        L.append(f'    {", ".join(vals)},')
    L.append('};')
    L.append('#endif /* FFT_W64_STATIC_ARRAYS_DEFINED */')
    L.append('')
    return L


# ═══════════════════════════════════════════════════════════════
# LOG3 EXTERNAL TWIDDLE DERIVATION DATA
# ═══════════════════════════════════════════════════════════════

ROW_BASES_FULL = {0: None, 8: ('ew8r','ew8i'), 16: ('ew16r','ew16i'),
                  24: ('ew24r','ew24i'), 32: ('ew32r','ew32i'),
                  40: ('ew40r','ew40i'), 48: ('ew48r','ew48i'), 56: ('ew56r','ew56i')}
COL_BASES_FULL = {0: None, 1: ('ew1r','ew1i'), 2: ('ew2r','ew2i'),
                  3: ('ew3r','ew3i'), 4: ('ew4r','ew4i'), 5: ('ew5r','ew5i'),
                  6: ('ew6r','ew6i'), 7: ('ew7r','ew7i')}


# ═══════════════════════════════════════════════════════════════
# ISA CONFIGURATION
# ═══════════════════════════════════════════════════════════════

class ISAConfig:
    """ISA-specific parameters. Each field matches the R=32 generator exactly."""
    def __init__(self, name, reg_type, width, k_step,
                 load_macro, store_macro, spill_mul,
                 target_attr, align_attr, align_bytes,
                 nfuse_tw, nfuse_notw):
        self.name = name
        self.reg_type = reg_type
        self.width = width          # doubles per register
        self.k_step = k_step
        self.load_macro = load_macro
        self.store_macro = store_macro
        self.spill_mul = spill_mul
        self.target_attr = target_attr
        self.align_attr = align_attr
        self.align_bytes = align_bytes
        self.nfuse_tw = nfuse_tw
        self.nfuse_notw = nfuse_notw

ISA_SCALAR = ISAConfig(
    name='scalar', reg_type='double', width=1, k_step=1,
    load_macro=None, store_macro=None, spill_mul=1,
    target_attr='', align_attr='', align_bytes=0,
    nfuse_tw=2, nfuse_notw=2,
)

ISA_AVX2 = ISAConfig(
    name='avx2', reg_type='__m256d', width=4, k_step=4,
    load_macro='LD', store_macro='ST', spill_mul=4,
    target_attr='__attribute__((target("avx2,fma")))',
    align_attr='__attribute__((aligned(32)))',
    align_bytes=32,
    nfuse_tw=0, nfuse_notw=0,
)

ISA_AVX512 = ISAConfig(
    name='avx512', reg_type='__m512d', width=8, k_step=8,
    load_macro='LD', store_macro='ST', spill_mul=8,
    target_attr='__attribute__((target("avx512f,avx512dq,fma")))',
    align_attr='__attribute__((aligned(64)))',
    align_bytes=64,
    nfuse_tw=2, nfuse_notw=2,
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
        # Arithmetic counters
        self.n_add = 0
        self.n_sub = 0
        self.n_mul = 0
        self.n_neg = 0
        self.n_fma = 0
        self.n_fms = 0
        self.n_load = 0
        self.n_store = 0

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
        total_arith = self.n_add + self.n_sub + self.n_mul + self.n_neg + self.n_fma + self.n_fms
        total_mem = self.n_load + self.n_store + self.spill_c + self.reload_c
        flops = (self.n_add + self.n_sub + self.n_neg) + self.n_mul + 2 * (self.n_fma + self.n_fms)
        return {
            'add': self.n_add, 'sub': self.n_sub, 'mul': self.n_mul,
            'neg': self.n_neg, 'fma': self.n_fma, 'fms': self.n_fms,
            'total_arith': total_arith, 'flops': flops,
            'load': self.n_load, 'store': self.n_store,
            'spill': self.spill_c, 'reload': self.reload_c,
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

    def fnma(self, a, b, c):
        """fnmadd: c - a*b"""
        if self.isa.name == 'scalar':
            return f"({c})-({a})*({b})"
        elif self.isa.name == 'avx2':
            return f"_mm256_fnmadd_pd({a},{b},{c})"
        else:
            return f"_mm512_fnmadd_pd({a},{b},{c})"

    def set1(self, v):
        if self.isa.name == 'scalar':
            return str(v)
        elif self.isa.name == 'avx2':
            return f"_mm256_set1_pd({v})"
        else:
            return f"_mm512_set1_pd({v})"

    # ── Load / Store / Spill / Reload ──

    # addr_mode: 'K' (default), 'n1' (separate is/os), 't1' (in-place ios/ms)
    addr_mode = 'K'

    def _in_addr(self, n, k_expr="k"):
        if self.addr_mode == 'n1': return f"{n}*is+{k_expr}"
        if self.addr_mode == 't1':
            if self.isa.name == 'scalar': return f"m*ms+{n}*ios"
            return f"m+{n}*ios"
        return f"{n}*K+{k_expr}"

    def _out_addr(self, m, k_expr="k"):
        if self.addr_mode == 'n1': return f"{m}*os+{k_expr}"
        if self.addr_mode == 't1':
            if self.isa.name == 'scalar': return f"m*ms+{m}*ios"
            return f"m+{m}*ios"
        return f"{m}*K+{k_expr}"

    def _in_buf(self):
        return "rio_re" if self.addr_mode == 't1' else "in_re"
    def _in_buf_im(self):
        return "rio_im" if self.addr_mode == 't1' else "in_im"
    def _out_buf(self):
        return "rio_re" if self.addr_mode == 't1' else "out_re"
    def _out_buf_im(self):
        return "rio_im" if self.addr_mode == 't1' else "out_im"

    def _tw_addr(self, tw_idx, k_expr="k"):
        if self.addr_mode == 't1': return f"{tw_idx}*me+m"
        return f"{tw_idx}*K+{k_expr}"
    def _tw_buf(self):
        return "W_re" if self.addr_mode == 't1' else "tw_re"
    def _tw_buf_im(self):
        return "W_im" if self.addr_mode == 't1' else "tw_im"

    def emit_load(self, v, n, k_expr="k"):
        self.n_load += 2
        ib, ibi = self._in_buf(), self._in_buf_im()
        addr = self._in_addr(n, k_expr)
        if self.isa.name == 'scalar':
            self.o(f"{v}_re = {ib}[{addr}];")
            self.o(f"{v}_im = {ibi}[{addr}];")
        else:
            self.o(f"{v}_re = {self.isa.load_macro}(&{ib}[{addr}]);")
            self.o(f"{v}_im = {self.isa.load_macro}(&{ibi}[{addr}]);")

    def emit_store(self, v, m, k_expr="k"):
        self.n_store += 2
        ob, obi = self._out_buf(), self._out_buf_im()
        addr = self._out_addr(m, k_expr)
        if self.isa.name == 'scalar':
            self.o(f"{ob}[{addr}] = {v}_re;")
            self.o(f"{obi}[{addr}] = {v}_im;")
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

    # ── Radix-8 butterfly (monolithic — for AVX-512/scalar) ──

    def emit_radix8(self, v, d, label=""):
        """Emit DFT-8 on v[0]..v[7] using _re/_im suffix convention."""
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
        # W8 twiddles on odd outputs
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
        # Combine
        for i, j in [(0, 4), (1, 5), (2, 6), (3, 7)]:
            self.o(f"  {v[i]}_re={self.add(f'e{i}r',f'o{i}r')}; {v[i]}_im={self.add(f'e{i}i',f'o{i}i')};")
            self.o(f"  {v[j]}_re={self.sub(f'e{i}r',f'o{i}r')}; {v[j]}_im={self.sub(f'e{i}i',f'o{i}i')};")
        self.o(f"}}")

    # ── Radix-8 split butterfly (3-phase, for AVX2 with bfr/bfi) ──

    def emit_radix8_split(self, v, d, label=""):
        """Split radix-8 butterfly for AVX2 (16 YMM).

        3 phases: Even DFT-4 -> spill to bfr -> Odd DFT-4 with W8 -> reload -> combine.
        """
        fwd = (d == 'fwd')
        T = self.isa.reg_type
        C = self.isa.width
        if label:
            self.c(f"{label} [{d}] (split)")
        self.o('{')
        self.ind += 1

        # Phase 1: Even DFT-4 -> spill A0..A3
        self.c('Phase 1: Even DFT-4 -> spill A0..A3')
        self.o(f'{{ {T} epr={self.add(f"{v[0]}_re",f"{v[4]}_re")}, epi={self.add(f"{v[0]}_im",f"{v[4]}_im")};')
        self.o(f'  {T} eqr={self.sub(f"{v[0]}_re",f"{v[4]}_re")}, eqi={self.sub(f"{v[0]}_im",f"{v[4]}_im")};')
        self.o(f'  {T} err={self.add(f"{v[2]}_re",f"{v[6]}_re")}, eri={self.add(f"{v[2]}_im",f"{v[6]}_im")};')
        self.o(f'  {T} esr={self.sub(f"{v[2]}_re",f"{v[6]}_re")}, esi={self.sub(f"{v[2]}_im",f"{v[6]}_im")};')
        st = lambda addr, val: f'_mm256_store_pd({addr},{val})'
        self.o(f'  {st(f"&bfr[0*{C}]",self.add("epr","err"))}; {st(f"&bfi[0*{C}]",self.add("epi","eri"))};')
        self.o(f'  {st(f"&bfr[2*{C}]",self.sub("epr","err"))}; {st(f"&bfi[2*{C}]",self.sub("epi","eri"))};')
        if fwd:
            self.o(f'  {st(f"&bfr[1*{C}]",self.add("eqr","esi"))}; {st(f"&bfi[1*{C}]",self.sub("eqi","esr"))};')
            self.o(f'  {st(f"&bfr[3*{C}]",self.sub("eqr","esi"))}; {st(f"&bfi[3*{C}]",self.add("eqi","esr"))};')
        else:
            self.o(f'  {st(f"&bfr[1*{C}]",self.sub("eqr","esi"))}; {st(f"&bfi[1*{C}]",self.add("eqi","esr"))};')
            self.o(f'  {st(f"&bfr[3*{C}]",self.add("eqr","esi"))}; {st(f"&bfi[3*{C}]",self.sub("eqi","esr"))};')
        self.o(f'}}')

        # Phase 2: Odd DFT-4 + W8 twiddles
        self.c('Phase 2: Odd DFT-4 + W8 twiddles')
        self.o(f'{{ {T} opr={self.add(f"{v[1]}_re",f"{v[5]}_re")}, opi={self.add(f"{v[1]}_im",f"{v[5]}_im")};')
        self.o(f'  {T} oqr={self.sub(f"{v[1]}_re",f"{v[5]}_re")}, oqi={self.sub(f"{v[1]}_im",f"{v[5]}_im")};')
        self.o(f'  {T} orr={self.add(f"{v[3]}_re",f"{v[7]}_re")}, ori={self.add(f"{v[3]}_im",f"{v[7]}_im")};')
        self.o(f'  {T} osr={self.sub(f"{v[3]}_re",f"{v[7]}_re")}, osi={self.sub(f"{v[3]}_im",f"{v[7]}_im")};')
        self.o(f'  const {T} B0r={self.add("opr","orr")}, B0i={self.add("opi","ori")};')
        self.o(f'  const {T} B2r={self.sub("opr","orr")}, B2i={self.sub("opi","ori")};')
        if fwd:
            self.o(f'  const {T} _B1r={self.add("oqr","osi")}, _B1i={self.sub("oqi","osr")};')
            self.o(f'  const {T} B1r={self.mul("sqrt2_inv",self.add("_B1r","_B1i"))}, B1i={self.mul("sqrt2_inv",self.sub("_B1i","_B1r"))};')
        else:
            self.o(f'  const {T} _B1r={self.sub("oqr","osi")}, _B1i={self.add("oqi","osr")};')
            self.o(f'  const {T} B1r={self.mul("sqrt2_inv",self.sub("_B1r","_B1i"))}, B1i={self.mul("sqrt2_inv",self.add("_B1r","_B1i"))};')
        if fwd:
            self.o(f'  const {T} _B3r={self.sub("oqr","osi")}, _B3i={self.add("oqi","osr")};')
            self.o(f'  const {T} B3r={self.mul("nsqrt2_inv",self.sub("_B3r","_B3i"))}, B3i={self.mul("nsqrt2_inv",self.add("_B3r","_B3i"))};')
        else:
            self.o(f'  const {T} _B3r={self.add("oqr","osi")}, _B3i={self.sub("oqi","osr")};')
            self.o(f'  const {T} B3r={self.mul("nsqrt2_inv",self.add("_B3r","_B3i"))}, B3i={self.mul("sqrt2_inv",self.sub("_B3r","_B3i"))};')

        # Phase 3: Reload A, combine
        self.c('Phase 3: Reload A, combine A +/- B')
        ld = lambda addr: f'_mm256_load_pd({addr})'
        self.o(f'{{ const {T} A0r={ld(f"&bfr[0*{C}]")}, A0i={ld(f"&bfi[0*{C}]")};')
        self.o(f'  {v[0]}_re={self.add("A0r","B0r")}; {v[0]}_im={self.add("A0i","B0i")};')
        self.o(f'  {v[4]}_re={self.sub("A0r","B0r")}; {v[4]}_im={self.sub("A0i","B0i")}; }}')
        self.o(f'{{ const {T} A1r={ld(f"&bfr[1*{C}]")}, A1i={ld(f"&bfi[1*{C}]")};')
        self.o(f'  {v[1]}_re={self.add("A1r","B1r")}; {v[1]}_im={self.add("A1i","B1i")};')
        self.o(f'  {v[5]}_re={self.sub("A1r","B1r")}; {v[5]}_im={self.sub("A1i","B1i")}; }}')
        self.o(f'{{ const {T} A2r={ld(f"&bfr[2*{C}]")}, A2i={ld(f"&bfi[2*{C}]")};')
        if fwd:
            self.o(f'  {v[2]}_re={self.add("A2r","B2i")}; {v[2]}_im={self.sub("A2i","B2r")};')
            self.o(f'  {v[6]}_re={self.sub("A2r","B2i")}; {v[6]}_im={self.add("A2i","B2r")}; }}')
        else:
            self.o(f'  {v[2]}_re={self.sub("A2r","B2i")}; {v[2]}_im={self.add("A2i","B2r")};')
            self.o(f'  {v[6]}_re={self.add("A2r","B2i")}; {v[6]}_im={self.sub("A2i","B2r")}; }}')
        self.o(f'{{ const {T} A3r={ld(f"&bfr[3*{C}]")}, A3i={ld(f"&bfi[3*{C}]")};')
        self.o(f'  {v[3]}_re={self.add("A3r","B3r")}; {v[3]}_im={self.add("A3i","B3i")};')
        self.o(f'  {v[7]}_re={self.sub("A3r","B3r")}; {v[7]}_im={self.sub("A3i","B3i")}; }}')

        self.o(f'}}')  # close odd scope
        self.ind -= 1
        self.o('}')

    # ── Radix-8 dispatch ──

    def emit_r8(self, v, d, label=""):
        """ISA dispatch: split for AVX2, monolithic for AVX-512/scalar."""
        if self.isa.name == 'avx2':
            self.emit_radix8_split(v, d, label)
        else:
            self.emit_radix8(v, d, label)

    # ── Internal twiddle W64^e ──

    def emit_itw_apply(self, xr, xi, e, fwd):
        """Apply W64^e to (xr, xi) in-place. 5 special cases + general cmul from static arrays."""
        e = e % 64
        cls = classify_itw(e)
        T = self.isa.reg_type

        if cls == 'one':
            return

        if cls == 'neg_j':
            if fwd: self.o(f'{{ {T} tr = {xr}; {xr} = {xi}; {xi} = {self.neg("tr")}; }}')
            else:   self.o(f'{{ {T} tr = {xr}; {xr} = {self.neg(xi)}; {xi} = tr; }}')
            return

        if cls == 'neg_one':
            self.o(f'{xr} = {self.neg(xr)};')
            self.o(f'{xi} = {self.neg(xi)};')
            return

        if cls == 'pos_j':
            if fwd: self.o(f'{{ {T} tr = {xr}; {xr} = {self.neg(xi)}; {xi} = tr; }}')
            else:   self.o(f'{{ {T} tr = {xr}; {xr} = {xi}; {xi} = {self.neg("tr")}; }}')
            return

        if cls == 'w8':
            if e == 8:
                if fwd:
                    self.o(f'{{ {T} tr = {self.mul("sqrt2_inv",self.add(xr,xi))};')
                    self.o(f'  {xi} = {self.mul("sqrt2_inv",self.sub(xi,xr))}; {xr} = tr; }}')
                else:
                    self.o(f'{{ {T} tr = {self.mul("sqrt2_inv",self.sub(xr,xi))};')
                    self.o(f'  {xi} = {self.mul("sqrt2_inv",self.add(xr,xi))}; {xr} = tr; }}')
            elif e == 24:
                if fwd:
                    self.o(f'{{ {T} tr = {self.mul("nsqrt2_inv",self.sub(xr,xi))};')
                    self.o(f'  {xi} = {self.mul("nsqrt2_inv",self.add(xr,xi))}; {xr} = tr; }}')
                else:
                    self.o(f'{{ {T} tr = {self.mul("nsqrt2_inv",self.add(xr,xi))};')
                    self.o(f'  {xi} = {self.mul("nsqrt2_inv",self.sub(xi,xr))}; {xr} = tr; }}')
            return

        # General: broadcast from static array
        wr = self.set1(f'iw_re[{e}]')
        wi = self.set1(f'iw_im[{e}]')
        self.o(f'{{ {T} tr = {xr};')
        if fwd:
            self.o(f'  {xr} = {self.fms(xr,wr,self.mul(xi,wi))};')
            self.o(f'  {xi} = {self.fma("tr",wi,self.mul(xi,wr))}; }}')
        else:
            self.o(f'  {xr} = {self.fma(xr,wr,self.mul(xi,wi))};')
            self.o(f'  {xi} = {self.fnma("tr",wi,self.mul(xi,wr))}; }}')

    # ── External twiddle DIT (pre-butterfly, direct loads) ──

    def emit_ext_twiddle_dit(self, n1, n, d, k_expr="k"):
        """Load external twiddle for element n and apply to x{n1} (DIT: pre-multiply)."""
        if n == 0:
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

    # ── External twiddle DIF (post-butterfly) ──

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

    # ── External twiddle log3 (from pre-derived bases) ──

    def emit_ext_twiddle_log3(self, xr, xi, n, fwd):
        """Apply W^n from pre-derived log3 bases."""
        col = n % 8
        row8 = n - col
        if n == 0:
            return
        elif row8 == 0:
            cr, ci = COL_BASES_FULL[col]
            self.emit_cmul_inplace_raw(xr, xi, cr, ci, fwd)
        elif col == 0:
            rr, ri = ROW_BASES_FULL[row8]
            self.emit_cmul_inplace_raw(xr, xi, rr, ri, fwd)
        else:
            cr, ci = COL_BASES_FULL[col]
            rr, ri = ROW_BASES_FULL[row8]
            self.emit_cmul_inplace_raw(xr, xi, cr, ci, fwd)
            self.emit_cmul_inplace_raw(xr, xi, rr, ri, fwd)

    # ── Complex multiply helpers ──

    def emit_cmul(self, dst_r, dst_i, ar, ai, br, bi, d):
        """Emit dst = a * b (fwd) or a * conj(b) (bwd)."""
        fwd = (d == 'fwd')
        if fwd:
            self.o(f"{dst_r} = {self.fms(ar,br,self.mul(ai,bi))};")
            self.o(f"{dst_i} = {self.fma(ar,bi,self.mul(ai,br))};")
        else:
            self.o(f"{dst_r} = {self.fma(ar,br,self.mul(ai,bi))};")
            self.o(f"{dst_i} = {self.fms(ai,br,self.mul(ar,bi))};")

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

    def emit_cmul_inplace_raw(self, xr, xi, wr, wi, fwd):
        """Emit (xr,xi) *= (wr,wi) using raw variable names (not _re/_im suffixed)."""
        T = self.isa.reg_type
        self.o(f'{{ {T} _tr = {xr};')
        if fwd:
            self.o(f'  {xr} = {self.fms(xr,wr,self.mul(xi,wi))};')
            self.o(f'  {xi} = {self.fma("_tr",wi,self.mul(xi,wr))}; }}')
        else:
            self.o(f'  {xr} = {self.fma(xr,wr,self.mul(xi,wi))};')
            self.o(f'  {xi} = {self.fnma("_tr",wi,self.mul(xi,wr))}; }}')

    def emit_cmul_split(self, dst_r, dst_i, ar, ai, br, bi):
        """Emit dst = a * b (always forward, for log3 derivation)."""
        self.o(f'const {self.isa.reg_type} {dst_r} = {self.fms(ar,br,self.mul(ai,bi))};')
        self.o(f'const {self.isa.reg_type} {dst_i} = {self.fma(ar,bi,self.mul(ai,br))};')


# ═══════════════════════════════════════════════════════════════
# LOG3 DERIVATION EMITTERS
# ═══════════════════════════════════════════════════════════════

def emit_log3_full(em, k_expr="k"):
    """Derive all 14 bases at loop top. Load W^1, W^8 → derive 12 more."""
    tb, tbi = em._tw_buf(), em._tw_buf_im()
    em.c('Log3 (full): load W^1, W^8, derive all 14 bases')
    if em.isa.name == 'scalar':
        em.o(f'const {em.isa.reg_type} ew1r = {tb}[{em._tw_addr(0, k_expr)}], ew1i = {tbi}[{em._tw_addr(0, k_expr)}];')
        em.o(f'const {em.isa.reg_type} ew8r = {tb}[{em._tw_addr(7, k_expr)}], ew8i = {tbi}[{em._tw_addr(7, k_expr)}];')
    else:
        load_fn = '_mm256_load_pd' if em.isa.name == 'avx2' else '_mm512_load_pd'
        em.o(f'const {em.isa.reg_type} ew1r = {load_fn}(&{tb}[{em._tw_addr(0, k_expr)}]), ew1i = {load_fn}(&{tbi}[{em._tw_addr(0, k_expr)}]);')
        em.o(f'const {em.isa.reg_type} ew8r = {load_fn}(&{tb}[{em._tw_addr(7, k_expr)}]), ew8i = {load_fn}(&{tbi}[{em._tw_addr(7, k_expr)}]);')
    em.b()
    em.c('Column bases: W^2..W^7')
    em.emit_cmul_split('ew2r','ew2i', 'ew1r','ew1i', 'ew1r','ew1i')
    em.emit_cmul_split('ew3r','ew3i', 'ew1r','ew1i', 'ew2r','ew2i')
    em.emit_cmul_split('ew4r','ew4i', 'ew2r','ew2i', 'ew2r','ew2i')
    em.emit_cmul_split('ew5r','ew5i', 'ew1r','ew1i', 'ew4r','ew4i')
    em.emit_cmul_split('ew6r','ew6i', 'ew3r','ew3i', 'ew3r','ew3i')
    em.emit_cmul_split('ew7r','ew7i', 'ew3r','ew3i', 'ew4r','ew4i')
    em.b()
    em.c('Row bases: W^16..W^56')
    em.emit_cmul_split('ew16r','ew16i', 'ew8r','ew8i', 'ew8r','ew8i')
    em.emit_cmul_split('ew24r','ew24i', 'ew8r','ew8i', 'ew16r','ew16i')
    em.emit_cmul_split('ew32r','ew32i', 'ew16r','ew16i', 'ew16r','ew16i')
    em.emit_cmul_split('ew40r','ew40i', 'ew8r','ew8i', 'ew32r','ew32i')
    em.emit_cmul_split('ew48r','ew48i', 'ew24r','ew24i', 'ew24r','ew24i')
    em.emit_cmul_split('ew56r','ew56i', 'ew24r','ew24i', 'ew32r','ew32i')
    em.b()


# ═══════════════════════════════════════════════════════════════
# KERNEL EMITTERS
# ═══════════════════════════════════════════════════════════════

def emit_notw_kernel(em, d, nfuse, k_expr="k"):
    """Emit one N1 (notw) kernel — no external twiddles."""
    fwd = (d == 'fwd')
    last_n2 = N2 - 1
    xv = [f"x{i}" for i in range(N1)]

    # PASS 1: N2 radix-8 sub-FFTs
    for n2 in range(N2):
        is_last = (n2 == last_n2)
        em.c(f"sub-FFT n2={n2}")
        for n1 in range(N1):
            n = N2 * n1 + n2
            em.emit_load(f"x{n1}", n, k_expr)
        em.b()
        em.emit_r8(xv, d, f"radix-8 n2={n2}")
        em.b()

        if is_last and nfuse > 0:
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
                em.emit_itw_apply(f"x{n2}_re", f"x{n2}_im", e, fwd)
            em.b()

        em.emit_r8(xv, d, f"radix-8 k1={k1}")
        em.b()

        for k2 in range(N2):
            em.emit_store(f"x{k2}", k1 + N1 * k2, k_expr)
        em.b()


def emit_dit_tw_flat_kernel(em, d, nfuse, k_expr="k"):
    """Emit one DIT twiddled flat kernel (fwd or bwd)."""
    fwd = (d == 'fwd')
    last_n2 = N2 - 1
    xv = [f"x{i}" for i in range(N1)]

    # PASS 1: N2 sub-FFTs with external twiddle pre-multiply
    for n2 in range(N2):
        is_last = (n2 == last_n2)
        em.c(f"sub-FFT n2={n2}")
        for n1 in range(N1):
            n = N2 * n1 + n2
            em.emit_load(f"x{n1}", n, k_expr)
            em.emit_ext_twiddle_dit(n1, n, d, k_expr)
        em.b()
        em.emit_r8(xv, d, f"radix-8 n2={n2}")
        em.b()

        if is_last and nfuse > 0:
            em.c(f"FUSED: save x0..x{nfuse-1} to s-regs, spill x{nfuse}..x{N1-1}")
            for k1 in range(nfuse):
                em.o(f"s{k1}_re = x{k1}_re; s{k1}_im = x{k1}_im;")
            for k1 in range(nfuse, N1):
                em.emit_spill(f"x{k1}", n2 * N1 + k1)
        else:
            for k1 in range(N1):
                em.emit_spill(f"x{k1}", n2 * N1 + k1)
        em.b()

    # PASS 2: column combines
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
                em.emit_itw_apply(f"x{n2}_re", f"x{n2}_im", e, fwd)
            em.b()

        em.emit_r8(xv, d, f"radix-8 k1={k1}")
        em.b()

        for k2 in range(N2):
            em.emit_store(f"x{k2}", k1 + N1 * k2, k_expr)
        em.b()


def emit_dif_tw_flat_kernel(em, d, nfuse, k_expr="k"):
    """Emit one DIF twiddled flat kernel (fwd or bwd)."""
    fwd = (d == 'fwd')
    last_n2 = N2 - 1
    xv = [f"x{i}" for i in range(N1)]

    # PASS 1: load (no twiddle) -> radix-8 -> spill
    for n2 in range(N2):
        is_last = (n2 == last_n2)
        em.c(f"sub-FFT n2={n2}")
        for n1 in range(N1):
            n = N2 * n1 + n2
            em.emit_load(f"x{n1}", n, k_expr)
        em.b()
        em.emit_r8(xv, d, f"radix-8 n2={n2}")
        em.b()

        if is_last and nfuse > 0:
            em.c(f"FUSED: save x0..x{nfuse-1} to locals, spill x{nfuse}..x{N1-1}")
            for k1 in range(nfuse):
                em.o(f"s{k1}_re = x{k1}_re; s{k1}_im = x{k1}_im;")
            for k1 in range(nfuse, N1):
                em.emit_spill(f"x{k1}", n2 * N1 + k1)
        else:
            for k1 in range(N1):
                em.emit_spill(f"x{k1}", n2 * N1 + k1)
        em.b()

    # PASS 2: reload -> int twiddle -> radix-8 -> ext twiddle on OUTPUT -> store
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
                em.emit_itw_apply(f"x{n2}_re", f"x{n2}_im", e, fwd)
            em.b()

        em.emit_r8(xv, d, f"radix-8 k1={k1}")
        em.b()

        # DIF: external twiddle on OUTPUT
        for k2 in range(N2):
            m = k1 + N1 * k2
            if m > 0:
                em.emit_ext_twiddle_dif(f"x{k2}", m - 1, d, k_expr)
        em.b()

        for k2 in range(N2):
            em.emit_store(f"x{k2}", k1 + N1 * k2, k_expr)
        em.b()


def emit_dit_tw_log3_kernel(em, d, nfuse, k_expr="k"):
    """Emit DIT twiddled log3 kernel — derive 63 twiddles from 2 base loads."""
    fwd = (d == 'fwd')
    last_n2 = N2 - 1
    xv = [f"x{i}" for i in range(N1)]

    emit_log3_full(em, k_expr)

    # PASS 1: load -> log3 ext twiddle -> radix-8 -> spill
    for n2 in range(N2):
        is_last = (n2 == last_n2)
        em.c(f"sub-FFT n2={n2}")
        for n1 in range(N1):
            n = N2 * n1 + n2
            em.emit_load(f"x{n1}", n, k_expr)
            em.emit_ext_twiddle_log3(f"x{n1}_re", f"x{n1}_im", n, fwd)
        em.b()
        em.emit_r8(xv, d, f"radix-8 n2={n2}")
        em.b()

        if is_last and nfuse > 0:
            em.c(f"FUSED: save x0..x{nfuse-1} to s-regs, spill x{nfuse}..x{N1-1}")
            for k1 in range(nfuse):
                em.o(f"s{k1}_re = x{k1}_re; s{k1}_im = x{k1}_im;")
            for k1 in range(nfuse, N1):
                em.emit_spill(f"x{k1}", n2 * N1 + k1)
        else:
            for k1 in range(N1):
                em.emit_spill(f"x{k1}", n2 * N1 + k1)
        em.b()

    # PASS 2
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
                em.emit_itw_apply(f"x{n2}_re", f"x{n2}_im", e, fwd)
            em.b()

        em.emit_r8(xv, d, f"radix-8 k1={k1}")
        em.b()

        for k2 in range(N2):
            em.emit_store(f"x{k2}", k1 + N1 * k2, k_expr)
        em.b()


def emit_dif_tw_log3_kernel(em, d, nfuse, k_expr="k"):
    """Emit DIF twiddled log3 kernel."""
    fwd = (d == 'fwd')
    last_n2 = N2 - 1
    xv = [f"x{i}" for i in range(N1)]

    emit_log3_full(em, k_expr)

    # PASS 1: load (no ext tw) -> radix-8 -> spill
    for n2 in range(N2):
        is_last = (n2 == last_n2)
        em.c(f"sub-FFT n2={n2}")
        for n1 in range(N1):
            n = N2 * n1 + n2
            em.emit_load(f"x{n1}", n, k_expr)
        em.b()
        em.emit_r8(xv, d, f"radix-8 n2={n2}")
        em.b()

        if is_last and nfuse > 0:
            em.c(f"FUSED: save x0..x{nfuse-1}, spill x{nfuse}..x{N1-1}")
            for k1 in range(nfuse):
                em.o(f"s{k1}_re = x{k1}_re; s{k1}_im = x{k1}_im;")
            for k1 in range(nfuse, N1):
                em.emit_spill(f"x{k1}", n2 * N1 + k1)
        else:
            for k1 in range(N1):
                em.emit_spill(f"x{k1}", n2 * N1 + k1)
        em.b()

    # PASS 2: reload -> int tw -> radix-8 -> log3 ext tw on output -> store
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
                em.emit_itw_apply(f"x{n2}_re", f"x{n2}_im", e, fwd)
            em.b()

        em.emit_r8(xv, d, f"radix-8 k1={k1}")
        em.b()

        for k2 in range(N2):
            m = k1 + N1 * k2
            em.emit_ext_twiddle_log3(f"x{k2}_re", f"x{k2}_im", m, fwd)
        em.b()

        for k2 in range(N2):
            em.emit_store(f"x{k2}", k1 + N1 * k2, k_expr)
        em.b()


# ═══════════════════════════════════════════════════════════════
# FILE BOILERPLATE HELPERS
# ═══════════════════════════════════════════════════════════════

def insert_stats_into_header(lines, stats):
    """Insert operation count table into the file header comment block."""
    table = []
    table.append(" *")
    table.append(" * -- Operation counts per k-step --")
    table.append(" *")
    table.append(f" *   {'kernel':<20s} {'add':>5s} {'sub':>5s} {'mul':>5s} {'neg':>5s}"
                 f" {'fma':>5s} {'fms':>5s} | {'arith':>5s} {'flops':>5s}"
                 f" | {'ld':>3s} {'st':>3s} {'sp':>3s} {'rl':>3s} {'mem':>4s}")
    sep = '-' * 20
    s5 = '-' * 5
    s3 = '-' * 3
    s4 = '-' * 4
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
    lines.insert(0, "/* " + " | ".join(f"{k}:{v}" for k, v in stats.items()) + " */")


def _emit_common_boilerplate(em, isa, T, nfuse):
    """Emit shared function-body boilerplate: constants, spill buffer, working regs."""
    if isa.name == 'scalar':
        em.o(f"const double sqrt2_inv = 0.70710678118654752440;")
        em.o(f"const double nsqrt2_inv = -0.70710678118654752440;")
        em.b()
    else:
        set1 = '_mm256_set1_pd' if isa.name == 'avx2' else '_mm512_set1_pd'
        em.o(f"const {T} sign_flip = {set1}(-0.0);")
        em.o(f"const {T} sqrt2_inv = {set1}(0.70710678118654752440);")
        em.o(f"const {T} nsqrt2_inv = {set1}(-0.70710678118654752440);")
        em.b()

    spill_total = N * isa.spill_mul
    if isa.name == 'scalar':
        em.o(f"double spill_re[{N}], spill_im[{N}];")
    else:
        em.o(f"{isa.align_attr} double spill_re[{spill_total}];")
        em.o(f"{isa.align_attr} double spill_im[{spill_total}];")
    em.b()

    # Split butterfly auxiliary buffer for AVX2
    if isa.name == 'avx2':
        em.o(f"{isa.align_attr} double bfr[4*{isa.width}], bfi[4*{isa.width}];")
        em.b()

    em.o(f"{T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
    em.o(f"{T} x4_re,x4_im,x5_re,x5_im,x6_re,x6_im,x7_re,x7_im;")
    if nfuse > 0:
        slist = ",".join(f"s{i}_re,s{i}_im" for i in range(nfuse))
        em.o(f"{T} {slist};")
    em.b()


def _emit_ld_st_macros(lines, isa, prefix):
    """Emit LD/ST macro definitions."""
    if isa.name == 'avx2':
        lines.append(f"#ifndef {prefix}A_LD"); lines.append(f"#define {prefix}A_LD(p) _mm256_loadu_pd(p)"); lines.append("#endif")
        lines.append(f"#ifndef {prefix}A_ST"); lines.append(f"#define {prefix}A_ST(p,v) _mm256_storeu_pd((p),(v))"); lines.append("#endif")
        lines.append(f"#define LD {prefix}A_LD"); lines.append(f"#define ST {prefix}A_ST"); lines.append("")
    elif isa.name == 'avx512':
        lines.append(f"#ifndef {prefix}5_LD"); lines.append(f"#define {prefix}5_LD(p) _mm512_loadu_pd(p)"); lines.append("#endif")
        lines.append(f"#ifndef {prefix}5_ST"); lines.append(f"#define {prefix}5_ST(p,v) _mm512_storeu_pd((p),(v))"); lines.append("#endif")
        lines.append(f"#define LD {prefix}5_LD"); lines.append(f"#define ST {prefix}5_ST"); lines.append("")


# ═══════════════════════════════════════════════════════════════
# FILE EMITTERS
# ═══════════════════════════════════════════════════════════════

def emit_notw_file(isa):
    """Emit complete notw header for one ISA. Returns (lines, stats)."""
    nfuse = isa.nfuse_notw
    T = isa.reg_type
    em = Emitter(isa)

    guard = f"FFT_RADIX64_{isa.name.upper()}_NOTW_H"
    em.L.append(f"/**")
    em.L.append(f" * @file fft_radix64_{isa.name}_notw.h")
    em.L.append(f" * @brief DFT-64 {isa.name.upper()} notw codelet -- NFUSE={nfuse}")
    em.L.append(f" *")
    em.L.append(f" * No external twiddles. Pure DFT-64 butterfly.")
    em.L.append(f" * 8x8 decomposition, fwd + bwd")
    em.L.append(f" * Generated by gen_radix64.py")
    em.L.append(f" */")
    em.L.append(f"")
    em.L.append(f"#ifndef {guard}")
    em.L.append(f"#define {guard}")
    em.L.append(f"")
    em.L.append(f"#include <stddef.h>")
    if isa.name != 'scalar':
        em.L.append(f"#include <immintrin.h>")
    em.L.append(f"")

    em.L.extend(emit_itw_static_arrays())

    _emit_ld_st_macros(em.L, isa, 'R64N')

    em.L.append(f"/* === NOTW ({isa.name.upper()}, NFUSE={nfuse}) === */")
    em.L.append("")

    stats = {}
    for d in ['fwd', 'bwd']:
        em.reset_counters()
        if isa.target_attr:
            em.L.append(f"static {isa.target_attr} void")
        else:
            em.L.append(f"static void")
        em.L.append(f"radix64_n1_dit_kernel_{d}_{isa.name}(")
        em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
        em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
        em.L.append(f"    size_t K)")
        em.L.append(f"{{")
        em.ind = 1

        _emit_common_boilerplate(em, isa, T, nfuse)

        if isa.name == 'scalar':
            em.o(f"for (size_t k = 0; k < K; k++) {{")
        else:
            em.o(f"for (size_t k = 0; k < K; k += {isa.k_step}) {{")
        em.ind += 1
        emit_notw_kernel(em, d, nfuse)
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


def emit_dit_tw_flat_file(isa):
    """Emit complete DIT tw flat header for one ISA. Returns (lines, stats)."""
    nfuse = isa.nfuse_tw
    T = isa.reg_type
    em = Emitter(isa)

    guard = f"FFT_RADIX64_{isa.name.upper()}_TW_H"
    em.L.append(f"/**")
    em.L.append(f" * @file fft_radix64_{isa.name}_tw.h")
    em.L.append(f" * @brief DFT-64 {isa.name.upper()} twiddled codelet -- flat twiddles, NFUSE={nfuse}")
    em.L.append(f" *")
    em.L.append(f" * 8x8 decomposition, fwd + bwd")
    em.L.append(f" * Generated by gen_radix64.py")
    em.L.append(f" */")
    em.L.append(f"")
    em.L.append(f"#ifndef {guard}")
    em.L.append(f"#define {guard}")
    em.L.append(f"")
    em.L.append(f"#include <stddef.h>")
    if isa.name != 'scalar':
        em.L.append(f"#include <immintrin.h>")
    em.L.append(f"")

    em.L.extend(emit_itw_static_arrays())

    _emit_ld_st_macros(em.L, isa, 'R64T')

    em.L.append(f"/* === FLAT DIT ({isa.name.upper()}, NFUSE={nfuse}) === */")
    em.L.append("")

    stats = {}
    for d in ['fwd', 'bwd']:
        em.reset_counters()
        if isa.target_attr:
            em.L.append(f"static {isa.target_attr} void")
        else:
            em.L.append(f"static void")
        em.L.append(f"radix64_tw_flat_dit_kernel_{d}_{isa.name}(")
        em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
        em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
        em.L.append(f"    const double * __restrict__ tw_re, const double * __restrict__ tw_im,")
        em.L.append(f"    size_t K)")
        em.L.append(f"{{")
        em.ind = 1

        _emit_common_boilerplate(em, isa, T, nfuse)

        if isa.name == 'scalar':
            em.o(f"for (size_t k = 0; k < K; k++) {{")
        else:
            em.o(f"for (size_t k = 0; k < K; k += {isa.k_step}) {{")
        em.ind += 1
        emit_dit_tw_flat_kernel(em, d, nfuse)
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


def emit_dif_tw_flat_file(isa):
    """Emit complete DIF tw flat header for one ISA."""
    nfuse = isa.nfuse_tw
    T = isa.reg_type
    em = Emitter(isa)

    guard = f"FFT_RADIX64_{isa.name.upper()}_DIF_TW_H"
    em.L.append(f"/**")
    em.L.append(f" * @file fft_radix64_{isa.name}_dif_tw.h")
    em.L.append(f" * @brief DFT-64 {isa.name.upper()} DIF twiddled codelet -- flat twiddles, NFUSE={nfuse}")
    em.L.append(f" *")
    em.L.append(f" * DIF: external twiddle on OUTPUT (after butterfly)")
    em.L.append(f" * 8x8 decomposition, fwd + bwd")
    em.L.append(f" * Generated by gen_radix64.py")
    em.L.append(f" */")
    em.L.append(f"")
    em.L.append(f"#ifndef {guard}")
    em.L.append(f"#define {guard}")
    em.L.append(f"")
    em.L.append(f"#include <stddef.h>")
    if isa.name != 'scalar':
        em.L.append(f"#include <immintrin.h>")
    em.L.append(f"")

    em.L.extend(emit_itw_static_arrays())

    _emit_ld_st_macros(em.L, isa, 'R64D')

    em.L.append(f"/* === DIF FLAT ({isa.name.upper()}, NFUSE={nfuse}) === */")
    em.L.append("")

    stats = {}
    for d in ['fwd', 'bwd']:
        em.reset_counters()
        if isa.target_attr:
            em.L.append(f"static {isa.target_attr} void")
        else:
            em.L.append(f"static void")
        em.L.append(f"radix64_tw_flat_dif_kernel_{d}_{isa.name}(")
        em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
        em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
        em.L.append(f"    const double * __restrict__ tw_re, const double * __restrict__ tw_im,")
        em.L.append(f"    size_t K)")
        em.L.append(f"{{")
        em.ind = 1

        _emit_common_boilerplate(em, isa, T, nfuse)

        if isa.name == 'scalar':
            em.o(f"for (size_t k = 0; k < K; k++) {{")
        else:
            em.o(f"for (size_t k = 0; k < K; k += {isa.k_step}) {{")
        em.ind += 1
        emit_dif_tw_flat_kernel(em, d, nfuse)
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


def emit_dit_tw_log3_file(isa):
    """Emit DIT tw log3 header."""
    nfuse = isa.nfuse_tw
    T = isa.reg_type
    em = Emitter(isa)

    guard = f"FFT_RADIX64_{isa.name.upper()}_TW_LOG3_H"
    em.L.append(f"/**")
    em.L.append(f" * @file fft_radix64_{isa.name}_tw_log3.h")
    em.L.append(f" * @brief DFT-64 {isa.name.upper()} DIT log3 twiddled codelet -- NFUSE={nfuse}")
    em.L.append(f" *")
    em.L.append(f" * 2 base twiddle loads + 12 cmul derivations per k-step.")
    em.L.append(f" * Generated by gen_radix64.py")
    em.L.append(f" */")
    em.L.append(f"")
    em.L.append(f"#ifndef {guard}")
    em.L.append(f"#define {guard}")
    em.L.append(f"")
    em.L.append(f"#include <stddef.h>")
    if isa.name != 'scalar':
        em.L.append(f"#include <immintrin.h>")
    em.L.append(f"")

    em.L.extend(emit_itw_static_arrays())

    _emit_ld_st_macros(em.L, isa, 'R64TL')

    em.L.append(f"/* === DIT LOG3 ({isa.name.upper()}, NFUSE={nfuse}) === */")
    em.L.append("")

    stats = {}
    for d in ['fwd', 'bwd']:
        em.reset_counters()
        if isa.target_attr:
            em.L.append(f"static {isa.target_attr} void")
        else:
            em.L.append(f"static void")
        em.L.append(f"radix64_tw_log3_dit_kernel_{d}_{isa.name}(")
        em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
        em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
        em.L.append(f"    const double * __restrict__ tw_re, const double * __restrict__ tw_im,")
        em.L.append(f"    size_t K)")
        em.L.append(f"{{")
        em.ind = 1

        _emit_common_boilerplate(em, isa, T, nfuse)

        if isa.name == 'scalar':
            em.o(f"for (size_t k = 0; k < K; k++) {{")
        else:
            em.o(f"for (size_t k = 0; k < K; k += {isa.k_step}) {{")
        em.ind += 1
        emit_dit_tw_log3_kernel(em, d, nfuse)
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


def emit_dif_tw_log3_file(isa):
    """Emit DIF tw log3 header."""
    nfuse = isa.nfuse_tw
    T = isa.reg_type
    em = Emitter(isa)

    guard = f"FFT_RADIX64_{isa.name.upper()}_DIF_TW_LOG3_H"
    em.L.append(f"/**")
    em.L.append(f" * @file fft_radix64_{isa.name}_dif_tw_log3.h")
    em.L.append(f" * @brief DFT-64 {isa.name.upper()} DIF log3 twiddled codelet -- NFUSE={nfuse}")
    em.L.append(f" *")
    em.L.append(f" * DIF log3: 2 base loads + 12 cmuls, post-twiddle via derived bases.")
    em.L.append(f" * Generated by gen_radix64.py")
    em.L.append(f" */")
    em.L.append(f"")
    em.L.append(f"#ifndef {guard}")
    em.L.append(f"#define {guard}")
    em.L.append(f"")
    em.L.append(f"#include <stddef.h>")
    if isa.name != 'scalar':
        em.L.append(f"#include <immintrin.h>")
    em.L.append(f"")

    em.L.extend(emit_itw_static_arrays())

    _emit_ld_st_macros(em.L, isa, 'R64DL')

    em.L.append(f"/* === DIF LOG3 ({isa.name.upper()}, NFUSE={nfuse}) === */")
    em.L.append("")

    stats = {}
    for d in ['fwd', 'bwd']:
        em.reset_counters()
        if isa.target_attr:
            em.L.append(f"static {isa.target_attr} void")
        else:
            em.L.append(f"static void")
        em.L.append(f"radix64_tw_log3_dif_kernel_{d}_{isa.name}(")
        em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
        em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
        em.L.append(f"    const double * __restrict__ tw_re, const double * __restrict__ tw_im,")
        em.L.append(f"    size_t K)")
        em.L.append(f"{{")
        em.ind = 1

        _emit_common_boilerplate(em, isa, T, nfuse)

        if isa.name == 'scalar':
            em.o(f"for (size_t k = 0; k < K; k++) {{")
        else:
            em.o(f"for (size_t k = 0; k < K; k += {isa.k_step}) {{")
        em.ind += 1
        emit_dif_tw_log3_kernel(em, d, nfuse)
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
# FILE EMITTER — CT codelets (n1, t1_dit, t1_dit_log3, t1_dif)
# ═══════════════════════════════════════════════════════════════

def emit_ct_file(isa, ct_variant):
    """Emit FFTW-style CT codelet for R=64."""
    is_n1 = ct_variant == 'ct_n1'
    is_t1_dif = ct_variant == 'ct_t1_dif'
    is_t1_dit_log3 = ct_variant == 'ct_t1_dit_log3'
    nfuse = isa.nfuse_notw if is_n1 else isa.nfuse_tw
    T = isa.reg_type
    em = Emitter(isa)
    em.addr_mode = 'n1' if is_n1 else 't1'

    if is_n1:
        func_base = "radix64_n1"
        vname = "n1 (separate is/os)"
    elif is_t1_dif:
        func_base = "radix64_t1_dif"
        vname = "t1 DIF (in-place twiddle)"
    elif is_t1_dit_log3:
        func_base = "radix64_t1_dit_log3"
        vname = "t1 DIT log3 (in-place twiddle)"
    else:
        func_base = "radix64_t1_dit"
        vname = "t1 DIT (in-place twiddle)"

    guard = f"FFT_RADIX64_{isa.name.upper()}_CT_{ct_variant.upper()}_H"

    em.L.append(f"/**")
    em.L.append(f" * @file fft_radix64_{isa.name}_{ct_variant}.h")
    em.L.append(f" * @brief DFT-64 {isa.name.upper()} {vname} -- NFUSE={nfuse}")
    em.L.append(f" *")
    em.L.append(f" * FFTW-style codelet for recursive CT executor.")
    em.L.append(f" * Generated by gen_radix64.py --variant {ct_variant}")
    em.L.append(f" */")
    em.L.append(f"")
    em.L.append(f"#ifndef {guard}")
    em.L.append(f"#define {guard}")
    em.L.append(f"")
    em.L.append(f"#include <stddef.h>")
    if isa.name != 'scalar':
        em.L.append(f"#include <immintrin.h>")
    em.L.append(f"")

    em.L.extend(emit_itw_static_arrays())

    _emit_ld_st_macros(em.L, isa, 'R64CT')

    for d in ['fwd', 'bwd']:
        em.reset_counters()
        em.addr_mode = 'n1' if is_n1 else 't1'

        if isa.target_attr:
            em.L.append(f"static {isa.target_attr} void")
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
        else:
            em.L.append(f"{func_base}_{d}_{isa.name}(")
            em.L.append(f"    double * __restrict__ rio_re, double * __restrict__ rio_im,")
            em.L.append(f"    const double * __restrict__ W_re, const double * __restrict__ W_im,")
            if isa.name == 'scalar':
                em.L.append(f"    size_t ios, size_t mb, size_t me, size_t ms)")
            else:
                em.L.append(f"    size_t ios, size_t me)")

        em.L.append(f"{{")
        em.ind = 1

        _emit_common_boilerplate(em, isa, T, nfuse)

        # Loop
        if is_n1:
            if isa.name == 'scalar':
                em.o(f"for (size_t k = 0; k < vl; k++) {{")
            else:
                em.o(f"for (size_t k = 0; k < vl; k += {isa.k_step}) {{")
        else:
            if isa.name == 'scalar':
                em.o(f"for (size_t m = mb; m < me; m++) {{")
            else:
                em.o(f"for (size_t m = 0; m < me; m += {isa.k_step}) {{")

        em.ind += 1
        if is_n1:
            emit_notw_kernel(em, d, nfuse)
        elif is_t1_dif:
            emit_dif_tw_flat_kernel(em, d, nfuse)
        elif is_t1_dit_log3:
            emit_dit_tw_log3_kernel(em, d, nfuse)
        else:
            emit_dit_tw_flat_kernel(em, d, nfuse)
        em.ind -= 1
        em.o("}")
        em.L.append("}")
        em.L.append("")

    if isa.name != 'scalar':
        em.L.append("#undef LD"); em.L.append("#undef ST"); em.L.append("")
    em.L.append(f"#endif /* {guard} */")
    return em.L


# ═══════════════════════════════════════════════════════════════
# SV CODELET GENERATION — text transform from t2 output
# ═══════════════════════════════════════════════════════════════

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

    text = '\n'.join(t2_lines)

    if variant == 'notw':
        t2_pattern = 'radix64_n1_dit_kernel'
        sv_name = 'radix64_n1sv_kernel'
    elif variant == 'dit_tw':
        t2_pattern = 'radix64_tw_flat_dit_kernel'
        sv_name = 'radix64_t1sv_dit_kernel'
    elif variant == 'dif_tw':
        t2_pattern = 'radix64_tw_flat_dif_kernel'
        sv_name = 'radix64_t1sv_dif_kernel'
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
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Unified R=64 codelet generator')
    parser.add_argument('--isa', default='avx2',
                        choices=['scalar', 'avx2', 'avx512', 'all'])
    parser.add_argument('--variant', default='notw',
                        choices=['notw', 'dit_tw', 'dif_tw', 'dit_tw_log3', 'dif_tw_log3',
                                 'ct_n1', 'ct_t1_dit', 'ct_t1_dit_log3', 'ct_t1_dif', 'all'])
    # Legacy positional
    parser.add_argument('isa_pos', nargs='?', default=None)
    args = parser.parse_args()

    if args.isa_pos and args.isa_pos in ('scalar', 'avx2', 'avx512'):
        args.isa = args.isa_pos

    if args.isa == 'all':
        targets = [ISA_SCALAR, ISA_AVX2, ISA_AVX512]
    else:
        targets = [ALL_ISA[args.isa]]

    def print_file(lines, label, stats=None, isa_obj=None, variant_name=None):
        # Insert sv variants before #undef LD
        if isa_obj and variant_name:
            sv_lines = emit_sv_variants(lines, isa_obj, variant_name)
            if sv_lines:
                for i in range(len(lines)):
                    if lines[i].strip() == '#undef LD':
                        lines[i:i] = sv_lines
                        break
        print("\n".join(lines))
        if stats:
            print(f"\n{'='*72}", file=sys.stderr)
            print(f"  {label} -- Operation Counts (per k-step)", file=sys.stderr)
            print(f"{'='*72}", file=sys.stderr)
            print(f"  {'kernel':<20s} {'add':>5s} {'sub':>5s} {'mul':>5s} {'neg':>5s}"
                  f" {'fma':>5s} {'fms':>5s} | {'arith':>6s} {'flops':>6s}"
                  f" | {'ld':>4s} {'st':>4s} {'sp':>4s} {'rl':>4s} {'mem':>5s}",
                  file=sys.stderr)
            print(f"  {'-'*20} {'-'*5} {'-'*5} {'-'*5} {'-'*5}"
                  f" {'-'*5} {'-'*5} + {'-'*6} {'-'*6}"
                  f" + {'-'*4} {'-'*4} {'-'*4} {'-'*4} {'-'*5}",
                  file=sys.stderr)
            for k in sorted(stats.keys()):
                s = stats[k]
                print(f"  {k:<20s} {s['add']:5d} {s['sub']:5d} {s['mul']:5d} {s['neg']:5d}"
                      f" {s['fma']:5d} {s['fms']:5d} | {s['total_arith']:6d} {s['flops']:6d}"
                      f" | {s['load']:4d} {s['store']:4d} {s['spill']:4d} {s['reload']:4d} {s['total_mem']:5d}",
                      file=sys.stderr)
            print(f"{'='*72}", file=sys.stderr)

    for isa in targets:
        if args.variant in ('notw', 'all'):
            lines, stats = emit_notw_file(isa)
            print_file(lines, f"{isa.name.upper()} NOTW", stats, isa, 'notw')

        if args.variant in ('dit_tw', 'all'):
            lines, stats = emit_dit_tw_flat_file(isa)
            print_file(lines, f"{isa.name.upper()} DIT TW", stats, isa, 'dit_tw')

        if args.variant in ('dif_tw', 'all'):
            lines, stats = emit_dif_tw_flat_file(isa)
            print_file(lines, f"{isa.name.upper()} DIF TW", stats, isa, 'dif_tw')

        if args.variant in ('dit_tw_log3', 'all'):
            lines, stats = emit_dit_tw_log3_file(isa)
            print_file(lines, f"{isa.name.upper()} DIT TW LOG3", stats)

        if args.variant in ('dif_tw_log3', 'all'):
            lines, stats = emit_dif_tw_log3_file(isa)
            print_file(lines, f"{isa.name.upper()} DIF TW LOG3", stats)

        if args.variant in ('ct_n1', 'all'):
            lines = emit_ct_file(isa, 'ct_n1')
            print_file(lines, f"{isa.name.upper()} CT N1")

        if args.variant in ('ct_t1_dit', 'all') and isa.name != 'scalar':
            lines = emit_ct_file(isa, 'ct_t1_dit')
            print_file(lines, f"{isa.name.upper()} CT T1 DIT")

        if args.variant in ('ct_t1_dit_log3', 'all') and isa.name != 'scalar':
            lines = emit_ct_file(isa, 'ct_t1_dit_log3')
            print_file(lines, f"{isa.name.upper()} CT T1 DIT LOG3")

        if args.variant in ('ct_t1_dif', 'all') and isa.name != 'scalar':
            lines = emit_ct_file(isa, 'ct_t1_dif')
            print_file(lines, f"{isa.name.upper()} CT T1 DIF")


if __name__ == '__main__':
    main()
