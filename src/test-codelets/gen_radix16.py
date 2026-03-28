#!/usr/bin/env python3
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

    def _in_addr(self, n, ke="k"):
        if self.addr_mode == 'n1':
            return f"{n}*is+{ke}"
        elif self.addr_mode == 't1':
            return f"m*ms+{n}*ios"  # ke unused for t1
        return f"{n}*K+{ke}"

    def _out_addr(self, m, ke="k"):
        if self.addr_mode == 'n1':
            return f"{m}*os+{ke}"
        elif self.addr_mode == 't1':
            return f"m*ms+{m}*ios"
        return f"{m}*K+{ke}"

    def _in_buf(self):
        return "rio_re" if self.addr_mode == 't1' else "in_re"

    def _in_buf_im(self):
        return "rio_im" if self.addr_mode == 't1' else "in_im"

    def _out_buf(self):
        return "rio_re" if self.addr_mode == 't1' else "out_re"

    def _out_buf_im(self):
        return "rio_im" if self.addr_mode == 't1' else "out_im"

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

    # ── Complex multiply (log3 derivation) ──
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
#
# Bases: W^1, W^2, W^4, W^8 (loaded from table, indices 0-3)
# Per sub-FFT derivation chain:
#   n2=0: W^4(base), W^8(base), W^12=W^4*W^8           [1 cmul]
#   n2=1: W^1(base), W^5=W^1*W^4, W^9=W^1*W^8,
#          W^13=W^5*W^8                                  [3 cmuls]
#   n2=2: W^2(base), W^6=W^2*W^4, W^10=W^2*W^8,
#          W^14=W^6*W^8                                  [3 cmuls]
#   n2=3: W^3=W^1*W^2, W^7=W^3*W^4, W^11=W^3*W^8,
#          W^15=W^7*W^8                                  [4 cmuls]
# Total: 11 cmuls = 44 FMA-port ops
# ═══════════════════════════════════════════════════════════════

# For each sub-FFT n2, for each n1=0..3:
#   (n1, twiddle_source)
# twiddle_source = None (no twiddle, n=0) or
#   ('base', base_idx) or
#   ('derive', target_name, a_name, b_name)
#
# Names: 'b1','b2','b4','b8' for bases, 'w3'..'w15' for derived

LOG3_BASE_MAP = {'b1': 0, 'b2': 1, 'b4': 2, 'b8': 3}

def _log3_subfft_chain(n2):
    """Return [(n1, tw_action), ...] for sub-FFT n2.
    tw_action = None | ('base', name) | ('derive', wname, a, b) | ('use', wname)
    After 'derive', the result is available as wname for subsequent 'use'."""
    if n2 == 0:
        return [
            (0, None),                                    # n=0
            (1, ('base', 'b4')),                          # n=4, W^4 base
            (2, ('base', 'b8')),                          # n=8, W^8 base
            (3, ('derive', 'w12', 'b4', 'b8')),          # n=12, W^12=W^4*W^8
        ]
    elif n2 == 1:
        return [
            (0, ('base', 'b1')),                          # n=1, W^1 base
            (1, ('derive', 'w5', 'b1', 'b4')),           # n=5, W^5=W^1*W^4
            (2, ('derive', 'w9', 'b1', 'b8')),           # n=9, W^9=W^1*W^8
            (3, ('derive', 'w13', 'w5', 'b8')),          # n=13, W^13=W^5*W^8
        ]
    elif n2 == 2:
        return [
            (0, ('base', 'b2')),                          # n=2, W^2 base
            (1, ('derive', 'w6', 'b2', 'b4')),           # n=6, W^6=W^2*W^4
            (2, ('derive', 'w10', 'b2', 'b8')),          # n=10, W^10=W^2*W^8
            (3, ('derive', 'w14', 'w6', 'b8')),          # n=14, W^14=W^6*W^8
        ]
    else:  # n2 == 3
        return [
            (0, ('derive', 'w3', 'b1', 'b2')),           # n=3, W^3=W^1*W^2
            (1, ('derive', 'w7', 'w3', 'b4')),           # n=7, W^7=W^3*W^4
            (2, ('derive', 'w11', 'w3', 'b8')),          # n=11, W^11=W^3*W^8
            (3, ('derive', 'w15', 'w7', 'b8')),          # n=15, W^15=W^7*W^8
        ]


def emit_kernel_body_log3(em, d, itw_set, variant):
    """Emit DIT or DIF log3 kernel body. AVX-512 only."""
    assert variant in ('dit_tw_log3', 'dif_tw_log3')
    is_dit = variant == 'dit_tw_log3'
    T = em.isa.T
    xv4 = [f"x{i}" for i in range(N1)]
    ld = f"{em.isa.p}_load_pd"

    # Load 4 base twiddles
    em.c("Load 4 log3 base twiddles: W^1, W^2, W^4, W^8")
    for name, idx in sorted(LOG3_BASE_MAP.items(), key=lambda x: x[1]):
        em.o(f"const {T} {name}_re = {ld}(&base_tw_re[{idx}*K+k]);")
        em.o(f"const {T} {name}_im = {ld}(&base_tw_im[{idx}*K+k]);")
    em.n_load += 8
    em.b()

    # Declare derived twiddle scratch (all sub-FFTs share these names,
    # but they're overwritten each sub-FFT)
    em.o(f"{T} tw_dr, tw_di;")  # temp for derivation
    em.b()

    # PASS 1
    for n2 in range(N2):
        chain = _log3_subfft_chain(n2)
        em.c(f"sub-FFT n2={n2}")

        # Track derived twiddle vars alive in this sub-FFT
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
                    # Derive then apply
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

    # PASS 2 (same as flat — internal twiddles + radix-4)
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

        # DIF: post-twiddle outputs using log3 derived twiddles
        if not is_dit:
            # Need to re-derive twiddles for outputs m=1..15
            # Outputs are m = k1 + N1*k2 for k2=0..3
            for k2 in range(N2):
                m = k1 + N1 * k2
                if m == 0:
                    continue
                # Map m to its twiddle source
                # m's binary decomposition → chain of bases
                tw_pairs = []
                bits = []
                for bit in [1, 2, 4, 8]:
                    if m & bit:
                        bits.append(bit)
                if len(bits) == 1:
                    # Single base
                    bname = f"b{bits[0]}"
                    em.emit_cmul_inplace(f"x{k2}", f"{bname}_re", f"{bname}_im", d)
                else:
                    # Need to derive: accumulate product
                    em.o(f"{{ {T} pr = b{bits[0]}_re, pi = b{bits[0]}_im;")
                    for i in range(1, len(bits)):
                        em.emit_cmul("tw_dr", "tw_di",
                                     "pr", "pi",
                                     f"b{bits[i]}_re", f"b{bits[i]}_im", 'fwd')
                        em.o(f"pr = tw_dr; pi = tw_di;")
                    em.emit_cmul_inplace(f"x{k2}", "pr", "pi", d)
                    em.o(f"}}")
            em.b()

        for k2 in range(N2):
            em.emit_store(f"x{k2}", k1 + N1 * k2)
        em.b()

def emit_kernel_body(em, d, itw_set, variant):
    """Emit the inner loop body. variant: 'notw', 'dit_tw', 'dif_tw'."""
    xv4 = [f"x{i}" for i in range(N1)]

    # PASS 1: N2 radix-4 sub-FFTs
    for n2 in range(N2):
        em.c(f"sub-FFT n2={n2}")
        for n1 in range(N1):
            n = N2 * n1 + n2
            em.emit_load(f"x{n1}", n)
            if variant == 'dit_tw' and n > 0:
                em.emit_ext_tw(f"x{n1}", n - 1, d)
        em.b()
        em.emit_radix4(xv4, d, f"radix-4 n2={n2}")
        em.b()
        for k1 in range(N1):
            em.emit_spill(f"x{k1}", n2 * N1 + k1)
        em.b()

    # PASS 2: N1 radix-4 column combines
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
        # DIF: external twiddle on outputs
        if variant == 'dif_tw':
            for k2 in range(N2):
                m = k1 + N1 * k2
                if m > 0:
                    em.emit_ext_tw(f"x{k2}", m - 1, d)
            em.b()
        for k2 in range(N2):
            em.emit_store(f"x{k2}", k1 + N1 * k2)
        em.b()


# ═══════════════════════════════════════════════════════════════
# FILE EMITTER
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
    """Emit complete header for one ISA and one variant."""
    em = Emitter(isa)
    T = isa.T
    is_log3 = variant.endswith('_log3')

    # Function name mapping
    if variant == 'notw':
        func_base = 'radix16_n1_dit_kernel'
        tw_params = None
    elif variant == 'dit_tw':
        func_base = 'radix16_tw_flat_dit_kernel'
        tw_params = 'flat'
    elif variant == 'dif_tw':
        func_base = 'radix16_tw_flat_dif_kernel'
        tw_params = 'flat'
    elif variant == 'dit_tw_log3':
        func_base = 'radix16_tw_log3_dit_kernel'
        tw_params = 'log3'
    elif variant == 'dif_tw_log3':
        func_base = 'radix16_tw_log3_dif_kernel'
        tw_params = 'log3'

    # File header
    vname = {
        'notw': 'N1 (no twiddle)', 'dit_tw': 'DIT twiddled (flat)',
        'dif_tw': 'DIF twiddled (flat)',
        'dit_tw_log3': 'DIT twiddled (log3 derived)',
        'dif_tw_log3': 'DIF twiddled (log3 derived)',
    }[variant]
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

    # Twiddle constants
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
        elif tw_params == 'log3':
            em.L.append(f"    const double * __restrict__ base_tw_re, const double * __restrict__ base_tw_im,")
        em.L.append(f"    size_t K)")
        em.L.append(f"{{")
        em.ind = 1

        # Constants
        if isa.name != 'scalar':
            em.o(f"const {T} sign_flip = {isa.p}_set1_pd(-0.0);")
            em.o(f"const {T} sqrt2_inv = {isa.p}_set1_pd(0.70710678118654752440);")
        else:
            em.o(f"const double sqrt2_inv = 0.70710678118654752440;")
        em.b()

        # Spill buffer
        spill_total = N * isa.sm
        if isa.name == 'scalar':
            em.o(f"double spill_re[{N}], spill_im[{N}];")
        else:
            em.o(f"{isa.align} double spill_re[{spill_total}];")
            em.o(f"{isa.align} double spill_im[{spill_total}];")
        em.b()

        # Working registers
        em.o(f"{T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
        em.b()

        # Hoisted internal twiddle broadcasts
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

        # K loop
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
        # Find the for-loop line — skip it and start tracking depth
        if not in_loop and 'for (size_t k' in stripped and 'k < K' in stripped:
            in_loop = True
            depth = 1  # The '{' at end of for-line
            continue
        if in_loop:
            # Count braces on this line
            for ch in stripped:
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
            # When depth reaches 0, we've found the for-loop's closing '}'
            if depth <= 0:
                in_loop = False
                if stripped == '}':
                    continue  # Skip the bare closing brace
                # If '}' is embedded in a line with code, still process the line
        # Transform addressing
        line = re.sub(r'(\d+)\*K\+k', r'\1*vs', line)
        line = re.sub(r'\[k\]', '[0]', line)
        # Dedent by 4 spaces (was inside for-loop)
        if line.startswith('        '):
            line = line[4:]
        out.append(line)
    return '\n'.join(out)


def emit_sv_variants(t2_lines, isa, variant):
    """Extract t2 functions from generated lines, emit sv versions."""
    if isa.name == 'scalar':
        return []
    if variant.endswith('_log3'):
        return []  # No sv for log3

    text = '\n'.join(t2_lines)

    # Map variant to function name patterns
    if variant == 'notw':
        t2_pattern = 'radix16_n1_dit_kernel'
        sv_name = 'radix16_n1sv_kernel'
    elif variant == 'dit_tw':
        t2_pattern = 'radix16_tw_flat_dit_kernel'
        sv_name = 'radix16_t1sv_dit_kernel'
    elif variant == 'dif_tw':
        t2_pattern = 'radix16_tw_flat_dif_kernel'
        sv_name = 'radix16_t1sv_dif_kernel'
    else:
        return []

    out = []
    out.append('')
    out.append(f'/* === sv codelets: no loop, elements at stride vs === */')
    out.append(f'/* Executor calls K/{isa.k_step} times, advancing base pointers by {isa.k_step}. */')

    for d in ['fwd', 'bwd']:
        func_name = f'{t2_pattern}_{d}_{isa.name}'
        sv_func_name = f'{sv_name}_{d}_{isa.name}'

        # Find the function in the text
        func_start = text.find(f'{func_name}(')
        if func_start < 0:
            continue

        # Walk back to find 'static'
        static_start = text.rfind('static', 0, func_start)
        if static_start < 0:
            continue

        # Find function body: first '{' after signature, then matching '}'
        brace_start = text.find('{', func_start)
        if brace_start < 0:
            continue

        # Find matching closing brace
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

        # Transform body
        sv_body = _t2_to_sv(func_body)

        # Build sv function signature
        sig = text[static_start:brace_start]
        # Replace function name
        sig = sig.replace(func_name, sv_func_name)
        # Replace 'size_t K)' with 'size_t vs)'
        sig = sig.replace('size_t K)', 'size_t vs)')

        out.append(sig + '{')
        out.append(sv_body)
        out.append('}')
        out.append('')

    return out


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def emit_file_ct(isa, itw_set, ct_variant):
    """Emit FFTW-style n1 or t1_dit codelet."""
    em = Emitter(isa)
    T = isa.T

    is_n1 = ct_variant == 'ct_n1'
    is_t1 = ct_variant == 'ct_t1_dit'
    em.addr_mode = 'n1' if is_n1 else 't1'

    func_base = f"radix16_n1" if is_n1 else f"radix16_t1_dit"
    vname = "n1 (separate is/os)" if is_n1 else "t1 DIT (in-place twiddle)"
    guard = f"FFT_RADIX16_{isa.name.upper()}_CT_{ct_variant.upper()}_H"

    em.L.append(f"/**")
    em.L.append(f" * @file fft_radix16_{isa.name}_{ct_variant}.h")
    em.L.append(f" * @brief DFT-16 {isa.name.upper()} {vname}")
    em.L.append(f" *")
    em.L.append(f" * FFTW-style codelet for recursive CT executor.")
    em.L.append(f" * Generated by gen_radix16.py --variant {ct_variant}")
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

        # Constants
        if isa.name != 'scalar':
            em.o(f"const {T} sign_flip = {isa.p}_set1_pd(-0.0);")
            em.o(f"const {T} sqrt2_inv = {isa.p}_set1_pd(0.70710678118654752440);")
        else:
            em.o(f"const double sqrt2_inv = 0.70710678118654752440;")
        em.b()

        # Spill buffer
        spill_total = N * isa.sm
        if isa.name == 'scalar':
            em.o(f"double spill_re[{N}], spill_im[{N}];")
        else:
            em.o(f"{isa.align} double spill_re[{spill_total}];")
            em.o(f"{isa.align} double spill_im[{spill_total}];")
        em.b()

        em.o(f"{T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
        em.b()

        # Hoisted internal twiddle broadcasts
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
        # Reuse the same kernel body — addressing differs via addr_mode
        kernel_variant = 'notw' if is_n1 else 'dit_tw'
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


def main():
    parser = argparse.ArgumentParser(description='Unified R=16 codelet generator')
    parser.add_argument('--isa', default='avx512',
                        choices=['scalar', 'avx2', 'avx512', 'all'])
    parser.add_argument('--variant', default='dit_tw',
                        choices=['notw', 'dit_tw', 'dif_tw', 'dit_tw_log3', 'dif_tw_log3',
                                 'ct_n1', 'ct_t1_dit', 'all'])
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

    for isa in targets:
        for variant in variants:
            if variant.startswith('ct_'):
                lines = emit_file_ct(isa, itw_set, variant)
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


if __name__ == '__main__':
    main()
