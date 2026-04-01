#!/usr/bin/env python3
"""
gen_radix11.py -- Unified DFT-11 codelet generator for VectorFFT

Monolithic DFT-11 butterfly (straight-line genfft DAG translation).
10 constants (no sign_flip-free path — all S-terms use xor-negate).

DFT-11 forward butterfly:
  Load pairs: (x1,x10), (x5,x6), (x4,x7), (x3,x8), (x2,x9)
  Sums: T4=x1+x10, Tg=x5+x6, Td=x4+x7, Ta=x3+x8, T7=x2+x9
  Diffs: Ti=x10-x1, Tl=x6-x5, Tk=x7-x4, Tj=x8-x3, Tm=x9-x2
  y0 = x0 + T4 + T7 + Ta + Td + Tg
  5 output pairs (R_m +/- S_m):
    Pair (4,7):  Th, Tn
    Pair (5,6):  Tu, Tv
    Pair (3,8):  Ts, Tt
    Pair (1,10): Tq, Tr
    Pair (2,9):  To, Tp

10 constants (genfft names):
  KP654860733, KP142314838, KP959492973, KP415415013, KP841253532,
  KP989821441, KP909631995, KP281732556, KP540640817, KP755749574

Backward butterfly swaps which index gets +S and which gets -S for each pair.

AVX2 spill budget:
  Phase 1: compute 5 sums + 5 diffs, spill diffs (slots 0-4)
  Phase 2: compute y0 + 5 R terms from {x0, sums}, spill R terms (slots 5-9)
  Phase 3: one pair at a time — reload R, reload 1 diff, compute S, combine, store
  Total: 10 spills + 10 reloads, peak register use stays <= 16 YMM.

AVX-512 no-spill:
  10 const + x0(2) + 5 sums(10) + 5 diffs(10) = 32. Exactly fits in 32 ZMM.

Log3 twiddle for R=11 (9 cmuls from w1):
  w2=w1*w1, w3=w1*w2, w4=w1*w3, w5=w1*w4,
  w6=w3*w3, w7=w3*w4, w8=w4*w4, w9=w4*w5, w10=w5*w5

Usage:
  python3 gen_radix11.py --isa avx2 --variant all
  python3 gen_radix11.py --isa all --variant ct_n1
  python3 gen_radix11.py --isa avx2 --variant ct_t1_dit
"""

import sys, math, argparse, re

R = 11

# DFT-11 constants (from genfft)
KP654860733_val = +0.654860733945285064056925072466293553183791199
KP142314838_val = +0.142314838273285140443792668616369668791051361
KP959492973_val = +0.959492973614497389890368057066327699062454848
KP415415013_val = +0.415415013001886425529274149229623203524004910
KP841253532_val = +0.841253532831181168861811648919367717513292498
KP989821441_val = +0.989821441880932732376092037776718787376519372
KP909631995_val = +0.909631995354518371411715383079028460060241051
KP281732556_val = +0.281732556841429697711417915346616899035777899
KP540640817_val = +0.540640817455597582107635954318691695431770608
KP755749574_val = +0.755749574354258283774035843972344420179717445

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

ISA_SCALAR = ISAConfig('scalar', 'double', 1, 1, '', 1, '', '', 'R11S')
ISA_AVX2   = ISAConfig('avx2', '__m256d', 4, 4, '_mm256', 4,
    '__attribute__((target("avx2,fma")))',
    '__attribute__((aligned(32)))', 'R11A')
ISA_AVX512 = ISAConfig('avx512', '__m512d', 8, 8, '_mm512', 8,
    '__attribute__((target("avx512f,avx512dq,fma")))',
    '__attribute__((aligned(64)))', 'R11L')

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

    # -- DFT-11 butterfly --
    def emit_radix11_butterfly(self, d, out_names=None):
        """Emit DFT-11 butterfly, translating genfft DAG exactly.

        AVX2/scalar: 3-phase Sethi-Ullman with spill buffer (peak 16 YMM).
        AVX-512: single-pass no-spill (all 32 ZMM).
        out_names: if given, list of 11 output variable names.
                   Default: overwrite x0..x10.
        """
        fwd = (d == 'fwd')
        T = self.isa.T

        if out_names is None:
            out_names = [f'x{i}' for i in range(11)]

        if self.isa.name == 'avx512':
            return self._emit_radix11_butterfly_nospill(d, out_names)

        self.c(f"DFT-11 butterfly [{d}] (AVX2/scalar: 3-phase + spill)")

        # -------------------------------------------------------
        # Phase 1: load pairs, compute sums + diffs, spill diffs
        # -------------------------------------------------------
        # Inputs x0..x10 are already in registers (loaded by caller).
        # Working vars: sum4/Ti (pair 1,10), sumg/Tl (pair 5,6),
        #               sumd/Tk (pair 4,7), suma/Tj (pair 3,8), sum7/Tm (pair 2,9)
        # Naming matches genfft: T4=sum, Ti=diff for pair (x1,x10); etc.

        sm = self.isa.sm

        self.c("Phase 1 — sums and diffs of conjugate pairs")
        # pair (x1, x10) -> T4 = sum, Ti = diff  (diff = x10-x1)
        self.o(f"T4r={self.add('x1_re','x10_re')}; T4i={self.add('x1_im','x10_im')};")
        self.o(f"Tir={self.sub('x10_re','x1_re')}; Tii={self.sub('x10_im','x1_im')};")
        # pair (x5, x6) -> Tg = sum, Tl = diff  (diff = x6-x5)
        self.o(f"Tgr={self.add('x5_re','x6_re')}; Tgi={self.add('x5_im','x6_im')};")
        self.o(f"Tlr={self.sub('x6_re','x5_re')}; Tli={self.sub('x6_im','x5_im')};")
        # pair (x4, x7) -> Td = sum, Tk = diff  (diff = x7-x4)
        self.o(f"Tdr={self.add('x4_re','x7_re')}; Tdi={self.add('x4_im','x7_im')};")
        self.o(f"Tkr={self.sub('x7_re','x4_re')}; Tki={self.sub('x7_im','x4_im')};")
        # pair (x3, x8) -> Ta = sum, Tj = diff  (diff = x8-x3)
        self.o(f"Tar={self.add('x3_re','x8_re')}; Tai={self.add('x3_im','x8_im')};")
        self.o(f"Tjr={self.sub('x8_re','x3_re')}; Tji={self.sub('x8_im','x3_im')};")
        # pair (x2, x9) -> T7 = sum, Tm = diff  (diff = x9-x2)
        self.o(f"T7r={self.add('x2_re','x9_re')}; T7i={self.add('x2_im','x9_im')};")
        self.o(f"Tmr={self.sub('x9_re','x2_re')}; Tmi={self.sub('x9_im','x2_im')};")

        # Spill all 5 diffs: Ti->slot0, Tj->slot1, Tk->slot2, Tl->slot3, Tm->slot4
        self.c("Spill diffs Ti,Tj,Tk,Tl,Tm (slots 0-4)")
        if self.isa.name == 'scalar':
            self.o(f"spill_re[0]=Tir; spill_im[0]=Tii;")
            self.o(f"spill_re[1]=Tjr; spill_im[1]=Tji;")
            self.o(f"spill_re[2]=Tkr; spill_im[2]=Tki;")
            self.o(f"spill_re[3]=Tlr; spill_im[3]=Tli;")
            self.o(f"spill_re[4]=Tmr; spill_im[4]=Tmi;")
        else:
            self.o(f"{self.isa.p}_store_pd(&spill_re[0*{sm}],Tir);")
            self.o(f"{self.isa.p}_store_pd(&spill_im[0*{sm}],Tii);")
            self.o(f"{self.isa.p}_store_pd(&spill_re[1*{sm}],Tjr);")
            self.o(f"{self.isa.p}_store_pd(&spill_im[1*{sm}],Tji);")
            self.o(f"{self.isa.p}_store_pd(&spill_re[2*{sm}],Tkr);")
            self.o(f"{self.isa.p}_store_pd(&spill_im[2*{sm}],Tki);")
            self.o(f"{self.isa.p}_store_pd(&spill_re[3*{sm}],Tlr);")
            self.o(f"{self.isa.p}_store_pd(&spill_im[3*{sm}],Tli);")
            self.o(f"{self.isa.p}_store_pd(&spill_re[4*{sm}],Tmr);")
            self.o(f"{self.isa.p}_store_pd(&spill_im[4*{sm}],Tmi);")
        self.spill_c += 5

        # -------------------------------------------------------
        # Phase 2: y0 + 5 R (cosine) terms from {x0, T4, T7, Ta, Td, Tg}
        # -------------------------------------------------------
        self.b()
        self.c("Phase 2 — 5 R (cosine) terms THEN y0 (compute R before overwriting x0)")

        # R terms — exact genfft expression translated to FMA chains.
        # Each R is a linear combination of x0 and the 5 sums (T4,T7,Ta,Td,Tg).
        # We chain fnma (subtract = fnmadd) from innermost out.
        #
        # Pair (4,7): Th = KP841253532*Ta + KP415415013*Tg + ((x0 - KP654860733*T4) - KP142314838*T7) - KP959492973*Td)
        #   = fma(KP841,Ta, fma(KP415,Tg, fnma(KP959,Td, fnma(KP142,T7, fnma(KP654,T4, x0)))))
        self.o(f"Thr={self.fma('KP841253532','Tar',self.fma('KP415415013','Tgr',self.fnma('KP959492973','Tdr',self.fnma('KP142314838','T7r',self.fnma('KP654860733','T4r','x0_re')))))};")
        self.o(f"Thi={self.fma('KP841253532','Tai',self.fma('KP415415013','Tgi',self.fnma('KP959492973','Tdi',self.fnma('KP142314838','T7i',self.fnma('KP654860733','T4i','x0_im')))))};")

        # Pair (5,6): Tu = KP841253532*T7 + KP415415013*Td + ((x0 - KP959492973*T4) - KP654860733*Ta) - KP142314838*Tg)
        self.o(f"Tur={self.fma('KP841253532','T7r',self.fma('KP415415013','Tdr',self.fnma('KP142314838','Tgr',self.fnma('KP654860733','Tar',self.fnma('KP959492973','T4r','x0_re')))))};")
        self.o(f"Tui={self.fma('KP841253532','T7i',self.fma('KP415415013','Tdi',self.fnma('KP142314838','Tgi',self.fnma('KP654860733','Tai',self.fnma('KP959492973','T4i','x0_im')))))};")

        # Pair (3,8): Ts = KP415415013*Ta + KP841253532*Td + ((x0 - KP142314838*T4) - KP959492973*T7) - KP654860733*Tg)
        self.o(f"Tsr={self.fma('KP415415013','Tar',self.fma('KP841253532','Tdr',self.fnma('KP654860733','Tgr',self.fnma('KP959492973','T7r',self.fnma('KP142314838','T4r','x0_re')))))};")
        self.o(f"Tsi={self.fma('KP415415013','Tai',self.fma('KP841253532','Tdi',self.fnma('KP654860733','Tgi',self.fnma('KP959492973','T7i',self.fnma('KP142314838','T4i','x0_im')))))};")

        # Pair (1,10): Tq = KP841253532*T4 + KP415415013*T7 + ((x0 - KP142314838*Ta) - KP654860733*Td) - KP959492973*Tg)
        self.o(f"Tqr={self.fma('KP841253532','T4r',self.fma('KP415415013','T7r',self.fnma('KP959492973','Tgr',self.fnma('KP654860733','Tdr',self.fnma('KP142314838','Tar','x0_re')))))};")
        self.o(f"Tqi={self.fma('KP841253532','T4i',self.fma('KP415415013','T7i',self.fnma('KP959492973','Tgi',self.fnma('KP654860733','Tdi',self.fnma('KP142314838','Tai','x0_im')))))};")

        # Pair (2,9): To = KP415415013*T4 + KP841253532*Tg + ((x0 - KP654860733*T7) - KP959492973*Ta) - KP142314838*Td)
        self.o(f"Tor={self.fma('KP415415013','T4r',self.fma('KP841253532','Tgr',self.fnma('KP142314838','Tdr',self.fnma('KP959492973','Tar',self.fnma('KP654860733','T7r','x0_re')))))};")
        self.o(f"Toi={self.fma('KP415415013','T4i',self.fma('KP841253532','Tgi',self.fnma('KP142314838','Tdi',self.fnma('KP959492973','Tai',self.fnma('KP654860733','T7i','x0_im')))))};")

        # y0 = x0 + T4 + T7 + Ta + Td + Tg  [NOW safe to overwrite x0]
        y0 = out_names[0]
        self.o(f"{y0}_re={self.add('x0_re',self.add('T4r',self.add('T7r',self.add('Tar',self.add('Tdr','Tgr')))))};")
        self.o(f"{y0}_im={self.add('x0_im',self.add('T4i',self.add('T7i',self.add('Tai',self.add('Tdi','Tgi')))))};")

        # Spill R terms: Th->slot5, Tu->slot6, Ts->slot7, Tq->slot8, To->slot9
        self.c("Spill R terms Th,Tu,Ts,Tq,To (slots 5-9)")
        if self.isa.name == 'scalar':
            self.o(f"spill_re[5]=Thr; spill_im[5]=Thi;")
            self.o(f"spill_re[6]=Tur; spill_im[6]=Tui;")
            self.o(f"spill_re[7]=Tsr; spill_im[7]=Tsi;")
            self.o(f"spill_re[8]=Tqr; spill_im[8]=Tqi;")
            self.o(f"spill_re[9]=Tor; spill_im[9]=Toi;")
        else:
            self.o(f"{self.isa.p}_store_pd(&spill_re[5*{sm}],Thr);")
            self.o(f"{self.isa.p}_store_pd(&spill_im[5*{sm}],Thi);")
            self.o(f"{self.isa.p}_store_pd(&spill_re[6*{sm}],Tur);")
            self.o(f"{self.isa.p}_store_pd(&spill_im[6*{sm}],Tui);")
            self.o(f"{self.isa.p}_store_pd(&spill_re[7*{sm}],Tsr);")
            self.o(f"{self.isa.p}_store_pd(&spill_im[7*{sm}],Tsi);")
            self.o(f"{self.isa.p}_store_pd(&spill_re[8*{sm}],Tqr);")
            self.o(f"{self.isa.p}_store_pd(&spill_im[8*{sm}],Tqi);")
            self.o(f"{self.isa.p}_store_pd(&spill_re[9*{sm}],Tor);")
            self.o(f"{self.isa.p}_store_pd(&spill_im[9*{sm}],Toi);")
        self.spill_c += 5

        # -------------------------------------------------------
        # Phase 3: for each pair, reload R + diffs, compute S, combine, store
        # -------------------------------------------------------
        # S terms are the sine cross-combinations:
        #   S_re   = -(combination of _im of diffs)  [negated in genfft via leading minus]
        #   S_im   =  (combination of _re of diffs)  [positive]
        # fwd: y_lo = R + S,  y_hi = R - S   (where lo < hi index)
        # bwd: y_lo = R - S,  y_hi = R + S   (reversed sign of S cross)
        #
        # From genfft fwd:
        #   out[4]  = Th + Tn;  out[7]  = Th - Tn   (pair 4,7)
        #   out[5]  = Tu + Tv;  out[6]  = Tu - Tv   (pair 5,6)
        #   out[3]  = Ts + Tt;  out[8]  = Ts - Tt   (pair 3,8)
        #   out[1]  = Tq + Tr;  out[10] = Tq - Tr   (pair 1,10)
        #   out[2]  = To + Tp;  out[9]  = To - Tp   (pair 2,9)
        #
        # S expressions (fwd, from scalar genfft, negating the _re part):
        #   Tn_re = -(KP755749574*Ti_im + KP540640817*Tj_im + KP281732556*Tk_im - KP989821441*Tm_im - KP909631995*Tl_im)
        #   Tn_im =   KP755749574*Ti_re + KP540640817*Tj_re + KP281732556*Tk_re - KP989821441*Tm_re - KP909631995*Tl_re
        #
        #   Tv_re = -(KP281732556*Ti_im + KP755749574*Tj_im + KP989821441*Tl_im - KP540640817*Tm_im - KP909631995*Tk_im)
        #   Tv_im =   KP281732556*Ti_re + KP755749574*Tj_re + KP989821441*Tl_re - KP540640817*Tm_re - KP909631995*Tk_re
        #
        #   Tt_re = -(KP989821441*Ti_im + KP540640817*Tk_im + KP755749574*Tl_im - KP281732556*Tm_im - KP909631995*Tj_im)
        #   Tt_im =   KP989821441*Ti_re + KP540640817*Tk_re + KP755749574*Tl_re - KP281732556*Tm_re - KP909631995*Tj_re
        #
        #   Tr_re = -(KP540640817*Ti_im + KP909631995*Tm_im + KP989821441*Tj_im + KP755749574*Tk_im + KP281732556*Tl_im)
        #   Tr_im =   KP540640817*Ti_re + KP909631995*Tm_re + KP989821441*Tj_re + KP755749574*Tk_re + KP281732556*Tl_re
        #
        #   Tp_re = -(KP909631995*Ti_im + KP755749574*Tm_im - KP281732556*Tj_im - KP989821441*Tk_im - KP540640817*Tl_im)
        #   Tp_im =   KP909631995*Ti_re + KP755749574*Tm_re - KP281732556*Tj_re - KP989821441*Tk_re - KP540640817*Tl_re
        #
        # bwd from genfft bwd (diff order is reversed within pairs, sum/diff semantics swap):
        #   Note: in bwd, Ti=x1-x10, Tj=x2-x9, Tk=x4-x7, Tl=x5-x6, Tm=x3-x8 (negated diffs)
        #   i.e. bwd_Ti_re = -fwd_Ti_re, etc. for all diffs.
        #   The bwd S expressions have the same constants but negated diffs -> equivalent to
        #   negating S, which swaps + and - in the combine step.
        #   Simpler: in bwd the store pairs are swapped:
        #     out[lo] = R - S,  out[hi] = R + S

        self.b()
        self.c("Phase 3 — S terms + combine (reload one diff set + R at a time)")

        # We need Ti_im for all 5 S terms. To minimise register pressure we
        # reload only what each S block needs: Ti is used in all 5, but we
        # keep it live only per scoped block. The genfft DAG uses Ti in all
        # 5 S computations, so we reload it each time (alternative: keep it
        # live and spill something else, but 1 extra reload per pair is cheaper).

        # -- Pair (4,7): Th (slot5) + Tn --
        self.b()
        self.c("Pair (4,7): reload Th, compute Tn, store out[4] and out[7]")
        if self.isa.name == 'scalar':
            self.o(f"Thr=spill_re[5]; Thi=spill_im[5];")
            self.o(f"Tir=spill_re[0]; Tii=spill_im[0];")
            self.o(f"Tjr=spill_re[1]; Tji=spill_im[1];")
            self.o(f"Tkr=spill_re[2]; Tki=spill_im[2];")
            self.o(f"Tlr=spill_re[3]; Tli=spill_im[3];")
            self.o(f"Tmr=spill_re[4]; Tmi=spill_im[4];")
        else:
            self.o(f"Thr={self.isa.p}_load_pd(&spill_re[5*{sm}]);")
            self.o(f"Thi={self.isa.p}_load_pd(&spill_im[5*{sm}]);")
            self.o(f"Tir={self.isa.p}_load_pd(&spill_re[0*{sm}]);")
            self.o(f"Tii={self.isa.p}_load_pd(&spill_im[0*{sm}]);")
            self.o(f"Tjr={self.isa.p}_load_pd(&spill_re[1*{sm}]);")
            self.o(f"Tji={self.isa.p}_load_pd(&spill_im[1*{sm}]);")
            self.o(f"Tkr={self.isa.p}_load_pd(&spill_re[2*{sm}]);")
            self.o(f"Tki={self.isa.p}_load_pd(&spill_im[2*{sm}]);")
            self.o(f"Tlr={self.isa.p}_load_pd(&spill_re[3*{sm}]);")
            self.o(f"Tli={self.isa.p}_load_pd(&spill_im[3*{sm}]);")
            self.o(f"Tmr={self.isa.p}_load_pd(&spill_re[4*{sm}]);")
            self.o(f"Tmi={self.isa.p}_load_pd(&spill_im[4*{sm}]);")
        self.reload_c += 6
        # Tn_im = KP755749574*Ti_re + KP540640817*Tj_re + KP281732556*Tk_re - KP989821441*Tm_re - KP909631995*Tl_re
        # Tn_re = -(above with _re -> _im)
        self.o(f"{{ {T} Tnr={self.neg(self.fma('KP755749574','Tii',self.fma('KP540640817','Tji',self.fnma('KP909631995','Tli',self.fnma('KP989821441','Tmi',self.mul('KP281732556','Tki'))))))};")
        self.o(f"  {T} Tni={self.fma('KP755749574','Tir',self.fma('KP540640817','Tjr',self.fnma('KP909631995','Tlr',self.fnma('KP989821441','Tmr',self.mul('KP281732556','Tkr')))))};")
        y4, y7 = out_names[4], out_names[7]
        if fwd:
            self.o(f"  {y4}_re={self.add('Thr','Tnr')}; {y4}_im={self.add('Thi','Tni')};")
            self.o(f"  {y7}_re={self.sub('Thr','Tnr')}; {y7}_im={self.sub('Thi','Tni')}; }}")
        else:
            self.o(f"  {y7}_re={self.add('Thr','Tnr')}; {y7}_im={self.add('Thi','Tni')};")
            self.o(f"  {y4}_re={self.sub('Thr','Tnr')}; {y4}_im={self.sub('Thi','Tni')}; }}")

        # -- Pair (5,6): Tu (slot6) + Tv --
        self.b()
        self.c("Pair (5,6): reload Tu, compute Tv, store out[5] and out[6]")
        if self.isa.name == 'scalar':
            self.o(f"Tur=spill_re[6]; Tui=spill_im[6];")
        else:
            self.o(f"Tur={self.isa.p}_load_pd(&spill_re[6*{sm}]);")
            self.o(f"Tui={self.isa.p}_load_pd(&spill_im[6*{sm}]);")
        self.reload_c += 1
        # Tv_re = -(KP281732556*Ti_im + KP755749574*Tj_im + KP989821441*Tl_im - KP540640817*Tm_im - KP909631995*Tk_im)
        # Tv_im =  KP281732556*Ti_re + KP755749574*Tj_re + KP989821441*Tl_re - KP540640817*Tm_re - KP909631995*Tk_re
        self.o(f"{{ {T} Tvr={self.neg(self.fma('KP281732556','Tii',self.fma('KP755749574','Tji',self.fnma('KP909631995','Tki',self.fnma('KP540640817','Tmi',self.mul('KP989821441','Tli'))))))};")
        self.o(f"  {T} Tvi={self.fma('KP281732556','Tir',self.fma('KP755749574','Tjr',self.fnma('KP909631995','Tkr',self.fnma('KP540640817','Tmr',self.mul('KP989821441','Tlr')))))};")
        y5, y6 = out_names[5], out_names[6]
        if fwd:
            self.o(f"  {y5}_re={self.add('Tur','Tvr')}; {y5}_im={self.add('Tui','Tvi')};")
            self.o(f"  {y6}_re={self.sub('Tur','Tvr')}; {y6}_im={self.sub('Tui','Tvi')}; }}")
        else:
            self.o(f"  {y6}_re={self.add('Tur','Tvr')}; {y6}_im={self.add('Tui','Tvi')};")
            self.o(f"  {y5}_re={self.sub('Tur','Tvr')}; {y5}_im={self.sub('Tui','Tvi')}; }}")

        # -- Pair (3,8): Ts (slot7) + Tt --
        self.b()
        self.c("Pair (3,8): reload Ts, compute Tt, store out[3] and out[8]")
        if self.isa.name == 'scalar':
            self.o(f"Tsr=spill_re[7]; Tsi=spill_im[7];")
        else:
            self.o(f"Tsr={self.isa.p}_load_pd(&spill_re[7*{sm}]);")
            self.o(f"Tsi={self.isa.p}_load_pd(&spill_im[7*{sm}]);")
        self.reload_c += 1
        # Tt_re = -(KP989821441*Ti_im + KP540640817*Tk_im + KP755749574*Tl_im - KP281732556*Tm_im - KP909631995*Tj_im)
        # Tt_im =   KP989821441*Ti_re + KP540640817*Tk_re + KP755749574*Tl_re - KP281732556*Tm_re - KP909631995*Tj_re
        self.o(f"{{ {T} Ttr={self.neg(self.fma('KP989821441','Tii',self.fma('KP540640817','Tki',self.fnma('KP909631995','Tji',self.fnma('KP281732556','Tmi',self.mul('KP755749574','Tli'))))))};")
        self.o(f"  {T} Tti={self.fma('KP989821441','Tir',self.fma('KP540640817','Tkr',self.fnma('KP909631995','Tjr',self.fnma('KP281732556','Tmr',self.mul('KP755749574','Tlr')))))};")
        y3, y8 = out_names[3], out_names[8]
        if fwd:
            self.o(f"  {y3}_re={self.add('Tsr','Ttr')}; {y3}_im={self.add('Tsi','Tti')};")
            self.o(f"  {y8}_re={self.sub('Tsr','Ttr')}; {y8}_im={self.sub('Tsi','Tti')}; }}")
        else:
            self.o(f"  {y8}_re={self.add('Tsr','Ttr')}; {y8}_im={self.add('Tsi','Tti')};")
            self.o(f"  {y3}_re={self.sub('Tsr','Ttr')}; {y3}_im={self.sub('Tsi','Tti')}; }}")

        # -- Pair (1,10): Tq (slot8) + Tr --
        self.b()
        self.c("Pair (1,10): reload Tq, compute Tr, store out[1] and out[10]")
        if self.isa.name == 'scalar':
            self.o(f"Tqr=spill_re[8]; Tqi=spill_im[8];")
        else:
            self.o(f"Tqr={self.isa.p}_load_pd(&spill_re[8*{sm}]);")
            self.o(f"Tqi={self.isa.p}_load_pd(&spill_im[8*{sm}]);")
        self.reload_c += 1
        # Tr_re = -(KP540640817*Ti_im + KP909631995*Tm_im + KP989821441*Tj_im + KP755749574*Tk_im + KP281732556*Tl_im)
        # Tr_im =   KP540640817*Ti_re + KP909631995*Tm_re + KP989821441*Tj_re + KP755749574*Tk_re + KP281732556*Tl_re
        self.o(f"{{ {T} Trr={self.neg(self.fma('KP540640817','Tii',self.fma('KP909631995','Tmi',self.fma('KP989821441','Tji',self.fma('KP755749574','Tki',self.mul('KP281732556','Tli'))))))};")
        self.o(f"  {T} Tri={self.fma('KP540640817','Tir',self.fma('KP909631995','Tmr',self.fma('KP989821441','Tjr',self.fma('KP755749574','Tkr',self.mul('KP281732556','Tlr')))))};")
        y1, y10 = out_names[1], out_names[10]
        if fwd:
            self.o(f"  {y1}_re={self.add('Tqr','Trr')}; {y1}_im={self.add('Tqi','Tri')};")
            self.o(f"  {y10}_re={self.sub('Tqr','Trr')}; {y10}_im={self.sub('Tqi','Tri')}; }}")
        else:
            self.o(f"  {y10}_re={self.add('Tqr','Trr')}; {y10}_im={self.add('Tqi','Tri')};")
            self.o(f"  {y1}_re={self.sub('Tqr','Trr')}; {y1}_im={self.sub('Tqi','Tri')}; }}")

        # -- Pair (2,9): To (slot9) + Tp --
        self.b()
        self.c("Pair (2,9): reload To, compute Tp, store out[2] and out[9]")
        if self.isa.name == 'scalar':
            self.o(f"Tor=spill_re[9]; Toi=spill_im[9];")
        else:
            self.o(f"Tor={self.isa.p}_load_pd(&spill_re[9*{sm}]);")
            self.o(f"Toi={self.isa.p}_load_pd(&spill_im[9*{sm}]);")
        self.reload_c += 1
        # Tp_re = -(KP909631995*Ti_im + KP755749574*Tm_im - KP281732556*Tj_im - KP989821441*Tk_im - KP540640817*Tl_im)
        # Tp_im =   KP909631995*Ti_re + KP755749574*Tm_re - KP281732556*Tj_re - KP989821441*Tk_re - KP540640817*Tl_re
        self.o(f"{{ {T} Tpr={self.neg(self.fma('KP909631995','Tii',self.fnma('KP540640817','Tli',self.fnma('KP989821441','Tki',self.fnma('KP281732556','Tji',self.mul('KP755749574','Tmi'))))))};")
        self.o(f"  {T} Tpi={self.fma('KP909631995','Tir',self.fnma('KP540640817','Tlr',self.fnma('KP989821441','Tkr',self.fnma('KP281732556','Tjr',self.mul('KP755749574','Tmr')))))};")
        y2, y9 = out_names[2], out_names[9]
        if fwd:
            self.o(f"  {y2}_re={self.add('Tor','Tpr')}; {y2}_im={self.add('Toi','Tpi')};")
            self.o(f"  {y9}_re={self.sub('Tor','Tpr')}; {y9}_im={self.sub('Toi','Tpi')}; }}")
        else:
            self.o(f"  {y9}_re={self.add('Tor','Tpr')}; {y9}_im={self.add('Toi','Tpi')};")
            self.o(f"  {y2}_re={self.sub('Tor','Tpr')}; {y2}_im={self.sub('Toi','Tpi')}; }}")

    def _emit_radix11_butterfly_nospill(self, d, out_names):
        """AVX-512 path: single-pass, all 32 ZMM. No spill buffer.
        10 const + x0(2) + 5 sums(10) + 5 diffs(10) = 32 ZMM exactly.
        Each output pair is computed in a scoped block, freeing registers as we go.
        """
        fwd = (d == 'fwd')
        T = self.isa.T

        self.c(f"DFT-11 butterfly [{d}] (AVX-512 no-spill, 32 ZMM)")

        # Sums and diffs — all live simultaneously (holds 10+10=20 ZMMs plus x0(2)+consts(10)=32)
        self.c("Sums and diffs (all live)")
        self.o(f"{T} T4r={self.add('x1_re','x10_re')}, T4i={self.add('x1_im','x10_im')};")
        self.o(f"{T} Tir={self.sub('x10_re','x1_re')}, Tii={self.sub('x10_im','x1_im')};")
        self.o(f"{T} Tgr={self.add('x5_re','x6_re')}, Tgi={self.add('x5_im','x6_im')};")
        self.o(f"{T} Tlr={self.sub('x6_re','x5_re')}, Tli={self.sub('x6_im','x5_im')};")
        self.o(f"{T} Tdr={self.add('x4_re','x7_re')}, Tdi={self.add('x4_im','x7_im')};")
        self.o(f"{T} Tkr={self.sub('x7_re','x4_re')}, Tki={self.sub('x7_im','x4_im')};")
        self.o(f"{T} Tar={self.add('x3_re','x8_re')}, Tai={self.add('x3_im','x8_im')};")
        self.o(f"{T} Tjr={self.sub('x8_re','x3_re')}, Tji={self.sub('x8_im','x3_im')};")
        self.o(f"{T} T7r={self.add('x2_re','x9_re')}, T7i={self.add('x2_im','x9_im')};")
        self.o(f"{T} Tmr={self.sub('x9_re','x2_re')}, Tmi={self.sub('x9_im','x2_im')};")

        # Pair (4,7)  [y0 computed AFTER all pairs that read x0]
        self.b()
        self.c("Pair (4,7)")
        self.o(f"{{ {T} Rr={self.fma('KP841253532','Tar',self.fma('KP415415013','Tgr',self.fnma('KP959492973','Tdr',self.fnma('KP142314838','T7r',self.fnma('KP654860733','T4r','x0_re')))))};")
        self.o(f"  {T} Ri={self.fma('KP841253532','Tai',self.fma('KP415415013','Tgi',self.fnma('KP959492973','Tdi',self.fnma('KP142314838','T7i',self.fnma('KP654860733','T4i','x0_im')))))};")
        self.o(f"  {T} Sr={self.neg(self.fma('KP755749574','Tii',self.fma('KP540640817','Tji',self.fnma('KP909631995','Tli',self.fnma('KP989821441','Tmi',self.mul('KP281732556','Tki'))))))};")
        self.o(f"  {T} Si={self.fma('KP755749574','Tir',self.fma('KP540640817','Tjr',self.fnma('KP909631995','Tlr',self.fnma('KP989821441','Tmr',self.mul('KP281732556','Tkr')))))};")
        y4, y7 = out_names[4], out_names[7]
        if fwd:
            self.o(f"  {y4}_re={self.add('Rr','Sr')}; {y4}_im={self.add('Ri','Si')};")
            self.o(f"  {y7}_re={self.sub('Rr','Sr')}; {y7}_im={self.sub('Ri','Si')}; }}")
        else:
            self.o(f"  {y7}_re={self.add('Rr','Sr')}; {y7}_im={self.add('Ri','Si')};")
            self.o(f"  {y4}_re={self.sub('Rr','Sr')}; {y4}_im={self.sub('Ri','Si')}; }}")

        # Pair (5,6)
        self.b()
        self.c("Pair (5,6)")
        self.o(f"{{ {T} Rr={self.fma('KP841253532','T7r',self.fma('KP415415013','Tdr',self.fnma('KP142314838','Tgr',self.fnma('KP654860733','Tar',self.fnma('KP959492973','T4r','x0_re')))))};")
        self.o(f"  {T} Ri={self.fma('KP841253532','T7i',self.fma('KP415415013','Tdi',self.fnma('KP142314838','Tgi',self.fnma('KP654860733','Tai',self.fnma('KP959492973','T4i','x0_im')))))};")
        self.o(f"  {T} Sr={self.neg(self.fma('KP281732556','Tii',self.fma('KP755749574','Tji',self.fnma('KP909631995','Tki',self.fnma('KP540640817','Tmi',self.mul('KP989821441','Tli'))))))};")
        self.o(f"  {T} Si={self.fma('KP281732556','Tir',self.fma('KP755749574','Tjr',self.fnma('KP909631995','Tkr',self.fnma('KP540640817','Tmr',self.mul('KP989821441','Tlr')))))};")
        y5, y6 = out_names[5], out_names[6]
        if fwd:
            self.o(f"  {y5}_re={self.add('Rr','Sr')}; {y5}_im={self.add('Ri','Si')};")
            self.o(f"  {y6}_re={self.sub('Rr','Sr')}; {y6}_im={self.sub('Ri','Si')}; }}")
        else:
            self.o(f"  {y6}_re={self.add('Rr','Sr')}; {y6}_im={self.add('Ri','Si')};")
            self.o(f"  {y5}_re={self.sub('Rr','Sr')}; {y5}_im={self.sub('Ri','Si')}; }}")

        # Pair (3,8)
        self.b()
        self.c("Pair (3,8)")
        self.o(f"{{ {T} Rr={self.fma('KP415415013','Tar',self.fma('KP841253532','Tdr',self.fnma('KP654860733','Tgr',self.fnma('KP959492973','T7r',self.fnma('KP142314838','T4r','x0_re')))))};")
        self.o(f"  {T} Ri={self.fma('KP415415013','Tai',self.fma('KP841253532','Tdi',self.fnma('KP654860733','Tgi',self.fnma('KP959492973','T7i',self.fnma('KP142314838','T4i','x0_im')))))};")
        self.o(f"  {T} Sr={self.neg(self.fma('KP989821441','Tii',self.fma('KP540640817','Tki',self.fnma('KP909631995','Tji',self.fnma('KP281732556','Tmi',self.mul('KP755749574','Tli'))))))};")
        self.o(f"  {T} Si={self.fma('KP989821441','Tir',self.fma('KP540640817','Tkr',self.fnma('KP909631995','Tjr',self.fnma('KP281732556','Tmr',self.mul('KP755749574','Tlr')))))};")
        y3, y8 = out_names[3], out_names[8]
        if fwd:
            self.o(f"  {y3}_re={self.add('Rr','Sr')}; {y3}_im={self.add('Ri','Si')};")
            self.o(f"  {y8}_re={self.sub('Rr','Sr')}; {y8}_im={self.sub('Ri','Si')}; }}")
        else:
            self.o(f"  {y8}_re={self.add('Rr','Sr')}; {y8}_im={self.add('Ri','Si')};")
            self.o(f"  {y3}_re={self.sub('Rr','Sr')}; {y3}_im={self.sub('Ri','Si')}; }}")

        # Pair (1,10)
        self.b()
        self.c("Pair (1,10)")
        self.o(f"{{ {T} Rr={self.fma('KP841253532','T4r',self.fma('KP415415013','T7r',self.fnma('KP959492973','Tgr',self.fnma('KP654860733','Tdr',self.fnma('KP142314838','Tar','x0_re')))))};")
        self.o(f"  {T} Ri={self.fma('KP841253532','T4i',self.fma('KP415415013','T7i',self.fnma('KP959492973','Tgi',self.fnma('KP654860733','Tdi',self.fnma('KP142314838','Tai','x0_im')))))};")
        self.o(f"  {T} Sr={self.neg(self.fma('KP540640817','Tii',self.fma('KP909631995','Tmi',self.fma('KP989821441','Tji',self.fma('KP755749574','Tki',self.mul('KP281732556','Tli'))))))};")
        self.o(f"  {T} Si={self.fma('KP540640817','Tir',self.fma('KP909631995','Tmr',self.fma('KP989821441','Tjr',self.fma('KP755749574','Tkr',self.mul('KP281732556','Tlr')))))};")
        y1, y10 = out_names[1], out_names[10]
        if fwd:
            self.o(f"  {y1}_re={self.add('Rr','Sr')}; {y1}_im={self.add('Ri','Si')};")
            self.o(f"  {y10}_re={self.sub('Rr','Sr')}; {y10}_im={self.sub('Ri','Si')}; }}")
        else:
            self.o(f"  {y10}_re={self.add('Rr','Sr')}; {y10}_im={self.add('Ri','Si')};")
            self.o(f"  {y1}_re={self.sub('Rr','Sr')}; {y1}_im={self.sub('Ri','Si')}; }}")

        # Pair (2,9)
        self.b()
        self.c("Pair (2,9)")
        self.o(f"{{ {T} Rr={self.fma('KP415415013','T4r',self.fma('KP841253532','Tgr',self.fnma('KP142314838','Tdr',self.fnma('KP959492973','Tar',self.fnma('KP654860733','T7r','x0_re')))))};")
        self.o(f"  {T} Ri={self.fma('KP415415013','T4i',self.fma('KP841253532','Tgi',self.fnma('KP142314838','Tdi',self.fnma('KP959492973','Tai',self.fnma('KP654860733','T7i','x0_im')))))};")
        self.o(f"  {T} Sr={self.neg(self.fma('KP909631995','Tii',self.fnma('KP540640817','Tli',self.fnma('KP989821441','Tki',self.fnma('KP281732556','Tji',self.mul('KP755749574','Tmi'))))))};")
        self.o(f"  {T} Si={self.fma('KP909631995','Tir',self.fnma('KP540640817','Tlr',self.fnma('KP989821441','Tkr',self.fnma('KP281732556','Tjr',self.mul('KP755749574','Tmr')))))};")
        y2, y9 = out_names[2], out_names[9]
        if fwd:
            self.o(f"  {y2}_re={self.add('Rr','Sr')}; {y2}_im={self.add('Ri','Si')};")
            self.o(f"  {y9}_re={self.sub('Rr','Sr')}; {y9}_im={self.sub('Ri','Si')}; }}")
        else:
            self.o(f"  {y9}_re={self.add('Rr','Sr')}; {y9}_im={self.add('Ri','Si')};")
            self.o(f"  {y2}_re={self.sub('Rr','Sr')}; {y2}_im={self.sub('Ri','Si')}; }}")

        # y0 = x0 + T4 + T7 + Ta + Td + Tg — LAST, after all pairs that read x0
        self.b()
        y0 = out_names[0]
        self.o(f"{y0}_re={self.add('x0_re',self.add('T4r',self.add('T7r',self.add('Tar',self.add('Tdr','Tgr')))))};")
        self.o(f"{y0}_im={self.add('x0_im',self.add('T4i',self.add('T7i',self.add('Tai',self.add('Tdi','Tgi')))))};")


# ================================================================
# HELPERS: constants
# ================================================================

def emit_dft11_constants(em):
    """Emit the 10 DFT-11 constants + sign_flip as SIMD broadcasts or scalars."""
    T = em.isa.T
    vals = [
        ('KP654860733', KP654860733_val),
        ('KP142314838', KP142314838_val),
        ('KP959492973', KP959492973_val),
        ('KP415415013', KP415415013_val),
        ('KP841253532', KP841253532_val),
        ('KP989821441', KP989821441_val),
        ('KP909631995', KP909631995_val),
        ('KP281732556', KP281732556_val),
        ('KP540640817', KP540640817_val),
        ('KP755749574', KP755749574_val),
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


def emit_dft11_constants_raw(lines, isa, indent=1):
    """Append DFT-11 constant declarations to a line list."""
    pad = "    " * indent
    T = isa.T
    vals = [
        ('KP654860733', KP654860733_val),
        ('KP142314838', KP142314838_val),
        ('KP959492973', KP959492973_val),
        ('KP415415013', KP415415013_val),
        ('KP841253532', KP841253532_val),
        ('KP989821441', KP989821441_val),
        ('KP909631995', KP909631995_val),
        ('KP281732556', KP281732556_val),
        ('KP540640817', KP540640817_val),
        ('KP755749574', KP755749574_val),
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


def emit_spill_decl(em):
    """Emit the aligned spill buffer declaration (10 slots for R=11)."""
    N_SPILL = 10
    sm = em.isa.sm
    T = em.isa.T
    if em.isa.name == 'scalar':
        em.o(f"double spill_re[{N_SPILL}], spill_im[{N_SPILL}];")
    else:
        em.o(f"{em.isa.align} {T} spill_re_buf[{N_SPILL}];")
        em.o(f"{em.isa.align} {T} spill_im_buf[{N_SPILL}];")
        em.o(f"double * __restrict__ spill_re = (double*)spill_re_buf;")
        em.o(f"double * __restrict__ spill_im = (double*)spill_im_buf;")


def emit_spill_decl_raw(lines, isa, indent=1):
    """Append spill buffer declarations to a line list."""
    pad = "    " * indent
    N_SPILL = 10
    T = isa.T
    if isa.name == 'scalar':
        lines.append(f"{pad}double spill_re[{N_SPILL}], spill_im[{N_SPILL}];")
    else:
        lines.append(f"{pad}{isa.align} {T} spill_re_buf[{N_SPILL}];")
        lines.append(f"{pad}{isa.align} {T} spill_im_buf[{N_SPILL}];")
        lines.append(f"{pad}double * __restrict__ spill_re = (double*)spill_re_buf;")
        lines.append(f"{pad}double * __restrict__ spill_im = (double*)spill_im_buf;")


# ================================================================
# KERNEL BODY EMITTERS
# ================================================================

def emit_kernel_body(em, d, variant):
    """Emit the inner loop body for notw, dit_tw, dif_tw."""
    T = em.isa.T

    # Load all 11 inputs
    for n in range(R):
        em.emit_load(f"x{n}", n)

    # DIT: twiddle inputs 1..10 before butterfly
    if variant == 'dit_tw':
        for n in range(1, R):
            em.emit_ext_tw(f"x{n}", n - 1, d)

    em.b()
    em.emit_radix11_butterfly(d)
    em.b()

    # DIF: twiddle outputs 1..10 after butterfly
    if variant == 'dif_tw':
        for m in range(1, R):
            em.emit_ext_tw(f"x{m}", m - 1, d)

    # Store all 11 outputs
    for m in range(R):
        em.emit_store(f"x{m}", m)


def emit_kernel_body_log3(em, d, variant):
    """Emit the log3 variant: derive w2..w10 from w1 (9 cmuls)."""
    T = em.isa.T
    is_dit = variant == 'dit_tw_log3'

    em.c("Load base twiddle w1, derive w2..w10 (9 cmuls)")
    if em.isa.name == 'scalar':
        em.o(f"const double w1r = tw_re[0*K+k], w1i = tw_im[0*K+k];")
    else:
        em.o(f"const {T} w1r = LD(&tw_re[0*K+k]), w1i = LD(&tw_im[0*K+k]);")
    em.n_load += 2

    # w2 = w1 * w1
    em.o(f"{T} w2r, w2i;")
    em.emit_cmul("w2r", "w2i", "w1r", "w1i", "w1r", "w1i", 'fwd')

    # w3 = w1 * w2
    em.o(f"{T} w3r, w3i;")
    em.emit_cmul("w3r", "w3i", "w1r", "w1i", "w2r", "w2i", 'fwd')

    # w4 = w1 * w3
    em.o(f"{T} w4r, w4i;")
    em.emit_cmul("w4r", "w4i", "w1r", "w1i", "w3r", "w3i", 'fwd')

    # w5 = w1 * w4
    em.o(f"{T} w5r, w5i;")
    em.emit_cmul("w5r", "w5i", "w1r", "w1i", "w4r", "w4i", 'fwd')

    # w6 = w3 * w3
    em.o(f"{T} w6r, w6i;")
    em.emit_cmul("w6r", "w6i", "w3r", "w3i", "w3r", "w3i", 'fwd')

    # w7 = w3 * w4
    em.o(f"{T} w7r, w7i;")
    em.emit_cmul("w7r", "w7i", "w3r", "w3i", "w4r", "w4i", 'fwd')

    # w8 = w4 * w4
    em.o(f"{T} w8r, w8i;")
    em.emit_cmul("w8r", "w8i", "w4r", "w4i", "w4r", "w4i", 'fwd')

    # w9 = w4 * w5
    em.o(f"{T} w9r, w9i;")
    em.emit_cmul("w9r", "w9i", "w4r", "w4i", "w5r", "w5i", 'fwd')

    # w10 = w5 * w5
    em.o(f"{T} w10r, w10i;")
    em.emit_cmul("w10r", "w10i", "w5r", "w5i", "w5r", "w5i", 'fwd')
    em.b()

    # Load inputs
    for n in range(R):
        em.emit_load(f"x{n}", n)

    # DIT: apply twiddles to x1..x10
    if is_dit:
        for n in range(1, R):
            em.emit_cmul_inplace(f"x{n}", f"w{n}r", f"w{n}i", d)
    em.b()

    em.emit_radix11_butterfly(d)
    em.b()

    # DIF: apply twiddles to outputs x1..x10
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
        func_base = 'radix11_n1_dit_kernel'
    elif variant == 'dit_tw':
        func_base = 'radix11_tw_flat_dit_kernel'
        tw_params = 'flat'
    elif variant == 'dif_tw':
        func_base = 'radix11_tw_flat_dif_kernel'
        tw_params = 'flat'
    elif variant == 'dit_tw_log3':
        func_base = 'radix11_tw_log3_dit_kernel'
        tw_params = 'flat'
    elif variant == 'dif_tw_log3':
        func_base = 'radix11_tw_log3_dif_kernel'
        tw_params = 'flat'

    vname = {
        'notw': 'N1 (no twiddle)',
        'dit_tw': 'DIT twiddled (flat)',
        'dif_tw': 'DIF twiddled (flat)',
        'dit_tw_log3': 'DIT twiddled (log3 derived)',
        'dif_tw_log3': 'DIF twiddled (log3 derived)',
    }[variant]
    guard = f"FFT_RADIX11_{isa.name.upper()}_{variant.upper()}_H"

    em.L.append(f"/**")
    em.L.append(f" * @file fft_radix11_{isa.name}_{variant}.h")
    em.L.append(f" * @brief DFT-11 {isa.name.upper()} {vname}")
    em.L.append(f" *")
    em.L.append(f" * Monolithic DFT-11 butterfly (genfft DAG), 10 constants.")
    em.L.append(f" * AVX2: 3-phase Sethi-Ullman, 10 spills, peak 16 YMM.")
    em.L.append(f" * AVX-512: single-pass no-spill, 32 ZMM exactly.")
    em.L.append(f" * k-step={isa.k_step}")
    em.L.append(f" * Generated by gen_radix11.py")
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
        emit_dft11_constants(em)

        # Working registers for all 11 inputs/outputs
        em.o(f"{T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
        em.o(f"{T} x4_re,x4_im,x5_re,x5_im,x6_re,x6_im,x7_re,x7_im;")
        em.o(f"{T} x8_re,x8_im,x9_re,x9_im,x10_re,x10_im;")

        # AVX2/scalar: spill buffer + butterfly working temps
        if isa.name != 'avx512':
            emit_spill_decl(em)
            em.b()
            em.o(f"{T} T4r,T4i,Tgr,Tgi,Tdr,Tdi,Tar,Tai,T7r,T7i;")
            em.o(f"{T} Tir,Tii,Tlr,Tli,Tkr,Tki,Tjr,Tji,Tmr,Tmi;")
            em.o(f"{T} Thr,Thi,Tur,Tui,Tsr,Tsi,Tqr,Tqi,Tor,Toi;")
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
    if isa.name == 'avx512' and not is_log3 and func_base:
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

            emit_dft11_constants_raw(em.L, isa, indent=1)
            xdecl = (f"        {T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im,"
                     f"x4_re,x4_im,x5_re,x5_im,x6_re,x6_im,x7_re,x7_im,"
                     f"x8_re,x8_im,x9_re,x9_im,x10_re,x10_im;")
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

    is_n1     = ct_variant == 'ct_n1'
    is_t1_dit = ct_variant == 'ct_t1_dit'
    is_t1_dif = ct_variant == 'ct_t1_dif'
    em.addr_mode = 'n1' if is_n1 else 't1'

    if is_n1:
        func_base = "radix11_n1"
        vname = "n1 (separate is/os)"
    elif is_t1_dif:
        func_base = "radix11_t1_dif"
        vname = "t1 DIF (in-place twiddle)"
    else:
        func_base = "radix11_t1_dit"
        vname = "t1 DIT (in-place twiddle)"

    guard = f"FFT_RADIX11_{isa.name.upper()}_CT_{ct_variant.upper()}_H"

    em.L.append(f"/**")
    em.L.append(f" * @file fft_radix11_{isa.name}_{ct_variant}.h")
    em.L.append(f" * @brief DFT-11 {isa.name.upper()} {vname}")
    em.L.append(f" *")
    em.L.append(f" * FFTW-style codelet for recursive CT executor.")
    em.L.append(f" * Generated by gen_radix11.py --variant {ct_variant}")
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

        emit_dft11_constants(em)

        em.o(f"{T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
        em.o(f"{T} x4_re,x4_im,x5_re,x5_im,x6_re,x6_im,x7_re,x7_im;")
        em.o(f"{T} x8_re,x8_im,x9_re,x9_im,x10_re,x10_im;")
        if isa.name != 'avx512':
            emit_spill_decl(em)
            em.o(f"{T} T4r,T4i,Tgr,Tgi,Tdr,Tdi,Tar,Tai,T7r,T7i;")
            em.o(f"{T} Tir,Tii,Tlr,Tli,Tkr,Tki,Tjr,Tji,Tmr,Tmi;")
            em.o(f"{T} Thr,Thi,Tur,Tui,Tsr,Tsi,Tqr,Tqi,Tor,Toi;")
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
        kernel_variant = 'notw' if is_n1 else ('dif_tw' if is_t1_dif else 'dit_tw')
        emit_kernel_body(em, d, kernel_variant)
        em.ind -= 1
        em.o("}")
        em.L.append("}")
        em.L.append("")

    # n1_ovs: butterfly with fused SIMD transpose stores (for R=11)
    # 11 bins = groups of 4 (bins 0-3, 4-7) + 3 remaining (8,9,10)
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
            em.L.append(f"radix11_n1_ovs_{d}_{isa.name}(")
            em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
            em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
            em.L.append(f"    size_t is, size_t os, size_t vl, size_t ovs)")
            em.L.append(f"{{")

            em.L.append(f"    /* n1_ovs: butterfly -> tbuf, then 4x4 transpose (bins 0-3, 4-7) + scatter (bins 8-10) */")
            em.L.append(f"    {isa.align} double tbuf_re[{R * VL}];")
            em.L.append(f"    {isa.align} double tbuf_im[{R * VL}];")

            emit_dft11_constants_raw(em.L, isa, indent=1)
            if isa.name != 'avx512':
                emit_spill_decl_raw(em.L, isa, indent=1)

            em.L.append(f"    {T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im,x3_re,x3_im;")
            em.L.append(f"    {T} x4_re,x4_im,x5_re,x5_im,x6_re,x6_im,x7_re,x7_im;")
            em.L.append(f"    {T} x8_re,x8_im,x9_re,x9_im,x10_re,x10_im;")
            if isa.name != 'avx512':
                em.L.append(f"    {T} T4r,T4i,Tgr,Tgi,Tdr,Tdi,Tar,Tai,T7r,T7i;")
                em.L.append(f"    {T} Tir,Tii,Tlr,Tli,Tkr,Tki,Tjr,Tji,Tmr,Tmi;")
                em.L.append(f"    {T} Thr,Thi,Tur,Tui,Tsr,Tsi,Tqr,Tqi,Tor,Toi;")
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

            # Bins 8,9,10: extract from tbuf via YMM load + scalar scatter
            for bn in [8, 9, 10]:
                em.L.append(f"        /* Bin {bn}: extract from tbuf -> scatter */")
                if isa.name == 'avx2':
                    for comp, arr in [('re', 'out_re'), ('im', 'out_im')]:
                        bname = f"tbuf_{comp}"
                        em.L.append(f"        {{ __m256d v=LD(&{bname}[{bn}*{VL}]);")
                        em.L.append(f"          __m128d lo=_mm256_castpd256_pd128(v), hi=_mm256_extractf128_pd(v,1);")
                        em.L.append(f"          _mm_storel_pd(&{arr}[(k+0)*ovs+os*{bn}], lo);")
                        em.L.append(f"          _mm_storeh_pd(&{arr}[(k+1)*ovs+os*{bn}], lo);")
                        em.L.append(f"          _mm_storel_pd(&{arr}[(k+2)*ovs+os*{bn}], hi);")
                        em.L.append(f"          _mm_storeh_pd(&{arr}[(k+3)*ovs+os*{bn}], hi); }}")
                else:  # avx512
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
        t2_pattern = 'radix11_n1_dit_kernel'
        sv_name = 'radix11_n1sv_kernel'
    elif variant == 'dit_tw':
        t2_pattern = 'radix11_tw_flat_dit_kernel'
        sv_name = 'radix11_t1sv_dit_kernel'
    elif variant == 'dif_tw':
        t2_pattern = 'radix11_tw_flat_dif_kernel'
        sv_name = 'radix11_t1sv_dif_kernel'
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
    parser = argparse.ArgumentParser(description='Unified R=11 codelet generator')
    parser.add_argument('--isa', default='avx2',
                        choices=['scalar', 'avx2', 'avx512', 'all'])
    parser.add_argument('--variant', default='notw',
                        choices=['notw', 'dit_tw', 'dif_tw', 'dit_tw_log3', 'dif_tw_log3',
                                 'ct_n1', 'ct_t1_dit', 'ct_t1_dif', 'all'])
    args = parser.parse_args()

    if args.isa == 'all':
        targets = [ISA_SCALAR, ISA_AVX2, ISA_AVX512]
    else:
        targets = [ALL_ISA[args.isa]]

    std_variants = ['notw', 'dit_tw', 'dif_tw', 'dit_tw_log3', 'dif_tw_log3']
    ct_variants  = ['ct_n1', 'ct_t1_dit', 'ct_t1_dif']

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
