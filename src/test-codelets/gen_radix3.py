#!/usr/bin/env python3
"""
gen_radix3.py -- Unified DFT-3 codelet generator for VectorFFT

Monolithic DFT-3 butterfly with 2 constants (KHALF, KS).
No spills on any ISA: 3 complex inputs = 6 regs + 2 constants = 8 total.
Peak 8 YMMs on AVX2, zero spills on ALL ISAs.

DFT-3 butterfly (forward):
  a = x1 + x2        (symmetric sum)
  d = x1 - x2        (antisymmetric diff)
  y0 = x0 + a
  R1 = fnmadd(KHALF, a, x0)   [= x0 - 0.5*a = x0 + cos(2pi/3)*a]
  S_re = KS * d_im   (cross real from imag: sin(2pi/3)*d_im)
  S_pos = KS * d_re  (sin(2pi/3)*d_re)
  fwd: y1_re = R1_re + S_re;  y1_im = R1_im - S_pos
       y2_re = R1_re - S_re;  y2_im = R1_im + S_pos
  bwd: swap y1/y2 (or equivalently negate S)

Constants:
  KHALF = 0.5
  KS = sin(2*pi/3) = sqrt(3)/2 = 0.866025403784438646763723170752936183471402627

Log3 twiddle derivation (R=3):
  Load w1 from table.  w2 = w1*w1  (1 cmul)

Usage:
  python3 gen_radix3.py --isa avx2 --variant all
  python3 gen_radix3.py --isa all --variant ct_n1
  python3 gen_radix3.py --isa avx2 --variant ct_t1_dit
"""

import sys, math, argparse, re

R = 3

# DFT-3 constants
KHALF_val = 0.500000000000000000000000000000000000000000000
KS_val    = 0.866025403784438646763723170752936183471402627

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

ISA_SCALAR = ISAConfig('scalar', 'double', 1, 1, '', 1, '', '', 'R3S')
ISA_AVX2   = ISAConfig('avx2', '__m256d', 4, 4, '_mm256', 4,
    '__attribute__((target("avx2,fma")))',
    '__attribute__((aligned(32)))', 'R3A')
ISA_AVX512 = ISAConfig('avx512', '__m512d', 8, 8, '_mm512', 8,
    '__attribute__((target("avx512f,avx512dq,fma")))',
    '__attribute__((aligned(64)))', 'R3L')

ALL_ISA = {'scalar': ISA_SCALAR, 'avx2': ISA_AVX2, 'avx512': ISA_AVX512}

# ================================================================
# EMITTER
# ================================================================

class Emitter:
    def __init__(self, isa):
        self.isa = isa
        self.L = []
        self.ind = 1
        self.n_add = self.n_sub = self.n_mul = self.n_neg = 0
        self.n_fma = self.n_fms = 0
        self.n_load = self.n_store = 0
        self.addr_mode = 'K'

    def reset(self):
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
            'total_mem': self.n_load + self.n_store,
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

    # -- DFT-3 butterfly --
    def emit_radix3_butterfly(self, d, out_names=None):
        """Emit DFT-3 butterfly on x0..x2.
        No ISA dispatch needed: 3 complex inputs = 6 regs + 2 constants = 8 total.
        Fits trivially on AVX2 (16 YMMs) and AVX-512 (32 ZMMs). Zero spills on all ISAs.
        out_names: if provided, list of 3 output variable names.
                   Default: overwrite x0..x2."""
        fwd = (d == 'fwd')
        T = self.isa.T

        if out_names is None:
            out_names = ['x0', 'x1', 'x2']

        self.c(f"DFT-3 butterfly [{d}]")

        # Symmetric/antisymmetric pair
        # a = x1 + x2  (symmetric sum)
        # d = x1 - x2  (antisymmetric diff)
        self.o(f"{T} ar={self.add('x1_re','x2_re')}, ai={self.add('x1_im','x2_im')};")
        self.o(f"{T} dr={self.sub('x1_re','x2_re')}, di={self.sub('x1_im','x2_im')};")

        # R1 = x0 - 0.5*a = fnmadd(KHALF, a, x0)  [compute BEFORE y0 to use x0 first]
        self.o(f"{T} R1r={self.fnma('KHALF','ar','x0_re')}, R1i={self.fnma('KHALF','ai','x0_im')};")

        # y0 = x0 + a  [now safe to overwrite x0]
        y0 = out_names[0]
        self.o(f"{y0}_re={self.add('x0_re','ar')}; {y0}_im={self.add('x0_im','ai')};")

        # Fuse S into combine: y1 = R1 + KS*d_cross, y2 = R1 - KS*d_cross
        # Eliminates 2 mul + 2 add/sub, replaced by 4 FMA.
        # S_re = KS*d_im (cross real/imag), S_im = -KS*d_re
        y1, y2 = out_names[1], out_names[2]
        if fwd:
            self.o(f"{y1}_re={self.fma('KS','di','R1r')}; {y1}_im={self.fnma('KS','dr','R1i')};")
            self.o(f"{y2}_re={self.fnma('KS','di','R1r')}; {y2}_im={self.fma('KS','dr','R1i')};")
        else:
            self.o(f"{y2}_re={self.fma('KS','di','R1r')}; {y2}_im={self.fnma('KS','dr','R1i')};")
            self.o(f"{y1}_re={self.fnma('KS','di','R1r')}; {y1}_im={self.fma('KS','dr','R1i')};")


# ================================================================
# HELPERS: constants
# ================================================================

def emit_dft3_constants(em):
    """Emit the 2 DFT-3 constants + sign_flip as SIMD broadcasts or scalars."""
    T = em.isa.T
    if em.isa.name == 'scalar':
        em.o(f"const double KHALF = {KHALF_val:.45f};")
        em.o(f"const double KS    = {KS_val:.45f};")
    else:
        set1 = f"{em.isa.p}_set1_pd"
        em.o(f"const {T} sign_flip = {set1}(-0.0);")
        em.o(f"const {T} KHALF = {set1}({KHALF_val:.45f});")
        em.o(f"const {T} KS    = {set1}({KS_val:.45f});")
        em.o(f"(void)sign_flip;  /* unused by current butterfly, reserved for neg() */")
    em.b()


def emit_dft3_constants_raw(lines, isa, indent=1):
    """Append DFT-3 constant declarations + sign_flip directly to a line list."""
    pad = "    " * indent
    T = isa.T
    if isa.name == 'scalar':
        lines.append(f"{pad}const double KHALF = {KHALF_val:.45f};")
        lines.append(f"{pad}const double KS    = {KS_val:.45f};")
    else:
        set1 = f"{isa.p}_set1_pd"
        lines.append(f"{pad}const {T} sign_flip = {set1}(-0.0);")
        lines.append(f"{pad}const {T} KHALF = {set1}({KHALF_val:.45f});")
        lines.append(f"{pad}const {T} KS    = {set1}({KS_val:.45f});")
        lines.append(f"{pad}(void)sign_flip;")
    lines.append(f"")


# ================================================================
# KERNEL BODY EMITTERS
# ================================================================

def emit_kernel_body(em, d, variant):
    """Emit the inner loop body for notw, dit_tw, dif_tw."""
    T = em.isa.T

    # Load inputs
    for n in range(R):
        em.emit_load(f"x{n}", n)

    # DIT: twiddle inputs 1..2 before butterfly
    if variant == 'dit_tw':
        for n in range(1, R):
            em.emit_ext_tw(f"x{n}", n - 1, d)

    em.b()
    em.emit_radix3_butterfly(d)
    em.b()

    # DIF: twiddle outputs 1..2 after butterfly
    if variant == 'dif_tw':
        for m in range(1, R):
            em.emit_ext_tw(f"x{m}", m - 1, d)

    # Store outputs
    for m in range(R):
        em.emit_store(f"x{m}", m)


def emit_kernel_body_log3(em, d, variant):
    """Emit the log3 variant: derive w2 from w1 (1 cmul for R=3)."""
    T = em.isa.T
    is_dit = variant == 'dit_tw_log3'

    # Load base twiddle w1
    em.c("Load base twiddle w1, derive w2 (1 cmul)")
    if em.isa.name == 'scalar':
        em.o(f"const double w1r = tw_re[0*K+k], w1i = tw_im[0*K+k];")
    else:
        em.o(f"const {T} w1r = LD(&tw_re[0*K+k]), w1i = LD(&tw_im[0*K+k]);")
    em.n_load += 2

    # w2 = w1 * w1
    em.o(f"{T} w2r, w2i;")
    em.emit_cmul("w2r", "w2i", "w1r", "w1i", "w1r", "w1i", 'fwd')
    em.b()

    # Load inputs
    for n in range(R):
        em.emit_load(f"x{n}", n)

    # DIT: apply twiddles to x1..x2
    if is_dit:
        for n in range(1, R):
            wname = f"w{n}"
            em.emit_cmul_inplace(f"x{n}", f"{wname}r", f"{wname}i", d)
    em.b()

    em.emit_radix3_butterfly(d)
    em.b()

    # DIF: apply twiddles to outputs x1..x2
    if not is_dit:
        for m in range(1, R):
            wname = f"w{m}"
            em.emit_cmul_inplace(f"x{m}", f"{wname}r", f"{wname}i", d)

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
        func_base = 'radix3_n1_dit_kernel'
    elif variant == 'dit_tw':
        func_base = 'radix3_tw_flat_dit_kernel'
        tw_params = 'flat'
    elif variant == 'dif_tw':
        func_base = 'radix3_tw_flat_dif_kernel'
        tw_params = 'flat'
    elif variant == 'dit_tw_log3':
        func_base = 'radix3_tw_log3_dit_kernel'
        tw_params = 'flat'  # log3 still reads from tw_re/tw_im (just 1 base row)
    elif variant == 'dif_tw_log3':
        func_base = 'radix3_tw_log3_dif_kernel'
        tw_params = 'flat'

    vname = {
        'notw': 'N1 (no twiddle)', 'dit_tw': 'DIT twiddled (flat)',
        'dif_tw': 'DIF twiddled (flat)',
        'dit_tw_log3': 'DIT twiddled (log3 derived)',
        'dif_tw_log3': 'DIF twiddled (log3 derived)',
    }[variant]
    guard = f"FFT_RADIX3_{isa.name.upper()}_{variant.upper()}_H"

    em.L.append(f"/**")
    em.L.append(f" * @file fft_radix3_{isa.name}_{variant}.h")
    em.L.append(f" * @brief DFT-3 {isa.name.upper()} {vname}")
    em.L.append(f" *")
    em.L.append(f" * Monolithic DFT-3 butterfly, 2 constants, zero spills on all ISAs.")
    em.L.append(f" * k-step={isa.k_step}")
    em.L.append(f" * Generated by gen_radix3.py")
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
        emit_dft3_constants(em)

        # Working registers
        em.o(f"{T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im;")
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
    # R=3: 2x6 working + 2 shared constants = 14/32 ZMM — very comfortable.
    if isa.name == 'avx512' and not is_log3 and func_base is not None:
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
            em.L.append(f"    /* U=2: two independent k-groups per iteration (14/32 ZMM) */")

            emit_dft3_constants_raw(em.L, isa, indent=1)
            xdecl = f"        {T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im;"
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

    is_n1 = ct_variant == 'ct_n1'
    is_t1_dit = ct_variant == 'ct_t1_dit'
    is_t1_dif = ct_variant == 'ct_t1_dif'
    em.addr_mode = 'n1' if is_n1 else 't1'

    if is_n1:
        func_base = "radix3_n1"
        vname = "n1 (separate is/os)"
    elif is_t1_dif:
        func_base = "radix3_t1_dif"
        vname = "t1 DIF (in-place twiddle)"
    else:
        func_base = "radix3_t1_dit"
        vname = "t1 DIT (in-place twiddle)"

    guard = f"FFT_RADIX3_{isa.name.upper()}_CT_{ct_variant.upper()}_H"

    em.L.append(f"/**")
    em.L.append(f" * @file fft_radix3_{isa.name}_{ct_variant}.h")
    em.L.append(f" * @brief DFT-3 {isa.name.upper()} {vname}")
    em.L.append(f" *")
    em.L.append(f" * FFTW-style codelet for recursive CT executor.")
    em.L.append(f" * Generated by gen_radix3.py --variant {ct_variant}")
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

        emit_dft3_constants(em)

        em.o(f"{T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im;")
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

    # n1_ovs: butterfly with scatter stores
    # R=3: NOT divisible by 4, so NO 4x4 transpose possible.
    # ALL 3 bins use scalar scatter via AVX2 extract (same as R=5 bin 4, R=7 bins 4-6).
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
            em.L.append(f"radix3_n1_ovs_{d}_{isa.name}(")
            em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
            em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
            em.L.append(f"    size_t is, size_t os, size_t vl, size_t ovs)")
            em.L.append(f"{{")

            em.L.append(f"    /* n1_ovs: butterfly -> tbuf, then scatter all 3 bins via AVX2 extract */")
            em.L.append(f"    /* R=3 is not divisible by 4, so no 4x4 transpose is possible. */")
            em.L.append(f"    {isa.align} double tbuf_re[{R * VL}];")
            em.L.append(f"    {isa.align} double tbuf_im[{R * VL}];")

            emit_dft3_constants_raw(em.L, isa, indent=1)

            em.L.append(f"    {T} x0_re,x0_im,x1_re,x1_im,x2_re,x2_im;")
            em.L.append(f"")

            em.L.append(f"    for (size_t k = 0; k < vl; k += {isa.k_step}) {{")

            # Use fresh Emitter for kernel body
            em2.L = []
            em2.ind = 2
            em2.reset()
            em2.addr_mode = 'n1_ovs'
            emit_kernel_body(em2, d, 'notw')
            em.L.extend(em2.L)

            # Scatter all 3 bins via AVX2 extract from tbuf
            # R=3 < 4: no 4x4 transpose possible. Each bin: load YMM from tbuf,
            # castpd256_pd128 + extractf128, storel/storeh for 4 lanes.
            if isa.name == 'avx2':
                for bn in range(R):
                    em.L.append(f"        /* Bin {bn}: extract from tbuf -> scatter */")
                    for comp, arr in [('re', 'out_re'), ('im', 'out_im')]:
                        bname = f"tbuf_{comp}"
                        em.L.append(f"        {{ __m256d v=LD(&{bname}[{bn}*{VL}]);")
                        em.L.append(f"          __m128d lo=_mm256_castpd256_pd128(v), hi=_mm256_extractf128_pd(v,1);")
                        em.L.append(f"          _mm_storel_pd(&{arr}[(k+0)*ovs+os*{bn}], lo);")
                        em.L.append(f"          _mm_storeh_pd(&{arr}[(k+1)*ovs+os*{bn}], lo);")
                        em.L.append(f"          _mm_storel_pd(&{arr}[(k+2)*ovs+os*{bn}], hi);")
                        em.L.append(f"          _mm_storeh_pd(&{arr}[(k+3)*ovs+os*{bn}], hi); }}")
            else:
                # AVX-512 fallback: scalar scatter from tbuf
                for bn in range(R):
                    em.L.append(f"        /* Bin {bn}: scalar scatter */")
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
        t2_pattern = 'radix3_n1_dit_kernel'
        sv_name = 'radix3_n1sv_kernel'
    elif variant == 'dit_tw':
        t2_pattern = 'radix3_tw_flat_dit_kernel'
        sv_name = 'radix3_t1sv_dit_kernel'
    elif variant == 'dif_tw':
        t2_pattern = 'radix3_tw_flat_dif_kernel'
        sv_name = 'radix3_t1sv_dif_kernel'
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
    s5 = '-' * 5; s3 = '-' * 3; s4 = '-' * 4; sep = '-' * 20
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
    parser = argparse.ArgumentParser(description='Unified R=3 codelet generator')
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
    ct_variants = ['ct_n1', 'ct_t1_dit', 'ct_t1_dif']

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
