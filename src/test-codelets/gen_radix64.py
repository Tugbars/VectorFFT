#!/usr/bin/env python3
"""
gen_radix64.py -- Unified DFT-64 codelet generator (8x8 CT)

8 radix-8 sub-FFTs + 8 radix-8 column combines.
Internal twiddles: W64 constants (trivial + cmul from static arrays).
External twiddles: flat table with log3 derivation.

Variants: notw (N1), dit_tw, dif_tw
ISAs:     scalar, avx2, avx512

Usage:
  python3 gen_radix64.py --isa avx2 --variant all
  python3 gen_radix64.py --isa avx2 --variant notw
  python3 gen_radix64.py --isa scalar --variant notw
  python3 gen_radix64.py avx2  # legacy positional
"""
import math, sys, re

# =======================================
# ISA abstraction
# =======================================

class ISA:
    def __init__(self, name):
        self.name = name
        if name == 'avx512':
            self.T    = '__m512d'
            self.Ti   = '__m512i'
            self.C    = 8
            self.P    = '_mm512'
            self.attr = '__attribute__((target("avx512f,avx512dq,fma")))'
            self.align = 64
            self.sign_mask = f'const __m512d sign_mask = _mm512_set1_pd(-0.0);'
        elif name == 'avx2':
            self.T    = '__m256d'
            self.Ti   = '__m256i'
            self.C    = 4
            self.P    = '_mm256'
            self.attr = '__attribute__((target("avx2,fma")))'
            self.align = 32
            self.sign_mask = f'const __m256d sign_mask = _mm256_set1_pd(-0.0);'
        elif name == 'scalar':
            self.T    = 'double'
            self.Ti   = None
            self.C    = 1
            self.P    = None
            self.attr = ''
            self.align = 8
            self.sign_mask = ''
        else:
            raise ValueError(f"Unknown ISA: {name}")

    @property
    def is_scalar(self):
        return self.name == 'scalar'

    def add(self, a, b):
        return f'({a})+({b})' if self.is_scalar else f'{self.P}_add_pd({a},{b})'
    def sub(self, a, b):
        return f'({a})-({b})' if self.is_scalar else f'{self.P}_sub_pd({a},{b})'
    def mul(self, a, b):
        return f'({a})*({b})' if self.is_scalar else f'{self.P}_mul_pd({a},{b})'
    def fmsub(self, a,b,c):
        return f'({a})*({b})-({c})' if self.is_scalar else f'{self.P}_fmsub_pd({a},{b},{c})'
    def fmadd(self, a,b,c):
        return f'({a})*({b})+({c})' if self.is_scalar else f'{self.P}_fmadd_pd({a},{b},{c})'
    def fnmadd(self,a,b,c):
        return f'({c})-({a})*({b})' if self.is_scalar else f'{self.P}_fnmadd_pd({a},{b},{c})'
    def load(self, addr):
        return f'(*({addr}))' if self.is_scalar else f'{self.P}_load_pd({addr})'
    def store(self, addr, v):
        return f'*({addr}) = {v}' if self.is_scalar else f'{self.P}_store_pd({addr},{v})'
    def set1(self, v):
        return str(v) if self.is_scalar else f'{self.P}_set1_pd({v})'
    def neg(self, v):
        return f'-({v})' if self.is_scalar else f'{self.P}_xor_pd({v}, sign_mask)'


# =======================================
# Twiddle math
# =======================================

def w64(e):
    e = e % 64
    a = 2.0 * math.pi * e / 64
    return (math.cos(a), -math.sin(a))

def collect_internal_twiddles():
    exps = set()
    for k1 in range(1, 8):
        for n2 in range(1, 8):
            exps.add((n2 * k1) % 64)
    return sorted(exps)

ITW_EXPS = collect_internal_twiddles()

def classify_itw(e):
    e = e % 64
    if e == 0: return 'one'
    if e == 16: return 'neg_j'
    if e == 32: return 'neg_one'
    if e == 48: return 'pos_j'
    if e == 8 or e == 24: return 'w8'
    return 'general'


# =======================================
# Emitter
# =======================================

class Emitter:
    def __init__(self): self.L=[]; self.ind=0
    def o(self, s=''): self.L.append('    '*self.ind + s)
    def c(self, s): self.o(f'/* {s} */')
    def b(self): self.L.append('')


# =======================================
# Static W64 arrays (file-scope)
# =======================================

def emit_itw_static_arrays():
    L = []
    L.append('#ifndef FFT_W64_STATIC_ARRAYS_DEFINED')
    L.append('#define FFT_W64_STATIC_ARRAYS_DEFINED')
    L.append('/* Internal W64 twiddle constants -- static arrays for broadcast */')
    L.append('static const double __attribute__((aligned(8))) iw_re[64] = {')
    for i in range(0, 64, 4):
        vals = [f'{w64(j)[0]:.20e}' for j in range(i, i+4)]
        L.append(f'    {", ".join(vals)},')
    L.append('};')
    L.append('static const double __attribute__((aligned(8))) iw_im[64] = {')
    for i in range(0, 64, 4):
        vals = [f'{w64(j)[1]:.20e}' for j in range(i, i+4)]
        L.append(f'    {", ".join(vals)},')
    L.append('};')
    L.append('#endif /* FFT_W64_STATIC_ARRAYS_DEFINED */')
    L.append('')
    return L


# =======================================
# ISA-dependent helpers
# =======================================

def cmul_split(isa, em, dst_r, dst_i, ar, ai, br, bi):
    em.o(f'const {isa.T} {dst_r} = {isa.fmsub(ar,br,isa.mul(ai,bi))};')
    em.o(f'const {isa.T} {dst_i} = {isa.fmadd(ar,bi,isa.mul(ai,br))};')

def cmul_inplace_split(isa, em, xr, xi, wr, wi, fwd):
    em.o(f'{{ {isa.T} tr = {xr};')
    if fwd:
        em.o(f'  {xr} = {isa.fmsub(xr,wr,isa.mul(xi,wi))};')
        em.o(f'  {xi} = {isa.fmadd("tr",wi,isa.mul(xi,wr))}; }}')
    else:
        em.o(f'  {xr} = {isa.fmadd(xr,wr,isa.mul(xi,wi))};')
        em.o(f'  {xi} = {isa.fnmadd("tr",wi,isa.mul(xi,wr))}; }}')


def emit_r8_butterfly(isa, em, fwd, xr, xi, yr, yi):
    """Monolithic radix-8 butterfly. For AVX-512 (32 regs)."""
    T = isa.T
    add, sub, mul = isa.add, isa.sub, isa.mul

    em.o('{')
    em.ind += 1

    em.o(f'const {T} epr={add(xr[0],xr[4])}, epi={add(xi[0],xi[4])};')
    em.o(f'const {T} eqr={sub(xr[0],xr[4])}, eqi={sub(xi[0],xi[4])};')
    em.o(f'const {T} err={add(xr[2],xr[6])}, eri={add(xi[2],xi[6])};')
    em.o(f'const {T} esr={sub(xr[2],xr[6])}, esi={sub(xi[2],xi[6])};')
    em.o(f'const {T} A0r={add("epr","err")}, A0i={add("epi","eri")};')
    em.o(f'const {T} A2r={sub("epr","err")}, A2i={sub("epi","eri")};')
    if fwd:
        em.o(f'const {T} A1r={add("eqr","esi")}, A1i={sub("eqi","esr")};')
        em.o(f'const {T} A3r={sub("eqr","esi")}, A3i={add("eqi","esr")};')
    else:
        em.o(f'const {T} A1r={sub("eqr","esi")}, A1i={add("eqi","esr")};')
        em.o(f'const {T} A3r={add("eqr","esi")}, A3i={sub("eqi","esr")};')

    em.o(f'const {T} opr={add(xr[1],xr[5])}, opi={add(xi[1],xi[5])};')
    em.o(f'const {T} oqr={sub(xr[1],xr[5])}, oqi={sub(xi[1],xi[5])};')
    em.o(f'const {T} orr={add(xr[3],xr[7])}, ori={add(xi[3],xi[7])};')
    em.o(f'const {T} osr={sub(xr[3],xr[7])}, osi={sub(xi[3],xi[7])};')
    em.o(f'const {T} B0r={add("opr","orr")}, B0i={add("opi","ori")};')
    em.o(f'const {T} B2r={sub("opr","orr")}, B2i={sub("opi","ori")};')
    if fwd:
        em.o(f'const {T} B1r={add("oqr","osi")}, B1i={sub("oqi","osr")};')
        em.o(f'const {T} B3r={sub("oqr","osi")}, B3i={add("oqi","osr")};')
    else:
        em.o(f'const {T} B1r={sub("oqr","osi")}, B1i={add("oqi","osr")};')
        em.o(f'const {T} B3r={add("oqr","osi")}, B3i={sub("oqi","osr")};')

    em.o(f'{yr[0]}={add("A0r","B0r")}; {yi[0]}={add("A0i","B0i")};')
    em.o(f'{yr[4]}={sub("A0r","B0r")}; {yi[4]}={sub("A0i","B0i")};')

    if fwd:
        em.o(f'{{ const {T} t1r={mul("vc",add("B1r","B1i"))}, t1i={mul("vc",sub("B1i","B1r"))};')
    else:
        em.o(f'{{ const {T} t1r={mul("vc",sub("B1r","B1i"))}, t1i={mul("vc",add("B1r","B1i"))};')
    em.o(f'  {yr[1]}={add("A1r","t1r")}; {yi[1]}={add("A1i","t1i")};')
    em.o(f'  {yr[5]}={sub("A1r","t1r")}; {yi[5]}={sub("A1i","t1i")}; }}')

    if fwd:
        em.o(f'{yr[2]}={add("A2r","B2i")}; {yi[2]}={sub("A2i","B2r")};')
        em.o(f'{yr[6]}={sub("A2r","B2i")}; {yi[6]}={add("A2i","B2r")};')
    else:
        em.o(f'{yr[2]}={sub("A2r","B2i")}; {yi[2]}={add("A2i","B2r")};')
        em.o(f'{yr[6]}={add("A2r","B2i")}; {yi[6]}={sub("A2i","B2r")};')

    if fwd:
        em.o(f'{{ const {T} t3r={mul("vnc",sub("B3r","B3i"))}, t3i={mul("vnc",add("B3r","B3i"))};')
    else:
        em.o(f'{{ const {T} t3r={mul("vnc",add("B3r","B3i"))}, t3i={mul("vc",sub("B3r","B3i"))};')
    em.o(f'  {yr[3]}={add("A3r","t3r")}; {yi[3]}={add("A3i","t3i")};')
    em.o(f'  {yr[7]}={sub("A3r","t3r")}; {yi[7]}={sub("A3i","t3i")}; }}')

    em.ind -= 1
    em.o('}')


def emit_r8_butterfly_split(isa, em, fwd, xr, xi, yr, yi):
    """Split radix-8 butterfly for AVX2 (16 YMM).

    Splits the butterfly into three phases to keep peak live <= 16 regs:
      Phase 1: Even DFT-4 (x0,x2,x4,x6) -> A0..A3, spill to bfr/bfi  [peak 12]
      Phase 2: Odd  DFT-4 (x1,x3,x5,x7) -> B0..B3, apply W8 twiddles [peak 12]
      Phase 3: Reload A one pair at a time, combine A +/- B -> output     [peak 14]

    Cost: 8 stores + 8 loads to bfr/bfi per butterfly (L1, hidden by OoO).
    Saves: ~20 compiler-inserted spill/reloads from monolithic register pressure.
    """
    T = isa.T
    C = isa.C
    add, sub, mul = isa.add, isa.sub, isa.mul
    st = isa.store
    ld = isa.load

    em.c('Split radix-8 butterfly (even/odd/combine)')
    em.o('{')
    em.ind += 1

    # -- Phase 1: Even DFT-4 -> spill A0..A3 --
    # Inputs: xr/xi[0,2,4,6] = 8 regs. Temps: 8. Peak: 12 (inputs die as consumed).
    em.c('Phase 1: Even DFT-4 -> spill A0..A3')
    em.o(f'{{ const {T} epr={add(xr[0],xr[4])}, epi={add(xi[0],xi[4])};')
    em.o(f'  const {T} eqr={sub(xr[0],xr[4])}, eqi={sub(xi[0],xi[4])};')
    em.o(f'  const {T} err={add(xr[2],xr[6])}, eri={add(xi[2],xi[6])};')
    em.o(f'  const {T} esr={sub(xr[2],xr[6])}, esi={sub(xi[2],xi[6])};')
    # A0, A2: straight add/sub
    em.o(f'  {st(f"&bfr[0*{C}]", add("epr","err"))}; {st(f"&bfi[0*{C}]", add("epi","eri"))};')
    em.o(f'  {st(f"&bfr[2*{C}]", sub("epr","err"))}; {st(f"&bfi[2*{C}]", sub("epi","eri"))};')
    # A1, A3: xj variant
    if fwd:
        em.o(f'  {st(f"&bfr[1*{C}]", add("eqr","esi"))}; {st(f"&bfi[1*{C}]", sub("eqi","esr"))};')
        em.o(f'  {st(f"&bfr[3*{C}]", sub("eqr","esi"))}; {st(f"&bfi[3*{C}]", add("eqi","esr"))};')
    else:
        em.o(f'  {st(f"&bfr[1*{C}]", sub("eqr","esi"))}; {st(f"&bfi[1*{C}]", add("eqi","esr"))};')
        em.o(f'  {st(f"&bfr[3*{C}]", add("eqr","esi"))}; {st(f"&bfi[3*{C}]", sub("eqi","esr"))};')
    em.o(f'}}')
    # All even inputs and temps dead. Registers free.

    # -- Phase 2: Odd DFT-4 + W8 twiddles --
    # Inputs: xr/xi[1,3,5,7] = 8 regs. vc,vnc = 2 regs. Peak: 12.
    em.c('Phase 2: Odd DFT-4 + W8 twiddles')
    em.o(f'{{ const {T} opr={add(xr[1],xr[5])}, opi={add(xi[1],xi[5])};')
    em.o(f'  const {T} oqr={sub(xr[1],xr[5])}, oqi={sub(xi[1],xi[5])};')
    em.o(f'  const {T} orr={add(xr[3],xr[7])}, ori={add(xi[3],xi[7])};')
    em.o(f'  const {T} osr={sub(xr[3],xr[7])}, osi={sub(xi[3],xi[7])};')
    # B0 (no W8 twiddle -- kept for combine)
    em.o(f'  const {T} B0r={add("opr","orr")}, B0i={add("opi","ori")};')
    # B2 (W8^2 = xj, handled implicitly in combine)
    em.o(f'  const {T} B2r={sub("opr","orr")}, B2i={sub("opi","ori")};')
    # B1 raw, then W8^1 twiddle
    if fwd:
        em.o(f'  const {T} _B1r={add("oqr","osi")}, _B1i={sub("oqi","osr")};')
        em.o(f'  const {T} B1r={mul("vc",add("_B1r","_B1i"))}, B1i={mul("vc",sub("_B1i","_B1r"))};')
    else:
        em.o(f'  const {T} _B1r={sub("oqr","osi")}, _B1i={add("oqi","osr")};')
        em.o(f'  const {T} B1r={mul("vc",sub("_B1r","_B1i"))}, B1i={mul("vc",add("_B1r","_B1i"))};')
    # B3 raw, then W8^3 twiddle
    if fwd:
        em.o(f'  const {T} _B3r={sub("oqr","osi")}, _B3i={add("oqi","osr")};')
        em.o(f'  const {T} B3r={mul("vnc",sub("_B3r","_B3i"))}, B3i={mul("vnc",add("_B3r","_B3i"))};')
    else:
        em.o(f'  const {T} _B3r={add("oqr","osi")}, _B3i={sub("oqi","osr")};')
        em.o(f'  const {T} B3r={mul("vnc",add("_B3r","_B3i"))}, B3i={mul("vc",sub("_B3r","_B3i"))};')

    # -- Phase 3: Reload A, combine A +/- B incrementally --
    # B0,B1,B2,B3 in 8 regs. Load 2 A regs at a time. Peak: 10.
    # After each pair: B[k] dies, output written. Regs decrease.
    em.c('Phase 3: Reload A, combine A +/- B (incremental)')
    # Pair 0: A0 +/- B0
    em.o(f'{{ const {T} A0r={ld(f"&bfr[0*{C}]")}, A0i={ld(f"&bfi[0*{C}]")};')
    em.o(f'  {yr[0]}={add("A0r","B0r")}; {yi[0]}={add("A0i","B0i")};')
    em.o(f'  {yr[4]}={sub("A0r","B0r")}; {yi[4]}={sub("A0i","B0i")}; }}')
    # Pair 1: A1 +/- B1 (W8 already applied)
    em.o(f'{{ const {T} A1r={ld(f"&bfr[1*{C}]")}, A1i={ld(f"&bfi[1*{C}]")};')
    em.o(f'  {yr[1]}={add("A1r","B1r")}; {yi[1]}={add("A1i","B1i")};')
    em.o(f'  {yr[5]}={sub("A1r","B1r")}; {yi[5]}={sub("A1i","B1i")}; }}')
    # Pair 2: A2 +/- j*B2 (implicit xj: swap re/im)
    em.o(f'{{ const {T} A2r={ld(f"&bfr[2*{C}]")}, A2i={ld(f"&bfi[2*{C}]")};')
    if fwd:
        em.o(f'  {yr[2]}={add("A2r","B2i")}; {yi[2]}={sub("A2i","B2r")};')
        em.o(f'  {yr[6]}={sub("A2r","B2i")}; {yi[6]}={add("A2i","B2r")}; }}')
    else:
        em.o(f'  {yr[2]}={sub("A2r","B2i")}; {yi[2]}={add("A2i","B2r")};')
        em.o(f'  {yr[6]}={add("A2r","B2i")}; {yi[6]}={sub("A2i","B2r")}; }}')
    # Pair 3: A3 +/- B3 (W8 already applied)
    em.o(f'{{ const {T} A3r={ld(f"&bfr[3*{C}]")}, A3i={ld(f"&bfi[3*{C}]")};')
    em.o(f'  {yr[3]}={add("A3r","B3r")}; {yi[3]}={add("A3i","B3i")};')
    em.o(f'  {yr[7]}={sub("A3r","B3r")}; {yi[7]}={sub("A3i","B3i")}; }}')

    em.o(f'}}')   # close odd DFT-4 scope (B vars die)
    em.ind -= 1
    em.o('}')


def emit_r8(isa, em, fwd, xr, xi, yr, yi):
    """ISA dispatch: split butterfly for AVX2 (16 regs), monolithic for AVX-512/scalar (32 regs or no pressure)."""
    if isa.name == 'avx2':
        emit_r8_butterfly_split(isa, em, fwd, xr, xi, yr, yi)
    else:
        emit_r8_butterfly(isa, em, fwd, xr, xi, yr, yi)


def emit_itw_apply(isa, em, xr, xi, e, fwd):
    e = e % 64
    cls = classify_itw(e)
    T, P = isa.T, isa.P

    if cls == 'one': return

    if cls == 'neg_j':
        if fwd: em.o(f'{{ {T} tr = {xr}; {xr} = {xi}; {xi} = {isa.neg("tr")}; }}')
        else:   em.o(f'{{ {T} tr = {xr}; {xr} = {isa.neg(xi)}; {xi} = tr; }}')
        return

    if cls == 'neg_one':
        em.o(f'{xr} = {isa.neg(xr)};')
        em.o(f'{xi} = {isa.neg(xi)};')
        return

    if cls == 'pos_j':
        if fwd: em.o(f'{{ {T} tr = {xr}; {xr} = {isa.neg(xi)}; {xi} = tr; }}')
        else:   em.o(f'{{ {T} tr = {xr}; {xr} = {xi}; {xi} = {isa.neg("tr")}; }}')
        return

    if cls == 'w8':
        if e == 8:
            if fwd:
                em.o(f'{{ {T} tr = {isa.mul("vc",isa.add(xr,xi))};')
                em.o(f'  {xi} = {isa.mul("vc",isa.sub(xi,xr))}; {xr} = tr; }}')
            else:
                em.o(f'{{ {T} tr = {isa.mul("vc",isa.sub(xr,xi))};')
                em.o(f'  {xi} = {isa.mul("vc",isa.add(xr,xi))}; {xr} = tr; }}')
        elif e == 24:
            if fwd:
                em.o(f'{{ {T} tr = {isa.mul("vnc",isa.sub(xr,xi))};')
                em.o(f'  {xi} = {isa.mul("vnc",isa.add(xr,xi))}; {xr} = tr; }}')
            else:
                em.o(f'{{ {T} tr = {isa.mul("vnc",isa.add(xr,xi))};')
                em.o(f'  {xi} = {isa.mul("vnc",isa.sub(xi,xr))}; {xr} = tr; }}')
        return

    # General: broadcast from static array
    wr = isa.set1(f'iw_re[{e}]')
    wi = isa.set1(f'iw_im[{e}]')
    em.o(f'{{ {T} tr = {xr};')
    if fwd:
        em.o(f'  {xr} = {isa.fmsub(xr,wr,isa.mul(xi,wi))};')
        em.o(f'  {xi} = {isa.fmadd("tr",wi,isa.mul(xi,wr))}; }}')
    else:
        em.o(f'  {xr} = {isa.fmadd(xr,wr,isa.mul(xi,wi))};')
        em.o(f'  {xi} = {isa.fnmadd("tr",wi,isa.mul(xi,wr))}; }}')


# =======================================
# External twiddle strategies
# =======================================

# --- Strategy 1: Full log3 (AVX-512, 32 ZMM) ---
# Derive all 14 bases upfront. 28 ZMM alive, fits in 32.

ROW_BASES_FULL = {0: None, 8: ('ew8r','ew8i'), 16: ('ew16r','ew16i'),
                  24: ('ew24r','ew24i'), 32: ('ew32r','ew32i'),
                  40: ('ew40r','ew40i'), 48: ('ew48r','ew48i'), 56: ('ew56r','ew56i')}
COL_BASES_FULL = {0: None, 1: ('ew1r','ew1i'), 2: ('ew2r','ew2i'),
                  3: ('ew3r','ew3i'), 4: ('ew4r','ew4i'), 5: ('ew5r','ew5i'),
                  6: ('ew6r','ew6i'), 7: ('ew7r','ew7i')}

def emit_log3_full(isa, em):
    """Derive all 14 bases at loop top. For AVX-512."""
    em.c('Log3 (full): load W^1, W^8, derive all 14 bases (32 ZMM)')
    em.o(f'const {isa.T} ew1r = {isa.load("&tw_re[0*K+k]")}, ew1i = {isa.load("&tw_im[0*K+k]")};')
    em.o(f'const {isa.T} ew8r = {isa.load("&tw_re[7*K+k]")}, ew8i = {isa.load("&tw_im[7*K+k]")};')
    em.b()
    em.c('Column bases: W^2..W^7')
    cmul_split(isa, em, 'ew2r','ew2i', 'ew1r','ew1i', 'ew1r','ew1i')
    cmul_split(isa, em, 'ew3r','ew3i', 'ew1r','ew1i', 'ew2r','ew2i')
    cmul_split(isa, em, 'ew4r','ew4i', 'ew2r','ew2i', 'ew2r','ew2i')
    cmul_split(isa, em, 'ew5r','ew5i', 'ew1r','ew1i', 'ew4r','ew4i')
    cmul_split(isa, em, 'ew6r','ew6i', 'ew3r','ew3i', 'ew3r','ew3i')
    cmul_split(isa, em, 'ew7r','ew7i', 'ew3r','ew3i', 'ew4r','ew4i')
    em.b()
    em.c('Row bases: W^16..W^56')
    cmul_split(isa, em, 'ew16r','ew16i', 'ew8r','ew8i', 'ew8r','ew8i')
    cmul_split(isa, em, 'ew24r','ew24i', 'ew8r','ew8i', 'ew16r','ew16i')
    cmul_split(isa, em, 'ew32r','ew32i', 'ew16r','ew16i', 'ew16r','ew16i')
    cmul_split(isa, em, 'ew40r','ew40i', 'ew8r','ew8i', 'ew32r','ew32i')
    cmul_split(isa, em, 'ew48r','ew48i', 'ew24r','ew24i', 'ew24r','ew24i')
    cmul_split(isa, em, 'ew56r','ew56i', 'ew24r','ew24i', 'ew32r','ew32i')
    em.b()

def emit_ext_twiddle_full(isa, em, xr, xi, n, fwd):
    """Apply W^n from full pre-derived bases. For AVX-512."""
    col = n % 8
    row8 = n - col
    if n == 0: return
    elif row8 == 0:
        cr, ci = COL_BASES_FULL[col]
        cmul_inplace_split(isa, em, xr, xi, cr, ci, fwd)
    elif col == 0:
        rr, ri = ROW_BASES_FULL[row8]
        cmul_inplace_split(isa, em, xr, xi, rr, ri, fwd)
    else:
        cr, ci = COL_BASES_FULL[col]
        rr, ri = ROW_BASES_FULL[row8]
        cmul_inplace_split(isa, em, xr, xi, cr, ci, fwd)
        cmul_inplace_split(isa, em, xr, xi, rr, ri, fwd)


# --- Strategy 2: Lean log3 (AVX2, 16 YMM) ---
# Derive 8 core intermediates upfront (16 YMM).
# Extended bases (ew5,6,7 + ew40,48,56) derived inline at point of use -- scoped,
# so compiler sees them as 2-instruction-lived temps, not loop-wide constants.
# Cost: ~48 extra cmuls per k-iteration. Saves: 12 YMM of liveness.

# Core bases: directly addressable
COL_BASES_CORE = {0: None, 1: ('ew1r','ew1i'), 2: ('ew2r','ew2i'),
                  3: ('ew3r','ew3i'), 4: ('ew4r','ew4i')}
ROW_BASES_CORE = {0: None, 8: ('ew8r','ew8i'), 16: ('ew16r','ew16i'),
                  24: ('ew24r','ew24i'), 32: ('ew32r','ew32i')}

# Extended bases: inline-derive from core pairs (a x b)
COL_EXTENDED = {5: ('ew1r','ew1i','ew4r','ew4i'),   # ew5 = ew1 x ew4
                6: ('ew3r','ew3i','ew3r','ew3i'),   # ew6 = ew3^2
                7: ('ew3r','ew3i','ew4r','ew4i')}   # ew7 = ew3 x ew4
ROW_EXTENDED = {40: ('ew8r','ew8i','ew32r','ew32i'),   # ew40 = ew8 x ew32
                48: ('ew16r','ew16i','ew32r','ew32i'),  # ew48 = ew16 x ew32
                56: ('ew24r','ew24i','ew32r','ew32i')}  # ew56 = ew24 x ew32

def emit_log3_lean(isa, em):
    """Derive 8 core bases only. For AVX2 (16 YMM)."""
    em.c('Log3 (lean): load W^1, W^8, derive 8 core intermediates (16 YMM)')
    em.o(f'const {isa.T} ew1r = {isa.load("&tw_re[0*K+k]")}, ew1i = {isa.load("&tw_im[0*K+k]")};')
    em.o(f'const {isa.T} ew8r = {isa.load("&tw_re[7*K+k]")}, ew8i = {isa.load("&tw_im[7*K+k]")};')
    em.b()
    em.c('Core column bases: W^2..W^4 (3 cmuls)')
    cmul_split(isa, em, 'ew2r','ew2i', 'ew1r','ew1i', 'ew1r','ew1i')
    cmul_split(isa, em, 'ew3r','ew3i', 'ew1r','ew1i', 'ew2r','ew2i')
    cmul_split(isa, em, 'ew4r','ew4i', 'ew2r','ew2i', 'ew2r','ew2i')
    em.b()
    em.c('Core row bases: W^16, W^24, W^32 (3 cmuls)')
    cmul_split(isa, em, 'ew16r','ew16i', 'ew8r','ew8i', 'ew8r','ew8i')
    cmul_split(isa, em, 'ew24r','ew24i', 'ew8r','ew8i', 'ew16r','ew16i')
    cmul_split(isa, em, 'ew32r','ew32i', 'ew16r','ew16i', 'ew16r','ew16i')
    em.c('Extended bases (ew5,6,7,40,48,56) derived inline at point of use')
    em.b()

def _emit_inline_derive_cmul(isa, em, xr, xi, ar, ai, br, bi, fwd):
    """Inline: derive tmp = a x b, then x *= tmp (or x *= conj(tmp)). All scoped."""
    T = isa.T
    em.o(f'{{ {T} _wr = {isa.fmsub(ar,br,isa.mul(ai,bi))}, '
         f'_wi = {isa.fmadd(ar,bi,isa.mul(ai,br))}; {T} _tr = {xr};')
    if fwd:
        em.o(f'  {xr} = {isa.fmsub(xr,"_wr",isa.mul(xi,"_wi"))};')
        em.o(f'  {xi} = {isa.fmadd("_tr","_wi",isa.mul(xi,"_wr"))}; }}')
    else:
        em.o(f'  {xr} = {isa.fmadd(xr,"_wr",isa.mul(xi,"_wi"))};')
        em.o(f'  {xi} = {isa.fnmadd("_tr","_wi",isa.mul(xi,"_wr"))}; }}')

def _emit_apply_base(isa, em, xr, xi, base_or_ext, core_dict, ext_dict, fwd):
    """Apply a single twiddle component (col or row). Core -> direct, extended -> inline derive."""
    if base_or_ext in core_dict and core_dict[base_or_ext] is not None:
        br, bi = core_dict[base_or_ext]
        cmul_inplace_split(isa, em, xr, xi, br, bi, fwd)
    elif base_or_ext in ext_dict:
        ar, ai, br, bi = ext_dict[base_or_ext]
        _emit_inline_derive_cmul(isa, em, xr, xi, ar, ai, br, bi, fwd)
    # else: base_or_ext == 0, no twiddle needed

def emit_ext_twiddle_lean(isa, em, xr, xi, n, fwd):
    """Apply W^n using lean log3. Core bases direct, extended bases inline-derived."""
    col = n % 8
    row8 = n - col
    if n == 0: return
    elif row8 == 0:
        _emit_apply_base(isa, em, xr, xi, col, COL_BASES_CORE, COL_EXTENDED, fwd)
    elif col == 0:
        _emit_apply_base(isa, em, xr, xi, row8, ROW_BASES_CORE, ROW_EXTENDED, fwd)
    else:
        _emit_apply_base(isa, em, xr, xi, col, COL_BASES_CORE, COL_EXTENDED, fwd)
        _emit_apply_base(isa, em, xr, xi, row8, ROW_BASES_CORE, ROW_EXTENDED, fwd)


# --- Strategy 3: Direct table load (kept for reference/benchmarking) ---

def emit_ext_twiddle_direct(isa, em, xr, xi, n, fwd):
    """Apply W^n by loading directly from twiddle table."""
    if n == 0:
        return
    T = isa.T
    wr = f'{isa.load(f"&tw_re[{n-1}*K+k]")}'
    wi = f'{isa.load(f"&tw_im[{n-1}*K+k]")}'
    em.o(f'{{ {T} _wr = {wr}, _wi = {wi}; {T} tr = {xr};')
    if fwd:
        em.o(f'  {xr} = {isa.fmsub(xr, "_wr", isa.mul(xi, "_wi"))};')
        em.o(f'  {xi} = {isa.fmadd("tr", "_wi", isa.mul(xi, "_wr"))}; }}')
    else:
        em.o(f'  {xr} = {isa.fmadd(xr, "_wr", isa.mul(xi, "_wi"))};')
        em.o(f'  {xi} = {isa.fnmadd("tr", "_wi", isa.mul(xi, "_wr"))}; }}')


# --- ISA dispatch ---

def emit_log3_derivation(isa, em):
    """Full log3 for both ISAs. Split butterfly handles AVX2 register pressure."""
    emit_log3_full(isa, em)

def emit_ext_twiddle(isa, em, xr, xi, n, fwd):
    """Full pre-derived bases for both ISAs."""
    emit_ext_twiddle_full(isa, em, xr, xi, n, fwd)


# =======================================
# DIT generator
# =======================================

def gen_dit_tw(isa, direction):
    fwd = direction == 'fwd'
    em = Emitter()
    T, C = isa.T, isa.C

    em.L.append(isa.attr)
    em.L.append(f'static void')
    em.L.append(f'radix64_tw_flat_dit_kernel_{direction}_{isa.name}(')
    em.L.append(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
    em.L.append(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    em.L.append(f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,')
    em.L.append(f'    size_t K)')
    em.L.append(f'{{')
    em.ind = 1

    em.o(isa.sign_mask)
    em.o(f'const {T} vc = {isa.set1("0.707106781186547524400844362104849039284835938")};')
    em.o(f'const {T} vnc = {isa.set1("-0.707106781186547524400844362104849039284835938")};')
    em.o(f'__attribute__((aligned({isa.align}))) double sp_re[64*{C}], sp_im[64*{C}];')
    if isa.name == 'avx2':
        em.o(f'__attribute__((aligned({isa.align}))) double bfr[4*{C}], bfi[4*{C}];')
    em.o(f'{T} x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i;')
    em.b()

    em.o(f'for (size_t k = 0; k < K; k += {C}) {{')
    em.ind += 1

    # Both ISAs use log3 derivation (2 loads + 12 cmuls, bandwidth-optimal).
    # AVX2: split butterfly keeps peak <=12 YMM, so log3 bases are dormant
    #   during butterfly phases -- compiler spills/reloads them from stack
    #   (sequential, predictable) instead of 63 strided table loads.
    # AVX-512: 32 ZMM holds everything with monolithic butterfly.
    emit_log3_derivation(isa, em)

    xr = [f'x{i}r' for i in range(8)]
    xi = [f'x{i}i' for i in range(8)]

    for n2 in range(8):
        em.c(f'Sub-FFT n2={n2}')
        for n1 in range(8):
            n = 8*n1 + n2
            em.o(f'{xr[n1]} = {isa.load(f"&in_re[{n}*K+k]")}; {xi[n1]} = {isa.load(f"&in_im[{n}*K+k]")};')
            emit_ext_twiddle(isa, em, xr[n1], xi[n1], n, fwd)
        em.b()
        emit_r8(isa, em, fwd, xr, xi, xr, xi)
        em.b()
        for k1 in range(8):
            slot = n2 * 8 + k1
            em.o(f'{isa.store(f"&sp_re[{slot}*{C}]", xr[k1])}; {isa.store(f"&sp_im[{slot}*{C}]", xi[k1])};')
        em.b()

    for k1 in range(8):
        em.c(f'Column k1={k1}')
        for n2 in range(8):
            slot = n2 * 8 + k1
            em.o(f'{xr[n2]} = {isa.load(f"&sp_re[{slot}*{C}]")}; {xi[n2]} = {isa.load(f"&sp_im[{slot}*{C}]")};')
        if k1 > 0:
            for n2 in range(1, 8):
                emit_itw_apply(isa, em, xr[n2], xi[n2], (n2*k1)%64, fwd)
        em.b()
        emit_r8(isa, em, fwd, xr, xi, xr, xi)
        em.b()
        for k2 in range(8):
            m = k1 + 8 * k2
            em.o(f'{isa.store(f"&out_re[{m}*K+k]", xr[k2])}; {isa.store(f"&out_im[{m}*K+k]", xi[k2])};')
        em.b()

    em.ind -= 1
    em.o('}')
    em.L.append('}')
    em.L.append('')
    return em.L


# =======================================
# DIF generator
# =======================================

def gen_dif_tw(isa, direction):
    fwd = direction == 'fwd'
    em = Emitter()
    T, C = isa.T, isa.C

    em.L.append(isa.attr)
    em.L.append(f'static void')
    em.L.append(f'radix64_tw_flat_dif_kernel_{direction}_{isa.name}(')
    em.L.append(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
    em.L.append(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    em.L.append(f'    const double * __restrict__ tw_re, const double * __restrict__ tw_im,')
    em.L.append(f'    size_t K)')
    em.L.append(f'{{')
    em.ind = 1

    em.o(isa.sign_mask)
    em.o(f'const {T} vc = {isa.set1("0.707106781186547524400844362104849039284835938")};')
    em.o(f'const {T} vnc = {isa.set1("-0.707106781186547524400844362104849039284835938")};')
    em.o(f'__attribute__((aligned({isa.align}))) double sp_re[64*{C}], sp_im[64*{C}];')
    em.o(f'{T} x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i;')
    em.b()
    if isa.name == 'avx2':
        em.o(f'__attribute__((aligned({isa.align}))) double bfr[4*{C}], bfi[4*{C}];')

    em.o(f'for (size_t k = 0; k < K; k += {C}) {{')
    em.ind += 1

    emit_log3_derivation(isa, em)

    xr = [f'x{i}r' for i in range(8)]
    xi = [f'x{i}i' for i in range(8)]

    for n2 in range(8):
        em.c(f'Sub-FFT n2={n2} (DIF: no external twiddle on inputs)')
        for n1 in range(8):
            n = 8*n1 + n2
            em.o(f'{xr[n1]} = {isa.load(f"&in_re[{n}*K+k]")}; {xi[n1]} = {isa.load(f"&in_im[{n}*K+k]")};')
        em.b()
        emit_r8(isa, em, fwd, xr, xi, xr, xi)
        em.b()
        for k1 in range(8):
            slot = n2 * 8 + k1
            em.o(f'{isa.store(f"&sp_re[{slot}*{C}]", xr[k1])}; {isa.store(f"&sp_im[{slot}*{C}]", xi[k1])};')
        em.b()

    for k1 in range(8):
        em.c(f'Column k1={k1}')
        for n2 in range(8):
            slot = n2 * 8 + k1
            em.o(f'{xr[n2]} = {isa.load(f"&sp_re[{slot}*{C}]")}; {xi[n2]} = {isa.load(f"&sp_im[{slot}*{C}]")};')
        if k1 > 0:
            for n2 in range(1, 8):
                emit_itw_apply(isa, em, xr[n2], xi[n2], (n2*k1)%64, fwd)
        em.b()
        emit_r8(isa, em, fwd, xr, xi, xr, xi)
        em.b()
        for k2 in range(8):
            m = k1 + 8 * k2
            emit_ext_twiddle(isa, em, xr[k2], xi[k2], m, fwd)
            em.o(f'{isa.store(f"&out_re[{m}*K+k]", xr[k2])}; {isa.store(f"&out_im[{m}*K+k]", xi[k2])};')
        em.b()

    em.ind -= 1
    em.o('}')
    em.L.append('}')
    em.L.append('')
    return em.L


# =======================================
# Notw (N1) generator
# =======================================

def gen_notw(isa, direction):
    fwd = direction == 'fwd'
    em = Emitter()
    T, C = isa.T, isa.C

    if isa.attr:
        em.L.append(isa.attr)
    em.L.append(f'static void')
    em.L.append(f'radix64_n1_dit_kernel_{direction}_{isa.name}(')
    em.L.append(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
    em.L.append(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    em.L.append(f'    size_t K)')
    em.L.append(f'{{')
    em.ind = 1

    if isa.sign_mask:
        em.o(isa.sign_mask)
    em.o(f'const {T} vc = {isa.set1("0.707106781186547524400844362104849039284835938")};')
    em.o(f'const {T} vnc = {isa.set1("-0.707106781186547524400844362104849039284835938")};')
    if isa.is_scalar:
        em.o(f'double sp_re[64], sp_im[64];')
    else:
        em.o(f'__attribute__((aligned({isa.align}))) double sp_re[64*{C}], sp_im[64*{C}];')
    if isa.name == 'avx2':
        em.o(f'__attribute__((aligned({isa.align}))) double bfr[4*{C}], bfi[4*{C}];')
    em.o(f'{T} x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i;')
    em.b()

    k_step = C if not isa.is_scalar else 1
    em.o(f'for (size_t k = 0; k < K; k += {k_step}) {{')
    em.ind += 1

    xr = [f'x{i}r' for i in range(8)]
    xi = [f'x{i}i' for i in range(8)]
    sp_mul = C if not isa.is_scalar else 1

    # PASS 1: 8 radix-8 sub-FFTs (no external twiddle)
    for n2 in range(8):
        em.c(f'Sub-FFT n2={n2}')
        for n1 in range(8):
            n = 8*n1 + n2
            em.o(f'{xr[n1]} = {isa.load(f"&in_re[{n}*K+k]")}; {xi[n1]} = {isa.load(f"&in_im[{n}*K+k]")};')
        em.b()
        emit_r8(isa, em, fwd, xr, xi, xr, xi)
        em.b()
        for k1 in range(8):
            slot = n2 * 8 + k1
            em.o(f'{isa.store(f"&sp_re[{slot}*{sp_mul}]", xr[k1])}; {isa.store(f"&sp_im[{slot}*{sp_mul}]", xi[k1])};')
        em.b()

    # PASS 2: 8 radix-8 column combines with internal W64 twiddles
    for k1 in range(8):
        em.c(f'Column k1={k1}')
        for n2 in range(8):
            slot = n2 * 8 + k1
            em.o(f'{xr[n2]} = {isa.load(f"&sp_re[{slot}*{sp_mul}]")}; {xi[n2]} = {isa.load(f"&sp_im[{slot}*{sp_mul}]")};')
        if k1 > 0:
            for n2 in range(1, 8):
                emit_itw_apply(isa, em, xr[n2], xi[n2], (n2*k1)%64, fwd)
        em.b()
        emit_r8(isa, em, fwd, xr, xi, xr, xi)
        em.b()
        for k2 in range(8):
            m = k1 + 8 * k2
            em.o(f'{isa.store(f"&out_re[{m}*K+k]", xr[k2])}; {isa.store(f"&out_im[{m}*K+k]", xi[k2])};')
        em.b()

    em.ind -= 1
    em.o('}')
    em.L.append('}')
    em.L.append('')
    return em.L


# =======================================
# File generator
# =======================================

def gen_ct_n1(isa, direction):
    """FFTW-style n1: out-of-place, separate is/os, loop over vl."""
    fwd = direction == 'fwd'
    em = Emitter()
    T, C = isa.T, isa.C

    if isa.attr:
        em.L.append(isa.attr)
    em.L.append(f'static void')
    em.L.append(f'radix64_n1_{direction}_{isa.name}(')
    em.L.append(f'    const double * __restrict__ in_re, const double * __restrict__ in_im,')
    em.L.append(f'    double * __restrict__ out_re, double * __restrict__ out_im,')
    if isa.is_scalar:
        em.L.append(f'    size_t is, size_t os, size_t vl, size_t ivs, size_t ovs)')
    else:
        em.L.append(f'    size_t is, size_t os, size_t vl)')
    em.L.append(f'{{')
    em.ind = 1

    if isa.sign_mask:
        em.o(isa.sign_mask)
    em.o(f'const {T} vc = {isa.set1("0.707106781186547524400844362104849039284835938")};')
    em.o(f'const {T} vnc = {isa.set1("-0.707106781186547524400844362104849039284835938")};')
    if isa.is_scalar:
        em.o(f'double sp_re[64], sp_im[64];')
    else:
        em.o(f'__attribute__((aligned({isa.align}))) double sp_re[64*{C}], sp_im[64*{C}];')
    if isa.name == 'avx2':
        em.o(f'__attribute__((aligned({isa.align}))) double bfr[4*{C}], bfi[4*{C}];')
    em.o(f'{T} x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i;')
    em.b()

    k_step = C if not isa.is_scalar else 1
    em.o(f'for (size_t k = 0; k < vl; k += {k_step}) {{')
    em.ind += 1

    xr = [f'x{i}r' for i in range(8)]
    xi = [f'x{i}i' for i in range(8)]
    sp_mul = C if not isa.is_scalar else 1

    # PASS 1: 8 radix-8 sub-FFTs — loads use is
    for n2 in range(8):
        em.c(f'Sub-FFT n2={n2}')
        for n1 in range(8):
            n = 8*n1 + n2
            em.o(f'{xr[n1]} = {isa.load(f"&in_re[{n}*is+k]")}; {xi[n1]} = {isa.load(f"&in_im[{n}*is+k]")};')
        em.b()
        emit_r8(isa, em, fwd, xr, xi, xr, xi)
        em.b()
        for k1 in range(8):
            slot = n2 * 8 + k1
            em.o(f'{isa.store(f"&sp_re[{slot}*{sp_mul}]", xr[k1])}; {isa.store(f"&sp_im[{slot}*{sp_mul}]", xi[k1])};')
        em.b()

    # PASS 2: 8 radix-8 column combines — stores use os
    for k1 in range(8):
        em.c(f'Column k1={k1}')
        for n2 in range(8):
            slot = n2 * 8 + k1
            em.o(f'{xr[n2]} = {isa.load(f"&sp_re[{slot}*{sp_mul}]")}; {xi[n2]} = {isa.load(f"&sp_im[{slot}*{sp_mul}]")};')
        if k1 > 0:
            for n2 in range(1, 8):
                emit_itw_apply(isa, em, xr[n2], xi[n2], (n2*k1)%64, fwd)
        em.b()
        emit_r8(isa, em, fwd, xr, xi, xr, xi)
        em.b()
        for k2 in range(8):
            m = k1 + 8 * k2
            em.o(f'{isa.store(f"&out_re[{m}*os+k]", xr[k2])}; {isa.store(f"&out_im[{m}*os+k]", xi[k2])};')
        em.b()

    em.ind -= 1
    em.o('}')
    em.L.append('}')
    em.L.append('')
    return em.L


def gen_ct_t1_dit(isa, direction, log3=False):
    """FFTW-style t1 DIT: in-place twiddle + butterfly, loop over m.
    log3=True: derive 63 twiddles from 2 base loads (better at high me).
    log3=False: direct 63 loads (better at small me, L1-resident)."""
    fwd = direction == 'fwd'
    em = Emitter()
    T, C = isa.T, isa.C
    suffix = '_log3' if log3 else ''

    if isa.attr:
        em.L.append(isa.attr)
    em.L.append(f'static void')
    em.L.append(f'radix64_t1_dit{suffix}_{direction}_{isa.name}(')
    em.L.append(f'    double * __restrict__ rio_re, double * __restrict__ rio_im,')
    em.L.append(f'    const double * __restrict__ W_re, const double * __restrict__ W_im,')
    if isa.is_scalar:
        em.L.append(f'    size_t ios, size_t mb, size_t me, size_t ms)')
    else:
        em.L.append(f'    size_t ios, size_t me)')
    em.L.append(f'{{')
    em.ind = 1

    if isa.sign_mask:
        em.o(isa.sign_mask)
    em.o(f'const {T} vc = {isa.set1("0.707106781186547524400844362104849039284835938")};')
    em.o(f'const {T} vnc = {isa.set1("-0.707106781186547524400844362104849039284835938")};')
    if isa.is_scalar:
        em.o(f'double sp_re[64], sp_im[64];')
    else:
        em.o(f'__attribute__((aligned({isa.align}))) double sp_re[64*{C}], sp_im[64*{C}];')
    if isa.name == 'avx2':
        em.o(f'__attribute__((aligned({isa.align}))) double bfr[4*{C}], bfi[4*{C}];')
    em.o(f'{T} x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i;')
    em.b()

    k_step = C if not isa.is_scalar else 1
    if isa.is_scalar:
        em.o(f'for (size_t m = mb; m < me; m++) {{')
    else:
        em.o(f'for (size_t m = 0; m < me; m += {k_step}) {{')
    em.ind += 1

    if log3:
        # Log3: load W^1 and W^8, derive W^2..W^56 (2 loads + 12 cmuls)
        em.c('Log3: derive 63 external twiddles from W^1 and W^8')
        em.o(f'const {T} ew1r = {isa.load("&W_re[0*me+m]")}, ew1i = {isa.load("&W_im[0*me+m]")};')
        em.o(f'const {T} ew8r = {isa.load("&W_re[7*me+m]")}, ew8i = {isa.load("&W_im[7*me+m]")};')
        em.b()
        em.c('Column bases: W^2..W^7')
        cmul_split(isa, em, 'ew2r','ew2i', 'ew1r','ew1i', 'ew1r','ew1i')
        cmul_split(isa, em, 'ew3r','ew3i', 'ew1r','ew1i', 'ew2r','ew2i')
        cmul_split(isa, em, 'ew4r','ew4i', 'ew2r','ew2i', 'ew2r','ew2i')
        cmul_split(isa, em, 'ew5r','ew5i', 'ew1r','ew1i', 'ew4r','ew4i')
        cmul_split(isa, em, 'ew6r','ew6i', 'ew3r','ew3i', 'ew3r','ew3i')
        cmul_split(isa, em, 'ew7r','ew7i', 'ew3r','ew3i', 'ew4r','ew4i')
        em.b()
        em.c('Row bases: W^16..W^56')
        cmul_split(isa, em, 'ew16r','ew16i', 'ew8r','ew8i', 'ew8r','ew8i')
        cmul_split(isa, em, 'ew24r','ew24i', 'ew8r','ew8i', 'ew16r','ew16i')
        cmul_split(isa, em, 'ew32r','ew32i', 'ew16r','ew16i', 'ew16r','ew16i')
        cmul_split(isa, em, 'ew40r','ew40i', 'ew8r','ew8i', 'ew32r','ew32i')
        cmul_split(isa, em, 'ew48r','ew48i', 'ew24r','ew24i', 'ew24r','ew24i')
        cmul_split(isa, em, 'ew56r','ew56i', 'ew24r','ew24i', 'ew32r','ew32i')
        em.b()

    xr = [f'x{i}r' for i in range(8)]
    xi = [f'x{i}i' for i in range(8)]
    sp_mul = C if not isa.is_scalar else 1

    # PASS 1: 8 radix-8 sub-FFTs — load from rio, apply external twiddle
    for n2 in range(8):
        em.c(f'Sub-FFT n2={n2}')
        for n1 in range(8):
            n = 8*n1 + n2
            if isa.is_scalar:
                em.o(f'{xr[n1]} = rio_re[m*ms+{n}*ios]; {xi[n1]} = rio_im[m*ms+{n}*ios];')
            else:
                em.o(f'{xr[n1]} = {isa.load(f"&rio_re[m+{n}*ios]")}; {xi[n1]} = {isa.load(f"&rio_im[m+{n}*ios]")};')
            # Apply external twiddle W^n (DIT: pre-multiply)
            if log3:
                emit_ext_twiddle_full(isa, em, xr[n1], xi[n1], n, fwd)
            elif n > 0:
                if isa.is_scalar:
                    em.o(f'{{ double wr = W_re[{n-1}*me+m], wi = W_im[{n-1}*me+m], tr = {xr[n1]};')
                else:
                    em.o(f'{{ {T} wr = {isa.load(f"&W_re[{n-1}*me+m]")}; {T} wi = {isa.load(f"&W_im[{n-1}*me+m]")};')
                    em.o(f'  {T} tr = {xr[n1]};')
                if fwd:
                    if isa.is_scalar:
                        em.o(f'  {xr[n1]} = {xr[n1]}*wr - {xi[n1]}*wi;')
                        em.o(f'  {xi[n1]} = tr*wi + {xi[n1]}*wr; }}')
                    else:
                        em.o(f'  {xr[n1]} = {isa.fmsub(xr[n1],"wr",isa.mul(xi[n1],"wi"))};')
                        em.o(f'  {xi[n1]} = {isa.fmadd("tr","wi",isa.mul(xi[n1],"wr"))}; }}')
                else:
                    if isa.is_scalar:
                        em.o(f'  {xr[n1]} = {xr[n1]}*wr + {xi[n1]}*wi;')
                        em.o(f'  {xi[n1]} = {xi[n1]}*wr - tr*wi; }}')
                    else:
                        em.o(f'  {xr[n1]} = {isa.fmadd(xr[n1],"wr",isa.mul(xi[n1],"wi"))};')
                        em.o(f'  {xi[n1]} = {isa.fmsub(xi[n1],"wr",isa.mul("tr","wi"))}; }}')
        em.b()
        emit_r8(isa, em, fwd, xr, xi, xr, xi)
        em.b()
        for k1 in range(8):
            slot = n2 * 8 + k1
            em.o(f'{isa.store(f"&sp_re[{slot}*{sp_mul}]", xr[k1])}; {isa.store(f"&sp_im[{slot}*{sp_mul}]", xi[k1])};')
        em.b()

    # PASS 2: 8 radix-8 column combines — write back in-place
    for k1 in range(8):
        em.c(f'Column k1={k1}')
        for n2 in range(8):
            slot = n2 * 8 + k1
            em.o(f'{xr[n2]} = {isa.load(f"&sp_re[{slot}*{sp_mul}]")}; {xi[n2]} = {isa.load(f"&sp_im[{slot}*{sp_mul}]")};')
        if k1 > 0:
            for n2 in range(1, 8):
                emit_itw_apply(isa, em, xr[n2], xi[n2], (n2*k1)%64, fwd)
        em.b()
        emit_r8(isa, em, fwd, xr, xi, xr, xi)
        em.b()
        for k2 in range(8):
            m_out = k1 + 8 * k2
            if isa.is_scalar:
                em.o(f'rio_re[m*ms+{m_out}*ios] = {xr[k2]}; rio_im[m*ms+{m_out}*ios] = {xi[k2]};')
            else:
                em.o(f'{isa.store(f"&rio_re[m+{m_out}*ios]", xr[k2])}; {isa.store(f"&rio_im[m+{m_out}*ios]", xi[k2])};')
        em.b()

    em.ind -= 1
    em.o('}')
    em.L.append('}')
    em.L.append('')
    return em.L


def gen_ct_t1_dif(isa, direction):
    """FFTW-style t1 DIF: in-place butterfly then post-twiddle, loop over m."""
    fwd = direction == 'fwd'
    em = Emitter()
    T, C = isa.T, isa.C

    if isa.attr:
        em.L.append(isa.attr)
    em.L.append(f'static void')
    em.L.append(f'radix64_t1_dif_{direction}_{isa.name}(')
    em.L.append(f'    double * __restrict__ rio_re, double * __restrict__ rio_im,')
    em.L.append(f'    const double * __restrict__ W_re, const double * __restrict__ W_im,')
    if isa.is_scalar:
        em.L.append(f'    size_t ios, size_t mb, size_t me, size_t ms)')
    else:
        em.L.append(f'    size_t ios, size_t me)')
    em.L.append(f'{{')
    em.ind = 1

    if isa.sign_mask:
        em.o(isa.sign_mask)
    em.o(f'const {T} vc = {isa.set1("0.707106781186547524400844362104849039284835938")};')
    em.o(f'const {T} vnc = {isa.set1("-0.707106781186547524400844362104849039284835938")};')
    if isa.is_scalar:
        em.o(f'double sp_re[64], sp_im[64];')
    else:
        em.o(f'__attribute__((aligned({isa.align}))) double sp_re[64*{C}], sp_im[64*{C}];')
    if isa.name == 'avx2':
        em.o(f'__attribute__((aligned({isa.align}))) double bfr[4*{C}], bfi[4*{C}];')
    em.o(f'{T} x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i;')
    em.b()

    k_step = C if not isa.is_scalar else 1
    if isa.is_scalar:
        em.o(f'for (size_t m = mb; m < me; m++) {{')
    else:
        em.o(f'for (size_t m = 0; m < me; m += {k_step}) {{')
    em.ind += 1

    xr = [f'x{i}r' for i in range(8)]
    xi = [f'x{i}i' for i in range(8)]
    sp_mul = C if not isa.is_scalar else 1

    # PASS 1: 8 radix-8 sub-FFTs — no external twiddle on input (DIF)
    for n2 in range(8):
        em.c(f'Sub-FFT n2={n2}')
        for n1 in range(8):
            n = 8*n1 + n2
            if isa.is_scalar:
                em.o(f'{xr[n1]} = rio_re[m*ms+{n}*ios]; {xi[n1]} = rio_im[m*ms+{n}*ios];')
            else:
                em.o(f'{xr[n1]} = {isa.load(f"&rio_re[m+{n}*ios]")}; {xi[n1]} = {isa.load(f"&rio_im[m+{n}*ios]")};')
        em.b()
        emit_r8(isa, em, fwd, xr, xi, xr, xi)
        em.b()
        for k1 in range(8):
            slot = n2 * 8 + k1
            em.o(f'{isa.store(f"&sp_re[{slot}*{sp_mul}]", xr[k1])}; {isa.store(f"&sp_im[{slot}*{sp_mul}]", xi[k1])};')
        em.b()

    # PASS 2: 8 radix-8 column combines + internal W64 twiddles
    for k1 in range(8):
        em.c(f'Column k1={k1}')
        for n2 in range(8):
            slot = n2 * 8 + k1
            em.o(f'{xr[n2]} = {isa.load(f"&sp_re[{slot}*{sp_mul}]")}; {xi[n2]} = {isa.load(f"&sp_im[{slot}*{sp_mul}]")};')
        if k1 > 0:
            for n2 in range(1, 8):
                emit_itw_apply(isa, em, xr[n2], xi[n2], (n2*k1)%64, fwd)
        em.b()
        emit_r8(isa, em, fwd, xr, xi, xr, xi)
        em.b()
        # Post-twiddle outputs (DIF), then store in-place
        for k2 in range(8):
            m_out = k1 + 8 * k2
            if m_out > 0:
                if isa.is_scalar:
                    em.o(f'{{ double wr=W_re[{m_out-1}*me+m],wi=W_im[{m_out-1}*me+m],tr={xr[k2]};')
                    if fwd:
                        em.o(f'  {xr[k2]}=tr*wr-{xi[k2]}*wi; {xi[k2]}=tr*wi+{xi[k2]}*wr; }}')
                    else:
                        em.o(f'  {xr[k2]}=tr*wr+{xi[k2]}*wi; {xi[k2]}={xi[k2]}*wr-tr*wi; }}')
                else:
                    em.o(f'{{ {T} wr={isa.load(f"&W_re[{m_out-1}*me+m]")};')
                    em.o(f'  {T} wi={isa.load(f"&W_im[{m_out-1}*me+m]")};')
                    em.o(f'  {T} tr={xr[k2]};')
                    if fwd:
                        em.o(f'  {xr[k2]}={isa.fmsub(xr[k2],"wr",isa.mul(xi[k2],"wi"))};')
                        em.o(f'  {xi[k2]}={isa.fmadd("tr","wi",isa.mul(xi[k2],"wr"))}; }}')
                    else:
                        em.o(f'  {xr[k2]}={isa.fmadd(xr[k2],"wr",isa.mul(xi[k2],"wi"))};')
                        em.o(f'  {xi[k2]}={isa.fmsub(xi[k2],"wr",isa.mul("tr","wi"))}; }}')
            if isa.is_scalar:
                em.o(f'rio_re[m*ms+{m_out}*ios]={xr[k2]}; rio_im[m*ms+{m_out}*ios]={xi[k2]};')
            else:
                em.o(f'{isa.store(f"&rio_re[m+{m_out}*ios]", xr[k2])}; {isa.store(f"&rio_im[m+{m_out}*ios]", xi[k2])};')
        em.b()

    em.ind -= 1
    em.o('}')
    em.L.append('}')
    em.L.append('')
    return em.L


def gen_file(isa_name, variant='all'):
    isa = ISA(isa_name)
    ISA_U = isa_name.upper()

    if variant == 'all':
        guard = f'FFT_RADIX64_{ISA_U}_H'
        fname = f'fft_radix64_{isa_name}.h'
        desc = 'all variants'
    elif variant == 'notw':
        guard = f'FFT_RADIX64_{ISA_U}_NOTW_H'
        fname = f'fft_radix64_{isa_name}_notw.h'
        desc = 'N1 (no twiddle)'
    elif variant == 'dit_tw':
        guard = f'FFT_RADIX64_{ISA_U}_TW_H'
        fname = f'fft_radix64_{isa_name}_tw.h'
        desc = 'DIT/DIF twiddled'
    else:
        guard = f'FFT_RADIX64_{ISA_U}_{variant.upper()}_H'
        fname = f'fft_radix64_{isa_name}_{variant}.h'
        desc = variant

    L = ['/**',
         f' * @file {fname}',
         f' * @brief DFT-64 {ISA_U} -- 8x8 CT, {desc}',
         f' *',
         f' * 8 radix-8 sub-FFTs + 8 radix-8 column combines.',
         f' * Internal twiddles: W64 constants (trivial + cmul from static arrays).',
         f' * Vector width: {isa.C} doubles, k-step: {isa.C}',
         f' * Generated by gen_radix64.py --isa {isa_name} --variant {variant}',
         ' */', '',
         f'#ifndef {guard}', f'#define {guard}',
         '#include <stddef.h>']
    if not isa.is_scalar:
        L.append('#include <immintrin.h>')
    L.append('')

    L.extend(emit_itw_static_arrays())

    if variant in ('notw', 'all'):
        for d in ('fwd', 'bwd'):
            L.extend(gen_notw(isa, d))
            L.append('')

    if variant in ('dit_tw', 'all') and not isa.is_scalar:
        for d in ('fwd', 'bwd'):
            L.extend(gen_dit_tw(isa, d))
            L.append('')

    if variant in ('dif_tw', 'dit_tw', 'all') and not isa.is_scalar:
        for d in ('fwd', 'bwd'):
            L.extend(gen_dif_tw(isa, d))
            L.append('')

    if variant in ('ct_n1', 'all'):
        L.append('/* ================================================================')
        L.append(' * FFTW-style n1: out-of-place, separate is/os')
        L.append(' * ================================================================ */')
        L.append('')
        for d in ('fwd', 'bwd'):
            L.extend(gen_ct_n1(isa, d))
            L.append('')

    if variant in ('ct_t1_dit', 'all') and not isa.is_scalar:
        L.append('/* ================================================================')
        L.append(' * FFTW-style t1 DIT: in-place twiddle + butterfly (direct loads)')
        L.append(' * ================================================================ */')
        L.append('')
        for d in ('fwd', 'bwd'):
            L.extend(gen_ct_t1_dit(isa, d, log3=False))
            L.append('')

    if variant in ('ct_t1_dit_log3', 'all') and not isa.is_scalar:
        L.append('/* ================================================================')
        L.append(' * FFTW-style t1 DIT log3: 2 base loads + 12 cmuls (high me)')
        L.append(' * ================================================================ */')
        L.append('')
        for d in ('fwd', 'bwd'):
            L.extend(gen_ct_t1_dit(isa, d, log3=True))
            L.append('')

    if variant in ('ct_t1_dif', 'all') and not isa.is_scalar:
        L.append('/* ================================================================')
        L.append(' * FFTW-style t1 DIF: in-place butterfly + post-twiddle')
        L.append(' * ================================================================ */')
        L.append('')
        for d in ('fwd', 'bwd'):
            L.extend(gen_ct_t1_dif(isa, d))
            L.append('')

    L.append(f'#endif /* {guard} */')
    return L


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Unified R=64 codelet generator')
    parser.add_argument('--isa', default='avx2',
                        choices=['scalar', 'avx2', 'avx512'])
    parser.add_argument('--variant', default='all',
                        choices=['notw', 'dit_tw', 'dif_tw', 'ct_n1', 'ct_t1_dit', 'ct_t1_dit_log3', 'ct_t1_dif', 'all'])
    # Legacy positional: gen_radix64.py avx2
    parser.add_argument('isa_pos', nargs='?', default=None)
    args = parser.parse_args()

    if args.isa_pos and args.isa_pos in ('scalar', 'avx2', 'avx512'):
        args.isa = args.isa_pos

    lines = gen_file(args.isa, args.variant)

    # Add sv variants for SIMD ISAs
    if args.isa != 'scalar':
        isa = ISA(args.isa)

        def _t2_to_sv(body):
            blines = body.split('\n')
            out = []
            in_loop = False
            depth = 0
            for line in blines:
                stripped = line.strip()
                if not in_loop and 'for (size_t k' in stripped and 'k < K' in stripped:
                    in_loop = True
                    depth = 1
                    continue
                if in_loop:
                    for ch in stripped:
                        if ch == '{': depth += 1
                        elif ch == '}': depth -= 1
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

        text = '\n'.join(lines)

        # Name mapping for R=64
        mappings = [
            ('radix64_n1_dit_kernel', 'radix64_n1sv_kernel'),
            ('radix64_tw_flat_dit_kernel', 'radix64_t1sv_dit_kernel'),
            ('radix64_tw_flat_dif_kernel', 'radix64_t1sv_dif_kernel'),
        ]

        sv_lines = []
        sv_lines.append('')
        sv_lines.append(f'/* === sv codelets: no loop, elements at stride vs === */')
        sv_lines.append(f'/* Executor calls K/{isa.C} times, advancing base pointers by {isa.C}. */')

        for t2_pat, sv_pat in mappings:
            for d in ('fwd', 'bwd'):
                func_name = f'{t2_pat}_{d}_{args.isa}'
                sv_func = f'{sv_pat}_{d}_{args.isa}'

                func_start = text.find(f'{func_name}(')
                if func_start < 0:
                    continue
                static_start = text.rfind('static', 0, func_start)
                if static_start < 0:
                    # Try attribute line
                    attr_start = text.rfind('__attribute__', 0, func_start)
                    if attr_start < 0:
                        continue
                    static_start = attr_start
                brace_start = text.find('{', func_start)
                if brace_start < 0:
                    continue

                depth = 0
                pos = brace_start
                while pos < len(text):
                    if text[pos] == '{': depth += 1
                    elif text[pos] == '}':
                        depth -= 1
                        if depth == 0: break
                    pos += 1

                func_body = text[brace_start + 1:pos]
                sv_body = _t2_to_sv(func_body)

                sig = text[static_start:brace_start]
                sig = sig.replace(func_name, sv_func)
                sig = sig.replace('size_t K)', 'size_t vs)')

                sv_lines.append(sig + '{')
                sv_lines.append(sv_body)
                sv_lines.append('}')
                sv_lines.append('')

        # Insert before #endif
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith('#endif'):
                lines[i:i] = sv_lines
                break

    print('\n'.join(lines))
