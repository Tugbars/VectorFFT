#!/usr/bin/env python3
"""
gen_radix11_n1.py — DFT-11 N1 codelet via Rader's algorithm

Rader (g=2): converts prime DFT-11 to circular convolution of length 10.
DFT-10 = 2×5 Cooley-Tukey with Winograd DFT-5 (5 muls per component).

Flow per k-step:
  1. DC accumulate: sum x[0..10]
  2. Rader permute: a[q] = x[g^q mod 11]
  3. DFT-10(a): 5 radix-2 + W₁₀ twiddles + 2×Winograd-5
  4. Pointwise multiply by precomputed kernel B[k]
  5. IDFT-10: same structure, backward
  6. Scale /10, add x[0], unpermute store

Usage:
  python3 gen_radix11_n1.py scalar > fft_radix11_scalar_n1_gen.h
  python3 gen_radix11_n1.py avx2   > fft_radix11_avx2_n1_gen.h
"""

import math, sys

N = 11

# ── Rader constants ──
PRIM_ROOT = 2
PERM  = [(PRIM_ROOT**q) % N for q in range(N-1)]       # [1,2,4,8,5,10,9,7,3,6]
GINV  = pow(PRIM_ROOT, N-2, N)                           # 6
IPERM = [(GINV**q) % N for q in range(N-1)]              # [1,6,3,7,9,10,5,8,4,2]

# ── Winograd DFT-5 constants ──
W5_ALPHA =  -0.25
W5_BETA  =   0.55901699437494745    # √5/4
W5_S2    =   0.58778525229247325    # sin(4π/5)
W5_S1mS2 =   0.36327126400268028   # sin(2π/5) - sin(4π/5)
W5_S1pS2 =   1.53884176858762678   # sin(2π/5) + sin(4π/5)

# ── W₁₀ twiddle constants ──
W10_C36 = math.cos(math.pi/5)      # cos(36°)
W10_S36 = math.sin(math.pi/5)      # sin(36°)
W10_C72 = math.cos(2*math.pi/5)    # cos(72°)
W10_S72 = math.sin(2*math.pi/5)    # sin(72°)

# ── Rader kernel B[k] = DFT-10(reversed kernel) ──
import cmath
def compute_kernel(direction):
    """Compute precomputed kernel for Rader convolution."""
    W11 = lambda n: cmath.exp(-2j * math.pi * n / 11)
    W10f = lambda k: cmath.exp(-2j * math.pi * k / 10)
    if direction == 'fwd':
        kern = [W11(PERM[q]) for q in range(10)]
    else:
        kern = [W11(PERM[q]).conjugate() for q in range(10)]
    kern_rev = [kern[(-q) % 10] for q in range(10)]
    B = [sum(kern_rev[q] * W10f(q*k) for q in range(10)) for k in range(10)]
    return [(b.real, b.imag) for b in B]

B_FWD = compute_kernel('fwd')
B_BWD = compute_kernel('bwd')

# ── ISA abstraction ──
class Scalar:
    name='scalar'; vtype='double'; k_step=1; prefix='R11S'
    target=''
    sign_flip=None
    @staticmethod
    def add(a,b): return f"({a}+{b})"
    @staticmethod
    def sub(a,b): return f"({a}-{b})"
    @staticmethod
    def mul(a,b): return f"({a}*{b})"
    @staticmethod
    def neg(a):   return f"(-{a})"
    @staticmethod
    def fmadd(a,b,c): return f"({a}*{b}+{c})"
    @staticmethod
    def fmsub(a,b,c): return f"({a}*{b}-{c})"
    @staticmethod
    def bcast(v): return str(v)
    @staticmethod
    def spill_st(ptr,v): return f"*({ptr})={v};"
    @staticmethod
    def spill_ld(ptr): return f"*({ptr})"
    spill_align=''; spill_mul=1

class AVX2:
    name='avx2'; vtype='__m256d'; k_step=4; prefix='R11A'
    target='__attribute__((target("avx2,fma")))'
    sign_flip='const __m256d sign_flip = _mm256_set1_pd(-0.0);'
    @staticmethod
    def add(a,b): return f"_mm256_add_pd({a},{b})"
    @staticmethod
    def sub(a,b): return f"_mm256_sub_pd({a},{b})"
    @staticmethod
    def mul(a,b): return f"_mm256_mul_pd({a},{b})"
    @staticmethod
    def neg(a):   return f"_mm256_xor_pd({a},sign_flip)"
    @staticmethod
    def fmadd(a,b,c): return f"_mm256_fmadd_pd({a},{b},{c})"
    @staticmethod
    def fmsub(a,b,c): return f"_mm256_fmsub_pd({a},{b},{c})"
    @staticmethod
    def bcast(v): return f"_mm256_set1_pd({v})"
    @staticmethod
    def spill_st(ptr,v): return f"_mm256_store_pd({ptr},{v});"
    @staticmethod
    def spill_ld(ptr): return f"_mm256_load_pd({ptr})"
    spill_align='__attribute__((aligned(32)))'; spill_mul=4

class AVX512:
    name='avx512'; vtype='__m512d'; k_step=8; prefix='R11Z'
    target='__attribute__((target("avx512f,avx512dq,fma")))'
    sign_flip='const __m512d sign_flip = _mm512_set1_pd(-0.0);'
    @staticmethod
    def add(a,b): return f"_mm512_add_pd({a},{b})"
    @staticmethod
    def sub(a,b): return f"_mm512_sub_pd({a},{b})"
    @staticmethod
    def mul(a,b): return f"_mm512_mul_pd({a},{b})"
    @staticmethod
    def neg(a):   return f"_mm512_xor_pd({a},sign_flip)"
    @staticmethod
    def fmadd(a,b,c): return f"_mm512_fmadd_pd({a},{b},{c})"
    @staticmethod
    def fmsub(a,b,c): return f"_mm512_fmsub_pd({a},{b},{c})"
    @staticmethod
    def bcast(v): return f"_mm512_set1_pd({v})"
    @staticmethod
    def spill_st(ptr,v): return f"_mm512_store_pd({ptr},{v});"
    @staticmethod
    def spill_ld(ptr): return f"_mm512_load_pd({ptr})"
    spill_align='__attribute__((aligned(64)))'; spill_mul=8

ISAS = {'scalar': Scalar, 'avx2': AVX2, 'avx512': AVX512}

class E:
    """Emitter."""
    def __init__(s, I):
        s.I=I; s.L=[]; s.ind=1
    def o(s,t=""): s.L.append("    "*s.ind + t)
    def c(s,t): s.o(f"/* {t} */")
    def b(s): s.L.append("")
    def load(s,v,n):
        s.o(f"{v}_re={s.I.prefix}_LD(&in_re[{n}*K+k]);")
        s.o(f"{v}_im={s.I.prefix}_LD(&in_im[{n}*K+k]);")
    def store(s,v,m):
        s.o(f"{s.I.prefix}_ST(&out_re[{m}*K+k],{v}_re);")
        s.o(f"{s.I.prefix}_ST(&out_im[{m}*K+k],{v}_im);")
    def spill(s,v,slot):
        I=s.I; m=I.spill_mul; w='256' if I.k_step==4 else '512'
        if I.name=='scalar':
            s.o(f"sp_re[{slot}]={v}_re; sp_im[{slot}]={v}_im;")
        else:
            s.o(f"_mm{w}_store_pd(&sp_re[{slot}*{m}],{v}_re);")
            s.o(f"_mm{w}_store_pd(&sp_im[{slot}*{m}],{v}_im);")
    def reload(s,v,slot):
        I=s.I; m=I.spill_mul; w='256' if I.k_step==4 else '512'
        if I.name=='scalar':
            s.o(f"{v}_re=sp_re[{slot}]; {v}_im=sp_im[{slot}];")
        else:
            s.o(f"{v}_re=_mm{w}_load_pd(&sp_re[{slot}*{m}]);")
            s.o(f"{v}_im=_mm{w}_load_pd(&sp_im[{slot}*{m}]);")

    def emit_winograd5(s, r, d, label=""):
        """Emit Winograd DFT-5 on vars r[0..4], direction d='fwd'/'bwd'."""
        I=s.I; T=I.vtype; fwd=(d=='fwd')
        if label: s.c(f"{label} [{d}]")
        v = r  # list of 5 var name prefixes
        s.o(f"{{ {T} a1r,a1i,a2r,a2i,b1r,b1i,b2r,b2i;")
        s.o(f"  {T} t1r,t1i,t2r,t2i,x0sr,x0si;")
        s.o(f"  {T} m1r,m1i,m2r,m2i;")
        s.o(f"  {T} p1r,p1i,p2r,p2i,p3r,p3i,q1r,q1i,q2r,q2i;")
        s.o(f"  {T} w1r,w1i,w2r,w2i,w3r,w3i;")
        s.o(f"  a1r={I.add(f'{v[1]}_re',f'{v[4]}_re')}; a1i={I.add(f'{v[1]}_im',f'{v[4]}_im')};")
        s.o(f"  a2r={I.add(f'{v[2]}_re',f'{v[3]}_re')}; a2i={I.add(f'{v[2]}_im',f'{v[3]}_im')};")
        s.o(f"  b1r={I.sub(f'{v[1]}_re',f'{v[4]}_re')}; b1i={I.sub(f'{v[1]}_im',f'{v[4]}_im')};")
        s.o(f"  b2r={I.sub(f'{v[2]}_re',f'{v[3]}_re')}; b2i={I.sub(f'{v[2]}_im',f'{v[3]}_im')};")
        s.o(f"  t1r={I.add('a1r','a2r')}; t1i={I.add('a1i','a2i')};")
        s.o(f"  t2r={I.sub('a1r','a2r')}; t2i={I.sub('a1i','a2i')};")
        s.o(f"  x0sr={v[0]}_re; x0si={v[0]}_im;")
        s.o(f"  {v[0]}_re={I.add('x0sr','t1r')}; {v[0]}_im={I.add('x0si','t1i')};")
        s.o(f"  m1r={I.mul('t1r','cW5a')}; m1i={I.mul('t1i','cW5a')};")
        s.o(f"  m2r={I.mul('t2r','cW5b')}; m2i={I.mul('t2i','cW5b')};")
        s.o(f"  p1r={I.mul('cW5s2',I.add('b1r','b2r'))}; p1i={I.mul('cW5s2',I.add('b1i','b2i'))};")
        s.o(f"  p2r={I.mul('cW5d',  'b1r')};              p2i={I.mul('cW5d',  'b1i')};")
        s.o(f"  p3r={I.mul('cW5p',  'b2r')};              p3i={I.mul('cW5p',  'b2i')};")
        s.o(f"  q1r={I.add('p1r','p2r')}; q1i={I.add('p1i','p2i')};")
        s.o(f"  q2r={I.sub('p1r','p3r')}; q2i={I.sub('p1i','p3i')};")
        s.o(f"  w1r={I.add('x0sr','m1r')}; w1i={I.add('x0si','m1i')};")
        s.o(f"  w2r={I.add('w1r','m2r')}; w2i={I.add('w1i','m2i')};")
        s.o(f"  w3r={I.sub('w1r','m2r')}; w3i={I.sub('w1i','m2i')};")
        if fwd:
            s.o(f"  {v[1]}_re={I.add('w2r','q1i')}; {v[1]}_im={I.sub('w2i','q1r')};")
            s.o(f"  {v[4]}_re={I.sub('w2r','q1i')}; {v[4]}_im={I.add('w2i','q1r')};")
            s.o(f"  {v[2]}_re={I.add('w3r','q2i')}; {v[2]}_im={I.sub('w3i','q2r')};")
            s.o(f"  {v[3]}_re={I.sub('w3r','q2i')}; {v[3]}_im={I.add('w3i','q2r')};")
        else:
            s.o(f"  {v[1]}_re={I.sub('w2r','q1i')}; {v[1]}_im={I.add('w2i','q1r')};")
            s.o(f"  {v[4]}_re={I.add('w2r','q1i')}; {v[4]}_im={I.sub('w2i','q1r')};")
            s.o(f"  {v[2]}_re={I.sub('w3r','q2i')}; {v[2]}_im={I.add('w3i','q2r')};")
            s.o(f"  {v[3]}_re={I.add('w3r','q2i')}; {v[3]}_im={I.sub('w3i','q2r')};")
        s.o(f"}}")

    def emit_cmul(s, dr, di, ar, ai, wr, wi):
        """Complex multiply (ar,ai)×(wr,wi) → (dr,di)."""
        I=s.I
        s.o(f"{{ {I.vtype} _tr={ar};")
        s.o(f"  {dr}={I.fmsub(ar,wr,I.mul(ai,wi))};")
        s.o(f"  {di}={I.fmadd('_tr',wi,I.mul(ai,wr))}; }}")


def emit_kernel_spillfree(em, direction, B_kernel):
    """Spill-free DFT-11 for AVX-512: all 10 DFT-10 values in registers.

    Register budget: v0..v9 (20 ZMM) + dc,x0 (4 ZMM) = 24 of 32.
    8 ZMM free for Winograd-5 temps (peak live = 8). Exact fit.
    100 L1 spill ops eliminated vs generic path.
    B[0]=-1 → negate. B[5]=pure imag → j-rotation.
    """
    I = em.I; T = I.vtype; fwd = (direction == 'fwd')
    perm = PERM; iperm = IPERM

    em.L.append(f"static {I.target} void")
    em.L.append(f"radix11_n1_dit_kernel_{direction}_{I.name}(")
    em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
    em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
    em.L.append(f"    size_t K)")
    em.L.append(f"{{")
    em.ind = 1

    em.o(I.sign_flip)
    em.b()

    # 10 complex working vars + dc + x0
    em.o(f"{T} v0_re,v0_im,v1_re,v1_im,v2_re,v2_im,v3_re,v3_im,v4_re,v4_im;")
    em.o(f"{T} v5_re,v5_im,v6_re,v6_im,v7_re,v7_im,v8_re,v8_im,v9_re,v9_im;")
    em.o(f"{T} dc_re,dc_im,x0_re,x0_im;")
    em.b()

    # Broadcast constants
    em.c("Winograd DFT-5 constants")
    em.o(f"const {T} cW5a  = {I.bcast(W5_ALPHA)};")
    em.o(f"const {T} cW5b  = {I.bcast(W5_BETA)};")
    em.o(f"const {T} cW5s2 = {I.bcast(W5_S2)};")
    em.o(f"const {T} cW5d  = {I.bcast(W5_S1mS2)};")
    em.o(f"const {T} cW5p  = {I.bcast(W5_S1pS2)};")
    em.b()

    em.c("W₁₀ twiddle constants")
    em.o(f"const {T} cC36 = {I.bcast(W10_C36)};")
    em.o(f"const {T} cS36 = {I.bcast(W10_S36)};")
    em.o(f"const {T} cC72 = {I.bcast(W10_C72)};")
    em.o(f"const {T} cS72 = {I.bcast(W10_S72)};")
    em.b()

    em.c("Rader kernel (only non-trivial entries)")
    # B[0] = -1 (trivial), B[5] = pure imag
    for k in range(10):
        br, bi = B_kernel[k]
        if k == 0: continue  # B[0] = -1, handled as negate
        if abs(br) < 1e-15:  # pure imaginary
            em.o(f"const {T} Bk{k}_im = {I.bcast(f'{bi:.20e}')};")
        else:
            em.o(f"const {T} Bk{k}_re = {I.bcast(f'{br:.20e}')};")
            em.o(f"const {T} Bk{k}_im = {I.bcast(f'{bi:.20e}')};")
    em.b()

    em.o(f"const {T} inv10 = {I.bcast('0.1')};")
    em.b()

    vn_even = [f"v{i}" for i in range(5)]
    vn_odd = [f"v{i}" for i in range(5, 10)]

    em.o(f"for (size_t k = 0; k < K; k += {I.k_step}) {{")
    em.ind += 1

    # ── 1. DC ──
    em.c("1. DC = sum(x[0..10])")
    em.o(f"dc_re = {I.prefix}_LD(&in_re[0*K+k]); dc_im = {I.prefix}_LD(&in_im[0*K+k]);")
    em.o(f"x0_re = dc_re; x0_im = dc_im;")
    for n in range(1, 11):
        em.o(f"{{ {T} t = {I.prefix}_LD(&in_re[{n}*K+k]); dc_re = {I.add('dc_re','t')};")
        em.o(f"  t = {I.prefix}_LD(&in_im[{n}*K+k]); dc_im = {I.add('dc_im','t')}; }}")
    em.b()

    # ── 2. Load all 10 permuted inputs ──
    em.c(f"2. Rader permute: load all 10 into v0..v9")
    em.c(f"   perm = {perm}")
    for q in range(10):
        em.o(f"v{q}_re = {I.prefix}_LD(&in_re[{perm[q]}*K+k]); v{q}_im = {I.prefix}_LD(&in_im[{perm[q]}*K+k]);")
    em.b()

    # ── 3. DFT-10 pass 1: 5 radix-2 in-place ──
    em.c("3. DFT-10 radix-2: v[n]±v[n+5]")
    for n in range(5):
        em.o(f"{{ {T} tr={I.add(f'v{n}_re',f'v{n+5}_re')}, ti={I.add(f'v{n}_im',f'v{n+5}_im')};")
        em.o(f"  v{n+5}_re={I.sub(f'v{n}_re',f'v{n+5}_re')}; v{n+5}_im={I.sub(f'v{n}_im',f'v{n+5}_im')};")
        em.o(f"  v{n}_re=tr; v{n}_im=ti; }}")
    em.b()

    # ── 4. W₁₀ twiddles on odd half (v5..v9) ──
    em.c("4. W₁₀ twiddles on v5..v9 (forward)")
    # W₁₀^1=(C36,-S36), W₁₀^2=(C72,-S72), W₁₀^3=(-C72,-S72), W₁₀^4=(-C36,-S36)
    tw_specs = [(6,'cC36','cS36',False), (7,'cC72','cS72',False),
                (8,'cC72','cS72',True),  (9,'cC36','cS36',True)]
    for vi, cr, si, neg_cos in tw_specs:
        em.o(f"{{ {T} _tr = v{vi}_re;")
        # Forward W₁₀: (cr,-si) or (-cr,-si)
        if not neg_cos:
            em.o(f"  v{vi}_re = {I.fmadd(f'v{vi}_re',cr,I.mul(f'v{vi}_im',si))};")
            em.o(f"  v{vi}_im = {I.fmsub(f'v{vi}_im',cr,I.mul('_tr',si))}; }}")
        else:
            em.o(f"  v{vi}_re = {I.neg(I.fmsub(f'v{vi}_re',cr,I.mul(f'v{vi}_im',si)))};")
            em.o(f"  v{vi}_im = {I.neg(I.fmadd('_tr',si,I.mul(f'v{vi}_im',cr)))}; }}")
    em.b()

    # ── 5. Winograd-5 on even (v0..v4) and odd (v5..v9) ──
    em.emit_winograd5(vn_even, 'fwd', "DFT-10 Winograd-5 even (v0..v4)")
    em.b()
    em.emit_winograd5(vn_odd, 'fwd', "DFT-10 Winograd-5 odd (v5..v9)")
    em.b()

    # ── 6. Pointwise multiply ──
    # Even: v0×B[0], v1×B[2], v2×B[4], v3×B[6], v4×B[8]
    # Odd:  v5×B[1], v6×B[3], v7×B[5], v8×B[7], v9×B[9]
    em.c("6. Pointwise multiply by Rader kernel")

    # v0 × B[0] = -1: just negate
    em.c("B[0] = -1: negate")
    em.o(f"v0_re = {I.neg('v0_re')}; v0_im = {I.neg('v0_im')};")

    # Generic cmul for the rest of even
    even_bk = [(1, 2), (2, 4), (3, 6), (4, 8)]
    for vi, bk in even_bk:
        em.emit_cmul(f"v{vi}_re", f"v{vi}_im",
                     f"v{vi}_re", f"v{vi}_im",
                     f"Bk{bk}_re", f"Bk{bk}_im")

    # Odd cmuls
    odd_bk = [(5, 1), (6, 3), (8, 7), (9, 9)]
    for vi, bk in odd_bk:
        em.emit_cmul(f"v{vi}_re", f"v{vi}_im",
                     f"v{vi}_re", f"v{vi}_im",
                     f"Bk{bk}_re", f"Bk{bk}_im")

    # v7 × B[5] = pure imaginary: (re,im)×(0,b) = (-im*b, re*b)
    b5_bk = 5
    em.c(f"B[{b5_bk}] = pure imaginary: j-rotation")
    em.o(f"{{ {T} tr = v7_re;")
    em.o(f"  v7_re = {I.neg(I.mul('v7_im',f'Bk{b5_bk}_im'))};")
    em.o(f"  v7_im = {I.mul('tr',f'Bk{b5_bk}_im')}; }}")
    em.b()

    # ── 7. IDFT-10 ──
    em.c("7. IDFT-10: backward Winograd-5 + conjugate W₁₀ + radix-2")
    em.emit_winograd5(vn_even, 'bwd', "IDFT Winograd-5 even (v0..v4)")
    em.b()
    em.emit_winograd5(vn_odd, 'bwd', "IDFT Winograd-5 odd (v5..v9)")
    em.b()

    # IDFT W₁₀ conjugate twiddles on v5..v9
    em.c("IDFT W₁₀ conjugate twiddles on v5..v9")
    for vi, cr, si, neg_cos in tw_specs:
        em.o(f"{{ {T} _tr = v{vi}_re;")
        # Conjugate W₁₀: (cr,+si) or (-cr,+si)
        if not neg_cos:
            em.o(f"  v{vi}_re = {I.fmsub(f'v{vi}_re',cr,I.mul(f'v{vi}_im',si))};")
            em.o(f"  v{vi}_im = {I.fmadd('_tr',si,I.mul(f'v{vi}_im',cr))}; }}")
        else:
            em.o(f"  v{vi}_re = {I.neg(I.fmadd(f'v{vi}_re',cr,I.mul(f'v{vi}_im',si)))};")
            em.o(f"  v{vi}_im = {I.neg(I.fmsub(f'v{vi}_im',cr,I.mul('_tr',si)))}; }}")
    em.b()

    # ── 8. Radix-2 combine + scale + unpermute + store ──
    em.c("8. Radix-2 combine, scale /10, add x[0], unpermute, store")
    em.c(f"   iperm = {iperm}")
    # c[n] = (v[n] + v[n+5]) / 10,  c[n+5] = (v[n] - v[n+5]) / 10
    # X[iperm[n]] = x[0] + c[n],  X[iperm[n+5]] = x[0] + c[n+5]
    for n in range(5):
        q1, q2 = n, n + 5
        idx1, idx2 = iperm[q1], iperm[q2]
        em.o(f"{{ {T} s_re = {I.mul(I.add(f'v{n}_re',f'v{n+5}_re'),'inv10')};")
        em.o(f"  {T} s_im = {I.mul(I.add(f'v{n}_im',f'v{n+5}_im'),'inv10')};")
        em.o(f"  {T} d_re = {I.mul(I.sub(f'v{n}_re',f'v{n+5}_re'),'inv10')};")
        em.o(f"  {T} d_im = {I.mul(I.sub(f'v{n}_im',f'v{n+5}_im'),'inv10')};")
        em.o(f"  {I.prefix}_ST(&out_re[{idx1}*K+k], {I.add('x0_re','s_re')});")
        em.o(f"  {I.prefix}_ST(&out_im[{idx1}*K+k], {I.add('x0_im','s_im')});")
        em.o(f"  {I.prefix}_ST(&out_re[{idx2}*K+k], {I.add('x0_re','d_re')});")
        em.o(f"  {I.prefix}_ST(&out_im[{idx2}*K+k], {I.add('x0_im','d_im')}); }}")
    em.b()

    em.c("DC output")
    em.o(f"{I.prefix}_ST(&out_re[0*K+k], dc_re);")
    em.o(f"{I.prefix}_ST(&out_im[0*K+k], dc_im);")
    em.b()

    em.ind -= 1
    em.o("}")
    em.L.append("}")
    em.L.append("")


def emit_kernel(em, direction, B_kernel):
    I = em.I; T = I.vtype; fwd = (direction == 'fwd')
    m = I.spill_mul; w = '256' if I.k_step == 4 else '512'
    perm = PERM; iperm = IPERM

    if I.target:
        em.L.append(f"static {I.target} void")
    else:
        em.L.append(f"static void")
    em.L.append(f"radix11_n1_dit_kernel_{direction}_{I.name}(")
    em.L.append(f"    const double * __restrict__ in_re, const double * __restrict__ in_im,")
    em.L.append(f"    double * __restrict__ out_re, double * __restrict__ out_im,")
    em.L.append(f"    size_t K)")
    em.L.append(f"{{")
    em.ind = 1

    if I.sign_flip: em.o(I.sign_flip)
    em.b()

    # Spill buffer for DFT-10 intermediate
    m = I.spill_mul
    if I.name == 'scalar':
        em.o(f"double sp_re[10], sp_im[10];")
    else:
        em.o(f"{I.spill_align} double sp_re[10*{m}];")
        em.o(f"{I.spill_align} double sp_im[10*{m}];")
    em.b()

    # Working vars: 5 complex for Winograd-5 + DC accum + x0 save
    em.o(f"{T} v0_re,v0_im,v1_re,v1_im,v2_re,v2_im,v3_re,v3_im,v4_re,v4_im;")
    em.o(f"{T} dc_re,dc_im,x0_re,x0_im;")
    em.b()

    # Broadcast constants
    em.c("Winograd DFT-5 constants")
    em.o(f"const {T} cW5a  = {I.bcast(W5_ALPHA)};")
    em.o(f"const {T} cW5b  = {I.bcast(W5_BETA)};")
    em.o(f"const {T} cW5s2 = {I.bcast(W5_S2)};")
    em.o(f"const {T} cW5d  = {I.bcast(W5_S1mS2)};")
    em.o(f"const {T} cW5p  = {I.bcast(W5_S1pS2)};")
    em.b()

    em.c("W₁₀ twiddle constants")
    em.o(f"const {T} cC36 = {I.bcast(W10_C36)};")
    em.o(f"const {T} cS36 = {I.bcast(W10_S36)};")
    em.o(f"const {T} cC72 = {I.bcast(W10_C72)};")
    em.o(f"const {T} cS72 = {I.bcast(W10_S72)};")
    em.b()

    em.c("Rader kernel B[k] (precomputed)")
    for k in range(10):
        br, bi = B_kernel[k]
        em.o(f"const {T} Bk{k}_re = {I.bcast(f'{br:.20e}')};")
        em.o(f"const {T} Bk{k}_im = {I.bcast(f'{bi:.20e}')};")
    em.b()

    em.o(f"const {T} inv10 = {I.bcast('0.1')};")
    em.b()

    # k-loop
    em.o(f"for (size_t k = 0; k < K; k += {I.k_step}) {{")
    em.ind += 1

    # ── 1. DC accumulate ──
    em.c("1. DC = sum(x[0..10])")
    em.load("dc", 0)
    em.o(f"x0_re = dc_re; x0_im = dc_im;")  # save x[0]
    for n in range(1, 11):
        em.o(f"{{ {T} tr,ti;")
        em.o(f"  tr = {I.prefix}_LD(&in_re[{n}*K+k]);")
        em.o(f"  ti = {I.prefix}_LD(&in_im[{n}*K+k]);")
        em.o(f"  dc_re = {I.add('dc_re','tr')}; dc_im = {I.add('dc_im','ti')}; }}")
    em.b()

    # ── 2. Rader permute + DFT-10 ──
    em.c("2. Rader permute: load a[q] = x[g^q mod 11]")
    em.c(f"   perm = {perm}")
    # Load first 5 into v0..v4 (first half of DFT-10)
    for q in range(5):
        em.load(f"v{q}", perm[q])
    em.b()

    # ── 3. DFT-10 pass 1: 5 radix-2 with second half ──
    em.c("3. DFT-10 pass 1: 5 radix-2, spill even half")
    for n in range(5):
        em.o(f"{{ {T} tr,ti;")
        em.o(f"  tr = {I.prefix}_LD(&in_re[{perm[n+5]}*K+k]);")
        em.o(f"  ti = {I.prefix}_LD(&in_im[{perm[n+5]}*K+k]);")
        em.o(f"  {T} a_re = {I.add(f'v{n}_re','tr')};")
        em.o(f"  {T} a_im = {I.add(f'v{n}_im','ti')};")
        em.o(f"  v{n}_re = {I.sub(f'v{n}_re','tr')};")
        em.o(f"  v{n}_im = {I.sub(f'v{n}_im','ti')};")
        if I.name == 'scalar':
            em.o(f"  sp_re[{n}] = a_re; sp_im[{n}] = a_im;")
        else:
            em.o(f"  _mm{w}_store_pd(&sp_re[{n}*{m}], a_re);")
            em.o(f"  _mm{w}_store_pd(&sp_im[{n}*{m}], a_im);")
        em.o(f"}}")
    em.b()

    # v0..v4 now hold b[0..4] (odd half)
    # Apply W₁₀ twiddles to b[1..4]
    em.c("W₁₀ twiddles on b[1..4]")
    # W₁₀¹ = (C36, -S36)  fwd, (C36, +S36) bwd
    # W₁₀² = (C72, -S72)  fwd, (C72, +S72) bwd
    # W₁₀³ = (-C72, -S72) fwd, (-C72, +S72) bwd
    # W₁₀⁴ = (-C36, -S36) fwd, (-C36, +S36) bwd
    tw_specs = [
        (1, 'cC36', 'cS36'),
        (2, 'cC72', 'cS72'),
        (3, 'cC72', 'cS72'),  # negated cos
        (4, 'cC36', 'cS36'),  # negated cos
    ]
    for idx, (bidx, cr, si) in enumerate(tw_specs):
        neg_cos = (bidx >= 3)
        wr = I.neg(cr) if neg_cos else cr
        if True:  # DFT-10 always forward
            wi = I.neg(si)
        else:
            wi = si
            if neg_cos:
                wi = si  # still positive for bwd with neg cos
        # Actually let me be more careful
        # W₁₀^k = cos(2πk/10) - j·sin(2πk/10) for forward
        # k=1: ( cos36, -sin36)
        # k=2: ( cos72, -sin72)
        # k=3: (-cos72, -sin72)
        # k=4: (-cos36, -sin36)
        # For backward (conjugate): flip sign of imaginary
        em.o(f"{{ {T} _tr = v{bidx}_re;")
        if bidx == 1:
            if True:  # DFT-10 always forward
                em.o(f"  v{bidx}_re = {I.fmadd(f'v{bidx}_re','cC36',I.mul(f'v{bidx}_im','cS36'))};")
                em.o(f"  v{bidx}_im = {I.fmsub(f'v{bidx}_im','cC36',I.mul('_tr','cS36'))}; }}")
            else:
                em.o(f"  v{bidx}_re = {I.fmsub(f'v{bidx}_re','cC36',I.mul(f'v{bidx}_im','cS36'))};")
                em.o(f"  v{bidx}_im = {I.fmadd('_tr','cS36',I.mul(f'v{bidx}_im','cC36'))}; }}")
        elif bidx == 2:
            if True:  # DFT-10 always forward
                em.o(f"  v{bidx}_re = {I.fmadd(f'v{bidx}_re','cC72',I.mul(f'v{bidx}_im','cS72'))};")
                em.o(f"  v{bidx}_im = {I.fmsub(f'v{bidx}_im','cC72',I.mul('_tr','cS72'))}; }}")
            else:
                em.o(f"  v{bidx}_re = {I.fmsub(f'v{bidx}_re','cC72',I.mul(f'v{bidx}_im','cS72'))};")
                em.o(f"  v{bidx}_im = {I.fmadd('_tr','cS72',I.mul(f'v{bidx}_im','cC72'))}; }}")
        elif bidx == 3:
            if True:  # DFT-10 always forward
                em.o(f"  v{bidx}_re = {I.neg(I.fmsub(f'v{bidx}_re','cC72',I.mul(f'v{bidx}_im','cS72')))};")
                em.o(f"  v{bidx}_im = {I.neg(I.fmadd('_tr','cS72',I.mul(f'v{bidx}_im','cC72')))}; }}")
            else:
                em.o(f"  v{bidx}_re = {I.neg(I.fmadd(f'v{bidx}_re','cC72',I.mul(f'v{bidx}_im','cS72')))};")
                em.o(f"  v{bidx}_im = {I.neg(I.fmsub(f'v{bidx}_im','cC72',I.mul('_tr','cS72')))}; }}")
        elif bidx == 4:
            if True:  # DFT-10 always forward
                em.o(f"  v{bidx}_re = {I.neg(I.fmsub(f'v{bidx}_re','cC36',I.mul(f'v{bidx}_im','cS36')))};")
                em.o(f"  v{bidx}_im = {I.neg(I.fmadd('_tr','cS36',I.mul(f'v{bidx}_im','cC36')))}; }}")
            else:
                em.o(f"  v{bidx}_re = {I.neg(I.fmadd(f'v{bidx}_re','cC36',I.mul(f'v{bidx}_im','cS36')))};")
                em.o(f"  v{bidx}_im = {I.neg(I.fmsub(f'v{bidx}_im','cC36',I.mul('_tr','cS36')))}; }}")
    em.b()

    # Winograd-5 on odd half (b[0..4] in v0..v4)
    vn = [f"v{i}" for i in range(5)]
    em.emit_winograd5(vn, 'fwd', "DFT-10 Winograd-5 odd half")
    em.b()

    # Spill odd results to sp[5..9]
    em.c("Spill odd DFT-10 outputs")
    for k in range(5):
        em.spill(f"v{k}", 5+k)
    em.b()

    # Reload even half, Winograd-5
    em.c("Reload even half, Winograd-5")
    for k in range(5):
        em.reload(f"v{k}", k)
    em.emit_winograd5(vn, 'fwd', "DFT-10 Winograd-5 even half")
    em.b()

    # ── 4. Pointwise multiply ──
    # Even outputs in v0..v4, odd in spill[5..9]
    # DFT-10 output: X[2k] = even[k], X[2k+1] = odd[k]
    # Multiply: C[2k] = v[k] * B[2k], C[2k+1] = odd[k] * B[2k+1]
    em.c("4. Pointwise multiply by Rader kernel")
    # Even (in registers): v[k] × B[2k]
    for k in range(5):
        bk = 2*k
        em.emit_cmul(f"v{k}_re", f"v{k}_im",
                     f"v{k}_re", f"v{k}_im",
                     f"Bk{bk}_re", f"Bk{bk}_im")
    # Spill even products
    for k in range(5):
        em.spill(f"v{k}", k)
    em.b()

    # Reload odd, multiply, spill
    em.c("Odd pointwise multiply")
    for k in range(5):
        em.reload(f"v{k}", 5+k)
    for k in range(5):
        bk = 2*k+1
        em.emit_cmul(f"v{k}_re", f"v{k}_im",
                     f"v{k}_re", f"v{k}_im",
                     f"Bk{bk}_re", f"Bk{bk}_im")
    em.b()

    # ── 5. IDFT-10 ──
    # Now: even products in spill[0..4], odd products in v0..v4
    # IDFT-10 = backward DFT-10
    # Structure: de-interleave → backward Winograd-5 × 2 → backward W₁₀ twiddles → 5 radix-2

    # First, backward Winograd-5 on odd half (already in v0..v4)
    bwd_dir = 'bwd'  # IDFT-10 always backward
    em.c("5. IDFT-10: Winograd-5 backward on odd half")
    em.emit_winograd5(vn, bwd_dir, "IDFT Winograd-5 odd")
    em.b()

    # Backward W₁₀ twiddles on b[1..4] (conjugate of forward)
    em.c("IDFT W₁₀ twiddles (conjugate)")
    for bidx in [1,2,3,4]:
        em.o(f"{{ {T} _tr = v{bidx}_re;")
        if bidx == 1:
            if True:  # IDFT-10 always conjugate
                em.o(f"  v{bidx}_re = {I.fmsub(f'v{bidx}_re','cC36',I.mul(f'v{bidx}_im','cS36'))};")
                em.o(f"  v{bidx}_im = {I.fmadd('_tr','cS36',I.mul(f'v{bidx}_im','cC36'))}; }}")
            else:
                em.o(f"  v{bidx}_re = {I.fmadd(f'v{bidx}_re','cC36',I.mul(f'v{bidx}_im','cS36'))};")
                em.o(f"  v{bidx}_im = {I.fmsub(f'v{bidx}_im','cC36',I.mul('_tr','cS36'))}; }}")
        elif bidx == 2:
            if True:  # IDFT-10 always conjugate
                em.o(f"  v{bidx}_re = {I.fmsub(f'v{bidx}_re','cC72',I.mul(f'v{bidx}_im','cS72'))};")
                em.o(f"  v{bidx}_im = {I.fmadd('_tr','cS72',I.mul(f'v{bidx}_im','cC72'))}; }}")
            else:
                em.o(f"  v{bidx}_re = {I.fmadd(f'v{bidx}_re','cC72',I.mul(f'v{bidx}_im','cS72'))};")
                em.o(f"  v{bidx}_im = {I.fmsub(f'v{bidx}_im','cC72',I.mul('_tr','cS72'))}; }}")
        elif bidx == 3:
            if True:  # IDFT-10 always conjugate
                em.o(f"  v{bidx}_re = {I.neg(I.fmadd(f'v{bidx}_re','cC72',I.mul(f'v{bidx}_im','cS72')))};")
                em.o(f"  v{bidx}_im = {I.neg(I.fmsub(f'v{bidx}_im','cC72',I.mul('_tr','cS72')))}; }}")
            else:
                em.o(f"  v{bidx}_re = {I.neg(I.fmsub(f'v{bidx}_re','cC72',I.mul(f'v{bidx}_im','cS72')))};")
                em.o(f"  v{bidx}_im = {I.neg(I.fmadd('_tr','cS72',I.mul(f'v{bidx}_im','cC72')))}; }}")
        elif bidx == 4:
            if True:  # IDFT-10 always conjugate
                em.o(f"  v{bidx}_re = {I.neg(I.fmadd(f'v{bidx}_re','cC36',I.mul(f'v{bidx}_im','cS36')))};")
                em.o(f"  v{bidx}_im = {I.neg(I.fmsub(f'v{bidx}_im','cC36',I.mul('_tr','cS36')))}; }}")
            else:
                em.o(f"  v{bidx}_re = {I.neg(I.fmsub(f'v{bidx}_re','cC36',I.mul(f'v{bidx}_im','cS36')))};")
                em.o(f"  v{bidx}_im = {I.neg(I.fmadd('_tr','cS36',I.mul(f'v{bidx}_im','cC36')))}; }}")
    em.b()

    # Spill odd after IDFT twiddles
    for k in range(5):
        em.spill(f"v{k}", 5+k)
    em.b()

    # Reload even products, backward Winograd-5
    em.c("IDFT Winograd-5 on even half")
    for k in range(5):
        em.reload(f"v{k}", k)
    em.emit_winograd5(vn, bwd_dir, "IDFT Winograd-5 even")
    em.b()

    # Radix-2 combine: c[n] = even[n] + odd[n], c[n+5] = even[n] - odd[n]
    # Then scale by 1/10
    em.c("IDFT radix-2 combine + scale by 1/10")
    for n in range(5):
        em.o(f"{{ {T} or_,oi;")
        if I.name == 'scalar':
            em.o(f"  or_ = sp_re[{5+n}]; oi = sp_im[{5+n}];")
        else:
            w = '256' if I.k_step==4 else '512'
            em.o(f"  or_ = _mm{w}_load_pd(&sp_re[{5+n}*{m}]); oi = _mm{w}_load_pd(&sp_im[{5+n}*{m}]);")
        # c[n] = (even[n] + odd[n]) / 10
        # c[n+5] = (even[n] - odd[n]) / 10
        if I.name == 'scalar':
            em.o(f"  sp_re[{n}] = (v{n}_re + or_) * 0.1;")
            em.o(f"  sp_im[{n}] = (v{n}_im + oi) * 0.1;")
            em.o(f"  sp_re[{5+n}] = (v{n}_re - or_) * 0.1;")
            em.o(f"  sp_im[{5+n}] = (v{n}_im - oi) * 0.1;")
        else:
            em.o(f"  _mm{w}_store_pd(&sp_re[{n}*{m}], {I.mul(I.add(f'v{n}_re','or_'),'inv10')});")
            em.o(f"  _mm{w}_store_pd(&sp_im[{n}*{m}], {I.mul(I.add(f'v{n}_im','oi'),'inv10')});")
            em.o(f"  _mm{w}_store_pd(&sp_re[{5+n}*{m}], {I.mul(I.sub(f'v{n}_re','or_'),'inv10')});")
            em.o(f"  _mm{w}_store_pd(&sp_im[{5+n}*{m}], {I.mul(I.sub(f'v{n}_im','oi'),'inv10')});")
        em.o(f"}}")
    em.b()

    # ── 6. Unpermute + DC fixup ──
    # c[q] is now in spill[q] for q=0..9
    # X[g^{-q} mod 11] = x[0] + c[q]
    em.c(f"6. Unpermute: X[g^{{-q}}] = x[0] + c[q]")
    em.c(f"   iperm = {iperm}")
    for q in range(10):
        idx = iperm[q]
        em.reload("v0", q)
        em.o(f"{I.prefix}_ST(&out_re[{idx}*K+k], {I.add('x0_re','v0_re')});")
        em.o(f"{I.prefix}_ST(&out_im[{idx}*K+k], {I.add('x0_im','v0_im')});")
    em.b()

    em.c("7. DC output")
    em.o(f"{I.prefix}_ST(&out_re[0*K+k], dc_re);")
    em.o(f"{I.prefix}_ST(&out_im[0*K+k], dc_im);")
    em.b()

    em.ind -= 1
    em.o("}")
    em.L.append("}")
    em.L.append("")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ISAS:
        print(f"Usage: {sys.argv[0]} <scalar|avx2>", file=sys.stderr)
        sys.exit(1)

    isa_name = sys.argv[1]
    I = ISAS[isa_name]
    em = E(I)
    guard = f"FFT_RADIX11_{I.name.upper()}_N1_GEN_H"

    em.L.append(f"/**")
    em.L.append(f" * @file fft_radix11_{I.name}_n1_gen.h")
    em.L.append(f" * @brief GENERATED DFT-11 {I.name.upper()} N1 (Rader + Winograd DFT-5)")
    em.L.append(f" *")
    em.L.append(f" * Rader g=2, DFT-10 = 2×5 Cooley-Tukey, Winograd DFT-5 (5 muls/comp)")
    em.L.append(f" * k-step={I.k_step}")
    em.L.append(f" * Generated by gen_radix11_n1.py")
    em.L.append(f" */")
    em.L.append(f"")
    em.L.append(f"#ifndef {guard}")
    em.L.append(f"#define {guard}")
    em.L.append(f"")
    if I.name in ('avx2', 'avx512'):
        em.L.append(f"#include <immintrin.h>")
        em.L.append(f"")

    # LD/ST macros
    if I.name == 'scalar':
        em.L.append(f"#ifndef {I.prefix}_LD")
        em.L.append(f"#define {I.prefix}_LD(p) (*(p))")
        em.L.append(f"#endif")
        em.L.append(f"#ifndef {I.prefix}_ST")
        em.L.append(f"#define {I.prefix}_ST(p,v) (*(p)=(v))")
        em.L.append(f"#endif")
    else:
        em.L.append(f"#ifndef {I.prefix}_LD")
        w = '256' if I.k_step==4 else '512'
        em.L.append(f"#define {I.prefix}_LD(p) _mm{w}_load_pd(p)")
        em.L.append(f"#endif")
        em.L.append(f"#ifndef {I.prefix}_ST")
        em.L.append(f"#define {I.prefix}_ST(p,v) _mm{w}_store_pd((p),(v))")
        em.L.append(f"#endif")
    em.L.append(f"")

    if I.name == 'avx512':
        emit_kernel_spillfree(em, 'fwd', B_FWD)
        emit_kernel_spillfree(em, 'bwd', B_BWD)
    else:
        emit_kernel(em, 'fwd', B_FWD)
        emit_kernel(em, 'bwd', B_BWD)

    em.L.append(f"#endif /* {guard} */")

    print("\n".join(em.L))
    print(f"\n=== DFT-11 {I.name.upper()} N1 ===", file=sys.stderr)
    print(f"  Lines: {len(em.L)}", file=sys.stderr)

if __name__ == '__main__':
    main()
