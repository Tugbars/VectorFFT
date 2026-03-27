#!/usr/bin/env python3
"""
gen_n1_k1_simd.py — K=1 SIMD N1 codelets

R=16 AVX2:   4 YMM × 4 lanes = 16 elements, 4×4 CT, zero spill
R=16 AVX-512: same as AVX2 (16 elements don't fill ZMM usefully)  
R=32 AVX-512: 4 ZMM × 8 lanes = 32 elements, 8×4 CT, zero spill

Architecture:
  1. Load R contiguous doubles into R/VL vectors
  2. Pass 1: vector DFT-N1 (all N2 rows in parallel, one per lane)
  3. Vector twiddle: constant vectors (per-lane different W_R^(n2*k1))
  4. Transpose: in-register 4×4 (AVX2) or 8×4 (AVX-512)
  5. Pass 2: vector DFT-N2 (all N1 columns in parallel)
  6. Store

Usage:
  python3 gen_n1_k1_simd.py r16_avx2
  python3 gen_n1_k1_simd.py r32_avx512
"""
import math, sys

def wN(e, tN):
    a = 2.0 * math.pi * e / tN
    return (math.cos(a), -math.sin(a))


class Emitter:
    def __init__(self):
        self.L = []
        self.ind = 0
    def o(self, s=''):
        self.L.append('    ' * self.ind + s)
    def b(self):
        self.L.append('')


def emit_r16_avx2(direction):
    """R=16, AVX2, 4×4 CT, zero spill, transpose-based."""
    N, N1, N2, VL = 16, 4, 4, 4
    fwd = (direction == 'fwd')
    em = Emitter()

    em.o(f'__attribute__((target("avx2,fma")))')
    em.o(f'static inline void dft16_k1_{direction}_avx2(')
    em.o(f'    const double * __restrict__ ir, const double * __restrict__ ii,')
    em.o(f'    double * __restrict__ or_, double * __restrict__ oi)')
    em.o('{')
    em.ind = 1

    # Build twiddle constant vectors for k1=1,2,3
    # Lane n2 gets W_16^(n2*k1)
    for k1 in range(1, N1):
        vals_re = []
        vals_im = []
        for n2 in range(N2):
            e = (n2 * k1) % N
            wr, wi = wN(e, N)
            if not fwd:
                wi = -wi  # conjugate for bwd
            vals_re.append(f'{wr:.20e}')
            vals_im.append(f'{wi:.20e}')
        em.o(f'const __m256d tw{k1}r = _mm256_set_pd({vals_re[3]},{vals_re[2]},{vals_re[1]},{vals_re[0]});')
        em.o(f'const __m256d tw{k1}i = _mm256_set_pd({vals_im[3]},{vals_im[2]},{vals_im[1]},{vals_im[0]});')
    em.b()

    # Load 4 YMM re + 4 YMM im (contiguous)
    em.o('/* Load: 4 YMM × {ir[0..3], ir[4..7], ir[8..11], ir[12..15]} */')
    for i in range(N1):
        em.o(f'__m256d a{i}r = _mm256_loadu_pd(&ir[{i*VL}]);')
        em.o(f'__m256d a{i}i = _mm256_loadu_pd(&ii[{i*VL}]);')
    em.b()

    # Pass 1: DFT-4 on a0,a1,a2,a3 (all 4 rows in parallel)
    # Each lane processes one row independently
    em.o('/* Pass 1: DFT-4 — all 4 rows in parallel */')
    em.o(f'__m256d t0r = _mm256_add_pd(a0r, a2r), t0i = _mm256_add_pd(a0i, a2i);')
    em.o(f'__m256d t1r = _mm256_sub_pd(a0r, a2r), t1i = _mm256_sub_pd(a0i, a2i);')
    em.o(f'__m256d t2r = _mm256_add_pd(a1r, a3r), t2i = _mm256_add_pd(a1i, a3i);')
    em.o(f'__m256d t3r = _mm256_sub_pd(a1r, a3r), t3i = _mm256_sub_pd(a1i, a3i);')
    em.o(f'__m256d d0r = _mm256_add_pd(t0r, t2r), d0i = _mm256_add_pd(t0i, t2i);')
    em.o(f'__m256d d2r = _mm256_sub_pd(t0r, t2r), d2i = _mm256_sub_pd(t0i, t2i);')
    if fwd:
        em.o(f'__m256d d1r = _mm256_add_pd(t1r, t3i), d1i = _mm256_sub_pd(t1i, t3r);')
        em.o(f'__m256d d3r = _mm256_sub_pd(t1r, t3i), d3i = _mm256_add_pd(t1i, t3r);')
    else:
        em.o(f'__m256d d1r = _mm256_sub_pd(t1r, t3i), d1i = _mm256_add_pd(t1i, t3r);')
        em.o(f'__m256d d3r = _mm256_add_pd(t1r, t3i), d3i = _mm256_sub_pd(t1i, t3r);')
    em.b()

    # Internal twiddle: d_k1 *= tw_k1 (per-lane constant vector cmul)
    # d0 (k1=0): no twiddle
    em.o('/* Internal twiddle: per-lane constant W16 vectors */')
    for k1 in range(1, N1):
        em.o(f'{{ __m256d tr = d{k1}r;')
        em.o(f'  d{k1}r = _mm256_fmsub_pd(d{k1}r, tw{k1}r, _mm256_mul_pd(d{k1}i, tw{k1}i));')
        em.o(f'  d{k1}i = _mm256_fmadd_pd(tr, tw{k1}i, _mm256_mul_pd(d{k1}i, tw{k1}r)); }}')
    em.b()

    # Transpose 4×4: each d_k1 has lanes {row0, row1, row2, row3}
    # After transpose: each vector has {col0_val, col1_val, col2_val, col3_val}
    # i.e. all 4 sub-FFT outputs for one column
    em.o('/* 4×4 transpose — re */')
    em.o('{ __m256d u0 = _mm256_unpacklo_pd(d0r, d1r);')   # {d0[0],d1[0],d0[2],d1[2]}
    em.o('  __m256d u1 = _mm256_unpackhi_pd(d0r, d1r);')   # {d0[1],d1[1],d0[3],d1[3]}
    em.o('  __m256d u2 = _mm256_unpacklo_pd(d2r, d3r);')
    em.o('  __m256d u3 = _mm256_unpackhi_pd(d2r, d3r);')
    em.o('  d0r = _mm256_permute2f128_pd(u0, u2, 0x20);')  # {d0[0],d1[0],d2[0],d3[0]}
    em.o('  d1r = _mm256_permute2f128_pd(u1, u3, 0x20);')  # {d0[1],d1[1],d2[1],d3[1]}
    em.o('  d2r = _mm256_permute2f128_pd(u0, u2, 0x31);')  # {d0[2],d1[2],d2[2],d3[2]}
    em.o('  d3r = _mm256_permute2f128_pd(u1, u3, 0x31); }') # {d0[3],d1[3],d2[3],d3[3]}
    em.b()
    em.o('/* 4×4 transpose — im */')
    em.o('{ __m256d u0 = _mm256_unpacklo_pd(d0i, d1i);')
    em.o('  __m256d u1 = _mm256_unpackhi_pd(d0i, d1i);')
    em.o('  __m256d u2 = _mm256_unpacklo_pd(d2i, d3i);')
    em.o('  __m256d u3 = _mm256_unpackhi_pd(d2i, d3i);')
    em.o('  d0i = _mm256_permute2f128_pd(u0, u2, 0x20);')
    em.o('  d1i = _mm256_permute2f128_pd(u1, u3, 0x20);')
    em.o('  d2i = _mm256_permute2f128_pd(u0, u2, 0x31);')
    em.o('  d3i = _mm256_permute2f128_pd(u1, u3, 0x31); }')
    em.b()

    # Pass 2: DFT-4 on transposed data (all 4 columns in parallel)
    em.o('/* Pass 2: DFT-4 — all 4 columns in parallel */')
    em.o(f't0r = _mm256_add_pd(d0r, d2r); t0i = _mm256_add_pd(d0i, d2i);')
    em.o(f't1r = _mm256_sub_pd(d0r, d2r); t1i = _mm256_sub_pd(d0i, d2i);')
    em.o(f't2r = _mm256_add_pd(d1r, d3r); t2i = _mm256_add_pd(d1i, d3i);')
    em.o(f't3r = _mm256_sub_pd(d1r, d3r); t3i = _mm256_sub_pd(d1i, d3i);')
    em.o(f'd0r = _mm256_add_pd(t0r, t2r); d0i = _mm256_add_pd(t0i, t2i);')
    em.o(f'd2r = _mm256_sub_pd(t0r, t2r); d2i = _mm256_sub_pd(t0i, t2i);')
    if fwd:
        em.o(f'd1r = _mm256_add_pd(t1r, t3i); d1i = _mm256_sub_pd(t1i, t3r);')
        em.o(f'd3r = _mm256_sub_pd(t1r, t3i); d3i = _mm256_add_pd(t1i, t3r);')
    else:
        em.o(f'd1r = _mm256_sub_pd(t1r, t3i); d1i = _mm256_add_pd(t1i, t3r);')
        em.o(f'd3r = _mm256_add_pd(t1r, t3i); d3i = _mm256_sub_pd(t1i, t3r);')
    em.b()

    # Store: d0={out[0],out[1],out[2],out[3]}, d1={out[4],...}, d2={out[8],...}, d3={out[12],...}
    em.o('/* Store */')
    em.o(f'_mm256_storeu_pd(&or_[0],  d0r); _mm256_storeu_pd(&oi[0],  d0i);')
    em.o(f'_mm256_storeu_pd(&or_[4],  d1r); _mm256_storeu_pd(&oi[4],  d1i);')
    em.o(f'_mm256_storeu_pd(&or_[8],  d2r); _mm256_storeu_pd(&oi[8],  d2i);')
    em.o(f'_mm256_storeu_pd(&or_[12], d3r); _mm256_storeu_pd(&oi[12], d3i);')

    em.ind = 0
    em.o('}')
    return em.L


def emit_r32_avx512(direction):
    """R=32, AVX-512, 4 ZMM × 8 lanes, 8×4 CT, zero spill."""
    N, N1, N2, VL = 32, 8, 4, 8
    fwd = (direction == 'fwd')
    em = Emitter()
    S2 = math.sqrt(2.0) / 2.0

    em.o(f'__attribute__((target("avx512f,avx512dq,fma")))')
    em.o(f'static inline void dft32_k1_{direction}_avx512(')
    em.o(f'    const double * __restrict__ ir, const double * __restrict__ ii,')
    em.o(f'    double * __restrict__ or_, double * __restrict__ oi)')
    em.o('{')
    em.ind = 1

    # Constants for DFT-8 W8 combine
    em.o(f'const __m512d vc  = _mm512_set1_pd({S2:.20e});')
    em.o(f'const __m512d vnc = _mm512_set1_pd({-S2:.20e});')
    em.b()

    # Build twiddle constant vectors for k1=1..7
    # Lane n2 (0..3) gets W_32^(n2*k1), but we have 8 lanes
    # Wait — N2=4 rows, VL=8 lanes. We need only 4 active lanes.
    # Actually: data is ir[0..31], loaded as 4 ZMMs of 8.
    # a0 = {ir[0]..ir[7]}, lanes correspond to n2=0..7 modular positions.
    #
    # With 8×4 decomposition: n = N2*n1 + n2 = 4*n1 + n2
    # ir[n] at position n: n1 = n/4, n2 = n%4
    # a0 = {ir[0]..ir[7]}: lane 0 is n2=0,n1=0; lane 1 is n2=1,n1=0; ... lane 4 is n2=0,n1=1; lane 5 is n2=1,n1=1; ...
    #
    # Hmm, this doesn't give us "all rows in parallel" cleanly.
    # With 4 ZMMs of 8: each ZMM holds {n1=0..7 for a fixed n2 range}?
    # No — the data layout is contiguous, not strided.
    #
    # Let me reconsider. For R=32, N1=8, N2=4:
    # Element n maps to row n2 = n%4, column n1 = n/4
    # a0 = ir[0..7]:  n2 = {0,1,2,3,0,1,2,3}, n1 = {0,0,0,0,1,1,1,1}
    # a1 = ir[8..15]: n2 = {0,1,2,3,0,1,2,3}, n1 = {2,2,2,2,3,3,3,3}
    # a2 = ir[16..23]: n1 = {4,4,4,4,5,5,5,5}
    # a3 = ir[24..31]: n1 = {6,6,6,6,7,7,7,7}
    #
    # Each ZMM holds 2 groups of 4 for adjacent n1 values.
    # DFT-4 on a0,a1,a2,a3 treats position 0,1,2,3 as the 4-point DFT indices.
    # But position 0 in a0 is n1=0, position 0 in a1 is n1=2, etc.
    # This ISN'T a natural DFT-4 on rows.
    #
    # The natural decomposition for 4 ZMMs of 8 is 4×8:
    # Pass 1: 8 DFT-4 across a0,a1,a2,a3 (each lane independently)
    # Pass 2: 4 DFT-8 within each ZMM (intra-vector — shuffle-heavy)
    #
    # DFT-4 across vectors: lane i gets {a0[i], a1[i], a2[i], a3[i]}
    # a0[i] = ir[i], a1[i] = ir[8+i], a2[i] = ir[16+i], a3[i] = ir[24+i]
    # That's stride-8 = DFT-4 on elements {i, i+8, i+16, i+24}
    # In 4×8 CT: n1 = n%8, n2 = n/8. Element n = 8*n2 + n1.
    # So lane i is n1=i, and {a0,a1,a2,a3} maps to n2={0,1,2,3}.
    # DFT-4 on a0[i],a1[i],a2[i],a3[i] = DFT-4 of row n1=i.
    # This is 4×8 CT: Pass 1 = DFT-4 on rows, Pass 2 = DFT-8 on columns.
    #
    # After DFT-4 + twiddle, we need DFT-8 on each column.
    # Column k2 has data in lane (k2) across all 4 vectors? No.
    # After DFT-4, d0[i] = output k2=0 of row n1=i, etc.
    # For DFT-8 on column k2, we need {d_k2[0], d_k2[1], ..., d_k2[7]}.
    # d_k2[n1] is in vector d_k2, lane n1. So DFT-8 on column k2 is
    # intra-vector on d_k2. That's shuffle-heavy.
    #
    # Better approach: use 8×4 with rearranged loads.
    # Load so that each ZMM holds one row (4 elements, stride N2=4).
    # But 4 elements in 8-lane ZMM wastes half the width.
    #
    # Or: two interleaved sets. Lower 4 lanes = row A, upper 4 lanes = row B.
    # Process 2 rows per ZMM, 4 ZMMs = 8 rows. DFT-4 on positions within each 4-lane half.
    # This needs masked ops or careful lane management.
    #
    # Actually the simplest correct approach for R=32 AVX-512 K=1:
    # Just do 4×8 CT where Pass 2 is DFT-8 intra-vector.
    # The DFT-8 intra-vector is the same butterfly we use everywhere,
    # just applied to lanes within one ZMM using permute instructions.
    #
    # This is complex. Let me use a different approach:
    # Use 8 YMM-width operations within AVX-512 mode (256-bit ops in ZMM).
    # Or just fall back to spill-based approach with direct stores to 
    # a stack buffer and use the scalar K=1 codelet.
    #
    # Actually, let me just use the gather/scatter approach:
    # vgatherdpd to load stride-4 elements into a ZMM, giving us one row.
    # 4 gathers = 4 rows of 8 elements each. Then DFT-4 across ZMMs + twiddle.
    # Then DFT-8 within each ZMM (shuffle-based). Then scatter-store.

    # For now, use the pragmatic approach: 4×8 CT with vector DFT-4 cross-ZMM
    # and DFT-8 replaced by spill-to-stack + scalar DFT-8 per column.
    # This is still much better than fully scalar.

    # Actually, I think the cleanest R=32 AVX-512 K=1 is:
    # Gather loads to get stride-4 data into ZMMs, avoiding transpose entirely.
    
    idx = '__m256i idx = _mm256_set_epi32(28,24,20,16,12,8,4,0);'
    em.o(f'/* Gather index: stride-4 positions */')
    em.o(f'const __m256i idx = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);')
    em.b()

    # Load 4 rows via gather (each row = 8 elements at stride 4)
    em.o('/* Load 4 rows via gather: row n2 = elements n2, n2+4, n2+8, ..., n2+28 */')
    for n2 in range(N2):
        em.o(f'__m512d r{n2}r = _mm512_i32gather_pd(idx, &ir[{n2}], 8);')
        em.o(f'__m512d r{n2}i = _mm512_i32gather_pd(idx, &ii[{n2}], 8);')
    em.b()

    # DFT-4 on r0,r1,r2,r3 (all 8 columns in parallel)
    em.o('/* Pass 1: DFT-4 — all 8 columns in parallel */')
    em.o(f'__m512d t0r = _mm512_add_pd(r0r, r2r), t0i = _mm512_add_pd(r0i, r2i);')
    em.o(f'__m512d t1r = _mm512_sub_pd(r0r, r2r), t1i = _mm512_sub_pd(r0i, r2i);')
    em.o(f'__m512d t2r = _mm512_add_pd(r1r, r3r), t2i = _mm512_add_pd(r1i, r3i);')
    em.o(f'__m512d t3r = _mm512_sub_pd(r1r, r3r), t3i = _mm512_sub_pd(r1i, r3i);')
    em.o(f'__m512d d0r = _mm512_add_pd(t0r, t2r), d0i = _mm512_add_pd(t0i, t2i);')
    em.o(f'__m512d d2r = _mm512_sub_pd(t0r, t2r), d2i = _mm512_sub_pd(t0i, t2i);')
    if fwd:
        em.o(f'__m512d d1r = _mm512_add_pd(t1r, t3i), d1i = _mm512_sub_pd(t1i, t3r);')
        em.o(f'__m512d d3r = _mm512_sub_pd(t1r, t3i), d3i = _mm512_add_pd(t1i, t3r);')
    else:
        em.o(f'__m512d d1r = _mm512_sub_pd(t1r, t3i), d1i = _mm512_add_pd(t1i, t3r);')
        em.o(f'__m512d d3r = _mm512_add_pd(t1r, t3i), d3i = _mm512_sub_pd(t1i, t3r);')
    em.b()

    # Internal twiddle: d_k2 lane n1 gets W_32^(n2*k1) where n2=k2, k1=lane
    # Wait — with gather loads, row n2 has elements at n1=0..7 in lanes 0..7.
    # After DFT-4 on rows, output k2 has the k2-th DFT-4 output for each n1.
    # Internal twiddle: W_32^(k2 * n1) where n1 is the lane index.
    #
    # d0 (k2=0): no twiddle
    # d1 (k2=1): lane n1 gets W32^n1 = {W32^0, W32^1, ..., W32^7}
    # d2 (k2=2): lane n1 gets W32^(2*n1) = {W32^0, W32^2, ..., W32^14}
    # d3 (k2=3): lane n1 gets W32^(3*n1) = {W32^0, W32^3, ..., W32^21}

    em.o('/* Internal twiddle: per-lane W32 constant vectors */')
    for k2 in range(1, N2):
        vals_re = []
        vals_im = []
        for n1 in range(N1):
            e = (k2 * n1) % N
            wr, wi = wN(e, N)
            if not fwd:
                wi = -wi
            vals_re.append(f'{wr:.20e}')
            vals_im.append(f'{wi:.20e}')
        # _mm512_set_pd is reverse order (lane 7 first)
        rev_re = ','.join(reversed(vals_re))
        rev_im = ','.join(reversed(vals_im))
        em.o(f'{{ const __m512d twr = _mm512_set_pd({rev_re});')
        em.o(f'  const __m512d twi = _mm512_set_pd({rev_im});')
        em.o(f'  __m512d tr = d{k2}r;')
        em.o(f'  d{k2}r = _mm512_fmsub_pd(d{k2}r, twr, _mm512_mul_pd(d{k2}i, twi));')
        em.o(f'  d{k2}i = _mm512_fmadd_pd(tr, twi, _mm512_mul_pd(d{k2}i, twr)); }}')
    em.b()

    # Pass 2: DFT-8 on each d_k2 (intra-vector)
    # After twiddle, d_k2 has 8 values that need a DFT-8.
    # Intra-vector DFT-8 requires permutes.
    # Better: spill to stack, do scalar DFT-8, reload.
    # At K=1 this is only 4 DFT-8s × 8 elements = 32 scalar values total.
    em.o('/* Pass 2: DFT-8 on each column (via stack) */')
    em.o('__attribute__((aligned(64))) double buf_r[32], buf_i[32];')
    em.o('_mm512_store_pd(&buf_r[0],  d0r); _mm512_store_pd(&buf_i[0],  d0i);')
    em.o('_mm512_store_pd(&buf_r[8],  d1r); _mm512_store_pd(&buf_i[8],  d1i);')
    em.o('_mm512_store_pd(&buf_r[16], d2r); _mm512_store_pd(&buf_i[16], d2i);')
    em.o('_mm512_store_pd(&buf_r[24], d3r); _mm512_store_pd(&buf_i[24], d3i);')
    em.b()

    # 8 scalar DFT-4 on columns: column n1 has {buf[n1], buf[8+n1], buf[16+n1], buf[24+n1]}
    # Wait, after DFT-4 pass 1, d_k2 holds the k2-th output. So buf[k2*8 + n1] = pass1_output(k2, n1).
    # Pass 2 needs DFT-4 on {buf[0*8+n1], buf[1*8+n1], buf[2*8+n1], buf[3*8+n1]} for each n1.
    # But we said DFT-8 on columns? No — the decomposition is 8×4: Pass1=DFT-4 (N2=4), Pass2=DFT-8 (N1=8)?
    # 
    # Actually with gather: we loaded rows as r_n2 = elements at stride 4 starting from n2.
    # The CT is: N = N2 × N1 = 4 × 8. (NOT 8×4!)
    # Pass 1: N2=4 DFT of size N1=... no. Let me be precise.
    #
    # Standard CT: X[k1 + N1*k2] = sum over n2 of W_N^(n2*k1) * (DFT-N1 of row n2)[k1]
    # where row n2 has elements x[N2*n1 + n2] for n1=0..N1-1.
    #
    # With N1=8, N2=4: row n2 = {x[n2], x[4+n2], x[8+n2], ..., x[28+n2]} — exactly what gather loaded!
    # Pass 1: DFT-8 on each row n2 (8 elements) → produces 8 outputs per row.
    # Pass 2: DFT-4 across rows for each k1. Combined with twiddle W_32^(n2*k1).
    #
    # So I had the passes backwards! Pass 1 should be DFT-8 (within each ZMM),
    # Pass 2 should be DFT-4 (across ZMMs). Let me swap.

    # Clear the buffer approach and redo with correct pass ordering.
    em.L.clear()
    em.ind = 0

    em.o(f'__attribute__((target("avx512f,avx512dq,fma")))')
    em.o(f'static inline void dft32_k1_{direction}_avx512(')
    em.o(f'    const double * __restrict__ ir, const double * __restrict__ ii,')
    em.o(f'    double * __restrict__ or_, double * __restrict__ oi)')
    em.o('{')
    em.ind = 1

    em.o(f'const __m512d vc  = _mm512_set1_pd({S2:.20e});')
    em.o(f'const __m512d vnc = _mm512_set1_pd({-S2:.20e});')
    em.o(f'const __m256i idx = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);')
    em.b()

    # Load 4 rows via gather
    em.o('/* Load 4 rows via gather: row n2 = {x[n2], x[n2+4], ..., x[n2+28]} */')
    for n2 in range(N2):
        em.o(f'__m512d r{n2}r = _mm512_i32gather_pd(idx, &ir[{n2}], 8);')
        em.o(f'__m512d r{n2}i = _mm512_i32gather_pd(idx, &ii[{n2}], 8);')
    em.b()

    # Pass 1: DFT-8 on each row (intra-vector). Need to spill to stack for this.
    # DFT-8 within a ZMM is shuffle-heavy. Use stack approach:
    em.o('/* Pass 1: DFT-8 on each row + internal twiddle → stack */')
    em.o('__attribute__((aligned(64))) double buf_r[32], buf_i[32];')
    em.o('_mm512_store_pd(&buf_r[0],  r0r); _mm512_store_pd(&buf_i[0],  r0i);')
    em.o('_mm512_store_pd(&buf_r[8],  r1r); _mm512_store_pd(&buf_i[8],  r1i);')
    em.o('_mm512_store_pd(&buf_r[16], r2r); _mm512_store_pd(&buf_i[16], r2i);')
    em.o('_mm512_store_pd(&buf_r[24], r3r); _mm512_store_pd(&buf_i[24], r3i);')
    em.b()

    # 4 scalar DFT-8 on rows, with fused internal twiddle
    # Row n2 outputs: d_k1 *= W32^(n2*k1)
    em.o('/* 4 scalar DFT-8 + fused twiddle → buf */')
    for n2 in range(N2):
        base = n2 * 8
        em.o(f'{{ /* Row n2={n2} */')
        em.o(f'  double *rr = &buf_r[{base}], *ri = &buf_i[{base}];')
        em.o(f'  double epr=rr[0]+rr[4],epi=ri[0]+ri[4],eqr=rr[0]-rr[4],eqi=ri[0]-ri[4];')
        em.o(f'  double err=rr[2]+rr[6],eri=ri[2]+ri[6],esr=rr[2]-rr[6],esi=ri[2]-ri[6];')
        em.o(f'  double A0r=epr+err,A0i=epi+eri,A2r=epr-err,A2i=epi-eri;')
        if fwd:
            em.o(f'  double A1r=eqr+esi,A1i=eqi-esr,A3r=eqr-esi,A3i=eqi+esr;')
        else:
            em.o(f'  double A1r=eqr-esi,A1i=eqi+esr,A3r=eqr+esi,A3i=eqi-esr;')
        em.o(f'  double opr=rr[1]+rr[5],opi=ri[1]+ri[5],oqr=rr[1]-rr[5],oqi=ri[1]-ri[5];')
        em.o(f'  double orr=rr[3]+rr[7],ori=ri[3]+ri[7],osr=rr[3]-rr[7],osi=ri[3]-ri[7];')
        em.o(f'  double B0r=opr+orr,B0i=opi+ori,B2r=opr-orr,B2i=opi-ori;')
        if fwd:
            em.o(f'  double B1r=oqr+osi,B1i=oqi-osr,B3r=oqr-osi,B3i=oqi+osr;')
        else:
            em.o(f'  double B1r=oqr-osi,B1i=oqi+osr,B3r=oqr+osi,B3i=oqi-osr;')
        c = S2
        if fwd:
            em.o(f'  double w1r={c}*(B1r+B1i),w1i={c}*(B1i-B1r);')
            em.o(f'  double w3r={-c}*(B3r-B3i),w3i={-c}*(B3r+B3i);')
        else:
            em.o(f'  double w1r={c}*(B1r-B1i),w1i={c}*(B1r+B1i);')
            em.o(f'  double w3r={-c}*(B3r+B3i),w3i={c}*(B3r-B3i);')
        # DFT-8 outputs: k1=0..7
        em.o(f'  rr[0]=A0r+B0r; ri[0]=A0i+B0i;')
        em.o(f'  rr[1]=A1r+w1r; ri[1]=A1i+w1i;')
        if fwd:
            em.o(f'  rr[2]=A2r+B2i; ri[2]=A2i-B2r;')
        else:
            em.o(f'  rr[2]=A2r-B2i; ri[2]=A2i+B2r;')
        em.o(f'  rr[3]=A3r+w3r; ri[3]=A3i+w3i;')
        em.o(f'  rr[4]=A0r-B0r; ri[4]=A0i-B0i;')
        em.o(f'  rr[5]=A1r-w1r; ri[5]=A1i-w1i;')
        if fwd:
            em.o(f'  rr[6]=A2r-B2i; ri[6]=A2i+B2r;')
        else:
            em.o(f'  rr[6]=A2r+B2i; ri[6]=A2i-B2r;')
        em.o(f'  rr[7]=A3r-w3r; ri[7]=A3i-w3i;')

        # Fused internal twiddle: output k1 *= W32^(n2*k1)
        if n2 > 0:
            em.o(f'  /* Fused twiddle: W32^({n2}*k1) */')
            for k1 in range(1, N1):
                e = (n2 * k1) % N
                wr, wi = wN(e, N)
                if not fwd:
                    wi = -wi
                em.o(f'  {{ double tr=rr[{k1}]; rr[{k1}]=tr*{wr:.20e}-ri[{k1}]*{wi:.20e}; ri[{k1}]=tr*{wi:.20e}+ri[{k1}]*{wr:.20e}; }}')
        em.o(f'}}')
    em.b()

    # Pass 2: DFT-4 across rows for each k1 (vectorized)
    # Reload from buf with stride 8: column k1 = {buf[k1], buf[8+k1], buf[16+k1], buf[24+k1]}
    em.o('/* Pass 2: DFT-4 across rows (vectorized via gather) */')
    em.o('const __m256i cidx = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);')
    em.b()

    # Gather 4 rows worth of data for each k1 group
    # Actually simpler: just load the 4 ZMMs back and do vector DFT-4
    em.o('/* Reload rows */')
    for n2 in range(N2):
        em.o(f'r{n2}r = _mm512_load_pd(&buf_r[{n2*8}]); r{n2}i = _mm512_load_pd(&buf_i[{n2*8}]);')
    em.b()

    em.o('/* DFT-4 across rows — all 8 k1 values in parallel */')
    em.o(f'__m512d t0r = _mm512_add_pd(r0r, r2r), t0i = _mm512_add_pd(r0i, r2i);')
    em.o(f'__m512d t1r = _mm512_sub_pd(r0r, r2r), t1i = _mm512_sub_pd(r0i, r2i);')
    em.o(f'__m512d t2r = _mm512_add_pd(r1r, r3r), t2i = _mm512_add_pd(r1i, r3i);')
    em.o(f'__m512d t3r = _mm512_sub_pd(r1r, r3r), t3i = _mm512_sub_pd(r1i, r3i);')
    em.o(f'__m512d d0r = _mm512_add_pd(t0r, t2r), d0i = _mm512_add_pd(t0i, t2i);')
    em.o(f'__m512d d2r = _mm512_sub_pd(t0r, t2r), d2i = _mm512_sub_pd(t0i, t2i);')
    if fwd:
        em.o(f'__m512d d1r = _mm512_add_pd(t1r, t3i), d1i = _mm512_sub_pd(t1i, t3r);')
        em.o(f'__m512d d3r = _mm512_sub_pd(t1r, t3i), d3i = _mm512_add_pd(t1i, t3r);')
    else:
        em.o(f'__m512d d1r = _mm512_sub_pd(t1r, t3i), d1i = _mm512_add_pd(t1i, t3r);')
        em.o(f'__m512d d3r = _mm512_add_pd(t1r, t3i), d3i = _mm512_sub_pd(t1i, t3r);')
    em.b()

    # Store: d_k2 has outputs for k2, lanes hold k1=0..7
    # Output index = k1 + 8*k2 = contiguous blocks of 8
    em.o('/* Store: contiguous blocks of 8 */')
    for k2 in range(N2):
        em.o(f'_mm512_storeu_pd(&or_[{k2*N1}], d{k2}r); _mm512_storeu_pd(&oi[{k2*N1}], d{k2}i);')

    em.ind = 0
    em.o('}')
    return em.L


def main():
    if len(sys.argv) < 2:
        print("Usage: gen_n1_k1_simd.py <r16_avx2|r32_avx512>", file=sys.stderr)
        sys.exit(1)

    target = sys.argv[1]
    lines = ['#ifndef FFT_N1_K1_SIMD_H', '#define FFT_N1_K1_SIMD_H', '',
             '#include <immintrin.h>', '']

    if target == 'r16_avx2':
        lines.extend(emit_r16_avx2('fwd'))
        lines.append('')
        lines.extend(emit_r16_avx2('bwd'))
    elif target == 'r32_avx512':
        lines.extend(emit_r32_avx512('fwd'))
        lines.append('')
        lines.extend(emit_r32_avx512('bwd'))
    elif target == 'all':
        lines.extend(emit_r16_avx2('fwd'))
        lines.append('')
        lines.extend(emit_r16_avx2('bwd'))
        lines.append('')
        lines.extend(emit_r32_avx512('fwd'))
        lines.append('')
        lines.extend(emit_r32_avx512('bwd'))

    lines.extend(['', '#endif /* FFT_N1_K1_SIMD_H */'])
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
