#ifndef FFT_N1_K1_SIMD_H
#define FFT_N1_K1_SIMD_H

#include <immintrin.h>

__attribute__((target("avx2,fma")))
static inline void dft16_k1_fwd_avx2(
    const double * __restrict__ ir, const double * __restrict__ ii,
    double * __restrict__ or_, double * __restrict__ oi)
{
    const __m256d tw1r = _mm256_set_pd(3.82683432365089837290e-01,7.07106781186547572737e-01,9.23879532511286738483e-01,1.00000000000000000000e+00);
    const __m256d tw1i = _mm256_set_pd(-9.23879532511286738483e-01,-7.07106781186547572737e-01,-3.82683432365089781779e-01,-0.00000000000000000000e+00);
    const __m256d tw2r = _mm256_set_pd(-7.07106781186547461715e-01,6.12323399573676603587e-17,7.07106781186547572737e-01,1.00000000000000000000e+00);
    const __m256d tw2i = _mm256_set_pd(-7.07106781186547572737e-01,-1.00000000000000000000e+00,-7.07106781186547572737e-01,-0.00000000000000000000e+00);
    const __m256d tw3r = _mm256_set_pd(-9.23879532511286849505e-01,-7.07106781186547461715e-01,3.82683432365089837290e-01,1.00000000000000000000e+00);
    const __m256d tw3i = _mm256_set_pd(3.82683432365089670757e-01,-7.07106781186547572737e-01,-9.23879532511286738483e-01,-0.00000000000000000000e+00);

    /* Load: 4 YMM × {ir[0..3], ir[4..7], ir[8..11], ir[12..15]} */
    __m256d a0r = _mm256_loadu_pd(&ir[0]);
    __m256d a0i = _mm256_loadu_pd(&ii[0]);
    __m256d a1r = _mm256_loadu_pd(&ir[4]);
    __m256d a1i = _mm256_loadu_pd(&ii[4]);
    __m256d a2r = _mm256_loadu_pd(&ir[8]);
    __m256d a2i = _mm256_loadu_pd(&ii[8]);
    __m256d a3r = _mm256_loadu_pd(&ir[12]);
    __m256d a3i = _mm256_loadu_pd(&ii[12]);

    /* Pass 1: DFT-4 — all 4 rows in parallel */
    __m256d t0r = _mm256_add_pd(a0r, a2r), t0i = _mm256_add_pd(a0i, a2i);
    __m256d t1r = _mm256_sub_pd(a0r, a2r), t1i = _mm256_sub_pd(a0i, a2i);
    __m256d t2r = _mm256_add_pd(a1r, a3r), t2i = _mm256_add_pd(a1i, a3i);
    __m256d t3r = _mm256_sub_pd(a1r, a3r), t3i = _mm256_sub_pd(a1i, a3i);
    __m256d d0r = _mm256_add_pd(t0r, t2r), d0i = _mm256_add_pd(t0i, t2i);
    __m256d d2r = _mm256_sub_pd(t0r, t2r), d2i = _mm256_sub_pd(t0i, t2i);
    __m256d d1r = _mm256_add_pd(t1r, t3i), d1i = _mm256_sub_pd(t1i, t3r);
    __m256d d3r = _mm256_sub_pd(t1r, t3i), d3i = _mm256_add_pd(t1i, t3r);

    /* Internal twiddle: per-lane constant W16 vectors */
    { __m256d tr = d1r;
      d1r = _mm256_fmsub_pd(d1r, tw1r, _mm256_mul_pd(d1i, tw1i));
      d1i = _mm256_fmadd_pd(tr, tw1i, _mm256_mul_pd(d1i, tw1r)); }
    { __m256d tr = d2r;
      d2r = _mm256_fmsub_pd(d2r, tw2r, _mm256_mul_pd(d2i, tw2i));
      d2i = _mm256_fmadd_pd(tr, tw2i, _mm256_mul_pd(d2i, tw2r)); }
    { __m256d tr = d3r;
      d3r = _mm256_fmsub_pd(d3r, tw3r, _mm256_mul_pd(d3i, tw3i));
      d3i = _mm256_fmadd_pd(tr, tw3i, _mm256_mul_pd(d3i, tw3r)); }

    /* 4×4 transpose — re */
    { __m256d u0 = _mm256_unpacklo_pd(d0r, d1r);
      __m256d u1 = _mm256_unpackhi_pd(d0r, d1r);
      __m256d u2 = _mm256_unpacklo_pd(d2r, d3r);
      __m256d u3 = _mm256_unpackhi_pd(d2r, d3r);
      d0r = _mm256_permute2f128_pd(u0, u2, 0x20);
      d1r = _mm256_permute2f128_pd(u1, u3, 0x20);
      d2r = _mm256_permute2f128_pd(u0, u2, 0x31);
      d3r = _mm256_permute2f128_pd(u1, u3, 0x31); }

    /* 4×4 transpose — im */
    { __m256d u0 = _mm256_unpacklo_pd(d0i, d1i);
      __m256d u1 = _mm256_unpackhi_pd(d0i, d1i);
      __m256d u2 = _mm256_unpacklo_pd(d2i, d3i);
      __m256d u3 = _mm256_unpackhi_pd(d2i, d3i);
      d0i = _mm256_permute2f128_pd(u0, u2, 0x20);
      d1i = _mm256_permute2f128_pd(u1, u3, 0x20);
      d2i = _mm256_permute2f128_pd(u0, u2, 0x31);
      d3i = _mm256_permute2f128_pd(u1, u3, 0x31); }

    /* Pass 2: DFT-4 — all 4 columns in parallel */
    t0r = _mm256_add_pd(d0r, d2r); t0i = _mm256_add_pd(d0i, d2i);
    t1r = _mm256_sub_pd(d0r, d2r); t1i = _mm256_sub_pd(d0i, d2i);
    t2r = _mm256_add_pd(d1r, d3r); t2i = _mm256_add_pd(d1i, d3i);
    t3r = _mm256_sub_pd(d1r, d3r); t3i = _mm256_sub_pd(d1i, d3i);
    d0r = _mm256_add_pd(t0r, t2r); d0i = _mm256_add_pd(t0i, t2i);
    d2r = _mm256_sub_pd(t0r, t2r); d2i = _mm256_sub_pd(t0i, t2i);
    d1r = _mm256_add_pd(t1r, t3i); d1i = _mm256_sub_pd(t1i, t3r);
    d3r = _mm256_sub_pd(t1r, t3i); d3i = _mm256_add_pd(t1i, t3r);

    /* Store */
    _mm256_storeu_pd(&or_[0],  d0r); _mm256_storeu_pd(&oi[0],  d0i);
    _mm256_storeu_pd(&or_[4],  d1r); _mm256_storeu_pd(&oi[4],  d1i);
    _mm256_storeu_pd(&or_[8],  d2r); _mm256_storeu_pd(&oi[8],  d2i);
    _mm256_storeu_pd(&or_[12], d3r); _mm256_storeu_pd(&oi[12], d3i);
}

__attribute__((target("avx2,fma")))
static inline void dft16_k1_bwd_avx2(
    const double * __restrict__ ir, const double * __restrict__ ii,
    double * __restrict__ or_, double * __restrict__ oi)
{
    const __m256d tw1r = _mm256_set_pd(3.82683432365089837290e-01,7.07106781186547572737e-01,9.23879532511286738483e-01,1.00000000000000000000e+00);
    const __m256d tw1i = _mm256_set_pd(9.23879532511286738483e-01,7.07106781186547572737e-01,3.82683432365089781779e-01,0.00000000000000000000e+00);
    const __m256d tw2r = _mm256_set_pd(-7.07106781186547461715e-01,6.12323399573676603587e-17,7.07106781186547572737e-01,1.00000000000000000000e+00);
    const __m256d tw2i = _mm256_set_pd(7.07106781186547572737e-01,1.00000000000000000000e+00,7.07106781186547572737e-01,0.00000000000000000000e+00);
    const __m256d tw3r = _mm256_set_pd(-9.23879532511286849505e-01,-7.07106781186547461715e-01,3.82683432365089837290e-01,1.00000000000000000000e+00);
    const __m256d tw3i = _mm256_set_pd(-3.82683432365089670757e-01,7.07106781186547572737e-01,9.23879532511286738483e-01,0.00000000000000000000e+00);

    /* Load: 4 YMM × {ir[0..3], ir[4..7], ir[8..11], ir[12..15]} */
    __m256d a0r = _mm256_loadu_pd(&ir[0]);
    __m256d a0i = _mm256_loadu_pd(&ii[0]);
    __m256d a1r = _mm256_loadu_pd(&ir[4]);
    __m256d a1i = _mm256_loadu_pd(&ii[4]);
    __m256d a2r = _mm256_loadu_pd(&ir[8]);
    __m256d a2i = _mm256_loadu_pd(&ii[8]);
    __m256d a3r = _mm256_loadu_pd(&ir[12]);
    __m256d a3i = _mm256_loadu_pd(&ii[12]);

    /* Pass 1: DFT-4 — all 4 rows in parallel */
    __m256d t0r = _mm256_add_pd(a0r, a2r), t0i = _mm256_add_pd(a0i, a2i);
    __m256d t1r = _mm256_sub_pd(a0r, a2r), t1i = _mm256_sub_pd(a0i, a2i);
    __m256d t2r = _mm256_add_pd(a1r, a3r), t2i = _mm256_add_pd(a1i, a3i);
    __m256d t3r = _mm256_sub_pd(a1r, a3r), t3i = _mm256_sub_pd(a1i, a3i);
    __m256d d0r = _mm256_add_pd(t0r, t2r), d0i = _mm256_add_pd(t0i, t2i);
    __m256d d2r = _mm256_sub_pd(t0r, t2r), d2i = _mm256_sub_pd(t0i, t2i);
    __m256d d1r = _mm256_sub_pd(t1r, t3i), d1i = _mm256_add_pd(t1i, t3r);
    __m256d d3r = _mm256_add_pd(t1r, t3i), d3i = _mm256_sub_pd(t1i, t3r);

    /* Internal twiddle: per-lane constant W16 vectors */
    { __m256d tr = d1r;
      d1r = _mm256_fmsub_pd(d1r, tw1r, _mm256_mul_pd(d1i, tw1i));
      d1i = _mm256_fmadd_pd(tr, tw1i, _mm256_mul_pd(d1i, tw1r)); }
    { __m256d tr = d2r;
      d2r = _mm256_fmsub_pd(d2r, tw2r, _mm256_mul_pd(d2i, tw2i));
      d2i = _mm256_fmadd_pd(tr, tw2i, _mm256_mul_pd(d2i, tw2r)); }
    { __m256d tr = d3r;
      d3r = _mm256_fmsub_pd(d3r, tw3r, _mm256_mul_pd(d3i, tw3i));
      d3i = _mm256_fmadd_pd(tr, tw3i, _mm256_mul_pd(d3i, tw3r)); }

    /* 4×4 transpose — re */
    { __m256d u0 = _mm256_unpacklo_pd(d0r, d1r);
      __m256d u1 = _mm256_unpackhi_pd(d0r, d1r);
      __m256d u2 = _mm256_unpacklo_pd(d2r, d3r);
      __m256d u3 = _mm256_unpackhi_pd(d2r, d3r);
      d0r = _mm256_permute2f128_pd(u0, u2, 0x20);
      d1r = _mm256_permute2f128_pd(u1, u3, 0x20);
      d2r = _mm256_permute2f128_pd(u0, u2, 0x31);
      d3r = _mm256_permute2f128_pd(u1, u3, 0x31); }

    /* 4×4 transpose — im */
    { __m256d u0 = _mm256_unpacklo_pd(d0i, d1i);
      __m256d u1 = _mm256_unpackhi_pd(d0i, d1i);
      __m256d u2 = _mm256_unpacklo_pd(d2i, d3i);
      __m256d u3 = _mm256_unpackhi_pd(d2i, d3i);
      d0i = _mm256_permute2f128_pd(u0, u2, 0x20);
      d1i = _mm256_permute2f128_pd(u1, u3, 0x20);
      d2i = _mm256_permute2f128_pd(u0, u2, 0x31);
      d3i = _mm256_permute2f128_pd(u1, u3, 0x31); }

    /* Pass 2: DFT-4 — all 4 columns in parallel */
    t0r = _mm256_add_pd(d0r, d2r); t0i = _mm256_add_pd(d0i, d2i);
    t1r = _mm256_sub_pd(d0r, d2r); t1i = _mm256_sub_pd(d0i, d2i);
    t2r = _mm256_add_pd(d1r, d3r); t2i = _mm256_add_pd(d1i, d3i);
    t3r = _mm256_sub_pd(d1r, d3r); t3i = _mm256_sub_pd(d1i, d3i);
    d0r = _mm256_add_pd(t0r, t2r); d0i = _mm256_add_pd(t0i, t2i);
    d2r = _mm256_sub_pd(t0r, t2r); d2i = _mm256_sub_pd(t0i, t2i);
    d1r = _mm256_sub_pd(t1r, t3i); d1i = _mm256_add_pd(t1i, t3r);
    d3r = _mm256_add_pd(t1r, t3i); d3i = _mm256_sub_pd(t1i, t3r);

    /* Store */
    _mm256_storeu_pd(&or_[0],  d0r); _mm256_storeu_pd(&oi[0],  d0i);
    _mm256_storeu_pd(&or_[4],  d1r); _mm256_storeu_pd(&oi[4],  d1i);
    _mm256_storeu_pd(&or_[8],  d2r); _mm256_storeu_pd(&oi[8],  d2i);
    _mm256_storeu_pd(&or_[12], d3r); _mm256_storeu_pd(&oi[12], d3i);
}

__attribute__((target("avx512f,avx512dq,fma")))
static inline void dft32_k1_fwd_avx512(
    const double * __restrict__ ir, const double * __restrict__ ii,
    double * __restrict__ or_, double * __restrict__ oi)
{
    const __m512d vc  = _mm512_set1_pd(7.07106781186547572737e-01);
    const __m512d vnc = _mm512_set1_pd(-7.07106781186547572737e-01);
    const __m256i idx = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);

    /* Load 4 rows via gather: row n2 = {x[n2], x[n2+4], ..., x[n2+28]} */
    __m512d r0r = _mm512_i32gather_pd(idx, &ir[0], 8);
    __m512d r0i = _mm512_i32gather_pd(idx, &ii[0], 8);
    __m512d r1r = _mm512_i32gather_pd(idx, &ir[1], 8);
    __m512d r1i = _mm512_i32gather_pd(idx, &ii[1], 8);
    __m512d r2r = _mm512_i32gather_pd(idx, &ir[2], 8);
    __m512d r2i = _mm512_i32gather_pd(idx, &ii[2], 8);
    __m512d r3r = _mm512_i32gather_pd(idx, &ir[3], 8);
    __m512d r3i = _mm512_i32gather_pd(idx, &ii[3], 8);

    /* Pass 1: DFT-8 on each row + internal twiddle → stack */
    __attribute__((aligned(64))) double buf_r[32], buf_i[32];
    _mm512_store_pd(&buf_r[0],  r0r); _mm512_store_pd(&buf_i[0],  r0i);
    _mm512_store_pd(&buf_r[8],  r1r); _mm512_store_pd(&buf_i[8],  r1i);
    _mm512_store_pd(&buf_r[16], r2r); _mm512_store_pd(&buf_i[16], r2i);
    _mm512_store_pd(&buf_r[24], r3r); _mm512_store_pd(&buf_i[24], r3i);

    /* 4 scalar DFT-8 + fused twiddle → buf */
    { /* Row n2=0 */
      double *rr = &buf_r[0], *ri = &buf_i[0];
      double epr=rr[0]+rr[4],epi=ri[0]+ri[4],eqr=rr[0]-rr[4],eqi=ri[0]-ri[4];
      double err=rr[2]+rr[6],eri=ri[2]+ri[6],esr=rr[2]-rr[6],esi=ri[2]-ri[6];
      double A0r=epr+err,A0i=epi+eri,A2r=epr-err,A2i=epi-eri;
      double A1r=eqr+esi,A1i=eqi-esr,A3r=eqr-esi,A3i=eqi+esr;
      double opr=rr[1]+rr[5],opi=ri[1]+ri[5],oqr=rr[1]-rr[5],oqi=ri[1]-ri[5];
      double orr=rr[3]+rr[7],ori=ri[3]+ri[7],osr=rr[3]-rr[7],osi=ri[3]-ri[7];
      double B0r=opr+orr,B0i=opi+ori,B2r=opr-orr,B2i=opi-ori;
      double B1r=oqr+osi,B1i=oqi-osr,B3r=oqr-osi,B3i=oqi+osr;
      double w1r=0.7071067811865476*(B1r+B1i),w1i=0.7071067811865476*(B1i-B1r);
      double w3r=-0.7071067811865476*(B3r-B3i),w3i=-0.7071067811865476*(B3r+B3i);
      rr[0]=A0r+B0r; ri[0]=A0i+B0i;
      rr[1]=A1r+w1r; ri[1]=A1i+w1i;
      rr[2]=A2r+B2i; ri[2]=A2i-B2r;
      rr[3]=A3r+w3r; ri[3]=A3i+w3i;
      rr[4]=A0r-B0r; ri[4]=A0i-B0i;
      rr[5]=A1r-w1r; ri[5]=A1i-w1i;
      rr[6]=A2r-B2i; ri[6]=A2i+B2r;
      rr[7]=A3r-w3r; ri[7]=A3i-w3i;
    }
    { /* Row n2=1 */
      double *rr = &buf_r[8], *ri = &buf_i[8];
      double epr=rr[0]+rr[4],epi=ri[0]+ri[4],eqr=rr[0]-rr[4],eqi=ri[0]-ri[4];
      double err=rr[2]+rr[6],eri=ri[2]+ri[6],esr=rr[2]-rr[6],esi=ri[2]-ri[6];
      double A0r=epr+err,A0i=epi+eri,A2r=epr-err,A2i=epi-eri;
      double A1r=eqr+esi,A1i=eqi-esr,A3r=eqr-esi,A3i=eqi+esr;
      double opr=rr[1]+rr[5],opi=ri[1]+ri[5],oqr=rr[1]-rr[5],oqi=ri[1]-ri[5];
      double orr=rr[3]+rr[7],ori=ri[3]+ri[7],osr=rr[3]-rr[7],osi=ri[3]-ri[7];
      double B0r=opr+orr,B0i=opi+ori,B2r=opr-orr,B2i=opi-ori;
      double B1r=oqr+osi,B1i=oqi-osr,B3r=oqr-osi,B3i=oqi+osr;
      double w1r=0.7071067811865476*(B1r+B1i),w1i=0.7071067811865476*(B1i-B1r);
      double w3r=-0.7071067811865476*(B3r-B3i),w3i=-0.7071067811865476*(B3r+B3i);
      rr[0]=A0r+B0r; ri[0]=A0i+B0i;
      rr[1]=A1r+w1r; ri[1]=A1i+w1i;
      rr[2]=A2r+B2i; ri[2]=A2i-B2r;
      rr[3]=A3r+w3r; ri[3]=A3i+w3i;
      rr[4]=A0r-B0r; ri[4]=A0i-B0i;
      rr[5]=A1r-w1r; ri[5]=A1i-w1i;
      rr[6]=A2r-B2i; ri[6]=A2i+B2r;
      rr[7]=A3r-w3r; ri[7]=A3i-w3i;
      /* Fused twiddle: W32^(1*k1) */
      { double tr=rr[1]; rr[1]=tr*9.80785280403230430579e-01-ri[1]*-1.95090322016128248084e-01; ri[1]=tr*-1.95090322016128248084e-01+ri[1]*9.80785280403230430579e-01; }
      { double tr=rr[2]; rr[2]=tr*9.23879532511286738483e-01-ri[2]*-3.82683432365089781779e-01; ri[2]=tr*-3.82683432365089781779e-01+ri[2]*9.23879532511286738483e-01; }
      { double tr=rr[3]; rr[3]=tr*8.31469612302545235671e-01-ri[3]*-5.55570233019602177649e-01; ri[3]=tr*-5.55570233019602177649e-01+ri[3]*8.31469612302545235671e-01; }
      { double tr=rr[4]; rr[4]=tr*7.07106781186547572737e-01-ri[4]*-7.07106781186547572737e-01; ri[4]=tr*-7.07106781186547572737e-01+ri[4]*7.07106781186547572737e-01; }
      { double tr=rr[5]; rr[5]=tr*5.55570233019602288671e-01-ri[5]*-8.31469612302545235671e-01; ri[5]=tr*-8.31469612302545235671e-01+ri[5]*5.55570233019602288671e-01; }
      { double tr=rr[6]; rr[6]=tr*3.82683432365089837290e-01-ri[6]*-9.23879532511286738483e-01; ri[6]=tr*-9.23879532511286738483e-01+ri[6]*3.82683432365089837290e-01; }
      { double tr=rr[7]; rr[7]=tr*1.95090322016128331351e-01-ri[7]*-9.80785280403230430579e-01; ri[7]=tr*-9.80785280403230430579e-01+ri[7]*1.95090322016128331351e-01; }
    }
    { /* Row n2=2 */
      double *rr = &buf_r[16], *ri = &buf_i[16];
      double epr=rr[0]+rr[4],epi=ri[0]+ri[4],eqr=rr[0]-rr[4],eqi=ri[0]-ri[4];
      double err=rr[2]+rr[6],eri=ri[2]+ri[6],esr=rr[2]-rr[6],esi=ri[2]-ri[6];
      double A0r=epr+err,A0i=epi+eri,A2r=epr-err,A2i=epi-eri;
      double A1r=eqr+esi,A1i=eqi-esr,A3r=eqr-esi,A3i=eqi+esr;
      double opr=rr[1]+rr[5],opi=ri[1]+ri[5],oqr=rr[1]-rr[5],oqi=ri[1]-ri[5];
      double orr=rr[3]+rr[7],ori=ri[3]+ri[7],osr=rr[3]-rr[7],osi=ri[3]-ri[7];
      double B0r=opr+orr,B0i=opi+ori,B2r=opr-orr,B2i=opi-ori;
      double B1r=oqr+osi,B1i=oqi-osr,B3r=oqr-osi,B3i=oqi+osr;
      double w1r=0.7071067811865476*(B1r+B1i),w1i=0.7071067811865476*(B1i-B1r);
      double w3r=-0.7071067811865476*(B3r-B3i),w3i=-0.7071067811865476*(B3r+B3i);
      rr[0]=A0r+B0r; ri[0]=A0i+B0i;
      rr[1]=A1r+w1r; ri[1]=A1i+w1i;
      rr[2]=A2r+B2i; ri[2]=A2i-B2r;
      rr[3]=A3r+w3r; ri[3]=A3i+w3i;
      rr[4]=A0r-B0r; ri[4]=A0i-B0i;
      rr[5]=A1r-w1r; ri[5]=A1i-w1i;
      rr[6]=A2r-B2i; ri[6]=A2i+B2r;
      rr[7]=A3r-w3r; ri[7]=A3i-w3i;
      /* Fused twiddle: W32^(2*k1) */
      { double tr=rr[1]; rr[1]=tr*9.23879532511286738483e-01-ri[1]*-3.82683432365089781779e-01; ri[1]=tr*-3.82683432365089781779e-01+ri[1]*9.23879532511286738483e-01; }
      { double tr=rr[2]; rr[2]=tr*7.07106781186547572737e-01-ri[2]*-7.07106781186547572737e-01; ri[2]=tr*-7.07106781186547572737e-01+ri[2]*7.07106781186547572737e-01; }
      { double tr=rr[3]; rr[3]=tr*3.82683432365089837290e-01-ri[3]*-9.23879532511286738483e-01; ri[3]=tr*-9.23879532511286738483e-01+ri[3]*3.82683432365089837290e-01; }
      { double tr=rr[4]; rr[4]=tr*6.12323399573676603587e-17-ri[4]*-1.00000000000000000000e+00; ri[4]=tr*-1.00000000000000000000e+00+ri[4]*6.12323399573676603587e-17; }
      { double tr=rr[5]; rr[5]=tr*-3.82683432365089726268e-01-ri[5]*-9.23879532511286738483e-01; ri[5]=tr*-9.23879532511286738483e-01+ri[5]*-3.82683432365089726268e-01; }
      { double tr=rr[6]; rr[6]=tr*-7.07106781186547461715e-01-ri[6]*-7.07106781186547572737e-01; ri[6]=tr*-7.07106781186547572737e-01+ri[6]*-7.07106781186547461715e-01; }
      { double tr=rr[7]; rr[7]=tr*-9.23879532511286738483e-01-ri[7]*-3.82683432365089892802e-01; ri[7]=tr*-3.82683432365089892802e-01+ri[7]*-9.23879532511286738483e-01; }
    }
    { /* Row n2=3 */
      double *rr = &buf_r[24], *ri = &buf_i[24];
      double epr=rr[0]+rr[4],epi=ri[0]+ri[4],eqr=rr[0]-rr[4],eqi=ri[0]-ri[4];
      double err=rr[2]+rr[6],eri=ri[2]+ri[6],esr=rr[2]-rr[6],esi=ri[2]-ri[6];
      double A0r=epr+err,A0i=epi+eri,A2r=epr-err,A2i=epi-eri;
      double A1r=eqr+esi,A1i=eqi-esr,A3r=eqr-esi,A3i=eqi+esr;
      double opr=rr[1]+rr[5],opi=ri[1]+ri[5],oqr=rr[1]-rr[5],oqi=ri[1]-ri[5];
      double orr=rr[3]+rr[7],ori=ri[3]+ri[7],osr=rr[3]-rr[7],osi=ri[3]-ri[7];
      double B0r=opr+orr,B0i=opi+ori,B2r=opr-orr,B2i=opi-ori;
      double B1r=oqr+osi,B1i=oqi-osr,B3r=oqr-osi,B3i=oqi+osr;
      double w1r=0.7071067811865476*(B1r+B1i),w1i=0.7071067811865476*(B1i-B1r);
      double w3r=-0.7071067811865476*(B3r-B3i),w3i=-0.7071067811865476*(B3r+B3i);
      rr[0]=A0r+B0r; ri[0]=A0i+B0i;
      rr[1]=A1r+w1r; ri[1]=A1i+w1i;
      rr[2]=A2r+B2i; ri[2]=A2i-B2r;
      rr[3]=A3r+w3r; ri[3]=A3i+w3i;
      rr[4]=A0r-B0r; ri[4]=A0i-B0i;
      rr[5]=A1r-w1r; ri[5]=A1i-w1i;
      rr[6]=A2r-B2i; ri[6]=A2i+B2r;
      rr[7]=A3r-w3r; ri[7]=A3i-w3i;
      /* Fused twiddle: W32^(3*k1) */
      { double tr=rr[1]; rr[1]=tr*8.31469612302545235671e-01-ri[1]*-5.55570233019602177649e-01; ri[1]=tr*-5.55570233019602177649e-01+ri[1]*8.31469612302545235671e-01; }
      { double tr=rr[2]; rr[2]=tr*3.82683432365089837290e-01-ri[2]*-9.23879532511286738483e-01; ri[2]=tr*-9.23879532511286738483e-01+ri[2]*3.82683432365089837290e-01; }
      { double tr=rr[3]; rr[3]=tr*-1.95090322016128192573e-01-ri[3]*-9.80785280403230430579e-01; ri[3]=tr*-9.80785280403230430579e-01+ri[3]*-1.95090322016128192573e-01; }
      { double tr=rr[4]; rr[4]=tr*-7.07106781186547461715e-01-ri[4]*-7.07106781186547572737e-01; ri[4]=tr*-7.07106781186547572737e-01+ri[4]*-7.07106781186547461715e-01; }
      { double tr=rr[5]; rr[5]=tr*-9.80785280403230430579e-01-ri[5]*-1.95090322016128608906e-01; ri[5]=tr*-1.95090322016128608906e-01+ri[5]*-9.80785280403230430579e-01; }
      { double tr=rr[6]; rr[6]=tr*-9.23879532511286849505e-01-ri[6]*3.82683432365089670757e-01; ri[6]=tr*3.82683432365089670757e-01+ri[6]*-9.23879532511286849505e-01; }
      { double tr=rr[7]; rr[7]=tr*-5.55570233019602177649e-01-ri[7]*8.31469612302545235671e-01; ri[7]=tr*8.31469612302545235671e-01+ri[7]*-5.55570233019602177649e-01; }
    }

    /* Pass 2: DFT-4 across rows (vectorized via gather) */
    const __m256i cidx = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);

    /* Reload rows */
    r0r = _mm512_load_pd(&buf_r[0]); r0i = _mm512_load_pd(&buf_i[0]);
    r1r = _mm512_load_pd(&buf_r[8]); r1i = _mm512_load_pd(&buf_i[8]);
    r2r = _mm512_load_pd(&buf_r[16]); r2i = _mm512_load_pd(&buf_i[16]);
    r3r = _mm512_load_pd(&buf_r[24]); r3i = _mm512_load_pd(&buf_i[24]);

    /* DFT-4 across rows — all 8 k1 values in parallel */
    __m512d t0r = _mm512_add_pd(r0r, r2r), t0i = _mm512_add_pd(r0i, r2i);
    __m512d t1r = _mm512_sub_pd(r0r, r2r), t1i = _mm512_sub_pd(r0i, r2i);
    __m512d t2r = _mm512_add_pd(r1r, r3r), t2i = _mm512_add_pd(r1i, r3i);
    __m512d t3r = _mm512_sub_pd(r1r, r3r), t3i = _mm512_sub_pd(r1i, r3i);
    __m512d d0r = _mm512_add_pd(t0r, t2r), d0i = _mm512_add_pd(t0i, t2i);
    __m512d d2r = _mm512_sub_pd(t0r, t2r), d2i = _mm512_sub_pd(t0i, t2i);
    __m512d d1r = _mm512_add_pd(t1r, t3i), d1i = _mm512_sub_pd(t1i, t3r);
    __m512d d3r = _mm512_sub_pd(t1r, t3i), d3i = _mm512_add_pd(t1i, t3r);

    /* Store: contiguous blocks of 8 */
    _mm512_storeu_pd(&or_[0], d0r); _mm512_storeu_pd(&oi[0], d0i);
    _mm512_storeu_pd(&or_[8], d1r); _mm512_storeu_pd(&oi[8], d1i);
    _mm512_storeu_pd(&or_[16], d2r); _mm512_storeu_pd(&oi[16], d2i);
    _mm512_storeu_pd(&or_[24], d3r); _mm512_storeu_pd(&oi[24], d3i);
}

__attribute__((target("avx512f,avx512dq,fma")))
static inline void dft32_k1_bwd_avx512(
    const double * __restrict__ ir, const double * __restrict__ ii,
    double * __restrict__ or_, double * __restrict__ oi)
{
    const __m512d vc  = _mm512_set1_pd(7.07106781186547572737e-01);
    const __m512d vnc = _mm512_set1_pd(-7.07106781186547572737e-01);
    const __m256i idx = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);

    /* Load 4 rows via gather: row n2 = {x[n2], x[n2+4], ..., x[n2+28]} */
    __m512d r0r = _mm512_i32gather_pd(idx, &ir[0], 8);
    __m512d r0i = _mm512_i32gather_pd(idx, &ii[0], 8);
    __m512d r1r = _mm512_i32gather_pd(idx, &ir[1], 8);
    __m512d r1i = _mm512_i32gather_pd(idx, &ii[1], 8);
    __m512d r2r = _mm512_i32gather_pd(idx, &ir[2], 8);
    __m512d r2i = _mm512_i32gather_pd(idx, &ii[2], 8);
    __m512d r3r = _mm512_i32gather_pd(idx, &ir[3], 8);
    __m512d r3i = _mm512_i32gather_pd(idx, &ii[3], 8);

    /* Pass 1: DFT-8 on each row + internal twiddle → stack */
    __attribute__((aligned(64))) double buf_r[32], buf_i[32];
    _mm512_store_pd(&buf_r[0],  r0r); _mm512_store_pd(&buf_i[0],  r0i);
    _mm512_store_pd(&buf_r[8],  r1r); _mm512_store_pd(&buf_i[8],  r1i);
    _mm512_store_pd(&buf_r[16], r2r); _mm512_store_pd(&buf_i[16], r2i);
    _mm512_store_pd(&buf_r[24], r3r); _mm512_store_pd(&buf_i[24], r3i);

    /* 4 scalar DFT-8 + fused twiddle → buf */
    { /* Row n2=0 */
      double *rr = &buf_r[0], *ri = &buf_i[0];
      double epr=rr[0]+rr[4],epi=ri[0]+ri[4],eqr=rr[0]-rr[4],eqi=ri[0]-ri[4];
      double err=rr[2]+rr[6],eri=ri[2]+ri[6],esr=rr[2]-rr[6],esi=ri[2]-ri[6];
      double A0r=epr+err,A0i=epi+eri,A2r=epr-err,A2i=epi-eri;
      double A1r=eqr-esi,A1i=eqi+esr,A3r=eqr+esi,A3i=eqi-esr;
      double opr=rr[1]+rr[5],opi=ri[1]+ri[5],oqr=rr[1]-rr[5],oqi=ri[1]-ri[5];
      double orr=rr[3]+rr[7],ori=ri[3]+ri[7],osr=rr[3]-rr[7],osi=ri[3]-ri[7];
      double B0r=opr+orr,B0i=opi+ori,B2r=opr-orr,B2i=opi-ori;
      double B1r=oqr-osi,B1i=oqi+osr,B3r=oqr+osi,B3i=oqi-osr;
      double w1r=0.7071067811865476*(B1r-B1i),w1i=0.7071067811865476*(B1r+B1i);
      double w3r=-0.7071067811865476*(B3r+B3i),w3i=0.7071067811865476*(B3r-B3i);
      rr[0]=A0r+B0r; ri[0]=A0i+B0i;
      rr[1]=A1r+w1r; ri[1]=A1i+w1i;
      rr[2]=A2r-B2i; ri[2]=A2i+B2r;
      rr[3]=A3r+w3r; ri[3]=A3i+w3i;
      rr[4]=A0r-B0r; ri[4]=A0i-B0i;
      rr[5]=A1r-w1r; ri[5]=A1i-w1i;
      rr[6]=A2r+B2i; ri[6]=A2i-B2r;
      rr[7]=A3r-w3r; ri[7]=A3i-w3i;
    }
    { /* Row n2=1 */
      double *rr = &buf_r[8], *ri = &buf_i[8];
      double epr=rr[0]+rr[4],epi=ri[0]+ri[4],eqr=rr[0]-rr[4],eqi=ri[0]-ri[4];
      double err=rr[2]+rr[6],eri=ri[2]+ri[6],esr=rr[2]-rr[6],esi=ri[2]-ri[6];
      double A0r=epr+err,A0i=epi+eri,A2r=epr-err,A2i=epi-eri;
      double A1r=eqr-esi,A1i=eqi+esr,A3r=eqr+esi,A3i=eqi-esr;
      double opr=rr[1]+rr[5],opi=ri[1]+ri[5],oqr=rr[1]-rr[5],oqi=ri[1]-ri[5];
      double orr=rr[3]+rr[7],ori=ri[3]+ri[7],osr=rr[3]-rr[7],osi=ri[3]-ri[7];
      double B0r=opr+orr,B0i=opi+ori,B2r=opr-orr,B2i=opi-ori;
      double B1r=oqr-osi,B1i=oqi+osr,B3r=oqr+osi,B3i=oqi-osr;
      double w1r=0.7071067811865476*(B1r-B1i),w1i=0.7071067811865476*(B1r+B1i);
      double w3r=-0.7071067811865476*(B3r+B3i),w3i=0.7071067811865476*(B3r-B3i);
      rr[0]=A0r+B0r; ri[0]=A0i+B0i;
      rr[1]=A1r+w1r; ri[1]=A1i+w1i;
      rr[2]=A2r-B2i; ri[2]=A2i+B2r;
      rr[3]=A3r+w3r; ri[3]=A3i+w3i;
      rr[4]=A0r-B0r; ri[4]=A0i-B0i;
      rr[5]=A1r-w1r; ri[5]=A1i-w1i;
      rr[6]=A2r+B2i; ri[6]=A2i-B2r;
      rr[7]=A3r-w3r; ri[7]=A3i-w3i;
      /* Fused twiddle: W32^(1*k1) */
      { double tr=rr[1]; rr[1]=tr*9.80785280403230430579e-01-ri[1]*1.95090322016128248084e-01; ri[1]=tr*1.95090322016128248084e-01+ri[1]*9.80785280403230430579e-01; }
      { double tr=rr[2]; rr[2]=tr*9.23879532511286738483e-01-ri[2]*3.82683432365089781779e-01; ri[2]=tr*3.82683432365089781779e-01+ri[2]*9.23879532511286738483e-01; }
      { double tr=rr[3]; rr[3]=tr*8.31469612302545235671e-01-ri[3]*5.55570233019602177649e-01; ri[3]=tr*5.55570233019602177649e-01+ri[3]*8.31469612302545235671e-01; }
      { double tr=rr[4]; rr[4]=tr*7.07106781186547572737e-01-ri[4]*7.07106781186547572737e-01; ri[4]=tr*7.07106781186547572737e-01+ri[4]*7.07106781186547572737e-01; }
      { double tr=rr[5]; rr[5]=tr*5.55570233019602288671e-01-ri[5]*8.31469612302545235671e-01; ri[5]=tr*8.31469612302545235671e-01+ri[5]*5.55570233019602288671e-01; }
      { double tr=rr[6]; rr[6]=tr*3.82683432365089837290e-01-ri[6]*9.23879532511286738483e-01; ri[6]=tr*9.23879532511286738483e-01+ri[6]*3.82683432365089837290e-01; }
      { double tr=rr[7]; rr[7]=tr*1.95090322016128331351e-01-ri[7]*9.80785280403230430579e-01; ri[7]=tr*9.80785280403230430579e-01+ri[7]*1.95090322016128331351e-01; }
    }
    { /* Row n2=2 */
      double *rr = &buf_r[16], *ri = &buf_i[16];
      double epr=rr[0]+rr[4],epi=ri[0]+ri[4],eqr=rr[0]-rr[4],eqi=ri[0]-ri[4];
      double err=rr[2]+rr[6],eri=ri[2]+ri[6],esr=rr[2]-rr[6],esi=ri[2]-ri[6];
      double A0r=epr+err,A0i=epi+eri,A2r=epr-err,A2i=epi-eri;
      double A1r=eqr-esi,A1i=eqi+esr,A3r=eqr+esi,A3i=eqi-esr;
      double opr=rr[1]+rr[5],opi=ri[1]+ri[5],oqr=rr[1]-rr[5],oqi=ri[1]-ri[5];
      double orr=rr[3]+rr[7],ori=ri[3]+ri[7],osr=rr[3]-rr[7],osi=ri[3]-ri[7];
      double B0r=opr+orr,B0i=opi+ori,B2r=opr-orr,B2i=opi-ori;
      double B1r=oqr-osi,B1i=oqi+osr,B3r=oqr+osi,B3i=oqi-osr;
      double w1r=0.7071067811865476*(B1r-B1i),w1i=0.7071067811865476*(B1r+B1i);
      double w3r=-0.7071067811865476*(B3r+B3i),w3i=0.7071067811865476*(B3r-B3i);
      rr[0]=A0r+B0r; ri[0]=A0i+B0i;
      rr[1]=A1r+w1r; ri[1]=A1i+w1i;
      rr[2]=A2r-B2i; ri[2]=A2i+B2r;
      rr[3]=A3r+w3r; ri[3]=A3i+w3i;
      rr[4]=A0r-B0r; ri[4]=A0i-B0i;
      rr[5]=A1r-w1r; ri[5]=A1i-w1i;
      rr[6]=A2r+B2i; ri[6]=A2i-B2r;
      rr[7]=A3r-w3r; ri[7]=A3i-w3i;
      /* Fused twiddle: W32^(2*k1) */
      { double tr=rr[1]; rr[1]=tr*9.23879532511286738483e-01-ri[1]*3.82683432365089781779e-01; ri[1]=tr*3.82683432365089781779e-01+ri[1]*9.23879532511286738483e-01; }
      { double tr=rr[2]; rr[2]=tr*7.07106781186547572737e-01-ri[2]*7.07106781186547572737e-01; ri[2]=tr*7.07106781186547572737e-01+ri[2]*7.07106781186547572737e-01; }
      { double tr=rr[3]; rr[3]=tr*3.82683432365089837290e-01-ri[3]*9.23879532511286738483e-01; ri[3]=tr*9.23879532511286738483e-01+ri[3]*3.82683432365089837290e-01; }
      { double tr=rr[4]; rr[4]=tr*6.12323399573676603587e-17-ri[4]*1.00000000000000000000e+00; ri[4]=tr*1.00000000000000000000e+00+ri[4]*6.12323399573676603587e-17; }
      { double tr=rr[5]; rr[5]=tr*-3.82683432365089726268e-01-ri[5]*9.23879532511286738483e-01; ri[5]=tr*9.23879532511286738483e-01+ri[5]*-3.82683432365089726268e-01; }
      { double tr=rr[6]; rr[6]=tr*-7.07106781186547461715e-01-ri[6]*7.07106781186547572737e-01; ri[6]=tr*7.07106781186547572737e-01+ri[6]*-7.07106781186547461715e-01; }
      { double tr=rr[7]; rr[7]=tr*-9.23879532511286738483e-01-ri[7]*3.82683432365089892802e-01; ri[7]=tr*3.82683432365089892802e-01+ri[7]*-9.23879532511286738483e-01; }
    }
    { /* Row n2=3 */
      double *rr = &buf_r[24], *ri = &buf_i[24];
      double epr=rr[0]+rr[4],epi=ri[0]+ri[4],eqr=rr[0]-rr[4],eqi=ri[0]-ri[4];
      double err=rr[2]+rr[6],eri=ri[2]+ri[6],esr=rr[2]-rr[6],esi=ri[2]-ri[6];
      double A0r=epr+err,A0i=epi+eri,A2r=epr-err,A2i=epi-eri;
      double A1r=eqr-esi,A1i=eqi+esr,A3r=eqr+esi,A3i=eqi-esr;
      double opr=rr[1]+rr[5],opi=ri[1]+ri[5],oqr=rr[1]-rr[5],oqi=ri[1]-ri[5];
      double orr=rr[3]+rr[7],ori=ri[3]+ri[7],osr=rr[3]-rr[7],osi=ri[3]-ri[7];
      double B0r=opr+orr,B0i=opi+ori,B2r=opr-orr,B2i=opi-ori;
      double B1r=oqr-osi,B1i=oqi+osr,B3r=oqr+osi,B3i=oqi-osr;
      double w1r=0.7071067811865476*(B1r-B1i),w1i=0.7071067811865476*(B1r+B1i);
      double w3r=-0.7071067811865476*(B3r+B3i),w3i=0.7071067811865476*(B3r-B3i);
      rr[0]=A0r+B0r; ri[0]=A0i+B0i;
      rr[1]=A1r+w1r; ri[1]=A1i+w1i;
      rr[2]=A2r-B2i; ri[2]=A2i+B2r;
      rr[3]=A3r+w3r; ri[3]=A3i+w3i;
      rr[4]=A0r-B0r; ri[4]=A0i-B0i;
      rr[5]=A1r-w1r; ri[5]=A1i-w1i;
      rr[6]=A2r+B2i; ri[6]=A2i-B2r;
      rr[7]=A3r-w3r; ri[7]=A3i-w3i;
      /* Fused twiddle: W32^(3*k1) */
      { double tr=rr[1]; rr[1]=tr*8.31469612302545235671e-01-ri[1]*5.55570233019602177649e-01; ri[1]=tr*5.55570233019602177649e-01+ri[1]*8.31469612302545235671e-01; }
      { double tr=rr[2]; rr[2]=tr*3.82683432365089837290e-01-ri[2]*9.23879532511286738483e-01; ri[2]=tr*9.23879532511286738483e-01+ri[2]*3.82683432365089837290e-01; }
      { double tr=rr[3]; rr[3]=tr*-1.95090322016128192573e-01-ri[3]*9.80785280403230430579e-01; ri[3]=tr*9.80785280403230430579e-01+ri[3]*-1.95090322016128192573e-01; }
      { double tr=rr[4]; rr[4]=tr*-7.07106781186547461715e-01-ri[4]*7.07106781186547572737e-01; ri[4]=tr*7.07106781186547572737e-01+ri[4]*-7.07106781186547461715e-01; }
      { double tr=rr[5]; rr[5]=tr*-9.80785280403230430579e-01-ri[5]*1.95090322016128608906e-01; ri[5]=tr*1.95090322016128608906e-01+ri[5]*-9.80785280403230430579e-01; }
      { double tr=rr[6]; rr[6]=tr*-9.23879532511286849505e-01-ri[6]*-3.82683432365089670757e-01; ri[6]=tr*-3.82683432365089670757e-01+ri[6]*-9.23879532511286849505e-01; }
      { double tr=rr[7]; rr[7]=tr*-5.55570233019602177649e-01-ri[7]*-8.31469612302545235671e-01; ri[7]=tr*-8.31469612302545235671e-01+ri[7]*-5.55570233019602177649e-01; }
    }

    /* Pass 2: DFT-4 across rows (vectorized via gather) */
    const __m256i cidx = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);

    /* Reload rows */
    r0r = _mm512_load_pd(&buf_r[0]); r0i = _mm512_load_pd(&buf_i[0]);
    r1r = _mm512_load_pd(&buf_r[8]); r1i = _mm512_load_pd(&buf_i[8]);
    r2r = _mm512_load_pd(&buf_r[16]); r2i = _mm512_load_pd(&buf_i[16]);
    r3r = _mm512_load_pd(&buf_r[24]); r3i = _mm512_load_pd(&buf_i[24]);

    /* DFT-4 across rows — all 8 k1 values in parallel */
    __m512d t0r = _mm512_add_pd(r0r, r2r), t0i = _mm512_add_pd(r0i, r2i);
    __m512d t1r = _mm512_sub_pd(r0r, r2r), t1i = _mm512_sub_pd(r0i, r2i);
    __m512d t2r = _mm512_add_pd(r1r, r3r), t2i = _mm512_add_pd(r1i, r3i);
    __m512d t3r = _mm512_sub_pd(r1r, r3r), t3i = _mm512_sub_pd(r1i, r3i);
    __m512d d0r = _mm512_add_pd(t0r, t2r), d0i = _mm512_add_pd(t0i, t2i);
    __m512d d2r = _mm512_sub_pd(t0r, t2r), d2i = _mm512_sub_pd(t0i, t2i);
    __m512d d1r = _mm512_sub_pd(t1r, t3i), d1i = _mm512_add_pd(t1i, t3r);
    __m512d d3r = _mm512_add_pd(t1r, t3i), d3i = _mm512_sub_pd(t1i, t3r);

    /* Store: contiguous blocks of 8 */
    _mm512_storeu_pd(&or_[0], d0r); _mm512_storeu_pd(&oi[0], d0i);
    _mm512_storeu_pd(&or_[8], d1r); _mm512_storeu_pd(&oi[8], d1i);
    _mm512_storeu_pd(&or_[16], d2r); _mm512_storeu_pd(&oi[16], d2i);
    _mm512_storeu_pd(&or_[24], d3r); _mm512_storeu_pd(&oi[24], d3i);
}

#endif /* FFT_N1_K1_SIMD_H */
