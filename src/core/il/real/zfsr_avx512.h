/* zfsr_avx512.h — the real four-step's block gather, scatter and its two
 * order sweeps at 512 bits (the avx512 build).
 *
 * Included by zfsr.h in place of its 256-bit forms: the method, the 16 x 16
 * block, the seventeenth lane and the two laws of the sweep are zfsr.h's.
 * Here a vector holds four complexes: the block and its mirror block turn in
 * 4 x 4 complex pieces (two rounds of the 128-bit-lane deinterleave), the
 * fold runs four lanes a step, a mirror run is reversed by one 128-bit-lane
 * shuffle, and an output run leaves as four 64-byte vectors -- streamed whole
 * where the caller's buffer sits on a 64-byte boundary, as two streamed
 * halves where it sits on a 32-byte one. Every value is computed by the
 * operations of the 256-bit sweep: the output is bitwise the same. */
#ifndef VFFT_ZFSR_AVX512_H
#define VFFT_ZFSR_AVX512_H

/* the 4 x 4 complex turn: four vectors of four points (a row each) into the four columns' vectors */
static inline __attribute__((always_inline)) void _zfsr_turn4(const __m512d a0, const __m512d a1, const __m512d a2,
                                                             const __m512d a3, __m512d *o0, __m512d *o1,
                                                             __m512d *o2, __m512d *o3)
{
    const __m512d t0 = _mm512_shuffle_f64x2(a0, a1, 0x88), t1 = _mm512_shuffle_f64x2(a0, a1, 0xDD);
    const __m512d t2 = _mm512_shuffle_f64x2(a2, a3, 0x88), t3 = _mm512_shuffle_f64x2(a2, a3, 0xDD);
    *o0 = _mm512_shuffle_f64x2(t0, t2, 0x88);
    *o1 = _mm512_shuffle_f64x2(t1, t3, 0x88);
    *o2 = _mm512_shuffle_f64x2(t0, t2, 0xDD);
    *o3 = _mm512_shuffle_f64x2(t1, t3, 0xDD);
}

/* the block gathers (zfsr.h): bufA[i][j] = the bin k1b+j + N1*(k2b+i), bufB[i][j] its partner, from the
 * segments of k1b+j at the positions k2b+i and the segments of N1-(k1b+j) at the positions N2-1-(k2b+i) */
static inline void _zfsr_gather(const vfft_zfsr_plan_t *p, const double *plane, int k1b, int k2b,
                                double *bufA, double *bufB)
{
    const int N1 = p->N1, N2 = p->N2, cmax = N2 - 1 - k2b;
    const int *P = p->fs->p_of_k1;
    int i, j;
    for (j = 0; j < VFFT_ZFSR_TB; j += 4)
    {
        const double *r0 = plane + 2 * ((size_t)P[k1b + j] * (size_t)N2 + (size_t)k2b);
        const double *r1 = plane + 2 * ((size_t)P[k1b + j + 1] * (size_t)N2 + (size_t)k2b);
        const double *r2 = plane + 2 * ((size_t)P[k1b + j + 2] * (size_t)N2 + (size_t)k2b);
        const double *r3 = plane + 2 * ((size_t)P[k1b + j + 3] * (size_t)N2 + (size_t)k2b);
        const double *q0 = plane + 2 * ((size_t)P[(N1 - k1b - j) % N1] * (size_t)N2);
        const double *q1 = plane + 2 * ((size_t)P[N1 - k1b - j - 1] * (size_t)N2);
        const double *q2 = plane + 2 * ((size_t)P[N1 - k1b - j - 2] * (size_t)N2);
        const double *q3 = plane + 2 * ((size_t)P[N1 - k1b - j - 3] * (size_t)N2);
        for (i = 0; i < VFFT_ZFSR_TB; i += 4)
        {
            __m512d o0, o1, o2, o3;
            _zfsr_turn4(_mm512_loadu_pd(r0 + 2 * i), _mm512_loadu_pd(r1 + 2 * i), _mm512_loadu_pd(r2 + 2 * i),
                        _mm512_loadu_pd(r3 + 2 * i), &o0, &o1, &o2, &o3);
            _mm512_store_pd(bufA + 2 * (i * VFFT_ZFSR_TB + j), o0);
            _mm512_store_pd(bufA + 2 * ((i + 1) * VFFT_ZFSR_TB + j), o1);
            _mm512_store_pd(bufA + 2 * ((i + 2) * VFFT_ZFSR_TB + j), o2);
            _mm512_store_pd(bufA + 2 * ((i + 3) * VFFT_ZFSR_TB + j), o3);
            /* the mirror segments' positions cmax-i-3 .. cmax-i: the last one is the block's row i */
            _zfsr_turn4(_mm512_loadu_pd(q0 + 2 * (cmax - i - 3)), _mm512_loadu_pd(q1 + 2 * (cmax - i - 3)),
                        _mm512_loadu_pd(q2 + 2 * (cmax - i - 3)), _mm512_loadu_pd(q3 + 2 * (cmax - i - 3)), &o0, &o1, &o2, &o3);
            _mm512_store_pd(bufB + 2 * ((i + 3) * VFFT_ZFSR_TB + j), o0);
            _mm512_store_pd(bufB + 2 * ((i + 2) * VFFT_ZFSR_TB + j), o1);
            _mm512_store_pd(bufB + 2 * ((i + 1) * VFFT_ZFSR_TB + j), o2);
            _mm512_store_pd(bufB + 2 * (i * VFFT_ZFSR_TB + j), o3);
        }
    }
}
/* the scatter: both blocks into the plane, whole 64-B vectors at aligned addresses (the plane is 64-B
 * aligned, N2 and the block edges multiples of 16 complexes), streamed */
static inline void _zfsr_scatter(const vfft_zfsr_plan_t *p, double *plane, int k1b, int k2b,
                                 const double *bufA, const double *bufB)
{
    const int N1 = p->N1, N2 = p->N2, cmax = N2 - 1 - k2b;
    const int *P = p->fs->p_of_k1;
    int i, j;
    for (j = 0; j < VFFT_ZFSR_TB; j += 4)
    {
        double *r0 = plane + 2 * ((size_t)P[k1b + j] * (size_t)N2 + (size_t)k2b);
        double *r1 = plane + 2 * ((size_t)P[k1b + j + 1] * (size_t)N2 + (size_t)k2b);
        double *r2 = plane + 2 * ((size_t)P[k1b + j + 2] * (size_t)N2 + (size_t)k2b);
        double *r3 = plane + 2 * ((size_t)P[k1b + j + 3] * (size_t)N2 + (size_t)k2b);
        double *q0 = plane + 2 * ((size_t)P[(N1 - k1b - j) % N1] * (size_t)N2);
        double *q1 = plane + 2 * ((size_t)P[N1 - k1b - j - 1] * (size_t)N2);
        double *q2 = plane + 2 * ((size_t)P[N1 - k1b - j - 2] * (size_t)N2);
        double *q3 = plane + 2 * ((size_t)P[N1 - k1b - j - 3] * (size_t)N2);
        for (i = 0; i < VFFT_ZFSR_TB; i += 4)
        {
            __m512d o0, o1, o2, o3;
            _zfsr_turn4(_mm512_load_pd(bufA + 2 * (i * VFFT_ZFSR_TB + j)), _mm512_load_pd(bufA + 2 * ((i + 1) * VFFT_ZFSR_TB + j)),
                        _mm512_load_pd(bufA + 2 * ((i + 2) * VFFT_ZFSR_TB + j)), _mm512_load_pd(bufA + 2 * ((i + 3) * VFFT_ZFSR_TB + j)),
                        &o0, &o1, &o2, &o3);
            _mm512_stream_pd(r0 + 2 * i, o0);
            _mm512_stream_pd(r1 + 2 * i, o1);
            _mm512_stream_pd(r2 + 2 * i, o2);
            _mm512_stream_pd(r3 + 2 * i, o3);
            _zfsr_turn4(_mm512_load_pd(bufB + 2 * ((i + 3) * VFFT_ZFSR_TB + j)), _mm512_load_pd(bufB + 2 * ((i + 2) * VFFT_ZFSR_TB + j)),
                        _mm512_load_pd(bufB + 2 * ((i + 1) * VFFT_ZFSR_TB + j)), _mm512_load_pd(bufB + 2 * (i * VFFT_ZFSR_TB + j)),
                        &o0, &o1, &o2, &o3);
            _mm512_stream_pd(q0 + 2 * (cmax - i - 3), o0);
            _mm512_stream_pd(q1 + 2 * (cmax - i - 3), o1);
            _mm512_stream_pd(q2 + 2 * (cmax - i - 3), o2);
            _mm512_stream_pd(q3 + 2 * (cmax - i - 3), o3);
        }
    }
}

/* the block's lane twiddles w^(k1b + j), four lanes a vector, each value doubled over its complex */
static inline void _zfsr_lane_tw(const vfft_zfsr_plan_t *p, int k1b, __m512d *vc, __m512d *vs)
{
    int j;
    for (j = 0; j < VFFT_ZFSR_TB; j += 4)
    {
        const __m512d v = _mm512_loadu_pd(p->V + 2 * (k1b + j));   /* (c_j, s_j, .., c_j+3, s_j+3) */
        vc[j / 4] = _mm512_movedup_pd(v);
        vs[j / 4] = _mm512_permute_pd(v, 0xFF);
    }
}
/* an output run's vector: mode 2 one streamed 64-B vector (the address on a 64-byte boundary), 1 two streamed
 * halves (on a 32-byte one), 0 a plain store */
static inline __attribute__((always_inline)) void _zfsr_put(double *o, const __m512d v, const int mode)
{
    if (mode == 2) _mm512_stream_pd(o, v);
    else if (mode == 1)
    {
        _mm256_stream_pd(o, _mm512_castpd512_pd256(v));
        _mm256_stream_pd(o + 4, _mm512_extractf64x4_pd(v, 1));
    }
    else _mm512_storeu_pd(o, v);
}

/* r2c's last sweep (zfsr.h): the four-step's plane -> X */
static void _zfsr_sweep_fwd_range(const vfft_zfsr_plan_t *p, const double *plane, double *X, int k1lo, int k1hi)
{
    const int N1 = p->N1, N2 = p->N2, M = p->M;
    const int *P = p->fs->p_of_k1;
    const __m512d CONJ = _mm512_setr_pd(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
    const __m512d HALF = _mm512_set1_pd(0.5), NHALF = _mm512_set1_pd(-0.5);
    const int mode = (((uintptr_t)X & 63) == 0) ? 2 : (((uintptr_t)X & 31) == 0) ? 1 : 0;
    __attribute__((aligned(64))) double bufA[VFFT_ZFSR_TB * VFFT_ZFSR_TB * 2];
    __attribute__((aligned(64))) double bufB[VFFT_ZFSR_TB * VFFT_ZFSR_TB * 2];
    __attribute__((aligned(64))) double m16[VFFT_ZFSR_TB * 2];        /* the seventeenth lane's mirrors, per i */
    __attribute__((aligned(64))) double mb[(VFFT_ZFSR_TB + 2) * 2];   /* one mirror run + the unused lane-0 slot */
    __m512d vc[VFFT_ZFSR_TB / 4], vs[VFFT_ZFSR_TB / 4];
    int k1b, k2b, i, j;
    for (k1b = k1lo; k1b < k1hi; k1b += VFFT_ZFSR_TB)
    {
        const int kx = (k1b + VFFT_ZFSR_TB) % N1;
        const double *a16row = plane + 2 * (size_t)P[kx] * (size_t)N2;
        const double *b16row = plane + 2 * (size_t)P[(N1 - kx) % N1] * (size_t)N2;
        const __m512d c16 = _mm512_set1_pd(p->V[2 * (k1b + VFFT_ZFSR_TB)]), s16 = _mm512_set1_pd(p->V[2 * (k1b + VFFT_ZFSR_TB) + 1]);
        _zfsr_lane_tw(p, k1b, vc, vs);
        for (k2b = 0; k2b < N2 / 2; k2b += VFFT_ZFSR_TB)
        {
            const int cmax = N2 - 1 - k2b;
            _zfsr_gather(p, plane, k1b, k2b, bufA, bufB);
            /* the seventeenth lane, k1 = k1b + 16: its mirrors only, four positions a vector */
            for (i = 0; i < VFFT_ZFSR_TB; i += 4)
            {
                const __m512d a = _mm512_loadu_pd(a16row + 2 * (k2b + i));
                const __m512d br = _mm512_loadu_pd(b16row + 2 * (cmax - i - 3));
                const __m512d cb = _mm512_xor_pd(_mm512_shuffle_f64x2(br, br, 0x1B), CONJ);
                const __m512d u = _mm512_loadu_pd(p->U + 2 * (k2b + i));
                const __m512d c2 = _mm512_movedup_pd(u), s2 = _mm512_permute_pd(u, 0xFF);
                const __m512d c = _mm512_fnmadd_pd(s16, s2, _mm512_mul_pd(c16, c2));
                const __m512d s = _mm512_fmadd_pd(s16, c2, _mm512_mul_pd(c16, s2));
                const __m512d wr = _mm512_fnmadd_pd(HALF, s, HALF), wi = _mm512_mul_pd(NHALF, c);
                const __m512d t = _mm512_sub_pd(a, cb);
                const __m512d x = _mm512_fmaddsub_pd(wr, t, _mm512_mul_pd(wi, _mm512_permute_pd(t, 0x55)));
                _mm512_store_pd(m16 + 2 * i, _mm512_xor_pd(_mm512_sub_pd(a, x), CONJ));
            }
            for (i = 0; i < VFFT_ZFSR_TB; i++)
            {
                const size_t f0 = (size_t)k1b + (size_t)N1 * (size_t)(k2b + i);
                const double *ra = bufA + 2 * i * VFFT_ZFSR_TB, *rb = bufB + 2 * i * VFFT_ZFSR_TB;
                double *of = X + 2 * f0, *om = X + 2 * ((size_t)M - f0 - VFFT_ZFSR_TB);
                const __m512d c2 = _mm512_set1_pd(p->U[2 * (k2b + i)]), s2 = _mm512_set1_pd(p->U[2 * (k2b + i) + 1]);
                for (j = 0; j < VFFT_ZFSR_TB; j += 4)
                {
                    const __m512d a = _mm512_load_pd(ra + 2 * j);
                    const __m512d cb = _mm512_xor_pd(_mm512_load_pd(rb + 2 * j), CONJ);
                    const __m512d c = _mm512_fnmadd_pd(vs[j / 4], s2, _mm512_mul_pd(vc[j / 4], c2));
                    const __m512d s = _mm512_fmadd_pd(vs[j / 4], c2, _mm512_mul_pd(vc[j / 4], s2));
                    const __m512d wr = _mm512_fnmadd_pd(HALF, s, HALF), wi = _mm512_mul_pd(NHALF, c);
                    const __m512d t = _mm512_sub_pd(a, cb);
                    const __m512d x = _mm512_fmaddsub_pd(wr, t, _mm512_mul_pd(wi, _mm512_permute_pd(t, 0x55)));
                    const __m512d xf = _mm512_add_pd(cb, x);
                    const __m512d xm = _mm512_xor_pd(_mm512_sub_pd(a, x), CONJ);
                    _zfsr_put(of + 2 * j, xf, mode);
                    /* lanes (j .. j+3) mirror to run slots (16-j .. 13-j): slot 16 is lane 0's, not this run's */
                    _mm512_storeu_pd(mb + 2 * (VFFT_ZFSR_TB - 3 - j), _mm512_shuffle_f64x2(xm, xm, 0x1B));
                }
                _mm_store_pd(mb, _mm_load_pd(m16 + 2 * i));   /* slot 0: the mirror of k1b + 16 */
                for (j = 0; j < 2 * VFFT_ZFSR_TB; j += 8) _zfsr_put(om + j, _mm512_load_pd(mb + j), mode);
            }
        }
    }
    if (mode) _mm_sfence();
}

/* c2r's first sweep (zfsr.h): X -> the four-step's plane */
static void _zfsr_sweep_bwd_range(const vfft_zfsr_plan_t *p, const double *X, double *plane, int k1lo, int k1hi)
{
    const int N1 = p->N1, N2 = p->N2, M = p->M;
    const __m512d CONJ = _mm512_setr_pd(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
    const __m512d NEG = _mm512_set1_pd(-0.0);
    __attribute__((aligned(64))) double bufA[VFFT_ZFSR_TB * VFFT_ZFSR_TB * 2];
    __attribute__((aligned(64))) double bufB[VFFT_ZFSR_TB * VFFT_ZFSR_TB * 2];
    __m512d vc[VFFT_ZFSR_TB / 4], vs[VFFT_ZFSR_TB / 4];
    int k1b, k2b, i, j;
    for (k1b = k1lo; k1b < k1hi; k1b += VFFT_ZFSR_TB)
    {
        _zfsr_lane_tw(p, k1b, vc, vs);
        for (k2b = 0; k2b < N2 / 2; k2b += VFFT_ZFSR_TB)
        {
            for (i = 0; i < VFFT_ZFSR_TB; i++)
            {
                const size_t f0 = (size_t)k1b + (size_t)N1 * (size_t)(k2b + i);
                double *ra = bufA + 2 * i * VFFT_ZFSR_TB, *rb = bufB + 2 * i * VFFT_ZFSR_TB;
                const double *xf = X + 2 * f0, *xm = X + 2 * ((size_t)M - f0);
                const __m512d c2 = _mm512_set1_pd(p->U[2 * (k2b + i)]), s2 = _mm512_set1_pd(p->U[2 * (k2b + i) + 1]);
                for (j = 0; j < VFFT_ZFSR_TB; j += 4)
                {
                    const __m512d F = _mm512_loadu_pd(xf + 2 * j);
                    const __m512d mv = _mm512_loadu_pd(xm - 2 * (j + 3));
                    const __m512d cm = _mm512_xor_pd(_mm512_shuffle_f64x2(mv, mv, 0x1B), CONJ);
                    const __m512d c = _mm512_fnmadd_pd(vs[j / 4], s2, _mm512_mul_pd(vc[j / 4], c2));
                    const __m512d s = _mm512_fmadd_pd(vs[j / 4], c2, _mm512_mul_pd(vc[j / 4], s2));
                    const __m512d wi = _mm512_xor_pd(c, NEG);
                    const __m512d t = _mm512_sub_pd(F, cm);
                    const __m512d cy = _mm512_fmaddsub_pd(s, t, _mm512_mul_pd(wi, _mm512_permute_pd(t, 0x55)));
                    const __m512d E = _mm512_add_pd(F, cm);
                    _mm512_store_pd(ra + 2 * j, _mm512_sub_pd(E, cy));
                    _mm512_store_pd(rb + 2 * j, _mm512_xor_pd(_mm512_add_pd(E, cy), CONJ));
                }
            }
            _zfsr_scatter(p, plane, k1b, k2b, bufA, bufB);
        }
    }
    _mm_sfence();
}

#endif /* VFFT_ZFSR_AVX512_H */
