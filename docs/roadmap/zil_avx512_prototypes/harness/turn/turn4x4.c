/* turn4x4.c -- proves the AVX-512 corner-turn store network produces the
 * SAME MEMORY IMAGE as the AVX2 permute2f128 pairing (c2c_il.ml monolithic
 * N1T edge, lines ~1559-1591) for every radix 1..17, every count 1..13,
 * several OLs, and unaligned bases.
 *
 * Isolates the STORE EDGE: "outs" = the loaded legs (identity butterfly),
 * so any difference is the turn itself.
 *
 * AVX2 reference (exactly what the generator emits today):
 *   wide loop k += 2 : legs paired (l, l+1): permute2f128 0x20 -> col k,
 *                      0x31 -> col k+1; odd radix last leg: castpd256_pd128 /
 *                      extractf128(.,1) scatter
 *   tail  k < count  : per leg _mm_storeu_pd(&zout[2*(k*OLs + l)])
 *
 * AVX-512 proposal (per = 4):
 *   full leg groups of 4: two rounds of "complex-lane deinterleave"
 *        even(x,y) = shuffle_f64x2(x, y, 0x88) = [x0,x2,y0,y2]
 *        odd (x,y) = shuffle_f64x2(x, y, 0xDD) = [x1,x3,y1,y3]
 *      round1 on (a,b),(c,d) -> [e0,e1,o0,o1]; round2 on (e0,e1),(o0,o1)
 *      -> [col0,col1,col2,col3]  (8 vshuff64x2, 4 full zmm stores)
 *   leftover r = R mod 4:
 *      r == 1: quarter scatter, extractf64x2(v, c) -> col c (store-only uops)
 *      r >= 2: same network, missing partners replaced by the vector itself
 *              (self-pad), each column stored with a prefix mask (2r doubles)
 *   column tail (count mod 4 in 1..3): ONE masked iteration -- maskz loads of
 *      the rem valid columns, the SAME network, column c stored with mask
 *      (c < rem ? legmask : 0). Masked-off stores fault-suppress, proven
 *      below with a PROT_NONE guard page right after the output buffer.
 *
 * Build: gcc -O2 -mavx2 -mfma -mavx512f -mavx512dq -mavx512vl turn4x4.c
 */
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <unistd.h>

/* ---------------- AVX2 reference: today's emitted edge ---------------- */
__attribute__((target("avx2,fma")))
static void turn_avx2(const double *zin, double *zout, size_t Ls, size_t OLs,
                      size_t count, int R)
{
    size_t k = 0;
    for (; k + 2 <= count; k += 2) {
        __m256d o[64];
        for (int l = 0; l < R; l++) o[l] = _mm256_loadu_pd(&zin[2*((size_t)l*Ls + k)]);
        int l = 0;
        for (; l + 1 < R; l += 2) {
            _mm256_storeu_pd(&zout[2*((size_t)k*OLs + l)], _mm256_permute2f128_pd(o[l], o[l+1], 0x20));
            _mm256_storeu_pd(&zout[2*(((size_t)k + 1)*OLs + l)], _mm256_permute2f128_pd(o[l], o[l+1], 0x31));
        }
        if (l < R) {
            _mm_storeu_pd(&zout[2*((size_t)k*OLs + l)], _mm256_castpd256_pd128(o[l]));
            _mm_storeu_pd(&zout[2*(((size_t)k + 1)*OLs + l)], _mm256_extractf128_pd(o[l], 1));
        }
    }
    for (; k < count; ++k)
        for (int l = 0; l < R; l++)
            _mm_storeu_pd(&zout[2*((size_t)k*OLs + l)], _mm_loadu_pd(&zin[2*((size_t)l*Ls + k)]));
}

/* ---------------- AVX-512 proposal ---------------- */
#define EVEN(x, y) _mm512_shuffle_f64x2((x), (y), 0x88)
#define ODD(x, y)  _mm512_shuffle_f64x2((x), (y), 0xDD)

/* The generator-side network, written as the generic even/odd rounds
 * (n = 4 -> 2 rounds). v[0..r-1] real, r in 2..4; missing partner = self. */
__attribute__((target("avx512f,avx512dq,avx512vl")))
static inline void transpose4(const __m512d *v, int r, __m512d *col)
{
    __m512d a = v[0], b = r > 1 ? v[1] : v[0];
    __m512d c = r > 2 ? v[2] : a, d = r > 3 ? v[3] : (r > 2 ? v[2] : b);
    /* round 1 (pairs (a,b), (c,d)) */
    __m512d e0 = EVEN(a, b), o0 = ODD(a, b);
    __m512d e1 = EVEN(c, d), o1 = ODD(c, d);
    /* round 2 (pairs (e0,e1), (o0,o1)) -> columns 0,1,2,3 in order */
    col[0] = EVEN(e0, e1);
    col[1] = EVEN(o0, o1);
    col[2] = ODD(e0, e1);
    col[3] = ODD(o0, o1);
}

__attribute__((target("avx512f,avx512dq,avx512vl")))
static inline void store_group(double *zout, size_t k, size_t OLs, int l,
                               const __m512d *v, int r, int rem)
{
    /* rem = valid columns in this iteration (4 in the wide loop) */
    if (r == 1) {
        for (int c = 0; c < 4; c++) {
            const __mmask8 m = (c < rem) ? 0x3 : 0x0;
            double *p = &zout[2*(((size_t)k + c)*OLs + l)];
            if (rem == 4) {
                __m128d q = c == 0 ? _mm512_castpd512_pd128(v[0])
                          : c == 1 ? _mm512_extractf64x2_pd(v[0], 1)
                          : c == 2 ? _mm512_extractf64x2_pd(v[0], 2)
                                   : _mm512_extractf64x2_pd(v[0], 3);
                _mm_storeu_pd(p, q);
            } else {
                /* masked tail: whole-zmm masked store, column c's quarter
                   moved to lane 0 -- or equivalently mask the quarter */
                __m128d q = c == 0 ? _mm512_castpd512_pd128(v[0])
                          : c == 1 ? _mm512_extractf64x2_pd(v[0], 1)
                          : c == 2 ? _mm512_extractf64x2_pd(v[0], 2)
                                   : _mm512_extractf64x2_pd(v[0], 3);
                _mm_mask_storeu_pd(p, m, q);          /* AVX512VL */
            }
        }
        return;
    }
    __m512d col[4];
    transpose4(v, r, col);
    const __mmask8 legm = (__mmask8)((1u << (2 * r)) - 1u);
    for (int c = 0; c < 4; c++) {
        double *p = &zout[2*(((size_t)k + c)*OLs + l)];
        const __mmask8 m = (c < rem) ? legm : 0;
        if (m == 0xFF) _mm512_storeu_pd(p, col[c]);
        else _mm512_mask_storeu_pd(p, m, col[c]);
    }
}

__attribute__((target("avx512f,avx512dq,avx512vl")))
static void turn_avx512(const double *zin, double *zout, size_t Ls, size_t OLs,
                        size_t count, int R)
{
    size_t k = 0;
    for (; k + 4 <= count; k += 4) {
        __m512d o[64];
        for (int l = 0; l < R; l++) o[l] = _mm512_loadu_pd(&zin[2*((size_t)l*Ls + k)]);
        int l = 0;
        for (; l + 4 <= R; l += 4) store_group(zout, k, OLs, l, &o[l], 4, 4);
        if (l < R) store_group(zout, k, OLs, l, &o[l], R - l, 4);
    }
    if (k < count) {           /* ONE masked iteration, rem in 1..3 */
        const int rem = (int)(count - k);
        const __mmask8 lm = (__mmask8)((1u << (2 * rem)) - 1u);
        __m512d o[64];
        for (int l = 0; l < R; l++) o[l] = _mm512_maskz_loadu_pd(lm, &zin[2*((size_t)l*Ls + k)]);
        int l = 0;
        for (; l + 4 <= R; l += 4) store_group(zout, k, OLs, l, &o[l], 4, rem);
        if (l < R) store_group(zout, k, OLs, l, &o[l], R - l, rem);
    }
}

/* ---------------- leg-strided turn (t2tg): quarter scatter ---------------- */
__attribute__((target("avx2,fma")))
static void turng_avx2(const double *zin, double *zout, size_t Ls, size_t OLs,
                       size_t OGs, size_t count, int R)
{
    size_t k = 0;
    for (; k + 2 <= count; k += 2)
        for (int l = 0; l < R; l++) {
            __m256d v = _mm256_loadu_pd(&zin[2*((size_t)l*Ls + k)]);
            _mm_storeu_pd(&zout[2*((size_t)k*OLs + (size_t)l*OGs)], _mm256_castpd256_pd128(v));
            _mm_storeu_pd(&zout[2*(((size_t)k + 1)*OLs + (size_t)l*OGs)], _mm256_extractf128_pd(v, 1));
        }
    for (; k < count; ++k)
        for (int l = 0; l < R; l++)
            _mm_storeu_pd(&zout[2*((size_t)k*OLs + (size_t)l*OGs)], _mm_loadu_pd(&zin[2*((size_t)l*Ls + k)]));
}
__attribute__((target("avx512f,avx512dq,avx512vl")))
static void turng_avx512(const double *zin, double *zout, size_t Ls, size_t OLs,
                         size_t OGs, size_t count, int R)
{
    size_t k = 0;
    for (; k < count; k += 4) {
        const int rem = (count - k) >= 4 ? 4 : (int)(count - k);
        const __mmask8 lm = (__mmask8)((1u << (2 * rem)) - 1u);
        for (int l = 0; l < R; l++) {
            __m512d v = rem == 4 ? _mm512_loadu_pd(&zin[2*((size_t)l*Ls + k)])
                                 : _mm512_maskz_loadu_pd(lm, &zin[2*((size_t)l*Ls + k)]);
            __m128d q[4] = { _mm512_castpd512_pd128(v), _mm512_extractf64x2_pd(v, 1),
                             _mm512_extractf64x2_pd(v, 2), _mm512_extractf64x2_pd(v, 3) };
            for (int c = 0; c < 4; c++)
                _mm_mask_storeu_pd(&zout[2*(((size_t)k + c)*OLs + (size_t)l*OGs)],
                                   c < rem ? 0x3 : 0x0, q[c]);
        }
    }
}

/* ---------------- harness ---------------- */
static double *guarded(size_t ndbl, size_t off_bytes, void **base, size_t *mlen)
{
    /* the buffer ENDS exactly at a PROT_NONE page: any masked-off store that
       did not fault-suppress would SIGSEGV */
    long pg = sysconf(_SC_PAGESIZE);
    size_t bytes = ndbl * sizeof(double) + off_bytes;
    size_t npg = (bytes + pg - 1) / pg;
    *mlen = (npg + 1) * pg;
    char *m = mmap(0, *mlen, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) { perror("mmap"); exit(2); }
    mprotect(m + npg * pg, pg, PROT_NONE);
    *base = m;
    return (double *)(m + npg * pg - ndbl * sizeof(double));   /* flush against the guard */
}

int main(void)
{
    long fails = 0, cases = 0;
    const size_t offs[] = { 0, 16, 32, 48 };   /* byte offsets of zin within a 64B line */
    for (int R = 1; R <= 17; R++)
    for (size_t count = 1; count <= 13; count++)
    for (int oi = 0; oi < 3; oi++)
    for (int ai = 0; ai < 4; ai++) {
        const size_t Ls = count + (size_t)ai;           /* leg stride >= count */
        const size_t OLs = (size_t)R + (size_t)(oi == 0 ? 0 : oi == 1 ? 3 : 8);
        /* input: R legs x Ls complex, plus 1 spare */
        size_t nin = 2 * ((size_t)R * Ls + 4);
        double *inbuf = aligned_alloc(64, (nin + 8) * sizeof(double));
        double *zin = (double *)((char *)inbuf + offs[ai]);
        for (size_t i = 0; i < nin; i++) zin[i] = (double)(i + 1) * 1.25 + 0.5;
        /* output extent: (count-1)*OLs + R complex exactly */
        size_t nout = 2 * ((count - 1) * OLs + (size_t)R);
        void *b1, *b2; size_t m1, m2;
        double *z1 = guarded(nout, 0, &b1, &m1);
        double *z2 = guarded(nout, 0, &b2, &m2);
        for (size_t i = 0; i < nout; i++) { z1[i] = -777.0; z2[i] = -777.0; }
        turn_avx2(zin, z1, Ls, OLs, count, R);
        turn_avx512(zin, z2, Ls, OLs, count, R);
        cases++;
        if (memcmp(z1, z2, nout * sizeof(double))) {
            fails++;
            if (fails < 10) printf("MISMATCH turn R=%d count=%zu OLs=%zu Ls=%zu\n", R, count, OLs, Ls);
        }
        /* t2tg: leg stride OGs */
        const size_t OGs = 3, OLs2 = (size_t)R * OGs + 1;
        size_t nout2 = 2 * ((count - 1) * OLs2 + (size_t)(R - 1) * OGs + 1);
        void *b3, *b4; size_t m3, m4;
        double *z3 = guarded(nout2, 0, &b3, &m3);
        double *z4 = guarded(nout2, 0, &b4, &m4);
        for (size_t i = 0; i < nout2; i++) { z3[i] = -777.0; z4[i] = -777.0; }
        turng_avx2(zin, z3, Ls, OLs2, OGs, count, R);
        turng_avx512(zin, z4, Ls, OLs2, OGs, count, R);
        cases++;
        if (memcmp(z3, z4, nout2 * sizeof(double))) {
            fails++;
            if (fails < 10) printf("MISMATCH turng R=%d count=%zu\n", R, count);
        }
        munmap(b1, m1); munmap(b2, m2); munmap(b3, m3); munmap(b4, m4);
        free(inbuf);
    }
    /* explicit 4 legs x 4 complex dump */
    {
        double zin[32], z1[32], z2[32];
        for (int i = 0; i < 32; i++) zin[i] = i;           /* leg l col k = (8l+2k, 8l+2k+1) */
        turn_avx2(zin, z1, 4, 4, 4, 4);
        turn_avx512(zin, z2, 4, 4, 4, 4);
        printf("4x4 image avx2  :"); for (int i = 0; i < 32; i += 2) printf(" %g", z1[i]); printf("\n");
        printf("4x4 image avx512:"); for (int i = 0; i < 32; i += 2) printf(" %g", z2[i]); printf("\n");
        printf("4x4 memcmp = %d\n", memcmp(z1, z2, sizeof z1));
    }
    printf("cases=%ld fails=%ld\n", cases, fails);
    return fails != 0;
}
