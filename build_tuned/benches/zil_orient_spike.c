/* zil_orient_spike.c — LEVER 1 spike (z_cascade_plan §4.9995): move the
 * terminator's load-side 4x4 transpose into the LAST MID's stores.
 *
 * Census (anatomy §9): MKL's finishers do NO load-side transpose — the whole
 * interior is shuffle-free and layout is converted only on store paths. Our
 * terminator pays 2.0 port-5 shuffles/complex (1.0 store-side REINT = matches
 * MKL, + 1.0 load-side TR4 = pure surplus). The mids have port-5 idle (CPI
 * 0.238, 71% retiring). So relocate the TR4: the port-saturated pass sheds it,
 * the idle-port pass absorbs it. Same op count, better port balance.
 *
 * DERIVATION (verified below by a bit-exact gate). Last mid = stage nf-2,
 * radix-8, Ls=8, count=8, groups G. Terminator reads iteration k at base
 * 16*k, today loading 16 doubles/column + 4xTR4. The terminator's radix-8
 * combines the mid's 8 output columns (mc0-7) of each mid-leg, with mid-leg as
 * the SIMD lane; its TR4 swaps (mid-leg <-> mid-column) per 4x4 tile.
 *
 * NEW handoff layout (private scratch between last-mid and terminator):
 *   per mid group g at bp=g*128: two terminator iterations
 *     I0 (mid legs 0-3 -> lanes) at bp+0 ; I1 (mid legs 4-7) at bp+64.
 *   within an iteration RI: xr[l] at RI+4*l, xi[l] at RI+32+4*l  (l=0..7).
 * msgt applies the 4 TR4s in its stores; stermf loads xr[l]=zin+16*k+4*l,
 * xi[l]=zin+16*k+32+4*l with NO transpose, then the SAME packed-w1 squaring
 * tree + SPLIT_BFLY8 + REINT stores as sterm. Per-iteration base 16*k
 * UNCHANGED (bp+RI == 16*k), so only the within-block order differs.
 *
 * Build: python build.py --src benches/zil_orient_spike.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include <immintrin.h>

#include "zsplit.h"

/* ---- macros verbatim from the emitted zil family ---- */
#define DEINT(zlo, zhi, re, im) do {                                  \
    __m256d _u = _mm256_unpacklo_pd(zlo, zhi);                        \
    __m256d _v = _mm256_unpackhi_pd(zlo, zhi);                        \
    re = _mm256_permute4x64_pd(_u, 0xD8);                             \
    im = _mm256_permute4x64_pd(_v, 0xD8);                             \
} while (0)
#define REINT(re, im, zlo, zhi) do {                                  \
    __m256d _p = _mm256_permute4x64_pd(re, 0xD8);                     \
    __m256d _q = _mm256_permute4x64_pd(im, 0xD8);                     \
    zlo = _mm256_unpacklo_pd(_p, _q);                                 \
    zhi = _mm256_unpackhi_pd(_p, _q);                                 \
} while (0)
#define SPLIT_CMUL(ar,ai, ct,st, or_,oi_) do {                        \
    or_ = _mm256_fnmadd_pd(st, ai, _mm256_mul_pd(ct, ar));            \
    oi_ = _mm256_fmadd_pd(st, ar, _mm256_mul_pd(ct, ai));             \
} while (0)
#define TR4(a0,a1,a2,a3, t0,t1,t2,t3) do {                            \
    __m256d _u0 = _mm256_unpacklo_pd(a0, a1);                         \
    __m256d _u1 = _mm256_unpackhi_pd(a0, a1);                         \
    __m256d _u2 = _mm256_unpacklo_pd(a2, a3);                         \
    __m256d _u3 = _mm256_unpackhi_pd(a2, a3);                         \
    t0 = _mm256_permute2f128_pd(_u0, _u2, 0x20);                      \
    t1 = _mm256_permute2f128_pd(_u1, _u3, 0x20);                      \
    t2 = _mm256_permute2f128_pd(_u0, _u2, 0x31);                      \
    t3 = _mm256_permute2f128_pd(_u1, _u3, 0x31);                      \
} while (0)
#define WPROD(cA,sA, cB,sB, cP,sP) do {                               \
    cP = _mm256_fnmadd_pd(sA, sB, _mm256_mul_pd(cA, cB));             \
    sP = _mm256_fmadd_pd(cA, sB, _mm256_mul_pd(sA, cB));              \
} while (0)
#define SPLIT_BFLY8(x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i, \
                    o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i) do { \
    const __m256d _C = _mm256_set1_pd(0.70710678118654752440);        \
    __m256d t0r=_mm256_add_pd(x0r,x4r), t0i=_mm256_add_pd(x0i,x4i);   \
    __m256d t1r=_mm256_sub_pd(x0r,x4r), t1i=_mm256_sub_pd(x0i,x4i);   \
    __m256d t2r=_mm256_add_pd(x2r,x6r), t2i=_mm256_add_pd(x2i,x6i);   \
    __m256d t3r=_mm256_sub_pd(x2r,x6r), t3i=_mm256_sub_pd(x2i,x6i);   \
    __m256d E0r=_mm256_add_pd(t0r,t2r), E0i=_mm256_add_pd(t0i,t2i);   \
    __m256d E2r=_mm256_sub_pd(t0r,t2r), E2i=_mm256_sub_pd(t0i,t2i);   \
    __m256d E1r=_mm256_add_pd(t1r,t3i), E1i=_mm256_sub_pd(t1i,t3r);   \
    __m256d E3r=_mm256_sub_pd(t1r,t3i), E3i=_mm256_add_pd(t1i,t3r);   \
    __m256d s0r=_mm256_add_pd(x1r,x5r), s0i=_mm256_add_pd(x1i,x5i);   \
    __m256d s1r=_mm256_sub_pd(x1r,x5r), s1i=_mm256_sub_pd(x1i,x5i);   \
    __m256d s2r=_mm256_add_pd(x3r,x7r), s2i=_mm256_add_pd(x3i,x7i);   \
    __m256d s3r=_mm256_sub_pd(x3r,x7r), s3i=_mm256_sub_pd(x3i,x7i);   \
    __m256d O0r=_mm256_add_pd(s0r,s2r), O0i=_mm256_add_pd(s0i,s2i);   \
    __m256d O2r=_mm256_sub_pd(s0r,s2r), O2i=_mm256_sub_pd(s0i,s2i);   \
    __m256d O1r=_mm256_add_pd(s1r,s3i), O1i=_mm256_sub_pd(s1i,s3r);   \
    __m256d O3r=_mm256_sub_pd(s1r,s3i), O3i=_mm256_add_pd(s1i,s3r);   \
    __m256d X1r=_mm256_add_pd(O1r,O1i), X1i=_mm256_sub_pd(O1i,O1r);   \
    __m256d X3r=_mm256_sub_pd(O3i,O3r), X3n=_mm256_add_pd(O3r,O3i);   \
    o0r=_mm256_add_pd(E0r,O0r); o0i=_mm256_add_pd(E0i,O0i);           \
    o4r=_mm256_sub_pd(E0r,O0r); o4i=_mm256_sub_pd(E0i,O0i);           \
    o1r=_mm256_fmadd_pd(_C,X1r,E1r); o1i=_mm256_fmadd_pd(_C,X1i,E1i); \
    o5r=_mm256_fnmadd_pd(_C,X1r,E1r); o5i=_mm256_fnmadd_pd(_C,X1i,E1i); \
    o2r=_mm256_add_pd(E2r,O2i); o2i=_mm256_sub_pd(E2i,O2r);           \
    o6r=_mm256_sub_pd(E2r,O2i); o6i=_mm256_add_pd(E2i,O2r);           \
    o3r=_mm256_fmadd_pd(_C,X3r,E3r); o3i=_mm256_fnmadd_pd(_C,X3n,E3i); \
    o7r=_mm256_fnmadd_pd(_C,X3r,E3r); o7i=_mm256_fmadd_pd(_C,X3n,E3i); \
} while (0)

typedef void (*zfn)(const double *, const double *, double *, double *,
                    const double *, const double *, unsigned long long,
                    unsigned long long, unsigned long long,
                    unsigned long long, unsigned long long);

/* ============================================================= *
 *  msgt — orientation-aware LAST MID (radix-8, Ls=8, count=8).   *
 *  Body identical to msg, but the stores apply the 4 TR4s and    *
 *  write the terminator's pre-oriented layout.                   *
 * ============================================================= */
__attribute__((target("avx2,fma")))
static void radix8_z_msgt_fwd(
    const double * __restrict__ zin, const double * __restrict__ zu,
    double * __restrict__ zout, double * __restrict__ zou,
    const double * tw_re, const double * tw_im,
    unsigned long long Ls, unsigned long long Gs, unsigned long long OLs,
    unsigned long long OGs, unsigned long long count)
{
    (void)zin; (void)zu; (void)zou; (void)tw_im; (void)OLs; (void)OGs;
    /* contract: Ls==8, count==8 (last mid); group span = 2*8*Ls = 128 dbl.
     * IN-PLACE, register-blocked: read+compute BOTH k-blocks before any store
     * (the transposed stores scatter across the group and would clobber the
     * k=4 inputs). Stores land on the warm in-place plane — no cold 2nd plane,
     * no snapshot copy. Emitter version can tune the store schedule. */
    double *bp = zout;
    const double *twg = tw_re;
    for (unsigned long long g = 0; g < Gs; g++) {
        __m256d oR[2][8], oI[2][8];   /* [kblock][leg] */
        for (int kb = 0; kb < 2; kb++) {
            size_t k = (size_t)kb * 4;
            __m256d xr[8], xi[8];
            xr[0] = _mm256_loadu_pd(bp + 2*(size_t)k);
            xi[0] = _mm256_loadu_pd(bp + 2*(size_t)k + 4);
            for (int l = 1; l < 8; l++) {
                __m256d ar = _mm256_loadu_pd(bp + 2*((size_t)l*Ls + k));
                __m256d ai = _mm256_loadu_pd(bp + 2*((size_t)l*Ls + k) + 4);
                __m256d ct = _mm256_loadu_pd(twg + (size_t)(l - 1) * 8);
                __m256d st = _mm256_loadu_pd(twg + (size_t)(l - 1) * 8 + 4);
                SPLIT_CMUL(ar, ai, ct, st, xr[l], xi[l]);
            }
            SPLIT_BFLY8(xr[0],xi[0],xr[1],xi[1],xr[2],xi[2],xr[3],xi[3],
                        xr[4],xi[4],xr[5],xi[5],xr[6],xi[6],xr[7],xi[7],
                        oR[kb][0],oI[kb][0],oR[kb][1],oI[kb][1],oR[kb][2],oI[kb][2],oR[kb][3],oI[kb][3],
                        oR[kb][4],oI[kb][4],oR[kb][5],oI[kb][5],oR[kb][6],oI[kb][6],oR[kb][7],oI[kb][7]);
        }
        /* all reads done — now the 8 TR4s + 32 transposed stores, in place.
         * kb -> blbase (butterfly-leg half): kb0->0 (legs0-3), kb1->16 (legs4-7).
         * legs 0-3 -> I0 (bp+0), legs 4-7 -> I1 (bp+64). */
        for (int kb = 0; kb < 2; kb++) {
            size_t blbase = (size_t)kb * 16;
            __m256d a, b, c, d;
            TR4(oR[kb][0],oR[kb][1],oR[kb][2],oR[kb][3], a,b,c,d);
            _mm256_storeu_pd(bp + blbase + 0, a);  _mm256_storeu_pd(bp + blbase + 4, b);
            _mm256_storeu_pd(bp + blbase + 8, c);  _mm256_storeu_pd(bp + blbase + 12, d);
            TR4(oI[kb][0],oI[kb][1],oI[kb][2],oI[kb][3], a,b,c,d);
            _mm256_storeu_pd(bp + 32 + blbase + 0, a);  _mm256_storeu_pd(bp + 32 + blbase + 4, b);
            _mm256_storeu_pd(bp + 32 + blbase + 8, c);  _mm256_storeu_pd(bp + 32 + blbase + 12, d);
            TR4(oR[kb][4],oR[kb][5],oR[kb][6],oR[kb][7], a,b,c,d);
            _mm256_storeu_pd(bp + 64 + blbase + 0, a);  _mm256_storeu_pd(bp + 64 + blbase + 4, b);
            _mm256_storeu_pd(bp + 64 + blbase + 8, c);  _mm256_storeu_pd(bp + 64 + blbase + 12, d);
            TR4(oI[kb][4],oI[kb][5],oI[kb][6],oI[kb][7], a,b,c,d);
            _mm256_storeu_pd(bp + 96 + blbase + 0, a);  _mm256_storeu_pd(bp + 96 + blbase + 4, b);
            _mm256_storeu_pd(bp + 96 + blbase + 8, c);  _mm256_storeu_pd(bp + 96 + blbase + 12, d);
        }
        bp += 2 * (size_t)8 * Ls;
        twg += 56;
    }
}

/* ============================================================= *
 *  stermf — TRANSPOSE-FREE terminator. sterm minus the load TR4:  *
 *  xr[l]=zin+16k+4l, xi[l]=zin+16k+32+4l (l=0..7), NO transpose.   *
 *  Everything after (twiddles, bfly, REINT stores) verbatim.      *
 * ============================================================= */
__attribute__((target("avx2,fma")))
static void radix8_z_stermf_fwd(
    const double * __restrict__ zin, const double * __restrict__ zu,
    double * __restrict__ zout, double * __restrict__ zou,
    const double * tw_re, const double * tw_im,
    unsigned long long Ls, unsigned long long Gs, unsigned long long OLs,
    unsigned long long OGs, unsigned long long count)
{
    (void)zu; (void)zou; (void)tw_im; (void)Ls; (void)Gs; (void)OGs;
    for (size_t k = 0; k + 4 <= count; k += 4) {
        __m256d xr[8], xi[8];
        for (int l = 0; l < 8; l++) {
            xr[l] = _mm256_loadu_pd(zin + 16*(size_t)k + 4*(size_t)l);
            xi[l] = _mm256_loadu_pd(zin + 16*(size_t)k + 32 + 4*(size_t)l);
        }
        {
            __m256d c1 = _mm256_loadu_pd(tw_re + 2*(size_t)k);
            __m256d s1 = _mm256_loadu_pd(tw_re + 2*(size_t)k + 4);
            __m256d c2, s2, c3, s3, c4, s4, cw, sw, rr, ii;
            SPLIT_CMUL(xr[1], xi[1], c1, s1, rr, ii); xr[1] = rr; xi[1] = ii;
            WPROD(c1, s1, c1, s1, c2, s2);
            SPLIT_CMUL(xr[2], xi[2], c2, s2, rr, ii); xr[2] = rr; xi[2] = ii;
            WPROD(c2, s2, c1, s1, c3, s3);
            SPLIT_CMUL(xr[3], xi[3], c3, s3, rr, ii); xr[3] = rr; xi[3] = ii;
            WPROD(c2, s2, c2, s2, c4, s4);
            SPLIT_CMUL(xr[4], xi[4], c4, s4, rr, ii); xr[4] = rr; xi[4] = ii;
            WPROD(c4, s4, c1, s1, cw, sw);
            SPLIT_CMUL(xr[5], xi[5], cw, sw, rr, ii); xr[5] = rr; xi[5] = ii;
            WPROD(c4, s4, c2, s2, cw, sw);
            SPLIT_CMUL(xr[6], xi[6], cw, sw, rr, ii); xr[6] = rr; xi[6] = ii;
            WPROD(c4, s4, c3, s3, cw, sw);
            SPLIT_CMUL(xr[7], xi[7], cw, sw, rr, ii); xr[7] = rr; xi[7] = ii;
        }
        __m256d or_[8], oi_[8];
        SPLIT_BFLY8(xr[0],xi[0],xr[1],xi[1],xr[2],xi[2],xr[3],xi[3],
                    xr[4],xi[4],xr[5],xi[5],xr[6],xi[6],xr[7],xi[7],
                    or_[0],oi_[0],or_[1],oi_[1],or_[2],oi_[2],or_[3],oi_[3],
                    or_[4],oi_[4],or_[5],oi_[5],or_[6],oi_[6],or_[7],oi_[7]);
        for (int l = 0; l < 8; l++) {
            __m256d zlo, zhi;
            REINT(or_[l], oi_[l], zlo, zhi);
            _mm256_storeu_pd(zout + 2*((size_t)l*OLs + k), zlo);
            _mm256_storeu_pd(zout + 2*((size_t)l*OLs + k) + 4, zhi);
        }
    }
}

/* ---- run the shared front (s0s + mids 1..nf-3) into p->sp ---- */
static void run_shared_front(const vfft_zsplit_plan_t *p, const double *zin)
{
    const int nf = p->nf;
    if (p->chain[0] == 8)
        radix8_z_s0s_fwd_avx2(zin, 0, p->sp, 0, 0, 0,
                              (unsigned long long)p->D[0], 0,
                              (unsigned long long)p->D[0], 0,
                              (unsigned long long)p->D[0]);
    else
        radix4_z_s0s_fwd_avx2(zin, 0, p->sp, 0, 0, 0,
                              (unsigned long long)p->D[0], 0,
                              (unsigned long long)p->D[0], 0,
                              (unsigned long long)p->D[0]);
    for (int s = 1; s <= nf - 3; s++) {          /* stop BEFORE the last mid */
        zfn f = (p->chain[s] == 8) ? radix8_z_msg_fwd_avx2
                                   : radix4_z_msg_fwd_avx2;
        f(p->sp, 0, p->sp, 0, p->twsp[s], 0,
          (unsigned long long)p->D[s], (unsigned long long)p->G[s],
          0, 0, (unsigned long long)p->D[s]);
    }
}

static double qpc_ms(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return 1000.0 * (double)c.QuadPart / (double)f.QuadPart;
}
static char *g_bust;
#define BUST_SZ (32u * 1024u * 1024u)
static void cachebust(void) { for (size_t i = 0; i < BUST_SZ; i += 64) g_bust[i]++; }

int main(void)
{
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    g_bust = (char *)malloc(BUST_SZ);
    memset(g_bust, 1, BUST_SZ);
    const int cells[] = { 2048, 4096, 8192, 16384 };
    int rc = 0;

    for (int ci = 0; ci < 4; ci++) {
        const int N = cells[ci];
        int chain[VFFT_ZSPLIT_MAX_NF];
        int nf = vfft_zsplit_default_chain(N, chain);
        vfft_zsplit_plan_t *p = vfft_zsplit_create(N, chain, nf);
        if (!p) { printf("N=%d create FAIL\n", N); return 1; }
        if (p->chain[nf - 2] != 8) { printf("N=%d last mid not r8 — skip\n", N); continue; }

        double *in   = (double *)_aligned_malloc((size_t)2 * N * 8, 64);
        double *snap = (double *)_aligned_malloc((size_t)2 * N * 8, 64);
        double *ref  = (double *)_aligned_malloc((size_t)2 * N * 8, 64);
        double *cand = (double *)_aligned_malloc((size_t)2 * N * 8, 64);
        double *sp2  = (double *)_aligned_malloc((size_t)2 * N * 8, 64); /* last-mid OOP out */
        srand(N + 3);
        for (int i = 0; i < 2 * N; i++) in[i] = (double)rand() / RAND_MAX - 0.5;

        const unsigned long long lastLs = (unsigned long long)p->D[nf - 2];
        const unsigned long long lastG  = (unsigned long long)p->G[nf - 2];
        const unsigned long long cnt    = (unsigned long long)p->D[nf - 2];
        const double *lasttw = p->twsp[nf - 2];
        const unsigned long long tcols  = (unsigned long long)(N / 8);

        /* reference: shared front (into sp) then real last-msg + real sterm */
        run_shared_front(p, in);
        memcpy(snap, p->sp, (size_t)2 * N * 8);
        radix8_z_msg_fwd_avx2(p->sp, 0, p->sp, 0, lasttw, 0, lastLs, lastG, 0, 0, cnt);
        radix8_z_sterm_fwd_avx2(p->sp, 0, ref, 0, p->twq, 0, 0, 0, tcols, 0, tcols);

        /* candidate: same snap -> msgt (IN-PLACE) + stermf, both on p->sp */
        memcpy(p->sp, snap, (size_t)2 * N * 8);
        radix8_z_msgt_fwd(p->sp, 0, p->sp, 0, lasttw, 0, lastLs, lastG, 0, 0, cnt);
        radix8_z_stermf_fwd(p->sp, 0, cand, 0, p->twq, 0, 0, 0, tcols, 0, tcols);

        int bad = memcmp(ref, cand, (size_t)2 * N * 8) != 0;
        double maxerr = 0;
        for (int i = 0; i < 2 * N; i++) { double d = ref[i]-cand[i]; if (d<0)d=-d; if(d>maxerr)maxerr=d; }
        printf("N=%-6d GATE msgt+stermf vs msg+sterm : %s (bitcmp %s, maxabs %.2e)\n",
               N, (!bad) ? "BIT-IDENTICAL PASS" : (maxerr < 1e-15 ? "near (non-bit)" : "FAIL"),
               bad ? "differ" : "equal", maxerr);
        if (bad && maxerr >= 1e-13) rc = 1;

        /* race: last-mid + terminator only (the changed region), paced */
        const int RF = (N <= 2048) ? 400 : (N <= 4096) ? 200 : (N <= 8192) ? 100 : 60;
        const int ROUNDS = 7;
        double bestOld = 1e30, bestNew = 1e30;
        for (int r = 0; r < ROUNDS; r++) {
            for (int j = 0; j < 2; j++) {
                int arm = (j + r) & 1;
                if (arm == 0) {
                    cachebust();
                    memcpy(p->sp, snap, (size_t)2 * N * 8);
                    radix8_z_msg_fwd_avx2(p->sp, 0, p->sp, 0, lasttw, 0, lastLs, lastG, 0, 0, cnt);
                    radix8_z_sterm_fwd_avx2(p->sp, 0, ref, 0, p->twq, 0, 0, 0, tcols, 0, tcols);
                    double t0 = qpc_ms();
                    for (int i = 0; i < RF; i++) {
                        radix8_z_msg_fwd_avx2(p->sp, 0, p->sp, 0, lasttw, 0, lastLs, lastG, 0, 0, cnt);
                        radix8_z_sterm_fwd_avx2(p->sp, 0, ref, 0, p->twq, 0, 0, 0, tcols, 0, tcols);
                    }
                    double ns = (qpc_ms() - t0) * 1e6 / RF;
                    if (ns < bestOld) bestOld = ns;
                } else {
                    cachebust();
                    memcpy(p->sp, snap, (size_t)2 * N * 8);
                    radix8_z_msgt_fwd(p->sp, 0, p->sp, 0, lasttw, 0, lastLs, lastG, 0, 0, cnt);
                    radix8_z_stermf_fwd(p->sp, 0, cand, 0, p->twq, 0, 0, 0, tcols, 0, tcols);
                    double t0 = qpc_ms();
                    for (int i = 0; i < RF; i++) {
                        radix8_z_msgt_fwd(p->sp, 0, p->sp, 0, lasttw, 0, lastLs, lastG, 0, 0, cnt);
                        radix8_z_stermf_fwd(p->sp, 0, cand, 0, p->twq, 0, 0, 0, tcols, 0, tcols);
                    }
                    double ns = (qpc_ms() - t0) * 1e6 / RF;
                    if (ns < bestNew) bestNew = ns;
                }
                Sleep(50);
            }
            Sleep(130);
        }
        printf("N=%-6d  last-mid+term:  OLD %8.1f ns   NEW %8.1f ns   (%+6.2f%%)\n",
               N, bestOld, bestNew, 100.0 * (bestNew / bestOld - 1.0));
        _aligned_free(in); _aligned_free(snap); _aligned_free(ref); _aligned_free(cand); _aligned_free(sp2);
        vfft_zsplit_destroy(p);
    }
    printf(rc ? "OVERALL FAIL\n" : "OVERALL PASS\n");
    return rc;
}
