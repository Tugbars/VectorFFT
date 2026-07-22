/* bench_k1_vs_mkl.c — K=1 single-transform: BAILEY2V plan API vs MKL, same
 * process, canonical methodology (modeled on bench_1d_vs_mkl.c: pinned core 2,
 * HIGH prio, per-arm hot warmup inside each trial, best-of-5 with 32MB
 * cachebust between trials, ORDER FLIPPED per trial to neutralize thermal /
 * cache ordering bias). Pair selection = quick pre-pass sweep per (N, layout)
 * — mimics create-time adoption; the measured table then uses the chosen pair.
 *
 * Arms per N: MKL-IL (in-place CCE), MKL-split (REAL_REAL), B2V-split (OOP
 * x->d), B2V-IL (z->z), LEAF (N<=128, today's production route), MONO64
 * split+IL (N=64, the P3 tier's hand reference).
 * Speedups reported as MKL/ours (>1 = we win, v1_0 convention).
 *
 * Build: python build.py --src benches/bench_k1_vs_mkl.c --mkl
 * Run:   bench_k1_vs_mkl [N ...]   (default 64 128 256 512 1024 2048 4096 8192)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include <immintrin.h>
#include <mkl_dfti.h>

#include "executor.h"
#include "planner.h"
#include "oop_plan.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double now_ms(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return 1000.0 * (double)c.QuadPart / (double)f.QuadPart;
}
static void cachebust(void)
{
    size_t s = 32u * 1024u * 1024u / 8u;
    double *j = (double *)malloc(s * 8);
    volatile double a = 0;
    for (size_t i = 0; i < s; i++) j[i] = (double)i * 0.5;
    for (size_t i = 0; i < s; i++) a += j[i];
    (void)a; free(j);
}
static double *ad(size_t n)
{
    double *p = NULL;
    if (vfft_proto_posix_memalign((void **)&p, 64, n * sizeof(double)) != 0) exit(1);
    return p;
}

/* ---- mono64 (P3 hand reference, split + IL) — from k1_fourstep_spike.c ---- */
static inline void dft8v(__m256d *re, __m256d *im)
{
    const __m256d C = _mm256_set1_pd(0.70710678118654752440);
    const __m256d CN = _mm256_set1_pd(-0.70710678118654752440);
    __m256d t0r = _mm256_add_pd(re[0], re[4]), t0i = _mm256_add_pd(im[0], im[4]);
    __m256d t1r = _mm256_sub_pd(re[0], re[4]), t1i = _mm256_sub_pd(im[0], im[4]);
    __m256d t2r = _mm256_add_pd(re[2], re[6]), t2i = _mm256_add_pd(im[2], im[6]);
    __m256d t3r = _mm256_sub_pd(re[2], re[6]), t3i = _mm256_sub_pd(im[2], im[6]);
    __m256d E0r = _mm256_add_pd(t0r, t2r), E0i = _mm256_add_pd(t0i, t2i);
    __m256d E2r = _mm256_sub_pd(t0r, t2r), E2i = _mm256_sub_pd(t0i, t2i);
    __m256d E1r = _mm256_add_pd(t1r, t3i), E1i = _mm256_sub_pd(t1i, t3r);
    __m256d E3r = _mm256_sub_pd(t1r, t3i), E3i = _mm256_add_pd(t1i, t3r);
    __m256d s0r = _mm256_add_pd(re[1], re[5]), s0i = _mm256_add_pd(im[1], im[5]);
    __m256d s1r = _mm256_sub_pd(re[1], re[5]), s1i = _mm256_sub_pd(im[1], im[5]);
    __m256d s2r = _mm256_add_pd(re[3], re[7]), s2i = _mm256_add_pd(im[3], im[7]);
    __m256d s3r = _mm256_sub_pd(re[3], re[7]), s3i = _mm256_sub_pd(im[3], im[7]);
    __m256d O0r = _mm256_add_pd(s0r, s2r), O0i = _mm256_add_pd(s0i, s2i);
    __m256d O2r = _mm256_sub_pd(s0r, s2r), O2i = _mm256_sub_pd(s0i, s2i);
    __m256d O1r = _mm256_add_pd(s1r, s3i), O1i = _mm256_sub_pd(s1i, s3r);
    __m256d O3r = _mm256_sub_pd(s1r, s3i), O3i = _mm256_add_pd(s1i, s3r);
    __m256d W1r = _mm256_mul_pd(C, _mm256_add_pd(O1r, O1i));
    __m256d W1i = _mm256_mul_pd(C, _mm256_sub_pd(O1i, O1r));
    __m256d W2r = O2i, W2i = _mm256_sub_pd(_mm256_setzero_pd(), O2r);
    __m256d W3r = _mm256_mul_pd(C, _mm256_sub_pd(O3i, O3r));
    __m256d W3i = _mm256_mul_pd(CN, _mm256_add_pd(O3r, O3i));
    re[0] = _mm256_add_pd(E0r, O0r); im[0] = _mm256_add_pd(E0i, O0i);
    re[4] = _mm256_sub_pd(E0r, O0r); im[4] = _mm256_sub_pd(E0i, O0i);
    re[1] = _mm256_add_pd(E1r, W1r); im[1] = _mm256_add_pd(E1i, W1i);
    re[5] = _mm256_sub_pd(E1r, W1r); im[5] = _mm256_sub_pd(E1i, W1i);
    re[2] = _mm256_add_pd(E2r, W2r); im[2] = _mm256_add_pd(E2i, W2i);
    re[6] = _mm256_sub_pd(E2r, W2r); im[6] = _mm256_sub_pd(E2i, W2i);
    re[3] = _mm256_add_pd(E3r, W3r); im[3] = _mm256_add_pd(E3i, W3i);
    re[7] = _mm256_sub_pd(E3r, W3r); im[7] = _mm256_sub_pd(E3i, W3i);
}
#define T4(r0, r1, r2, r3, o0, o1, o2, o3) do {                          \
    __m256d _u0 = _mm256_unpacklo_pd(r0, r1), _u1 = _mm256_unpackhi_pd(r0, r1); \
    __m256d _u2 = _mm256_unpacklo_pd(r2, r3), _u3 = _mm256_unpackhi_pd(r2, r3); \
    o0 = _mm256_permute2f128_pd(_u0, _u2, 0x20);                          \
    o1 = _mm256_permute2f128_pd(_u1, _u3, 0x20);                          \
    o2 = _mm256_permute2f128_pd(_u0, _u2, 0x31);                          \
    o3 = _mm256_permute2f128_pd(_u1, _u3, 0x31);                          \
} while (0)
static double M64TWR[2][8][4], M64TWI[2][8][4];
static void m64_init(void)
{
    for (int h = 0; h < 2; h++)
        for (int m = 0; m < 8; m++)
            for (int j = 0; j < 4; j++) {
                double a = -2.0 * M_PI * (double)(m * (4 * h + j)) / 64.0;
                M64TWR[h][m][j] = cos(a); M64TWI[h][m][j] = sin(a);
            }
}
static void mono64_core(__m256d Ur[2][8], __m256d Ui[2][8], double *dr, double *di,
                        double *z_out)
{
    __m256d Vr[2][8], Vi[2][8];
    for (int h = 0; h < 2; h++)
        for (int mh = 0; mh < 2; mh++) {
            T4(Ur[h][4*mh+0], Ur[h][4*mh+1], Ur[h][4*mh+2], Ur[h][4*mh+3],
               Vr[mh][4*h+0], Vr[mh][4*h+1], Vr[mh][4*h+2], Vr[mh][4*h+3]);
            T4(Ui[h][4*mh+0], Ui[h][4*mh+1], Ui[h][4*mh+2], Ui[h][4*mh+3],
               Vi[mh][4*h+0], Vi[mh][4*h+1], Vi[mh][4*h+2], Vi[mh][4*h+3]);
        }
    for (int mh = 0; mh < 2; mh++) {
        dft8v(Vr[mh], Vi[mh]);
        for (int jp = 0; jp < 8; jp++) {
            if (z_out) {
                __m256d lo = _mm256_unpacklo_pd(Vr[mh][jp], Vi[mh][jp]);
                __m256d hi = _mm256_unpackhi_pd(Vr[mh][jp], Vi[mh][jp]);
                _mm256_storeu_pd(z_out + 2*((size_t)jp*8 + 4*mh), _mm256_permute2f128_pd(lo, hi, 0x20));
                _mm256_storeu_pd(z_out + 2*((size_t)jp*8 + 4*mh) + 4, _mm256_permute2f128_pd(lo, hi, 0x31));
            } else {
                _mm256_storeu_pd(dr + (size_t)jp*8 + 4*mh, Vr[mh][jp]);
                _mm256_storeu_pd(di + (size_t)jp*8 + 4*mh, Vi[mh][jp]);
            }
        }
    }
}
static void mono64_split(const double *xr, const double *xi, double *dr, double *di)
{
    __m256d Ur[2][8], Ui[2][8];
    for (int h = 0; h < 2; h++) {
        __m256d r[8], q[8];
        for (int j = 0; j < 8; j++) {
            r[j] = _mm256_loadu_pd(xr + (size_t)j*8 + 4*h);
            q[j] = _mm256_loadu_pd(xi + (size_t)j*8 + 4*h);
        }
        dft8v(r, q);
        for (int m = 1; m < 8; m++) {
            __m256d cr = _mm256_loadu_pd(M64TWR[h][m]), ci = _mm256_loadu_pd(M64TWI[h][m]);
            __m256d zr = _mm256_sub_pd(_mm256_mul_pd(r[m], cr), _mm256_mul_pd(q[m], ci));
            __m256d zi = _mm256_add_pd(_mm256_mul_pd(r[m], ci), _mm256_mul_pd(q[m], cr));
            r[m] = zr; q[m] = zi;
        }
        for (int m = 0; m < 8; m++) { Ur[h][m] = r[m]; Ui[h][m] = q[m]; }
    }
    mono64_core(Ur, Ui, dr, di, NULL);
}
static void mono64_il(const double *z_in, double *z_out)
{
    __m256d Ur[2][8], Ui[2][8];
    for (int h = 0; h < 2; h++) {
        __m256d r[8], q[8];
        for (int j = 0; j < 8; j++) {
            __m256d za = _mm256_loadu_pd(z_in + 2*((size_t)j*8 + 4*h));
            __m256d zb = _mm256_loadu_pd(z_in + 2*((size_t)j*8 + 4*h) + 4);
            r[j] = _mm256_permute4x64_pd(_mm256_unpacklo_pd(za, zb), 0xD8);
            q[j] = _mm256_permute4x64_pd(_mm256_unpackhi_pd(za, zb), 0xD8);
        }
        dft8v(r, q);
        for (int m = 1; m < 8; m++) {
            __m256d cr = _mm256_loadu_pd(M64TWR[h][m]), ci = _mm256_loadu_pd(M64TWI[h][m]);
            __m256d zr = _mm256_sub_pd(_mm256_mul_pd(r[m], cr), _mm256_mul_pd(q[m], ci));
            __m256d zi = _mm256_add_pd(_mm256_mul_pd(r[m], ci), _mm256_mul_pd(q[m], cr));
            r[m] = zr; q[m] = zi;
        }
        for (int m = 0; m < 8; m++) { Ur[h][m] = r[m]; Ui[h][m] = q[m]; }
    }
    mono64_core(Ur, Ui, NULL, NULL, z_out);
}

/* ---------------- arms ---------------- */
enum { A_MKL_IL = 0, A_MKL_SP, A_B2V_SP, A_B2V_IP, A_B2V_IL, A_2PA_IP, A_2PB_IP,
       A_TWL_IP, A_LEAF, A_M64_SP, A_M64_IL, A_K1M, NARMS };
static const char *ANAME[NARMS] = { "MKL-IL", "MKL-sp", "B2V-sp", "B2V-ip", "B2V-il",
                                    "2pa-ip", "2pb-ip", "twl-ip", "LEAF", "M64-sp", "M64-il",
                                    "K1M-emit" };

typedef struct {
    int N;
    DFTI_DESCRIPTOR_HANDLE hi, hs;
    vfft_oop_plan_t *psp, *pip, *pil, *p2a, *p2b;   /* chosen pairs per arm */
    vfft_oop11_fn leafN;
    double *xr, *xi, *dr, *di, *zi, *zs;  /* zi = MKL IL buf, zs = ours IL buf */
    double *mre, *mim;                     /* MKL split bufs */
    double *wr, *wi;                       /* in-place arm's own evolving bufs */
} bctx_t;

static void run_arm(bctx_t *c, int a)
{
    switch (a) {
    case A_MKL_IL: DftiComputeForward(c->hi, c->zi); break;
    case A_MKL_SP: DftiComputeForward(c->hs, c->mre, c->mim); break;
    case A_B2V_SP: vfft_oop_execute_fwd(c->psp, c->xr, c->xi, c->dr, c->di); break;
    case A_B2V_IP: /* in-place: d == x (leaf drains x into scratch first) */
        vfft_oop_execute_fwd(c->pip, c->wr, c->wi, c->wr, c->wi); break;
    case A_B2V_IL: vfft_oop_execute_fwd_il(c->pil, c->zs, c->zs); break;
    case A_2PA_IP: /* two-pass route a, in-place: leaf w->scr, t1-UL scr->w */
        vfft_oop_execute_fwd_2pa(c->p2a, c->wr, c->wi, c->wr, c->wi); break;
    case A_2PB_IP: /* two-pass route b, in-place: leaf-UL w->scr, t1 scr->w */
        vfft_oop_execute_fwd_2pb(c->p2b, c->wr, c->wi, c->wr, c->wi); break;
    case A_TWL_IP: /* route a + LINEAR twiddle stream (one cursor) */
        vfft_oop_execute_fwd_2pa_twl(c->p2a, c->wr, c->wi, c->wr, c->wi); break;
    case A_LEAF:   c->leafN(c->xr, c->xi, c->dr, c->di, 0, 0, 1, 1, 1, 1, 1); break;
    case A_M64_SP: mono64_split(c->xr, c->xi, c->dr, c->di); break;
    case A_M64_IL: mono64_il(c->zs, c->zs); break;
    case A_K1M: {  /* emitted mono (M1) — same shape as the hand kernel */
        vfft_oop11_fn f = vfft_k1_mono_fn(c->N);
        f(c->xr, c->xi, c->dr, c->di, 0, 0, 0, 0, 0, 0, 0);
        break; }
    }
}

static int arm_avail(bctx_t *c, int a)
{
    switch (a) {
    case A_MKL_IL: return c->hi != NULL;
    case A_MKL_SP: return c->hs != NULL;
    case A_B2V_SP: return c->psp != NULL;
    case A_B2V_IP: return c->pip != NULL;
    case A_B2V_IL: return c->pil && c->pil->il_leaf && c->pil->t1_il;
    case A_2PA_IP: return c->p2a && c->p2a->t1_ul;
    case A_2PB_IP: return c->p2b && c->p2b->leaf_ul;
    case A_TWL_IP: return c->p2a && c->p2a->t1_ul_twl;
    case A_LEAF:   return c->leafN != NULL;
    case A_M64_SP: case A_M64_IL: return c->N == 64;
    case A_K1M:    return vfft_k1_mono_fn(c->N) != 0;
    }
    return 0;
}

/* time one arm hot: 10 warmup + reps timed; returns ns/exec */
static double time_arm(bctx_t *c, int a, int reps)
{
    for (int w = 0; w < 10; w++) run_arm(c, a);
    double t0 = now_ms();
    for (int i = 0; i < reps; i++) run_arm(c, a);
    return (now_ms() - t0) * 1e6 / reps;
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    m64_init();
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg); (void)reg;

    int Nd[] = { 64, 128, 256, 512, 1024, 2048, 4096, 8192 };
    int nN = argc > 1 ? argc - 1 : 8;

    printf("# K=1 plan-API vs MKL — same process, order-flipped best-of-5, cachebust between trials\n");
    printf("# speedups = MKL/ours (>1 = we win)\n");
    /* NOTE: the pair pre-pass sweeps candidates in FIXED R2-descending order —
     * first-tried pairs run coolest and win ties (the cmp_old_new fixed-order
     * bias). Good enough for a bench snapshot; the production pair choice is
     * the calibrator (isolated per-candidate, order-neutralized) writing
     * per-cell (pair x placement x layout) wisdom. */
    printf("%-6s %9s %9s %13s %13s %13s %13s %9s %13s | %7s %7s\n",
           "N", "MKL-IL", "MKL-sp", "B2V-sp", "B2V-ip", "2pa-ip", "2pb-ip", "twl-ip",
           "B2V-il", "sp/sp", "best/IL");

    for (int ni = 0; ni < nN; ni++) {
        int N = argc > 1 ? atoi(argv[ni + 1]) : Nd[ni];
        bctx_t c; memset(&c, 0, sizeof c);
        c.N = N;
        c.xr = ad(N); c.xi = ad(N); c.dr = ad(N); c.di = ad(N);
        c.zi = ad((size_t)2 * N); c.zs = ad((size_t)2 * N);
        c.mre = ad(N); c.mim = ad(N);
        c.wr = ad(N); c.wi = ad(N);
        srand(42 + N);
        for (int n = 0; n < N; n++) {
            c.xr[n] = (double)rand() / RAND_MAX - 0.5;
            c.xi[n] = (double)rand() / RAND_MAX - 0.5;
            c.zi[2*n] = c.zs[2*n] = c.xr[n];
            c.zi[2*n+1] = c.zs[2*n+1] = c.xi[n];
            c.mre[n] = c.xr[n]; c.mim[n] = c.xi[n];
            c.wr[n] = c.xr[n]; c.wi[n] = c.xi[n];
        }
        DftiCreateDescriptor(&c.hi, DFTI_DOUBLE, DFTI_COMPLEX, 1, N);
        DftiCommitDescriptor(c.hi);
        DftiCreateDescriptor(&c.hs, DFTI_DOUBLE, DFTI_COMPLEX, 1, N);
        DftiSetValue(c.hs, DFTI_COMPLEX_STORAGE, DFTI_REAL_REAL);
        DftiCommitDescriptor(c.hs);
        c.leafN = (N <= 128) ? vfft_oop_leaf_fn(N) : NULL;

        int reps = (int)(2e6 / (double)N);
        if (reps < 100) reps = 100;
        if (reps > 400000) reps = 400000;

        /* pair pre-pass: per (placement/layout), best-of-2 at FULL reps with a
         * cachebust between (a quarter-reps single-shot mispicked pairs at
         * 4096 — the in-tree wisdom/tuner replaces this eventually). */
        int bs1 = 0, bs2 = 0, bp1 = 0, bp2 = 0, bi1 = 0, bi2 = 0;
        int ba1 = 0, ba2 = 0, bb1 = 0, bb2 = 0;
        double bsns = 1e18, bpns = 1e18, bins = 1e18, bans = 1e18, bbns = 1e18;
        for (int R2 = (N < 128 ? N : 128); R2 >= 4; R2--) {
            if (N % R2) continue;
            int R1 = N / R2;
            if (R1 < 4 || R1 > 128 || (R1 % 4) || (R2 % 4)) continue;
            vfft_oop_plan_t *p = vfft_oop_plan_create_k1(N, R1, R2);
            if (!p) continue;
            c.psp = p; c.pip = p; c.pil = p; c.p2a = p; c.p2b = p;
            double t_sp = 1e18, t_ip = 1e18, t_il = 1e18, t_2a = 1e18, t_2b = 1e18;
            for (int t = 0; t < 2; t++) {
                if (t) cachebust();
                double v = time_arm(&c, A_B2V_SP, reps);
                if (v < t_sp) t_sp = v;
                v = time_arm(&c, A_B2V_IP, reps);
                if (v < t_ip) t_ip = v;
                if (p->il_leaf && p->t1_il) {
                    v = time_arm(&c, A_B2V_IL, reps);
                    if (v < t_il) t_il = v;
                }
                if (p->t1_ul) {
                    v = time_arm(&c, A_2PA_IP, reps);
                    if (v < t_2a) t_2a = v;
                }
                if (p->leaf_ul) {
                    v = time_arm(&c, A_2PB_IP, reps);
                    if (v < t_2b) t_2b = v;
                }
            }
            if (t_sp < bsns) { bsns = t_sp; bs1 = R1; bs2 = R2; }
            if (t_ip < bpns) { bpns = t_ip; bp1 = R1; bp2 = R2; }
            if (t_il < bins) { bins = t_il; bi1 = R1; bi2 = R2; }
            if (t_2a < bans) { bans = t_2a; ba1 = R1; ba2 = R2; }
            if (t_2b < bbns) { bbns = t_2b; bb1 = R1; bb2 = R2; }
            vfft_oop_plan_destroy(p);
        }
        c.psp = bs1 ? vfft_oop_plan_create_k1(N, bs1, bs2) : NULL;
        c.pip = bp1 ? vfft_oop_plan_create_k1(N, bp1, bp2) : NULL;
        c.pil = bi1 ? vfft_oop_plan_create_k1(N, bi1, bi2) : NULL;
        c.p2a = ba1 ? vfft_oop_plan_create_k1(N, ba1, ba2) : NULL;
        c.p2b = bb1 ? vfft_oop_plan_create_k1(N, bb1, bb2) : NULL;

        /* measured table: best-of-5, order flipped per trial, cachebust between */
        double best[NARMS];
        for (int a = 0; a < NARMS; a++) best[a] = 1e18;
        for (int t = 0; t < 5; t++) {
            if (t) cachebust();
            for (int k = 0; k < NARMS; k++) {
                int a = (t & 1) ? (NARMS - 1 - k) : k;
                if (!arm_avail(&c, a)) continue;
                double ns = time_arm(&c, a, reps);
                if (ns < best[a]) best[a] = ns;
            }
        }

        char sp[32], ip[32], il[32], a2[32], b2[32], tl[32];
        snprintf(sp, sizeof sp, "%.0f(%dx%d)", best[A_B2V_SP], bs1, bs2);
        snprintf(ip, sizeof ip, "%.0f(%dx%d)", best[A_B2V_IP], bp1, bp2);
        snprintf(il, sizeof il, arm_avail(&c, A_B2V_IL) ? "%.0f(%dx%d)" : "-", best[A_B2V_IL], bi1, bi2);
        snprintf(a2, sizeof a2, arm_avail(&c, A_2PA_IP) ? "%.0f(%dx%d)" : "-", best[A_2PA_IP], ba1, ba2);
        snprintf(b2, sizeof b2, arm_avail(&c, A_2PB_IP) ? "%.0f(%dx%d)" : "-", best[A_2PB_IP], bb1, bb2);
        snprintf(tl, sizeof tl, arm_avail(&c, A_TWL_IP) ? "%.0f" : "-", best[A_TWL_IP]);
        double ours_best_sp = best[A_B2V_SP] < best[A_B2V_IP] ? best[A_B2V_SP] : best[A_B2V_IP];
        if (arm_avail(&c, A_2PA_IP) && best[A_2PA_IP] < ours_best_sp) ours_best_sp = best[A_2PA_IP];
        if (arm_avail(&c, A_2PB_IP) && best[A_2PB_IP] < ours_best_sp) ours_best_sp = best[A_2PB_IP];
        if (arm_avail(&c, A_TWL_IP) && best[A_TWL_IP] < ours_best_sp) ours_best_sp = best[A_TWL_IP];
        double ours_best_il = best[A_B2V_IL];
        if (N == 64) {
            if (best[A_M64_SP] < ours_best_sp) ours_best_sp = best[A_M64_SP];
            if (best[A_M64_IL] < ours_best_il) ours_best_il = best[A_M64_IL];
        }
        printf("%-6d %9.1f %9.1f %13s %13s %13s %13s %9s %13s | %7.2f %7.2f\n",
               N, best[A_MKL_IL], best[A_MKL_SP], sp, ip, a2, b2, tl, il,
               best[A_MKL_SP] / ours_best_sp,
               arm_avail(&c, A_B2V_IL) || N == 64 ? best[A_MKL_IL] / ours_best_il : 0.0);
        if (arm_avail(&c, A_LEAF) || N == 64 || arm_avail(&c, A_K1M))
            printf("       LEAF=%.0f  M64-sp=%.0f  M64-il=%.0f  K1M-emit=%.0f\n",
                   arm_avail(&c, A_LEAF) ? best[A_LEAF] : 0.0,
                   N == 64 ? best[A_M64_SP] : 0.0,
                   N == 64 ? best[A_M64_IL] : 0.0,
                   arm_avail(&c, A_K1M) ? best[A_K1M] : 0.0);

        DftiFreeDescriptor(&c.hi); DftiFreeDescriptor(&c.hs);
        if (c.psp) vfft_oop_plan_destroy(c.psp);
        if (c.pip) vfft_oop_plan_destroy(c.pip);
        if (c.pil) vfft_oop_plan_destroy(c.pil);
        if (c.p2a) vfft_oop_plan_destroy(c.p2a);
        if (c.p2b) vfft_oop_plan_destroy(c.p2b);
    }
    printf("\nDONE\n");
    return 0;
}
