/* zil_sterm_pipe.c — sterm SOFTWARE-PIPELINING race (z_cascade_plan §4.9993).
 *
 * VTune (§4.9991): sterm is the stall mass — CPI 0.371, 44% retiring, store
 * latency + FB-full + L1-latency dominant; 16+ live ymm ⇒ gcc spills. This
 * bench races scheduling-only variants of the terminator inside the REAL
 * production cascade (zsplit.h plan + s0s/msg front), all gated BIT-IDENTICAL
 * against the emitted baseline before timing.
 *
 * Arms: emit  = linked emitted radix8_z_sterm_fwd_avx2 (control)
 *       copy  = same source pasted in this TU (TU/flag-parity control)
 *       rot   = cross-iteration rotation of the twiddle squaring tree
 *       phase = intra-iteration live-range minimization (anti-spill)
 *       pfw   = PREFETCHW on the 8 output RFO streams (+1 input line)
 *       uj2   = 2-quad unroll-and-jam (ILP/MLP hypothesis test)
 *       nt    = non-temporal full-line stores (informational: RFO killer;
 *               needs aligned zout — bench allocs are 64B-aligned)
 *
 * Discipline (canonical_mkl_bench): pin logical core 2 (mask 4), HIGH prio,
 * 32MB cachebust per measurement, arm order rotated per round, best-of-R,
 * Sleep pacing. Timings: FULL cascade (ground truth) + terminator-only
 * (resolution; input = plan->sp left hot by a front pass, like real life).
 *
 * Build:  python build.py --src benches/zil_sterm_pipe.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include <immintrin.h>

#include "zsplit.h"

/* ---- macros: verbatim from radix8_z_sterm_avx2.c (emitter-owned) ---- */
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

/* ---- arm "copy": baseline body pasted (TU/flag parity control) ---- */
__attribute__((target("avx2,fma")))
static void sterm_fwd_copy(
    const double * __restrict__ zin, const double * __restrict__ zu,
    double * __restrict__ zout, double * __restrict__ zou,
    const double * tw_re, const double * tw_im,
    unsigned long long Ls, unsigned long long Gs, unsigned long long OLs,
    unsigned long long OGs, unsigned long long count)
{
    (void)zu; (void)zou; (void)tw_im; (void)Ls; (void)Gs; (void)OGs;
    for (size_t k = 0; k + 4 <= count; k += 4) {
        __m256d xr[8], xi[8];
        {
            __m256d rl0 = _mm256_loadu_pd(zin + 16*(size_t)k);
            __m256d il0 = _mm256_loadu_pd(zin + 16*(size_t)k + 4);
            __m256d rh0 = _mm256_loadu_pd(zin + 16*(size_t)k + 8);
            __m256d ih0 = _mm256_loadu_pd(zin + 16*(size_t)k + 12);
            __m256d rl1 = _mm256_loadu_pd(zin + 16*((size_t)k+1));
            __m256d il1 = _mm256_loadu_pd(zin + 16*((size_t)k+1) + 4);
            __m256d rh1 = _mm256_loadu_pd(zin + 16*((size_t)k+1) + 8);
            __m256d ih1 = _mm256_loadu_pd(zin + 16*((size_t)k+1) + 12);
            __m256d rl2 = _mm256_loadu_pd(zin + 16*((size_t)k+2));
            __m256d il2 = _mm256_loadu_pd(zin + 16*((size_t)k+2) + 4);
            __m256d rh2 = _mm256_loadu_pd(zin + 16*((size_t)k+2) + 8);
            __m256d ih2 = _mm256_loadu_pd(zin + 16*((size_t)k+2) + 12);
            __m256d rl3 = _mm256_loadu_pd(zin + 16*((size_t)k+3));
            __m256d il3 = _mm256_loadu_pd(zin + 16*((size_t)k+3) + 4);
            __m256d rh3 = _mm256_loadu_pd(zin + 16*((size_t)k+3) + 8);
            __m256d ih3 = _mm256_loadu_pd(zin + 16*((size_t)k+3) + 12);
            TR4(rl0, rl1, rl2, rl3, xr[0], xr[1], xr[2], xr[3]);
            TR4(il0, il1, il2, il3, xi[0], xi[1], xi[2], xi[3]);
            TR4(rh0, rh1, rh2, rh3, xr[4], xr[5], xr[6], xr[7]);
            TR4(ih0, ih1, ih2, ih3, xi[4], xi[5], xi[6], xi[7]);
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

/* ===================== WORKFLOW ARMS PASTED BELOW ===================== */
/* ARMS_PLACEHOLDER */
/* ===================== WORKFLOW ARMS PASTED ABOVE ===================== */

/* ---- cascade helpers: mirror vfft_zsplit_execute_fwd with swappable term */
static void run_front(const vfft_zsplit_plan_t *p, const double *zin)
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
    for (int s = 1; s <= nf - 2; s++) {
        zfn f = (p->chain[s] == 8) ? radix8_z_msg_fwd_avx2
                                   : radix4_z_msg_fwd_avx2;
        f(p->sp, 0, p->sp, 0, p->twsp[s], 0,
          (unsigned long long)p->D[s], (unsigned long long)p->G[s],
          0, 0, (unsigned long long)p->D[s]);
    }
}
static void exec_term(const vfft_zsplit_plan_t *p, double *zout, zfn term)
{
    term(p->sp, 0, zout, 0, p->twq, 0, 0, 0,
         (unsigned long long)(p->N / 8), 0, (unsigned long long)(p->N / 8));
}

/* ---- harness ---- */
static double qpc_ms(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return 1000.0 * (double)c.QuadPart / (double)f.QuadPart;
}
static char *g_bust;
#define BUST_SZ (32u * 1024u * 1024u)
static void cachebust(void)
{
    for (size_t i = 0; i < BUST_SZ; i += 64) g_bust[i]++;
}

typedef struct { const char *name; zfn fn; } arm_t;

int main(void)
{
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    g_bust = (char *)malloc(BUST_SZ);
    memset(g_bust, 1, BUST_SZ);

    arm_t arms[] = {
        { "emit",  radix8_z_sterm_fwd_avx2 },
        { "copy",  sterm_fwd_copy },
        /* ARM_TABLE_PLACEHOLDER */
    };
    const int NA = (int)(sizeof(arms) / sizeof(arms[0]));
    const int cells[] = { 2048, 4096, 8192, 16384 };
    const int reps_full[] = { 300, 150, 80, 40 };
    const int ROUNDS = 7;
    int rc = 0;

    for (int ci = 0; ci < 4; ci++) {
        const int N = cells[ci];
        int chain[VFFT_ZSPLIT_MAX_NF];
        int nf = vfft_zsplit_default_chain(N, chain);
        vfft_zsplit_plan_t *p = vfft_zsplit_create(N, chain, nf);
        if (!p) { printf("N=%d create FAIL\n", N); return 1; }

        double *in  = (double *)_aligned_malloc((size_t)2 * N * 8, 64);
        double *ref = (double *)_aligned_malloc((size_t)2 * N * 8, 64);
        double *out = (double *)_aligned_malloc((size_t)2 * N * 8, 64);
        srand(20260725 + N);
        for (int i = 0; i < 2 * N; i++)
            in[i] = (double)rand() / RAND_MAX - 0.5;

        /* gates: every arm bit-identical to the production fwd */
        vfft_zsplit_execute_fwd(p, in, ref);
        for (int a = 0; a < NA; a++) {
            memset(out, 0, (size_t)2 * N * 8);
            run_front(p, in);
            exec_term(p, out, arms[a].fn);
            if (memcmp(out, ref, (size_t)2 * N * 8) != 0) {
                printf("N=%-6d GATE %-5s BIT-MISMATCH FAIL\n", N, arms[a].name);
                rc = 1;
            } else {
                printf("N=%-6d GATE %-5s bit-identical PASS\n", N, arms[a].name);
            }
        }
        if (rc) { printf("gates failed at N=%d — not timing\n", N); return rc; }

        /* race: full cascade + terminator-only, rotated arms, best-of-ROUNDS */
        double best_full[16], best_term[16];
        for (int a = 0; a < NA; a++) { best_full[a] = 1e30; best_term[a] = 1e30; }
        const int RF = reps_full[ci], RT = RF * 3;
        for (int r = 0; r < ROUNDS; r++) {
            for (int j = 0; j < NA; j++) {
                int a = (j + r) % NA;               /* order-neutral rotation */
                cachebust();
                run_front(p, in);                    /* warm + fresh sp */
                exec_term(p, out, arms[a].fn);
                double t0 = qpc_ms();
                for (int i = 0; i < RF; i++) {
                    run_front(p, in);
                    exec_term(p, out, arms[a].fn);
                }
                double ns = (qpc_ms() - t0) * 1e6 / RF;
                if (ns < best_full[a]) best_full[a] = ns;

                cachebust();
                run_front(p, in);                    /* sp hot, like real life */
                exec_term(p, out, arms[a].fn);
                t0 = qpc_ms();
                for (int i = 0; i < RT; i++)
                    exec_term(p, out, arms[a].fn);
                ns = (qpc_ms() - t0) * 1e6 / RT;
                if (ns < best_term[a]) best_term[a] = ns;
                Sleep(60);
            }
            Sleep(150);
        }
        for (int a = 0; a < NA; a++)
            printf("N=%-6d %-5s full %9.1f ns (%+6.2f%%)   term %8.1f ns (%+6.2f%%)\n",
                   N, arms[a].name,
                   best_full[a], 100.0 * (best_full[a] / best_full[0] - 1.0),
                   best_term[a], 100.0 * (best_term[a] / best_term[0] - 1.0));
        _aligned_free(in); _aligned_free(ref); _aligned_free(out);
        vfft_zsplit_destroy(p);
    }
    printf(rc ? "OVERALL FAIL\n" : "OVERALL PASS\n");
    return rc;
}
