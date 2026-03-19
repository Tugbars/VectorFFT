/**
 * bench_factorize.c — Find optimal factorization for each N
 *
 * For each N, enumerates ALL valid factorizations (using registered radixes),
 * creates a plan for each, benchmarks fwd, and reports the best.
 * Writes results to vfft_wisdom.txt for use by bench_full_fft.
 *
 * Usage:
 *   bench_factorize                    — run default set (same as bench_full_fft)
 *   bench_factorize 1024 4096 8192     — specific sizes only
 *   bench_factorize -o my_wisdom.txt   — custom output path
 *   bench_factorize -q                 — quiet (only summary + wisdom)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef _WIN32
#include <windows.h>
static double get_ns(void) {
    static LARGE_INTEGER freq = {0};
    if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
    LARGE_INTEGER t; QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart * 1e9;
}
#else
#include <time.h>
static double get_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}
#endif

#ifndef VFFT_ISA_LEVEL_DEFINED
#define VFFT_ISA_LEVEL_DEFINED
typedef enum { VFFT_ISA_SCALAR=0, VFFT_ISA_AVX2=1, VFFT_ISA_AVX512=2 } vfft_isa_level_t;
#endif
static inline vfft_isa_level_t vfft_detect_isa(void) {
#if defined(__AVX512F__)
    return VFFT_ISA_AVX512;
#elif defined(__AVX2__)
    return VFFT_ISA_AVX2;
#else
    return VFFT_ISA_SCALAR;
#endif
}

/* ── All dispatch headers ── */
#include "fft_radix2_dispatch.h"
#include "fft_radix3_dispatch.h"
#include "fft_radix4_dispatch.h"
#include "fft_radix5_dispatch.h"
#define vfft_detect_isa _bf_isa_r7
#include "fft_radix7_dispatch.h"
#undef vfft_detect_isa
#include "fft_radix8_dispatch.h"
#include "fft_radix16_dispatch.h"
#define vfft_detect_isa _bf_isa_r32
#include "fft_radix32_dispatch.h"
#undef vfft_detect_isa
#include "fft_radix10_dispatch.h"
#include "fft_radix25_dispatch.h"
#include "fft_radix2_dif_dispatch.h"
#include "fft_radix3_dif_dispatch.h"
#include "fft_radix4_dif_dispatch.h"
#include "fft_radix5_dif_dispatch.h"
#include "fft_radix7_dif_dispatch.h"
#include "fft_radix8_dif_dispatch.h"
#include "fft_radix16_dif_dispatch.h"
#include "fft_radix32_dif_dispatch.h"
#include "fft_radix10_dif_dispatch.h"
#include "fft_radix25_dif_dispatch.h"
#include "fft_radix11_genfft.h"
#include "fft_radix13_genfft.h"
#include "fft_radix17_genfft.h"
#include "fft_radix19_genfft.h"
#include "fft_radix23_genfft.h"
#include "fft_radix64_n1.h"
#include "fft_radix128_n1.h"
#include "vfft_planner.h"
#define vfft_detect_isa _bf_isa_reg
#include "vfft_register_codelets.h"
#undef vfft_detect_isa

#include "vfft_wisdom.h"

/* ═══ Enumerator ═══ */

#define MAX_STAGES 8
#define MAX_FACTS 2048

typedef struct { size_t factors[MAX_STAGES]; size_t nfactors; } fact_t;
static fact_t g_facts[MAX_FACTS];
static size_t g_nfacts;

static const size_t RADIXES[] = {2,3,4,5,7,8,10,11,13,16,17,19,23,25,32,64,128,0};

static void enum_rec(size_t rem, size_t *f, size_t d,
                     const vfft_codelet_registry *reg, size_t minr) {
    if (rem == 1) {
        if (d > 0 && g_nfacts < MAX_FACTS) {
            g_facts[g_nfacts].nfactors = d;
            memcpy(g_facts[g_nfacts].factors, f, d * sizeof(size_t));
            g_nfacts++;
        }
        return;
    }
    for (const size_t *rp = RADIXES; *rp; rp++) {
        size_t r = *rp;
        if (r < minr || r > rem || rem % r || d >= MAX_STAGES) continue;
        if (r >= VFFT_MAX_RADIX || !reg->fwd[r]) continue;
        f[d] = r;
        enum_rec(rem / r, f, d + 1, reg, r);
    }
}

/* ═══ Plan with specific factors ═══ */

static vfft_plan *make_plan(size_t N, const size_t *factors, size_t nf,
                            const vfft_codelet_registry *reg) {
    size_t prod = 1;
    for (size_t i = 0; i < nf; i++) prod *= factors[i];
    if (prod != N) return NULL;

    vfft_plan *p = (vfft_plan *)calloc(1, sizeof(vfft_plan));
    p->N = N; p->nstages = nf;
    size_t K = 1;
    for (size_t s = 0; s < nf; s++) {
        size_t si = nf - 1 - s;
        size_t R = factors[si];
        vfft_stage *st = &p->stages[s];
        st->radix = R; st->K = K; st->N_remaining = N / (R * K);
        if (R < VFFT_MAX_RADIX) {
            st->fwd = reg->fwd[R]; st->bwd = reg->bwd[R];
            st->tw_fwd = reg->tw_fwd[R]; st->tw_bwd = reg->tw_bwd[R];
            st->tw_dif_fwd = reg->tw_dif_fwd[R]; st->tw_dif_bwd = reg->tw_dif_bwd[R];
        }
        if (!st->fwd) { free(p); return NULL; }
        if (K > 1) {
            size_t tsz = (R - 1) * K;
            st->tw_re = (double *)vfft_aligned_alloc(64, tsz * 8);
            st->tw_im = (double *)vfft_aligned_alloc(64, tsz * 8);
            double Na = (double)(R * K);
            for (size_t n = 1; n < R; n++)
                for (size_t k = 0; k < K; k++) {
                    double a = -2.0 * M_PI * (double)(n * k) / Na;
                    st->tw_re[(n-1)*K+k] = cos(a);
                    st->tw_im[(n-1)*K+k] = sin(a);
                }
        }
        K *= R;
    }
    size_t rx[MAX_STAGES];
    for (size_t i = 0; i < nf; i++) rx[i] = p->stages[i].radix;
    p->perm = vfft_build_perm(rx, nf, N);
    p->inv_perm = (size_t *)malloc(N * sizeof(size_t));
    for (size_t i = 0; i < N; i++) p->inv_perm[p->perm[i]] = i;
    p->buf_a_re = (double *)vfft_aligned_alloc(64, N * 8);
    p->buf_a_im = (double *)vfft_aligned_alloc(64, N * 8);
    p->buf_b_re = (double *)vfft_aligned_alloc(64, N * 8);
    p->buf_b_im = (double *)vfft_aligned_alloc(64, N * 8);
    return p;
}

/* ═══ Bench one ═══ */

static double bench1(size_t N, const size_t *f, size_t nf,
                     const vfft_codelet_registry *reg,
                     double *ir, double *ii, double *vr, double *vi) {
    vfft_plan *p = make_plan(N, f, nf, reg);
    if (!p) return -1.0;
    for (int i = 0; i < 5; i++) vfft_execute_fwd(p, ir, ii, vr, vi);
    int reps = N <= 512 ? 5000 : N <= 2048 ? 2000 : N <= 8192 ? 1000 : N <= 32768 ? 200 : 100;
    double best = 1e18;
    for (int t = 0; t < 5; t++) {
        double t0 = get_ns();
        for (int r = 0; r < reps; r++) vfft_execute_fwd(p, ir, ii, vr, vi);
        double ns = (get_ns() - t0) / reps;
        if (ns < best) best = ns;
    }
    vfft_plan_destroy(p);
    return best;
}

static void fstr(const size_t *f, size_t n, char *b, size_t sz) {
    int p = 0;
    for (size_t i = 0; i < n; i++) {
        if (i) p += snprintf(b+p, sz-p, "x");
        p += snprintf(b+p, sz-p, "%zu", f[i]);
    }
}

/* ═══ Default test sizes ═══ */

static const size_t DEFAULT_Ns[] = {
    256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
    320, 448,
    200, 400, 1000, 2000, 5000, 10000,
    224, 896, 1792, 3584,
    88, 704, 5632, 104, 832, 6656, 136, 1088, 152, 1216, 184, 1472,
    80, 640, 4000, 8000,
    800, 20000, 40000,
    0
};

int main(int argc, char **argv) {
    printf("================================================================\n");
    printf("  VectorFFT Factorization Search + Wisdom Generator\n");
    printf("  ISA: ");
#if defined(__AVX512F__)
    printf("AVX-512\n");
#elif defined(__AVX2__)
    printf("AVX2\n");
#else
    printf("Scalar\n");
#endif
    printf("================================================================\n\n");

    vfft_codelet_registry reg;
    vfft_register_all(&reg);

    const char *wpath = "vfft_wisdom.txt";
    size_t cust[256]; size_t nc = 0; int verbose = 1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-o") && i+1 < argc) wpath = argv[++i];
        else if (!strcmp(argv[i], "-q")) verbose = 0;
        else cust[nc++] = (size_t)atol(argv[i]);
    }
    const size_t *Ns = nc ? (cust[nc]=0, cust) : DEFAULT_Ns;

    /* Deduplicate */
    size_t uq[256]; size_t nu = 0;
    for (const size_t *p = Ns; *p; p++) {
        int d = 0;
        for (size_t j = 0; j < nu; j++) if (uq[j] == *p) { d=1; break; }
        if (!d) uq[nu++] = *p;
    }

    vfft_wisdom wis;
    vfft_wisdom_init(&wis);

    for (size_t ni = 0; ni < nu; ni++) {
        size_t N = uq[ni];
        printf("── N = %zu ", N); fflush(stdout);
        g_nfacts = 0;
        { size_t tmp[MAX_STAGES]; enum_rec(N, tmp, 0, &reg, 2); }
        printf("(%zu factorizations) ──\n", g_nfacts);
        if (!g_nfacts) { printf("  (none)\n\n"); continue; }

        double *ir = (double*)vfft_aligned_alloc(64,N*8);
        double *ii = (double*)vfft_aligned_alloc(64,N*8);
        double *vr = (double*)vfft_aligned_alloc(64,N*8);
        double *vi = (double*)vfft_aligned_alloc(64,N*8);
        srand(42+(unsigned)N);
        for (size_t i = 0; i < N; i++) {
            ir[i]=(double)rand()/RAND_MAX*2-1;
            ii[i]=(double)rand()/RAND_MAX*2-1;
        }

        /* Current planner */
        vfft_plan *cp = vfft_plan_create(N, &reg);
        char cfact[128] = "?"; double cns = -1;
        if (cp) {
            int pos = 0;
            for (size_t s = 0; s < cp->nstages; s++) {
                if (s) pos += snprintf(cfact+pos, sizeof(cfact)-pos, "x");
                pos += snprintf(cfact+pos, sizeof(cfact)-pos, "%zu", cp->stages[s].radix);
            }
            for (int w=0;w<5;w++) vfft_execute_fwd(cp,ir,ii,vr,vi);
            int reps = N<=512?5000:N<=2048?2000:N<=8192?1000:N<=32768?200:100;
            cns = 1e18;
            for (int t=0;t<5;t++){double t0=get_ns();
                for(int r=0;r<reps;r++) vfft_execute_fwd(cp,ir,ii,vr,vi);
                double ns=(get_ns()-t0)/reps; if(ns<cns)cns=ns;}
            vfft_plan_destroy(cp);
        }

        /* All factorizations */
        double bns = 1e18; size_t bi = 0;
        double *ans = (double*)calloc(g_nfacts, sizeof(double));
        for (size_t fi = 0; fi < g_nfacts; fi++) {
            ans[fi] = bench1(N, g_facts[fi].factors, g_facts[fi].nfactors, &reg, ir,ii,vr,vi);
            if (ans[fi] > 0 && ans[fi] < bns) { bns = ans[fi]; bi = fi; }
        }

        if (verbose) {
            size_t *ord = (size_t*)malloc(g_nfacts*sizeof(size_t));
            for (size_t i=0;i<g_nfacts;i++) ord[i]=i;
            for (size_t i=1;i<g_nfacts;i++){
                size_t k=ord[i]; double kn=ans[k]>0?ans[k]:1e18; size_t j=i;
                while(j>0&&(ans[ord[j-1]]<=0||ans[ord[j-1]]>kn)){ord[j]=ord[j-1];j--;}
                ord[j]=k;
            }
            size_t show = g_nfacts < 15 ? g_nfacts : 15;
            printf("  %-4s %-28s %9s %7s\n","rank","factorization","ns","ratio");
            printf("  %-4s %-28s %9s %7s\n","----","----------------------------","---------","-------");
            for (size_t i=0;i<show;i++){
                size_t fi=ord[i]; char lb[128];
                fstr(g_facts[fi].factors,g_facts[fi].nfactors,lb,sizeof(lb));
                if(ans[fi]>0) printf("  %-4zu %-28s %9.0f %6.2fx%s\n",
                    i+1,lb,ans[fi],ans[fi]/bns,fi==bi?" ***":"");
                else printf("  %-4zu %-28s     FAIL\n",i+1,lb);
            }
            if(g_nfacts>show) printf("  ... (%zu more)\n",g_nfacts-show);
            free(ord);
        }

        char bl[128]; fstr(g_facts[bi].factors,g_facts[bi].nfactors,bl,sizeof(bl));
        printf("\n  PLANNER:  %-24s %9.0f ns\n", cfact, cns);
        printf("  OPTIMAL:  %-24s %9.0f ns", bl, bns);
        if(cns>0&&bns>0&&cns>bns*1.05) printf("  (%.1fx faster!)",cns/bns);
        else if(cns>0&&bns>0) printf("  (planner OK)");
        printf("\n\n");

        /* Wisdom stores inner→outer (same as fact.factors[]/stages[]),
         * enum stores outer→inner (ascending). Reverse. */
        {
            size_t rev[MAX_STAGES];
            size_t nf = g_facts[bi].nfactors;
            for (size_t i = 0; i < nf; i++) rev[i] = g_facts[bi].factors[nf-1-i];
            vfft_wisdom_set(&wis, N, rev, nf, bns);
        }
        free(ans);
        vfft_aligned_free(ir); vfft_aligned_free(ii);
        vfft_aligned_free(vr); vfft_aligned_free(vi);
    }

    if (vfft_wisdom_save(&wis, wpath) == 0) {
        printf("================================================================\n");
        printf("  Wisdom written to: %s (%zu entries)\n", wpath, wis.count);
        printf("================================================================\n");
    } else {
        printf("  ERROR: failed to write %s\n", wpath);
    }

    vfft_wisdom_destroy(&wis);
    return 0;
}
