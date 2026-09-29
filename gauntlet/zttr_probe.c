/* zttr_probe.c -- ZTURN-T REAL (il/real/zttr.h): the gate and the stage
 * timing. For each even N (real) the legal ZTT chains of M = N/2 are built,
 * gated against a long double DFT, timed (untiled, and at the tile widths
 * 512 / 1024 / 2048 complexes), and set beside the front door's zr2c plan at
 * N (the store handed in argv[1], its banked route replayed) and its parts
 * (the ZTT c2c at M through vfft_ztt_execute_fwd + the fold).
 * Build: python gauntlet/build.py --compile --vfft --src gauntlet/zttr_probe.c
 * Run:   zttr_probe <wisdom dir> [N] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"
#include "zttr.h"
#include "zr2c.h"
#include "race_timing.h"
#ifdef _WIN32
#include <windows.h>
#endif

static double urand(unsigned *s)
{
    *s = *s * 1664525u + 1013904223u;
    return (double)(*s >> 8) / (double)(1u << 24) - 0.5;
}

static void ref_r2c(int N, const double *x, double *X)
{
    const long double tp = 6.283185307179586476925286766559L;
    for (int k = 0; k <= N / 2; k++) {
        long double re = 0, im = 0;
        for (int n = 0; n < N; n++) {
            long double a = tp * (long double)((long long)k * n % N) / (long double)N;
            re += (long double)x[n] * cosl(a);
            im -= (long double)x[n] * sinl(a);
        }
        X[2 * k] = (double)re;
        X[2 * k + 1] = (double)im;
    }
}

static double relerr(const double *a, const double *b, size_t n)
{
    double e = 0, m = 0;
    for (size_t i = 0; i < n; i++) { if (fabs(a[i] - b[i]) > e) e = fabs(a[i] - b[i]); if (fabs(b[i]) > m) m = fabs(b[i]); }
    return m > 0 ? e / m : e;
}

typedef void (*body_fn)(void *);
static double time_body(body_fn f, void *ctx)
{
    double t0 = vfft_now_ns();
    f(ctx);
    double e = vfft_now_ns() - t0;
    int reps = (int)(2.0e5 / (e > 1.0 ? e : 1.0));
    if (reps < 3) reps = 3;
    if (reps > 4000) reps = 4000;
    double best[7];
    for (int r = 0; r < 7; r++)
    {
        double s = vfft_now_ns();
        for (int i = 0; i < reps; i++) f(ctx);
        best[r] = (vfft_now_ns() - s) / reps;
    }
    for (int i = 1; i < 7; i++)
        for (int j = i; j > 0 && best[j] < best[j - 1]; j--) { double t = best[j]; best[j] = best[j - 1]; best[j - 1] = t; }
    return best[3];
}

typedef struct { vfft_zttr_plan_t *p; const double *x; double *X; } zctx_t;
static void zttr_fwd(void *v) { zctx_t *c = (zctx_t *)v; vfft_zttr_execute_fwd(c->p, c->x, c->X); }
typedef struct { vfft_plan h; double *x, *X; } hctx_t;
static void door_fwd(void *v) { hctx_t *c = (hctx_t *)v; vfft_execute(c->h, VFFT_FORWARD, c->x, NULL, c->X, NULL); }
typedef struct { vfft_ztt_plan_t *zt; const double *x; double *X; } cctx_t;
static void ztt_c2c(void *v) { cctx_t *c = (cctx_t *)v; vfft_ztt_execute_fwd(c->zt, c->x, c->X); }
typedef struct { int N; const double *z; double *X, *aff; } fctx_t;
static void fold(void *v) { fctx_t *c = (fctx_t *)v; _zr2c_fold_fwd(c->z, c->X, c->aff, c->aff + (c->N / 4 + 1), c->N, 1, (size_t)c->N + 2, (size_t)c->N + 2); }

/* every ordered {4,8} chain with product m, nf >= 2, (m / r0) % 4 == 0 */
static int chains_of(int m, int out[][8], int *nfs, int cap)
{
    int n = 0, ch[8];
    /* iterative enumeration over nf up to 7 */
    for (int nf = 2; nf <= 7 && n < cap; nf++)
    {
        int idx = 0;
        for (long code = 0; code < (1L << nf) && n < cap; code++)
        {
            long prod = 1;
            for (int s = 0; s < nf; s++) { ch[s] = (code >> s) & 1 ? 8 : 4; prod *= ch[s]; }
            if (prod != m || (m / ch[0]) % 4) continue;
            memcpy(out[n], ch, sizeof(int) * (size_t)nf);
            nfs[n] = nf;
            n++;
            (void)idx;
        }
    }
    return n;
}

int main(int argc, char **argv)
{
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), 0x4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif
    if (argc < 2) { fprintf(stderr, "usage: zttr_probe <wisdom dir> [N]\n"); return 2; }
    vfft_wisdom *W = vfft_wisdom_load(argv[1]);
    static const int Ns[] = { 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536 };
    const int only = argc > 2 ? atoi(argv[2]) : 0;
    unsigned seed = 0x7654321u;
    int fails = 0;
    for (int ni = 0; ni < (int)(sizeof Ns / sizeof Ns[0]); ni++)
    {
        const int N = Ns[ni], M = N / 2;
        if (only && N != only) continue;
        double *x = (double *)vfft_aligned_alloc((size_t)(N + 2) * sizeof(double));
        double *X = (double *)vfft_aligned_alloc((size_t)(N + 2) * sizeof(double));
        double *Rf = (double *)vfft_aligned_alloc((size_t)(N + 2) * sizeof(double));
        for (int i = 0; i < N; i++) x[i] = urand(&seed);
        x[N] = x[N + 1] = 0;
        if (N <= 8192) ref_r2c(N, x, Rf);
        /* the front door's plan at N (its banked engine) */
        vfft_config_t cfg; memset(&cfg, 0, sizeof cfg);
        cfg.transform = VFFT_R2C; cfg.placement = VFFT_OUTOFPLACE; cfg.dims = 1; cfg.n[0] = N;
        cfg.howmany = 1; cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.order = VFFT_ORDER_DEFAULT;
        cfg.rigor = VFFT_PATIENT; cfg.wisdom = W; cfg.nthreads = 1;
        vfft_plan h = vfft_create(&cfg);
        double tdoor = 0;
        if (h) { hctx_t hc = { h, x, X }; tdoor = time_body(door_fwd, &hc); }
        int chains[64][8], nfs[64];
        const int nc = chains_of(M, chains, nfs, 64);
        printf("N=%d (M=%d): door zr2c %.0f ns; %d chains\n", N, M, tdoor, nc);
        double best = 1e30; int bi = -1; size_t bt = 0;
        for (int c = 0; c < nc; c++)
        {
            static const size_t tiles[] = { 0, 512, 1024, 2048 };
            for (int ti = 0; ti < 4; ti++)
            {
                if (tiles[ti] && (tiles[ti] >= (size_t)M || tiles[ti] < (size_t)(chains[c][0] * chains[c][1]))) continue;
                vfft_zttr_plan_t *p = vfft_zttr_create(N, chains[c], nfs[c], tiles[ti]);
                if (!p) { if (ti == 0) printf("  chain "); if (ti == 0) { for (int s = 0; s < nfs[c]; s++) printf("%s%d", s ? "." : "", chains[c][s]); printf(": refused\n"); } continue; }
                memset(X, 0, (size_t)(N + 2) * sizeof(double));
                vfft_zttr_execute_fwd(p, x, X);
                double e = N <= 8192 ? relerr(X, Rf, (size_t)N + 2) : -1;
                zctx_t zc = { p, x, X };
                double t = time_body(zttr_fwd, &zc);
                /* the parts: the ZTT c2c at M alone, and the fold */
                cctx_t cc = { p->zt, x, X };
                double tc = time_body(ztt_c2c, &cc);
                printf("  chain ");
                for (int s = 0; s < nfs[c]; s++) printf("%s%d", s ? "." : "", chains[c][s]);
                printf(" tile %4zu: %7.0f ns  (c2c(M) %6.0f)  err %.2e%s\n", tiles[ti], t, tc, e, (e > 1e-12) ? "  FAIL" : "");
                if (e > 1e-12) fails++;
                if (t < best) { best = t; bi = c; bt = tiles[ti]; }
                vfft_zttr_destroy(p);
            }
        }
        {
            int top = N / 4;
            double *aff = (double *)vfft_aligned_alloc(sizeof(double) * 4u * (size_t)(top + 1));
            _zr2c_init_aff(N, aff, aff + (top + 1), aff + 2 * (top + 1), aff + 3 * (top + 1));
            fctx_t fc = { N, x, X, aff };
            printf("  fold alone %.0f ns; best zttr ", time_body(fold, &fc));
            if (bi >= 0) { for (int s = 0; s < nfs[bi]; s++) printf("%s%d", s ? "." : "", chains[bi][s]); printf(" tile %zu = %.0f ns -> %.2fx vs door\n", bt, best, tdoor > 0 ? tdoor / best : 0.0); }
            else printf("none\n");
            vfft_aligned_free(aff);
        }
        if (h) vfft_destroy(h);
        vfft_aligned_free(x); vfft_aligned_free(X); vfft_aligned_free(Rf);
    }
    printf("%d gate failures\n", fails);
    return fails ? 1 : 0;
}
