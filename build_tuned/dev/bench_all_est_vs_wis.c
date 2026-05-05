/* bench_all_est_vs_wis.c — diagnostic: estimate vs wisdom for every
 * public transform. Reveals where estimate and wisdom diverge (good)
 * and where they silently produce identical plans (gap in flag honoring).
 *
 * For each transform × cell: plan with VFFT_ESTIMATE and VFFT_MEASURE,
 * time both, report ratio. Wisdom must be loaded; cells without a
 * wisdom hit will auto-calibrate at first MEASURE which is slow.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>

#include "vfft.h"

static double now_ns(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / (double)f.QuadPart;
}

#define BENCH_REPS 21
#define BENCH_WARMUP 5

static double time_ns(void (*fn)(void *), void *ctx) {
    for (int i = 0; i < BENCH_WARMUP; i++) fn(ctx);
    double best = 1e30;
    for (int i = 0; i < BENCH_REPS; i++) {
        double t0 = now_ns();
        fn(ctx);
        double dt = now_ns() - t0;
        if (dt < best) best = dt;
    }
    return best;
}

/* ── Bench contexts (one per transform shape) ─────────────────────── */
typedef struct {
    vfft_plan plan;
    double *re, *im, *out;
} c2c_ctx;
static void run_c2c_fwd(void *c) {
    c2c_ctx *x = c; vfft_execute_fwd(x->plan, x->re, x->im);
}

typedef struct {
    vfft_plan plan;
    double *in, *out;
} real_ctx;
static void run_dct2(void *c) { real_ctx *x = c; vfft_execute_dct2(x->plan, x->in, x->out); }
static void run_dct4(void *c) { real_ctx *x = c; vfft_execute_dct4(x->plan, x->in, x->out); }
static void run_dst2(void *c) { real_ctx *x = c; vfft_execute_dst2(x->plan, x->in, x->out); }
static void run_dht (void *c) { real_ctx *x = c; vfft_execute_dht (x->plan, x->in, x->out); }

/* 2D row-major buffers. For 2D we just need to time, not roundtrip. */
typedef struct { vfft_plan plan; double *re, *im; } twoD_ctx;
static void run_2d(void *c) { twoD_ctx *x = c; vfft_execute_fwd(x->plan, x->re, x->im); }

/* ── 1D bench cells (loaded in wisdom file) ────────────────────────── */
static struct { int N; size_t K; const char *name; } CELLS_1D[] = {
    {   64,  256, "C2C N=64 K=256"  },
    {  256,  256, "C2C N=256 K=256" },
    { 1024,  256, "C2C N=1024 K=256"},
    { 4096,  256, "C2C N=4096 K=256"},
};
static struct { int N; size_t K; const char *name; } CELLS_REAL[] = {
    {   64,  256, "N=64 K=256"   },
    {  256,  256, "N=256 K=256"  },
    { 1024,  256, "N=1024 K=256" },
};
static struct { int N1, N2; const char *name; } CELLS_2D[] = {
    {  64,  64, "N1=64 N2=64"   },
    { 128, 128, "N1=128 N2=128" },
    { 256, 256, "N1=256 N2=256" },
};

static void bench_c2c_cell(int N, size_t K, const char *name) {
    size_t NK = (size_t)N * K;
    double *re = vfft_alloc(NK*sizeof(double));
    double *im = vfft_alloc(NK*sizeof(double));
    srand(42 + N + (int)K);
    for (size_t i = 0; i < NK; i++) {
        re[i] = (double)rand()/RAND_MAX - 0.5;
        im[i] = (double)rand()/RAND_MAX - 0.5;
    }

    vfft_plan pe = vfft_plan_c2c(N, K, VFFT_ESTIMATE);
    vfft_plan pw = vfft_plan_c2c(N, K, VFFT_MEASURE);
    if (!pe || !pw) { printf("  C2C %s: PLAN FAIL\n", name); goto done; }

    c2c_ctx ce = {pe, re, im, NULL};
    c2c_ctx cw = {pw, re, im, NULL};
    double te = time_ns(run_c2c_fwd, &ce);
    double tw = time_ns(run_c2c_fwd, &cw);
    printf("  C2C   %-22s  est=%9.0f  wis=%9.0f  ratio=%5.2fx %s\n",
           name, te, tw, te/tw,
           (te > tw*1.05) ? "(estimate slower)" :
           (tw > te*1.05) ? "(wisdom slower!?)" : "(tied)");

done:
    if (pe) vfft_destroy(pe);
    if (pw) vfft_destroy(pw);
    vfft_free(re); vfft_free(im);
}

#define BENCH_REAL_CELL(label, N, K, name, plan_fn, run_fn)                  \
    do {                                                                     \
        size_t NK = (size_t)N * K;                                           \
        double *in  = vfft_alloc(NK*sizeof(double));                         \
        double *out = vfft_alloc(NK*sizeof(double));                         \
        srand(42 + N + (int)K);                                              \
        for (size_t i = 0; i < NK; i++)                                      \
            in[i] = (double)rand()/RAND_MAX - 0.5;                           \
        vfft_plan pe = plan_fn(N, K, VFFT_ESTIMATE);                         \
        vfft_plan pw = plan_fn(N, K, VFFT_MEASURE);                          \
        if (!pe || !pw) {                                                    \
            printf("  %-5s %-22s  PLAN FAIL\n", label, name);                \
        } else {                                                             \
            real_ctx ce = {pe, in, out};                                     \
            real_ctx cw = {pw, in, out};                                     \
            double te = time_ns(run_fn, &ce);                                \
            double tw = time_ns(run_fn, &cw);                                \
            printf("  %-5s %-22s  est=%9.0f  wis=%9.0f  ratio=%5.2fx %s\n",  \
                   label, name, te, tw, te/tw,                               \
                   (te > tw*1.05) ? "(estimate slower)" :                    \
                   (tw > te*1.05) ? "(wisdom slower!?)" : "(tied)");         \
        }                                                                    \
        if (pe) vfft_destroy(pe);                                            \
        if (pw) vfft_destroy(pw);                                            \
        vfft_free(in); vfft_free(out);                                       \
    } while (0)

static void bench_2d_cell(int N1, int N2, const char *name) {
    size_t NK = (size_t)N1 * N2;
    double *re = vfft_alloc(NK*sizeof(double));
    double *im = vfft_alloc(NK*sizeof(double));
    srand(42 + N1 + N2);
    for (size_t i = 0; i < NK; i++) {
        re[i] = (double)rand()/RAND_MAX - 0.5;
        im[i] = (double)rand()/RAND_MAX - 0.5;
    }

    vfft_plan pe = vfft_plan_2d(N1, N2, VFFT_ESTIMATE);
    vfft_plan pw = vfft_plan_2d(N1, N2, VFFT_MEASURE);
    if (!pe || !pw) { printf("  2D    %s: PLAN FAIL\n", name); goto done; }

    twoD_ctx ce = {pe, re, im};
    twoD_ctx cw = {pw, re, im};
    double te = time_ns(run_2d, &ce);
    double tw = time_ns(run_2d, &cw);
    printf("  2D    %-22s  est=%9.0f  wis=%9.0f  ratio=%5.2fx %s\n",
           name, te, tw, te/tw,
           (fabs(te - tw)/te < 0.02) ? "(IDENTICAL — flag silently ignored)" :
           (te > tw*1.05) ? "(estimate slower)" :
           (tw > te*1.05) ? "(wisdom slower!?)" : "(tied)");

done:
    if (pe) vfft_destroy(pe);
    if (pw) vfft_destroy(pw);
    vfft_free(re); vfft_free(im);
}

static void bench_2d_r2c_cell(int N1, int N2, const char *name) {
    size_t hp1 = (size_t)(N2/2 + 1);
    size_t in_sz = (size_t)N1 * N2;
    size_t out_sz = (size_t)N1 * hp1;
    double *in   = vfft_alloc(in_sz  * sizeof(double));
    double *outr = vfft_alloc(out_sz * sizeof(double));
    double *outi = vfft_alloc(out_sz * sizeof(double));
    srand(42 + N1 + N2);
    for (size_t i = 0; i < in_sz; i++) in[i] = (double)rand()/RAND_MAX - 0.5;

    vfft_plan pe = vfft_plan_2d_r2c(N1, N2, VFFT_ESTIMATE);
    vfft_plan pw = vfft_plan_2d_r2c(N1, N2, VFFT_MEASURE);
    if (!pe || !pw) { printf("  2DR2C %s: PLAN FAIL\n", name); goto done; }

    typedef struct { vfft_plan p; double *in, *outr, *outi; } r2c_ctx;
    r2c_ctx ce = {pe, in, outr, outi};
    r2c_ctx cw = {pw, in, outr, outi};

    /* warmup + min-of-21 */
    for (int i = 0; i < BENCH_WARMUP; i++) vfft_execute_2d_r2c(ce.p, ce.in, ce.outr, ce.outi);
    double te = 1e30;
    for (int i = 0; i < BENCH_REPS; i++) {
        double t0 = now_ns();
        vfft_execute_2d_r2c(ce.p, ce.in, ce.outr, ce.outi);
        double dt = now_ns() - t0;
        if (dt < te) te = dt;
    }
    for (int i = 0; i < BENCH_WARMUP; i++) vfft_execute_2d_r2c(cw.p, cw.in, cw.outr, cw.outi);
    double tw = 1e30;
    for (int i = 0; i < BENCH_REPS; i++) {
        double t0 = now_ns();
        vfft_execute_2d_r2c(cw.p, cw.in, cw.outr, cw.outi);
        double dt = now_ns() - t0;
        if (dt < tw) tw = dt;
    }

    printf("  2DR2C %-22s  est=%9.0f  wis=%9.0f  ratio=%5.2fx %s\n",
           name, te, tw, te/tw,
           (fabs(te - tw)/te < 0.02) ? "(IDENTICAL — flag silently ignored)" :
           (te > tw*1.05) ? "(estimate slower)" :
           (tw > te*1.05) ? "(wisdom slower!?)" : "(tied)");

done:
    if (pe) vfft_destroy(pe);
    if (pw) vfft_destroy(pw);
    vfft_free(in); vfft_free(outr); vfft_free(outi);
}

int main(void) {
    vfft_init();
    vfft_pin_thread(0);
    vfft_set_num_threads(1);

    int rc = vfft_load_wisdom("vfft_wisdom_tuned.txt");
    if (rc != 0) {
        /* Try fully-qualified path as fallback when run from project root. */
        rc = vfft_load_wisdom(
            "c:/Users/Tugbars/Desktop/highSpeedFFT/build_tuned/vfft_wisdom_tuned.txt");
    }
    if (rc != 0) {
        fprintf(stderr, "[error] wisdom file not found.\n");
        return 1;
    }
    printf("=== bench_all_est_vs_wis -- estimate vs wisdom across transforms ===\n");
    printf("Reps: %d (warmup %d). Lower ns = faster. ratio = est/wis.\n\n",
           BENCH_REPS, BENCH_WARMUP);

    /* 1D C2C */
    printf("--- 1D C2C ---\n");
    for (int i = 0; i < (int)(sizeof(CELLS_1D)/sizeof(*CELLS_1D)); i++)
        bench_c2c_cell(CELLS_1D[i].N, CELLS_1D[i].K, CELLS_1D[i].name);

    /* 1D R2C — split output (separate outr / outi buffers) */
    printf("\n--- 1D R2C ---\n");
    for (int i = 0; i < (int)(sizeof(CELLS_REAL)/sizeof(*CELLS_REAL)); i++) {
        int N = CELLS_REAL[i].N; size_t K = CELLS_REAL[i].K;
        size_t in_sz  = (size_t)N * K;
        size_t out_sz = (size_t)(N/2 + 1) * K;
        double *in   = vfft_alloc(in_sz  * sizeof(double));
        double *outr = vfft_alloc(out_sz * sizeof(double));
        double *outi = vfft_alloc(out_sz * sizeof(double));
        srand(42 + N + (int)K);
        for (size_t j = 0; j < in_sz; j++) in[j] = (double)rand()/RAND_MAX - 0.5;

        vfft_plan pe = vfft_plan_r2c(N, K, VFFT_ESTIMATE);
        vfft_plan pw = vfft_plan_r2c(N, K, VFFT_MEASURE);
        if (!pe || !pw) {
            printf("  R2C   %-22s  PLAN FAIL\n", CELLS_REAL[i].name);
        } else {
            for (int j = 0; j < BENCH_WARMUP; j++) vfft_execute_r2c(pe, in, outr, outi);
            double te = 1e30;
            for (int j = 0; j < BENCH_REPS; j++) {
                double t0 = now_ns();
                vfft_execute_r2c(pe, in, outr, outi);
                double dt = now_ns() - t0;
                if (dt < te) te = dt;
            }
            for (int j = 0; j < BENCH_WARMUP; j++) vfft_execute_r2c(pw, in, outr, outi);
            double tw = 1e30;
            for (int j = 0; j < BENCH_REPS; j++) {
                double t0 = now_ns();
                vfft_execute_r2c(pw, in, outr, outi);
                double dt = now_ns() - t0;
                if (dt < tw) tw = dt;
            }
            printf("  R2C   %-22s  est=%9.0f  wis=%9.0f  ratio=%5.2fx %s\n",
                   CELLS_REAL[i].name, te, tw, te/tw,
                   (fabs(te - tw)/te < 0.02) ? "(IDENTICAL)" :
                   (te > tw*1.05) ? "(estimate slower)" :
                   (tw > te*1.05) ? "(wisdom slower!?)" : "(tied)");
        }
        if (pe) vfft_destroy(pe);
        if (pw) vfft_destroy(pw);
        vfft_free(in); vfft_free(outr); vfft_free(outi);
    }

    /* DCT-II */
    printf("\n--- DCT-II ---\n");
    for (int i = 0; i < (int)(sizeof(CELLS_REAL)/sizeof(*CELLS_REAL)); i++)
        BENCH_REAL_CELL("DCT2", CELLS_REAL[i].N, CELLS_REAL[i].K,
                        CELLS_REAL[i].name, vfft_plan_dct2, run_dct2);

    /* DCT-IV */
    printf("\n--- DCT-IV ---\n");
    for (int i = 0; i < (int)(sizeof(CELLS_REAL)/sizeof(*CELLS_REAL)); i++)
        BENCH_REAL_CELL("DCT4", CELLS_REAL[i].N, CELLS_REAL[i].K,
                        CELLS_REAL[i].name, vfft_plan_dct4, run_dct4);

    /* DST-II */
    printf("\n--- DST-II ---\n");
    for (int i = 0; i < (int)(sizeof(CELLS_REAL)/sizeof(*CELLS_REAL)); i++)
        BENCH_REAL_CELL("DST2", CELLS_REAL[i].N, CELLS_REAL[i].K,
                        CELLS_REAL[i].name, vfft_plan_dst2, run_dst2);

    /* DHT */
    printf("\n--- DHT ---\n");
    for (int i = 0; i < (int)(sizeof(CELLS_REAL)/sizeof(*CELLS_REAL)); i++)
        BENCH_REAL_CELL("DHT", CELLS_REAL[i].N, CELLS_REAL[i].K,
                        CELLS_REAL[i].name, vfft_plan_dht, run_dht);

    /* 2D C2C */
    printf("\n--- 2D C2C ---\n");
    for (int i = 0; i < (int)(sizeof(CELLS_2D)/sizeof(*CELLS_2D)); i++)
        bench_2d_cell(CELLS_2D[i].N1, CELLS_2D[i].N2, CELLS_2D[i].name);

    /* 2D R2C */
    printf("\n--- 2D R2C ---\n");
    for (int i = 0; i < (int)(sizeof(CELLS_2D)/sizeof(*CELLS_2D)); i++)
        bench_2d_r2c_cell(CELLS_2D[i].N1, CELLS_2D[i].N2, CELLS_2D[i].name);

    return 0;
}
