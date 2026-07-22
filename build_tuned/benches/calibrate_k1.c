/* calibrate_k1.c — the K=1 four-axis calibrator (row_major_engine.md step 2 of
 * productization): per cell N, measure ALL candidates over
 *     route {mono, mono-alt, mono-il, 3p, 3p-ip, 2pa-ip, 2pb-ip, twl-ip,
 *            3p-il, 2p-il}  x  pair (R1xR2)  x  placement  x  layout
 * and print per-axis verdicts (split-oop / split-ip / il) as wisdom-ready
 * lines.
 *
 * Methodology (fixes the cmp_old_new fixed-order bias that mispicked pairs in
 * the bench pre-pass): ONE process per cell; per-candidate hot timing (10
 * warmup + reps); T trials with 32MB cachebust between; candidate ORDER
 * ROTATED per trial (start = trial mod ncand) so no candidate systematically
 * runs coolest; best-of-T per candidate. Pin core 2, HIGH prio.
 *
 * Usage: calibrate_k1 <N> [trials=4]
 * Build: python build.py --src benches/calibrate_k1.c --compile
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>

#include "executor.h"
#include "planner.h"
#include "oop_plan.h"

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

/* routes */
enum { R_3P = 0, R_3P_IP, R_2PA_IP, R_2PB_IP, R_TWL_IP, R_3P_IL, R_2P_IL,
       R_MONO, R_MONO_ALT, R_MONO_IL, R_3PL3_IP, R_2PAL3_IP,
       R_2PB_SPEC, R_2PAL3_SPEC };
static const char *RNAME[] = { "3p", "3p-ip", "2pa-ip", "2pb-ip", "twl-ip",
                               "3p-il", "2p-il", "mono", "mono-alt", "mono-il",
                               "3p-l3-ip", "2pa-l3-ip",
                               "2pb-spec", "2pa-l3-spec" };
/* axis of each route: 0=split-oop 1=split-ip 2=il */
static const int RAXIS[] = { 0, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1 };

/* per-cell stride-specialized twins (§13.3 item A): 7-arg ABI, strides baked */
extern void radix32_n1_oop_fwd_avx2_UG_UL_spec32_1_1_32(
    const double *, const double *, double *, double *,
    const double *, const double *, size_t);
extern void radix32_t1_oop_fwd_avx2_UG_UG_spec32_1_32_1(
    const double *, const double *, double *, double *,
    const double *, const double *, size_t);
extern void radix64_n1_oop_fwd_avx2_UG_UG_spec64_1_64_1(
    const double *, const double *, double *, double *,
    const double *, const double *, size_t);
extern void radix64_t1_oop_fwd_avx2_UL_UG_log3_spec1_64_64_1(
    const double *, const double *, double *, double *,
    const double *, const double *, size_t);

typedef struct {
    int route, R1, R2;
    vfft_oop_plan_t *p;      /* NULL for mono routes */
    double best;
} cand_t;

static int N_g;
static double *xr, *xi, *dr, *di, *wr, *wi, *z;

static void run_cand(cand_t *c)
{
    switch (c->route) {
    case R_3P:     vfft_oop_execute_fwd(c->p, xr, xi, dr, di); break;
    case R_3P_IP:  vfft_oop_execute_fwd(c->p, wr, wi, wr, wi); break;
    case R_2PA_IP: vfft_oop_execute_fwd_2pa(c->p, wr, wi, wr, wi); break;
    case R_2PB_IP: vfft_oop_execute_fwd_2pb(c->p, wr, wi, wr, wi); break;
    case R_TWL_IP: vfft_oop_execute_fwd_2pa_twl(c->p, wr, wi, wr, wi); break;
    case R_3P_IL:  vfft_oop_execute_fwd_il(c->p, z, z); break;
    case R_2P_IL:  vfft_oop_execute_fwd_2p_il(c->p, z, z); break;
    case R_MONO:     vfft_k1_mono_fn(N_g)(xr, xi, dr, di, 0,0,0,0,0,0,0); break;
    case R_MONO_ALT: vfft_k1_mono_alt_fn(N_g)(xr, xi, dr, di, 0,0,0,0,0,0,0); break;
    case R_MONO_IL:  vfft_k1_mono_il_fn(N_g, 0)(z, 0, z, 0, 0,0,0,0,0,0,0); break;
    /* l3 routes: the candidate's dedicated plan has t1p/t1_ul overwritten
     * with the log3 twins at setup (same ABI, same Qr/Qi) */
    case R_3PL3_IP:  vfft_oop_execute_fwd(c->p, wr, wi, wr, wi); break;
    case R_2PAL3_IP: vfft_oop_execute_fwd_2pa(c->p, wr, wi, wr, wi); break;
    /* stride-spec pipelines (baked-constant twins; scratch/Qr from the plan) */
    case R_2PB_SPEC: /* 1024 = 2pb 32x32 */
        radix32_n1_oop_fwd_avx2_UG_UL_spec32_1_1_32(wr, wi, c->p->col_re, c->p->col_im, 0, 0, 32);
        radix32_t1_oop_fwd_avx2_UG_UG_spec32_1_32_1(c->p->col_re, c->p->col_im, wr, wi, c->p->Qr, c->p->Qi, 32);
        break;
    case R_2PAL3_SPEC: /* 4096 = 2pa-l3 64x64 */
        radix64_n1_oop_fwd_avx2_UG_UG_spec64_1_64_1(wr, wi, c->p->col_re, c->p->col_im, 0, 0, 64);
        radix64_t1_oop_fwd_avx2_UL_UG_log3_spec1_64_64_1(c->p->col_re, c->p->col_im, wr, wi, c->p->Qr, c->p->Qi, 64);
        break;
    }
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg); (void)reg;

    int N = argc > 1 ? atoi(argv[1]) : 256;
    int trials = argc > 2 ? atoi(argv[2]) : 4;
    N_g = N;

    xr = ad(N); xi = ad(N); dr = ad(N); di = ad(N);
    wr = ad(N); wi = ad(N); z = ad((size_t)2 * N);
    srand(77 + N);
    for (int n = 0; n < N; n++) {
        xr[n] = (double)rand() / RAND_MAX - 0.5;
        xi[n] = (double)rand() / RAND_MAX - 0.5;
        wr[n] = xr[n]; wi[n] = xi[n];
        z[2*n] = xr[n]; z[2*n+1] = xi[n];
    }

    /* enumerate candidates */
    cand_t cand[128]; int nc = 0;
    vfft_oop_plan_t *plans[16]; int np = 0;
    for (int R2 = (N < 128 ? N : 128); R2 >= 4; R2--) {
        if (N % R2) continue;
        int R1 = N / R2;
        if (R1 < 4 || R1 > 128 || (R1 % 4) || (R2 % 4)) continue;
        vfft_oop_plan_t *p = vfft_oop_plan_create_k1(N, R1, R2);
        if (!p) continue;
        plans[np++] = p;
        struct { int route; int avail; } rs[] = {
            { R_3P, 1 }, { R_3P_IP, 1 },
            { R_2PA_IP, p->t1_ul != 0 }, { R_2PB_IP, p->leaf_ul != 0 },
            { R_TWL_IP, p->t1_ul_twl != 0 },
            { R_3P_IL, p->il_leaf && p->t1_il },
            { R_2P_IL, p->il_leaf && p->t1_ul_il },
        };
        for (int i = 0; i < 7 && nc < 120; i++)
            if (rs[i].avail) {
                cand[nc].route = rs[i].route; cand[nc].R1 = R1; cand[nc].R2 = R2;
                cand[nc].p = p; cand[nc].best = 1e18; nc++;
            }
        /* stride-spec candidates (cell-gated: emitted for the two winner
         * shapes only; §13.3 item A measure-first) */
        if (N == 1024 && R1 == 32 && R2 == 32 && nc < 118) {
            cand[nc].route = R_2PB_SPEC; cand[nc].R1 = R1; cand[nc].R2 = R2;
            cand[nc].p = p; cand[nc].best = 1e18; nc++;
        }
        if (N == 4096 && R1 == 64 && R2 == 64 && nc < 118) {
            cand[nc].route = R_2PAL3_SPEC; cand[nc].R1 = R1; cand[nc].R2 = R2;
            cand[nc].p = p; cand[nc].best = 1e18; nc++;
        }
        /* LOG3 candidates: dedicated plan with the t1 pointers swapped ONCE
         * at setup (outside timing); same Qr/Qi tables. */
        if ((p->t1_l3 || p->t1_ul_l3) && np < 15 && nc < 118) {
            vfft_oop_plan_t *pl = vfft_oop_plan_create_k1(N, R1, R2);
            if (pl) {
                plans[np++] = pl;
                if (pl->t1_l3) {
                    pl->t1p = pl->t1_l3;
                    cand[nc].route = R_3PL3_IP; cand[nc].R1 = R1; cand[nc].R2 = R2;
                    cand[nc].p = pl; cand[nc].best = 1e18; nc++;
                }
                if (pl->t1_ul_l3) {
                    pl->t1_ul = pl->t1_ul_l3;
                    cand[nc].route = R_2PAL3_IP; cand[nc].R1 = R1; cand[nc].R2 = R2;
                    cand[nc].p = pl; cand[nc].best = 1e18; nc++;
                }
            }
        }
    }
    if (vfft_k1_mono_fn(N))    { cand[nc].route = R_MONO;     cand[nc].R1 = cand[nc].R2 = 0; cand[nc].p = 0; cand[nc].best = 1e18; nc++; }
    if (vfft_k1_mono_alt_fn(N)){ cand[nc].route = R_MONO_ALT; cand[nc].R1 = cand[nc].R2 = 0; cand[nc].p = 0; cand[nc].best = 1e18; nc++; }
    if (vfft_k1_mono_il_fn(N, 0)) { cand[nc].route = R_MONO_IL; cand[nc].R1 = cand[nc].R2 = 0; cand[nc].p = 0; cand[nc].best = 1e18; nc++; }

    int reps = (int)(2e6 / (double)N);
    if (reps < 100) reps = 100;
    if (reps > 400000) reps = 400000;

    printf("# calibrate_k1 N=%d candidates=%d trials=%d reps=%d (order-rotated)\n",
           N, nc, trials, reps);

    for (int t = 0; t < trials; t++) {
        if (t) cachebust();
        for (int k = 0; k < nc; k++) {
            cand_t *c = &cand[(k + t) % nc];   /* rotate start per trial */
            for (int w = 0; w < 10; w++) run_cand(c);
            double t0 = now_ms();
            for (int i = 0; i < reps; i++) run_cand(c);
            double ns = (now_ms() - t0) * 1e6 / reps;
            if (ns < c->best) c->best = ns;
        }
    }

    /* full dump + per-axis verdicts */
    int win[3] = { -1, -1, -1 };
    for (int k = 0; k < nc; k++) {
        int ax = RAXIS[cand[k].route];
        if (win[ax] < 0 || cand[k].best < cand[win[ax]].best) win[ax] = k;
        printf("cand,%d,%s,%d,%d,%.1f\n", N, RNAME[cand[k].route],
               cand[k].R1, cand[k].R2, cand[k].best);
    }
    static const char *AXN[3] = { "split-oop", "split-ip", "il" };
    for (int a = 0; a < 3; a++)
        if (win[a] >= 0)
            printf("K1VERDICT,%d,%s,%s,%d,%d,%.1f\n", N, AXN[a],
                   RNAME[cand[win[a]].route], cand[win[a]].R1, cand[win[a]].R2,
                   cand[win[a]].best);

    /* ready-to-append kind-3 OOP wisdom line: split verdict = the split-ip
     * winner (route code identical for oop/ip calls), il verdict = the il
     * winner. Format: N 1 3 spr R1 R2 ilr iR1 iR2 ns */
    if (win[1] >= 0)
    {
        static const int SPMAP[] = { /* internal route -> VFFT_K1_SP_* */
            VFFT_K1_SP_3P, VFFT_K1_SP_3P, VFFT_K1_SP_2PA, VFFT_K1_SP_2PB,
            VFFT_K1_SP_TWL, -1, -1, VFFT_K1_SP_MONO, VFFT_K1_SP_MONO, -1,
            VFFT_K1_SP_3P_L3, VFFT_K1_SP_2PA_L3 };
        static const int ILMAP[] = { -1, -1, -1, -1, -1,
            VFFT_K1_IL_3P, VFFT_K1_IL_2P, -1, -1, VFFT_K1_IL_MONO, -1, -1 };
        cand_t *ws = &cand[win[1]];
        int spr = SPMAP[ws->route];
        int sR1 = ws->R1, sR2 = ws->R2;
        if (ws->route == R_MONO_ALT) { sR1 = 8; sR2 = N / 8; } /* alt pair marker */
        int ilr = 0, iR1 = 0, iR2 = 0;
        double ns = ws->best;
        if (win[2] >= 0)
        {
            cand_t *wi = &cand[win[2]];
            ilr = ILMAP[wi->route];
            if (ilr < 0) ilr = 0;
            iR1 = wi->R1; iR2 = wi->R2;
        }
        if (spr >= 0)
            printf("K1WISDOM %d 1 3 %d %d %d %d %d %d %.1f\n",
                   N, spr, sR1, sR2, ilr, iR1, iR2, ns);
    }

    for (int i = 0; i < np; i++) vfft_oop_plan_destroy(plans[i]);
    return 0;
}
