/* bench_k1_best_vs_mkl.c — K=1: the CALIBRATED per-cell winners vs MKL.
 *
 * Successor to bench_k1_vs_mkl.c: instead of that bench's fixed-order pair
 * pre-pass (known biased), the arms here are read from the kind-3 wisdom the
 * calibrator wrote (k1_wis/oop_wisdom.txt, median-of-runs), i.e. exactly the
 * plans the engine ships. Arms per N:
 *   MKL-IL  — DFTI in-place interleaved
 *   MKL-sp  — DFTI in-place REAL_REAL (split)
 *   W-sp    — wisdom split verdict (route x pair), executed as the front door
 *             would (mono via pair fn; l3 via pointer swap; twl linear tables)
 *   JIT     — plan-time-baked kernel for the SAME (route, pair) where the
 *             wrapper table covers it (2pa/2pb/twl/2pa-l3); compile-on-miss
 *   W-il    — wisdom interleaved verdict (3p-il / 2p-il / mono-il)
 * N=64 has no kind-3 line yet (aggregator name-map gap): falls back to the
 * calibrated winners recorded in memory — mono (emitted) + twl 8x8, mono-il.
 *
 * Methodology (canonical, bench_1d_vs_mkl.c-modeled): pinned core 2 P-core,
 * HIGH prio, per-arm hot 10-warmup + reps, best-of-5 trials, 32MB cachebust
 * between trials, arm ORDER FLIPPED per trial. Run one N per process (driver
 * script) for isolation. Speedups = MKL/ours (>1 = we win), layout-fair:
 * split vs MKL-sp, interleaved vs MKL-IL.
 *
 * Build: python build.py --src benches/bench_k1_best_vs_mkl.c --mkl
 * Run:   bench_k1_best_vs_mkl <N>   (wisdom: env VFFT_K1_WIS or k1_wis/oop_wisdom.txt)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include <mkl_dfti.h>

#include "executor.h"
#include "planner.h"
#include "oop_plan.h"
#include "oop_wisdom.h"
#include "k1_jit_runtime.h"

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

static const char *SPN[] = { "3p", "2pa", "2pb", "twl", "mono", "2pa-l3", "3p-l3", "ccol" };
static const char *ILN[] = { "-", "3p-il", "2p-il", "mono-il" };

enum { A_MKL_IL = 0, A_MKL_SP, A_WSP, A_JIT, A_WIL, A_XTRA, NARMS };

typedef struct {
    int N, spr, R1, R2, ilr, iR1, iR2;
    DFTI_DESCRIPTOR_HANDLE hi, hs;
    vfft_oop_plan_t *psp, *pil, *pxt;   /* pxt = N=64 extra arm (twl 8x8) */
    vfft_oop11_fn mono, monoil;
    vfft_k1_jit_fn jit;
    double *xr, *xi, *dr, *di, *wr, *wi, *zi, *zs, *mre, *mim;
} bctx_t;

static void run_arm(bctx_t *c, int a)
{
    switch (a) {
    case A_MKL_IL: DftiComputeForward(c->hi, c->zi); break;
    case A_MKL_SP: DftiComputeForward(c->hs, c->mre, c->mim); break;
    case A_WSP:
        switch (c->spr) {
        case VFFT_K1_SP_3P: case VFFT_K1_SP_3P_L3:
            vfft_oop_execute_fwd(c->psp, c->wr, c->wi, c->wr, c->wi); break;
        case VFFT_K1_SP_2PA: case VFFT_K1_SP_2PA_L3:
            vfft_oop_execute_fwd_2pa(c->psp, c->wr, c->wi, c->wr, c->wi); break;
        case VFFT_K1_SP_2PB:
            vfft_oop_execute_fwd_2pb(c->psp, c->wr, c->wi, c->wr, c->wi); break;
        case VFFT_K1_SP_TWL:
            vfft_oop_execute_fwd_2pa_twl(c->psp, c->wr, c->wi, c->wr, c->wi); break;
        case VFFT_K1_SP_MONO:
            c->mono(c->xr, c->xi, c->dr, c->di, 0, 0, 0, 0, 0, 0, 0); break;
        case VFFT_K1_SP_CCOL:
            vfft_oop_execute_fwd_ccol(c->psp, c->wr, c->wi, c->wr, c->wi); break;
        }
        break;
    case A_JIT:
        c->jit(c->wr, c->wi, c->wr, c->wi, c->psp->col_re, c->psp->col_im,
               c->spr == VFFT_K1_SP_TWL ? c->psp->Qlr : c->psp->Qr,
               c->spr == VFFT_K1_SP_TWL ? c->psp->Qli : c->psp->Qi);
        break;
    case A_WIL:
        switch (c->ilr) {
        case VFFT_K1_IL_3P:   vfft_oop_execute_fwd_il(c->pil, c->zs, c->zs); break;
        case VFFT_K1_IL_2P:   vfft_oop_execute_fwd_2p_il(c->pil, c->zs, c->zs); break;
        case VFFT_K1_IL_MONO: c->monoil(c->zs, 0, c->zs, 0, 0, 0, 0, 0, 0, 0, 0); break;
        }
        break;
    case A_XTRA:
        vfft_oop_execute_fwd_2pa_twl(c->pxt, c->wr, c->wi, c->wr, c->wi); break;
    }
}

static int arm_avail(bctx_t *c, int a)
{
    switch (a) {
    case A_MKL_IL: return c->hi != NULL;
    case A_MKL_SP: return c->hs != NULL;
    case A_WSP:    return c->spr == VFFT_K1_SP_MONO ? c->mono != NULL : c->psp != NULL;
    case A_JIT:    return c->jit != NULL && c->psp != NULL;
    case A_WIL:    return c->ilr == VFFT_K1_IL_MONO ? c->monoil != NULL
                                                    : (c->ilr != VFFT_K1_IL_NONE && c->pil != NULL);
    case A_XTRA:   return c->pxt != NULL && c->pxt->t1_ul_twl != NULL;
    }
    return 0;
}

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
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg); (void)reg;

    static vfft_oop_wisdom_t wis;
    const char *wp = getenv("VFFT_K1_WIS");
    if (!wp) wp = "k1_wis/oop_wisdom.txt";
    if (vfft_oop_wisdom_load(&wis, wp) != 0) {
        fprintf(stderr, "cannot load wisdom %s\n", wp);
        return 1;
    }

    int Nd[] = { 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384 };
    int nN = argc > 1 ? argc - 1 : 9;

    for (int ni = 0; ni < nN; ni++) {
        int N = argc > 1 ? atoi(argv[ni + 1]) : Nd[ni];
        bctx_t c; memset(&c, 0, sizeof c);
        c.N = N;

        const vfft_oop_wisdom_entry_t *e = vfft_oop_wisdom_lookup_k1(&wis, N);
        if (e) {
            c.spr = e->k1_sp_route; c.R1 = e->R1; c.R2 = e->R2;
            c.ilr = e->k1_il_route; c.iR1 = e->il_R1; c.iR2 = e->il_R2;
        } else if (N == 64) {   /* no kind-3 line yet: calibrated winners */
            c.spr = VFFT_K1_SP_MONO; c.R1 = 0; c.R2 = 0;
            c.ilr = VFFT_K1_IL_MONO;
            c.pxt = vfft_oop_plan_create_k1(64, 8, 8);
        } else {
            fprintf(stderr, "no kind-3 wisdom for N=%d\n", N);
            continue;
        }

        if (c.spr == VFFT_K1_SP_MONO)
            c.mono = c.R1 ? vfft_k1_mono_pair_fn(N, c.R1) : vfft_k1_mono_fn(N);
        else if (c.spr == VFFT_K1_SP_CCOL) {
            int ccf[6], ccn = e ? vfft_k1_cc_chain_decode(e->cc_chain, ccf) : 0;
            if (ccn)
                c.psp = vfft_oop_plan_create_k1_cc(N, c.R1, ccf, ccn, &reg);
        }
        else {
            c.psp = vfft_oop_plan_create_k1(N, c.R1, c.R2);
            if (c.psp) {
                if (c.spr == VFFT_K1_SP_2PA_L3 && c.psp->t1_ul_l3)
                    c.psp->t1_ul = c.psp->t1_ul_l3;
                if (c.spr == VFFT_K1_SP_3P_L3 && c.psp->t1_l3)
                    c.psp->t1p = c.psp->t1_l3;
                if (c.spr == VFFT_K1_SP_2PA || c.spr == VFFT_K1_SP_2PB ||
                    c.spr == VFFT_K1_SP_TWL || c.spr == VFFT_K1_SP_2PA_L3)
                    c.jit = vfft_k1_jit_resolve(N, c.R1, c.R2, c.spr);
            }
        }
        if (c.ilr == VFFT_K1_IL_MONO)
            c.monoil = vfft_k1_mono_il_fn(N, 0);
        else if (c.ilr != VFFT_K1_IL_NONE)
            c.pil = vfft_oop_plan_create_k1(N, c.iR1, c.iR2);

        c.xr = ad(N); c.xi = ad(N); c.dr = ad(N); c.di = ad(N);
        c.wr = ad(N); c.wi = ad(N);
        c.zi = ad((size_t)2 * N); c.zs = ad((size_t)2 * N);
        c.mre = ad(N); c.mim = ad(N);
        srand(42 + N);
        for (int n = 0; n < N; n++) {
            c.xr[n] = (double)rand() / RAND_MAX - 0.5;
            c.xi[n] = (double)rand() / RAND_MAX - 0.5;
            c.zi[2*n] = c.zs[2*n] = c.xr[n];
            c.zi[2*n+1] = c.zs[2*n+1] = c.xi[n];
            c.mre[n] = c.xr[n]; c.mim[n] = c.xi[n];
            c.wr[n] = c.xr[n]; c.wi[n] = c.xi[n];
        }
        DftiCreateDescriptor(&c.hi, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N);
        DftiCommitDescriptor(c.hi);
        DftiCreateDescriptor(&c.hs, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N);
        DftiSetValue(c.hs, DFTI_COMPLEX_STORAGE, DFTI_REAL_REAL);
        DftiCommitDescriptor(c.hs);

        int reps = (int)(2e6 / (double)N);
        if (reps < 100) reps = 100;
        if (reps > 400000) reps = 400000;

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

        double sp = best[A_WSP];
        const char *spwin = SPN[c.spr];
        if (arm_avail(&c, A_JIT) && best[A_JIT] < sp) { sp = best[A_JIT]; spwin = "jit"; }
        if (arm_avail(&c, A_XTRA) && best[A_XTRA] < sp) { sp = best[A_XTRA]; spwin = "twl8x8"; }
        printf("RES,%d,MKL-IL,%.1f,MKL-sp,%.1f,Wsp,%s,%dx%d,%.1f,jit,%.1f,Wil,%s,%.1f,xtra,%.1f\n",
               N, best[A_MKL_IL], best[A_MKL_SP],
               SPN[c.spr], c.R1, c.R2, best[A_WSP],
               arm_avail(&c, A_JIT) ? best[A_JIT] : 0.0,
               ILN[c.ilr], arm_avail(&c, A_WIL) ? best[A_WIL] : 0.0,
               arm_avail(&c, A_XTRA) ? best[A_XTRA] : 0.0);
        printf("VERDICT,%d,sp-best,%s,%.1f,vsMKLsp,%.2f,il,%s,%.1f,vsMKLil,%.2f\n",
               N, spwin, sp, best[A_MKL_SP] / sp,
               ILN[c.ilr], arm_avail(&c, A_WIL) ? best[A_WIL] : 0.0,
               arm_avail(&c, A_WIL) ? best[A_MKL_IL] / best[A_WIL] : 0.0);

        DftiFreeDescriptor(&c.hi); DftiFreeDescriptor(&c.hs);
        if (c.psp) vfft_oop_plan_destroy(c.psp);
        if (c.pil) vfft_oop_plan_destroy(c.pil);
        if (c.pxt) vfft_oop_plan_destroy(c.pxt);
    }
    printf("DONE\n");
    return 0;
}
