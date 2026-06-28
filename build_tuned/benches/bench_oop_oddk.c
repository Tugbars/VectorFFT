/* bench_oop_oddk.c — OOP c2c odd-K margin vs MKL, compared to v1_0_results.md.
 *
 * Mirrors bench_1d_vs_mkl.c --oop methodology: front-door plan
 * (vfft_oop_plan_create_dp_best), MKL DFTI NOT_INPLACE split (REAL_REAL), best-of-5
 * min per engine, cachebust + cool between engines, order-flip (both flips reported;
 * flip0 ~ flip1 = fair). Correctness = OOP roundtrip fwd+bwd == N*x.
 *
 * Benches the same even-K cells as the v1.0 OOP table (so the even row is a
 * regression check — the M-fence removal should make OOP equal-or-better) plus
 * each cell's odd neighbours. v1.0 reference printed inline.
 *
 * Build: python build.py --src benches/bench_oop_oddk.c --mkl
 * Run  : needs MKL on PATH + MKL_THREADING_LAYER=SEQUENTIAL.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "executor.h"
#include "planner.h"
#include "oop_dp.h"
#include "oop_wisdom.h"
#include <mkl_dfti.h>
#include <mkl_service.h>

static double *ad(size_t n)
{
    double *p = NULL;
    if (vfft_proto_posix_memalign((void **)&p, 64, n * sizeof(double)) != 0) { exit(1); }
    return p;
}
static void afree(double *p) { vfft_proto_aligned_free(p); }
static void pace(int ms) { if (ms > 0) { struct timespec ts = {ms/1000,(long)(ms%1000)*1000000L}; nanosleep(&ts, NULL);} }
static void cachebust(void)
{
    size_t s = 32*1024*1024/sizeof(double); double *j = ad(s);
    for (size_t i=0;i<s;i++) j[i]=(double)i; volatile double a=0; for (size_t i=0;i<s;i++) a+=j[i]; (void)a; afree(j);
}
static int reps_for(size_t total) { int r = (int)(50000000ull/(total?total:1)); return r<200?200:r; }

static DFTI_DESCRIPTOR_HANDLE mkl_make_oop(int N, size_t K)
{
    DFTI_DESCRIPTOR_HANDLE d = NULL; MKL_LONG str[2] = {0,(MKL_LONG)K};
    if (DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N) != DFTI_NO_ERROR) return NULL;
    DftiSetValue(d, DFTI_COMPLEX_STORAGE, DFTI_REAL_REAL);
    DftiSetValue(d, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(d, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
    DftiSetValue(d, DFTI_INPUT_DISTANCE, 1); DftiSetValue(d, DFTI_OUTPUT_DISTANCE, 1);
    DftiSetValue(d, DFTI_INPUT_STRIDES, str); DftiSetValue(d, DFTI_OUTPUT_STRIDES, str);
    if (DftiCommitDescriptor(d) != DFTI_NO_ERROR) { DftiFreeDescriptor(&d); return NULL; }
    return d;
}
static double best_mkl(int N, size_t K, const double *sr, const double *si, size_t total)
{
    DFTI_DESCRIPTOR_HANDLE d = mkl_make_oop(N, K); if (!d) return 0;
    double *mr = ad(total), *mi = ad(total);
    for (int w=0;w<10;w++) DftiComputeForward(d,(void*)sr,(void*)si,mr,mi);
    int reps = reps_for(total); double best = 1e18;
    for (int t=0;t<5;t++){ double t0=vfft_proto_now_ns(); for(int i=0;i<reps;i++) DftiComputeForward(d,(void*)sr,(void*)si,mr,mi); double e=(vfft_proto_now_ns()-t0)/reps; if(e<best)best=e; }
    afree(mr); afree(mi); DftiFreeDescriptor(&d); return best;
}
static double best_vfft(vfft_oop_plan_t *p, const double *sr, const double *si, double *dr, double *di, size_t total)
{
    for (int w=0;w<10;w++) vfft_oop_execute_fwd(p, sr, si, dr, di);
    int reps = reps_for(total); double best = 1e18;
    for (int t=0;t<5;t++){ double t0=vfft_proto_now_ns(); for(int i=0;i<reps;i++) vfft_oop_execute_fwd(p,sr,si,dr,di); double e=(vfft_proto_now_ns()-t0)/reps; if(e<best)best=e; }
    return best;
}

static void cell(int N, size_t K, vfft_proto_registry_t *reg, const vfft_oop_wisdom_t *w,
                 int cool_ms, const char *v1ref)
{
    /* same as v1.0: calibrated wisdom lookup first (pure build, no measure);
     * miss (e.g. odd K, never calibrated) -> dp_best. */
    vfft_oop_plan_t *p = w ? vfft_oop_plan_create_wisdom(N, K, w, reg) : NULL;
    const char *src = p ? "wis" : "dp";
    if (!p) {
        vfft_proto_dp_context_t ctx; vfft_proto_dp_init(&ctx, K, N);
        p = vfft_oop_plan_create_dp_best(N, K, &ctx, reg);
        vfft_proto_dp_destroy(&ctx);
    }
    if (!p) { printf("  N=%-5d K=%-4zu  NO PLAN\n", N, K); return; }
    const char *kind = p->kind==VFFT_OOP_KIND_LEAF?"LEAF":p->kind==VFFT_OOP_KIND_BAILEY2?"BAILEY2":"MODEB";
    const char *order = p->kind==VFFT_OOP_KIND_MODEB?"scram":"nat";
    size_t total=(size_t)N*K;
    double *sr=ad(total),*si=ad(total),*dr=ad(total),*di=ad(total),*er=ad(total),*ei=ad(total);
    srand(42+N+(int)K);
    for(size_t i=0;i<total;i++){ sr[i]=(double)rand()/RAND_MAX-0.5; si[i]=(double)rand()/RAND_MAX-0.5; }
    vfft_oop_execute_fwd(p,sr,si,dr,di); vfft_oop_execute_bwd(p,dr,di,er,ei);
    double rt=0; for(size_t i=0;i<total;i++){ double a=fabs(er[i]/N-sr[i]),b=fabs(ei[i]/N-si[i]); if(a>rt)rt=a; if(b>rt)rt=b; }
    double r0,r1;
    { double m=best_mkl(N,K,sr,si,total); cachebust(); pace(cool_ms); double v=best_vfft(p,sr,si,dr,di,total); r1=m/v; } /* flip1: mkl first */
    { double v=best_vfft(p,sr,si,dr,di,total); cachebust(); pace(cool_ms); double m=best_mkl(N,K,sr,si,total); r0=m/v; } /* flip0: vfft first */
    printf("  N=%-5d K=%-4zu rem%zu %-7s %-5s %-3s rt=%.0e  flip0=%.2fx flip1=%.2fx  avg=%.2fx   %s\n",
           N, K, K%4, kind, order, src, rt, r0, r1, (r0+r1)/2, v1ref?v1ref:"");
    afree(sr);afree(si);afree(dr);afree(di);afree(er);afree(ei);
    vfft_oop_plan_destroy(p);
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    mkl_set_num_threads(1);
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    int cool = 80;

    /* calibrated OOP wisdom — same source the v1.0 OOP table used (K=32 cells). */
    vfft_oop_wisdom_t oopw; int have = 0;
    const char *paths[] = {
        "../../src/dag-fft-compiler/generator/generated/oop_wisdom.txt",
        "oop_wisdom.txt",
        "../benches/oop_wisdom.txt",
    };
    const char *used = NULL;
    for (int i = 0; i < 3 && !have; i++) {
        if (vfft_oop_wisdom_load(&oopw, paths[i]) == 0) { have = 1; used = paths[i]; }
    }
    const vfft_oop_wisdom_t *w = have ? &oopw : NULL;
    printf("OOP c2c odd-K vs MKL (NOT_INPLACE split, measure_ab). wisdom=%s [%s].\n",
           have ? "loaded" : "MISS->dp", used ? used : "-");
    printf("src: wis=calibrated wisdom plan (even K), dp=dp_best (odd K, no wisdom entry)\n");

    /* even K=32 = the calibrated v1.0 cells; 31/33 = the odd neighbours. */
    int Ns[] = {8, 16, 64, 256, 1024};
    for (int i = 0; i < 5; i++) {
        int N = Ns[i];
        printf("== N=%d ==\n", N);
        cell(N, 32, &reg, w, cool, N==8?"<- v1.0 K=32 10.78x":N==16?"<- v1.0 K=32 5.97x":"(K=32 even, calibrated)");
        cell(N, 31, &reg, w, cool, NULL);
        cell(N, 33, &reg, w, cool, NULL);
    }
    return 0;
}
