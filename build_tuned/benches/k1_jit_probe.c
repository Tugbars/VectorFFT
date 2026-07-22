/* k1_jit_probe.c — direct A/B: the K=1 JIT DLL's exec vs the unbaked route vs
 * the AOT spec twins, same buffers, same process. Isolates codegen quality
 * from vfft dispatch. Build: python build.py --src benches/k1_jit_probe.c --compile */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>

#include "executor.h"
#include "planner.h"
#include "oop_plan.h"
#include "k1_jit_runtime.h"

typedef vfft_k1_jit_fn jit_fn;

extern void radix32_n1_oop_fwd_avx2_UG_UL_spec32_1_1_32(
    const double *, const double *, double *, double *,
    const double *, const double *, size_t);
extern void radix32_t1_oop_fwd_avx2_UG_UG_spec32_1_32_1(
    const double *, const double *, double *, double *,
    const double *, const double *, size_t);

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

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg); (void)reg;

    jit_fn jf = vfft_k1_jit_resolve(1024, 32, 32, 2 /* 2PB */);
    printf("exec=%p (ver%d)\n", (void *)jf, VFFT_K1JIT_VERSION);
    if (!jf) return 1;

    int N = 1024, R1 = 32, R2 = 32;
    vfft_oop_plan_t *p = vfft_oop_plan_create_k1(N, R1, R2);
    double *xr = ad(N), *xi = ad(N), *dr = ad(N), *di = ad(N), *er = ad(N), *ei = ad(N);
    srand(11);
    for (int n = 0; n < N; n++) {
        xr[n] = (double)rand() / RAND_MAX - 0.5;
        xi[n] = (double)rand() / RAND_MAX - 0.5;
    }
    /* correctness: jit vs unbaked route (expect bit-identical) */
    vfft_oop_execute_fwd_2pb(p, xr, xi, er, ei);
    jf(xr, xi, dr, di, p->col_re, p->col_im, p->Qr, p->Qi);
    double e = 0;
    for (int k = 0; k < N; k++) {
        double c1 = fabs(dr[k] - er[k]), c2 = fabs(di[k] - ei[k]);
        if (c1 > e) e = c1;
        if (c2 > e) e = c2;
    }
    printf("jit vs unbaked: %.1e\n", e);

    int reps = 2000;
    double bj = 1e18, bu = 1e18, bs = 1e18;
    for (int t = 0; t < 5; t++) {
        if (t) cachebust();
        for (int w = 0; w < 10; w++) jf(xr, xi, dr, di, p->col_re, p->col_im, p->Qr, p->Qi);
        double t0 = now_ms();
        for (int i = 0; i < reps; i++) jf(xr, xi, dr, di, p->col_re, p->col_im, p->Qr, p->Qi);
        double ns = (now_ms() - t0) * 1e6 / reps;
        if (ns < bj) bj = ns;
        for (int w = 0; w < 10; w++) vfft_oop_execute_fwd_2pb(p, xr, xi, dr, di);
        t0 = now_ms();
        for (int i = 0; i < reps; i++) vfft_oop_execute_fwd_2pb(p, xr, xi, dr, di);
        ns = (now_ms() - t0) * 1e6 / reps;
        if (ns < bu) bu = ns;
        for (int w = 0; w < 10; w++) {
            radix32_n1_oop_fwd_avx2_UG_UL_spec32_1_1_32(xr, xi, p->col_re, p->col_im, 0, 0, 32);
            radix32_t1_oop_fwd_avx2_UG_UG_spec32_1_32_1(p->col_re, p->col_im, dr, di, p->Qr, p->Qi, 32);
        }
        t0 = now_ms();
        for (int i = 0; i < reps; i++) {
            radix32_n1_oop_fwd_avx2_UG_UL_spec32_1_1_32(xr, xi, p->col_re, p->col_im, 0, 0, 32);
            radix32_t1_oop_fwd_avx2_UG_UG_spec32_1_32_1(p->col_re, p->col_im, dr, di, p->Qr, p->Qi, 32);
        }
        ns = (now_ms() - t0) * 1e6 / reps;
        if (ns < bs) bs = ns;
    }
    printf("jit=%.0fns  unbaked-2pb=%.0fns  aot-spec=%.0fns\n", bj, bu, bs);
    return 0;
}
