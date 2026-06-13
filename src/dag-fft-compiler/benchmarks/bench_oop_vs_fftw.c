/* bench_oop_vs_fftw.c — the OOP plan kinds vs FFTW guru PATIENT, same
 * binary, round-robin, min-of-rounds. Build per-ISA: an -mavx2 build
 * races our pure-avx2 engine against FFTW's best choice (FFTW dispatches
 * whatever its planner measures fastest; its pick is printed per cell).
 *
 * Natural-order kinds gate elementwise vs FFTW output (<=1e-9). MODEB is
 * scrambled order: timed with the caveat, correctness covered by the
 * bit-exact sweep elsewhere.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <x86intrin.h>
#include "fftw3.h"
#include "../core/executor.h"
#include "../core/oop_auto.h"

static unsigned long long mn2(unsigned long long a,unsigned long long b){return a<b?a:b;}

static void run_cell(int N, size_t K,
                     const vfft_proto_wisdom_t *wis,
                     vfft_proto_registry_t *reg)
{
    vfft_oop_pair_hint_t h = {0, 0, 0, 0};
    int nh = 0, r1 = 0, r2 = 0;
    if (vfft_oop_tune_pairs(N, K, &r1, &r2, 0) >= 2 && r1 > 0)
    {
        h.N = N; h.K = K; h.R1 = r1; h.R2 = r2; nh = 1;
    }
    vfft_oop_plan_t *p = vfft_oop_plan_create_auto(N, K, wis, &h, nh, reg);
    if (!p)
    {
        printf("  N=%-5d K=%-4zu  no plan (skipped)\n", N, K);
        return;
    }
    size_t T = (size_t)N * K;
    double *sr = aligned_alloc(64, T * 8), *si = aligned_alloc(64, T * 8);
    double *dr = aligned_alloc(64, T * 8), *di = aligned_alloc(64, T * 8);
    fftw_complex *gi = fftw_malloc(sizeof(fftw_complex) * T);
    fftw_complex *go = fftw_malloc(sizeof(fftw_complex) * T);
    srand(71 + N);
    for (size_t t = 0; t < K; t++)
        for (int e = 0; e < N; e++)
        {
            double vr = (double)rand() / RAND_MAX - 0.5;
            double vi = (double)rand() / RAND_MAX - 0.5;
            sr[(size_t)e * K + t] = vr; si[(size_t)e * K + t] = vi;
            gi[t * (size_t)N + e][0] = vr; gi[t * (size_t)N + e][1] = vi;
        }
    fftw_iodim64 d1 = {N, 1, 1}, h1 = {(ptrdiff_t)K, N, N};
    fftw_set_timelimit(20.0);
    fftw_plan pg = fftw_plan_guru64_dft(1, &d1, 1, &h1, gi, go,
                                        FFTW_FORWARD, FFTW_PATIENT);
    /* PATIENT planning scribbles on the arrays during measurement;
     * refill AFTER plan creation or the gate compares against
     * FFTW-of-garbage (found as a void gate printing exactly 0e+00) */
    for (size_t t = 0; t < K; t++)
        for (int e = 0; e < N; e++)
        {
            gi[t * (size_t)N + e][0] = sr[(size_t)e * K + t];
            gi[t * (size_t)N + e][1] = si[(size_t)e * K + t];
        }

    vfft_oop_execute_fwd(p, sr, si, dr, di);
    fftw_execute(pg);
    char gate[24];
    const char *order;
    if (p->kind != VFFT_OOP_KIND_MODEB)
    {
        double me = 0, mm = 0;
        for (size_t t = 0; t < K; t += (K > 3 ? K / 3 : 1))
            for (int k = 0; k < N; k++)
            {
                double a = dr[(size_t)k * K + t] - go[t * (size_t)N + k][0];
                double b = di[(size_t)k * K + t] - go[t * (size_t)N + k][1];
                double e2 = sqrt(a * a + b * b);
                double m = hypot(go[t * (size_t)N + k][0],
                                 go[t * (size_t)N + k][1]);
                if (e2 > me) me = e2;
                if (m > mm) mm = m;
            }
        double r = mm > 0 ? me / mm : 1.0; /* degenerate reference = FAIL */
        snprintf(gate, sizeof gate, "%.0e %s", r, r < 1e-9 ? "OK" : "FAIL");
        order = "natural";
        if (r >= 1e-9)
        {
            printf("  N=%-5d K=%-4zu GATE FAIL %s\n", N, K, gate);
            goto done;
        }
    }
    else
    {
        snprintf(gate, sizeof gate, "bit-exact*");
        order = "scrambled";
    }
    {
        enum { ROUNDS = 20 };
        unsigned long long mv = ~0ULL, mg = ~0ULL, c;
        for (int w = 0; w < 3; w++)
        {
            vfft_oop_execute_fwd(p, sr, si, dr, di);
            fftw_execute(pg);
        }
        for (int r = 0; r < ROUNDS; r++)
        {
            c = __rdtsc();
            vfft_oop_execute_fwd(p, sr, si, dr, di);
            mv = mn2(mv, __rdtsc() - c);
            c = __rdtsc();
            fftw_execute(pg);
            mg = mn2(mg, __rdtsc() - c);
        }
        const char *kn = p->kind == VFFT_OOP_KIND_LEAF ? "LEAF" :
                         p->kind == VFFT_OOP_KIND_BAILEY2 ? "BAILEY2" : "MODEB";
        char pair[16] = "";
        if (p->kind == VFFT_OOP_KIND_BAILEY2)
            snprintf(pair, sizeof pair, " %dx%d", p->R1, p->R2);
        printf("  N=%-5d K=%-4zu %-7s%-7s gate %-10s order %-9s | "
               "vfft %9llu | fftw %9llu | speed vs FFTW %.3f\n",
               N, K, kn, pair, gate, order, mv, mg, (double)mg / mv);
        printf("    fftw plan: ");
        fftw_print_plan(pg);
        printf("\n");
    }
done:
    fftw_destroy_plan(pg);
    free(sr); free(si); free(dr); free(di);
    fftw_free(gi); fftw_free(go);
    vfft_oop_plan_destroy(p);
}

int main(void)
{
    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);
    vfft_proto_wisdom_t wis;
    if (vfft_proto_wisdom_load(&wis, "/tmp/wis/wisdom_v198.txt") != 0)
    {
        printf("wisdom load FAIL\n");
        return 1;
    }
    printf("== OOP plan kinds vs FFTW guru PATIENT, OOP both sides, "
           "1 thread, higher is faster ==\n");
    run_cell(64, 512, &wis, &reg);
    run_cell(128, 512, &wis, &reg);
    run_cell(169, 512, &wis, &reg);
    run_cell(512, 120, &wis, &reg);
    run_cell(1024, 120, &wis, &reg);
    run_cell(1024, 256, &wis, &reg);
    run_cell(4096, 256, &wis, &reg);
    printf("(* MODEB correctness from the bit-exact sweep; scrambled "
           "order vs FFTW natural)\n");
    vfft_proto_wisdom_free(&wis);
    return 0;
}
