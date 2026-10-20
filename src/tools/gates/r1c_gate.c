/* r1c_gate.c -- the real flat leaf (radixR_z_r1c_{fwd,bwd} at the build's ISA,
 * the real flat DIT's one kind) against a naive digit transform, at every odd radix
 * of the kind and at counts that reach the wide loop, the two-column step
 * and the lone last column (1, 2, 3, 4, 5, 7, 9, 15).
 *   fwd: R real legs (leg l of column k at zin[l*D + k]) -> digit 0 real at
 *        zout[k], digit p in 1..(R-1)/2 complex at zout[2*p*D + 2*k]:
 *        digit p = sum_l x[l*D + k] * exp(-2 pi i l p / R).
 *   bwd: the digit runs -> R real legs, R times the input.
 * The second half of block 0 (doubles D..2D) is never written: checked.
 * The kernels link as their own objects (each file declares its own static
 * constants). They are the build's ISA's (il_isa.h): each radix resolves
 * through the registry's R1C list. An absent radix FAILS, except at avx512
 * where the registry declares the kind absent (its R1C list defined and empty:
 * the DECLARED ABSENT block, Corpus.zil_isa_gap): reported, not gated. At avx2
 * no kind is declared absent, so every absent radix FAILS.
 * Build: python gauntlet/build.py --compile --src src/tools/gates/r1c_gate.c (VFFT_ISA=avx512 for the avx512 kernels) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include "il_isa.h"   /* the IL registry at the build's ISA: the kernels' declarations and radix lists */

typedef void (*fn11)(const double *, const double *, double *, double *, const double *, const double *, size_t, size_t, size_t, size_t, size_t);
static const int K[] = { 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 37, 41, 43, 47 };

/* the kernel of radix R in one direction at the build's ISA, 0 when the registry lists none */
static fn11 r1c_fn(int R, int bwd)
{
    switch (R)
    {
#ifdef VFFT_IL_R1C_PAIR_RADICES
#define C(n) case n: return bwd ? VFFT_IL_SYM(radix##n##_z_r1c_bwd) : VFFT_IL_SYM(radix##n##_z_r1c_fwd);
    VFFT_IL_R1C_PAIR_RADICES(C)
#undef C
#endif
    default: return 0;
    }
}

/* the kind is declared absent at the build's ISA: the registry defines its list
 * and leaves it empty (il_registry_<isa>.h: "every kind of the union over ISAs
 * is defined (empty = absent at this ISA)"); at avx2 the corpus is whole
 * (CMakeLists.txt's VFFT_ABSENT_avx2 is empty) and nothing is declared */
static int declared_absent(void)
{
#if defined(VFFT_BUILD_ISA_AVX512)
#define C(n) + 1
    return 0 VFFT_IL_R1C_PAIR_RADICES(C) == 0;
#undef C
#else
    return 0;
#endif
}

int main(void)
{
    static const int counts[] = { 1, 2, 3, 4, 5, 7, 9, 15 };
    const double PI = 3.14159265358979323846;
    const double GUARD = 12345.0;
    const int nk = (int)(sizeof K / sizeof K[0]);
    int fails = 0, nabsent = 0;
    unsigned seed = 0x5eedu;
    for (int ki = 0; ki < nk; ki++)
    {
        const int R = K[ki], h = R / 2;
        const fn11 fwd = r1c_fn(R, 0), bwd = r1c_fn(R, 1);
        double worst_f = 0, worst_b = 0;
        if (!fwd || !bwd)
        {
            if (!declared_absent())
            {   /* a radix the registry dropped from a kind it lists at this ISA */
                printf("R=%-2d absent at %s, not declared absent there  *** FAIL ***\n", R, VFFT_IL_ISA_NAME);
                fails++;
                continue;
            }
            printf("R=%-2d declared absent at %s: not gated\n", R, VFFT_IL_ISA_NAME);
            nabsent++;
            continue;
        }
        for (int ci = 0; ci < (int)(sizeof counts / sizeof counts[0]); ci++)
        {
            const size_t D = (size_t)counts[ci], N = (size_t)R * D, P = ((size_t)R + 1) * D;
            double *x = (double *)_aligned_malloc((N + 8) * 8, 64), *y = (double *)_aligned_malloc((N + 8) * 8, 64);
            double *pl = (double *)_aligned_malloc((P + 8) * 8, 64);
            for (size_t i = 0; i < N; i++) { seed = seed * 1664525u + 1013904223u; x[i] = (double)(seed >> 8) / (double)(1u << 24) - 0.5; }
            for (size_t i = 0; i < P + 8; i++) pl[i] = GUARD;
            fwd(x, NULL, pl, NULL, NULL, NULL, D, 0, D, 0, D);
            for (size_t k = 0; k < D; k++)
                for (int p = 0; p <= h; p++)
                {
                    long double re = 0, im = 0;
                    for (int l = 0; l < R; l++)
                    {
                        const long double a = -2.0L * (long double)PI * (long double)((l * p) % R) / (long double)R;
                        re += (long double)x[(size_t)l * D + k] * cosl(a);
                        im += (long double)x[(size_t)l * D + k] * sinl(a);
                    }
                    const double gr = p ? pl[2 * (size_t)p * D + 2 * k] : pl[k];
                    const double gi = p ? pl[2 * (size_t)p * D + 2 * k + 1] : 0.0;
                    const double e = fabs((double)(re - gr)) + fabs((double)(im - gi));
                    if (e > worst_f) worst_f = e;
                }
            for (size_t i = D; i < 2 * D; i++) if (pl[i] != GUARD) { printf("R=%d count=%zu: fwd wrote the unused half of block 0\n", R, D); fails++; break; }
            for (size_t i = P; i < P + 8; i++) if (pl[i] != GUARD) { printf("R=%d count=%zu: fwd wrote past the plane\n", R, D); fails++; break; }
            for (size_t i = 0; i < N + 8; i++) y[i] = GUARD;
            bwd(pl, NULL, y, NULL, NULL, NULL, D, 0, D, 0, D);
            for (size_t i = 0; i < N; i++) { const double e = fabs(y[i] - (double)R * x[i]) / (double)R; if (e > worst_b) worst_b = e; }
            for (size_t i = N; i < N + 8; i++) if (y[i] != GUARD) { printf("R=%d count=%zu: bwd wrote past the legs\n", R, D); fails++; break; }
            _aligned_free(x); _aligned_free(y); _aligned_free(pl);
        }
        const int ok = worst_f < 1e-13 && worst_b < 1e-13;
        if (!ok) fails++;
        printf("R=%-2d fwd max err %.1e  roundtrip max err %.1e  %s\n", R, worst_f, worst_b, ok ? "ok" : "*** FAIL ***");
    }
    if (nabsent)
        printf("%d of %d radices declared absent at %s: not gated\n", nabsent, nk, VFFT_IL_ISA_NAME);
    if (nabsent == nk)
        printf("no r1c kernel at %s: nothing gated, every radix declared absent\n", VFFT_IL_ISA_NAME);
    else if (nabsent && !fails)
        printf("ALL PASS on the %d radices present at %s (%d declared absent, not gated)\n", nk - nabsent, VFFT_IL_ISA_NAME, nabsent);
    else
        printf("%s\n", fails ? "FAILURES" : "ALL PASS");
    return fails ? 1 : 0;
}
