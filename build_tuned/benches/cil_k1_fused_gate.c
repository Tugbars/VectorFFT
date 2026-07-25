/* FULL-IL FUSED K=1 gate + race vs MKL.
 * One function per N, interleaved end to end, compile-time twiddles.
 * MKL here is in its HOME configuration (DFTI_COMPLEX interleaved, K=1,
 * contiguous) — the exact setup docs/research reverse-engineered, so this
 * is a no-handicap comparison. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include <mkl_dfti.h>

#define D(fn) extern void fn(const double * __restrict__, double * __restrict__);
D(vfft_cil_16_fwd_avx2)   D(vfft_cil_16_bwd_avx2)
D(vfft_cil_64_fwd_avx2)   D(vfft_cil_64_bwd_avx2)
D(vfft_cil_256_fwd_avx2)  D(vfft_cil_256_bwd_avx2)
D(vfft_cil_1024_fwd_avx2) D(vfft_cil_1024_bwd_avx2)
typedef void (*k1fn)(const double * __restrict__, double * __restrict__);

static double qpc_ms(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return 1000.0 * (double)c.QuadPart / (double)f.QuadPart;
}
static char *g_bust;
#define BUST_SZ (32u * 1024u * 1024u)
static void cachebust(void) { for (size_t i = 0; i < BUST_SZ; i += 64) g_bust[i]++; }
static double urand(unsigned *s)
{
    *s = *s * 1664525u + 1013904223u;
    return ((double)(*s >> 8) / (double)(1u << 24)) - 0.5;
}

int main(void)
{
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    g_bust = (char *)malloc(BUST_SZ); memset(g_bust, 1, BUST_SZ);

    struct { int N; k1fn f, b; } C[] = {
        { 16,   vfft_cil_16_fwd_avx2,   vfft_cil_16_bwd_avx2 },
        { 64,   vfft_cil_64_fwd_avx2,   vfft_cil_64_bwd_avx2 },
        { 256,  vfft_cil_256_fwd_avx2,  vfft_cil_256_bwd_avx2 },
        { 1024, vfft_cil_1024_fwd_avx2, vfft_cil_1024_bwd_avx2 },
    };
    int ok = 1;
    printf("%6s | %10s %10s | %9s %9s %8s\n",
           "N", "vs MKL err", "rtrip err", "CIL ns", "MKL ns", "vs MKL");
    for (int c = 0; c < 4; c++) {
        int N = C[c].N; size_t nd = 2 * (size_t)N;
        double *in  = (double *)_aligned_malloc(nd * 8, 64);
        double *out = (double *)_aligned_malloc(nd * 8, 64);
        double *rt  = (double *)_aligned_malloc(nd * 8, 64);
        double *mk  = (double *)_aligned_malloc(nd * 8, 64);
        unsigned seed = 77 + N;
        for (size_t i = 0; i < nd; i++) in[i] = urand(&seed);

        DFTI_DESCRIPTOR_HANDLE d;
        DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N);
        DftiCommitDescriptor(d);

        C[c].f(in, out);
        memcpy(mk, in, nd * 8); DftiComputeForward(d, mk);
        double fe = 0, sc = 0;
        for (size_t i = 0; i < nd; i++) {
            double dd = fabs(out[i] - mk[i]); if (dd > fe) fe = dd;
            if (fabs(mk[i]) > sc) sc = fabs(mk[i]);
        }
        fe = sc > 0 ? fe / sc : fe;

        C[c].b(out, rt);
        double re = 0, rs = 0;
        for (size_t i = 0; i < nd; i++) {
            double dd = fabs(rt[i] - (double)N * in[i]); if (dd > re) re = dd;
            double v = fabs((double)N * in[i]); if (v > rs) rs = v;
        }
        re = rs > 0 ? re / rs : re;

        int reps = (int)(3e6 / (double)N); if (reps < 50) reps = 50; if (reps > 5000) reps = 5000;
        double bc = 1e30, bm = 1e30;
        for (int r = 0; r < 9; r++) {
            for (int a = 0; a < 2; a++) {
                int arm = (a + r) & 1;
                cachebust();
                if (arm == 0) {
                    C[c].f(in, out);
                    double t0 = qpc_ms();
                    for (int i = 0; i < reps; i++) C[c].f(in, out);
                    double ns = (qpc_ms() - t0) * 1e6 / reps; if (ns < bc) bc = ns;
                } else {
                    memcpy(mk, in, nd * 8); DftiComputeForward(d, mk);
                    double t0 = qpc_ms();
                    for (int i = 0; i < reps; i++) DftiComputeForward(d, mk);
                    double ns = (qpc_ms() - t0) * 1e6 / reps; if (ns < bm) bm = ns;
                }
                Sleep(20);
            }
            Sleep(60);
        }
        printf("%6d | %10.2e %10.2e | %9.1f %9.1f %7.2fx%s\n",
               N, fe, re, bc, bm, bm / bc,
               (fe <= 1e-13 && re <= 1e-13) ? "" : "   <-- ACCURACY FAIL");
        if (fe > 1e-13 || re > 1e-13) ok = 0;
        DftiFreeDescriptor(&d);
        _aligned_free(in); _aligned_free(out); _aligned_free(rt); _aligned_free(mk);
    }
    printf("\n%s\n", ok ? "FUSED K=1 IL GATE: PASS" : "FUSED K=1 IL GATE: FAIL");
    free(g_bust);
    return ok ? 0 : 1;
}
