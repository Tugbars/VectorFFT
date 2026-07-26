/* spike_s0t_ingest.c — ZTURN Phase 0 (docs/roadmap/cascade_load_path_restructure.md §5).
 *
 * s0t ingest: radix-R0=4 twiddle-free leaf. Reads the user's INTERLEAVED (z) input
 * with 4 raw leg loads at stride Ls = N/4 complex (ZERO load-side shuffles),
 * computes the radix-4 DIT butterfly elementwise on interleaved registers, and
 * stores BLOCK-SPLIT 64-byte [re x4][im x4] records at the ZTURN addresses
 * rho = R0*k + a0  ==>  re of position p at sp[8p + a0], im at sp[8p + 4 + a0].
 * The z->split deinterleave and the corner turn are fused into ONE store lattice
 * (4 unpcklo + 4 unpckhi + 8 perm2f128 per 16 complex; +2 permilpd +2 xorpd for
 * the -i rotate). Budget: 18 port-5 ops per 16-complex iteration = MKL's own r4
 * ingest count (fn 0x1825c3800, loop 0x38c5..0x39f0: 10 vshufpd + 8 vperm2f128).
 *
 * Phase 0 is STATIC: correctness gate + instruction census only, no timing.
 *
 * ===================== PHASE-0 CENSUS RECORD (2026-07-26) =====================
 * Correctness: maxabs = 0.0 (bit-identical) vs the scalar index-map reference at
 * N in {2048, 4096, 16384}, BOTH compilers (gcc 13.2 mingw64, gcc 15.2 mingw152).
 * Loop body between backward-branch bounds, per 16-complex iteration (denominator
 * proven from the bytes: 8 distinct 32-B leg loads = 32 doubles = 16 complex;
 * input cursor +0x40, output cursor +0x100 = 16 complex out; no unrolling):
 *
 *   class            gcc13.2   gcc15.2   MKL r4 ingest (0x38c5..0x39f0)
 *   vunpcklpd            4         4        0
 *   vunpckhpd            4         4        0
 *   vperm2f128           4         4        8
 *   vinsertf128          4         4        0   (gcc renders perm2f128 imm 0x20
 *   vshufpd              0         0       10    as vinsertf128 $1 — same 1-uop
 *   vpermilpd            2         2        0    port-5 class on Raptor Lake)
 *   PORT-5 TOTAL        18        18       18   == MATCH (MKL's own count)
 *   vxorpd               2         2        2   (ours: mask hoisted to ymm3
 *   vadd/vsubpd         16        16       16    outside the loop; MKL: rip-mem)
 *   ymm loads            4+8f      8        8   (gcc13 folds 4 loads into add/sub
 *   ymm stores           8         8        8    mem operands and reads each twice
 *   scalar               4         4        7    = 20 mem accesses vs 16; gcc15 is
 *   TOTAL INSNS         52        56       59    the faithful 8+8. ZERO stack
 *                                                spills in either loop.)
 *   shuf+xor convention: 18 p5 + 2 vxorpd = 20 = MKL's census row "shuf+xor 20".
 * =============================================================================
 */
#include <immintrin.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Pinned -0.0 convention (isa.ml:272-281): file-scope static const aggregate,
 * MEMORY order (lane 0 first) — -0.0 in the ODD (im-position) lanes, applied
 * AFTER the swap:  x*(-i) == xor(permilpd(x,0x5), _M_IM).
 * NEVER _mm256_set_pd here: its args are high-to-low, set_pd(0,-0.0,0,-0.0)
 * silently builds the BACKWARD mask and the static census cannot catch it. */
static const __m256d _M_IM = { 0.0, -0.0, 0.0, -0.0 };

/* ---------------------------------------------------------------- THE BODY */
__attribute__((noinline, target("avx2,fma")))
void s0t_fwd(const double * __restrict zin, double * __restrict sp, size_t N)
{
    const size_t Ls = N / 4;              /* leg stride, complex; contract Ls % 4 == 0 */
    for (size_t k = 0; k + 4 <= Ls; k += 4) {   /* one iter = 4 positions = 16 complex */
        /* ---------------- half A: positions k, k+1 ---------------- */
        /* raw interleaved leg loads: x_l = zin complex l*Ls + {k,k+1} */
        const __m256d a0 = _mm256_loadu_pd(&zin[2*(0*Ls + k)]);
        const __m256d a1 = _mm256_loadu_pd(&zin[2*(1*Ls + k)]);
        const __m256d a2 = _mm256_loadu_pd(&zin[2*(2*Ls + k)]);
        const __m256d a3 = _mm256_loadu_pd(&zin[2*(3*Ls + k)]);
        /* radix-4 twiddle-free DIT, elementwise on interleaved regs */
        const __m256d at0 = _mm256_add_pd(a0, a2);
        const __m256d at1 = _mm256_sub_pd(a0, a2);
        const __m256d at2 = _mm256_add_pd(a1, a3);
        const __m256d at3 = _mm256_sub_pd(a1, a3);
        const __m256d ar  = _mm256_xor_pd(_mm256_permute_pd(at3, 0x5), _M_IM); /* -i*t3 */
        const __m256d aY0 = _mm256_add_pd(at0, at2);
        const __m256d aY2 = _mm256_sub_pd(at0, at2);
        const __m256d aY1 = _mm256_add_pd(at1, ar);   /* t1 - i*t3 */
        const __m256d aY3 = _mm256_sub_pd(at1, ar);   /* t1 + i*t3 */
        /* fused deinterleave + corner turn: unpacklo gathers the four RE parts,
         * unpackhi the four IM parts (z->split), while unpack+perm2f128 turns the
         * register axis (output digit a0) into the lane axis (ZTURN). */
        const __m256d au0 = _mm256_unpacklo_pd(aY0, aY1);
        const __m256d au1 = _mm256_unpackhi_pd(aY0, aY1);
        const __m256d au2 = _mm256_unpacklo_pd(aY2, aY3);
        const __m256d au3 = _mm256_unpackhi_pd(aY2, aY3);
        _mm256_storeu_pd(&sp[8*k +  0], _mm256_permute2f128_pd(au0, au2, 0x20)); /* re(k)   */
        _mm256_storeu_pd(&sp[8*k +  4], _mm256_permute2f128_pd(au1, au3, 0x20)); /* im(k)   */
        _mm256_storeu_pd(&sp[8*k +  8], _mm256_permute2f128_pd(au0, au2, 0x31)); /* re(k+1) */
        _mm256_storeu_pd(&sp[8*k + 12], _mm256_permute2f128_pd(au1, au3, 0x31)); /* im(k+1) */
        /* ---------------- half B: positions k+2, k+3 ---------------- */
        const __m256d b0 = _mm256_loadu_pd(&zin[2*(0*Ls + k) + 4]);
        const __m256d b1 = _mm256_loadu_pd(&zin[2*(1*Ls + k) + 4]);
        const __m256d b2 = _mm256_loadu_pd(&zin[2*(2*Ls + k) + 4]);
        const __m256d b3 = _mm256_loadu_pd(&zin[2*(3*Ls + k) + 4]);
        const __m256d bt0 = _mm256_add_pd(b0, b2);
        const __m256d bt1 = _mm256_sub_pd(b0, b2);
        const __m256d bt2 = _mm256_add_pd(b1, b3);
        const __m256d bt3 = _mm256_sub_pd(b1, b3);
        const __m256d br  = _mm256_xor_pd(_mm256_permute_pd(bt3, 0x5), _M_IM);
        const __m256d bY0 = _mm256_add_pd(bt0, bt2);
        const __m256d bY2 = _mm256_sub_pd(bt0, bt2);
        const __m256d bY1 = _mm256_add_pd(bt1, br);
        const __m256d bY3 = _mm256_sub_pd(bt1, br);
        const __m256d bu0 = _mm256_unpacklo_pd(bY0, bY1);
        const __m256d bu1 = _mm256_unpackhi_pd(bY0, bY1);
        const __m256d bu2 = _mm256_unpacklo_pd(bY2, bY3);
        const __m256d bu3 = _mm256_unpackhi_pd(bY2, bY3);
        _mm256_storeu_pd(&sp[8*k + 16], _mm256_permute2f128_pd(bu0, bu2, 0x20)); /* re(k+2) */
        _mm256_storeu_pd(&sp[8*k + 20], _mm256_permute2f128_pd(bu1, bu3, 0x20)); /* im(k+2) */
        _mm256_storeu_pd(&sp[8*k + 24], _mm256_permute2f128_pd(bu0, bu2, 0x31)); /* re(k+3) */
        _mm256_storeu_pd(&sp[8*k + 28], _mm256_permute2f128_pd(bu1, bu3, 0x31)); /* im(k+3) */
    }
}

/* ----------------------------------------------- SCALAR REFERENCE (index map)
 * Independent transcription of the roadmap's derived index map, eq (2)/(3):
 * source complex of (leg l, position p) = l*D[0] + p in natural z;
 * destination rho = 4*p + a0, i.e. re of Y_{a0} at sp[8p+a0], im at sp[8p+4+a0].
 * Plain loops, no intrinsics. Same Step-7 butterfly, scalar complex arithmetic. */
void s0t_ref(const double *zin, double *sp, size_t N)
{
    const size_t Ls = N / 4;
    for (size_t p = 0; p < Ls; p++) {
        double xr[4], xi[4], Yr[4], Yi[4];
        for (int l = 0; l < 4; l++) {
            xr[l] = zin[2*((size_t)l*Ls + p)];
            xi[l] = zin[2*((size_t)l*Ls + p) + 1];
        }
        const double t0r = xr[0]+xr[2], t0i = xi[0]+xi[2];
        const double t1r = xr[0]-xr[2], t1i = xi[0]-xi[2];
        const double t2r = xr[1]+xr[3], t2i = xi[1]+xi[3];
        const double t3r = xr[1]-xr[3], t3i = xi[1]-xi[3];
        Yr[0] = t0r + t2r;  Yi[0] = t0i + t2i;
        Yr[2] = t0r - t2r;  Yi[2] = t0i - t2i;
        Yr[1] = t1r + t3i;  Yi[1] = t1i - t3r;   /* Y1 = t1 - i*t3 (fwd) */
        Yr[3] = t1r - t3i;  Yi[3] = t1i + t3r;   /* Y3 = t1 + i*t3 */
        for (int j = 0; j < 4; j++) {
            sp[8*p + j]     = Yr[j];
            sp[8*p + 4 + j] = Yi[j];
        }
    }
}

/* ------------------------------------------------------------------ DRIVER */
static double run_one(size_t N)
{
    double *zin = (double *)_mm_malloc(2*N*sizeof(double), 64);
    double *spA = (double *)_mm_malloc(2*N*sizeof(double), 64);
    double *spB = (double *)_mm_malloc(2*N*sizeof(double), 64);
    if (!zin || !spA || !spB) { fprintf(stderr, "alloc fail\n"); exit(2); }
    for (size_t i = 0; i < N; i++) {                 /* deterministic, non-symmetric */
        zin[2*i]   = sin(0.7*(double)i) + (double)i*1e-3;
        zin[2*i+1] = cos(1.3*(double)i);
    }
    for (size_t i = 0; i < 2*N; i++) { spA[i] = 0.0; spB[i] = 1.0; } /* distinct fill */
    s0t_fwd(zin, spA, N);
    s0t_ref(zin, spB, N);
    double maxabs = 0.0;
    for (size_t i = 0; i < 2*N; i++) {
        double d = fabs(spA[i] - spB[i]);
        if (d > maxabs) maxabs = d;
    }
    _mm_free(zin); _mm_free(spA); _mm_free(spB);
    return maxabs;
}

int main(void)
{
    static const size_t Ns[3] = { 2048, 4096, 16384 };
    int fail = 0;
    for (int i = 0; i < 3; i++) {
        double m = run_one(Ns[i]);
        int ok = (m <= 1e-13);
        printf("N=%6zu  maxabs=%.3e  %s\n", Ns[i], m, ok ? "PASS" : "FAIL");
        if (!ok) fail = 1;
    }
    printf(fail ? "RESULT: FAIL\n" : "RESULT: PASS\n");
    return fail;
}
