/* blocked_tail_gate.c — does a BLOCKED cil kernel with the new odd-count tail
 * compute the same thing as its monolithic twin, at EVERY count?
 *
 * BACKGROUND. Blocked cil kernels shipped with an even-count contract: their
 * bulk loop steps `k += per` (per = 2 on AVX2) and there was no residual arm,
 * so an odd `count` left the last column NEVER WRITTEN. The plan level worked
 * around it by refusing blocked whenever the partner factor was odd
 * (`count_ok` in il2p.h), which silently demoted ~20 cells of the form
 * N = 32*odd / 64*odd to the monolithic kernel.
 *
 * 2026-08-23 gave blocked the same inline narrow arm the monolithic kernels
 * have carried since 2026-07-29. THIS GATE IS THE ACCEPTANCE TEST FOR THAT.
 *
 * WHAT IT PROVES, per count in 1..9:
 *   1. EVERY output column is written  — a canary prefill that survives means
 *      no store reached that address. That is the exact failure the missing
 *      tail caused, so it is checked directly rather than inferred from a
 *      value comparison (a stale column can coincidentally hold a plausible
 *      number, but it cannot hold the canary and be correct).
 *   2. NOTHING outside the output region is written — guard bands on both
 *      sides, checked byte-wise.
 *   3. The values MATCH the monolithic twin. Note this is a tolerance
 *      comparison, not bitwise: the bulk runs the BLOCKED construction while
 *      the tail runs the MONOLITHIC one, and blocked forms already differ
 *      from their monolithic twins at ~1e-16 (they were A/B'd 12/12 at that
 *      level when introduced). A tolerance of 1e-12 rel is the same bar
 *      vfft_il2p_create uses when it races these kernels.
 *
 * ODD counts are the point, but EVEN counts are checked too: the change
 * hoisted `k` out of the bulk loop for blocked, which must not disturb the
 * even path that every shipping cell uses today.
 *
 * THE BUILD'S ISA (il_isa.h). The corpus kernels are VFFT_IL_SYM's, declared
 * by the registry. The r32 2.16 leaf and the tangent / wing32 forms are built
 * outside the corpus (design D3) and exist at avx2 only: VFFT_IL_AVX2_ONLY
 * names them, 0 at avx512, where that declaration makes their arms absent:
 * reported, not gated. An absent form at avx2 is NOT CORRECT. The
 * twiddle table is sized by the build's record (VFFT_IL_TWREC doubles per
 * VFFT_IL_TWPER columns).
 *
 * Build (from the repo root):
 *   python gauntlet/build.py --compile --src src/tools/gates/blocked_tail_gate.c
 *   (VFFT_ISA=avx512 for the avx512 kernels)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "il_isa.h"   /* the IL registry at the build's ISA: VFFT_IL_SYM, VFFT_IL_AVX2_ONLY, the record width */

#define GUARD  64          /* doubles of guard band on each side */
#define CANARY (-9.87654321e300)

typedef void (*kfn)(const double *, const double *, double *, double *,
                    const double *, const double *,
                    size_t, size_t, size_t, size_t, size_t);

#define DECL(SYM) void SYM(const double *, const double *, double *, double *, \
                           const double *, const double *,                     \
                           size_t, size_t, size_t, size_t, size_t);
DECL(radix32_z_n1tb_fwd_avx2)
/* TANGENT / wing32 forms. These gained the tail in 4b34dcd8 (a commit whose
   message is about FFTW bench metrics) and were never covered here, while
   il2p.h's count_ok gates were lifted for variants 3 and 4 too -- so an odd
   partner count can now reach them. Their own headers still claim
   "count %% 2 == 0 (blocked, no odd tail)", which is what makes verifying
   the CODE rather than the comment the point. */
DECL(radix32_z_n1tbw32_fwd_avx2)     DECL(radix32_z_n1tbw32t256_fwd_avx2)
DECL(radix32_z_t2bw32_fwd_avx2)      DECL(radix32_z_t2bw32m128_fwd_avx2)
#undef DECL

static uint64_t lcg = 0x243F6A8885A308D3ull;
static double rnd(void)
{ lcg = lcg*6364136223846793005ull + 1442695040888963407ull;
  return ((double)(int64_t)(lcg>>11))/4503599627370496.0; }

static int g_fail = 0;

static void arm(const char *what, int R, kfn blk, kfn mono, int inplace,
                size_t count)
{
    /* leaf n1t: OOP corner turn — reads zin[2*(l*Ls + k)] for l<R, k<count,
     *           writes zout[2*(k*OLs + l)], Ls = count, OLs = R.
     * mid  t2 : IN-PLACE, Ls = OLs = count, and it streams twiddles. */
    const size_t Ls = count, OLs = inplace ? count : (size_t)R;
    const size_t nin  = 2 * (size_t)R * count;
    const size_t nout = inplace ? nin : 2 * count * OLs;
    const size_t ntw  = ((count + VFFT_IL_TWPER - 1) / VFFT_IL_TWPER) * (size_t)(R - 1) * VFFT_IL_TWREC + 64u;
    double *zin = (double *)malloc((nin + 16) * sizeof(double));
    double *tw  = (double *)malloc(ntw * sizeof(double));
    double *ob  = (double *)malloc((nout + 2*GUARD) * sizeof(double));
    double *om  = (double *)malloc((nout + 2*GUARD) * sizeof(double));
    size_t i, stale = 0, guard_hit = 0;
    double worst = 0.0, mag = 0.0;

    for (i = 0; i < nin; i++) zin[i] = rnd();
    for (i = 0; i < ntw; i++) { double th = 0.37*(double)i;
                                tw[i] = ((i>>2)&1) ? sin(th) : cos(th); }
    for (i = 0; i < nout + 2*GUARD; i++) ob[i] = om[i] = CANARY;

    if (inplace) {
        /* in-place: the canary cannot survive where the kernel copies input,
         * so seed BOTH buffers with the same input and let the guard bands
         * carry the out-of-region check. */
        memcpy(ob + GUARD, zin, nout * sizeof(double));
        memcpy(om + GUARD, zin, nout * sizeof(double));
        blk (ob + GUARD, 0, ob + GUARD, 0, tw, 0, Ls, 0, OLs, 0, count);
        mono(om + GUARD, 0, om + GUARD, 0, tw, 0, Ls, 0, OLs, 0, count);
    } else {
        blk (zin, 0, ob + GUARD, 0, tw, 0, Ls, 0, OLs, 0, count);
        mono(zin, 0, om + GUARD, 0, tw, 0, Ls, 0, OLs, 0, count);
    }

    /* 1. every output slot written? (OOP only — an in-place kernel is
     *    seeded with its own input, so there is no canary left to survive) */
    if (!inplace)
        for (i = 0; i < nout; i++)
            if (ob[GUARD + i] == CANARY) stale++;
    /* 2. guards intact? */
    for (i = 0; i < GUARD; i++) {
        if (ob[i] != CANARY) guard_hit++;
        if (ob[GUARD + nout + i] != CANARY) guard_hit++;
    }
    /* 3. values agree with the monolithic twin */
    for (i = 0; i < nout; i++) {
        double d = fabs(ob[GUARD + i] - om[GUARD + i]);
        if (d > worst) worst = d;
        if (fabs(om[GUARD + i]) > mag) mag = fabs(om[GUARD + i]);
    }
    {
        double rel = mag > 0 ? worst / mag : worst;
        int ok = (stale == 0) && (guard_hit == 0) && (rel < 1e-12);
        printf("  %-18s count=%-2zu %s unwritten=%-3zu guard_hits=%-3zu rel=%.2e  %s\n",
               what, count, (count & 1) ? "ODD " : "even", stale, guard_hit, rel,
               ok ? "OK" : "*** FAIL ***");
        if (!ok) g_fail = 1;
    }
    free(zin); free(tw); free(ob); free(om);
}

/* one blocked form at every count 1..9; a form absent at the build's ISA
 * (blk == 0) is reported and not gated at avx512, where VFFT_IL_AVX2_ONLY
 * declares it absent, and NOT CORRECT at avx2 */
static int g_absent = 0;
static void arms(const char *what, int R, kfn blk, kfn mono, int inplace)
{
    size_t c;
    if (!blk) {
#if defined(VFFT_BUILD_ISA_AVX512)
        printf("  %-18s declared absent at %s (VFFT_IL_AVX2_ONLY, design D3): not gated\n", what, VFFT_IL_ISA_NAME);
        g_absent++;
#else
        printf("  %-18s absent at %s  *** FAIL ***\n", what, VFFT_IL_ISA_NAME);
        g_fail = 1;
#endif
        return;
    }
    for (c = 1; c <= 9; c++)
        arm(what, R, blk, mono, inplace, c);
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("blocked cil odd-count tail gate — blocked vs monolithic twin\n");
    printf("  canary prefill: an UNWRITTEN column is proof the tail is missing\n");
    printf("  guard bands: a hit is proof it wrote outside its output region\n\n");
    arms("r32 leaf 2.16", 32, VFFT_IL_AVX2_ONLY(radix32_z_n1tb_fwd),
         VFFT_IL_SYM(radix32_z_n1t_fwd), 0);
    printf("\n");
    arms("r64 leaf 8.8", 64, VFFT_IL_SYM(radix64_z_n1tb88_fwd),
         VFFT_IL_SYM(radix64_z_n1t_fwd), 0);
    printf("\n");
    arms("r64 leaf 4.16", 64, VFFT_IL_SYM(radix64_z_n1tb416_fwd),
         VFFT_IL_SYM(radix64_z_n1t_fwd), 0);
    printf("\n");
    arms("r64 mid 8.8", 64, VFFT_IL_SYM(radix64_z_t2b88_fwd),
         VFFT_IL_SYM(radix64_z_t2_fwd), 1);
    printf("\n");
    arms("r64 mid 4.16", 64, VFFT_IL_SYM(radix64_z_t2b416_fwd),
         VFFT_IL_SYM(radix64_z_t2_fwd), 1);
    printf("\n");
    arms("r32 wing leaf", 32, VFFT_IL_AVX2_ONLY(radix32_z_n1tbw32_fwd),
         VFFT_IL_SYM(radix32_z_n1t_fwd), 0);
    printf("\n");
    arms("r32 wing T256", 32, VFFT_IL_AVX2_ONLY(radix32_z_n1tbw32t256_fwd),
         VFFT_IL_SYM(radix32_z_n1t_fwd), 0);
    printf("\n");
    arms("r32 wing mid", 32, VFFT_IL_AVX2_ONLY(radix32_z_t2bw32_fwd),
         VFFT_IL_SYM(radix32_z_t2_fwd), 1);
    printf("\n");
    arms("r32 wing M128", 32, VFFT_IL_AVX2_ONLY(radix32_z_t2bw32m128_fwd),
         VFFT_IL_SYM(radix32_z_t2_fwd), 1);
    if (g_absent)
        printf("\n%d of 9 blocked forms declared absent at %s: not gated\n", g_absent, VFFT_IL_ISA_NAME);
    if (g_absent && !g_fail)
        printf("\nBLOCKED TAIL: correct at every count 1..9 on every blocked form present at %s\n",
               VFFT_IL_ISA_NAME);
    else
        printf("\n%s\n", g_fail ? "*** BLOCKED TAIL: NOT CORRECT ***"
                                : "BLOCKED TAIL: correct at every count 1..9, "
                                  "r32 and r64, leaf and mid");
    return g_fail;
}
