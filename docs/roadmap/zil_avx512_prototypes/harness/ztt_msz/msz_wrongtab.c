/* msz / mszb / mszt at VW=8 (masked zmm tail) vs the shipped AVX2 kernels
 * (sse2 + scalar arms): every count 1..70, Ls > count with sentinels in the
 * gap (a masked store must not write past column count), and the LAST
 * group's data ending exactly at a PROT_NONE guard page (a masked load must
 * not touch the next page). Tables: VW-record splats of the same (c, s). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>
#include <unistd.h>
typedef void (*kfn)(const double *, const double *, double *, double *, const double *, const double *,
                    size_t, size_t, size_t, size_t, size_t);
#define DECL(R, SFX) extern void radix##R##_z_msz_fwd_##SFX(const double *, const double *, double *, double *, const double *, const double *, size_t, size_t, size_t, size_t, size_t); \
    extern void radix##R##_z_msz_bwd_##SFX(const double *, const double *, double *, double *, const double *, const double *, size_t, size_t, size_t, size_t, size_t); \
    extern void radix##R##_z_mszt_bwd_##SFX(const double *, const double *, double *, double *, const double *, const double *, size_t, size_t, size_t, size_t, size_t);
DECL(3, avx2) DECL(5, avx2) DECL(7, avx2) DECL(9, avx2) DECL(15, avx2)
DECL(3, avx512) DECL(5, avx512) DECL(7, avx512) DECL(9, avx512) DECL(15, avx512)
#define ROW(R) { R, { radix##R##_z_msz_fwd_avx2, radix##R##_z_msz_bwd_avx2, radix##R##_z_mszt_bwd_avx2 }, \
                    { radix##R##_z_msz_fwd_avx512, radix##R##_z_msz_bwd_avx512, radix##R##_z_mszt_bwd_avx512 } }
static const struct { int R; kfn a2[3], a5[3]; } K[] = { ROW(3), ROW(5), ROW(7), ROW(9), ROW(15) };
static void build_tw(double *tw, int vw, int R, int Gs, int bwd)
{   /* per group (R-1) records [c x vw][s x vw], splat */
    int g, l, lane;
    for (g = 0; g < Gs; g++) for (l = 1; l < R; l++) {
        const double a = -2.0 * M_PI * (double)(l * (g + 1)) / (double)(R * Gs * 7);
        double *rec = tw + ((size_t)g * (R - 1) + (l - 1)) * 2 * vw;
        for (lane = 0; lane < vw; lane++) { rec[lane] = cos(a); rec[vw + lane] = bwd ? -sin(a) : sin(a); }
    }
}
int main(void)
{
    const long pg = sysconf(_SC_PAGESIZE);
    int ki, kind, fails = 0, tests = 0, bitwise = 0;
    for (ki = 0; ki < 5; ki++) for (kind = 0; kind < 3; kind++) {
        const int R = K[ki].R, Gs = 3;
        size_t count;
        for (count = 1; count <= 70; count++) {
            const size_t Ls = count + 5, span = 2 * (size_t)R * Ls * Gs;   /* doubles */
            /* place the buffer so its LAST double is the last before a guard page */
            const size_t bytes = span * sizeof(double), npg = (bytes + pg - 1) / pg + 1;
            char *m = mmap(0, (npg + 1) * pg, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            double *b5, *b2 = malloc(bytes);
            double tw2[4096], tw5[8192];
            size_t i, bad = 0, nbit = 0;
            mprotect(m + npg * pg, pg, PROT_NONE);
            b5 = (double *)(m + npg * pg - bytes);
            /* shrink the last group's span so its final column ends at the page: the kernel
               touches up to leg R-1, column count-1 of group Gs-1 = the whole span minus the gap */
            b5 = (double *)((char *)b5 + (Ls - count) * 2 * sizeof(double));
            for (i = 0; i < span - (Ls - count) * 2; i++) b5[i] = sin(0.37 * i + R) ;
            for (i = 0; i < span; i++) b2[i] = (i < span - (Ls - count) * 2) ? b5[i] : 0;
            build_tw(tw2, 4, R, Gs, kind != 0); build_tw(tw5, 8, R, Gs, kind != 0);
            /* sentinels in each gap [count, Ls) of every leg/group row */
            {
                size_t g, l, c;
                for (g = 0; g < (size_t)Gs; g++) for (l = 0; l < (size_t)R; l++) for (c = count; c < Ls; c++) {
                    const size_t o = 2 * ((g * R + l) * Ls + c);
                    if (o + 1 < span - (Ls - count) * 2) { b5[o] = b5[o + 1] = b2[o] = b2[o + 1] = 12345.0; }
                }
            }
            K[ki].a2[kind](0, 0, b2, 0, tw2, 0, Ls, Gs, 0, 0, count);
            K[ki].a5[kind](0, 0, b5, 0, tw2, 0, Ls, Gs, 0, 0, count);  /* AVX2-LAYOUT table */
            for (i = 0; i < span - (Ls - count) * 2; i++) {
                if (fabs(b5[i] - b2[i]) > 1e-13 * (1 + fabs(b2[i]))) bad++;
                if (b5[i] != b2[i]) nbit++;
            }
            tests++;
            if (bad) { fails++; printf("FAIL R=%d kind=%d count=%zu bad=%zu\n", R, kind, count, bad); }
            if (!nbit) bitwise++;
            munmap(m, (npg + 1) * pg); free(b2);
        }
    }
    printf("msz/mszb/mszt avx512 vs avx2: %d cases, %d FAIL, %d bitwise-identical (guard page after the last column: no fault)\n", tests, fails, bitwise);
    return fails != 0;
}
