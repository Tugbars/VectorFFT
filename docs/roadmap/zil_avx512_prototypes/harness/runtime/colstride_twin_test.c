/* n1ccs / n1c twin test: avx512 kernel vs its avx2 twin, bitwise, over counts 1..17 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
typedef void (*kfn)(const double *, const double *, double *, double *, const double *, const double *, size_t, size_t, size_t, size_t, size_t);
#define D(n) void n(const double *, const double *, double *, double *, const double *, const double *, size_t, size_t, size_t, size_t, size_t);
#define K(R) D(radix##R##_z_n1ccs_fwd_avx2) D(radix##R##_z_n1ccs_fwd_avx512) D(radix##R##_z_n1c_fwd_avx2) D(radix##R##_z_n1c_fwd_avx512)
K(7) K(13) K(15) K(3)
static double sample(uint64_t *s) { *s = *s * 6364136223846793005ull + 1442695040888963407ull; return (double)(int64_t)(*s >> 11) / 4503599627370496.0; }
static void run(const char *nm, int R, kfn f2, kfn f5, int cs)
{
    int bad = 0;
    for (size_t cnt = 1; cnt <= 17; cnt++) {
        size_t Ls, Gs, OLs, OGs, n;
        if (cs) { Ls = OLs = 1; Gs = OGs = R + 3; n = 2 * (cnt * Gs + R) + 64; }      /* column-stride: lane k = a transform at pitch Gs */
        else    { Ls = cnt + 2; OLs = cnt + 5; Gs = OGs = 0; n = 2 * R * (OLs > Ls ? OLs : Ls) + 64; }
        double *a = malloc(8 * n), *b = malloc(8 * n); uint64_t s = 7;
        for (size_t i = 0; i < n; i++) a[i] = b[i] = sample(&s);
        if (cs) { f2(a, 0, a, 0, 0, 0, Ls, Gs, OLs, OGs, cnt); f5(b, 0, b, 0, 0, 0, Ls, Gs, OLs, OGs, cnt); }
        else { double *o2 = calloc(n, 8), *o5 = calloc(n, 8);
               f2(a, 0, o2, 0, 0, 0, Ls, 0, OLs, 0, cnt); f5(a, 0, o5, 0, 0, 0, Ls, 0, OLs, 0, cnt);
               memcpy(a, o2, 8 * n); memcpy(b, o5, 8 * n); free(o2); free(o5); }
        size_t nd = 0; double w = 0;
        for (size_t i = 0; i < n; i++) if (memcmp(a + i, b + i, 8)) { nd++; if (fabs(a[i] - b[i]) > w) w = fabs(a[i] - b[i]); }
        if (nd) { printf("  %s R=%d count=%zu: %zu doubles differ, max |d| %.2e\n", nm, R, cnt, nd, w); bad++; }
        free(a); free(b);
    }
    printf("%s radix %d: %s\n", nm, R, bad ? "DIFFERS" : "bitwise equal, counts 1..17");
}
int main(void)
{
    run("n1ccs", 13, radix13_z_n1ccs_fwd_avx2, radix13_z_n1ccs_fwd_avx512, 1);
    run("n1ccs", 15, radix15_z_n1ccs_fwd_avx2, radix15_z_n1ccs_fwd_avx512, 1);
    run("n1ccs", 3, radix3_z_n1ccs_fwd_avx2, radix3_z_n1ccs_fwd_avx512, 1);
    run("n1c", 7, radix7_z_n1c_fwd_avx2, radix7_z_n1c_fwd_avx512, 0);
    run("n1c", 13, radix13_z_n1c_fwd_avx2, radix13_z_n1c_fwd_avx512, 0);
    return 0;
}
