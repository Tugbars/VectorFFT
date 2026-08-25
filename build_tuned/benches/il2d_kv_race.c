/* il2d_kv_race.c — kernel-variant race for the IL-2D column kinds
 * (design doc §2a): monolithic vs BLOCKED splits (t2c/n1c r32/r64) and
 * plain vs TANGENT interiors (r8/r16). Same buffers, per-variant GATE vs
 * the monolithic twin (tolerance — different op order), then same-run
 * rotated timing at two regimes (cache-resident and streaming N2).
 * Verdicts are radix-determined per the 1D precedent — static adoption.
 * Build: python build.py --src benches/il2d_kv_race.c --vfft --compile */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

typedef void (*zfn)(const double *, const double *, double *, double *,
                    const double *, const double *,
                    size_t, size_t, size_t, size_t, size_t);
#define D(sym) extern void sym(const double *, const double *, double *, \
    double *, const double *, const double *, size_t, size_t, size_t,    \
    size_t, size_t);
D(radix32_z_t2c_fwd_avx2)  D(radix32_z_t2cb48_fwd_avx2)  D(radix32_z_t2cb84_fwd_avx2)
D(radix64_z_t2c_fwd_avx2)  D(radix64_z_t2cb88_fwd_avx2)  D(radix64_z_t2cb416_fwd_avx2)
D(radix32_z_n1c_fwd_avx2)  D(radix32_z_n1cb48_fwd_avx2)  D(radix32_z_n1cb84_fwd_avx2)
D(radix64_z_n1c_fwd_avx2)  D(radix64_z_n1cb88_fwd_avx2)  D(radix64_z_n1cb416_fwd_avx2)
D(radix8_z_t2c_fwd_avx2)   D(radix8_z_t2ctan_fwd_avx2)
D(radix16_z_t2c_fwd_avx2)  D(radix16_z_t2ctan_fwd_avx2)
D(radix16_z_n1c_fwd_avx2)  D(radix16_z_n1ctan_fwd_avx2)

static double now_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}
static double *mk_table(int R, int D_, int L)
{
    const double pi = 3.14159265358979323846;
    double *f = malloc((size_t)D_ * (R - 1) * 8 * sizeof(double));
    int d, r, ln;
    for (d = 0; d < D_; d++)
        for (r = 1; r < R; r++) {
            double a = -2.0 * pi * (double)(d * r) / (double)L;
            double c = cos(a), si = sin(a);
            double *rec = f + ((size_t)d * (R - 1) + (r - 1)) * 8;
            for (ln = 0; ln < 4; ln++) {
                rec[ln] = c;
                rec[4 + ln] = (ln & 1) ? si : -si;
            }
        }
    return f;
}
static double med_of(double *v, int n)
{
    int i, j; double t;
    for (i = 0; i < n; i++)
        for (j = i + 1; j < n; j++)
            if (v[j] < v[i]) { t = v[i]; v[i] = v[j]; v[j] = t; }
    return v[n / 2];
}

typedef struct { const char *tag; zfn fn; } var_t;
typedef struct {
    const char *name;
    int R, D_;              /* D_ = 1 -> n1c leaf geometry (no table) */
    int nvar;
    var_t v[3];
} case_t;
static const case_t CASES[] = {
    { "t2c r32 (stage0 of 32.32)", 32, 32, 3,
      { { "mono", radix32_z_t2c_fwd_avx2 },
        { "b48", radix32_z_t2cb48_fwd_avx2 },
        { "b84", radix32_z_t2cb84_fwd_avx2 } } },
    { "t2c r64 (stage0 of 64.64)", 64, 64, 3,
      { { "mono", radix64_z_t2c_fwd_avx2 },
        { "b88", radix64_z_t2cb88_fwd_avx2 },
        { "b416", radix64_z_t2cb416_fwd_avx2 } } },
    { "n1c r32 (leaf)", 32, 1, 3,
      { { "mono", radix32_z_n1c_fwd_avx2 },
        { "b48", radix32_z_n1cb48_fwd_avx2 },
        { "b84", radix32_z_n1cb84_fwd_avx2 } } },
    { "n1c r64 (leaf)", 64, 1, 3,
      { { "mono", radix64_z_n1c_fwd_avx2 },
        { "b88", radix64_z_n1cb88_fwd_avx2 },
        { "b416", radix64_z_n1cb416_fwd_avx2 } } },
    { "t2c r8 (tan)", 8, 16, 2,
      { { "plain", radix8_z_t2c_fwd_avx2 },
        { "tan", radix8_z_t2ctan_fwd_avx2 } } },
    { "t2c r16 (tan, stage0 of 16.16)", 16, 16, 2,
      { { "plain", radix16_z_t2c_fwd_avx2 },
        { "tan", radix16_z_t2ctan_fwd_avx2 } } },
    { "n1c r16 (tan, leaf)", 16, 1, 2,
      { { "plain", radix16_z_n1c_fwd_avx2 },
        { "tan", radix16_z_n1ctan_fwd_avx2 } } },
};

int main(void)
{
    static const int N2S[] = { 64, 1024 }; /* resident vs streaming */
    int ci, ni;
    const int ROUNDS = 9;
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== il2d kernel-variant race (gate vs mono, then rotated timing) ===\n");
    for (ci = 0; ci < (int)(sizeof CASES / sizeof CASES[0]); ci++) {
        const case_t *c = &CASES[ci];
        const int L = c->R * c->D_;
        double *tab = c->D_ > 1 ? mk_table(c->R, c->D_, L) : NULL;
        for (ni = 0; ni < 2; ni++) {
            const size_t rn = (size_t)N2S[ni];
            const size_t T = (size_t)L * rn;
            double *x = malloc(2 * T * 8), *ref = malloc(2 * T * 8);
            double *z = malloc(2 * T * 8);
            double smp[3][16], medv[3];
            int a, r, k, ok = 1;
            const int reps = (int)(4e6 / T) < 3 ? 3 : (int)(4e6 / T);
            size_t i;
            srand(5 + ci);
            for (i = 0; i < 2 * T; i++)
                x[i] = (double)rand() / RAND_MAX - 0.5;
            /* gate every variant vs the monolithic twin */
            memcpy(ref, x, 2 * T * 8);
            c->v[0].fn(ref, NULL, ref, NULL, tab, NULL,
                       (size_t)c->D_ * rn, rn, (size_t)c->D_ * rn,
                       (size_t)c->D_, rn);
            for (a = 1; a < c->nvar; a++) {
                double mx = 0, mr = 0;
                memcpy(z, x, 2 * T * 8);
                c->v[a].fn(z, NULL, z, NULL, tab, NULL,
                           (size_t)c->D_ * rn, rn, (size_t)c->D_ * rn,
                           (size_t)c->D_, rn);
                for (i = 0; i < 2 * T; i++) {
                    double d = fabs(z[i] - ref[i]);
                    double m = fabs(ref[i]);
                    if (d > mx) mx = d;
                    if (m > mr) mr = m;
                }
                if (mx / (mr > 0 ? mr : 1) > 1e-13) {
                    printf("  %s N2=%zu %s: GATE FAIL rel %.1e\n",
                           c->name, rn, c->v[a].tag, mx / mr);
                    ok = 0;
                }
            }
            if (!ok) { free(x); free(ref); free(z); continue; }
            memcpy(z, x, 2 * T * 8);
            for (r = 0; r < ROUNDS; r++)
                for (a = 0; a < c->nvar; a++) {
                    const int ai = (r & 1) ? c->nvar - 1 - a : a;
                    double t0 = now_ns();
                    for (k = 0; k < reps; k++)
                        c->v[ai].fn(z, NULL, z, NULL, tab, NULL,
                                    (size_t)c->D_ * rn, rn,
                                    (size_t)c->D_ * rn, (size_t)c->D_,
                                    rn);
                    smp[ai][r] = (now_ns() - t0) / reps;
                }
            printf("  %-32s N2=%-5zu (gate OK, reps %d):", c->name, rn,
                   reps);
            for (a = 0; a < c->nvar; a++) {
                medv[a] = med_of(smp[a], ROUNDS);
                printf("  %s %.0f%s", c->v[a].tag, medv[a],
                       a ? "" : "ns");
            }
            printf("  |");
            for (a = 1; a < c->nvar; a++)
                printf(" %s x%.3f", c->v[a].tag, medv[0] / medv[a]);
            printf("\n");
            free(x); free(ref); free(z);
        }
        free(tab);
    }
    return 0;
}
