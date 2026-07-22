/* k1_ccol_spike.c — composed column pass spike (row_major_engine.md §12.4
 * item 5, greenlit after the 2026-07-23 residency probe).
 *
 * Replaces the K=1 four-step's monolithic leaf with a COMPOSED batch plan:
 * the column pass IS a K=R1 batch of length-R2 transforms in batch layout
 * (the §11 batch identity), so run it through the production stride engine
 * (vfft_proto_execute_fwd_oop, DIT, contiguous per-stage streams) instead of
 * one strided leaf sweep. The batch plan's digit-reversed output order is
 * absorbed for FREE into the transpose (row lookup perm[m]), then the
 * existing flat t1 finishes exactly as the 3p route does. Wins sought:
 *   - leaner multi-stage column kernels (radix 4-16 bodies, §12.2)
 *   - contiguous column-pass streams at N where the strided leaf is
 *     L2-latency-bound (the stage probe's 0.14 -> 1.29 ns/el curve)
 *   - kills the 128-leaf ceiling: N=16384 = 64x256 runs today
 *   - decouples pair space: R1=32 pairs (cheap combines) at any N
 *
 * The permutation is DISCOVERED at setup (all-columns-equal probe vs a naive
 * R2-point DFT, bijection asserted) — self-validating, no digit-reversal
 * convention hardcoded. Whole-route outputs gated vs a naive N-point DFT
 * (tolerance: multi-stage column FP differs from the leaf's rounding, so
 * bit-identity vs 3p is NOT expected).
 *
 * Arms per N: classic 3p / 2pa (create_k1 reference, N<=8192), then per
 * (R1, chain) candidate the composed route with plain and TILED permuted
 * transpose. Canonical methodology: pinned P-core, HIGH prio, best-of-5,
 * cachebust between trials, arm order flipped per trial, one N per process.
 *
 * Build: python build.py --src benches/k1_ccol_spike.c
 * Run:   k1_ccol_spike <N>    (2048 | 4096 | 8192 | 16384)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include <immintrin.h>

#include "executor.h"
#include "planner.h"
#include "oop_plan.h"
#include "oop_execute.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

/* naive N-point DFT via precomputed W table (index (j*k) % N) */
static void naive_dft(int N, const double *xr, const double *xi,
                      double *Xr, double *Xi)
{
    /* plain malloc: scalar-only use, and the harness's posix_memalign must
     * NOT be paired with free() on Windows (_aligned_malloc). */
    double *wr = (double *)malloc((size_t)N * 8), *wi = (double *)malloc((size_t)N * 8);
    for (int n = 0; n < N; n++) {
        double a = -2.0 * M_PI * (double)n / (double)N;
        wr[n] = cos(a); wi[n] = sin(a);
    }
    for (int k = 0; k < N; k++) {
        double sr = 0, si = 0;
        size_t idx = 0;
        for (int j = 0; j < N; j++) {
            sr += xr[j] * wr[idx] - xi[j] * wi[idx];
            si += xr[j] * wi[idx] + xi[j] * wr[idx];
            idx += (size_t)k; if (idx >= (size_t)N) idx -= (size_t)N;
        }
        Xr[k] = sr; Xi[k] = si;
    }
    free(wr); free(wi);
}

/* permuted transpose: d[t*R2 + m] = s[perm[m]*R1 + t]; 4x4 SIMD blocks with
 * per-row bases (perm = identity reproduces _vfft_k1_transpose). */
static void perm_transpose(const double *s, double *d, int R2, int R1,
                           const int *perm)
{
    for (int m = 0; m < R2; m += 4) {
        const double *r0 = s + (size_t)perm[m + 0] * R1;
        const double *r1 = s + (size_t)perm[m + 1] * R1;
        const double *r2 = s + (size_t)perm[m + 2] * R1;
        const double *r3 = s + (size_t)perm[m + 3] * R1;
        for (int t = 0; t < R1; t += 4) {
            __m256d a = _mm256_loadu_pd(r0 + t);
            __m256d b = _mm256_loadu_pd(r1 + t);
            __m256d c = _mm256_loadu_pd(r2 + t);
            __m256d e = _mm256_loadu_pd(r3 + t);
            __m256d u0 = _mm256_unpacklo_pd(a, b), u1 = _mm256_unpackhi_pd(a, b);
            __m256d u2 = _mm256_unpacklo_pd(c, e), u3 = _mm256_unpackhi_pd(c, e);
            _mm256_storeu_pd(d + (size_t)(t + 0) * R2 + m, _mm256_permute2f128_pd(u0, u2, 0x20));
            _mm256_storeu_pd(d + (size_t)(t + 1) * R2 + m, _mm256_permute2f128_pd(u1, u3, 0x20));
            _mm256_storeu_pd(d + (size_t)(t + 2) * R2 + m, _mm256_permute2f128_pd(u0, u2, 0x31));
            _mm256_storeu_pd(d + (size_t)(t + 3) * R2 + m, _mm256_permute2f128_pd(u1, u3, 0x31));
        }
    }
}

/* tiled variant: block t so each pass's store window stays L1-resident
 * (TB columns x 4 rows live at once; TB*R2*8 bytes of store lines per row
 * sweep drops to TB lines revisited across the m loop). */
static void perm_transpose_tiled(const double *s, double *d, int R2, int R1,
                                 const int *perm)
{
    const int TB = 32;
    for (int tb = 0; tb < R1; tb += TB) {
        int te = tb + TB < R1 ? tb + TB : R1;
        for (int m = 0; m < R2; m += 4) {
            const double *r0 = s + (size_t)perm[m + 0] * R1;
            const double *r1 = s + (size_t)perm[m + 1] * R1;
            const double *r2 = s + (size_t)perm[m + 2] * R1;
            const double *r3 = s + (size_t)perm[m + 3] * R1;
            for (int t = tb; t < te; t += 4) {
                __m256d a = _mm256_loadu_pd(r0 + t);
                __m256d b = _mm256_loadu_pd(r1 + t);
                __m256d c = _mm256_loadu_pd(r2 + t);
                __m256d e = _mm256_loadu_pd(r3 + t);
                __m256d u0 = _mm256_unpacklo_pd(a, b), u1 = _mm256_unpackhi_pd(a, b);
                __m256d u2 = _mm256_unpacklo_pd(c, e), u3 = _mm256_unpackhi_pd(c, e);
                _mm256_storeu_pd(d + (size_t)(t + 0) * R2 + m, _mm256_permute2f128_pd(u0, u2, 0x20));
                _mm256_storeu_pd(d + (size_t)(t + 1) * R2 + m, _mm256_permute2f128_pd(u1, u3, 0x20));
                _mm256_storeu_pd(d + (size_t)(t + 2) * R2 + m, _mm256_permute2f128_pd(u0, u2, 0x31));
                _mm256_storeu_pd(d + (size_t)(t + 3) * R2 + m, _mm256_permute2f128_pd(u1, u3, 0x31));
            }
        }
    }
}

/* one composed-route candidate */
typedef struct {
    int R1, R2, nf;
    int chain[4];
    stride_plan_t *colp;
    int perm[512];
    vfft_oop11_fn t1;
    double *Qr, *Qi;
    char name[48];
} cc_t;

static vfft_proto_registry_t g_reg;

/* build the candidate: column plan + discovered perm + K=1 flat-t1 Q tables */
static int cc_setup(cc_t *c, int N)
{
    c->colp = vfft_proto_plan_create(c->R2, (size_t)c->R1, c->chain, NULL,
                                     c->nf, &g_reg);
    if (!c->colp) return -1;
    c->t1 = vfft_oop_t1_fn(c->R1);
    if (!c->t1) return -1;

    /* perm discovery: all columns carry the same random signal; output row p
     * then holds S[m] for the m with perm[m] == p. */
    {
        int R1 = c->R1, R2 = c->R2;
        /* ad() allocations are LEAKED on purpose throughout the spike: the
         * harness memalign has no matching free and each cell is its own
         * process. */
        double *pr = ad((size_t)R1 * R2), *pi = ad((size_t)R1 * R2);
        double *or_ = ad((size_t)R1 * R2), *oi = ad((size_t)R1 * R2);
        double *sr = ad(R2), *si = ad(R2), *Sr = ad(R2), *Si = ad(R2);
        srand(777);
        for (int j = 0; j < R2; j++) {
            sr[j] = (double)rand() / RAND_MAX - 0.5;
            si[j] = (double)rand() / RAND_MAX - 0.5;
        }
        for (int j = 0; j < R2; j++)
            for (int t = 0; t < R1; t++) {
                pr[(size_t)j * R1 + t] = sr[j];
                pi[(size_t)j * R1 + t] = si[j];
            }
        if (vfft_proto_execute_fwd_oop(c->colp, pr, pi, or_, oi,
                                       (size_t)c->R1) != 0) return -1;
        naive_dft(R2, sr, si, Sr, Si);
        double mag = 0;
        for (int m = 0; m < R2; m++)
            if (fabs(Sr[m]) + fabs(Si[m]) > mag) mag = fabs(Sr[m]) + fabs(Si[m]);
        for (int m = 0; m < R2; m++) c->perm[m] = -1;
        for (int m = 0; m < R2; m++) {
            for (int p = 0; p < R2; p++) {
                double d = fabs(or_[(size_t)p * R1] - Sr[m]) +
                           fabs(oi[(size_t)p * R1] - Si[m]);
                if (d < 1e-6 * mag) {
                    if (c->perm[m] != -1) return -2;   /* ambiguous */
                    c->perm[m] = p;
                }
            }
            if (c->perm[m] < 0) return -3;             /* unmatched */
        }
    }

    /* K=1 flat-t1 twiddles, the _vfft_oop_fill_bailey odd-K layout:
     * Q[(l2-1)*R2 + k2] = W_N^{l2*k2} */
    c->Qr = ad((size_t)(c->R1 - 1) * c->R2);
    c->Qi = ad((size_t)(c->R1 - 1) * c->R2);
    for (int l2 = 1; l2 < c->R1; l2++)
        for (int k2 = 0; k2 < c->R2; k2++) {
            double a = -2.0 * M_PI * (double)((long)l2 * k2) / (double)N;
            c->Qr[(size_t)(l2 - 1) * c->R2 + k2] = cos(a);
            c->Qi[(size_t)(l2 - 1) * c->R2 + k2] = sin(a);
        }
    return 0;
}

/* execute: colp src->col (batch, contiguous), permuted transpose col->dst,
 * flat t1 in-place on dst — the 3p shape with the leaf swapped out. */
static void cc_run(const cc_t *c, const double *xr, const double *xi,
                   double *col_re, double *col_im, double *dr, double *di,
                   int tiled)
{
    vfft_proto_execute_fwd_oop(c->colp, xr, xi, col_re, col_im, (size_t)c->R1);
    if (tiled) {
        perm_transpose_tiled(col_re, dr, c->R2, c->R1, c->perm);
        perm_transpose_tiled(col_im, di, c->R2, c->R1, c->perm);
    } else {
        perm_transpose(col_re, dr, c->R2, c->R1, c->perm);
        perm_transpose(col_im, di, c->R2, c->R1, c->perm);
    }
    c->t1(dr, di, dr, di, c->Qr, c->Qi,
          (size_t)c->R2, 1, (size_t)c->R2, 1, (size_t)c->R2);
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    vfft_proto_registry_init(&g_reg);

    int Nd[] = { 2048, 4096, 8192, 16384 };
    int nN = argc > 1 ? argc - 1 : 4;

    for (int ni = 0; ni < nN; ni++) {
        int N = argc > 1 ? atoi(argv[ni + 1]) : Nd[ni];

        /* candidate table: (R1, chain...) — R1=64 keeps today's pairs,
         * R1=32 exploits the cheap radix-32 combine (§12.2). */
        cc_t cs[6]; int nc = 0;
        memset(cs, 0, sizeof cs);
#define CAND(r1, ...) do { \
            int ch_[] = { __VA_ARGS__ }; \
            cs[nc].R1 = r1; cs[nc].nf = (int)(sizeof ch_ / sizeof ch_[0]); \
            cs[nc].R2 = N / r1; \
            memcpy(cs[nc].chain, ch_, sizeof ch_); nc++; \
        } while (0)
        switch (N) {
        case 2048:  CAND(64, 8, 4);  CAND(64, 4, 8);  CAND(32, 8, 8);  CAND(32, 16, 4); break;
        case 4096:  CAND(64, 8, 8);  CAND(64, 16, 4); CAND(32, 16, 8); CAND(32, 8, 16); break;
        case 8192:  CAND(64, 16, 8); CAND(64, 8, 16); CAND(32, 16, 16); break;
        case 16384: CAND(64, 16, 16); CAND(64, 8, 8, 4); break;
        default: fprintf(stderr, "unsupported N=%d\n", N); continue;
        }
        for (int k = 0; k < nc; k++) {
            char *q = cs[k].name;
            q += sprintf(q, "cc%d[", cs[k].R1);
            for (int s = 0; s < cs[k].nf; s++)
                q += sprintf(q, "%s%d", s ? "," : "", cs[k].chain[s]);
            sprintf(q, "]");
        }

        double *xr = ad(N), *xi = ad(N), *dr = ad(N), *di = ad(N);
        double *cre = ad(N), *cim = ad(N);
        double *Rr = ad(N), *Ri = ad(N);
        srand(42 + N);
        for (int n = 0; n < N; n++) {
            xr[n] = (double)rand() / RAND_MAX - 0.5;
            xi[n] = (double)rand() / RAND_MAX - 0.5;
        }
        naive_dft(N, xr, xi, Rr, Ri);
        double rmag = 0;
        for (int n = 0; n < N; n++)
            if (fabs(Rr[n]) + fabs(Ri[n]) > rmag) rmag = fabs(Rr[n]) + fabs(Ri[n]);

        /* setup + correctness gate per candidate */
        int ok[6] = { 0 };
        for (int k = 0; k < nc; k++) {
            int rc = cc_setup(&cs[k], N);
            if (rc != 0) {
                printf("GATE,%d,%s,setup_fail,%d\n", N, cs[k].name, rc);
                continue;
            }
            for (int tiled = 0; tiled < 2; tiled++) {
                cc_run(&cs[k], xr, xi, cre, cim, dr, di, tiled);
                double err = 0;
                for (int n = 0; n < N; n++) {
                    double d = fabs(dr[n] - Rr[n]) + fabs(di[n] - Ri[n]);
                    if (d > err) err = d;
                }
                printf("GATE,%d,%s%s,relerr,%.3e\n", N, cs[k].name,
                       tiled ? "-tiled" : "", err / rmag);
                if (err / rmag > 1e-10) { ok[k] = -1; break; }
            }
            if (!ok[k]) ok[k] = 1;
        }

        /* classic reference arms (create_k1 exists only when leaf(R2) does) */
        vfft_oop_plan_t *pcl = vfft_oop_plan_create_k1(N, 64, N / 64);

        int reps = (int)(2e6 / (double)N);
        if (reps < 100) reps = 100;
        if (reps > 400000) reps = 400000;

        /* arms: 0 = 3p classic, 1 = 2pa classic, then 2 + 2k (+1 tiled) */
        int narms = 2 + 2 * nc;
        double best[16];
        for (int a = 0; a < narms; a++) best[a] = 1e18;
        for (int t = 0; t < 5; t++) {
            if (t) cachebust();
            for (int q = 0; q < narms; q++) {
                int a = (t & 1) ? (narms - 1 - q) : q;
                double t0, ns;
                if (a == 0 || a == 1) {
                    if (!pcl) continue;
                    if (a == 1 && !pcl->t1_ul) continue;
                    for (int w = 0; w < 10; w++)
                        a == 0 ? vfft_oop_execute_fwd(pcl, xr, xi, dr, di)
                               : vfft_oop_execute_fwd_2pa(pcl, xr, xi, dr, di);
                    t0 = now_ms();
                    for (int i = 0; i < reps; i++)
                        a == 0 ? vfft_oop_execute_fwd(pcl, xr, xi, dr, di)
                               : vfft_oop_execute_fwd_2pa(pcl, xr, xi, dr, di);
                    ns = (now_ms() - t0) * 1e6 / reps;
                } else {
                    int k = (a - 2) / 2, tiled = (a - 2) & 1;
                    if (ok[k] != 1) continue;
                    for (int w = 0; w < 10; w++)
                        cc_run(&cs[k], xr, xi, cre, cim, dr, di, tiled);
                    t0 = now_ms();
                    for (int i = 0; i < reps; i++)
                        cc_run(&cs[k], xr, xi, cre, cim, dr, di, tiled);
                    ns = (now_ms() - t0) * 1e6 / reps;
                }
                if (ns < best[a]) best[a] = ns;
            }
        }

        if (pcl) {
            printf("TIME,%d,3p-64,%.1f\n", N, best[0]);
            if (pcl->t1_ul) printf("TIME,%d,2pa-64,%.1f\n", N, best[1]);
        }
        double bcc = 1e18; const char *bname = "-"; int btiled = 0;
        for (int k = 0; k < nc; k++) {
            if (ok[k] != 1) continue;
            printf("TIME,%d,%s,%.1f\n", N, cs[k].name, best[2 + 2 * k]);
            printf("TIME,%d,%s-tiled,%.1f\n", N, cs[k].name, best[3 + 2 * k]);
            for (int v = 0; v < 2; v++)
                if (best[2 + 2 * k + v] < bcc) {
                    bcc = best[2 + 2 * k + v];
                    bname = cs[k].name; btiled = v;
                }
        }
        printf("CCVERDICT,%d,best-cc,%s%s,%.1f,classic-best,%.1f\n",
               N, bname, btiled ? "-tiled" : "",
               bcc, pcl ? (best[1] < best[0] ? best[1] : best[0]) : 0.0);

        if (pcl) vfft_oop_plan_destroy(pcl);
        /* leak candidate plans/tables: one cell per process by design */
    }
    printf("DONE\n");
    return 0;
}
