/* zil_chain_dp.c — the chain PLANNER for the z staged cascade: exhaustively
 * enumerate ALL factor chains of N over the available z kernels, gate each vs
 * naive, race all + MKL-IL, rank. "We can't just arbitrarily choose the stages"
 * — at these Ns the chain space is tiny (18 @4096, 34 @16384), so measured
 * exhaustive search IS the planner (cost models hit the OOO/cache ceiling;
 * plan-level measurement doesn't). Winner per cell -> wisdom (like cc_chain).
 *
 * Executor = generalized arm-B (grid-preserving DIT, gated in zil_cascade.c):
 *   S0   n1 leaf   z0 -> A          (Ls=OLs=D0, count=D0)
 *   mids t2 TRUE-IN-PLACE on A      (Ls=OLs=Ds, count=Ds, group base digit-decomposed,
 *                                    col-const VTW2 W_{N/Ds}^{l*brev(g)})
 *   term t2s gather A -> out        (Ls=1, Gs=Rt, OLs=N/Rt, per-col VTW2 W_N^{l*brev(k)})
 * Output digit-reversed: out[l*(N/Rt)+g] = X[drev(g*Rt+l)] (mixed-radix drev).
 * nf=2 chains ARE the flat two-pass (64.64 = the old champion) — subsumed.
 *
 * Kernel coverage: S0 n1 {4,8,16b,32b,64b2}; mid t2 {8,16,32,64}; term t2s
 * {8,16,32,64}. radix-4 t2/t2s = pending emitter run (MKL's census favors
 * radix-4 deep stages — follow-up once emitted).
 *
 * Build: python build.py --src benches/zil_chain_dp.c --mkl
 * Run:   zil_chain_dp.exe [N]      (N in {2048,4096,8192,16384}, default 4096)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include <windows.h>
#include <mkl_dfti.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef void (*zfn)(const double *, const double *, double *, double *,
                    const double *, const double *,
                    unsigned long long, unsigned long long,
                    unsigned long long, unsigned long long, unsigned long long);
#define D(fn) extern void fn(const double *, const double *, double *, double *, \
    const double *, const double *, unsigned long long, unsigned long long, \
    unsigned long long, unsigned long long, unsigned long long);
D(radix4_z_n1_fwd_avx2)  D(radix8_z_n1_fwd_avx2)  D(radix16_z_n1b_fwd_avx2)
D(radix32_z_n1b_fwd_avx2) D(radix64_z_n1b2_fwd_avx2)
D(radix8_z_t2_fwd_avx2)  D(radix16_z_t2_fwd_avx2) D(radix32_z_t2_fwd_avx2)
D(radix64_z_t2_fwd_avx2)
D(radix8_z_t2s_fwd_avx2) D(radix16_z_t2s_fwd_avx2) D(radix32_z_t2s_fwd_avx2)
D(radix64_z_t2s_fwd_avx2)
D(radix8_z_t2c_fwd_avx2) D(radix16_z_t2c_fwd_avx2) D(radix32_z_t2c_fwd_avx2)
D(radix64_z_t2c_fwd_avx2)
D(radix8_z_t2sp_fwd_avx2) D(radix16_z_t2sp_fwd_avx2)

static zfn n1_of(int R)  { switch (R) { case 4: return radix4_z_n1_fwd_avx2;
    case 8: return radix8_z_n1_fwd_avx2;   case 16: return radix16_z_n1b_fwd_avx2;
    case 32: return radix32_z_n1b_fwd_avx2; case 64: return radix64_z_n1b2_fwd_avx2; }
    return 0; }
static zfn t2_of(int R)  { switch (R) { case 8: return radix8_z_t2_fwd_avx2;
    case 16: return radix16_z_t2_fwd_avx2; case 32: return radix32_z_t2_fwd_avx2;
    case 64: return radix64_z_t2_fwd_avx2; } return 0; }
static zfn t2s_of(int R) { switch (R) { case 8: return radix8_z_t2s_fwd_avx2;
    case 16: return radix16_z_t2s_fwd_avx2; case 32: return radix32_z_t2s_fwd_avx2;
    case 64: return radix64_z_t2s_fwd_avx2; } return 0; }
static zfn t2c_of(int R) { switch (R) { case 8: return radix8_z_t2c_fwd_avx2;
    case 16: return radix16_z_t2c_fwd_avx2; case 32: return radix32_z_t2c_fwd_avx2;
    case 64: return radix64_z_t2c_fwd_avx2; } return 0; }
static zfn t2sp_of(int R) { switch (R) { case 8: return radix8_z_t2sp_fwd_avx2;
    case 16: return radix16_z_t2sp_fwd_avx2; } return 0; }

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

/* ── chain machinery ──────────────────────────────────────────────────── */
#define MAXNF 8
#define MAXC  64
/* executor variants: bit0 = t2c mids (group-constant tw), bit1 = t2sp term (w^1 powers) */
#define NV 4
typedef struct {
    int nf, R[MAXNF];
    long D[MAXNF], G[MAXNF];       /* D=stride (prod after), G=groups (prod before) */
    double *twm[MAXNF];            /* middle-stage VTW2, replicated (t2 arm) */
    double *twc[MAXNF];            /* middle-stage VTW2, one set/group (t2c arm) */
    double *twt;                   /* terminator VTW2, FLAT per-column (t2s arm) */
    double *twp1;                  /* terminator w^1-only stream (t2sp arm) */
    size_t twb[NV];                /* streamed table bytes per variant */
    char name[40];
    double best[NV];               /* <0 = gate fail / unavailable */
} chain_t;
static chain_t C[MAXC];
static int NC = 0;
static int N;

/* mixed-radix digit reversal of grid index x over the full chain */
static long drev_full(long x, const int *R, int nf)
{
    long r = 0;
    for (int i = nf - 1; i >= 0; i--) { r = r * R[i] + (x % R[i]); x /= R[i]; }
    return r;
}
/* reversed value of the s prefix digits of group/block index g:
 * brev = sum f_i * (prod_{k<i} R_k), f_0 = MSB digit of g */
static long brev_prefix(long g, int s, const int *R)
{
    long f[MAXNF];
    for (int i = s - 1; i >= 0; i--) { f[i] = g % R[i]; g /= R[i]; }
    long P = 1, r = 0;
    for (int i = 0; i < s; i++) { r += f[i] * P; P *= R[i]; }
    return r;
}
/* complex base offset of stage-s group g: sum f_i * D_i */
static long base_of(long g, int s, const int *R, const long *D)
{
    long b = 0;
    for (int i = s - 1; i >= 0; i--) { long f = g % R[i]; g /= R[i]; b += f * D[i]; }
    return b;
}

/* one VTW2 record (8 doubles): cos-first, sign-folded, cols k0/k1 */
static void vtw2_rec(double *rec, double a0, double a1)
{
    rec[0] = cos(a0); rec[1] = cos(a0); rec[2] = cos(a1); rec[3] = cos(a1);
    rec[4] = -sin(a0); rec[5] = sin(a0); rec[6] = -sin(a1); rec[7] = sin(a1);
}

static void build_chain(chain_t *c)
{
    const int nf = c->nf;
    c->D[nf - 1] = 1;
    for (int i = nf - 2; i >= 0; i--) c->D[i] = c->D[i + 1] * c->R[i + 1];
    c->G[0] = 1;
    for (int i = 1; i < nf; i++) c->G[i] = c->G[i - 1] * c->R[i - 1];
    c->twbytes = 0;
    const double TAU = 2.0 * M_PI;

    /* middles: col-constant per group, record repeated across the group's pairs */
    for (int s = 1; s <= nf - 2; s++) {
        long pairs = c->D[s] / 2, M = (long)N / c->D[s];
        size_t rl = (size_t)(c->R[s] - 1) * 8;                 /* rec doubles/pair */
        size_t sz = (size_t)c->G[s] * pairs * rl * 8;
        c->twm[s] = (double *)_mm_malloc(sz, 64);
        c->twbytes += sz;
        for (long g = 0; g < c->G[s]; g++) {
            long brev = brev_prefix(g, s, c->R);
            double *gp = c->twm[s] + (size_t)g * pairs * rl;
            for (int l = 1; l < c->R[s]; l++) {
                double a = -TAU * (double)(((long)l * brev) % M) / (double)M;
                vtw2_rec(gp + (size_t)(l - 1) * 8, a, a);
            }
            for (long p = 1; p < pairs; p++)                   /* replicate */
                memcpy(gp + (size_t)p * rl, gp, rl * 8);
        }
    }
    /* terminator: genuine per-column records, W_N^{l*brev(k)} */
    {
        int Rt = c->R[nf - 1];
        long cols = (long)N / Rt, pairs = cols / 2;
        size_t rl = (size_t)(Rt - 1) * 8;
        size_t sz = (size_t)pairs * rl * 8;
        c->twt = (double *)_mm_malloc(sz, 64);
        c->twbytes += sz;
        for (long p = 0; p < pairs; p++) {
            long b0 = brev_prefix(2 * p, nf - 1, c->R);
            long b1 = brev_prefix(2 * p + 1, nf - 1, c->R);
            for (int l = 1; l < Rt; l++) {
                double a0 = -TAU * (double)(((long)l * b0) % N) / (double)N;
                double a1 = -TAU * (double)(((long)l * b1) % N) / (double)N;
                vtw2_rec(c->twt + (size_t)p * rl + (size_t)(l - 1) * 8, a0, a1);
            }
        }
    }
    char *w = c->name;
    for (int i = 0; i < nf; i++) w += sprintf(w, "%s%d", i ? "." : "", c->R[i]);
}

static void run_chain(const chain_t *c, const double *z0, double *A, double *out)
{
    const int nf = c->nf;
    n1_of(c->R[0])(z0, 0, A, 0, 0, 0,
                   (unsigned long long)c->D[0], 0, (unsigned long long)c->D[0], 0,
                   (unsigned long long)c->D[0]);
    for (int s = 1; s <= nf - 2; s++) {
        zfn f = t2_of(c->R[s]);
        long pairs = c->D[s] / 2;
        size_t rl = (size_t)(c->R[s] - 1) * 8;
        for (long g = 0; g < c->G[s]; g++) {
            long b = base_of(g, s, c->R, c->D);
            f(A + 2 * b, 0, A + 2 * b, 0, c->twm[s] + (size_t)g * pairs * rl, 0,
              (unsigned long long)c->D[s], 0, (unsigned long long)c->D[s], 0,
              (unsigned long long)c->D[s]);
        }
    }
    {
        int Rt = c->R[nf - 1];
        t2s_of(Rt)(A, 0, out, 0, c->twt, 0,
                   1, (unsigned long long)Rt, (unsigned long long)(N / Rt), 1,
                   (unsigned long long)(N / Rt));
    }
}

/* enumerate chains: first log2-part in {2..6} (n1 4..64), rest in {3..6} */
static void enumerate(int rem, int pos, int *parts)
{
    if (rem == 0) {
        if (pos < 2 || NC >= MAXC) return;
        chain_t *c = &C[NC];
        memset(c, 0, sizeof(*c));
        c->nf = pos;
        for (int i = 0; i < pos; i++) c->R[i] = 1 << parts[i];
        NC++;
        return;
    }
    int lo = pos ? 3 : 2, hi = 6;
    for (int p = lo; p <= hi && p <= rem; p++) {
        parts[pos] = p;
        enumerate(rem - p, pos + 1, parts);
    }
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    N = argc > 1 ? atoi(argv[1]) : 4096;
    int L = 0; while ((1 << L) < N) L++;
    if ((1 << L) != N) { printf("N must be 2^k\n"); return 1; }

    int parts[MAXNF];
    enumerate(L, 0, parts);
    printf("# N=%d: %d candidate chains\n", N, NC);

    double *z0 = (double *)_mm_malloc((size_t)2 * N * 8, 64);
    double *A  = (double *)_mm_malloc((size_t)2 * N * 8, 64);
    double *z  = (double *)_mm_malloc((size_t)2 * N * 8, 64);
    double *Rr = (double *)_mm_malloc((size_t)N * 8, 64);
    double *Ri = (double *)_mm_malloc((size_t)N * 8, 64);
    srand(N);
    for (int i = 0; i < 2 * N; i++) z0[i] = (double)rand() / RAND_MAX - 0.5;

    /* naive reference */
    for (int m = 0; m < N; m++) {
        double sr = 0, si = 0;
        for (int n = 0; n < N; n++) {
            double a = -2.0 * M_PI * (double)(((long)n * m) % N) / (double)N;
            double cc = cos(a), ss = sin(a);
            sr += z0[2 * n] * cc - z0[2 * n + 1] * ss;
            si += z0[2 * n] * ss + z0[2 * n + 1] * cc;
        }
        Rr[m] = sr; Ri[m] = si;
    }
    double mag = 0;
    for (int m = 0; m < N; m++) {
        double g = fabs(Rr[m]) + fabs(Ri[m]);
        if (g > mag) mag = g;
    }

    /* build + gate every chain (digit-reversed compare) */
    int ok = 0;
    for (int ci = 0; ci < NC; ci++) {
        chain_t *c = &C[ci];
        build_chain(c);
        run_chain(c, z0, A, z);
        int Rt = c->R[c->nf - 1];
        long NR = (long)N / Rt;
        double err = 0;
        for (long idx = 0; idx < N; idx++) {
            long l = idx / NR, g = idx % NR;
            long m = drev_full(g * Rt + l, c->R, c->nf);
            double d = fabs(z[2 * idx] - Rr[m]) + fabs(z[2 * idx + 1] - Ri[m]);
            if (d > err) err = d;
        }
        double rel = err / mag;
        c->best = 1e18;
        if (rel < 1e-12) ok++;
        else { printf("GATE FAIL %-14s relerr=%.3e\n", c->name, rel); c->best = -1; }
    }
    printf("# gates: %d/%d PASS\n", ok, NC);

    /* race: all gated chains + MKL, best-of-7, cachebust, rotated order */
    DFTI_DESCRIPTOR_HANDLE h = NULL;
    DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N);
    DftiCommitDescriptor(h);
    int reps = (int)(4.0e6 / N); if (reps < 200) reps = 200;
    double mkl_best = 1e18;
    int narm = NC + 1;
    for (int t = 0; t < 7; t++) {
        if (t) cachebust();
        for (int q = 0; q < narm; q++) {
            int a = (t & 1) ? (narm - 1 - q) : q;
            if (a < NC) {
                chain_t *c = &C[a];
                if (c->best < 0) continue;
                for (int w = 0; w < 6; w++) run_chain(c, z0, A, z);
                double t0 = now_ms();
                for (int i = 0; i < reps; i++) run_chain(c, z0, A, z);
                double ns = (now_ms() - t0) * 1e6 / reps;
                if (ns < c->best) c->best = ns;
            } else {
                for (int w = 0; w < 6; w++) DftiComputeForward(h, z);
                double t0 = now_ms();
                for (int i = 0; i < reps; i++) DftiComputeForward(h, z);
                double ns = (now_ms() - t0) * 1e6 / reps;
                if (ns < mkl_best) mkl_best = ns;
            }
        }
    }

    /* ranked report */
    for (int i = 0; i < NC; i++)                 /* selection sort by best */
        for (int j = i + 1; j < NC; j++)
            if (C[j].best >= 0 && (C[i].best < 0 || C[j].best < C[i].best)) {
                chain_t tmp = C[i]; C[i] = C[j]; C[j] = tmp;
            }
    printf("\n# N=%d chain race (scrambled-out cascade, in-place mids) vs MKL-IL %.1f ns\n",
           N, mkl_best);
    printf("%-4s %-14s %-6s %9s %7s %7s\n", "rank", "chain", "twKB", "ns", "vsMKL", "note");
    for (int i = 0; i < NC; i++) {
        if (C[i].best < 0) continue;
        printf("%-4d %-14s %-6.0f %9.1f %7.2f %s\n", i + 1, C[i].name,
               (double)C[i].twbytes / 1024.0, C[i].best, mkl_best / C[i].best,
               C[i].nf == 2 ? "flat two-pass" : "");
    }
    printf("DONE\n");
    return 0;
}
