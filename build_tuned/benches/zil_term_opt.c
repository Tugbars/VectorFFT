/* zil_term_opt.c — TERMINATOR-WEIGHT attack (z_cascade_plan §4.997 follow-up):
 * the two candidate fixes raced on the four wisdom-winner chains, paced.
 *
 *   champ : emitted ms/msz mids + incumbent term (t2sp/t2spt per wisdom)
 *   tree  : same, term = t2sq/t2sqt (squaring-tree powers: critical path
 *           6 sequential VTW2-cmul links -> 3; emitted kinds)
 *   sterm : ALL-ms mids (no msz — the re-interleave chore moves INTO the
 *           terminator) + hand t2sps: SPLIT-INPUT terminator, 4 cols/iter,
 *           4x4 register transposes on load, SHUFFLE-FREE split butterfly +
 *           twiddles, PACKED per-column w^1 table (16 B/col — halves the
 *           stream again), tree powers, re-interleave fused in the stores.
 *
 * Build: python build.py --src benches/zil_term_opt.c --mkl
 * Run:   zil_term_opt.exe [N]    (2048|4096|8192|16384)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include <immintrin.h>
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
D(radix4_z_s0s_fwd_avx2)
D(radix4_z_ms_fwd_avx2)  D(radix8_z_ms_fwd_avx2)
D(radix4_z_msz_fwd_avx2) D(radix8_z_msz_fwd_avx2)
D(radix8_z_t2sp_fwd_avx2) D(radix8_z_t2spt_fwd_avx2)
D(radix8_z_t2sq_fwd_avx2) D(radix8_z_t2sqt_fwd_avx2)
D(radix8_z_sterm_fwd_avx2)   /* EMITTED promotion of the hand t2sps below */

/* ── hand t2sps: split-input z-output terminator, radix 8, 4 cols/iter ── */
#define TR4(a0,a1,a2,a3, t0,t1,t2,t3) do {                            \
    __m256d _u0 = _mm256_unpacklo_pd(a0, a1);                         \
    __m256d _u1 = _mm256_unpackhi_pd(a0, a1);                         \
    __m256d _u2 = _mm256_unpacklo_pd(a2, a3);                         \
    __m256d _u3 = _mm256_unpackhi_pd(a2, a3);                         \
    t0 = _mm256_permute2f128_pd(_u0, _u2, 0x20);                      \
    t1 = _mm256_permute2f128_pd(_u1, _u3, 0x20);                      \
    t2 = _mm256_permute2f128_pd(_u0, _u2, 0x31);                      \
    t3 = _mm256_permute2f128_pd(_u1, _u3, 0x31);                      \
} while (0)
#define REINT(re, im, zlo, zhi) do {                                  \
    __m256d _p = _mm256_permute4x64_pd(re, 0xD8);                     \
    __m256d _q = _mm256_permute4x64_pd(im, 0xD8);                     \
    zlo = _mm256_unpacklo_pd(_p, _q);                                 \
    zhi = _mm256_unpackhi_pd(_p, _q);                                 \
} while (0)
#define SPLIT_CMUL(ar,ai, ct,st, or_,oi_) do {                        \
    or_ = _mm256_fnmadd_pd(st, ai, _mm256_mul_pd(ct, ar));            \
    oi_ = _mm256_fmadd_pd(st, ar, _mm256_mul_pd(ct, ai));             \
} while (0)
#define WPROD(cA,sA, cB,sB, cP,sP) do {  /* plain (c,s) product */    \
    cP = _mm256_fnmadd_pd(sA, sB, _mm256_mul_pd(cA, cB));             \
    sP = _mm256_fmadd_pd(cA, sB, _mm256_mul_pd(sA, cB));              \
} while (0)
#define SPLIT_BFLY8(x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i, \
                    o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i) do { \
    const __m256d _C = _mm256_set1_pd(0.70710678118654752440);        \
    __m256d t0r=_mm256_add_pd(x0r,x4r), t0i=_mm256_add_pd(x0i,x4i);   \
    __m256d t1r=_mm256_sub_pd(x0r,x4r), t1i=_mm256_sub_pd(x0i,x4i);   \
    __m256d t2r=_mm256_add_pd(x2r,x6r), t2i=_mm256_add_pd(x2i,x6i);   \
    __m256d t3r=_mm256_sub_pd(x2r,x6r), t3i=_mm256_sub_pd(x2i,x6i);   \
    __m256d E0r=_mm256_add_pd(t0r,t2r), E0i=_mm256_add_pd(t0i,t2i);   \
    __m256d E2r=_mm256_sub_pd(t0r,t2r), E2i=_mm256_sub_pd(t0i,t2i);   \
    __m256d E1r=_mm256_add_pd(t1r,t3i), E1i=_mm256_sub_pd(t1i,t3r);   \
    __m256d E3r=_mm256_sub_pd(t1r,t3i), E3i=_mm256_add_pd(t1i,t3r);   \
    __m256d s0r=_mm256_add_pd(x1r,x5r), s0i=_mm256_add_pd(x1i,x5i);   \
    __m256d s1r=_mm256_sub_pd(x1r,x5r), s1i=_mm256_sub_pd(x1i,x5i);   \
    __m256d s2r=_mm256_add_pd(x3r,x7r), s2i=_mm256_add_pd(x3i,x7i);   \
    __m256d s3r=_mm256_sub_pd(x3r,x7r), s3i=_mm256_sub_pd(x3i,x7i);   \
    __m256d O0r=_mm256_add_pd(s0r,s2r), O0i=_mm256_add_pd(s0i,s2i);   \
    __m256d O2r=_mm256_sub_pd(s0r,s2r), O2i=_mm256_sub_pd(s0i,s2i);   \
    __m256d O1r=_mm256_add_pd(s1r,s3i), O1i=_mm256_sub_pd(s1i,s3r);   \
    __m256d O3r=_mm256_sub_pd(s1r,s3i), O3i=_mm256_add_pd(s1i,s3r);   \
    __m256d X1r=_mm256_add_pd(O1r,O1i), X1i=_mm256_sub_pd(O1i,O1r);   \
    __m256d X3r=_mm256_sub_pd(O3i,O3r), X3n=_mm256_add_pd(O3r,O3i);   \
    o0r=_mm256_add_pd(E0r,O0r); o0i=_mm256_add_pd(E0i,O0i);           \
    o4r=_mm256_sub_pd(E0r,O0r); o4i=_mm256_sub_pd(E0i,O0i);           \
    o1r=_mm256_fmadd_pd(_C,X1r,E1r); o1i=_mm256_fmadd_pd(_C,X1i,E1i); \
    o5r=_mm256_fnmadd_pd(_C,X1r,E1r); o5i=_mm256_fnmadd_pd(_C,X1i,E1i); \
    o2r=_mm256_add_pd(E2r,O2i); o2i=_mm256_sub_pd(E2i,O2r);           \
    o6r=_mm256_sub_pd(E2r,O2i); o6i=_mm256_add_pd(E2i,O2r);           \
    o3r=_mm256_fmadd_pd(_C,X3r,E3r); o3i=_mm256_fnmadd_pd(_C,X3n,E3i); \
    o7r=_mm256_fnmadd_pd(_C,X3r,E3r); o7i=_mm256_fmadd_pd(_C,X3n,E3i); \
} while (0)

/* sp = block-split plane (col k = 64B blocks 2k, 2k+1); twq = packed per-col
 * w^1 [c x4][s x4] per 4 cols; out z, digit-reversed comb (OLs = N/8). */
__attribute__((target("avx2,fma")))
static void t2sps_r8(const double *sp, double *out, const double *twq,
                     long OLs, long count)
{
    for (long k = 0; k + 4 <= count; k += 4) {
        __m256d xr[8], xi[8];
        {   /* loads + 4x4 transposes: leg-vectors over 4 columns */
            __m256d rl0 = _mm256_loadu_pd(sp + 16 * (size_t)k);
            __m256d il0 = _mm256_loadu_pd(sp + 16 * (size_t)k + 4);
            __m256d rh0 = _mm256_loadu_pd(sp + 16 * (size_t)k + 8);
            __m256d ih0 = _mm256_loadu_pd(sp + 16 * (size_t)k + 12);
            __m256d rl1 = _mm256_loadu_pd(sp + 16 * (size_t)(k + 1));
            __m256d il1 = _mm256_loadu_pd(sp + 16 * (size_t)(k + 1) + 4);
            __m256d rh1 = _mm256_loadu_pd(sp + 16 * (size_t)(k + 1) + 8);
            __m256d ih1 = _mm256_loadu_pd(sp + 16 * (size_t)(k + 1) + 12);
            __m256d rl2 = _mm256_loadu_pd(sp + 16 * (size_t)(k + 2));
            __m256d il2 = _mm256_loadu_pd(sp + 16 * (size_t)(k + 2) + 4);
            __m256d rh2 = _mm256_loadu_pd(sp + 16 * (size_t)(k + 2) + 8);
            __m256d ih2 = _mm256_loadu_pd(sp + 16 * (size_t)(k + 2) + 12);
            __m256d rl3 = _mm256_loadu_pd(sp + 16 * (size_t)(k + 3));
            __m256d il3 = _mm256_loadu_pd(sp + 16 * (size_t)(k + 3) + 4);
            __m256d rh3 = _mm256_loadu_pd(sp + 16 * (size_t)(k + 3) + 8);
            __m256d ih3 = _mm256_loadu_pd(sp + 16 * (size_t)(k + 3) + 12);
            TR4(rl0, rl1, rl2, rl3, xr[0], xr[1], xr[2], xr[3]);
            TR4(il0, il1, il2, il3, xi[0], xi[1], xi[2], xi[3]);
            TR4(rh0, rh1, rh2, rh3, xr[4], xr[5], xr[6], xr[7]);
            TR4(ih0, ih1, ih2, ih3, xi[4], xi[5], xi[6], xi[7]);
        }
        {   /* packed per-column twiddles + squaring tree; apply legs 1..7 */
            __m256d c1 = _mm256_loadu_pd(twq + 2 * (size_t)k);
            __m256d s1 = _mm256_loadu_pd(twq + 2 * (size_t)k + 4);
            __m256d c2, s2, c3, s3, c4, s4, cw, sw, r, i;
            SPLIT_CMUL(xr[1], xi[1], c1, s1, r, i); xr[1] = r; xi[1] = i;
            WPROD(c1, s1, c1, s1, c2, s2);
            SPLIT_CMUL(xr[2], xi[2], c2, s2, r, i); xr[2] = r; xi[2] = i;
            WPROD(c2, s2, c1, s1, c3, s3);
            SPLIT_CMUL(xr[3], xi[3], c3, s3, r, i); xr[3] = r; xi[3] = i;
            WPROD(c2, s2, c2, s2, c4, s4);
            SPLIT_CMUL(xr[4], xi[4], c4, s4, r, i); xr[4] = r; xi[4] = i;
            WPROD(c4, s4, c1, s1, cw, sw);
            SPLIT_CMUL(xr[5], xi[5], cw, sw, r, i); xr[5] = r; xi[5] = i;
            WPROD(c4, s4, c2, s2, cw, sw);
            SPLIT_CMUL(xr[6], xi[6], cw, sw, r, i); xr[6] = r; xi[6] = i;
            WPROD(c4, s4, c3, s3, cw, sw);
            SPLIT_CMUL(xr[7], xi[7], cw, sw, r, i); xr[7] = r; xi[7] = i;
        }
        __m256d or_[8], oi_[8];
        SPLIT_BFLY8(xr[0],xi[0],xr[1],xi[1],xr[2],xi[2],xr[3],xi[3],
                    xr[4],xi[4],xr[5],xi[5],xr[6],xi[6],xr[7],xi[7],
                    or_[0],oi_[0],or_[1],oi_[1],or_[2],oi_[2],or_[3],oi_[3],
                    or_[4],oi_[4],or_[5],oi_[5],or_[6],oi_[6],or_[7],oi_[7]);
        for (int l = 0; l < 8; l++) {   /* re-interleave fused in the stores */
            __m256d zlo, zhi;
            REINT(or_[l], oi_[l], zlo, zhi);
            _mm256_storeu_pd(out + 2 * ((size_t)l * OLs + k), zlo);
            _mm256_storeu_pd(out + 2 * ((size_t)l * OLs + k) + 4, zhi);
        }
    }
}

/* ── chain plumbing (as zil_split_baked.c) ── */
static int N, NF;
static int R[8];
static long DD[8], GG[8];
static double *twsp[8], *twp1, *twq;
static long *gb[8];
static double *SP;

static long drev_full(long x, const int *r, int nf)
{ long v = 0; for (int i = nf - 1; i >= 0; i--) { v = v * r[i] + (x % r[i]); x /= r[i]; } return v; }
static long brev_prefix(long g, int s, const int *r)
{
    long f[8];
    for (int i = s - 1; i >= 0; i--) { f[i] = g % r[i]; g /= r[i]; }
    long P = 1, v = 0;
    for (int i = 0; i < s; i++) { v += f[i] * P; P *= r[i]; }
    return v;
}
static long base_of(long g, int s, const int *r, const long *d)
{
    long b = 0;
    for (int i = s - 1; i >= 0; i--) { long f = g % r[i]; g /= r[i]; b += f * d[i]; }
    return b;
}
static void build_tables(void)
{
    const double TAU = 2.0 * M_PI;
    DD[NF - 1] = 1;
    for (int i = NF - 2; i >= 0; i--) DD[i] = DD[i + 1] * R[i + 1];
    GG[0] = 1;
    for (int i = 1; i < NF; i++) GG[i] = GG[i - 1] * R[i - 1];
    for (int s = 1; s <= NF - 2; s++) {
        int Rm1 = R[s] - 1;
        long M = (long)N / DD[s];
        twsp[s] = (double *)_mm_malloc((size_t)GG[s] * Rm1 * 8 * 8, 64);
        gb[s] = (long *)malloc((size_t)GG[s] * sizeof(long));
        for (long g = 0; g < GG[s]; g++) {
            gb[s][g] = base_of(g, s, R, DD);
            long brev = brev_prefix(g, s, R);
            for (int l = 1; l < R[s]; l++) {
                double a = -TAU * (double)(((long)l * brev) % M) / (double)M;
                double *sp2 = twsp[s] + ((size_t)g * Rm1 + (l - 1)) * 8;
                for (int j = 0; j < 4; j++) { sp2[j] = cos(a); sp2[4 + j] = sin(a); }
            }
        }
    }
    {
        long cols = (long)N / 8, pairs = cols / 2;
        twp1 = (double *)_mm_malloc((size_t)pairs * 64, 64);
        twq = (double *)_mm_malloc((size_t)cols * 16, 64);
        for (long k = 0; k < cols; k++) {
            double a = -TAU * (double)(brev_prefix(k, NF - 1, R) % N) / (double)N;
            twq[2 * (k & ~3L) + (k & 3L)] = cos(a);
            twq[2 * (k & ~3L) + 4 + (k & 3L)] = sin(a);
        }
        for (long p = 0; p < pairs; p++) {
            double a0 = -TAU * (double)(brev_prefix(2 * p, NF - 1, R) % N) / (double)N;
            double a1 = -TAU * (double)(brev_prefix(2 * p + 1, NF - 1, R) % N) / (double)N;
            double *rec = twp1 + (size_t)p * 8;
            rec[0] = cos(a0); rec[1] = cos(a0); rec[2] = cos(a1); rec[3] = cos(a1);
            rec[4] = -sin(a0); rec[5] = sin(a0); rec[6] = -sin(a1); rec[7] = sin(a1);
        }
    }
}

/* arm: 0 = champ, 1 = tree, 2 = sterm(split-input) */
static void run_arm(int arm, int termt, const double *z0, double *A, double *out)
{
    radix4_z_s0s_fwd_avx2(z0, 0, SP, 0, 0, 0, (unsigned long long)DD[0], 0,
                          (unsigned long long)DD[0], 0, (unsigned long long)DD[0]);
    int last_ms = (arm == 2) ? NF - 2 : NF - 3;   /* sterm: ALL mids are ms */
    for (int s = 1; s <= last_ms; s++) {
        int Rm1 = R[s] - 1;
        zfn f = (R[s] == 8) ? radix8_z_ms_fwd_avx2 : radix4_z_ms_fwd_avx2;
        const long *gbs = gb[s];
        for (long g = 0; g < GG[s]; g++) {
            long b = gbs[g];
            f(SP + 2 * b, 0, SP + 2 * b, 0, twsp[s] + (size_t)g * Rm1 * 8, 0,
              (unsigned long long)DD[s], 0, (unsigned long long)DD[s], 0,
              (unsigned long long)DD[s]);
        }
    }
    if (arm == 2) {
        /* EMITTED kernel (promotion bit-gate vs the hand t2sps_r8 above) */
        radix8_z_sterm_fwd_avx2(SP, 0, out, 0, twq, 0, 0, 0,
                                (unsigned long long)(N / 8), 0,
                                (unsigned long long)(N / 8));
        return;
    }
    {
        int s = NF - 2, Rm1 = R[s] - 1;
        const long *gbs = gb[s];
        for (long g = 0; g < GG[s]; g++) {
            long b = gbs[g];
            radix8_z_msz_fwd_avx2(SP + 2 * b, 0, A + 2 * b, 0,
                                  twsp[s] + (size_t)g * Rm1 * 8, 0,
                                  (unsigned long long)DD[s], 0,
                                  (unsigned long long)DD[s], 0,
                                  (unsigned long long)DD[s]);
        }
    }
    zfn tf;
    if (arm == 1) tf = termt ? radix8_z_t2sqt_fwd_avx2 : radix8_z_t2sq_fwd_avx2;
    else          tf = termt ? radix8_z_t2spt_fwd_avx2 : radix8_z_t2sp_fwd_avx2;
    tf(A, 0, out, 0, twp1, 0, 1, 8, (unsigned long long)(N / 8), 1,
       (unsigned long long)(N / 8));
}

static double now_ms(void)
{
    LARGE_INTEGER f, cc;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&cc);
    return 1000.0 * (double)cc.QuadPart / (double)f.QuadPart;
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

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    N = argc > 1 ? atoi(argv[1]) : 4096;
    int termt = 1;
    if (N == 2048)      { NF = 4; R[0]=4; R[1]=8; R[2]=8; R[3]=8; }
    else if (N == 4096) { NF = 5; R[0]=4; R[1]=4; R[2]=4; R[3]=8; R[4]=8; }
    else if (N == 8192) { NF = 5; R[0]=4; R[1]=4; R[2]=8; R[3]=8; R[4]=8; termt = 0; }
    else if (N == 16384){ NF = 5; R[0]=4; R[1]=8; R[2]=8; R[3]=8; R[4]=8; }
    else { printf("N must be 2048|4096|8192|16384\n"); return 1; }
    build_tables();

    double *z0 = (double *)_mm_malloc((size_t)2 * N * 8, 64);
    double *A  = (double *)_mm_malloc((size_t)2 * N * 8, 64);
    double *z  = (double *)_mm_malloc((size_t)2 * N * 8, 64);
    SP = (double *)_mm_malloc((size_t)2 * N * 8, 64);
    double *Rr = (double *)_mm_malloc((size_t)N * 8, 64);
    double *Ri = (double *)_mm_malloc((size_t)N * 8, 64);
    srand(N);
    for (int i = 0; i < 2 * N; i++) z0[i] = (double)rand() / RAND_MAX - 0.5;
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

    static const char *AN[3] = { "champ", "tree ", "sterm" };
    for (int a = 0; a < 3; a++) {
        run_arm(a, termt, z0, A, z);
        long NR = (long)N / 8;
        double err = 0;
        for (long idx = 0; idx < N; idx++) {
            long l = idx / NR, g = idx % NR;
            long m = drev_full(g * 8 + l, R, NF);
            double d = fabs(z[2 * idx] - Rr[m]) + fabs(z[2 * idx + 1] - Ri[m]);
            if (d > err) err = d;
        }
        printf("GATE %s relerr=%.3e %s\n", AN[a], err / mag,
               (err / mag < 1e-11) ? "PASS" : "FAIL");
        if (err / mag >= 1e-11) return 1;
    }

    DFTI_DESCRIPTOR_HANDLE h = NULL;
    DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N);
    DftiCommitDescriptor(h);
    int reps = (int)(4.0e6 / N); if (reps < 200) reps = 200;
    double best[4] = { 1e18, 1e18, 1e18, 1e18 };
    for (int t = 0; t < 9; t++) {
        if (t) { cachebust(); Sleep(150); }
        for (int q = 0; q < 4; q++) {
            int a = (t & 1) ? 3 - q : q;
            double t0, ns;
            if (a < 3) {
                for (int w = 0; w < 6; w++) run_arm(a, termt, z0, A, z);
                t0 = now_ms();
                for (int i = 0; i < reps; i++) run_arm(a, termt, z0, A, z);
            } else {
                for (int w = 0; w < 6; w++) DftiComputeForward(h, z);
                t0 = now_ms();
                for (int i = 0; i < reps; i++) DftiComputeForward(h, z);
            }
            ns = (now_ms() - t0) * 1e6 / reps;
            if (ns < best[a]) best[a] = ns;
        }
    }
    printf("\n# N=%d TERMINATOR RACE (paced): champ vs tree-powers vs split-input\n", N);
    printf("champ (t2sp%s)         %9.1f ns   vsMKL %.2f\n", termt ? "t" : "",
           best[0], best[3] / best[0]);
    printf("tree  (t2sq%s)         %9.1f ns   vsMKL %.2f   vs champ %.2f\n",
           termt ? "t" : "", best[1], best[3] / best[1], best[0] / best[1]);
    printf("sterm (t2sps, all-ms)  %9.1f ns   vsMKL %.2f   vs champ %.2f\n",
           best[2], best[3] / best[2], best[0] / best[2]);
    printf("MKL-IL                 %9.1f ns\n", best[3]);
    printf("DONE\n");
    return 0;
}
