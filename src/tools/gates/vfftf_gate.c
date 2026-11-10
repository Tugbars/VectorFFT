/* vfftf_gate.c -- THE FP32 LIBRARY'S GATE (2026-11-10): the whole served scope
 * of include/vfftf.h through its public door, against a double-precision
 * reference, and its refusals.
 *
 * SERVED (vfftf.h): 1D C2C, power-of-two N, interleaved, natural order, in
 * place or out of place, one transform or a batch, any nthreads. Per
 * N = 2, 4, ..., --max (default 2^20, at most 2^24), K = 1 and a contiguous
 * batch of K = 3 (transform t at complex offset t N), nthreads 1 and 4:
 *   1. an out-of-place plan and an in-place plan are created on the scratch
 *      store (COLD on purpose, run_gates: ("flag", False): every create races
 *      and banks into wisdom2f_oop.txt there);
 *   2. FORWARD and BACKWARD out of place, each against the double-precision
 *      reference of the SAME float input converted to double: an O(N log N)
 *      radix-2 FFT in double written here, its twiddles cos/sin per index, so
 *      the reference carries double's rounding only. The error is the
 *      elementwise max |y - Y| over the reference's max magnitude across the
 *      whole batch; the FP32 limit is 1e-5 and a NaN fails. The output buffers
 *      are filled with NaN before each execute, so an element the plan never
 *      writes fails too;
 *   3. the in-place plan's forward and backward, each on a fresh copy of the
 *      input, against the same references and limit (an element the plan never
 *      writes keeps the input's value and fails). Whether they are bitwise the
 *      out-of-place results is printed, never asserted: each placement is raced
 *      and banked as its own row, so the two may run different chains (cold,
 *      1024 raced 2p 32.32 out of place and ZTURN-T 8.8.4.4 in place); where
 *      the two rows hold the same plan the results were bitwise at every N.
 *   4. the OUT-OF-PLACE plan called with ONE buffer (out == in), forward and
 *      backward on a fresh copy of the input: served, against the same
 *      references and limit (the solo kernels' plan runs its alias-tolerant
 *      twins, every other route its own in-place form);
 *   5. the IN-PLACE plan called with TWO buffers: refused at the public door
 *      (vfftf.h: in place, out == in) -- vfft_error() names the plan IN-PLACE
 *      and the output buffer keeps its NaN fill.
 * REFUSED (vfftf.h: every other configuration): N = 3, N = 12, rank 2, a real
 * transform (R2C), the split layout -- vfftf_create returns NULL and
 * vfft_error() is not empty.
 * Each plan's route (vfftf_info) is printed, never asserted.
 *
 * Built only with VFFT_HAS_FLOAT (gauntlet/build.py --float, or VFFT_FLOAT=1,
 * with the library's FP32 core): without it the gate prints SKIP and passes.
 * Run:   vfftf_gate.exe --wisdir <scratch dir> [--max N]
 * Build: VFFT_FLOAT=1 python gauntlet/build.py --compile --vfft --src src/tools/gates/vfftf_gate.c
 *        (run_gates.py builds it; VFFT_FLOAT=1 in its environment reaches build.py) */
#include <stdio.h>

#ifndef VFFT_HAS_FLOAT
int main(void)
{
    printf("SKIP: no FP32 build (VFFT_HAS_FLOAT is not defined: gauntlet/build.py --float or VFFT_FLOAT=1)\n");
    return 0;
}
#else
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfftf.h"

#define LIMIT 1e-5        /* the FP32 pass limit: relative, elementwise max */
#define KB 3              /* the contiguous batch */

static const double PI = 3.14159265358979323846264338327950288;

/* the twiddle table of the largest N: cos and -sin of 2 pi j / NT, j < NT / 2 */
static double *g_twr, *g_twi;
static int g_nt;

static void tw_init(int nt)
{
    g_nt = nt;
    g_twr = (double *)malloc((size_t)(nt / 2 + 1) * sizeof(double));
    g_twi = (double *)malloc((size_t)(nt / 2 + 1) * sizeof(double));
    if (!g_twr || !g_twi) { printf("out of memory (twiddles)\n"); exit(2); }
    for (int j = 0; j < nt / 2; j++)
    {
        const double a = 2.0 * PI * (double)j / (double)nt;
        g_twr[j] = cos(a);
        g_twi[j] = -sin(a);
    }
}

/* the reference: an in-place iterative radix-2 DIT FFT in double over N complex
 * values (N a power of two, N <= g_nt), forward (e^-) or backward (e^+), unnormalized */
static void ref_fft(double *z, int N, int backward)
{
    for (int i = 1, j = 0; i < N; i++)
    {   /* the bit-reversal permutation */
        int bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j)
        {
            double t = z[2 * i]; z[2 * i] = z[2 * j]; z[2 * j] = t;
            t = z[2 * i + 1]; z[2 * i + 1] = z[2 * j + 1]; z[2 * j + 1] = t;
        }
    }
    for (int len = 2; len <= N; len <<= 1)
    {
        const int half = len >> 1, step = g_nt / len;   /* W_len^k = W_nt^(k nt / len) */
        for (int i = 0; i < N; i += len)
            for (int k = 0; k < half; k++)
            {
                const double wr = g_twr[k * step], wi = backward ? -g_twi[k * step] : g_twi[k * step];
                double *a = z + 2 * (i + k), *b = z + 2 * (i + k + half);
                const double br = b[0] * wr - b[1] * wi, bi = b[0] * wi + b[1] * wr;
                b[0] = a[0] - br; b[1] = a[1] - bi;
                a[0] += br; a[1] += bi;
            }
    }
}

/* max_j |a_j - b_j| / max_j |b_j| over n values; NaN when a holds a NaN */
static double relerr(const float *a, const double *b, size_t n)
{
    double e = 0.0, m = 0.0;
    for (size_t j = 0; j < n; j++)
    {
        const double d = fabs((double)a[j] - b[j]);
        if (d != d) return NAN;
        if (d > e) e = d;
        if (fabs(b[j]) > m) m = fabs(b[j]);
    }
    return m > 0 ? e / m : e;
}

/* the input: a fixed 64-bit LCG, uniform in [-0.5, 0.5), rounded to float */
static unsigned long long g_rng = 0x9E3779B97F4A7C15ull;
static float rnd(void)
{
    g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((double)(g_rng >> 11) * (1.0 / 9007199254740992.0) - 0.5);
}

static void *xalloc(size_t bytes)
{
    void *p = vfft_malloc(bytes);
    if (!p) { printf("out of memory (%zu bytes)\n", bytes); exit(2); }
    return p;
}

static vfft_config_t cfg_of(const char *W, int N, int ip, int K, int T)
{
    vfft_config_t c;
    memset(&c, 0, sizeof c);
    c.transform = VFFT_C2C;
    c.rank = 1;
    c.n[0] = N;
    c.howmany = (size_t)K;
    c.layout = VFFT_LAYOUT_INTERLEAVED;
    c.placement = ip ? VFFT_PLACEMENT_IN : VFFT_PLACEMENT_OUT;
    c.order = VFFT_ORDER_NATURAL;
    c.batch = K > 1 ? VFFT_BATCH_CONTIGUOUS : VFFT_BATCH_DEFAULT;
    c.nthreads = T;
    c.wisdom_dir = W;
    return c;
}

static const char *route_of(vfftf_plan p)
{
    vfft_info_t info;
    memset(&info, 0, sizeof info);
    if (vfftf_info(p, &info) != 0 || !info.route || !info.route[0]) return "-";
    return info.route;
}

/* one refused configuration: NULL and a reason */
static int refused(const char *name, const vfft_config_t *c)
{
    vfftf_plan p = vfftf_create(c);
    const char *why = vfft_error();
    if (p)
    {
        printf("  *** FAIL *** %-24s served (route %s): vfftf.h refuses it\n", name, route_of(p));
        vfftf_destroy(p);
        return 0;
    }
    if (!why || !why[0])
    {
        printf("  *** FAIL *** %-24s refused with an EMPTY vfft_error()\n", name);
        return 0;
    }
    printf("  ok  %-24s refused: %.100s\n", name, why);
    return 1;
}

int main(int argc, char **argv)
{
    const char *W = NULL;
    int nmax = 1 << 20, fails = 0, cells = 0;
    double maxf = 0, maxb = 0;
    int maxf_n = 0, maxb_n = 0;
    for (int a = 1; a + 1 < argc; a++)
    {
        if (!strcmp(argv[a], "--wisdir")) W = argv[a + 1];
        else if (!strcmp(argv[a], "--max")) nmax = atoi(argv[a + 1]);
    }
    if (!W || nmax < 2 || nmax > (1 << 24) || (nmax & (nmax - 1)))
    {
        printf("usage: %s --wisdir <scratch dir> [--max N]   (N a power of two, 2 .. 2^24; default 2^20)\n", argv[0]);
        return 2;
    }
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== FP32 gate (vfftf.h): 1D C2C pow2 N = 2..%d, K = 1 and %d (contiguous), T = 1 and 4, oop + ip, "
           "fwd + bwd vs a double FFT, limit %.0e, oop and ip each; the oop plan on one buffer; the ip plan "
           "with two buffers refused; store %s ===\n", nmax, KB, LIMIT, W);

    /* ── the refusals ── */
    {
        vfft_config_t c;
        printf("refusals:\n");
        c = cfg_of(W, 3, 0, 1, 1);
        if (!refused("N = 3", &c)) fails++;
        c = cfg_of(W, 12, 0, 1, 1);
        if (!refused("N = 12", &c)) fails++;
        c = cfg_of(W, 16, 0, 1, 1);
        c.rank = 2; c.n[1] = 16;
        if (!refused("rank 2, 16 x 16", &c)) fails++;
        c = cfg_of(W, 64, 0, 1, 1);
        c.transform = VFFT_R2C;
        if (!refused("real (R2C), N = 64", &c)) fails++;
        c = cfg_of(W, 64, 0, 1, 1);
        c.layout = VFFT_LAYOUT_SPLIT;
        if (!refused("split layout, N = 64", &c)) fails++;
    }

    /* ── the served scope ── */
    tw_init(nmax);
    printf("served:\n %-8s %-2s %-2s | %-14s %-14s | %-9s %-9s | %-9s %-9s | %-9s %-9s | %-7s | %s\n",
           "N", "K", "T", "route oop", "route ip", "fwd err", "bwd err", "ip fwd", "ip bwd", "1buf fwd", "1buf bwd",
           "ip 2buf", "ip vs oop");
    for (int N = 2; N <= nmax; N <<= 1)
    {
        const size_t tot = 2 * (size_t)N * KB;          /* floats of the K = 3 batch */
        float *x = (float *)xalloc(tot * sizeof(float));
        float *yf = (float *)xalloc(tot * sizeof(float)), *yb = (float *)xalloc(tot * sizeof(float));
        float *z = (float *)xalloc(tot * sizeof(float));
        double *rf = (double *)malloc(tot * sizeof(double)), *rb = (double *)malloc(tot * sizeof(double));
        if (!rf || !rb) { printf("out of memory (the reference at N = %d)\n", N); return 2; }
        for (size_t j = 0; j < tot; j++) x[j] = rnd();
        /* the references of the three transforms (K = 1 is transform 0) */
        for (size_t j = 0; j < tot; j++) rf[j] = rb[j] = (double)x[j];
        for (int t = 0; t < KB; t++)
        {
            ref_fft(rf + 2 * (size_t)N * t, N, 0);
            ref_fft(rb + 2 * (size_t)N * t, N, 1);
        }
        for (int ti = 0; ti < 2; ti++)
            for (int ki = 0; ki < 2; ki++)
            {
                const int T = ti ? 4 : 1, K = ki ? KB : 1;
                const size_t n = 2 * (size_t)N * K;      /* floats of this cell's data */
                vfft_config_t co = cfg_of(W, N, 0, K, T), ci = cfg_of(W, N, 1, K, T);
                vfftf_plan po = vfftf_create(&co);
                char why_o[160] = "";
                vfftf_plan pi;
                if (!po) snprintf(why_o, sizeof why_o, "%s", vfft_error());
                pi = vfftf_create(&ci);
                cells++;
                if (!po || !pi)
                {
                    printf(" %-8d %-2d %-2d | *** FAIL *** refused, a served cell\n", N, K, T);
                    if (!po) printf("        oop: %s\n", why_o);
                    if (!pi) printf("        ip:  %s\n", vfft_error());
                    fails++;
                    if (po) vfftf_destroy(po);
                    if (pi) vfftf_destroy(pi);
                    continue;
                }
                memset(yf, 0xff, n * sizeof(float));   /* NaN: an unwritten element fails */
                vfftf_execute(po, VFFT_FORWARD, x, yf);
                memset(yb, 0xff, n * sizeof(float));
                vfftf_execute(po, VFFT_BACKWARD, x, yb);
                const double ef = relerr(yf, rf, n), eb = relerr(yb, rb, n);
                memcpy(z, x, n * sizeof(float));
                vfftf_execute(pi, VFFT_FORWARD, z, z);
                const double eif = relerr(z, rf, n);
                const int bit_f = memcmp(z, yf, n * sizeof(float)) == 0;
                memcpy(z, x, n * sizeof(float));
                vfftf_execute(pi, VFFT_BACKWARD, z, z);
                const double eib = relerr(z, rb, n);
                const int bit_b = memcmp(z, yb, n * sizeof(float)) == 0;
                /* 4: the out-of-place plan on one buffer */
                memcpy(z, x, n * sizeof(float));
                vfftf_execute(po, VFFT_FORWARD, z, z);
                const double eaf = relerr(z, rf, n);
                memcpy(z, x, n * sizeof(float));
                vfftf_execute(po, VFFT_BACKWARD, z, z);
                const double eab = relerr(z, rb, n);
                /* 5: the in-place plan with two buffers (this cell's only refusal: both creates left
                 * vfft_error() empty); yf holds the forward result of law 2 until here */
                int refused2 = 1;
                {
                    const char *why;
                    memset(yf, 0xff, n * sizeof(float));
                    vfftf_execute(pi, VFFT_FORWARD, x, yf);
                    why = vfft_error();
                    if (!why || !strstr(why, "IN-PLACE")) refused2 = 0;
                    for (size_t j = 0; j < n * sizeof(float); j++)
                        if (((const unsigned char *)yf)[j] != 0xff) { refused2 = 0; break; }
                }
                const int ok = ef < LIMIT && eb < LIMIT && eif < LIMIT && eib < LIMIT && eaf < LIMIT && eab < LIMIT && refused2;
                if (maxf == maxf && !(ef <= maxf)) { maxf = ef; maxf_n = N; }   /* a NaN is the largest, and stays */
                if (maxb == maxb && !(eb <= maxb)) { maxb = eb; maxb_n = N; }
                if (maxf == maxf && !(eif <= maxf)) { maxf = eif; maxf_n = N; }
                if (maxb == maxb && !(eib <= maxb)) { maxb = eib; maxb_n = N; }
                if (maxf == maxf && !(eaf <= maxf)) { maxf = eaf; maxf_n = N; }
                if (maxb == maxb && !(eab <= maxb)) { maxb = eab; maxb_n = N; }
                printf(" %-8d %-2d %-2d | %-14s %-14s | %.2e  %.2e | %.2e  %.2e | %.2e  %.2e | %-7s | %s%s\n", N, K, T,
                       route_of(po), route_of(pi), ef, eb, eif, eib, eaf, eab, refused2 ? "refused" : "SERVED",
                       bit_f && bit_b ? "bitwise" : "own plan",
                       ok ? "" : "   *** FAIL ***");
                if (!ok) fails++;
                vfftf_destroy(po);
                vfftf_destroy(pi);
            }
        vfft_free(x); vfft_free(yf); vfft_free(yb); vfft_free(z);
        free(rf); free(rb);
    }
    free(g_twr); free(g_twi);
    printf("largest error: forward %.2e (N = %d), backward %.2e (N = %d); limit %.0e\n", maxf, maxf_n, maxb, maxb_n, LIMIT);
    if (fails)
        printf("=== *** FAIL *** (%d failed of %d served cells and 5 refusals) ===\n", fails, cells);
    else
        printf("=== ALL PASS === (%d served cells, 5 refusals)\n", cells);
    return fails ? 1 : 0;
}
#endif /* VFFT_HAS_FLOAT */
