/* natorder_g3_oopnat.c — GATE G3: is OOP(-natural) + COPY-BACK a free lunch for
 * API-level in-place NATURAL order, vs the current SCRAMBLED in-place c2c?
 *
 * Public vfft.h API only. For each cell (N,K):
 *   (1) in-place c2c plan (VFFT_C2C, VFFT_INPLACE, VFFT_MEASURE, nthreads=1),
 *       time fwd execute (QPC best-of-5, buffers refreshed between trials).
 *   (2) OOP c2c plan (VFFT_OUTOFPLACE), time fwd execute alone AND fwd execute
 *       PLUS memcpy of both output planes back over the input planes
 *       (simulating in-place-natural via scratch+copyback).
 *   Gates: roundtrip fwd->bwd/N == x for both plans; OOP forward order checked
 *   against a naive O(N^2) DFT on lane 0 (natural vs scrambled — MODEB would be
 *   scrambled, which voids the "natural for free" claim for that cell).
 *   OOP kind is read back from the scratch wisdom (natorder_wis/oop_wisdom.txt).
 *
 * Scratch wisdom dir: VFFT_WISDOM_DIR=natorder_wis (putenv BEFORE first create) —
 * never touches real wisdom. Creates calibrate-on-miss (slow) — only execute is timed.
 *
 * Build: python build.py --src test/natorder_g3_oopnat.c --vfft --compile
 * Run  : test/natorder_g3_oopnat.exe [N K]   (no args = full 6x3 sweep)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include "vfft.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double g_qpf; /* QueryPerformanceFrequency, cached */
static double now_ns(void)
{
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / g_qpf;
}

static double *ad(size_t n)
{
    void *p = _aligned_malloc(n * sizeof(double), 64);
    if (!p) { fprintf(stderr, "alloc fail\n"); exit(1); }
    return (double *)p;
}
static void af(double *p) { _aligned_free(p); }

static void cachebust(void)
{
    size_t s = 32u * 1024 * 1024 / sizeof(double);
    double *j = ad(s);
    volatile double a = 0;
    for (size_t i = 0; i < s; i++) j[i] = (double)i * 0.5;
    for (size_t i = 0; i < s; i++) a += j[i];
    (void)a;
    af(j);
}

/* canonical harness rep count: ~2e6 total elems per trial, clamped [8,100000] */
static int reps_for(size_t total)
{
    int r = (int)(2e6 / (double)(total + 1));
    if (r < 8) r = 8;
    if (r > 100000) r = 100000;
    return r;
}

static vfft_plan mk(int N, size_t K, vfft_placement_t pl)
{
    vfft_config_t c;
    memset(&c, 0, sizeof c);
    c.transform = VFFT_C2C;
    c.placement = pl;
    c.rigor     = VFFT_MEASURE;
    c.dims      = 1;
    c.n[0]      = N;
    c.howmany   = K;
    c.nthreads  = 1;
    return vfft_create(&c);
}

/* ── (1) scrambled in-place forward: best-of-5, refresh buffers between trials ── */
static double time_ip(vfft_plan p, double *re, double *im,
                      const double *sr, const double *si, size_t total)
{
    size_t B = total * sizeof(double);
    memcpy(re, sr, B); memcpy(im, si, B);
    for (int w = 0; w < 10; w++) vfft_execute(p, VFFT_FORWARD, re, im, re, im);
    int reps = reps_for(total);
    double best = 1e18;
    for (int t = 0; t < 5; t++)
    {
        memcpy(re, sr, B); memcpy(im, si, B);      /* touch/refresh outside timed region */
        double t0 = now_ns();
        for (int i = 0; i < reps; i++) vfft_execute(p, VFFT_FORWARD, re, im, re, im);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }
    return best;
}

/* ── (2a) OOP forward alone (src stays fresh; dst overwritten) ── */
static double time_oop(vfft_plan p, double *re, double *im, double *dr, double *di,
                       const double *sr, const double *si, size_t total)
{
    size_t B = total * sizeof(double);
    memcpy(re, sr, B); memcpy(im, si, B);
    for (int w = 0; w < 10; w++) vfft_execute(p, VFFT_FORWARD, re, im, dr, di);
    int reps = reps_for(total);
    double best = 1e18;
    for (int t = 0; t < 5; t++)
    {
        memcpy(re, sr, B); memcpy(im, si, B);
        double t0 = now_ns();
        for (int i = 0; i < reps; i++) vfft_execute(p, VFFT_FORWARD, re, im, dr, di);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }
    return best;
}

/* ── (2b) OOP forward + copyback of BOTH output planes over the input planes
 *        (the simulated in-place-natural: scratch + copyback) ── */
static double time_oop_cb(vfft_plan p, double *re, double *im, double *dr, double *di,
                          const double *sr, const double *si, size_t total)
{
    size_t B = total * sizeof(double);
    memcpy(re, sr, B); memcpy(im, si, B);
    for (int w = 0; w < 10; w++)
    {
        vfft_execute(p, VFFT_FORWARD, re, im, dr, di);
        memcpy(re, dr, B); memcpy(im, di, B);
    }
    int reps = reps_for(total);
    double best = 1e18;
    for (int t = 0; t < 5; t++)
    {
        memcpy(re, sr, B); memcpy(im, si, B);
        double t0 = now_ns();
        for (int i = 0; i < reps; i++)
        {
            vfft_execute(p, VFFT_FORWARD, re, im, dr, di);
            memcpy(re, dr, B); memcpy(im, di, B);
        }
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }
    return best;
}

/* roundtrip gate: fwd then bwd, /N == x. In-place variant runs on one buffer pair. */
static double rt_ip(vfft_plan p, double *re, double *im,
                    const double *sr, const double *si, int N, size_t total)
{
    size_t B = total * sizeof(double);
    memcpy(re, sr, B); memcpy(im, si, B);
    vfft_execute(p, VFFT_FORWARD, re, im, re, im);
    vfft_execute(p, VFFT_BACKWARD, re, im, re, im);
    double e = 0, inv = 1.0 / (double)N;
    for (size_t i = 0; i < total; i++)
    {
        double a = fabs(re[i] * inv - sr[i]), b = fabs(im[i] * inv - si[i]);
        if (a > e) e = a;
        if (b > e) e = b;
    }
    return e;
}
static double rt_oop(vfft_plan p, double *re, double *im, double *dr, double *di,
                     double *er, double *ei,
                     const double *sr, const double *si, int N, size_t total)
{
    size_t B = total * sizeof(double);
    memcpy(re, sr, B); memcpy(im, si, B);
    vfft_execute(p, VFFT_FORWARD, re, im, dr, di);
    vfft_execute(p, VFFT_BACKWARD, dr, di, er, ei);
    double e = 0, inv = 1.0 / (double)N;
    for (size_t i = 0; i < total; i++)
    {
        double a = fabs(er[i] * inv - sr[i]), b = fabs(ei[i] * inv - si[i]);
        if (a > e) e = a;
        if (b > e) e = b;
    }
    return e;
}

/* naive O(N^2) DFT on lane 0 -> is the OOP forward output NATURAL order?
 * (LEAF/BAILEY2 => natural; MODEB => scrambled, which voids the free-natural claim.) */
static int oop_is_natural(vfft_plan p, double *re, double *im, double *dr, double *di,
                          const double *sr, const double *si, int N, size_t K, size_t total)
{
    size_t B = total * sizeof(double);
    memcpy(re, sr, B); memcpy(im, si, B);
    vfft_execute(p, VFFT_FORWARD, re, im, dr, di);
    double maxe = 0, maxm = 0;
    for (int k = 0; k < N; k++)
    {
        double Xr = 0, Xi = 0;
        for (int n = 0; n < N; n++)
        {
            double x = sr[(size_t)n * K], y = si[(size_t)n * K];
            double ang = -2.0 * M_PI * (double)((long long)k * n % N) / (double)N;
            double c = cos(ang), s = sin(ang);
            Xr += x * c - y * s;
            Xi += x * s + y * c;
        }
        double a = dr[(size_t)k * K] - Xr, b = di[(size_t)k * K] - Xi;
        double e = sqrt(a * a + b * b), m = sqrt(Xr * Xr + Xi * Xi);
        if (e > maxe) maxe = e;
        if (m > maxm) maxm = m;
    }
    return (maxm > 0 ? maxe / maxm : maxe) < 1e-9;
}

/* read back the OOP kind the calibrator banked for (N,K) from the scratch wisdom */
static const char *oop_kind_from_wisdom(int N, size_t K)
{
    FILE *f = fopen("natorder_wis/oop_wisdom.txt", "r");
    if (!f) return "unknown";
    static const char *names[3] = {"LEAF", "BAILEY2", "MODEB"};
    const char *r = "unknown";
    char line[512];
    while (fgets(line, sizeof line, f))
    {
        int n, kind; unsigned long long k;
        if (sscanf(line, "%d %llu %d", &n, &k, &kind) == 3 &&
            n == N && (size_t)k == K && kind >= 0 && kind <= 2)
        { r = names[kind]; break; }
    }
    fclose(f);
    return r;
}

static void run_cell(int N, size_t K)
{
    size_t total = (size_t)N * K;
    double *sr = ad(total), *si = ad(total);
    double *re = ad(total), *im = ad(total);
    double *dr = ad(total), *di = ad(total);
    double *er = ad(total), *ei = ad(total);
    srand(97 + N + (int)K);
    for (size_t i = 0; i < total; i++)
    {
        sr[i] = (double)rand() / RAND_MAX - 0.5;
        si[i] = (double)rand() / RAND_MAX - 0.5;
    }

    /* create BOTH plans first: all calibrate-on-miss noise lands before timing */
    vfft_plan pip = mk(N, K, VFFT_INPLACE);
    vfft_plan pop = mk(N, K, VFFT_OUTOFPLACE);
    if (!pip || !pop)
    {
        printf("%d,%zu,CREATE_NULL(ip=%p oop=%p)\n", N, K, (void *)pip, (void *)pop);
        if (pip) vfft_destroy(pip);
        if (pop) vfft_destroy(pop);
        goto done;
    }

    double eip = rt_ip(pip, re, im, sr, si, N, total);
    double eop = rt_oop(pop, re, im, dr, di, er, ei, sr, si, N, total);
    int natural = oop_is_natural(pop, re, im, dr, di, sr, si, N, K, total);
    const char *kind = oop_kind_from_wisdom(N, K);

    /* measure: in-place | cachebust+idle | OOP pure | cachebust+idle | OOP+copyback */
    double ns_ip = time_ip(pip, re, im, sr, si, total);
    cachebust(); Sleep(150);
    double ns_op = time_oop(pop, re, im, dr, di, sr, si, total);
    cachebust(); Sleep(150);
    double ns_cb = time_oop_cb(pop, re, im, dr, di, sr, si, total);

    printf("%d,%zu,%s,%s,%.1e,%.1e,%.0f,%.0f,%.0f,%.3f,%.3f\n",
           N, K, kind, natural ? "natural" : "scrambled",
           eip, eop, ns_ip, ns_op, ns_cb, ns_cb / ns_ip, ns_op / ns_ip);

    vfft_destroy(pip);
    vfft_destroy(pop);
done:
    af(sr); af(si); af(re); af(im); af(dr); af(di); af(er); af(ei);
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    LARGE_INTEGER f;
    QueryPerformanceFrequency(&f);
    g_qpf = (double)f.QuadPart;
    SetThreadAffinityMask(GetCurrentThread(), 1); /* pin core 0 (P-core) */

    putenv("VFFT_WISDOM_DIR=natorder_wis");       /* scratch wisdom — BEFORE first create */
    system("mkdir natorder_wis 2>nul");
    vfft_set_num_threads(1);

    printf("# G3: OOP+copyback vs scrambled in-place (public API, MEASURE, nthreads=1)\n");
    printf("# isa=%s version=%s\n", vfft_isa(), vfft_version());
    printf("N,K,oop_kind,oop_order,rt_ip,rt_oop,ns_inplace,ns_oop_pure,ns_oop_copyback,ratio_cb_vs_ip,ratio_pure_vs_ip\n");

    if (argc >= 3)
    {
        run_cell(atoi(argv[1]), (size_t)atoi(argv[2]));
        return 0;
    }
    int Ns[] = {64, 128, 256, 512, 1024, 4096};
    int Ks[] = {8, 64, 256};
    for (int ni = 0; ni < 6; ni++)
        for (int ki = 0; ki < 3; ki++)
            run_cell(Ns[ni], (size_t)Ks[ki]);
    return 0;
}
