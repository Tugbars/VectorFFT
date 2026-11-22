/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Tuğbars Heptaşkın */
/* nd_probe.c -- a correctness probe for complex transforms of rank 2 to 7 (VFFT_C2C, interleaved, natural order),
 * through the public API alone.
 *
 * For one shape N1xN2x...xNr it creates the plan the arguments name (measured or estimate, out of place or in place,
 * at a thread count), runs it forward on seeded data and checks the output against a REFERENCE that shares nothing
 * with the rank-N tier, then runs it backward and checks the round trip:
 *   - the reference, by default: a direct O(n^2) DFT along every axis in long double (--ref direct), for shapes whose
 *     extents are small; or the library's own 1D plans applied line by line along every axis (--ref lines), for
 *     larger shapes;
 *   - fwd: the largest difference from the reference over the reference's largest magnitude;
 *   - rt:  the largest difference of backward(forward(x)) / (N1 ... Nr) from x over the largest |x|.
 * It prints one line, and the best time of the forward execute (--reps, default 5).
 *
 * usage: nd_probe N1xN2[x...] [--est] [--threads T] [--ip] [--store DIR] [--ref direct|lines] [--reps R] [--batch K]
 *   --est        plan with VFFT_FLAG_ESTIMATE (else by measuring, on the store --store names: REQUIRED then, so a
 *                test never plans into the library's own store)
 *   --batch K    a contiguous batch of K transforms (howmany = K); each transform is checked against the reference
 * exit 0 when fwd and rt are both below 1e-11, 1 otherwise, 2 on a usage or create error.
 *
 * build (from the repository root, after the gauntlet target built build/gauntlet/libvfft_core.a):
 *   gcc -O2 -Iinclude tests/rank_n/nd_probe.c build/gauntlet/libvfft_core.a -lm -o build/nd_probe.exe */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vfft.h"

#ifdef _WIN32
#include <windows.h>
static double now_ns(void)
{
    static LARGE_INTEGER f;
    LARGE_INTEGER c;
    if (!f.QuadPart)
        QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / (double)f.QuadPart;
}
#else
#include <time.h>
static double now_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}
#endif

#define MAXR 7

static uint64_t g_seed = 0x9E3779B97F4A7C15ull;
static double rnd(void)
{   /* splitmix64 -> [-0.5, 0.5) */
    uint64_t z = (g_seed += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return (double)(z >> 11) / 9007199254740992.0 - 0.5;
}

/* one axis of the reference, in place on z (interleaved, row-major): the direct DFT of every line along axis a */
static void ref_axis_direct(double *z, const int *n, int rank, int a, int sign)
{
    const int N = n[a];
    size_t inner = 1, outer = 1, o, i;
    int j, k;
    long double *wr = (long double *)malloc(sizeof(long double) * (size_t)N);
    long double *wi = (long double *)malloc(sizeof(long double) * (size_t)N);
    long double *lr = (long double *)malloc(sizeof(long double) * (size_t)N);
    long double *li = (long double *)malloc(sizeof(long double) * (size_t)N);
    const long double tau = 6.283185307179586476925286766559005768394L;
    for (j = a + 1; j < rank; j++) inner *= (size_t)n[j];
    for (j = 0; j < a; j++) outer *= (size_t)n[j];
    for (k = 0; k < N; k++)
    {
        wr[k] = cosl(tau * (long double)k / (long double)N);
        wi[k] = (long double)sign * sinl(tau * (long double)k / (long double)N);
    }
    for (o = 0; o < outer; o++)
        for (i = 0; i < inner; i++)
        {
            double *base = z + 2 * (o * (size_t)N * inner + i);
            for (j = 0; j < N; j++)
            {
                lr[j] = base[2 * (size_t)j * inner];
                li[j] = base[2 * (size_t)j * inner + 1];
            }
            for (k = 0; k < N; k++)
            {
                long double sr = 0, si = 0;
                for (j = 0; j < N; j++)
                {
                    const int t = (int)(((long long)j * k) % N);
                    sr += lr[j] * wr[t] - li[j] * wi[t];
                    si += lr[j] * wi[t] + li[j] * wr[t];
                }
                base[2 * (size_t)k * inner] = (double)sr;
                base[2 * (size_t)k * inner + 1] = (double)si;
            }
        }
    free(wr); free(wi); free(lr); free(li);
}

/* one axis of the reference through the library's 1D plan (estimate, one thread), line by line */
static int ref_axis_lines(double *z, const int *n, int rank, int a)
{
    const int N = n[a];
    size_t inner = 1, outer = 1, o, i;
    int j;
    vfft_config_t c;
    vfft_plan p;
    double *in = (double *)vfft_malloc(sizeof(double) * 2 * (size_t)N);
    double *out = (double *)vfft_malloc(sizeof(double) * 2 * (size_t)N);
    memset(&c, 0, sizeof c);
    c.transform = VFFT_C2C;
    c.n[0] = N;
    c.flags = VFFT_FLAG_ESTIMATE;
    p = vfft_create(&c);
    if (!p || !in || !out)
    {
        fprintf(stderr, "nd_probe: the reference's 1D plan at N = %d: %s\n", N, vfft_error());
        return -1;
    }
    for (j = a + 1; j < rank; j++) inner *= (size_t)n[j];
    for (j = 0; j < a; j++) outer *= (size_t)n[j];
    for (o = 0; o < outer; o++)
        for (i = 0; i < inner; i++)
        {
            double *base = z + 2 * (o * (size_t)N * inner + i);
            for (j = 0; j < N; j++)
            {
                in[2 * j] = base[2 * (size_t)j * inner];
                in[2 * j + 1] = base[2 * (size_t)j * inner + 1];
            }
            vfft_execute(p, VFFT_FORWARD, in, out);
            for (j = 0; j < N; j++)
            {
                base[2 * (size_t)j * inner] = out[2 * j];
                base[2 * (size_t)j * inner + 1] = out[2 * j + 1];
            }
        }
    vfft_destroy(p);
    vfft_free(in);
    vfft_free(out);
    return 0;
}

int main(int argc, char **argv)
{
    int n[MAXR], rank = 0, est = 0, T = 1, ip = 0, reps = 5, i, a, bad = 0;
    const char *store = NULL, *ref = "direct";
    size_t pts = 1, K = 1, q, t;
    double *x, *y, *r, *w, efwd = 0, ert = 0, mref = 0, mx = 0, best = 1e300;
    vfft_config_t c;
    vfft_plan p;
    vfft_info_t info;
    char shape[128];
    if (argc < 2)
    {
        fprintf(stderr, "usage: nd_probe N1xN2[x...] [--est] [--threads T] [--ip] [--store DIR] [--ref direct|lines] "
                        "[--reps R] [--batch K]\n");
        return 2;
    }
    snprintf(shape, sizeof shape, "%s", argv[1]);
    {
        char *s = argv[1];
        while (*s && rank < MAXR)
        {
            n[rank++] = (int)strtol(s, &s, 10);
            if (*s == 'x' || *s == 'X')
                s++;
            else
                break;
        }
        if (*s || rank < 1)
        {
            fprintf(stderr, "nd_probe: bad shape %s (up to %d extents, N1xN2x...)\n", shape, MAXR);
            return 2;
        }
    }
    for (i = 2; i < argc; i++)
    {
        if (!strcmp(argv[i], "--est")) est = 1;
        else if (!strcmp(argv[i], "--ip")) ip = 1;
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) T = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--store") && i + 1 < argc) store = argv[++i];
        else if (!strcmp(argv[i], "--ref") && i + 1 < argc) ref = argv[++i];
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch") && i + 1 < argc) K = (size_t)strtoull(argv[++i], NULL, 10);
        else
        {
            fprintf(stderr, "nd_probe: unknown argument %s\n", argv[i]);
            return 2;
        }
    }
    if (!est && !store)
    {
        fprintf(stderr, "nd_probe: a measured plan needs --store DIR (a scratch wisdom folder; never the library's own)\n");
        return 2;
    }
    for (i = 0; i < rank; i++)
        pts *= (size_t)n[i];
    memset(&c, 0, sizeof c);
    c.transform = VFFT_C2C;
    c.rank = rank;
    for (i = 0; i < rank; i++)
        c.n[i] = n[i];
    c.howmany = K;
    c.placement = ip ? VFFT_PLACEMENT_IN : VFFT_PLACEMENT_OUT;
    c.nthreads = T;
    c.flags = est ? VFFT_FLAG_ESTIMATE : 0;
    c.wisdom_dir = est ? NULL : store;
    p = vfft_create(&c);
    if (!p)
    {
        printf("%-22s rank %d T=%d %s %s K=%zu  CREATE REFUSED: %s\n", shape, rank, T, est ? "est " : "meas", ip ? "ip " : "oop",
               K, vfft_error());
        return 2;
    }
    vfft_info(p, &info);
    x = (double *)vfft_malloc(sizeof(double) * 2 * pts * K);
    y = (double *)vfft_malloc(sizeof(double) * 2 * pts * K);
    r = (double *)vfft_malloc(sizeof(double) * 2 * pts * K);
    w = (double *)vfft_malloc(sizeof(double) * 2 * pts * K);
    if (!x || !y || !r || !w)
    {
        fprintf(stderr, "nd_probe: out of memory\n");
        return 2;
    }
    for (q = 0; q < 2 * pts * K; q++)
        x[q] = rnd();
    memcpy(r, x, sizeof(double) * 2 * pts * K);
    for (t = 0; t < K; t++)
        for (a = 0; a < rank; a++)
        {
            if (!strcmp(ref, "lines"))
            {
                if (ref_axis_lines(r + 2 * pts * t, n, rank, a))
                    return 2;
            }
            else
                ref_axis_direct(r + 2 * pts * t, n, rank, a, -1);
        }
    /* forward */
    if (ip)
    {
        memcpy(y, x, sizeof(double) * 2 * pts * K);
        if (vfft_execute(p, VFFT_FORWARD, y, y)) { printf("%s: execute refused: %s\n", shape, vfft_error()); return 2; }
    }
    else if (vfft_execute(p, VFFT_FORWARD, x, y)) { printf("%s: execute refused: %s\n", shape, vfft_error()); return 2; }
    for (q = 0; q < 2 * pts * K; q++)
    {
        const double d = fabs(y[q] - r[q]), m = fabs(r[q]);
        if (d > efwd) efwd = d;
        if (m > mref) mref = m;
        if (y[q] != y[q]) bad = 1;
    }
    efwd /= mref > 0 ? mref : 1;
    /* backward: the round trip */
    if (ip)
    {
        memcpy(w, y, sizeof(double) * 2 * pts * K);
        vfft_execute(p, VFFT_BACKWARD, w, w);
    }
    else
        vfft_execute(p, VFFT_BACKWARD, y, w);
    for (q = 0; q < 2 * pts * K; q++)
    {
        const double d = fabs(w[q] / (double)pts - x[q]), m = fabs(x[q]);
        if (d > ert) ert = d;
        if (m > mx) mx = m;
    }
    ert /= mx > 0 ? mx : 1;
    /* the forward's best time */
    for (i = 0; i < reps; i++)
    {
        double t0;
        if (ip)
            memcpy(y, x, sizeof(double) * 2 * pts * K);
        t0 = now_ns();
        vfft_execute(p, VFFT_FORWARD, ip ? y : x, y);
        t0 = now_ns() - t0;
        if (t0 < best) best = t0;
    }
    bad |= !(efwd < 1e-11) || !(ert < 1e-11);
    printf("%-22s rank %d T=%d %s %s K=%zu  fwd %.2e  rt %.2e  %10.0f ns  %6.2f ns/pt  threads=%d  %s\n", shape, rank, T,
           est ? "est " : "meas", ip ? "ip " : "oop", K, efwd, ert, best, best / (double)(pts * K), info.nthreads,
           bad ? "FAIL" : "ok");
    vfft_destroy(p);
    vfft_free(x); vfft_free(y); vfft_free(r); vfft_free(w);
    return bad ? 1 : 0;
}
