/* dfti_shim_gate.c -- THE DFTI INTERFACE'S OWN REGRESSION (docs/roadmap/dfti_shim_design.md, 2026-10-16).
 *
 * A program of DFTI calls, built against src/shims/dfti/mkl_dfti.h and judged against a naive DFT:
 *   C  complex: one transform at ranks 1 to 3 in both placements, contiguous and lane-major batches,
 *      offsets, both directions (the backward scale's roundtrip; its recommit keeps the plans);
 *   R  real: rank 1 in both placements (both directions from one commit), ranks 2 and 3 with the
 *      strides swapped and recommitted between directions (the unfit direction refused with 3),
 *      contiguous and lane-major batches at ranks 1 to 3 in both placements (in place: padded planes or
 *      volumes one after another; the lane-major rows padded to (n + 2) K); the c2r judged on the exact
 *      spectrum;
 *   S  the statuses: an uncommitted compute 2, a NULL handle 5 and the NULL-handle VERSION query 0,
 *      single precision and split storage 6, the settings DFTI does not implement 3, a batch with no
 *      distance 3, a rank-2 real transform on the default strides (3 in place, 6 out of place), an
 *      unserved layout 6, a copy, BACKWARD_SCRAMBLED served in natural order, the readbacks;
 *   M  one descriptor computed from eight threads at once (NUMBER_OF_USER_THREADS 8, then 1), every
 *      result bitwise the one-thread result; THREAD_LIMIT 0 reads back vfft_info().cores.
 * Cold scratch store (VFFT_WISDOM_DIR): every cell races. Usage: dfti_shim_gate --wisdir <dir> */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../shims/dfti/vfft_dfti.c" /* the interface in this unit: the gate reads its plans */
#ifdef _WIN32
#include <process.h>
#else
#include <pthread.h>
#endif

static int g_checks = 0, g_fail = 0;
#define CHECK(c, ...) do { g_checks++; if (!(c)) { g_fail++; printf("  *** FAIL: " __VA_ARGS__); printf("\n"); } } while (0)
#define GPI 3.14159265358979323846

static void fill(double *x, size_t n, unsigned seed)
{
    unsigned s = seed * 2654435761u + 12345u;
    size_t i;
    for (i = 0; i < n; i++)
    {
        s = s * 1664525u + 1013904223u;
        x[i] = (double)(s >> 8) / 16777216.0 - 0.5;
    }
}
/* row-major strides over the extents e times m, the first element at s0 */
static void rm(const MKL_LONG *e, int r, MKL_LONG m, MKL_LONG s0, MKL_LONG *s)
{
    int k;
    s[0] = s0;
    s[r] = m;
    for (k = r - 1; k >= 1; k--)
        s[k] = s[k + 1] * e[k];
}
/* DFTI's address of element lin (row-major over the extents e) of transform t, in elements */
static size_t at(size_t lin, int r, const MKL_LONG *e, const MKL_LONG *s, MKL_LONG dist, size_t t)
{
    size_t o = (size_t)s[0] + t * (size_t)(dist > 0 ? dist : 0);
    int a;
    for (a = r - 1; a >= 0; a--)
    {
        o += (size_t)s[a + 1] * (lin % (size_t)e[a]);
        lin /= (size_t)e[a];
    }
    return o;
}
/* the elements a layout spans */
static size_t span(int r, const MKL_LONG *e, const MKL_LONG *s, MKL_LONG dist, size_t K)
{
    size_t o = (size_t)s[0] + (K - 1) * (size_t)(dist > 0 ? dist : 0);
    int a;
    for (a = 0; a < r; a++)
        o += (size_t)s[a + 1] * (size_t)(e[a] - 1);
    return o + 1;
}
/* the naive multidimensional DFT of a contiguous complex array, one axis at a time; sign -1 forward */
static void naive_nd(double *z, int r, const MKL_LONG *n, int sign)
{
    size_t P = 1, stride = 1;
    int a;
    for (a = 0; a < r; a++)
        P *= (size_t)n[a];
    for (a = r - 1; a >= 0; a--)
    {
        const size_t N = (size_t)n[a], outer = P / (N * stride);
        double *t = (double *)malloc(2 * N * sizeof(double));
        size_t o, s, k, j;
        for (o = 0; o < outer; o++)
            for (s = 0; s < stride; s++)
            {
                double *b = z + 2 * (o * N * stride + s);
                for (k = 0; k < N; k++)
                {
                    double sr = 0, si = 0;
                    for (j = 0; j < N; j++)
                    {
                        const double ang = sign * 2.0 * GPI * (double)((k * j) % N) / (double)N;
                        const double c = cos(ang), sn = sin(ang), xr = b[2 * j * stride], xi = b[2 * j * stride + 1];
                        sr += xr * c - xi * sn;
                        si += xr * sn + xi * c;
                    }
                    t[2 * k] = sr;
                    t[2 * k + 1] = si;
                }
                for (k = 0; k < N; k++)
                {
                    b[2 * k * stride] = t[2 * k];
                    b[2 * k * stride + 1] = t[2 * k + 1];
                }
            }
        free(t);
        stride *= N;
    }
}
/* the row-major index over n of the element at row-major index lin over e (e[k] <= n[k]) */
static size_t full_of(size_t lin, int r, const MKL_LONG *e, const MKL_LONG *n)
{
    size_t idx[4], f = 0;
    int a;
    for (a = r - 1; a >= 0; a--)
    {
        idx[a] = lin % (size_t)e[a];
        lin /= (size_t)e[a];
    }
    for (a = 0; a < r; a++)
        f = f * (size_t)n[a] + idx[a];
    return f;
}
static double relerr(const double *y, const double *ref, size_t nd)
{
    double m = 0, e = 0;
    size_t i;
    for (i = 0; i < nd; i++)
    {
        if (fabs(ref[i]) > m) m = fabs(ref[i]);
        if (fabs(y[i] - ref[i]) > e) e = fabs(y[i] - ref[i]);
    }
    return m > 0 ? e / m : e;
}
static MKL_LONG mkdesc(DFTI_DESCRIPTOR_HANDLE *h, int real, int r, const MKL_LONG *n)
{
    if (r == 1) /* rank 1 passes the length itself, as DFTI requires */
        return DftiCreateDescriptor(h, DFTI_DOUBLE, real ? DFTI_REAL : DFTI_COMPLEX, 1, n[0]);
    return DftiCreateDescriptor(h, DFTI_DOUBLE, real ? DFTI_REAL : DFTI_COMPLEX, r, n);
}
static void shape_of(char *buf, size_t sz, int r, const MKL_LONG *n)
{
    if (r == 1) snprintf(buf, sz, "%ld", (long)n[0]);
    else if (r == 2) snprintf(buf, sz, "%ldx%ld", (long)n[0], (long)n[1]);
    else snprintf(buf, sz, "%ldx%ldx%ld", (long)n[0], (long)n[1], (long)n[2]);
}

/* ── C: complex ────────────────────────────────────────────────────────── */
static void cell_c(int r, const MKL_LONG *n, int K, int lane, int ip, MKL_LONG s0i, MKL_LONG s0o, MKL_LONG tlim)
{
    MKL_LONG is[5], os[5], P = 1, dist;
    DFTI_DESCRIPTOR_HANDLE h = NULL;
    size_t ni, no, i, t;
    double *x, *y, *xin, *a, *b, e = 0, ert = 0;
    vfft_plan p0;
    char shape[48], what[160];
    int k;
    MKL_LONG st;
    for (k = 0; k < r; k++)
        P *= n[k];
    rm(n, r, lane ? K : 1, s0i, is);
    rm(n, r, lane ? K : 1, ip ? s0i : s0o, os);
    dist = lane ? 1 : P;
    ni = span(r, n, is, dist, (size_t)K);
    no = ip ? ni : span(r, n, os, dist, (size_t)K);
    shape_of(shape, sizeof shape, r, n);
    snprintf(what, sizeof what, "c2c %-9s K=%-2d %-10s %-3s s0=%ld/%ld T=%ld", shape, K,
             K > 1 ? (lane ? "lane-major" : "contiguous") : "one", ip ? "ip" : "oop", (long)s0i, (long)s0o, (long)tlim);
    x = (double *)calloc(2 * ni, sizeof(double));
    y = ip ? x : (double *)calloc(2 * no, sizeof(double));
    xin = (double *)malloc(2 * ni * sizeof(double));
    a = (double *)malloc(2 * (size_t)P * sizeof(double));
    b = (double *)malloc(2 * (size_t)P * sizeof(double));
    fill(x, 2 * ni, (unsigned)(P * 7 + K * 3 + ip + lane * 11));
    memcpy(xin, x, 2 * ni * sizeof(double));
    st = mkdesc(&h, 0, r, n);
    CHECK(st == 0, "%s: create %ld", what, (long)st);
    if (st) goto out;
    DftiSetValue(h, DFTI_PLACEMENT, ip ? DFTI_INPLACE : DFTI_NOT_INPLACE);
    DftiSetValue(h, DFTI_THREAD_LIMIT, tlim);
    if (K > 1)
    {
        DftiSetValue(h, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
        DftiSetValue(h, DFTI_INPUT_DISTANCE, dist);
        DftiSetValue(h, DFTI_OUTPUT_DISTANCE, dist);
    }
    DftiSetValue(h, DFTI_INPUT_STRIDES, is);
    DftiSetValue(h, DFTI_OUTPUT_STRIDES, os);
    st = DftiCommitDescriptor(h);
    CHECK(st == 0, "%s: commit %ld", what, (long)st);
    if (st) goto out;
    st = ip ? DftiComputeForward(h, x) : DftiComputeForward(h, x, y);
    CHECK(st == 0, "%s: forward %ld", what, (long)st);
    for (t = 0; t < (size_t)K; t++)
    {   /* each transform against the naive DFT of its own input */
        double et;
        for (i = 0; i < (size_t)P; i++)
        {
            const size_t o = at(i, r, n, is, dist, t);
            a[2 * i] = xin[2 * o];
            a[2 * i + 1] = xin[2 * o + 1];
        }
        naive_nd(a, r, n, -1);
        for (i = 0; i < (size_t)P; i++)
        {
            const size_t o = at(i, r, n, os, dist, t);
            b[2 * i] = y[2 * o];
            b[2 * i + 1] = y[2 * o + 1];
        }
        et = relerr(b, a, 2 * (size_t)P);
        if (et > e) e = et;
    }
    CHECK(e < 1e-12, "%s: forward vs naive %.2e", what, e);
    /* the backward at the scale 1/P gives the input back; a scale is no change to the plans */
    p0 = h->plan[0][0];
    if (!ip && s0i != s0o)
    {   /* the backward reads through the input strides: swap them */
        DftiSetValue(h, DFTI_INPUT_STRIDES, os);
        DftiSetValue(h, DFTI_OUTPUT_STRIDES, is);
    }
    DftiSetValue(h, DFTI_BACKWARD_SCALE, 1.0 / (double)P);
    st = DftiCommitDescriptor(h);
    CHECK(st == 0 && h->plan[0][0] == p0, "%s: the recommit for a scale failed (%ld) or rebuilt the plans", what, (long)st);
    if (ip)
        st = DftiComputeBackward(h, x);
    else
    {
        memset(x, 0, 2 * ni * sizeof(double));
        st = DftiComputeBackward(h, y, x);
    }
    CHECK(st == 0, "%s: backward %ld", what, (long)st);
    for (t = 0; t < (size_t)K; t++)
        for (i = 0; i < (size_t)P; i++)
        {
            const size_t o = at(i, r, n, is, dist, t);
            const double d = fabs(x[2 * o] - xin[2 * o]) + fabs(x[2 * o + 1] - xin[2 * o + 1]);
            if (d > ert) ert = d;
        }
    CHECK(ert < 1e-12, "%s: roundtrip at the backward scale 1/P %.2e", what, ert);
    printf("C %s: fwd vs naive %.1e, roundtrip %.1e, %d thread(s)\n", what, e, ert, h->threads);
out:
    if (h)
    {
        DftiFreeDescriptor(&h);
        CHECK(h == NULL, "%s: free left the handle", what);
    }
    free(x);
    if (!ip) free(y);
    free(xin); free(a); free(b);
}

/* ── R: real ───────────────────────────────────────────────────────────── */
static void cell_r(int r, const MKL_LONG *n, int K, int lane, int ip, MKL_LONG tlim)
{
    MKL_LONG er[4], ec[4], rs[5], cs[5], P = 1, PC, H, rdist, cdist;
    DFTI_DESCRIPTOR_HANDLE h = NULL;
    size_t nr, nc, nbuf, i, t;
    double *xr, *z, *xin, *spec, *a, *b, e = 0, eb = 0;
    char shape[48], what[160];
    int k, both;
    MKL_LONG st;
    H = n[r - 1] / 2 + 1;
    for (k = 0; k < r; k++)
    {
        er[k] = ec[k] = n[k];
        P *= n[k];
    }
    ec[r - 1] = H;
    if (ip) er[r - 1] = 2 * H; /* in place: the real rows padded to the spectrum's */
    PC = P / n[r - 1] * H;
    rm(er, r, lane ? K : 1, 0, rs);
    rm(ec, r, lane ? K : 1, 0, cs);
    rdist = lane ? 1 : ip ? 2 * PC : P; /* contiguous in place: a padded plane or volume per transform */
    cdist = lane ? 1 : PC;
    nr = span(r, n, rs, rdist, (size_t)K);
    nc = span(r, ec, cs, cdist, (size_t)K);
    nbuf = ip ? (nr > 2 * nc ? nr : 2 * nc) : nr;
    shape_of(shape, sizeof shape, r, n);
    snprintf(what, sizeof what, "real %-9s K=%-2d %-10s %-3s T=%ld", shape, K,
             K > 1 ? (lane ? "lane-major" : "contiguous") : "one", ip ? "ip" : "oop", (long)tlim);
    xr = (double *)calloc(nbuf, sizeof(double));
    z = ip ? xr : (double *)calloc(2 * nc, sizeof(double));
    xin = (double *)malloc(nbuf * sizeof(double));
    spec = (double *)malloc(2 * (size_t)PC * (size_t)K * sizeof(double));
    a = (double *)malloc(2 * (size_t)P * sizeof(double));
    b = (double *)malloc(2 * (size_t)PC * sizeof(double));
    fill(xr, nbuf, (unsigned)(P * 5 + K + ip * 3 + lane * 7));
    memcpy(xin, xr, nbuf * sizeof(double));
    st = mkdesc(&h, 1, r, n);
    CHECK(st == 0, "%s: create %ld", what, (long)st);
    if (st) goto out;
    DftiSetValue(h, DFTI_PLACEMENT, ip ? DFTI_INPLACE : DFTI_NOT_INPLACE);
    DftiSetValue(h, DFTI_THREAD_LIMIT, tlim);
    DftiSetValue(h, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    if (K > 1)
    {
        DftiSetValue(h, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
        DftiSetValue(h, DFTI_INPUT_DISTANCE, rdist);
        DftiSetValue(h, DFTI_OUTPUT_DISTANCE, cdist);
    }
    DftiSetValue(h, DFTI_INPUT_STRIDES, rs);
    DftiSetValue(h, DFTI_OUTPUT_STRIDES, cs);
    st = DftiCommitDescriptor(h);
    CHECK(st == 0, "%s: commit %ld", what, (long)st);
    if (st) goto out;
    st = ip ? DftiComputeForward(h, xr) : DftiComputeForward(h, xr, z);
    CHECK(st == 0, "%s: forward %ld", what, (long)st);
    for (t = 0; t < (size_t)K; t++)
    {   /* each transform's half spectrum against the naive DFT of its reals */
        double et;
        for (i = 0; i < (size_t)P; i++)
        {
            a[2 * i] = xin[at(i, r, n, rs, rdist, t)];
            a[2 * i + 1] = 0.0;
        }
        naive_nd(a, r, n, -1);
        for (i = 0; i < (size_t)PC; i++)
        {
            const size_t f = full_of(i, r, ec, n), o = at(i, r, ec, cs, cdist, t);
            spec[2 * (t * (size_t)PC + i)] = a[2 * f];
            spec[2 * (t * (size_t)PC + i) + 1] = a[2 * f + 1];
            b[2 * i] = z[2 * o];
            b[2 * i + 1] = z[2 * o + 1];
        }
        et = relerr(b, spec + 2 * t * (size_t)PC, 2 * (size_t)PC);
        if (et > e) e = et;
    }
    CHECK(e < 1e-12, "%s: forward vs naive %.2e", what, e);
    /* the backward: rank 1 at one transform or lane-major reads the same strides both ways; elsewhere
     * the committed strides serve the forward only, so the backward is refused until they are swapped */
    both = r == 1 && (K == 1 || lane);
    if (!both)
    {
        st = ip ? DftiComputeBackward(h, xr) : DftiComputeBackward(h, z, xr);
        CHECK(st == DFTI_INCONSISTENT_CONFIGURATION, "%s: a backward on the forward's strides gave %ld, not 3", what, (long)st);
        DftiSetValue(h, DFTI_INPUT_STRIDES, cs);
        DftiSetValue(h, DFTI_OUTPUT_STRIDES, rs);
        if (K > 1)
        {
            DftiSetValue(h, DFTI_INPUT_DISTANCE, cdist);
            DftiSetValue(h, DFTI_OUTPUT_DISTANCE, rdist);
        }
        st = DftiCommitDescriptor(h);
        CHECK(st == 0, "%s: the recommit for the backward %ld", what, (long)st);
    }
    for (t = 0; t < (size_t)K; t++)
        for (i = 0; i < (size_t)PC; i++)
        {   /* the exact spectrum in, P times the reals out */
            const size_t o = at(i, r, ec, cs, cdist, t);
            z[2 * o] = spec[2 * (t * (size_t)PC + i)];
            z[2 * o + 1] = spec[2 * (t * (size_t)PC + i) + 1];
        }
    if (!ip) memset(xr, 0, nbuf * sizeof(double));
    st = ip ? DftiComputeBackward(h, xr) : DftiComputeBackward(h, z, xr);
    CHECK(st == 0, "%s: backward %ld", what, (long)st);
    {
        double m = 0;
        for (t = 0; t < (size_t)K; t++)
            for (i = 0; i < (size_t)P; i++)
            {
                const size_t o = at(i, r, n, rs, rdist, t);
                const double want = (double)P * xin[o], d = fabs(xr[o] - want);
                if (fabs(want) > m) m = fabs(want);
                if (d > eb) eb = d;
            }
        eb = m > 0 ? eb / m : eb;
    }
    CHECK(eb < 1e-12, "%s: c2r of the exact spectrum vs P x the reals %.2e", what, eb);
    printf("R %s: r2c vs naive %.1e, c2r %.1e, %s, %d thread(s)\n", what, e, eb,
           both ? "both directions from one commit" : "swapped and recommitted", h->threads);
out:
    if (h) DftiFreeDescriptor(&h);
    free(xr);
    if (!ip) free(z);
    free(xin); free(spec); free(a); free(b);
}

/* ── S: the statuses ───────────────────────────────────────────────────── */
static void statuses(void)
{
    DFTI_DESCRIPTOR_HANDLE h = NULL, h2 = NULL, hn = NULL;
    char ver[DFTI_VERSION_LENGTH], name[DFTI_MAX_NAME_LENGTH + 1];
    MKL_LONG v, n64 = 64, n8[8] = {2, 2, 2, 2, 2, 2, 2, 2}, n2[2] = {8, 64}, s2[3] = {0, 70, 1}, len[2];
    double x[128], x0[128], y[128], ref[128];
    vfft_info_t in;
    int iv, k;
    double dv;
    vfft_info(NULL, &in);
    /* a NULL handle: 5, except the VERSION query */
    memset(ver, 0, sizeof ver);
    CHECK(DftiGetValue(NULL, DFTI_VERSION, ver) == 0 && !strncmp(ver, "VectorFFT", 9), "S: the NULL-handle VERSION query (%s)", ver);
    CHECK(DftiCommitDescriptor(NULL) == DFTI_BAD_DESCRIPTOR, "S: commit(NULL) is not 5");
    CHECK(DftiComputeForward(NULL, x) == DFTI_BAD_DESCRIPTOR, "S: compute(NULL) is not 5");
    CHECK(DftiSetValue(NULL, DFTI_PLACEMENT, DFTI_INPLACE) == DFTI_BAD_DESCRIPTOR, "S: set(NULL) is not 5");
    CHECK(DftiGetValue(NULL, DFTI_PLACEMENT, &iv) == DFTI_BAD_DESCRIPTOR, "S: get(NULL) is not 5");
    CHECK(DftiFreeDescriptor(&hn) == DFTI_BAD_DESCRIPTOR, "S: free(&NULL) is not 5");
    /* create */
    CHECK(DftiCreateDescriptor(&h, DFTI_SINGLE, DFTI_COMPLEX, 1, n64) == DFTI_UNIMPLEMENTED && h == NULL, "S: single precision is not 6");
    CHECK(DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_COMPLEX, 8, n8) == DFTI_INVALID_CONFIGURATION && h == NULL, "S: rank 8 is not 2");
    CHECK(DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)0) == DFTI_INVALID_CONFIGURATION, "S: length 0 is not 2");
    CHECK(DftiCreateDescriptor(&h, DFTI_COMPLEX, DFTI_DOUBLE, 1, n64) == DFTI_INVALID_CONFIGURATION, "S: precision and domain swapped is not 2");
    /* an uncommitted compute: 2, nothing written */
    CHECK(DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_COMPLEX, 1, n64) == 0, "S: create 64");
    fill(x, 128, 5);
    memcpy(x0, x, sizeof x);
    CHECK(DftiComputeForward(h, x) == DFTI_INVALID_CONFIGURATION && !memcmp(x, x0, sizeof x), "S: an uncommitted compute is not 2, or it wrote");
    CHECK(DftiGetValue(h, DFTI_COMMIT_STATUS, &iv) == 0 && iv == DFTI_UNCOMMITTED, "S: a new descriptor is not UNCOMMITTED");
    /* the settings refused */
    CHECK(DftiSetValue(h, DFTI_COMPLEX_STORAGE, DFTI_REAL_REAL) == DFTI_UNIMPLEMENTED, "S: split COMPLEX_STORAGE is not 6");
    CHECK(DftiSetValue(h, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_REAL_REAL) == DFTI_UNIMPLEMENTED, "S: split CONJUGATE_EVEN_STORAGE is not 6");
    CHECK(DftiSetValue(h, DFTI_REAL_STORAGE, DFTI_REAL_COMPLEX) == DFTI_UNIMPLEMENTED, "S: REAL_STORAGE REAL_COMPLEX is not 6");
    CHECK(DftiSetValue(h, DFTI_PACKED_FORMAT, DFTI_CCS_FORMAT) == DFTI_UNIMPLEMENTED, "S: CCS is not 6");
    CHECK(DftiSetValue(h, DFTI_TRANSPOSE, DFTI_NONE) == DFTI_INCONSISTENT_CONFIGURATION, "S: setting TRANSPOSE is not 3");
    CHECK(DftiSetValue(h, DFTI_FWD_DISTANCE, (MKL_LONG)1) == DFTI_INCONSISTENT_CONFIGURATION, "S: FWD_DISTANCE is not 3");
    CHECK(DftiSetValue(h, DFTI_DIMENSION, (MKL_LONG)2) == DFTI_INCONSISTENT_CONFIGURATION, "S: setting DIMENSION is not 3");
    CHECK(DftiSetValue(h, (enum DFTI_CONFIG_PARAM)6, (MKL_LONG)1) == DFTI_INCONSISTENT_CONFIGURATION, "S: FORWARD_SIGN is not 3");
    CHECK(DftiSetValue(h, (enum DFTI_CONFIG_PARAM)999, (MKL_LONG)1) == DFTI_INCONSISTENT_CONFIGURATION, "S: an unknown parameter is not 3");
    CHECK(DftiSetValue(h, DFTI_THREAD_LIMIT, (MKL_LONG)-1) == DFTI_INCONSISTENT_CONFIGURATION, "S: THREAD_LIMIT -1 is not 3");
    CHECK(DftiSetValue(h, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)0) == DFTI_INCONSISTENT_CONFIGURATION, "S: NUMBER_OF_TRANSFORMS 0 is not 3");
    CHECK(DftiSetValue(h, DFTI_PLACEMENT, 99) == DFTI_INCONSISTENT_CONFIGURATION, "S: PLACEMENT 99 is not 3");
    /* the readbacks */
    CHECK(DftiSetValue(h, DFTI_DESCRIPTOR_NAME, "gate") == 0, "S: set DESCRIPTOR_NAME");
    memset(name, 0, sizeof name);
    CHECK(DftiGetValue(h, DFTI_DESCRIPTOR_NAME, name) == 0 && !strcmp(name, "gate"), "S: DESCRIPTOR_NAME reads back %s", name);
    CHECK(DftiGetValue(h, DFTI_PRECISION, &iv) == 0 && iv == DFTI_DOUBLE, "S: PRECISION");
    CHECK(DftiGetValue(h, DFTI_FORWARD_DOMAIN, &iv) == 0 && iv == DFTI_COMPLEX, "S: FORWARD_DOMAIN");
    CHECK(DftiGetValue(h, DFTI_PLACEMENT, &iv) == 0 && iv == DFTI_INPLACE, "S: the default PLACEMENT is not INPLACE");
    CHECK(DftiGetValue(h, DFTI_DIMENSION, &v) == 0 && v == 1, "S: DIMENSION");
    CHECK(DftiGetValue(h, DFTI_LENGTHS, &v) == 0 && v == 64, "S: LENGTHS");
    CHECK(DftiGetValue(h, DFTI_FORWARD_SCALE, &dv) == 0 && dv == 1.0, "S: FORWARD_SCALE");
    CHECK(DftiGetValue(h, DFTI_THREAD_LIMIT, &v) == 0 && v == 0, "S: THREAD_LIMIT before commit reads %ld", (long)v);
    /* the default THREAD_LIMIT commits plans on the machine's performance cores */
    CHECK(DftiCommitDescriptor(h) == 0, "S: commit 64 at the default THREAD_LIMIT");
    CHECK(DftiGetValue(h, DFTI_COMMIT_STATUS, &iv) == 0 && iv == DFTI_COMMITTED, "S: COMMITTED");
    CHECK(DftiGetValue(h, DFTI_THREAD_LIMIT, &v) == 0 && v == in.cores, "S: THREAD_LIMIT 0 reads back %ld after commit, not the %d cores", (long)v, in.cores);
    CHECK(DftiSetValue(h, DFTI_THREAD_LIMIT, (MKL_LONG)1) == 0 && DftiGetValue(h, DFTI_COMMIT_STATUS, &iv) == 0 && iv == DFTI_UNCOMMITTED,
          "S: a setting after commit did not uncommit");
    CHECK(DftiComputeForward(h, x) == DFTI_INVALID_CONFIGURATION, "S: a compute after a setting is not 2");
    CHECK(DftiCommitDescriptor(h) == 0 && DftiGetValue(h, DFTI_THREAD_LIMIT, &v) == 0 && v == 1, "S: THREAD_LIMIT 1 reads back %ld", (long)v);
    /* a copy: committed, its own plans, the same result after the original is freed */
    memcpy(x, x0, sizeof x);
    CHECK(DftiComputeForward(h, x) == 0, "S: compute 64");
    memcpy(ref, x, sizeof x);
    CHECK(DftiCopyDescriptor(h, &h2) == 0 && h2 && h2 != h, "S: copy");
    CHECK(DftiGetValue(h2, DFTI_COMMIT_STATUS, &iv) == 0 && iv == DFTI_COMMITTED, "S: the copy of a committed descriptor is not committed");
    CHECK(h2->plan[0][0] != h->plan[0][0], "S: the copy shares a plan");
    DftiFreeDescriptor(&h);
    memcpy(x, x0, sizeof x);
    CHECK(DftiComputeForward(h2, x) == 0 && !memcmp(x, ref, sizeof x), "S: the copy computes differently after the original is freed");
    DftiFreeDescriptor(&h2);
    CHECK(DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_COMPLEX, 1, n64) == 0 && DftiCopyDescriptor(h, &h2) == 0 &&
          DftiGetValue(h2, DFTI_COMMIT_STATUS, &iv) == 0 && iv == DFTI_UNCOMMITTED, "S: the copy of an uncommitted descriptor is not UNCOMMITTED");
    DftiFreeDescriptor(&h2);
    /* BACKWARD_SCRAMBLED: accepted, the output natural (bitwise the ORDERED descriptor's) */
    DftiSetValue(h, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(h, DFTI_THREAD_LIMIT, (MKL_LONG)1);
    CHECK(DftiCommitDescriptor(h) == 0 && DftiComputeForward(h, x0, ref) == 0, "S: ORDERED 64 out of place");
    CHECK(DftiSetValue(h, DFTI_ORDERING, DFTI_BACKWARD_SCRAMBLED) == 0 && DftiCommitDescriptor(h) == 0 &&
          DftiComputeForward(h, x0, y) == 0 && !memcmp(y, ref, sizeof y), "S: BACKWARD_SCRAMBLED is not the natural output");
    DftiFreeDescriptor(&h);
    /* DFTI's own refusals: a batch with no distance; a rank-2 real transform on the default strides */
    CHECK(DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_COMPLEX, 1, n64) == 0, "S: create");
    DftiSetValue(h, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)4);
    DftiSetValue(h, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    CHECK(DftiCommitDescriptor(h) == DFTI_INCONSISTENT_CONFIGURATION, "S: a batch with no distance is not 3");
    DftiSetValue(h, DFTI_PLACEMENT, DFTI_INPLACE);
    DftiSetValue(h, DFTI_INPUT_DISTANCE, (MKL_LONG)64);
    DftiSetValue(h, DFTI_THREAD_LIMIT, (MKL_LONG)1);
    CHECK(DftiCommitDescriptor(h) == 0, "S: an in-place complex batch with only INPUT_DISTANCE does not commit");
    DftiFreeDescriptor(&h);
    CHECK(DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_REAL, 2, n2) == 0, "S: create real 8x64");
    DftiSetValue(h, DFTI_THREAD_LIMIT, (MKL_LONG)1);
    CHECK(DftiCommitDescriptor(h) == DFTI_INCONSISTENT_CONFIGURATION, "S: in-place real 8x64 on the default strides is not 3");
    DftiSetValue(h, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    CHECK(DftiCommitDescriptor(h) == DFTI_UNIMPLEMENTED, "S: out-of-place real 8x64 on the default (padded) output strides is not 6");
    DftiFreeDescriptor(&h);
    /* a layout no plan addresses: complex rows padded to 70 */
    CHECK(DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_COMPLEX, 2, n2) == 0, "S: create 8x64");
    DftiSetValue(h, DFTI_INPUT_STRIDES, s2);
    CHECK(DftiCommitDescriptor(h) == DFTI_UNIMPLEMENTED, "S: padded complex rows are not 6");
    CHECK(DftiGetValue(h, DFTI_LENGTHS, len) == 0 && len[0] == 8 && len[1] == 64, "S: LENGTHS at rank 2");
    DftiFreeDescriptor(&h);
    /* error classes and messages */
    CHECK(DftiErrorClass(3, 3) && !DftiErrorClass(3, 2), "S: DftiErrorClass is not equality");
    for (k = 0; k <= 11; k++)
        CHECK(DftiErrorMessage(k) && strlen(DftiErrorMessage(k)) <= DFTI_MAX_MESSAGE_LENGTH, "S: message %d", k);
    printf("S statuses: NULL handle, create, uncommitted, refused settings, readbacks, copy, scrambled, DFTI's refusals, %d cores\n", in.cores);
}

/* ── M: one descriptor, eight threads ──────────────────────────────────── */
typedef struct
{
    DFTI_DESCRIPTOR_HANDLE h;
    double *x, *y;
    const double *ref;
    size_t nd;
    int reps, bad;
} mt_t;
#ifdef _WIN32
static unsigned __stdcall mt_run(void *v)
#else
static void *mt_run(void *v)
#endif
{
    mt_t *m = (mt_t *)v;
    int i;
    for (i = 0; i < m->reps; i++)
        if (DftiComputeForward(m->h, m->x, m->y) != 0 || memcmp(m->y, m->ref, m->nd * sizeof(double)))
            m->bad++;
    return 0;
}
static void shared(int nut)
{
    enum { NT = 8, N = 1024 };
    DFTI_DESCRIPTOR_HANDLE h = NULL;
    MKL_LONG n = N;
    double *x0 = (double *)malloc(2 * N * sizeof(double)), *ref = (double *)malloc(2 * N * sizeof(double));
    mt_t m[NT];
    int t, bad = 0;
    fill(x0, 2 * N, 77);
    CHECK(DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_COMPLEX, 1, n) == 0, "M: create");
    DftiSetValue(h, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(h, DFTI_THREAD_LIMIT, (MKL_LONG)1);
    DftiSetValue(h, DFTI_NUMBER_OF_USER_THREADS, (MKL_LONG)nut);
    CHECK(DftiCommitDescriptor(h) == 0 && h->nplans == nut, "M: commit with %d user threads", nut);
    CHECK(DftiComputeForward(h, x0, ref) == 0, "M: the one-thread result");
    for (t = 0; t < NT; t++)
    {
        m[t].h = h;
        m[t].x = (double *)malloc(2 * N * sizeof(double));
        m[t].y = (double *)malloc(2 * N * sizeof(double));
        memcpy(m[t].x, x0, 2 * N * sizeof(double));
        m[t].ref = ref;
        m[t].nd = 2 * N;
        m[t].reps = 200;
        m[t].bad = 0;
    }
#ifdef _WIN32
    {
        HANDLE th[NT];
        for (t = 0; t < NT; t++)
            th[t] = (HANDLE)_beginthreadex(NULL, 0, mt_run, &m[t], 0, NULL);
        WaitForMultipleObjects(NT, th, TRUE, INFINITE);
        for (t = 0; t < NT; t++)
            CloseHandle(th[t]);
    }
#else
    {
        pthread_t th[NT];
        for (t = 0; t < NT; t++)
            pthread_create(&th[t], NULL, mt_run, &m[t]);
        for (t = 0; t < NT; t++)
            pthread_join(th[t], NULL);
    }
#endif
    for (t = 0; t < NT; t++)
    {
        bad += m[t].bad;
        free(m[t].x);
        free(m[t].y);
    }
    CHECK(bad == 0 && h->busy == 0, "M: %d of %d computes from %d threads differ (NUMBER_OF_USER_THREADS %d)", bad, NT * 200, NT, nut);
    printf("M one descriptor, %d threads x 200 computes, NUMBER_OF_USER_THREADS %d: %d differ\n", NT, nut, bad);
    DftiFreeDescriptor(&h);
    free(x0);
    free(ref);
}

int main(int argc, char **argv)
{
    const char *dir = NULL;
    vfft_info_t in;
    int a;
    for (a = 1; a + 1 < argc; a++)
        if (!strcmp(argv[a], "--wisdir"))
            dir = argv[a + 1];
    if (!dir)
    {
        printf("usage: dfti_shim_gate --wisdir <cold scratch dir>\n");
        return 2;
    }
#ifdef _WIN32
    _putenv_s("VFFT_WISDOM_DIR", dir); /* the interface's plans use the library's store: point it here */
#else
    setenv("VFFT_WISDOM_DIR", dir, 1);
#endif
    vfft_info(NULL, &in);
    printf("DFTI INTERFACE GATE: VectorFFT %s, %d performance cores, store %s\n", in.version, in.cores, dir);
    CHECK(in.cores >= 1, "vfft_info: %d cores", in.cores);
    {   /* C: complex */
        const MKL_LONG n64[1] = {64}, n1000[1] = {1000}, n7[1] = {7}, n137[2] = {13, 7}, n5137[3] = {5, 13, 7},
                       n32[1] = {32}, n48[1] = {48}, n168[2] = {16, 8}, n888[3] = {8, 8, 8}, n1210[2] = {12, 10};
        cell_c(1, n64, 1, 0, 1, 0, 0, 1);      /* DFTI's defaults: in place */
        cell_c(1, n1000, 1, 0, 0, 3, 5, 1);    /* out of place, offsets */
        cell_c(1, n7, 1, 0, 1, 0, 0, 1);       /* a prime */
        cell_c(2, n137, 1, 0, 1, 0, 0, 0);     /* rank 2, the default THREAD_LIMIT */
        cell_c(3, n5137, 1, 0, 1, 0, 0, 1);    /* rank 3 */
        cell_c(1, n32, 5, 0, 1, 0, 0, 1);      /* a contiguous batch in place */
        cell_c(1, n48, 6, 0, 0, 0, 0, 1);      /* a contiguous batch out of place */
        cell_c(1, n64, 8, 1, 0, 0, 0, 1);      /* the lane-major batch */
        cell_c(1, n64, 8, 1, 1, 0, 0, 1);
        cell_c(2, n168, 4, 1, 0, 0, 0, 1);     /* the 2D lane-major batch */
        cell_c(3, n888, 4, 1, 0, 0, 0, 1);     /* the 3D lane-major batch */
        cell_c(2, n1210, 3, 0, 0, 0, 0, 1);    /* a 2D contiguous batch */
        cell_c(3, n888, 2, 0, 0, 0, 0, 1);     /* a 3D contiguous batch: the volume queue */
    }
    {   /* R: real */
        const MKL_LONG n64[1] = {64}, n63[1] = {63}, n6[1] = {6}, n16[1] = {16}, n54[2] = {5, 4}, n13x[3] = {13, 5, 4},
                       n816[2] = {8, 16}, n468[3] = {4, 6, 8};
        cell_r(1, n64, 1, 0, 0, 1);
        cell_r(1, n63, 1, 0, 0, 1);            /* odd */
        cell_r(1, n64, 1, 0, 1, 1);            /* in place on DFTI's default strides */
        cell_r(1, n6, 1, 0, 0, 1);
        cell_r(2, n54, 1, 0, 0, 1);            /* swap and recommit for the backward */
        cell_r(3, n13x, 1, 0, 1, 0);           /* in place, padded rows, the default THREAD_LIMIT */
        cell_r(2, n816, 1, 0, 1, 1);
        cell_r(1, n16, 3, 0, 0, 1);            /* contiguous batches */
        cell_r(1, n16, 3, 0, 1, 1);
        cell_r(1, n64, 8, 1, 0, 1);            /* the lane-major batch */
        cell_r(1, n64, 8, 1, 1, 1);
        cell_r(2, n816, 4, 1, 0, 1);           /* the 2D lane-major batch */
        cell_r(2, n816, 4, 1, 1, 1);
        cell_r(2, n816, 4, 0, 1, 1);           /* a 2D contiguous batch in place: K padded planes (the plane queue) */
        cell_r(3, n468, 3, 0, 0, 1);           /* 3D contiguous batches (the volume queue) */
        cell_r(3, n468, 3, 0, 1, 1);           /*   in place: K padded volumes */
        cell_r(3, n468, 3, 1, 0, 1);           /* the 3D lane-major batch */
        cell_r(3, n468, 3, 1, 1, 1);           /*   in place: N1 N2 rows of (N3 + 2) K doubles */
    }
    statuses();
    shared(8);
    shared(1);
    printf("%d checks, %d failed\n%s\n", g_checks, g_fail, g_fail ? "*** DFTI INTERFACE GATE FAILED ***" : "DFTI INTERFACE GATE ALL PASS");
    return g_fail ? 1 : 0;
}
