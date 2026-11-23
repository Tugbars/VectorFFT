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
 *      result bitwise the one-thread result; THREAD_LIMIT 0 reads back vfft_info().cores;
 *   T  the Trigonometric Transforms (mkl_trig_transforms.h): all six -- the sine, cosine, staggered sine,
 *      staggered cosine, staggered2 cosine and staggered2 sine
 *      transforms at every n from 2 to 40 and at larger ones, both directions, MKL's scales and the
 *      FFTW mode (ipar[10] = 1), judged on every entry of f against the definitions summed in long
 *      double; the round trip; init's defaults; the statuses (n below 2, a type out of range, the
 *      staggered cosine's boundary value in both modes and the staggered2 pair's lack of one, ipar[6] not 0,
 *      the boundary warning and its tolerance, ipar[7] = 0, ipar[0]
 *      or ipar[5] changed after the commit, a NULL or foreign handle both ways, single precision); one
 *      handle computed from eight threads at once, every result bitwise the one-thread result.
 * The interface plans every cell with the estimate (VFFT_FLAG_ESTIMATE): nothing races and the store named by
 * VFFT_WISDOM_DIR is not opened. Usage: dfti_shim_gate --wisdir <dir> */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../shims/dfti/vfft_dfti.c" /* the interface in this unit: the gate reads its plans */
#include "../../shims/dfti/vfft_tt.c"   /* the Trigonometric Transforms, likewise */
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

/* ── T: the Trigonometric Transforms ───────────────────────────────────── */
#define TT_PI 3.14159265358979323846264338327950288L
static const char *tt_name(int t)
{
    static const char *const nm[6] = { "sine", "cosine", "staggered cosine", "staggered sine", "staggered2 cosine",
                                       "staggered2 sine" };
    return nm[t];
}
/* S[m] = sin(pi m / 2n), m < 4n: every angle the sine, cosine and staggered transforms take */
static long double *tt_table(int n)
{
    long double *S = (long double *)malloc((size_t)4 * (size_t)n * sizeof(long double));
    int m;
    for (m = 0; m < 4 * n; m++)
        S[m] = sinl(TT_PI * m / (2.0L * n));
    return S;
}
/* Q[m] = sin(pi m / 4n), m < 8n: the angles of the staggered2 pair */
static long double *tt_table2(int n)
{
    long double *Q = (long double *)malloc((size_t)8 * (size_t)n * sizeof(long double));
    int m;
    for (m = 0; m < 8 * n; m++)
        Q[m] = sinl(TT_PI * m / (4.0L * n));
    return Q;
}
/* the transform of g into F (n + 1 entries) by its definition: MKL's (the reference's formulas), or with
 * fftw FFTW's (ipar[10] = 1); the entries outside the transform's range keep g's values */
static void tt_ref(int t, int fwd, int fftw, int n, const double *g, long double *F, const long double *S)
{
    const long long P = 4LL * n;
    int k, i;
    for (k = 0; k <= n; k++)
        F[k] = g[k];
    if (t == MKL_STAGGERED_COSINE_TRANSFORM)
        for (k = 0; k < n; k++)
        {   /* forward f(0) / n + 2/n sum_{i=1}^{n-1} f(i) cos((2k+1) i pi / 2n), k = 0 .. n-1; backward
             * sum_{i=0}^{n-1} f(i) cos((2i+1) k pi / 2n); FFTW's REDFT01 forward (n times MKL's) and REDFT10
             * backward (twice MKL's). cos(pi q / 2n) = S[q + n] */
            long double s = 0.0L;
            if (fwd)
            {
                for (i = 1; i < n; i++)
                    s += g[i] * S[((2LL * k + 1) * i + n) % P];
                F[k] = fftw ? (long double)g[0] + 2.0L * s : ((long double)g[0] + 2.0L * s) / n;
            }
            else
            {
                for (i = 0; i < n; i++)
                    s += g[i] * S[((2LL * i + 1) * k + n) % P];
                F[k] = fftw ? 2.0L * s : s;
            }
        }
    else if (t == MKL_STAGGERED2_COSINE_TRANSFORM || t == MKL_STAGGERED2_SINE_TRANSFORM)
    {   /* 2/n sum_{i=0}^{n-1} f(i) cos or sin((2k+1)(2i+1) pi / 4n) forward, the sum backward; FFTW's REDFT11
         * and RODFT11: twice the sum. cos(pi q / 4n) = Q[q + 2n] */
        long double *Q = tt_table2(n);
        const long long P2 = 8LL * n, sh = t == MKL_STAGGERED2_COSINE_TRANSFORM ? 2LL * n : 0;
        for (k = 0; k < n; k++)
        {
            long double s = 0.0L;
            for (i = 0; i < n; i++)
                s += g[i] * Q[((2LL * k + 1) * (2LL * i + 1) + sh) % P2];
            F[k] = fftw ? 2.0L * s : fwd ? 2.0L * s / n : s;
        }
        free(Q);
    }
    else if (t == MKL_SINE_TRANSFORM)
        for (k = 1; k <= n - 1; k++)
        {   /* 2/n sum_{i=1}^{n-1} f(i) sin(k i pi / n) forward, the sum backward; FFTW's RODFT00: twice the sum */
            long double s = 0.0L;
            for (i = 1; i <= n - 1; i++)
                s += g[i] * S[(2LL * k * i) % P];
            F[k] = fftw ? 2.0L * s : fwd ? 2.0L * s / n : s;
        }
    else if (t == MKL_COSINE_TRANSFORM)
        for (k = 0; k <= n; k++)
        {   /* e = f(0) + f(n) cos k pi: (e + 2 sum) / n forward, e / 2 + sum backward; FFTW's REDFT00: e + 2 sum */
            const long double e = (long double)g[0] + ((k & 1) ? -(long double)g[n] : (long double)g[n]);
            long double s = 0.0L;
            for (i = 1; i <= n - 1; i++)
                s += g[i] * S[(2LL * k * i + n) % P];
            F[k] = fftw ? e + 2.0L * s : fwd ? (e + 2.0L * s) / n : 0.5L * e + s;
        }
    else if (!fftw)
        for (k = 1; k <= n; k++)
        {   /* forward sin((2k-1) pi/2) f(n) / n + 2/n sum_{i=1}^{n-1} f(i) sin((2k-1) i pi / 2n);
             * backward sum_{i=1}^{n} f(i) sin((2i-1) k pi / 2n) */
            long double s = 0.0L;
            if (fwd)
            {
                for (i = 1; i <= n - 1; i++)
                    s += g[i] * S[((2LL * k - 1) * i) % P];
                F[k] = ((k & 1) ? (long double)g[n] : -(long double)g[n]) / n + 2.0L * s / n;
            }
            else
            {
                for (i = 1; i <= n; i++)
                    s += g[i] * S[((2LL * i - 1) * k) % P];
                F[k] = s;
            }
        }
    else
        for (k = 0; k < n; k++)
        {   /* FFTW's RODFT01 (forward) and RODFT10 (backward) of f[0..n-1] */
            long double s = 0.0L;
            if (fwd)
            {
                s = (k & 1) ? -(long double)g[n - 1] : (long double)g[n - 1];
                for (i = 0; i < n - 1; i++)
                    s += 2.0L * g[i] * S[((long long)(i + 1) * (2 * k + 1)) % P];
            }
            else
                for (i = 0; i < n; i++)
                    s += 2.0L * g[i] * S[((long long)(2 * i + 1) * (k + 1)) % P];
            F[k] = s;
        }
}
static double tt_worst[6];
static int tt_cells[6];
static void tt_cell(int t, int n)
{
    const size_t m = (size_t)n + 1;
    MKL_INT nn = n, tt = t, ipar[128], st = 0;
    DFTI_DESCRIPTOR_HANDLE h = NULL;
    double *dpar = (double *)malloc(((size_t)5 * (size_t)n / 2 + 2) * sizeof(double));
    double *f = (double *)malloc(m * sizeof(double)), *g = (double *)malloc(m * sizeof(double));
    double *g2 = (double *)malloc(m * sizeof(double));
    long double *F = (long double *)malloc(m * sizeof(long double)), *S = tt_table(n);
    int fftw, fwd;
    size_t k;
    fill(g, m, 31u * (unsigned)n + (unsigned)t);
    if (t == MKL_SINE_TRANSFORM || t == MKL_STAGGERED_SINE_TRANSFORM)
        g[0] = 0.0; /* the reference's 0.0 */
    if (t == MKL_SINE_TRANSFORM || t == MKL_STAGGERED_COSINE_TRANSFORM)
        g[n] = 0.0; /* the staggered2 pair: f[n] is past the array's n values, here a guard that must stay */
    memcpy(g2, g, m * sizeof(double));
    g2[0] = 0.3; /* the FFTW mode of the staggered sine transform reads f[0] */
    memset(ipar, 0x5a, sizeof ipar);
    d_init_trig_transform(&nn, &tt, ipar, dpar, &st);
    CHECK(st == 0 && ipar[0] == n && ipar[1] == 1 && ipar[2] == 1 && ipar[5] == t && ipar[6] == 0 && ipar[7] == 1 &&
          ipar[8] == 1 && ipar[9] == 1 && ipar[10] == 0 && dpar[0] == 1.0e-12 * n,
          "T %s n=%d: init's defaults (stat %lld, dpar[0] %g)", tt_name(t), n, (long long)st, dpar[0]);
    memcpy(f, g, m * sizeof(double));
    d_commit_trig_transform(f, &h, ipar, dpar, &st);
    CHECK(st == 0 && h != NULL && !memcmp(f, g, m * sizeof(double)), "T %s n=%d: commit stat %lld (f untouched)",
          tt_name(t), n, (long long)st);
    if (!h)
        goto done;
    for (fftw = 0; fftw < 2; fftw++)
        for (fwd = 1; fwd >= 0; fwd--)
        {
            const double *in = (fftw && t == MKL_STAGGERED_SINE_TRANSFORM) ? g2 : g;
            double e = 0.0, mx = 0.0;
            ipar[10] = fftw;
            memcpy(f, in, m * sizeof(double));
            tt_ref(t, fwd, fftw, n, in, F, S);
            if (fwd)
                d_forward_trig_transform(f, &h, ipar, dpar, &st);
            else
                d_backward_trig_transform(f, &h, ipar, dpar, &st);
            for (k = 0; k < m; k++)
            {
                const double d = fabs((double)((long double)f[k] - F[k]));
                if (d > e)
                    e = d;
                if (fabs((double)F[k]) > mx)
                    mx = fabs((double)F[k]);
            }
            e = mx > 0.0 ? e / mx : e;
            CHECK(st == 0 && e <= 1e-13, "T %s n=%d %s%s: stat %lld, error %.2e", tt_name(t), n,
                  fwd ? "forward" : "backward", fftw ? " (FFTW mode)" : "", (long long)st, e);
            if (e > tt_worst[t])
                tt_worst[t] = e;
        }
    {   /* MKL's pair is an inverse */
        double e = 0.0, mx = 0.0;
        ipar[10] = 0;
        memcpy(f, g, m * sizeof(double));
        d_forward_trig_transform(f, &h, ipar, dpar, &st);
        d_backward_trig_transform(f, &h, ipar, dpar, &st);
        for (k = 0; k < m; k++)
        {
            if (fabs(f[k] - g[k]) > e)
                e = fabs(f[k] - g[k]);
            if (fabs(g[k]) > mx)
                mx = fabs(g[k]);
        }
        CHECK(st == 0 && e <= 1e-13 * mx, "T %s n=%d: backward of forward off by %.2e", tt_name(t), n, mx > 0 ? e / mx : e);
    }
    free_trig_transform(&h, ipar, &st);
    CHECK(st == 0 && h == NULL, "T %s n=%d: free stat %lld", tt_name(t), n, (long long)st);
done:
    tt_cells[t]++;
    free(dpar);
    free(f);
    free(g);
    free(g2);
    free(F);
    free(S);
}
/* one routine's status on a fresh init of (n, type) and a commit on f (n + 1 values, the reference's
 * zeros in place); the handle is left to the caller */
static MKL_INT tt_commit(int n, int t, MKL_INT *ipar, double *dpar, double *f, DFTI_DESCRIPTOR_HANDLE *h)
{
    MKL_INT nn = n, tt = t, st;
    d_init_trig_transform(&nn, &tt, ipar, dpar, &st);
    d_commit_trig_transform(f, h, ipar, dpar, &st);
    return st;
}
static void tt_statuses(void)
{
    MKL_INT ipar[128], st, nn, tt;
    double dpar[5 * 16 / 2 + 2], f[17], fs[17];
    float spar[4];
    DFTI_DESCRIPTOR_HANDLE h = NULL, d = NULL;
    int t;
    fill(f, 17, 5u);
    f[0] = 0.0;
    f[8] = 0.0;
    nn = 8, tt = MKL_SINE_TRANSFORM;
    d_init_trig_transform(&nn, &tt, NULL, dpar, &st);
    CHECK(st == -99999, "T: init without ipar %lld", (long long)st);
    /* n below 2 and a type out of range: init takes them, commit refuses (-100) */
    CHECK(tt_commit(1, MKL_SINE_TRANSFORM, ipar, dpar, f, &h) == -100 && ipar[6] == -100 && h == NULL, "T: n = 1");
    CHECK(tt_commit(0, MKL_COSINE_TRANSFORM, ipar, dpar, f, &h) == -100 && h == NULL, "T: n = 0");
    CHECK(tt_commit(8, 7, ipar, dpar, f, &h) == -100 && h == NULL, "T: type 7");
    for (t = 0; t <= 5; t++)
        if (t == MKL_STAGGERED_COSINE_TRANSFORM || t == MKL_STAGGERED2_COSINE_TRANSFORM || t == MKL_STAGGERED2_SINE_TRANSFORM)
        {   /* f[n] = 1: the staggered cosine's commit warns (1) with MKL's scales and not in the FFTW mode; the
             * staggered2 pair has no boundary value (MKL 2025.3) */
            int fftw;
            for (fftw = 0; fftw < 2; fftw++)
            {
                const MKL_INT want = (t == MKL_STAGGERED_COSINE_TRANSFORM && !fftw) ? 1 : 0;
                memcpy(fs, f, sizeof f);
                fs[0] = 0.5;
                fs[8] = 1.0;
                nn = 8, tt = t;
                d_init_trig_transform(&nn, &tt, ipar, dpar, &st);
                ipar[2] = 0;
                ipar[10] = fftw;
                d_commit_trig_transform(fs, &h, ipar, dpar, &st);
                CHECK(st == want && h != NULL, "T: type %d with f[n] = 1%s at commit %lld", t, fftw ? " (FFTW mode)" : "",
                      (long long)st);
                if (h)
                    free_trig_transform(&h, ipar, &st);
            }
        }
    /* ipar[6] not 0 at the commit */
    nn = 8, tt = MKL_COSINE_TRANSFORM;
    d_init_trig_transform(&nn, &tt, ipar, dpar, &st);
    ipar[6] = 1;
    d_commit_trig_transform(f, &h, ipar, dpar, &st);
    CHECK(st == -10000 && h == NULL, "T: commit with ipar[6] = 1 %lld", (long long)st);
    /* the boundary warning, its tolerance dpar[0] (8e-12 at n = 8), and the forward after it */
    memcpy(fs, f, sizeof f);
    fs[0] = 0.5e-12 * 8;
    CHECK(tt_commit(8, MKL_SINE_TRANSFORM, ipar, dpar, fs, &h) == 0 && h, "T: f[0] within the tolerance");
    free_trig_transform(&h, ipar, &st);
    fs[0] = 2e-12 * 8;
    CHECK(tt_commit(8, MKL_SINE_TRANSFORM, ipar, dpar, fs, &h) == 1 && ipar[6] == 1 && h, "T: f[0] above the tolerance");
    d_forward_trig_transform(fs, &h, ipar, dpar, &st);
    CHECK(st == -10000, "T: a forward after a commit that warned %lld", (long long)st);
    ipar[6] = 0;
    d_forward_trig_transform(fs, &h, ipar, dpar, &st);
    CHECK(st == 0, "T: the forward once ipar[6] = 0 %lld", (long long)st);
    /* ipar[0] or ipar[5] changed after the commit */
    ipar[0] = 9;
    d_forward_trig_transform(fs, &h, ipar, dpar, &st);
    CHECK(st == -10000, "T: ipar[0] changed %lld", (long long)st);
    ipar[0] = 8, ipar[6] = 0, ipar[5] = MKL_COSINE_TRANSFORM;
    d_backward_trig_transform(fs, &h, ipar, dpar, &st);
    CHECK(st == -100, "T: ipar[5] changed %lld", (long long)st);
    ipar[5] = MKL_SINE_TRANSFORM, ipar[6] = 0;
    /* the handle the other way: DFTI's routines refuse it, and the TT routines a DFTI descriptor */
    CHECK(DftiComputeForward(h, fs) == DFTI_BAD_DESCRIPTOR, "T: DftiComputeForward on a TT handle");
    CHECK(DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG)8) == 0, "T: a DFTI descriptor");
    d_forward_trig_transform(fs, &d, ipar, dpar, &st);
    CHECK(st == -1000, "T: d_forward_trig_transform on a DFTI descriptor %lld", (long long)st);
    ipar[6] = 0;
    free_trig_transform(&d, ipar, &st);
    CHECK(st == -1000 && d != NULL, "T: free_trig_transform on a DFTI descriptor %lld", (long long)st);
    DftiFreeDescriptor(&d);
    free_trig_transform(&h, ipar, &st);
    CHECK(st == 0 && h == NULL, "T: free %lld", (long long)st);
    free_trig_transform(&h, ipar, &st);
    CHECK(st == -1000, "T: free of a NULL handle %lld", (long long)st);
    /* the staggered sine transform in the FFTW mode reads f[0]: no warning */
    memcpy(fs, f, sizeof f);
    fs[0] = 0.5;
    nn = 8, tt = MKL_STAGGERED_SINE_TRANSFORM;
    d_init_trig_transform(&nn, &tt, ipar, dpar, &st);
    ipar[10] = 1;
    d_commit_trig_transform(fs, &h, ipar, dpar, &st);
    CHECK(st == 0 && h, "T: the FFTW-mode staggered sine with f[0] = 0.5 %lld", (long long)st);
    free_trig_transform(&h, ipar, &st);
    /* ipar[7] = 0: commit checks and creates no handle */
    nn = 8, tt = MKL_STAGGERED_SINE_TRANSFORM;
    d_init_trig_transform(&nn, &tt, ipar, dpar, &st);
    ipar[7] = 0;
    d_commit_trig_transform(f, &h, ipar, dpar, &st);
    CHECK(st == 0 && h == NULL, "T: commit with ipar[7] = 0 %lld", (long long)st);
    d_forward_trig_transform(f, &h, ipar, dpar, &st);
    CHECK(st == -1000, "T: a forward with no handle %lld", (long long)st);
    /* single precision */
    nn = 8, tt = MKL_SINE_TRANSFORM;
    s_init_trig_transform(&nn, &tt, ipar, spar, &st);
    CHECK(st == -99999 && ipar[6] == -99999, "T: s_init_trig_transform %lld", (long long)st);
    s_commit_trig_transform((float *)f, &h, ipar, spar, &st);
    CHECK(st == -10000 && h == NULL, "T: s_commit_trig_transform %lld", (long long)st);
    s_forward_trig_transform((float *)f, &h, ipar, spar, &st);
    CHECK(st == -10000, "T: s_forward_trig_transform %lld", (long long)st);
    s_backward_trig_transform((float *)f, &h, ipar, spar, &st);
    CHECK(st == -10000, "T: s_backward_trig_transform %lld", (long long)st);
    printf("T statuses checked\n");
}
/* one handle, eight threads: each thread its own f and ipar */
typedef struct
{
    DFTI_DESCRIPTOR_HANDLE *h;
    double *f, *dpar;
    const double *in, *ref;
    MKL_INT ipar[128];
    size_t m;
    int reps, bad;
} tt_mt_t;
#ifdef _WIN32
static unsigned __stdcall tt_mt_run(void *v)
#else
static void *tt_mt_run(void *v)
#endif
{
    tt_mt_t *c = (tt_mt_t *)v;
    MKL_INT st;
    int i;
    for (i = 0; i < c->reps; i++)
    {
        memcpy(c->f, c->in, c->m * sizeof(double));
        d_forward_trig_transform(c->f, c->h, c->ipar, c->dpar, &st);
        if (st != 0 || memcmp(c->f, c->ref, c->m * sizeof(double)))
            c->bad++;
    }
    return 0;
}
static void tt_shared(void)
{
    enum { NT = 8, N = 4096 };
    const size_t m = N + 1;
    MKL_INT nn = N, tt = MKL_STAGGERED_SINE_TRANSFORM, ipar[128], st;
    DFTI_DESCRIPTOR_HANDLE h = NULL;
    double *dpar = (double *)malloc((5 * N / 2 + 2) * sizeof(double));
    double *in = (double *)malloc(m * sizeof(double)), *ref = (double *)malloc(m * sizeof(double));
    static tt_mt_t c[NT];
    int t, bad = 0;
    fill(in, m, 99u);
    in[0] = 0.0;
    d_init_trig_transform(&nn, &tt, ipar, dpar, &st);
    d_commit_trig_transform(in, &h, ipar, dpar, &st);
    CHECK(st == 0 && h, "T M: commit %lld", (long long)st);
    if (!h)
        return;
    memcpy(ref, in, m * sizeof(double));
    d_forward_trig_transform(ref, &h, ipar, dpar, &st);
    for (t = 0; t < NT; t++)
    {
        c[t].h = &h;
        c[t].f = (double *)malloc(m * sizeof(double));
        c[t].dpar = dpar;
        c[t].in = in;
        c[t].ref = ref;
        memcpy(c[t].ipar, ipar, sizeof ipar);
        c[t].m = m;
        c[t].reps = 100;
        c[t].bad = 0;
    }
#ifdef _WIN32
    {
        HANDLE th[NT];
        for (t = 0; t < NT; t++)
            th[t] = (HANDLE)_beginthreadex(NULL, 0, tt_mt_run, &c[t], 0, NULL);
        WaitForMultipleObjects(NT, th, TRUE, INFINITE);
        for (t = 0; t < NT; t++)
            CloseHandle(th[t]);
    }
#else
    {
        pthread_t th[NT];
        for (t = 0; t < NT; t++)
            pthread_create(&th[t], NULL, tt_mt_run, &c[t]);
        for (t = 0; t < NT; t++)
            pthread_join(th[t], NULL);
    }
#endif
    for (t = 0; t < NT; t++)
    {
        bad += c[t].bad;
        free(c[t].f);
    }
    CHECK(bad == 0 && ((struct vt_handle *)(void *)h)->busy == 0, "T M: %d of %d forwards from %d threads differ", bad,
          NT * 100, NT);
    printf("T M one handle, %d threads x 100 forwards: %d differ\n", NT, bad);
    free_trig_transform(&h, ipar, &st);
    free(dpar);
    free(in);
    free(ref);
}
static void trig(void)
{
    static const int big[] = {63, 64, 65, 100, 127, 128, 255, 256, 257, 1000, 1024, 1031, 4096, 4099};
    static const int types[6] = {MKL_SINE_TRANSFORM, MKL_COSINE_TRANSFORM, MKL_STAGGERED_SINE_TRANSFORM,
                                 MKL_STAGGERED_COSINE_TRANSFORM, MKL_STAGGERED2_COSINE_TRANSFORM,
                                 MKL_STAGGERED2_SINE_TRANSFORM};
    int a, n, b;
    for (a = 0; a < 6; a++)
    {
        for (n = 2; n <= 40; n++)
            tt_cell(types[a], n);
        for (b = 0; b < (int)(sizeof big / sizeof big[0]); b++)
            tt_cell(types[a], big[b]);
        printf("T %s transform: %d lengths, both directions, both scalings: worst error %.2e\n", tt_name(types[a]),
               tt_cells[types[a]], tt_worst[types[a]]);
    }
    tt_statuses();
    tt_shared();
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
    trig();
    printf("%d checks, %d failed\n%s\n", g_checks, g_fail, g_fail ? "*** DFTI INTERFACE GATE FAILED ***" : "DFTI INTERFACE GATE ALL PASS");
    return g_fail ? 1 : 0;
}
