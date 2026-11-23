/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Tuğbars Heptaşkın */
/*
 * vfft_tt.c -- MKL's Trigonometric Transform interface on VectorFFT (mkl_trig_transforms.h).
 *
 * init writes ipar and dpar[0]; commit checks them and creates the handle: one VectorFFT plan of the
 * transform, planned by VectorFFT's estimate (VFFT_FLAG_ESTIMATE: nothing is measured and no wisdom is
 * read or written) on the machine's performance cores; forward and backward run it in place on the
 * part of f the transform reads, then apply MKL's scale as their own pass over those values:
 *
 *   MKL's transform       VectorFFT's plan    on         forward        backward
 *   sine (n)              VFFT_DST1 of n - 1  f[1..n-1]  DST-I / n      DST-I / 2
 *   cosine (n)            VFFT_DCT1 of n + 1  f[0..n]    DCT-I / n      DCT-I / 2
 *   staggered sine (n)    VFFT_DST3 of n      f[1..n]    DST-III / n    DST-II / 2
 *   staggered cosine (n)  VFFT_DCT3 of n      f[0..n-1]  DCT-III / n    DCT-II / 2
 *   staggered2 cosine (n) VFFT_DCT4 of n      f[0..n-1]  DCT-IV / n     DCT-IV / 2
 *   staggered2 sine (n)   VFFT_DST4 of n      f[0..n-1]  DST-IV / n     DST-IV / 2
 *
 * MKL's FFTW compatibility (ipar[10] = 1) skips the scale: FFTW's RODFT00, REDFT00, for the staggered
 * sine RODFT01 (forward) and RODFT10 (backward) on f[0..n-1], for the staggered cosine REDFT01 and
 * REDFT10, for the staggered2 pair REDFT11 and RODFT11. The values outside the range a transform reads
 * are not written. The statuses and the checks are MKL's: init takes any n and type, commit refuses
 * what it cannot serve, a routine refuses to run after a call that did not end with ipar[6] = 0.
 * Includes vfft.h and nothing from src/core.
 */
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sched.h>
#endif
#include "vfft.h"
#include "mkl_trig_transforms.h"

#define VT_MAGIC 0x56545454 /* "VTTT": a live handle (a DFTI descriptor carries another number) */

/* MKL's statuses */
#define VT_OK 0
#define VT_WARN 1
#define VT_EDATA (-100)
#define VT_EFFT (-1000)
#define VT_EINIT (-10000)
#define VT_EFAIL (-99999)

struct vt_handle
{
    int magic;
    int type;     /* MKL's transform type, 0 to 5 */
    MKL_INT n;    /* the transform's n */
    int len;      /* the plan's values */
    vfft_plan plan;
    int busy;     /* a forward or backward is running (atomic) */
};

static void vt_yield(void)
{
#if defined(_WIN32)
    SwitchToThread();
#else
    sched_yield();
#endif
}

/* a printed line, as ipar[1] (errors) or ipar[2] (warnings) asks: 0 none, -1 the file
 * MKL_Trig_Transforms_log.txt in the current folder, otherwise the screen (stderr); each distinct
 * line once per process */
static void vt_say(const MKL_INT *ipar, int warning, const char *fmt, ...)
{
    static unsigned long long seen[256];
    static int lock;
    char line[768];
    unsigned long long h = 1469598103934665603ull;
    const MKL_INT where = ipar ? ipar[warning ? 2 : 1] : 1;
    const char *p;
    int i, fresh = 1;
    va_list ap;
    if (where == 0)
        return;
    va_start(ap, fmt);
    vsnprintf(line, sizeof line, fmt, ap);
    va_end(ap);
    for (p = line; *p; p++)
        h = (h ^ (unsigned char)*p) * 1099511628211ull;
    if (!h)
        h = 1;
    while (__atomic_exchange_n(&lock, 1, __ATOMIC_ACQUIRE))
        vt_yield();
    for (i = 0; i < 256 && seen[i]; i++)
        if (seen[i] == h)
        {
            fresh = 0;
            break;
        }
    if (fresh && i < 256)
        seen[i] = h;
    __atomic_store_n(&lock, 0, __ATOMIC_RELEASE);
    if (!fresh)
        return;
    if (where == -1)
    {
        FILE *log = fopen("MKL_Trig_Transforms_log.txt", "a");
        if (log)
        {
            fprintf(log, "vfft tt: %s\n", line);
            fclose(log);
            return;
        }
        fprintf(stdout, "vfft tt: the file MKL_Trig_Transforms_log.txt cannot be created\n");
    }
    fprintf(stderr, "vfft tt: %s\n", line);
}

/* the status, also in ipar[6] */
static void vt_done(MKL_INT *ipar, MKL_INT *stat, MKL_INT s)
{
    if (ipar)
        ipar[6] = s;
    if (stat)
        *stat = s;
}

/* the plan a served type takes: its kind and its length n + dn; 0 for a type not served */
static int vt_form(MKL_INT type, vfft_transform_t *kind, int *dn)
{
    switch (type)
    {
    case MKL_SINE_TRANSFORM:
        *kind = VFFT_DST1, *dn = -1;
        return 1;
    case MKL_COSINE_TRANSFORM:
        *kind = VFFT_DCT1, *dn = 1;
        return 1;
    case MKL_STAGGERED_SINE_TRANSFORM:
        *kind = VFFT_DST3, *dn = 0;
        return 1;
    case MKL_STAGGERED_COSINE_TRANSFORM:
        *kind = VFFT_DCT3, *dn = 0;
        return 1;
    case MKL_STAGGERED2_COSINE_TRANSFORM:
        *kind = VFFT_DCT4, *dn = 0;
        return 1;
    case MKL_STAGGERED2_SINE_TRANSFORM:
        *kind = VFFT_DST4, *dn = 0;
        return 1;
    default:
        return 0;
    }
}

/* where the plan's values start in f */
static int vt_off(int type, const MKL_INT *ipar)
{
    if (type == MKL_COSINE_TRANSFORM || type == MKL_STAGGERED_COSINE_TRANSFORM ||
        type == MKL_STAGGERED2_COSINE_TRANSFORM || type == MKL_STAGGERED2_SINE_TRANSFORM)
        return 0;
    if (type == MKL_STAGGERED_SINE_TRANSFORM && ipar[10] == 1)
        return 0; /* the FFTW mode: RODFT01 / RODFT10 of f[0..n-1] */
    return 1;
}

static struct vt_handle *vt_live(DFTI_DESCRIPTOR_HANDLE *handle)
{
    struct vt_handle *h;
    if (!handle || !*handle)
        return NULL;
    h = (struct vt_handle *)(void *)*handle;
    return h->magic == VT_MAGIC ? h : NULL;
}

/* ── init ──────────────────────────────────────────────────────────────── */

void vfft_tt_d_init(MKL_INT *n, MKL_INT *tt_type, MKL_INT ipar[], double dpar[], MKL_INT *stat)
{
    if (!n || !tt_type || !ipar || !dpar)
    {
        vt_done(ipar, stat, VT_EFAIL);
        return;
    }
    ipar[0] = *n;
    ipar[1] = 1; /* errors on the screen */
    ipar[2] = 1; /* warnings on the screen */
    ipar[5] = *tt_type;
    ipar[7] = 1; /* commit creates the handle */
    ipar[8] = 1; /* C-style messages */
    ipar[9] = 1;
    ipar[10] = 0; /* MKL's scales */
    dpar[0] = 1.0e-12 * (double)*n; /* the boundary values' tolerance, MKL's default */
    vt_done(ipar, stat, VT_OK);
}

/* ── commit ────────────────────────────────────────────────────────────── */

void vfft_tt_d_commit(double f[], DFTI_DESCRIPTOR_HANDLE *handle, MKL_INT ipar[], double dpar[], MKL_INT *stat)
{
    const char *who = "d_commit_trig_transform";
    vfft_transform_t kind;
    MKL_INT n, type, s = VT_OK;
    int dn;
    if (!ipar || !handle || !f || !dpar)
    {
        vt_say(ipar, 0, "%s: f, handle, ipar and dpar must be given", who);
        vt_done(ipar, stat, VT_EDATA);
        return;
    }
    n = ipar[0];
    type = ipar[5];
    if (ipar[6] != 0)
    {
        vt_say(ipar, 0, "%s: ipar[6] = %lld: d_init_trig_transform did not complete, or ipar[6] was changed outside "
                        "the routines; set ipar[6] = 0 to go on", who, (long long)ipar[6]);
        vt_done(ipar, stat, VT_EINIT);
        return;
    }
    if (n < 2)
    {
        vt_say(ipar, 0, "%s: n = %lld; the size of the transform must be 2 or more", who, (long long)n);
        vt_done(ipar, stat, VT_EDATA);
        return;
    }
    if (type < 0 || type > 5)
    {
        vt_say(ipar, 0, "%s: the transform type %lld is not one of 0 to 5", who, (long long)type);
        vt_done(ipar, stat, VT_EDATA);
        return;
    }
    if (!vt_form(type, &kind, &dn))
    {   /* every type of 0 to 5 has a plan: not reached */
        vt_say(ipar, 0, "%s: the transform type %lld is not served", who, (long long)type);
        vt_done(ipar, stat, VT_EDATA);
        return;
    }
    if (n > (MKL_INT)INT_MAX - 1)
    {
        vt_say(ipar, 0, "%s: n = %lld is not served", who, (long long)n);
        vt_done(ipar, stat, VT_EDATA);
        return;
    }
    /* the values the reference asks to be 0.0: f[0] and f[n] (sine), f[0] (staggered sine, but in
     * the FFTW mode, where f[0] is data), f[n] (staggered cosine, but in the FFTW mode: MKL 2025.3
     * warns in the one and not in the other); the staggered2 pair has no such value */
    {
        const int c0 = type == MKL_SINE_TRANSFORM || (type == MKL_STAGGERED_SINE_TRANSFORM && ipar[10] != 1);
        const int cn = type == MKL_SINE_TRANSFORM || (type == MKL_STAGGERED_COSINE_TRANSFORM && ipar[10] != 1);
        const double tol = dpar[0];
        const char *name = type == MKL_SINE_TRANSFORM ? "the sine transform"
                           : type == MKL_STAGGERED_COSINE_TRANSFORM ? "the staggered cosine transform"
                                                                    : "the staggered sine transform";
        if (c0 && !(f[0] >= -tol && f[0] <= tol))
        {
            vt_say(ipar, 1, "%s: |f[0]| = %g is above the tolerance dpar[0] = %g; %s expects 0.0 there", who, f[0],
                   tol, name);
            s = VT_WARN;
        }
        if (cn && !(f[n] >= -tol && f[n] <= tol))
        {
            vt_say(ipar, 1, "%s: |f[n]| = %g is above the tolerance dpar[0] = %g; %s expects 0.0 there", who, f[n],
                   tol, name);
            s = VT_WARN;
        }
    }
    if (ipar[7] != 0)
    {   /* the handle: one plan, planned by the estimate */
        struct vt_handle *h = (struct vt_handle *)calloc(1, sizeof *h);
        vfft_config_t c;
        vfft_info_t in;
        if (!h)
        {
            vt_say(ipar, 0, "%s: out of memory", who);
            vt_done(ipar, stat, VT_EFFT);
            return;
        }
        vfft_info(NULL, &in);
        memset(&c, 0, sizeof c);
        c.transform = kind;
        c.rank = 1;
        c.n[0] = (int)(n + dn);
        c.howmany = 1;
        c.layout = VFFT_LAYOUT_INTERLEAVED;
        c.placement = VFFT_PLACEMENT_IN;
        c.order = VFFT_ORDER_NATURAL;
        c.nthreads = in.cores > 0 ? in.cores : 1;
        c.flags = (unsigned)VFFT_FLAG_ESTIMATE;
        c.wisdom_dir = NULL; /* the estimate opens no store */
        h->plan = vfft_create(&c);
        if (!h->plan)
        {
            const char *e = vfft_error();
            vt_say(ipar, 0, "%s: %s", who, e && *e ? e : "the plan was refused");
            free(h);
            vt_done(ipar, stat, VT_EFFT);
            return;
        }
        h->magic = VT_MAGIC;
        h->type = (int)type;
        h->n = n;
        h->len = (int)(n + dn);
        *handle = (DFTI_DESCRIPTOR_HANDLE)(void *)h;
    }
    vt_done(ipar, stat, s);
}

/* ── forward, backward ─────────────────────────────────────────────────── */

static void vt_run(double f[], DFTI_DESCRIPTOR_HANDLE *handle, MKL_INT ipar[], double dpar[], MKL_INT *stat,
                   int backward)
{
    const char *who = backward ? "d_backward_trig_transform" : "d_forward_trig_transform";
    struct vt_handle *h;
    double *x;
    int rc, k;
    if (!ipar || !f || !dpar)
    {
        vt_say(ipar, 0, "%s: f, ipar and dpar must be given", who);
        vt_done(ipar, stat, VT_EDATA);
        return;
    }
    h = vt_live(handle);
    if (!h)
    {
        vt_say(ipar, 0, "%s: the handle is not one d_commit_trig_transform created, or it was freed", who);
        vt_done(ipar, stat, VT_EFFT);
        return;
    }
    if (ipar[6] != 0)
    {
        vt_say(ipar, 0, "%s: ipar[6] = %lld: the previous call did not end with 0, or ipar[6] was changed outside "
                        "the routines; set ipar[6] = 0 to go on", who, (long long)ipar[6]);
        vt_done(ipar, stat, VT_EINIT);
        return;
    }
    if (ipar[0] != h->n)
    {
        vt_say(ipar, 0, "%s: ipar[0] = %lld, the handle was committed for n = %lld", who, (long long)ipar[0],
               (long long)h->n);
        vt_done(ipar, stat, VT_EINIT);
        return;
    }
    if (ipar[5] != h->type)
    {
        vt_say(ipar, 0, "%s: ipar[5] = %lld, the handle was committed for the type %d", who, (long long)ipar[5],
               h->type);
        vt_done(ipar, stat, VT_EDATA);
        return;
    }
    x = f + vt_off(h->type, ipar);
    while (__atomic_exchange_n(&h->busy, 1, __ATOMIC_ACQUIRE))
        vt_yield(); /* one plan: computes on one handle take turns */
    rc = vfft_execute(h->plan, backward ? VFFT_BACKWARD : VFFT_FORWARD, x, x);
    __atomic_store_n(&h->busy, 0, __ATOMIC_RELEASE);
    if (rc != 0)
    {
        const char *e = vfft_error();
        vt_say(ipar, 0, "%s: %s", who, e && *e ? e : "the plan refused the call");
        vt_done(ipar, stat, VT_EFFT);
        return;
    }
    if (ipar[10] != 1)
    {   /* MKL's scale: forward 1/n, backward 1/2 */
        const double sc = backward ? 0.5 : 1.0 / (double)h->n;
        for (k = 0; k < h->len; k++)
            x[k] *= sc;
    }
    vt_done(ipar, stat, VT_OK);
}

void vfft_tt_d_forward(double f[], DFTI_DESCRIPTOR_HANDLE *handle, MKL_INT ipar[], double dpar[], MKL_INT *stat)
{
    vt_run(f, handle, ipar, dpar, stat, 0);
}

void vfft_tt_d_backward(double f[], DFTI_DESCRIPTOR_HANDLE *handle, MKL_INT ipar[], double dpar[], MKL_INT *stat)
{
    vt_run(f, handle, ipar, dpar, stat, 1);
}

/* ── free ──────────────────────────────────────────────────────────────── */

void vfft_tt_free(DFTI_DESCRIPTOR_HANDLE *handle, MKL_INT ipar[], MKL_INT *stat)
{
    struct vt_handle *h = vt_live(handle);
    if (!h)
    {
        vt_say(ipar, 0, "free_trig_transform: the handle is not one d_commit_trig_transform created, or it was freed");
        vt_done(ipar, stat, VT_EFFT);
        return;
    }
    vfft_destroy(h->plan);
    h->magic = 0;
    free(h);
    *handle = NULL;
    vt_done(ipar, stat, VT_OK);
}

/* ── single precision: refused ─────────────────────────────────────────── */

static void vt_single(MKL_INT ipar[], MKL_INT *stat, const char *who, MKL_INT s)
{
    vt_say(ipar, 0, "%s: single precision is not served; VectorFFT computes the trigonometric transforms in double "
                    "precision (the d_ routines)", who);
    vt_done(ipar, stat, s);
}

void vfft_tt_s_init(MKL_INT *n, MKL_INT *tt_type, MKL_INT ipar[], float spar[], MKL_INT *stat)
{
    (void)n, (void)tt_type, (void)spar;
    vt_single(ipar, stat, "s_init_trig_transform", VT_EFAIL);
}

void vfft_tt_s_commit(float f[], DFTI_DESCRIPTOR_HANDLE *handle, MKL_INT ipar[], float spar[], MKL_INT *stat)
{
    (void)f, (void)handle, (void)spar;
    vt_single(ipar, stat, "s_commit_trig_transform", VT_EINIT);
}

void vfft_tt_s_forward(float f[], DFTI_DESCRIPTOR_HANDLE *handle, MKL_INT ipar[], float spar[], MKL_INT *stat)
{
    (void)f, (void)handle, (void)spar;
    vt_single(ipar, stat, "s_forward_trig_transform", VT_EINIT);
}

void vfft_tt_s_backward(float f[], DFTI_DESCRIPTOR_HANDLE *handle, MKL_INT ipar[], float spar[], MKL_INT *stat)
{
    (void)f, (void)handle, (void)spar;
    vt_single(ipar, stat, "s_backward_trig_transform", VT_EINIT);
}
