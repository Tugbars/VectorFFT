/*
 * vfft_dfti.c -- the DFTI interface of VectorFFT (docs/roadmap/dfti_shim_design.md).
 *
 * A descriptor holds DFTI's settings. Its commit recognises the layout its strides and
 * distances describe, translates the descriptor into one vfft_config_t per direction the
 * committed strides serve, and creates the plans, one per declared user thread. A compute
 * borrows a plan and runs it, then applies the direction's scale. Nothing here selects an
 * engine, converts a layout or reorders a spectrum: a recognised layout goes to vfft_create,
 * whose refusal is the commit's. Includes vfft.h and nothing from src/core.
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
#include "mkl_dfti.h"

#define VD_MAGIC 0x56444654 /* "VDFT": a live descriptor */
#define VD_MAXRANK 7        /* DFTI's ranks; the commit serves vfft.h's 1 to 4 */
#define VD_MAXPLANS 64      /* plans per direction: the width of the busy mask */

struct DFTI_DESCRIPTOR
{
    int magic;
    int domain; /* DFTI_COMPLEX or DFTI_REAL */
    int rank;
    MKL_LONG n[VD_MAXRANK];
    /* the settings */
    double scale[2]; /* forward, backward */
    MKL_LONG howmany;
    int complex_storage, real_storage, conj_storage, packed_format;
    int placement, workspace, ordering, destroy_input;
    MKL_LONG istride[VD_MAXRANK + 1], ostride[VD_MAXRANK + 1];
    MKL_LONG idist, odist;
    MKL_LONG user_threads, thread_limit;
    char name[DFTI_MAX_NAME_LENGTH + 1];
    /* the commit */
    int committed;
    int fits[2];           /* the directions the committed strides serve */
    vfft_config_t cfg[2];  /* their configurations; complex data: cfg[0] serves both */
    int nplans;            /* plans per direction, one per declared user thread */
    vfft_plan *plan[2];    /* plan[direction][slot]; complex data: plan[1] == plan[0] */
    size_t ioff[2], ooff[2];   /* the first elements, in doubles from the caller's pointers */
    size_t out_doubles[2];     /* what each direction writes: the extent of its scale */
    int threads;               /* the plans' thread count */
    unsigned long long busy;   /* the slots computing now (atomic) */
    int waited;                /* a compute has waited for a slot (atomic) */
};

static int vd_live(const struct DFTI_DESCRIPTOR *d)
{
    return d && d->magic == VD_MAGIC;
}

static void vd_yield(void)
{
#if defined(_WIN32)
    SwitchToThread();
#else
    sched_yield();
#endif
}

/* a printed line: stderr, each distinct line once per process */
static void vd_say(const char *fmt, ...)
{
    static unsigned long long seen[256];
    static int lock;
    char line[768];
    unsigned long long h = 1469598103934665603ull;
    const char *p;
    int i, fresh = 1;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(line, sizeof line, fmt, ap);
    va_end(ap);
    for (p = line; *p; p++)
        h = (h ^ (unsigned char)*p) * 1099511628211ull;
    if (!h)
        h = 1;
    while (__atomic_exchange_n(&lock, 1, __ATOMIC_ACQUIRE))
        vd_yield();
    for (i = 0; i < 256 && seen[i]; i++)
        if (seen[i] == h)
        {
            fresh = 0;
            break;
        }
    if (fresh && i < 256)
        seen[i] = h;
    __atomic_store_n(&lock, 0, __ATOMIC_RELEASE);
    if (fresh)
        fprintf(stderr, "vfft dfti: %s\n", line);
}

/* row-major strides over the extents e[0..r-1] with element stride m: s[r] = m, s[k] = s[k + 1] e[k] */
static void vd_rowmajor(const MKL_LONG *e, int r, MKL_LONG m, MKL_LONG *s)
{
    int k;
    s[0] = 0;
    s[r] = m;
    for (k = r - 1; k >= 1; k--)
        s[k] = s[k + 1] * e[k];
}

/* ── create, copy, free ───────────────────────────────────────────────────── */

static MKL_LONG vd_create(DFTI_DESCRIPTOR_HANDLE *desc, int precision, int domain, MKL_LONG rank,
                          const MKL_LONG *lengths, MKL_LONG len1)
{
    struct DFTI_DESCRIPTOR *d;
    int k;
    if (!desc)
        return DFTI_BAD_DESCRIPTOR;
    *desc = NULL;
    if (rank < 1 || rank > VD_MAXRANK || (domain != DFTI_COMPLEX && domain != DFTI_REAL) ||
        (precision != DFTI_DOUBLE && precision != DFTI_SINGLE) || (rank > 1 && !lengths))
        return DFTI_INVALID_CONFIGURATION;
    for (k = 0; k < rank; k++)
        if ((lengths ? lengths[k] : len1) <= 0)
            return DFTI_INVALID_CONFIGURATION;
    if (precision == DFTI_SINGLE)
    {
        vd_say("DftiCreateDescriptor: single precision is not served (VectorFFT computes in double precision)");
        return DFTI_UNIMPLEMENTED;
    }
    d = (struct DFTI_DESCRIPTOR *)calloc(1, sizeof *d);
    if (!d)
        return DFTI_MEMORY_ERROR;
    d->magic = VD_MAGIC;
    d->domain = domain;
    d->rank = (int)rank;
    for (k = 0; k < rank; k++)
        d->n[k] = lengths ? lengths[k] : len1;
    d->scale[0] = d->scale[1] = 1.0;
    d->howmany = 1;
    d->complex_storage = DFTI_COMPLEX_COMPLEX;
    d->real_storage = DFTI_REAL_REAL;
    d->conj_storage = DFTI_COMPLEX_COMPLEX;
    d->packed_format = DFTI_CCE_FORMAT;
    d->placement = DFTI_INPLACE;
    d->workspace = DFTI_ALLOW;
    d->ordering = DFTI_ORDERED;
    d->destroy_input = DFTI_AVOID;
    vd_rowmajor(d->n, d->rank, 1, d->istride); /* DFTI's defaults: tight in the forward domain, both sides */
    vd_rowmajor(d->n, d->rank, 1, d->ostride);
    d->user_threads = 1;
    *desc = d;
    return DFTI_NO_ERROR;
}

MKL_LONG vfft_dfti_create(DFTI_DESCRIPTOR_HANDLE *desc, enum DFTI_CONFIG_VALUE precision,
                          enum DFTI_CONFIG_VALUE domain, MKL_LONG dimension, ...)
{
    va_list ap;
    MKL_LONG one = 0;
    const MKL_LONG *many = NULL;
    va_start(ap, dimension);
    if (dimension == 1)
        one = va_arg(ap, MKL_LONG);
    else
        many = va_arg(ap, const MKL_LONG *);
    va_end(ap);
    return vd_create(desc, (int)precision, (int)domain, dimension, many, one);
}
MKL_LONG vfft_dfti_create_d_1d(DFTI_DESCRIPTOR_HANDLE *desc, enum DFTI_CONFIG_VALUE domain, ...)
{
    va_list ap;
    MKL_LONG n;
    va_start(ap, domain);
    n = va_arg(ap, MKL_LONG);
    va_end(ap);
    return vd_create(desc, DFTI_DOUBLE, (int)domain, 1, NULL, n);
}
MKL_LONG vfft_dfti_create_d_md(DFTI_DESCRIPTOR_HANDLE *desc, enum DFTI_CONFIG_VALUE domain, MKL_LONG dimension, ...)
{
    va_list ap;
    const MKL_LONG *n;
    va_start(ap, dimension);
    n = va_arg(ap, const MKL_LONG *);
    va_end(ap);
    return vd_create(desc, DFTI_DOUBLE, (int)domain, dimension, n, 0);
}
MKL_LONG vfft_dfti_create_s_1d(DFTI_DESCRIPTOR_HANDLE *desc, enum DFTI_CONFIG_VALUE domain, ...)
{
    va_list ap;
    MKL_LONG n;
    va_start(ap, domain);
    n = va_arg(ap, MKL_LONG);
    va_end(ap);
    return vd_create(desc, DFTI_SINGLE, (int)domain, 1, NULL, n);
}
MKL_LONG vfft_dfti_create_s_md(DFTI_DESCRIPTOR_HANDLE *desc, enum DFTI_CONFIG_VALUE domain, MKL_LONG dimension, ...)
{
    va_list ap;
    const MKL_LONG *n;
    va_start(ap, dimension);
    n = va_arg(ap, const MKL_LONG *);
    va_end(ap);
    return vd_create(desc, DFTI_SINGLE, (int)domain, dimension, n, 0);
}

static void vd_drop_plans(struct DFTI_DESCRIPTOR *d)
{
    int k, s;
    if (d->plan[1] == d->plan[0])
        d->plan[1] = NULL; /* complex data: one set serves both directions */
    for (k = 0; k < 2; k++)
    {
        if (!d->plan[k])
            continue;
        for (s = 0; s < d->nplans; s++)
            vfft_destroy(d->plan[k][s]);
        free(d->plan[k]);
        d->plan[k] = NULL;
    }
    d->nplans = 0;
    d->fits[0] = d->fits[1] = 0;
    d->committed = 0;
}

MKL_LONG vfft_dfti_free(DFTI_DESCRIPTOR_HANDLE *desc)
{
    struct DFTI_DESCRIPTOR *d;
    if (!desc || !vd_live(*desc))
        return DFTI_BAD_DESCRIPTOR;
    d = *desc;
    vd_drop_plans(d);
    d->magic = 0;
    free(d);
    *desc = NULL;
    return DFTI_NO_ERROR;
}

/* ── set and get ─────────────────────────────────────────────────────────── */

static MKL_LONG vd_split(const char *param)
{
    vd_say("DftiSetValue: %s = DFTI_REAL_REAL (split storage) is not served by this interface; VectorFFT serves split data through vfft.h", param);
    return DFTI_UNIMPLEMENTED;
}

MKL_LONG vfft_dfti_set_value(DFTI_DESCRIPTOR_HANDLE desc, enum DFTI_CONFIG_PARAM param, ...)
{
    struct DFTI_DESCRIPTOR *d = desc;
    MKL_LONG st = DFTI_NO_ERROR;
    va_list ap;
    if (!vd_live(d))
        return DFTI_BAD_DESCRIPTOR;
    va_start(ap, param);
    switch ((int)param)
    {
    case DFTI_FORWARD_SCALE:
        d->scale[0] = va_arg(ap, double);
        break;
    case DFTI_BACKWARD_SCALE:
        d->scale[1] = va_arg(ap, double);
        break;
    case DFTI_NUMBER_OF_TRANSFORMS:
    {
        const MKL_LONG v = va_arg(ap, MKL_LONG);
        if (v < 1)
            st = DFTI_INCONSISTENT_CONFIGURATION;
        else
            d->howmany = v;
        break;
    }
    case DFTI_COMPLEX_STORAGE:
    {
        const int v = va_arg(ap, int);
        if (v == DFTI_COMPLEX_COMPLEX)
            d->complex_storage = v;
        else
            st = v == DFTI_REAL_REAL ? vd_split("DFTI_COMPLEX_STORAGE") : DFTI_INCONSISTENT_CONFIGURATION;
        break;
    }
    case DFTI_REAL_STORAGE:
    {
        const int v = va_arg(ap, int);
        if (v == DFTI_REAL_REAL)
            d->real_storage = v;
        else
            st = v == DFTI_REAL_COMPLEX ? DFTI_UNIMPLEMENTED : DFTI_INCONSISTENT_CONFIGURATION;
        break;
    }
    case DFTI_CONJUGATE_EVEN_STORAGE:
    {
        const int v = va_arg(ap, int);
        if (v == DFTI_COMPLEX_COMPLEX)
            d->conj_storage = v;
        else if (v == DFTI_COMPLEX_REAL && d->rank == 1)
        {
            vd_say("DftiSetValue: DFTI_CONJUGATE_EVEN_STORAGE = DFTI_COMPLEX_REAL (the packed real formats) is not served; use DFTI_COMPLEX_COMPLEX");
            st = DFTI_UNIMPLEMENTED;
        }
        else
            st = v == DFTI_REAL_REAL ? vd_split("DFTI_CONJUGATE_EVEN_STORAGE") : DFTI_INCONSISTENT_CONFIGURATION;
        break;
    }
    case DFTI_PACKED_FORMAT:
    {
        const int v = va_arg(ap, int);
        if (v == DFTI_CCE_FORMAT)
            d->packed_format = v;
        else if (v == DFTI_CCS_FORMAT || v == DFTI_PACK_FORMAT || v == DFTI_PERM_FORMAT)
        {
            vd_say("DftiSetValue: DFTI_PACKED_FORMAT CCS, PACK and PERM are not served; use DFTI_CCE_FORMAT");
            st = DFTI_UNIMPLEMENTED;
        }
        else
            st = DFTI_INCONSISTENT_CONFIGURATION;
        break;
    }
    case DFTI_PLACEMENT:
    {
        const int v = va_arg(ap, int);
        if (v == DFTI_INPLACE || v == DFTI_NOT_INPLACE)
            d->placement = v;
        else
            st = DFTI_INCONSISTENT_CONFIGURATION;
        break;
    }
    case DFTI_INPUT_STRIDES:
    case DFTI_OUTPUT_STRIDES:
    {
        const MKL_LONG *v = va_arg(ap, const MKL_LONG *);
        if (!v)
            st = DFTI_INCONSISTENT_CONFIGURATION;
        else
            memcpy((int)param == DFTI_INPUT_STRIDES ? d->istride : d->ostride, v, sizeof(MKL_LONG) * (size_t)(d->rank + 1));
        break;
    }
    case DFTI_INPUT_DISTANCE:
        d->idist = va_arg(ap, MKL_LONG);
        break;
    case DFTI_OUTPUT_DISTANCE:
        d->odist = va_arg(ap, MKL_LONG);
        break;
    case DFTI_WORKSPACE:
    {
        const int v = va_arg(ap, int);
        if (v == DFTI_ALLOW || v == DFTI_AVOID)
            d->workspace = v;
        else
            st = DFTI_INCONSISTENT_CONFIGURATION;
        break;
    }
    case DFTI_ORDERING:
    {
        const int v = va_arg(ap, int);
        if (v == DFTI_ORDERED || v == DFTI_BACKWARD_SCRAMBLED)
            d->ordering = v;
        else
            st = DFTI_INCONSISTENT_CONFIGURATION;
        break;
    }
    case DFTI_DESTROY_INPUT:
    {
        const int v = va_arg(ap, int);
        if (v == DFTI_ALLOW || v == DFTI_AVOID)
            d->destroy_input = v;
        else
            st = DFTI_INCONSISTENT_CONFIGURATION;
        break;
    }
    case DFTI_NUMBER_OF_USER_THREADS:
    {
        const MKL_LONG v = va_arg(ap, MKL_LONG);
        if (v < 1)
            st = DFTI_INCONSISTENT_CONFIGURATION;
        else
            d->user_threads = v;
        break;
    }
    case DFTI_THREAD_LIMIT:
    {
        const MKL_LONG v = va_arg(ap, MKL_LONG);
        if (v < 0)
            st = DFTI_INCONSISTENT_CONFIGURATION;
        else
            d->thread_limit = v;
        break;
    }
    case DFTI_DESCRIPTOR_NAME:
    {
        const char *v = va_arg(ap, const char *);
        if (!v)
            st = DFTI_INCONSISTENT_CONFIGURATION;
        else
        {
            memset(d->name, 0, sizeof d->name);
            strncpy(d->name, v, DFTI_MAX_NAME_LENGTH);
        }
        break;
    }
    default: /* fixed at create, read only, not implemented by DFTI, or unknown */
        st = DFTI_INCONSISTENT_CONFIGURATION;
        break;
    }
    va_end(ap);
    if (st == DFTI_NO_ERROR)
        d->committed = 0; /* any setting uncommits, as in DFTI; the plans stay until the next commit */
    return st;
}

static void vd_version(char *buf)
{
    vfft_info_t in;
    vfft_info(NULL, &in);
    snprintf(buf, DFTI_VERSION_LENGTH, "VectorFFT %s, DFTI interface: double precision, interleaved", in.version);
}

MKL_LONG vfft_dfti_get_value(DFTI_DESCRIPTOR_HANDLE desc, enum DFTI_CONFIG_PARAM param, ...)
{
    struct DFTI_DESCRIPTOR *d = desc;
    MKL_LONG st = DFTI_NO_ERROR;
    va_list ap;
    int k;
    va_start(ap, param);
    if ((int)param == DFTI_VERSION)
    {   /* the one query a NULL handle answers */
        char *v = va_arg(ap, char *);
        va_end(ap);
        if (!v)
            return DFTI_INCONSISTENT_CONFIGURATION;
        vd_version(v);
        return DFTI_NO_ERROR;
    }
    if (!vd_live(d))
    {
        va_end(ap);
        return DFTI_BAD_DESCRIPTOR;
    }
    switch ((int)param)
    {
    case DFTI_FORWARD_DOMAIN:
        *va_arg(ap, int *) = d->domain;
        break;
    case DFTI_DIMENSION:
        *va_arg(ap, MKL_LONG *) = d->rank;
        break;
    case DFTI_LENGTHS:
    {
        MKL_LONG *v = va_arg(ap, MKL_LONG *);
        for (k = 0; k < d->rank; k++)
            v[k] = d->n[k];
        break;
    }
    case DFTI_PRECISION:
        *va_arg(ap, int *) = DFTI_DOUBLE;
        break;
    case DFTI_FORWARD_SCALE:
        *va_arg(ap, double *) = d->scale[0];
        break;
    case DFTI_BACKWARD_SCALE:
        *va_arg(ap, double *) = d->scale[1];
        break;
    case DFTI_NUMBER_OF_TRANSFORMS:
        *va_arg(ap, MKL_LONG *) = d->howmany;
        break;
    case DFTI_COMPLEX_STORAGE:
        *va_arg(ap, int *) = d->complex_storage;
        break;
    case DFTI_REAL_STORAGE:
        *va_arg(ap, int *) = d->real_storage;
        break;
    case DFTI_CONJUGATE_EVEN_STORAGE:
        *va_arg(ap, int *) = d->conj_storage;
        break;
    case DFTI_PLACEMENT:
        *va_arg(ap, int *) = d->placement;
        break;
    case DFTI_INPUT_STRIDES:
    case DFTI_OUTPUT_STRIDES:
    {
        MKL_LONG *v = va_arg(ap, MKL_LONG *);
        memcpy(v, (int)param == DFTI_INPUT_STRIDES ? d->istride : d->ostride, sizeof(MKL_LONG) * (size_t)(d->rank + 1));
        break;
    }
    case DFTI_INPUT_DISTANCE:
        *va_arg(ap, MKL_LONG *) = d->idist;
        break;
    case DFTI_OUTPUT_DISTANCE:
        *va_arg(ap, MKL_LONG *) = d->odist;
        break;
    case DFTI_WORKSPACE:
        *va_arg(ap, int *) = d->workspace;
        break;
    case DFTI_ORDERING:
        *va_arg(ap, int *) = d->ordering;
        break;
    case DFTI_TRANSPOSE:
        *va_arg(ap, int *) = DFTI_NONE;
        break;
    case DFTI_DESCRIPTOR_NAME:
        memcpy(va_arg(ap, char *), d->name, DFTI_MAX_NAME_LENGTH);
        break;
    case DFTI_PACKED_FORMAT:
        *va_arg(ap, int *) = d->packed_format;
        break;
    case DFTI_COMMIT_STATUS:
        *va_arg(ap, int *) = d->committed ? DFTI_COMMITTED : DFTI_UNCOMMITTED;
        break;
    case DFTI_NUMBER_OF_USER_THREADS:
        *va_arg(ap, MKL_LONG *) = d->user_threads;
        break;
    case DFTI_THREAD_LIMIT: /* the set value until a commit, the plans' thread count after */
        *va_arg(ap, MKL_LONG *) = d->committed ? (MKL_LONG)d->threads : d->thread_limit;
        break;
    case DFTI_DESTROY_INPUT:
        *va_arg(ap, int *) = d->destroy_input;
        break;
    default:
        st = DFTI_INCONSISTENT_CONFIGURATION;
        break;
    }
    va_end(ap);
    return st;
}

/* ── commit ──────────────────────────────────────────────────────────────── */

/* one layout the commit recognises: the strides and the distance of each side (the complex
 * domain reads side r for its input and its output), whether the first elements sit at
 * offset 0, and the batch geometry it is */
typedef struct
{
    MKL_LONG r[VD_MAXRANK + 1], c[VD_MAXRANK + 1];
    MKL_LONG rdist, cdist; /* -1: not read */
    int s0zero;
    vfft_batch_t batch;
} vd_form_t;

static int vd_side(const MKL_LONG *s, MKL_LONG dist, const MKL_LONG *want, MKL_LONG wdist, int r, int s0zero)
{
    int k;
    for (k = 1; k <= r; k++)
        if (s[k] != want[k])
            return 0;
    if (s[0] < 0 || (s0zero && s[0] != 0))
        return 0;
    return wdist < 0 || dist == wdist;
}

/* complex data: one transform, contiguous transforms, the lane-major batch */
static int vd_complex_forms(const struct DFTI_DESCRIPTOR *d, vd_form_t *f)
{
    const int r = d->rank;
    const MKL_LONG K = d->howmany;
    MKL_LONG P = 1;
    int k, m = 0;
    for (k = 0; k < r; k++)
        P *= d->n[k];
    memset(f, 0, 2 * sizeof *f);
    vd_rowmajor(d->n, r, 1, f[m].r);
    f[m].rdist = K > 1 ? P : -1;
    f[m].batch = K > 1 ? VFFT_BATCH_CONTIGUOUS : VFFT_BATCH_DEFAULT;
    m++;
    if (K > 1)
    {
        vd_rowmajor(d->n, r, K, f[m].r);
        f[m].rdist = 1;
        f[m].batch = VFFT_BATCH_LANE_MAJOR;
        m++;
    }
    return m;
}

/* real data: the reals on side r, the half spectrum (H = n_last / 2 + 1 bins a row) on side c */
static int vd_real_forms(const struct DFTI_DESCRIPTOR *d, vd_form_t *f)
{
    const int r = d->rank, ip = d->placement == DFTI_INPLACE;
    const MKL_LONG K = d->howmany, H = d->n[r - 1] / 2 + 1;
    MKL_LONG ec[VD_MAXRANK], ep[VD_MAXRANK], P = 1;
    int k, m = 0;
    for (k = 0; k < r; k++)
    {
        ec[k] = ep[k] = d->n[k];
        P *= d->n[k];
    }
    ec[r - 1] = H;     /* the half spectrum's rows */
    ep[r - 1] = 2 * H; /* in place: the real rows padded to the spectrum's */
    memset(f, 0, 2 * sizeof *f);
    if (!ip)
    {   /* tight reals, a packed spectrum; contiguous transforms; the lane-major batch at ranks 1 and 2 */
        vd_rowmajor(d->n, r, 1, f[m].r);
        vd_rowmajor(ec, r, 1, f[m].c);
        f[m].rdist = K > 1 ? P : -1;
        f[m].cdist = K > 1 ? P / d->n[r - 1] * H : -1;
        f[m].batch = K > 1 ? VFFT_BATCH_CONTIGUOUS : VFFT_BATCH_DEFAULT;
        m++;
        if (K > 1 && r <= 2)
        {
            vd_rowmajor(d->n, r, K, f[m].r);
            vd_rowmajor(ec, r, K, f[m].c);
            f[m].rdist = f[m].cdist = 1;
            f[m].batch = VFFT_BATCH_LANE_MAJOR;
            m++;
        }
    }
    else if (K == 1 || r == 1)
    {   /* in place: one transform at any rank (padded rows above rank 1), or a rank-1 batch */
        vd_rowmajor(ep, r, 1, f[m].r);
        vd_rowmajor(ec, r, 1, f[m].c);
        f[m].rdist = K > 1 ? 2 * H : -1;
        f[m].cdist = K > 1 ? H : -1;
        f[m].s0zero = 1;
        f[m].batch = K > 1 ? VFFT_BATCH_CONTIGUOUS : VFFT_BATCH_DEFAULT;
        m++;
        if (K > 1)
        {
            vd_rowmajor(d->n, 1, K, f[m].r);
            vd_rowmajor(ec, 1, K, f[m].c);
            f[m].rdist = f[m].cdist = 1;
            f[m].s0zero = 1;
            f[m].batch = VFFT_BATCH_LANE_MAJOR;
            m++;
        }
    }
    else if (r == 2)
    {   /* in place, the rank-2 lane-major batch: N1 blocks of (N2 + 2) K doubles */
        vd_rowmajor(ep, 2, K, f[m].r);
        vd_rowmajor(ec, 2, K, f[m].c);
        f[m].rdist = f[m].cdist = 1;
        f[m].s0zero = 1;
        f[m].batch = VFFT_BATCH_LANE_MAJOR;
        m++;
    }
    return m;
}

/* the plans' thread count: the limit set; else one when the program declared its own threads,
 * else the machine's performance cores */
static int vd_threads(const struct DFTI_DESCRIPTOR *d)
{
    vfft_info_t in;
    if (d->thread_limit > 0)
        return d->thread_limit > 1024 ? 1024 : (int)d->thread_limit;
    if (d->user_threads > 1)
        return 1;
    vfft_info(NULL, &in);
    return in.cores > 0 ? in.cores : 1;
}

static void vd_config(const struct DFTI_DESCRIPTOR *d, vfft_transform_t tr, vfft_batch_t batch, int threads,
                      vfft_config_t *c)
{
    int k;
    memset(c, 0, sizeof *c);
    c->transform = tr;
    c->rank = d->rank;
    for (k = 0; k < d->rank; k++)
        c->n[k] = (int)d->n[k];
    c->howmany = (size_t)d->howmany;
    c->layout = VFFT_LAYOUT_INTERLEAVED;
    c->placement = d->placement == DFTI_INPLACE ? VFFT_PLACEMENT_IN : VFFT_PLACEMENT_OUT;
    c->order = VFFT_ORDER_NATURAL;
    c->batch = batch;
    c->nthreads = threads;
    c->flags = d->destroy_input == DFTI_ALLOW ? (unsigned)VFFT_FLAG_DESTROY_INPUT : 0u;
    c->wisdom_dir = NULL;
}

MKL_LONG vfft_dfti_commit(DFTI_DESCRIPTOR_HANDLE desc)
{
    struct DFTI_DESCRIPTOR *d = desc;
    vd_form_t f[2], use[2];
    vfft_config_t cfg[2];
    MKL_LONG def[VD_MAXRANK + 1];
    int fits[2] = {0, 0}, real, ip, r, nf, k, s, nplans;
    if (!vd_live(d))
        return DFTI_BAD_DESCRIPTOR;
    if (d->committed)
        return DFTI_NO_ERROR;
    real = d->domain == DFTI_REAL;
    ip = d->placement == DFTI_INPLACE;
    r = d->rank;
    if (r > 4)
    {
        vd_say("DftiCommitDescriptor: rank %d is not served (VectorFFT serves ranks 1 to 4)", r);
        return DFTI_UNIMPLEMENTED;
    }
    for (k = 0; k < r; k++)
        if (d->n[k] > INT_MAX)
        {
            vd_say("DftiCommitDescriptor: a length of %lld is not served", (long long)d->n[k]);
            return DFTI_UNIMPLEMENTED;
        }
    /* DFTI's own refusals: a batch with no distance; an in-place real transform of rank 2 or
     * more on the default strides */
    if (d->howmany > 1 && (d->idist == 0 || (d->odist == 0 && (real || !ip))))
    {
        vd_say("DftiCommitDescriptor: a batch (DFTI_NUMBER_OF_TRANSFORMS > 1) needs DFTI_INPUT_DISTANCE%s",
               (real || !ip) ? " and DFTI_OUTPUT_DISTANCE" : "");
        return DFTI_INCONSISTENT_CONFIGURATION;
    }
    vd_rowmajor(d->n, r, 1, def);
    if (real && ip && r >= 2 && vd_side(d->istride, 0, def, -1, r, 1) && vd_side(d->ostride, 0, def, -1, r, 1))
    {
        vd_say("DftiCommitDescriptor: an in-place real transform of rank %d needs the padded strides "
               "(forward: DFTI_INPUT_STRIDES {0, ..., 2(n/2+1), 1}, DFTI_OUTPUT_STRIDES {0, ..., n/2+1, 1}), as DFTI requires", r);
        return DFTI_INCONSISTENT_CONFIGURATION;
    }
    /* the layouts the strides describe, per direction */
    if (!real)
    {
        nf = vd_complex_forms(d, f);
        for (k = 0; k < nf && !fits[0]; k++)
            if (vd_side(d->istride, d->idist, f[k].r, f[k].rdist, r, 0) &&
                (ip || vd_side(d->ostride, d->odist, f[k].r, f[k].rdist, r, 0)))
            {
                fits[0] = fits[1] = 1;
                use[0] = use[1] = f[k];
            }
    }
    else
    {   /* the strides belong to the call: the forward reads reals, the backward the spectrum */
        nf = vd_real_forms(d, f);
        for (k = 0; k < nf; k++)
        {
            if (!fits[0] && vd_side(d->istride, d->idist, f[k].r, f[k].rdist, r, f[k].s0zero) &&
                vd_side(d->ostride, d->odist, f[k].c, f[k].cdist, r, f[k].s0zero))
            {
                fits[0] = 1;
                use[0] = f[k];
            }
            if (!fits[1] && vd_side(d->istride, d->idist, f[k].c, f[k].cdist, r, f[k].s0zero) &&
                vd_side(d->ostride, d->odist, f[k].r, f[k].rdist, r, f[k].s0zero))
            {
                fits[1] = 1;
                use[1] = f[k];
            }
        }
    }
    if (!fits[0] && !fits[1])
    {
        if (real && !ip && r >= 2 && vd_side(d->ostride, 0, def, -1, r, 0))
            vd_say("DftiCommitDescriptor: an out-of-place real transform of rank %d writes packed rows: set "
                   "DFTI_OUTPUT_STRIDES to {0, ..., n/2+1, 1} (DFTI's default output rows are padded to the real pitch, "
                   "which VectorFFT does not write)", r);
        else
            vd_say("DftiCommitDescriptor: %s rank %d, %lld transform(s), %s: the strides and distances are not a layout "
                   "VectorFFT serves (tight strides for one transform or contiguous transforms; the lane-major batch, "
                   "strides times the batch count and distance 1; real data in place also with padded rows; "
                   "docs/roadmap/dfti_shim_design.md section 3)",
                   real ? "real" : "complex", r, (long long)d->howmany, ip ? "in place" : "out of place");
        return DFTI_UNIMPLEMENTED;
    }
    if (d->ordering == DFTI_BACKWARD_SCRAMBLED)
        vd_say("DFTI_ORDERING = DFTI_BACKWARD_SCRAMBLED: the output is in natural order, which DFTI permits");
    {
        const int threads = vd_threads(d);
        memset(cfg, 0, sizeof cfg);
        for (k = 0; k < 2; k++)
            if (fits[k])
                vd_config(d, !real ? VFFT_C2C : k ? VFFT_C2R : VFFT_R2C, use[k].batch, threads, &cfg[k]);
    }
    nplans = d->user_threads > VD_MAXPLANS ? VD_MAXPLANS : (int)d->user_threads;
    /* the first elements: the offsets in doubles (a complex element is two) */
    for (k = 0; k < 2; k++)
    {
        const int incplx = !real || k == 1, outcplx = !real || k == 0;
        d->ioff[k] = (size_t)d->istride[0] * (incplx ? 2u : 1u);
        d->ooff[k] = (ip && !real) ? d->ioff[k] : (size_t)d->ostride[0] * (outcplx ? 2u : 1u);
    }
    /* a recommit that changed nothing the plans were built from keeps them */
    if (d->plan[0] || d->plan[1])
    {
        if (d->nplans == nplans && d->fits[0] == fits[0] && d->fits[1] == fits[1] && !memcmp(d->cfg, cfg, sizeof cfg))
        {
            d->committed = 1;
            return DFTI_NO_ERROR;
        }
        vd_drop_plans(d);
    }
    d->nplans = nplans;
    for (k = 0; k < 2; k++)
    {
        vfft_info_t in;
        if (!fits[k])
            continue;
        if (k == 1 && !real)
        {   /* complex data: the forward's plans serve the backward */
            d->plan[1] = d->plan[0];
            d->out_doubles[1] = d->out_doubles[0];
            break;
        }
        d->plan[k] = (vfft_plan *)calloc((size_t)nplans, sizeof(vfft_plan));
        if (!d->plan[k])
        {
            vd_drop_plans(d);
            return DFTI_MEMORY_ERROR;
        }
        for (s = 0; s < nplans; s++)
        {   /* the first may measure; the rest replay its wisdom */
            d->plan[k][s] = vfft_create(&cfg[k]);
            if (!d->plan[k][s])
            {
                const char *why = vfft_error();
                vd_say("DftiCommitDescriptor: %s", why && *why ? why : "the configuration was refused");
                vd_drop_plans(d);
                return DFTI_UNIMPLEMENTED;
            }
        }
        vfft_info(d->plan[k][0], &in);
        d->out_doubles[k] = in.out_doubles;
        d->threads = in.nthreads;
    }
    d->fits[0] = fits[0];
    d->fits[1] = fits[1];
    memcpy(d->cfg, cfg, sizeof cfg);
    d->committed = 1;
    return DFTI_NO_ERROR;
}

MKL_LONG vfft_dfti_copy(DFTI_DESCRIPTOR_HANDLE from, DFTI_DESCRIPTOR_HANDLE *to)
{
    struct DFTI_DESCRIPTOR *d;
    if (!vd_live(from) || !to)
        return DFTI_BAD_DESCRIPTOR;
    *to = NULL;
    d = (struct DFTI_DESCRIPTOR *)malloc(sizeof *d);
    if (!d)
        return DFTI_MEMORY_ERROR;
    *d = *from; /* the settings; the copy builds plans of its own */
    d->committed = 0;
    d->nplans = 0;
    d->plan[0] = d->plan[1] = NULL;
    d->fits[0] = d->fits[1] = 0;
    memset(d->cfg, 0, sizeof d->cfg);
    d->busy = 0;
    d->waited = 0;
    *to = d;
    if (from->committed)
    {
        const MKL_LONG st = vfft_dfti_commit(d);
        if (st != DFTI_NO_ERROR)
        {
            vfft_dfti_free(to);
            return st;
        }
    }
    return DFTI_NO_ERROR;
}

/* ── compute ─────────────────────────────────────────────────────────────── */

/* a free plan for this compute: one compare-and-swap when uncontended; when every plan is
 * computing, wait for one */
static int vd_borrow(struct DFTI_DESCRIPTOR *d)
{
    const unsigned long long all = d->nplans >= 64 ? ~0ull : ((1ull << d->nplans) - 1ull);
    for (;;)
    {
        unsigned long long b = __atomic_load_n(&d->busy, __ATOMIC_RELAXED);
        const unsigned long long idle = ~b & all;
        if (idle)
        {
            const int s = __builtin_ctzll(idle);
            if (__atomic_compare_exchange_n(&d->busy, &b, b | (1ull << s), 0, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED))
                return s;
            continue;
        }
        if (!__atomic_exchange_n(&d->waited, 1, __ATOMIC_RELAXED))
            vd_say("one descriptor is computed from more threads at once than DFTI_NUMBER_OF_USER_THREADS = %d: "
                   "the computes take turns; set it to the number of threads",
                   d->nplans);
        vd_yield();
    }
}
static void vd_return(struct DFTI_DESCRIPTOR *d, int s)
{
    __atomic_fetch_and(&d->busy, ~(1ull << s), __ATOMIC_RELEASE);
}

static MKL_LONG vd_compute(struct DFTI_DESCRIPTOR *d, int dir, void *x, va_list ap)
{
    double *in, *out;
    int slot;
    if (!vd_live(d))
        return DFTI_BAD_DESCRIPTOR;
    if (!d->committed)
        return DFTI_INVALID_CONFIGURATION;
    if (!d->fits[dir])
    {
        vd_say("%s: the committed strides and distances serve the %s transform only; swap DFTI_INPUT_STRIDES and "
               "DFTI_OUTPUT_STRIDES (and the distances) and recommit, as DFTI requires",
               dir ? "DftiComputeBackward" : "DftiComputeForward", dir ? "forward" : "backward");
        return DFTI_INCONSISTENT_CONFIGURATION;
    }
    in = (double *)x;
    out = d->placement == DFTI_INPLACE ? in : (double *)va_arg(ap, void *);
    in += d->ioff[dir];
    out += d->ooff[dir];
    slot = vd_borrow(d);
    vfft_execute(d->plan[dir][slot], dir ? VFFT_BACKWARD : VFFT_FORWARD, in, out);
    vd_return(d, slot);
    if (d->scale[dir] != 1.0)
    {   /* the direction's scale: one pass over what the plan wrote */
        const double sc = d->scale[dir];
        const size_t nd = d->out_doubles[dir];
        size_t i;
        for (i = 0; i < nd; i++)
            out[i] *= sc;
    }
    return DFTI_NO_ERROR;
}

MKL_LONG vfft_dfti_compute_forward(DFTI_DESCRIPTOR_HANDLE desc, void *x, ...)
{
    va_list ap;
    MKL_LONG st;
    va_start(ap, x);
    st = vd_compute(desc, 0, x, ap);
    va_end(ap);
    return st;
}
MKL_LONG vfft_dfti_compute_backward(DFTI_DESCRIPTOR_HANDLE desc, void *x, ...)
{
    va_list ap;
    MKL_LONG st;
    va_start(ap, x);
    st = vd_compute(desc, 1, x, ap);
    va_end(ap);
    return st;
}

/* ── errors ──────────────────────────────────────────────────────────────── */

char *vfft_dfti_error_message(MKL_LONG status)
{
    static char m0[] = "VectorFFT DFTI: no error";
    static char m1[] = "VectorFFT DFTI: not enough memory";
    static char m2[] = "VectorFFT DFTI: invalid configuration, or the descriptor is not committed";
    static char m3[] = "VectorFFT DFTI: inconsistent configuration";
    static char m5[] = "VectorFFT DFTI: bad descriptor";
    static char m6[] = "VectorFFT DFTI: not implemented";
    static char m7[] = "VectorFFT DFTI: internal error";
    static char mx[] = "VectorFFT DFTI: unknown status";
    switch (status)
    {
    case DFTI_NO_ERROR: return m0;
    case DFTI_MEMORY_ERROR: return m1;
    case DFTI_INVALID_CONFIGURATION: return m2;
    case DFTI_INCONSISTENT_CONFIGURATION: return m3;
    case DFTI_BAD_DESCRIPTOR: return m5;
    case DFTI_UNIMPLEMENTED: return m6;
    case DFTI_MKL_INTERNAL_ERROR: return m7;
    default: return mx;
    }
}

MKL_LONG vfft_dfti_error_class(MKL_LONG status, MKL_LONG error_class)
{
    return status == error_class;
}
