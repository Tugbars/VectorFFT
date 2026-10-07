/* vfft_memory.h -- the public memory calls (vfft.h, "memory"): vfft_malloc,
 * vfft_free, vfft_alignment, vfft_plan_alloc and vfft_buffers_free.
 *
 * ONE ALLOCATOR. vfft_malloc is the library's own body, vfft_aligned_alloc
 * (common/support/zalloc.h): the races, the plans and the callers allocate
 * through one function, so a caller's buffer starts on the alignment every
 * race timed and every published number assumed. It takes no alignment
 * argument (nothing to raise or replace silently), rounds the size up to the
 * alignment, gives a unique block for 0 bytes, returns uninitialised memory,
 * keeps no state, and returns NULL quietly when memory runs out; a size the
 * rounding would overflow is a caller bug and prints one line. On Windows the
 * block comes from the process heap with its own pointer header (zalloc.h),
 * so a block allocated by one module that links this static library may be
 * released by any other.
 *
 * WHY 64 BYTES, NOT 32 -- IN THE AVX2 BUILD TOO. A 32-byte vector at a
 * 32-byte base never splits a cache line, so 32 looks enough for AVX2. It is
 * not, because the engines build 64-byte structures INSIDE the caller's
 * output:
 *   - ZTURN-T out of place runs every stage in zout itself, as 64-byte
 *     [re x4][im x4] blocks (il/rank1/ztt.h). At a 32-byte base every block
 *     straddles two lines. The ingest scatters each column's run to a
 *     digit-reversed base, and the run that shares the other half of a line
 *     is written hundreds to thousands of iterations later, after that line
 *     has left L1: the ingest's store traffic grows 1.5-2x.
 *   - The four-step streams its output rows with non-temporal stores behind
 *     a 32-byte test (il/rank1/k1_fourstep.h), which a 32-byte base passes;
 *     every 256-byte row burst then starts and ends on a half-written line,
 *     which the write-combining buffers flush as partial writes. The real
 *     four-step's sweep (il/real/zfsr.h) and the 2D row stream do the same.
 *   - Threaded, neighbouring blocks belong to different workers, so a
 *     32-byte base puts two workers on one line.
 * Measured on the AVX2 host (docs/research/user_allocator/M_placement.md): a
 * 32-byte output cost +24..28% from N = 4096 at one thread (ZTURN-T at 16384
 * +27.5%, r2c at 262144 1.35x) and 1.6-1.8x at eight threads; a 16-byte
 * output, what malloc guarantees, up to 1.7x and 2.0x. Page alignment and
 * 2 MB pages gained nothing beyond 64. (The mechanisms are read in the code;
 * hardware counters have not confirmed them.) In the AVX-512 build the
 * vector IS the line: every 64-byte access at any other base splits.
 *
 * THE PLAN'S SET (owner, 2026-10-04). vfft_plan_alloc hands out every buffer
 * vfft_execute(p, ...) takes, in the plan's own geometry, as one opaque set
 * with one release: each role is its own block (the way the races allocate:
 * separate blocks, never a carve), sized here with the arithmetic checked.
 * A SPLIT plan created with config.owned_buffers = 1 gets its planes at the
 * stride its pad race chose (the owned batch's Kp, vfft_plan_stride) with the
 * pad lanes zeroed -- padding pays only when the caller's planes already have
 * Kp lanes, and a pad lane is transformed like any other -- and every other
 * plan gets the tight planes of its data contract. Refused, with a printed
 * reason: a SPLIT plan of rank 2 or more, and an INTERLEAVED real batch in
 * its lane-major geometry that runs on the split engines through the bridge
 * (bridge/real_bridge.h, the one crossing left; no surface is added to it).
 *
 * POSITION IN vfft.c: after the plan struct, _vfft_warn and the owned-batch
 * cluster (split/rank1/vfft_batch.h).
 */
#ifndef VFFT_MEMORY_H
#define VFFT_MEMORY_H

/* ── the bytes ──────────────────────────────────────────────────────────── */

void *vfft_malloc(size_t bytes)
{
    if (bytes > SIZE_MAX - 2 * (size_t)VFFT_ALIGNMENT)
    {
        _vfft_warn("vfft_malloc: %llu bytes cannot be rounded up to the alignment "
                   "without overflowing size_t -- NULL (compute buffer sizes in "
                   "size_t, or take them from vfft_plan_alloc)",
                   (unsigned long long)bytes);
        return NULL;
    }
    return vfft_aligned_alloc(bytes);
}

void vfft_free(void *p)
{
    vfft_aligned_free(p);
}

size_t vfft_alignment(void)
{
    return (size_t)VFFT_ALIGNMENT;
}

/* ── the plan's set ─────────────────────────────────────────────────────── */

struct vfft_buffers_s
{
    double *blk[4]; /* one block per distinct role: sre, sim, dre, dim */
};

/* a plan's roles in doubles (0 = unused), the rows of Kp lanes of a padded
 * split plane, and whether the destination roles are the source's */
typedef struct
{
    size_t dbl[4];
    size_t rows[4];
    size_t K, Kp;
    int alias;
} _vm_geom_t;

static int _vm_mul(size_t *r, size_t a, size_t b)
{
    if (b && a > SIZE_MAX / b)
        return 0;
    *r = a * b;
    return 1;
}

/* the roles of plan h: 1 = filled, 0 = refused (printed) */
static int _vm_geometry(const struct vfft_plan_s *h, _vm_geom_t *g)
{
    const int real = (h->transform == VFFT_R2C || h->transform == VFFT_C2R);
    const int ip = (h->placement == VFFT_INPLACE);
    const size_t N = (size_t)h->N, K = h->K, hp = (size_t)(h->N / 2 + 1);
    size_t a = 0, b = 0;
    memset(g, 0, sizeof *g);
    g->K = g->Kp = K;
    if (h->layout == (int)VFFT_LAYOUT_INTERLEAVED)
    {
        if (real && h->N2 == 0 && K > 1 && !h->tcb && !h->zrbl)
        {
            _vfft_warn("vfft_plan_alloc: this interleaved real batch in its lane-major "
                       "geometry runs on the split engines through the bridge -- no set "
                       "is sized for it (allocate z_in and z_out with vfft_malloc)");
            return 0;
        }
        if (!real)
        { /* C2C, any rank: K transforms or planes of N (x N2 x N3 x N4) complexes */
            size_t E = N;
            if ((h->N2 && !_vm_mul(&E, E, (size_t)h->N2)) || (h->N3 && !_vm_mul(&E, E, (size_t)h->N3)) ||
                (h->N4 && !_vm_mul(&E, E, (size_t)h->N4)) || !_vm_mul(&E, E, K) || !_vm_mul(&a, E, 2))
                goto overflow;
            b = a;
        }
        else if (h->N2 == 0)
        { /* 1D real: N reals against N/2+1 bins, per transform */
            if (!_vm_mul(&a, N, K) || !_vm_mul(&b, hp, K) || !_vm_mul(&b, b, 2))
                goto overflow;
        }
        else if (h->N3 == 0)
        { /* 2D real: N rows of N2 reals against N rows of N2/2+1 bins, per plane */
            const size_t h2 = (size_t)(h->N2 / 2 + 1);
            if (!_vm_mul(&a, N, (size_t)h->N2) || !_vm_mul(&a, a, K) || !_vm_mul(&b, N, h2) ||
                !_vm_mul(&b, b, K) || !_vm_mul(&b, b, 2))
                goto overflow;
        }
        else
        {
            _vfft_warn("vfft_plan_alloc: an interleaved real plan of rank 3 or more has no "
                       "sized set");
            return 0;
        }
        if (h->transform == VFFT_C2R)
        { /* the spectrum is the source */
            const size_t t = a;
            a = b;
            b = t;
        }
        if (ip)
        { /* one buffer serves both roles: the larger (the real in-place plane) */
            g->dbl[0] = a > b ? a : b;
            g->alias = 1;
        }
        else
        {
            g->dbl[0] = a;
            g->dbl[2] = b;
        }
        return 1;
    }
    /* SPLIT */
    if (h->N2)
    {
        _vfft_warn("vfft_plan_alloc: a SPLIT plan of rank 2 or more has no sized set");
        return 0;
    }
    if (h->own_batch)
        g->Kp = h->own_batch->Kp; /* the stride the pad race chose */
    {
        size_t plane, spec;
        if (!_vm_mul(&plane, N, g->Kp) || !_vm_mul(&spec, hp, g->Kp))
            goto overflow;
        if (h->transform == VFFT_C2C)
        {
            g->dbl[0] = g->dbl[1] = plane;
            g->rows[0] = g->rows[1] = N;
            if (ip)
                g->alias = 1;
            else
            {
                g->dbl[2] = g->dbl[3] = plane;
                g->rows[2] = g->rows[3] = N;
            }
        }
        else if (real)
        {
            if (ip)
            {
                _vfft_warn("vfft_plan_alloc: a SPLIT real plan in place has no sized set");
                return 0;
            }
            if (h->transform == VFFT_R2C)
            { /* real in, split spectrum out */
                g->dbl[0] = plane;
                g->rows[0] = N;
                g->dbl[2] = g->dbl[3] = spec;
                g->rows[2] = g->rows[3] = hp;
            }
            else
            { /* split spectrum in, real out */
                g->dbl[0] = g->dbl[1] = spec;
                g->rows[0] = g->rows[1] = hp;
                g->dbl[2] = plane;
                g->rows[2] = N;
            }
        }
        else
        { /* real-to-real: real in, real out */
            g->dbl[0] = plane;
            g->rows[0] = N;
            if (ip)
                g->alias = 1;
            else
            {
                g->dbl[2] = plane;
                g->rows[2] = N;
            }
        }
    }
    return 1;
overflow:
    _vfft_warn("vfft_plan_alloc: this plan's buffer sizes overflow size_t");
    return 0;
}

vfft_buffers vfft_plan_alloc(vfft_plan p, double **sre, double **sim, double **dre, double **dim)
{
    const struct vfft_plan_s *h = (const struct vfft_plan_s *)p;
    struct vfft_buffers_s *s;
    double *r[4] = {NULL, NULL, NULL, NULL};
    _vm_geom_t g;
    int i;
    if (sre)
        *sre = NULL;
    if (sim)
        *sim = NULL;
    if (dre)
        *dre = NULL;
    if (dim)
        *dim = NULL;
    if (!h)
    {
        _vfft_warn("vfft_plan_alloc: NULL plan -- no set");
        return NULL;
    }
    if (!_vm_geometry(h, &g))
        return NULL;
    s = (struct vfft_buffers_s *)vfft_aligned_alloc(sizeof *s);
    if (!s)
        return NULL;
    memset(s, 0, sizeof *s);
    for (i = 0; i < 4; i++)
    {
        size_t bytes;
        if (!g.dbl[i])
            continue;
        if (!_vm_mul(&bytes, g.dbl[i], sizeof(double)))
        {
            _vfft_warn("vfft_plan_alloc: this plan's buffer sizes overflow size_t");
            vfft_buffers_free(s);
            return NULL;
        }
        s->blk[i] = (double *)vfft_aligned_alloc(bytes);
        if (!s->blk[i])
        {
            vfft_buffers_free(s);
            return NULL;
        }
        if (g.Kp > g.K && g.rows[i])
        { /* the pad lanes of every row: zero, they are transformed like any other */
            size_t row;
            for (row = 0; row < g.rows[i]; row++)
                memset(s->blk[i] + row * g.Kp + g.K, 0, (g.Kp - g.K) * sizeof(double));
        }
    }
    r[0] = s->blk[0];
    r[1] = s->blk[1];
    r[2] = g.alias ? s->blk[0] : s->blk[2];
    r[3] = g.alias ? s->blk[1] : s->blk[3];
    if (sre)
        *sre = r[0];
    if (sim)
        *sim = r[1];
    if (dre)
        *dre = r[2];
    if (dim)
        *dim = r[3];
    return s;
}

void vfft_buffers_free(vfft_buffers b)
{
    int i;
    if (!b)
        return;
    for (i = 0; i < 4; i++)
        vfft_aligned_free(b->blk[i]);
    vfft_aligned_free(b);
}

#endif /* VFFT_MEMORY_H */
