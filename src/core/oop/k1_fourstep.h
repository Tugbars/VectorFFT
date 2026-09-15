/* k1_fourstep.h — the K=1 INTERLEAVED four-step above ZTURN-T's ceiling
 * (docs/design/k1_fourstep_design.md, 2026-09-15; owner: "our Bailey engine
 * is the best solution for this and MKL's code also shows that they are
 * using Bailey for 256k and above").
 *
 * N = N1 x N2 on the 2D INTERLEAVED tier: the signal as N1 rows of N2,
 * step 1 = the column-axis chain (N2 transforms of length N1, stride N2),
 * step 2 = the inter-pass twiddle W_N^(k1 * n2), step 3 = the row plans
 * (N1 transforms of length N2, the K=1 door's own verdict at N2: ZTURN-T at
 * 2048/4096, the pairs and solos below). Steps 1 and 3 are exactly the 2D
 * tier's banded walk with the row pass fused per band; step 2 rides in that
 * walk's row seam (_il2d_row_exec, il2d_tier.h) through the child's
 * il2d_fs_tw table — one plane position p = one row = one two-level record.
 * The 2D child is created through the public create at (N1, N2), DEFAULT
 * order (its column chain leaves k1 digit-reversed; the twiddle is built
 * per POSITION so the plan never permutes at run time), the request's
 * placement and thread count, so every verdict of the child (chain, band
 * width, row route, threading) is the rank-2 cell's own, banked on its
 * own rows, and one 2D cell serves both 1D order classes.
 *
 * ORDER CLASSES. After step 3, plane position p*N2 + k2 holds frequency
 * k1(p) + N1*k2: a fixed permutation of the plan — the SCRAMBLED class
 * serves it as is (no extra pass; the matched backward inverts it). The
 * NATURAL class transposes with the row permutation folded in (row p ->
 * output column k1(p), k2-major): a blocked permuting transpose, one sweep;
 * out of place through the plan's scratch plane, in place through the
 * scratch plane and a copy back (the price of natural order in place at
 * this size — a transposed-write row twin is the lever if the race asks).
 * Backward = the mirror: (natural: the inverse transpose first), the 2D
 * child backward with the conjugate twiddle applied AFTER each row plan and
 * every row BEFORE any column stage (the tier's rows-first backward walks,
 * il2d_tier.h / vfft_execute.h).
 *
 * The planner's race builds candidates through vfft_k1fs_create with the
 * wisdom + config it was handed (_k1fs_ctx, the Bluestein inner-chain
 * precedent); the door replays the banked split (il_R1, il_R2). */
#ifndef VFFT_OOP_K1_FOURSTEP_H
#define VFFT_OOP_K1_FOURSTEP_H

#include <math.h>
#include <string.h>
#include "support/zalloc.h"

/* the 2D tier's column map (il2d_cols.h, included after this header in the
 * one-TU build): scr row j -> natural row */
static int *_il2d_nat_perm(const int *Rs, int nst, int N1);

#include "k1_fourstep_band.h"  /* the band + the side ladder (bench-visible) */

/* the splits (N1, N2) with N1 * N2 = N, both sides in the ladder; returns
 * the count (<= max) */
static int vfft_k1fs_splits(int N, int *n1, int *n2, int max)
{
    int i, j, n = 0;
    for (i = 0; i < VFFT_K1FS_NSIDES && n < max; i++)
    {
        const int a = VFFT_K1FS_SIDES[i];
        if (N % a) continue;
        for (j = 0; j < VFFT_K1FS_NSIDES; j++)
            if (VFFT_K1FS_SIDES[j] == N / a)
            {
                n1[n] = a;
                n2[n] = N / a;
                n++;
                break;
            }
    }
    return n;
}

typedef struct vfft_k1fs_s
{
    int N, N1, N2;
    int scr;                      /* the order class: 1 = SCRAMBLED (the plane as is), 0 = NATURAL (transposed) */
    int inplace;                  /* the placement the child was created for */
    int nthreads;
    struct vfft_plan_s *c2d;      /* the 2D interleaved child at (N1, N2), DEFAULT order */
    double *tw;                   /* N1 records of 2*(N2/B + B) doubles: per plane position */
    int B;                        /* the fine period */
    int *k1_of_p;                 /* plane position p -> column output index k1 */
    int *p_of_k1;                 /* its inverse */
    double *plane;                /* NATURAL: the transpose scratch, 2*N doubles (64-B aligned) */
} vfft_k1fs_plan_t;

/* the planner's race: the child's wisdom + config (set by _k1_il_plan_race
 * before the race; the Bluestein inner-chain provider's pattern) */
static struct { struct vfft_wisdom_s *W; const vfft_config_t *cfg; } _k1fs_ctx;

static void vfft_k1fs_destroy(vfft_k1fs_plan_t *p)
{
    if (!p) return;
    if (p->c2d)
    {
        p->c2d->il2d_fs_tw = NULL;
        vfft_destroy((vfft_plan)p->c2d);
    }
    VFFT_ZS_FREE(p->tw);
    VFFT_ZS_FREE(p->plane);
    free(p->k1_of_p);
    free(p->p_of_k1);
    free(p);
}

/* the scrambled class's permutation: frequency k -> plane index */
static inline long vfft_k1fs_pos_of_bin(const vfft_k1fs_plan_t *p, long k)
{
    const long k1 = k % p->N1, k2 = k / p->N1;
    return (long)p->p_of_k1[k1] * p->N2 + k2;
}

static vfft_k1fs_plan_t *vfft_k1fs_create(int N, int N1, int N2, int scr,
                                          struct vfft_wisdom_s *W, const vfft_config_t *cfg,
                                          int inplace, int nthreads)
{
    vfft_k1fs_plan_t *p;
    vfft_config_t rc;
    int nst, B, q, a, b;
    size_t rec;
    if (N != N1 * N2 || !vfft_k1fs_band(N) || N1 < 4 || N2 < 4) return NULL;
    p = (vfft_k1fs_plan_t *)calloc(1, sizeof *p);
    if (!p) return NULL;
    p->N = N; p->N1 = N1; p->N2 = N2; p->scr = scr; p->inplace = inplace;
    p->nthreads = nthreads > 0 ? nthreads : 1;
    memset(&rc, 0, sizeof rc);
    rc.transform = VFFT_C2C;
    rc.placement = inplace ? VFFT_INPLACE : VFFT_OUTOFPLACE;
    rc.rigor = cfg ? cfg->rigor : VFFT_MEASURE;
    rc.dims = 2;
    rc.n[0] = N1;
    rc.n[1] = N2;
    rc.howmany = 1;
    rc.order = VFFT_ORDER_DEFAULT;      /* the column chain's own order; the twiddle is per position */
    rc.layout = VFFT_LAYOUT_INTERLEAVED;
    rc.nthreads = p->nthreads;
    rc.wisdom = (vfft_wisdom *)W;
    rc.wisdom_write = cfg ? cfg->wisdom_write : 0;
    rc.recalibrate = cfg ? cfg->recalibrate : 0;
    p->c2d = (struct vfft_plan_s *)vfft_create(&rc);
    if (!p->c2d || !p->c2d->il2d_row || p->c2d->il2d_col.nat || p->c2d->il2d_col.blu)
    {   /* the child must be the native tier with the scrambled column chain */
        vfft_k1fs_destroy(p);
        return NULL;
    }
    /* the column map: position p holds k1(p) (identity for a single-stage
     * axis, the chain's digit reversal otherwise) */
    p->k1_of_p = (int *)malloc((size_t)N1 * sizeof(int));
    p->p_of_k1 = (int *)malloc((size_t)N1 * sizeof(int));
    if (!p->k1_of_p || !p->p_of_k1) { vfft_k1fs_destroy(p); return NULL; }
    nst = p->c2d->il2d_col.nst;
    if (nst >= 2)
    {
        int *perm = _il2d_nat_perm(p->c2d->il2d_col.R, nst, N1);
        if (!perm) { vfft_k1fs_destroy(p); return NULL; }
        memcpy(p->k1_of_p, perm, (size_t)N1 * sizeof(int));
        free(perm);
    }
    else
        for (q = 0; q < N1; q++) p->k1_of_p[q] = q;
    for (q = 0; q < N1; q++) p->p_of_k1[p->k1_of_p[q]] = q;
    /* the twiddle records: B = the largest power of two with B*B <= N2 */
    B = 1;
    while ((B << 1) * (B << 1) <= N2) B <<= 1;
    p->B = B;
    rec = 2 * ((size_t)N2 / (size_t)B + (size_t)B);
    p->tw = (double *)VFFT_ZS_ALLOC((size_t)N1 * rec * sizeof(double));
    if (!p->tw) { vfft_k1fs_destroy(p); return NULL; }
    for (q = 0; q < N1; q++)
    {
        const long k1 = p->k1_of_p[q];
        double *C = p->tw + (size_t)q * rec, *F = C + 2 * ((size_t)N2 / (size_t)B);
        const long double th = -2.0L * 3.141592653589793238462643383279L / (long double)N;
        for (a = 0; a < N2 / B; a++)
        {   /* W_N^(k1 * a * B) */
            const long e = (long)(((long long)k1 * (long long)a * (long long)B) % N);
            C[2 * a] = (double)cosl(th * (long double)e);
            C[2 * a + 1] = (double)sinl(th * (long double)e);
        }
        for (b = 0; b < B; b++)
        {   /* W_N^(k1 * b) */
            const long e = (long)(((long long)k1 * (long long)b) % N);
            F[2 * b] = (double)cosl(th * (long double)e);
            F[2 * b + 1] = (double)sinl(th * (long double)e);
        }
    }
    p->c2d->il2d_fs_tw = p->tw;
    p->c2d->il2d_fs_B = B;
    if (!scr)
    {
        p->plane = (double *)VFFT_ZS_ALLOC(2 * (size_t)N * sizeof(double));
        if (!p->plane) { vfft_k1fs_destroy(p); return NULL; }
    }
    return p;
}

/* the natural class's transposes, blocked 16 x 16 complexes, the row
 * permutation folded in: forward = plane row p -> output column k1(p)
 * (k2-major); backward = the inverse */
#define VFFT_K1FS_TB 16
static void _k1fs_transpose_fwd(const vfft_k1fs_plan_t *p, const double *src, double *dst)
{
    const int N1 = p->N1, N2 = p->N2;
    int k1b, k2b, i, j;
    for (k1b = 0; k1b < N1; k1b += VFFT_K1FS_TB)
        for (k2b = 0; k2b < N2; k2b += VFFT_K1FS_TB)
            for (j = 0; j < VFFT_K1FS_TB; j++)
            {
                const int k1 = k1b + j;
                const double *row = src + 2 * ((size_t)p->p_of_k1[k1] * (size_t)N2 + (size_t)k2b);
                for (i = 0; i < VFFT_K1FS_TB; i++)
                {
                    double *o = dst + 2 * ((size_t)(k2b + i) * (size_t)N1 + (size_t)k1);
                    o[0] = row[2 * i];
                    o[1] = row[2 * i + 1];
                }
            }
}
static void _k1fs_transpose_bwd(const vfft_k1fs_plan_t *p, const double *src, double *dst)
{
    const int N1 = p->N1, N2 = p->N2;
    int k1b, k2b, i, j;
    for (k1b = 0; k1b < N1; k1b += VFFT_K1FS_TB)
        for (k2b = 0; k2b < N2; k2b += VFFT_K1FS_TB)
            for (j = 0; j < VFFT_K1FS_TB; j++)
            {
                const int k1 = k1b + j;
                double *row = dst + 2 * ((size_t)p->p_of_k1[k1] * (size_t)N2 + (size_t)k2b);
                for (i = 0; i < VFFT_K1FS_TB; i++)
                {
                    const double *s = src + 2 * ((size_t)(k2b + i) * (size_t)N1 + (size_t)k1);
                    row[2 * i] = s[0];
                    row[2 * i + 1] = s[1];
                }
            }
}

/* both directions, both placements (zin == zout in place), both classes */
static void vfft_k1fs_execute(const vfft_k1fs_plan_t *p, vfft_dir_t dir, const double *zin, double *zout)
{
    const size_t bytes = 2 * (size_t)p->N * sizeof(double);
    if (p->scr)
    {
        vfft_execute((vfft_plan)p->c2d, dir, (double *)zin, NULL, zout, NULL);
        return;
    }
    if (dir == VFFT_FORWARD)
    {
        if (zin != zout)
        {
            vfft_execute((vfft_plan)p->c2d, dir, (double *)zin, NULL, p->plane, NULL);
            _k1fs_transpose_fwd(p, p->plane, zout);
        }
        else
        {
            vfft_execute((vfft_plan)p->c2d, dir, zout, NULL, zout, NULL);
            _k1fs_transpose_fwd(p, zout, p->plane);
            memcpy(zout, p->plane, bytes);
        }
        return;
    }
    if (zin != zout)
    {
        _k1fs_transpose_bwd(p, zin, p->plane);
        vfft_execute((vfft_plan)p->c2d, dir, p->plane, NULL, zout, NULL);
    }
    else
    {
        _k1fs_transpose_bwd(p, zout, p->plane);
        memcpy(zout, p->plane, bytes);
        vfft_execute((vfft_plan)p->c2d, dir, zout, NULL, zout, NULL);
    }
}
static inline void vfft_k1fs_execute_fwd(const vfft_k1fs_plan_t *p, const double *zin, double *zout)
{ vfft_k1fs_execute(p, VFFT_FORWARD, zin, zout); }
static inline void vfft_k1fs_execute_bwd(const vfft_k1fs_plan_t *p, const double *zin, double *zout)
{ vfft_k1fs_execute(p, VFFT_BACKWARD, zin, zout); }

#endif /* VFFT_OOP_K1_FOURSTEP_H */
