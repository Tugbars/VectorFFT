/* real_child.h -- THE REAL FFT INSIDE A REAL-TO-REAL PLAN, CREATED AS A CHILD IN ROLE (owner, 2026-11-14).
 *
 * A DST-II or DST-III plan (il/real/dst.h) and a DCT-I or DST-I plan (il/real/dct1_dst1.h) each run one 1D
 * real FFT between two passes of their own. That FFT is a component of the real-to-real cell, not a cell of
 * its own (il/wisdom/wisdom2_child.h): it is created against a private store seeded from the cell's row,
 * replays the recipe there or races, and what it banked goes back onto the cell's row under the cell's
 * prefix. Its race runs in the cell's role (il/real/real_role.h): every candidate FFT is timed as one whole
 * real-to-real transform, the two passes with that FFT between them, and the fastest whole transform wins.
 * A real-to-real create never reads or writes an r2c or a c2r row.
 *
 * Under VFFT_FLAG_ESTIMATE the FFT is planned by formulas: nothing is measured, no store is read or written. */
#ifndef VFFT_IL_REAL_CHILD_H
#define VFFT_IL_REAL_CHILD_H
#ifndef VFFT_FLOAT   /* FP64 only: the FP32 unit serves 1D C2C */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "il/real/real_role.h"         /* the role the real FFT is raced in: the whole real-to-real transform */
#include "il/wisdom/wisdom2_real_il.h" /* the real-to-real cell's row */
#include "il/wisdom/wisdom2_child.h"   /* the private store, the child's rows on the cell's row */

typedef struct
{
    int t;                                           /* the cell's transform in the store's words (VW2_T_DST2, ...) */
    int N;                                           /* the cell's length */
    const char *pre;                                 /* the FFT's prefix on the cell's row: "dst_", "dct_" */
    const char *eng;                                 /* the row's eng= value */
    int n;                                           /* the FFT's length */
    int is_c2r;                                      /* a C2R; else an R2C */
    void (*run)(void *ctx, struct vfft_plan_s *h);   /* the role: one whole transform, the candidate h inside */
    void *ctx;
} _real_child_t;

static vfft_plan _vfft_create_estimate(const vfft_cfg_t *cfg);   /* il/estimate/c2c_estimate_il.h */

/* The real FFT rc names, out of place, for the real-to-real request cfg. est: planned by formulas. Else a child
 * in role: its private store *Sp seeded from the cell's row in W (empty when the row carries no recipe, or
 * there is no store), the create against it with nothing persisted, every candidate of a race timed by
 * rc->run; where the child raced, or the row did not carry it yet, the cell's row is banked afresh -- the time
 * of one whole transform from the child's winner -- with the child's rows under rc->pre (*changed = 1: the
 * caller saves). NULL: the real door's reason is on the record. */
static struct vfft_plan_s *_real_child_create(const vfft_cfg_t *cfg, struct vfft_wisdom_s *W, const _real_child_t *rc,
                                              int est, struct vfft_wisdom_s **Sp, int *changed)
{
    vfft_cfg_t c = *cfg;
    c.transform = rc->is_c2r ? VFFT_C2R : VFFT_R2C;
    c.dims = 1;
    c.n[0] = rc->n;
    c.howmany = 1;
    c.layout = VFI_LAYOUT_IL;
    c.placement = VFFT_OUTOFPLACE;
    c.order = VFI_ORDER_DEFAULT;
    c.batch_geom = VFFT_BATCH_DEFAULT;
    c.owned_buffers = 0;
    c.destroy_input = 0;
    if (est)
        return (struct vfft_plan_s *)_vfft_create_estimate(&c);
    {
        const int T = _vfft_plan_threads(cfg), usable = W && !W->vw2_off_oop;
        struct vfft_plan_s *ch;
        vw2_key_t pk;
        vw2_r2r_key(&pk, rc->t, rc->N, T);
        *Sp = vfft_child_store_for(usable ? &W->vw2 : NULL, &pk, rc->pre);
        if (!*Sp)
        {
            _vfft_warn("vfft_create: %s N=%d: out of memory (the real FFT's store)", _vfft_tname(cfg->transform), rc->N);
            return NULL;
        }
        c.wisdom = (vfft_wisdom *)*Sp;
        c.wisdom_write = 0;   /* a private store's plan never saves (il/wisdom/wisdom2_child.h) */
        _real_il_role.run = rc->run;
        _real_il_role.ctx = rc->ctx;
        _real_il_role.transform = (int)c.transform;
        _real_il_role.N = rc->n;
        ch = (struct vfft_plan_s *)_vfft_create_cfg(&c);
        memset(&_real_il_role, 0, sizeof _real_il_role);
        if (ch && usable && (*Sp)->vw2.nrec > 0 &&
            (vfft_child_store_raced(*Sp) || !vfft_child_row_has(vfft_child_parent_row(&W->vw2, &pk), rc->pre)))
        {
            vw2_key_t ck;
            const vw2_rec_t *cr;
            const char *nsv;
            vw2_real_il_key(&ck, rc->n, rc->is_c2r, 0, T);
            cr = vw2_lookup(&(*Sp)->vw2, &ck);
            nsv = cr ? vw2_rec_get(cr, "ns") : NULL;
            if (vw2_r2r_bank(&W->vw2, &pk, rc->eng, nsv ? atof(nsv) : 0.0) == VW2_OK &&
                vfft_child_row_update(&W->vw2, &pk, rc->pre, *Sp))
                *changed = 1;
            else
                fprintf(stderr, "vfft: %s verdict NOT banked at N=%d -- the cell will re-race\n", vw2_t_name[rc->t], rc->N);
        }
        return ch;
    }
}

#endif /* !VFFT_FLOAT */
#endif /* VFFT_IL_REAL_CHILD_H */
