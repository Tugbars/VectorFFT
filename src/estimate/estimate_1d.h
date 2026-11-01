/* estimate_1d.h -- the recipe of ANY 1D request for this folder's tools, and the recipe printer. The formulas
 * are the library's: c2c est_plan_1d (src/core/il/estimate/estimate_il.h), r2c / c2r est_plan_real
 * (src/core/il/estimate/real_estimate_il.h, moved there from this file on 2026-11-01; the sketch
 * docs/design/estimate_1d_sketch.md sections 6 and 7, its defaults and gaps noted in that file).
 *
 * Implements est_plan_any and est_recipe_print (the contracts are in estimate_recipe.h). Library behaviour
 * is cited file:line, relative to src/core/.
 *
 * PURE: reads only the machine and the request. No clock, no store, no allocation, no I/O. */
#ifndef EST_1D_H
#define EST_1D_H

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "estimate_recipe.h"

/* ================================================================ the contracts */

static inline int est_plan_any(const est_machine_t *m, const est_request_t *rq, est_any_recipe_t *out, char *why,
                               size_t whylen)
{
    char dummy[1];
    if (!why || !whylen) { why = dummy; whylen = sizeof dummy; }
    why[0] = 0;
    if (!m || !rq || !out) return est_refuse(why, whylen, "missing argument");
    if (rq->transform == VFFT_C2C)
    {   /* the library's formulas */
        memset(out, 0, sizeof *out);
        snprintf(out->eng, sizeof out->eng, "k1");
        return est_plan_1d(m, rq, &out->c, why, whylen);
    }
    return est_plan_real(m, rq, out, why, whylen);   /* the library's real formulas (il/estimate/real_estimate_il.h) */
}

/* the c2c route row: the key, then the route's tokens (the library's est_c2c_tokens), the T row's
 * threaded tokens (il_mt il_mt_tw il_mtsb, set with vw2_update_field after the bank) and the
 * four-step's fs_* */
static inline void est_print_c2c(est_sb_t *s, const est_recipe_t *rc, const est_request_t *rq, int T)
{
    const char *pl = rq->placement == VFFT_PLACEMENT_IN ? "ip" : "oop";
    if (rc->k1_il_route == VFFT_K1_IL_PRIME)
    {   /* the method row first: vw2_prime_method_bank's key and tokens (wisdom2_oop_il.h:529-582) */
        est_sb_add(s, "t=c2c n=%d q=1 ord=scr place=ip role=method lay=il | eng=%s in=%s in_sh=%s in_tw=%d",
                   rq->N, rc->prime.method == 1 ? "rader" : "bluestein", rc->prime.in_kind, rc->prime.in_shape,
                   rc->prime.in_tw);
        if (!strcmp(rc->prime.in_kind, "fs") && rc->fs[0]) est_sb_add(s, " %s", rc->fs);
        est_sb_add(s, "\n");
    }
    est_sb_add(s, "t=c2c n=%d q=1 ord=%s place=%s role=comp lay=il", rq->N,
               rq->order == VFFT_ORDER_SCRAMBLED ? "scr" : "nat", pl);
    if (T > 1) est_sb_add(s, " nthreads=%d", T);
    est_sb_add(s, " | ");
    est_c2c_tokens(s, rc, "");
    if (T > 1)
    {
        if (rc->k1_il_route == VFFT_K1_IL_FLAT || rc->k1_il_route == VFFT_K1_IL_ZTT || rc->k1_il_route == VFFT_K1_IL_PRIME)
        {
            est_sb_add(s, " il_mt=%d", rc->il_mt);
            if (rc->k1_il_route == VFFT_K1_IL_FLAT && rc->il_mt == 2) est_sb_add(s, " il_mt_tw=%d", rc->il_mt_tw);
        }
        if (rc->k1_il_route == VFFT_K1_IL_FS)
        {
            est_sb_add(s, " il_mt=%d il_mtsb=", rc->il_mt);
            if (rc->il_zt_n) est_sb_chain(s, rc->il_zt, rc->il_zt_n);
            else est_sb_add(s, "0");
        }
    }
    if (rc->k1_il_route == VFFT_K1_IL_FS && rc->fs[0]) est_sb_add(s, " %s", rc->fs);
    est_sb_add(s, "\n");
    if ((rc->k1_il_route == VFFT_K1_IL_2P_PURE || rc->k1_il_route == VFFT_K1_IL_CHAIN3) && rc->il_bkv)
    {   /* the dir=bwd sibling, keyed ord=nat at the request's placement (dp_planner_il.h:2266-2290) */
        est_sb_add(s, "t=c2c n=%d q=1 ord=nat place=%s dir=bwd role=comp lay=il | ", rq->N, pl);
        est_bwd_tokens(s, rc, "");
        est_sb_add(s, "\n");
    }
}

static inline int est_recipe_print(const est_any_recipe_t *rc, const est_request_t *rq, char *buf, size_t cap)
{
    const int T = rq->nthreads > 1 ? rq->nthreads : 1;
    const est_recipe_t *ch = &rc->c;
    est_sb_t s;
    est_sb_init(&s, buf, cap);
    if (rq->transform == VFFT_C2C)
        est_print_c2c(&s, ch, rq, T);
    else
    {   /* the real row: vw2_real_il_key (wisdom2_real_il.h:60-68) and the engine's tokens
         * (wisdom2_real_il.h:1-44) */
        est_sb_add(&s, "t=%s n=%d q=1 ord=nat place=%s", rq->transform == VFFT_C2R ? "c2r" : "r2c", rq->N,
                   rq->placement == VFFT_PLACEMENT_IN ? "ip" : "oop");
        if (T > 1) est_sb_add(&s, " nthreads=%d", T);
        est_sb_add(&s, " | eng=%s", rc->eng);
        if (!strcmp(rc->eng, "zrp"))
            est_sb_add(&s, " pair=%d.%d leaf=%s", rc->zrp.R1, rc->zrp.R2, rc->zrp.form ? "r2z" : "n1t");
        else if (!strcmp(rc->eng, "zttr"))
        {
            est_sb_add(&s, " chain=");
            est_sb_chain(&s, rc->zttr.chain, rc->zttr.nf);
            est_sb_add(&s, " tile=%u stk=%d", (unsigned)rc->zttr.tile, rc->zttr.stk);
            if (rc->zttr.mt) est_sb_add(&s, " mt=%d", rc->zttr.mt);
        }
        else if (!strcmp(rc->eng, "zfsr"))
            est_sb_add(&s, " split=%dx%d %s", rc->zfsr.n1, rc->zfsr.n2, ch->fs);
        else if (!strcmp(rc->eng, "zrf"))
        {
            est_sb_add(&s, " chain=");
            est_sb_chain(&s, rc->zrf.chain, rc->zrf.K);
            est_sb_add(&s, " msz=%d tile=%d", rc->zrf.nomsz ? 0 : 1, rc->zrf.tile);
            if (rc->zrf.mt) est_sb_add(&s, " mt=%d", rc->zrf.mt);
        }
        else if (!strcmp(rc->eng, "zrb"))
        {
            est_sb_add(&s, " m=%d in=%s in_sh=%s in_tw=%d", rc->zrb.M, rc->zrb.kind, rc->zrb.shape, rc->zrb.tw);
            if (rc->zrb.mt) est_sb_add(&s, " mt=%d", rc->zrb.mt);
            if (!strcmp(rc->zrb.kind, "fs") && ch->fs[0]) est_sb_add(&s, " %s", ch->fs);
        }
        else if (!strcmp(rc->eng, "zr2c"))
        {   /* vw2_real_il_bank_zr2c's order (wisdom2_real_il.h:422-470) */
            est_sb_add(&s, " route=%s", rc->zr2c.route ? "child_nat_ip" : "child_oop_il");
            if (T > 1) est_sb_add(&s, " fold=%s", rc->zr2c.fold_mt ? "mt" : "st");
            est_sb_add(&s, " il_route=%s", vw2_oop_il_name[ch->k1_il_route]);
            if (ch->il_R1 || ch->il_R2) est_sb_add(&s, " il_pair=%d.%d", ch->il_R1, ch->il_R2);
            if (ch->k1_il_route == VFFT_K1_IL_CHAIN3) est_sb_add(&s, " il_chain=%d.%d.%d", ch->il_c3[0], ch->il_c3[1], ch->il_c3[2]);
            if (ch->k1_il_route == VFFT_K1_IL_FLAT)
            {
                est_sb_add(&s, " il_flat=");
                est_sb_chain(&s, ch->il_fl, ch->il_fl_n);
                if (ch->il_flf[0]) est_sb_add(&s, " il_forms=%s", ch->il_flf);
            }
            if (ch->k1_il_route == VFFT_K1_IL_ZTT || (ch->k1_il_route == VFFT_K1_IL_FS && ch->il_zt_n >= 2))
            {
                est_sb_add(&s, " %s=", ch->k1_il_route == VFFT_K1_IL_ZTT ? "il_ztt" : "il_sb");
                est_sb_chain(&s, ch->il_zt, ch->il_zt_n);
            }
            if (ch->il_tw > 0) est_sb_add(&s, " il_tw=%d", ch->il_tw);
            if (ch->k1_il_route == VFFT_K1_IL_PRIME)
            {
                est_sb_add(&s, " il_prime=%s il_prime_in=%s il_prime_sh=%s il_prime_tw=%d",
                           ch->prime.method == 1 ? "rader" : "bluestein", ch->prime.in_kind, ch->prime.in_shape,
                           ch->prime.in_tw);
                if (ch->il_mt > 0) est_sb_add(&s, " il_mt=%d", ch->il_mt);
            }
            if (ch->fs[0] && (ch->k1_il_route == VFFT_K1_IL_FS || ch->k1_il_route == VFFT_K1_IL_PRIME)) est_sb_add(&s, " %s", ch->fs);
            est_sb_add(&s, " il_kv=%d il_bkv=%d", ch->il_kv, ch->il_bkv);
        }
        est_sb_add(&s, "\n");
    }
    if (buf && cap) buf[s.len < cap ? s.len : cap - 1] = 0;
    return (int)s.len;
}

#endif /* EST_1D_H */
