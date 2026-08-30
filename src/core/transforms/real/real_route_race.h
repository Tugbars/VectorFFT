/* real_route_race.h - the r2c/c2r route racers.
 *
 * The ARMS and the CLOCK. Extracted from vfft.c as migration step 11; see
 * docs/design/refactor_migration_plan.md.
 *
 * RACERS, NOT DECIDERS - AND THE LINE IS SHARP
 * --------------------------------------------
 * A route decision here has two halves, and they have different dependencies:
 *
 *   the RACER   builds both arms, times them, reports two numbers. It needs a
 *               clock and nothing else. That is this file.
 *   the DECIDER walks the precedence ladder - env hook, then a banked verdict,
 *               then race-and-bank, then the structural default - and writes
 *               the winner to wisdom. It touches vfft_plan_s, the create-race
 *               counter, and the store.
 *
 * The deciders (_r2c_route_decide, _c2r_route_decide) therefore stay in vfft.c
 * with the wisdom write path. Splitting here is not a convenience: it is the
 * difference between "how fast are these two arms" and "what should this cell
 * be served with, and who may reuse that answer".
 *
 * THE PROTOCOL, AND WHY IT IS SHAPED THIS WAY
 * -------------------------------------------
 * Alternating-order median-of-9 on ONE buffer set. Alternating is what makes
 * the verdict survive a thermally noisy host: both arms see the same drift, so
 * WHICH ARM WINS is robust even when the nanoseconds are not. One buffer set is
 * what makes it fair: both arms honour the same split re/im I/O contract, so
 * neither is charged for a layout the other avoided.
 *
 * `as_z` times the arms through the INTERLEAVED door instead of the split one -
 * what an interleaved caller's execute actually runs. A verdict is only valid
 * for the door it was measured through, which is why this is an argument rather
 * than an assumption.
 *
 * The c2r racer carries a deliberate ASYMMETRY: it leans toward stride on a
 * near-tie, because stride threads and owns high K, and a calibration wobble
 * should not flip a tie into the arm that cannot scale.
 *
 * FLOOR-LEGAL BY CONSTRUCTION
 * ---------------------------
 * No vfft_plan_s, no wisdom, no counter, no mutable file-scope state. Does NOT
 * pull engine/stride_executor.h.
 */
#ifndef VFFT_TRANSFORMS_REAL_REAL_ROUTE_RACE_H
#define VFFT_TRANSFORMS_REAL_REAL_ROUTE_RACE_H

#include <stdlib.h>

#include "r2c.h"                    /* vfft_r2c_plan_t + the dispatch knobs */
#include "c2r_dispatch.h"           /* vfft_c2r_disp_t and its execute door */
#include "support/race_timing.h"    /* the shared clock and median-of-9 */

/* Build exactly one r2c arm: the rfft cascade, or the decoupled stride. */
static vfft_r2c_plan_t *_r2c_build_arm(int N, size_t K, int stride_arm,
                                       const vfft_proto_registry_t *reg)
{
    size_t saved = vfft_r2c_dispatch_get_decouple_min_k();
    vfft_r2c_dispatch_set_decouple_min_k(stride_arm ? 0 : (size_t)-1);
    vfft_r2c_plan_t *p = vfft_r2c_plan_create(N, K, VFFT_R2C_SPLIT, _rfft_registry(),
                                              NULL, (vfft_proto_registry_t *)reg);
    vfft_r2c_dispatch_set_decouple_min_k(saved);
    return p;
}

/* Alternating-order median-of-9 A/B on ONE buffer set (both arms share the
 * same split re/im I/O contract). 0 on success. */
static int _r2c_race_arms(vfft_r2c_plan_t *pr, vfft_r2c_plan_t *ps,
                          int N, size_t K, int as_z,
                          double *n_rfft, double *n_stride)
{
    /* as_z: time the arms through the INTERLEAVED z door
     * (vfft_r2c_execute_fwd_z) — the exact entry an interleaved caller's
     * execute uses — instead of the split planes. The banked label then
     * names what was measured (owner directive 2026-08-25: IL races too,
     * never inherits a split-timed verdict). Both doors wrap the SAME
     * plan; only the timed I/O contract differs. */
    size_t insz = (size_t)N * K, outsz = (size_t)(N / 2 + 1) * K;
    double *x = NULL, *orr = NULL, *oii = NULL, *z = NULL;
    double a[9], b[9];
    int reps, r;
    if (vfft_proto_posix_memalign((void **)&x, 64, insz * sizeof(double)) ||
        (as_z
             ? vfft_proto_posix_memalign((void **)&z, 64, 2 * outsz * sizeof(double))
             : (vfft_proto_posix_memalign((void **)&orr, 64, outsz * sizeof(double)) ||
                vfft_proto_posix_memalign((void **)&oii, 64, outsz * sizeof(double)))))
    {
        vfft_proto_aligned_free(x);
        vfft_proto_aligned_free(orr);
        vfft_proto_aligned_free(oii);
        vfft_proto_aligned_free(z);
        return -1;
    }
    for (size_t i = 0; i < insz; i++)
        x[i] = (double)((i * 2654435761u) & 0xffff) / 65536.0 - 0.5;
#define VFFT__R2C_ARM(P) do {                                   \
        if (as_z) vfft_r2c_execute_fwd_z((P), x, z);            \
        else      vfft_r2c_execute_fwd((P), x, orr, oii);       \
    } while (0)
    for (int w = 0; w < 5; w++)
    {
        VFFT__R2C_ARM(pr);
        VFFT__R2C_ARM(ps);
    }
    reps = (int)(2e6 / (double)(insz + 1));
    if (reps < 20)
        reps = 20;
    if (reps > 100000)
        reps = 100000;
    for (r = 0; r < 9; r++)
    {
        vfft_r2c_plan_t *first = (r & 1) ? ps : pr;
        vfft_r2c_plan_t *second = (r & 1) ? pr : ps;
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            VFFT__R2C_ARM(first);
        double tf = (vfft_proto_now_ns() - t0) / reps;
        t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            VFFT__R2C_ARM(second);
        double tsc = (vfft_proto_now_ns() - t0) / reps;
        a[r] = (r & 1) ? tsc : tf;  /* rfft   */
        b[r] = (r & 1) ? tf : tsc;  /* stride */
    }
#undef VFFT__R2C_ARM
    vfft_proto_aligned_free(x);
    vfft_proto_aligned_free(orr);
    vfft_proto_aligned_free(oii);
    vfft_proto_aligned_free(z);
    *n_rfft = _il_ab_med9(a);
    *n_stride = _il_ab_med9(b);
    return 0;
}

/* Build NATURAL + STRIDE c2r for (N,K), time ST, return the faster. The c2r analog
 * of _r2c_bakeoff: BOTH consume split re/im (same caller I/O contract), so the pick
 * is transparent. NATURAL = the fast packed cascade on split input (no repack, the
 * low/mid-K winner); STRIDE = the decoupled high-K path that also threads. Hysteresis
 * toward stride on a near-tie (it threads and owns high K; calibration noise can't
 * flip a tie to natural). */
/* Alternating-order median-of-9 A/B, c2r twin of _r2c_race_arms.
 * as_z: time through the interleaved-spectrum door (vfft_c2r_disp_execute_z)
 * — what an interleaved caller's execute runs — instead of split planes. */
static int _c2r_race_arms(vfft_c2r_disp_t *pn, vfft_c2r_disp_t *ps,
                          int N, size_t K, int as_z,
                          double *n_nat, double *n_split)
{
    size_t outsz = (size_t)N * K, hcsz = (size_t)(N / 2 + 1) * K;
    double *re = NULL, *im = NULL, *y = NULL, *z = NULL;
    double a[9], b[9];
    int reps, r;
    if (vfft_proto_posix_memalign((void **)&y, 64, outsz * sizeof(double)) ||
        (as_z
             ? vfft_proto_posix_memalign((void **)&z, 64, 2 * hcsz * sizeof(double))
             : (vfft_proto_posix_memalign((void **)&re, 64, hcsz * sizeof(double)) ||
                vfft_proto_posix_memalign((void **)&im, 64, hcsz * sizeof(double)))))
    {
        vfft_proto_aligned_free(re);
        vfft_proto_aligned_free(im);
        vfft_proto_aligned_free(y);
        vfft_proto_aligned_free(z);
        return -1;
    }
    if (as_z)
        for (size_t i = 0; i < 2 * hcsz; i++)
            z[i] = (double)((i * 2654435761u) & 0xffff) / 65536.0 - 0.5;
    else
        for (size_t i = 0; i < hcsz; i++)
        {
            re[i] = (double)((i * 2654435761u) & 0xffff) / 65536.0 - 0.5;
            im[i] = (double)((i * 40503u) & 0xffff) / 65536.0 - 0.5;
        }
#define VFFT__C2R_ARM(P) do {                                   \
        if (as_z) vfft_c2r_disp_execute_z((P), z, y);           \
        else      vfft_c2r_disp_execute((P), re, im, y);        \
    } while (0)
    for (int w = 0; w < 5; w++)
    {
        VFFT__C2R_ARM(pn);
        VFFT__C2R_ARM(ps);
    }
    reps = (int)(2e6 / (double)(outsz + 1));
    if (reps < 20)
        reps = 20;
    if (reps > 100000)
        reps = 100000;
    for (r = 0; r < 9; r++)
    {
        vfft_c2r_disp_t *first = (r & 1) ? ps : pn;
        vfft_c2r_disp_t *second = (r & 1) ? pn : ps;
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            VFFT__C2R_ARM(first);
        double tf = (vfft_proto_now_ns() - t0) / reps;
        t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            VFFT__C2R_ARM(second);
        double tsc = (vfft_proto_now_ns() - t0) / reps;
        a[r] = (r & 1) ? tsc : tf;  /* natural */
        b[r] = (r & 1) ? tf : tsc;  /* split   */
    }
#undef VFFT__C2R_ARM
    vfft_proto_aligned_free(re);
    vfft_proto_aligned_free(im);
    vfft_proto_aligned_free(y);
    *n_nat = _il_ab_med9(a);
    *n_split = _il_ab_med9(b);
    return 0;
}

#endif /* VFFT_TRANSFORMS_REAL_REAL_ROUTE_RACE_H */
