/* real_role.h -- THE ROLE OF THE REAL DOOR'S RACES (owner, 2026-11-14: "the OOO engine makes everything non
 * predictable").
 *
 * A plan that runs a 1D real FFT between passes of its own (il/real/dst.h: a reorder pass and a rotation pass
 * around an R2C or a C2R at N; il/real/dct1_dst1.h: a pass that extends the input and a pass that takes the
 * outputs, around an R2C of twice the length) picks that FFT by what the WHOLE plan costs, not by what the FFT
 * costs alone:
 * inside the plan the FFT's input was written a moment ago by one pass and its output is read at once by the
 * other. Such a plan sets the role around the create of its FFT; every timed run of a candidate in the real
 * door's races is then one run of the plan's whole pass with that candidate inside:
 *   - the even door's sweeps (ZTT-r's chains and tiles, the real four-step's splits), zr2c's two placements
 *     of its child, and the engine race among their winners (il/real/zr2c_build.h, zrp_build.h);
 *   - the odd door's sweeps (the flat chain, the real Bluestein) and its engine race (il/real/odd_build.h).
 * The checks against the reference before a candidate is timed run the transform alone, as do the races
 * below these: zr2c's complex child (raced in zr2c's own role, the child and the fold) and the threaded arms'
 * own races.
 *
 * The role names the transform and N it was set for, so a create nested inside the FFT's own (a child of
 * another length, another transform) races its own way. One process static: creates take turns (the create
 * lock, vfft.c). No role set = the transform alone, every race as it was. */
#ifndef VFFT_IL_REAL_ROLE_H
#define VFFT_IL_REAL_ROLE_H

#include "vfft_internal.h" /* struct vfft_plan_s */

typedef struct
{
    void (*run)(void *ctx, struct vfft_plan_s *h);   /* one run of the plan's whole pass, the candidate h inside */
    void *ctx;
    int transform;                                   /* VFFT_R2C or VFFT_C2R: the FFT the role was set for */
    int N;
} _real_il_role_t;
static _real_il_role_t _real_il_role;

/* 1 = a role is set and h is a candidate for its FFT */
static inline int _real_il_in_role(const struct vfft_plan_s *h)
{
    return _real_il_role.run && (int)h->transform == _real_il_role.transform && h->N == _real_il_role.N &&
           h->N2 == 0 && h->K <= 1;
}
/* one timed run of the candidate h: the plan's whole pass when h races in a role (1), else nothing done (0)
 * and the caller runs the transform alone */
static inline int _real_il_role_run(struct vfft_plan_s *h)
{
    if (!_real_il_in_role(h))
        return 0;
    _real_il_role.run(_real_il_role.ctx, h);
    return 1;
}

#endif /* VFFT_IL_REAL_ROLE_H */
