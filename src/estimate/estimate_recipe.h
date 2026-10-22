/* estimate_recipe.h -- the 1D estimate's REAL-TRANSFORM recipe and this folder's contracts
 * (tracked, not wired into the library).
 *
 * THE COMPLEX PART IS THE LIBRARY'S (released 2026-10-22 as VFFT_FLAG_ESTIMATE, 1D c2c only):
 *   src/core/il/planning/estimate_il.h  the machine (est_machine_t), the request (est_request_t), the
 *                                       c2c recipe (est_recipe_t), the c2c formulas (est_plan_1d) and
 *                                       the c2c row writer (est_put_rows), plus every helper the real
 *                                       formulas below share (est_c2c, the ZTURN-T / pair / four-step
 *                                       / prime-inner helpers, est_fs_child, est__fs_text ...);
 *   src/core/il/rank1/c2c_estimate_il.h the c2c build (est_build_c2c: pre-flight, private store, the
 *                                       door's replay, the detectors est__meter_read / est__replay,
 *                                       est__fs_preflight) and the front door's _vfft_create_estimate.
 * This folder adds r2c / c2r on top of it -- the real formulas (estimate_1d.h), the real build
 * (estimate_build.h) -- and the tools. Nothing here is part of the library or its build.
 *
 * The recipe of ANY request (c2c or real) is est_any_recipe_t: the library's c2c recipe as its member
 * `c`, plus the real engines' fields. For a c2c request `c` is the whole recipe; for a zr2c real
 * request it is the c2c child at N/2; zfsr's split child and zrb's four-step inner use `c.fs`.
 *
 * Specification: docs/design/estimate_1d_sketch.md. Paths are relative to src/core/ unless they
 * start with include/ or docs/. Zero means "absent" unless a comment says otherwise.
 *
 * NEEDS THE LIBRARY'S TYPES: include only after #include "vfft.c" (the tools) -- the TU then has the
 * tracked headers above in scope. */
#ifndef EST_RECIPE_H
#define EST_RECIPE_H

#ifndef VFFT_IL_C2C_ESTIMATE_IL_H
#error "src/estimate needs the library's estimate (il/rank1/c2c_estimate_il.h): #include \"vfft.c\" first"
#endif

#include <stddef.h>

/* ---------------------------------------------------------------- the recipe of any request */
typedef struct est_any_recipe_s
{
    char eng[8];          /* [eng] the row's engine: "k1" for c2c; zrm | zrp | zttr | zr2c | zfsr | zrf | zrb */
    est_recipe_t c;       /* c2c: the recipe (est_plan_1d). zr2c: the child at N/2, natural, T = 1, mapped
                           * 1:1 onto vw2_zr2c_child_t (wisdom2_real_il.h:388-406; c.il_mt = the prime
                           * child's threaded arm). zfsr / zrb kind 4: c.fs holds the four-step child. */

    /* ---- real: one row per (r2c|c2r, N, placement, T), ord=nat (wisdom2_real_il.h:1-44) ------- */
    struct { int R1, R2, form; } zrp;    /* [pair=R1.R2 leaf=n1t|r2z] form 0 A (n1t), 1 B (r2z);
                                          * 4 lanes only (sketch 6)                             */
    struct { int chain[8], nf; size_t tile; int stk, mt; } zttr;
                                         /* [chain= tile= stk= mt=] the chain at M = N/2, 2..8
                                          * stages; stk = 3; mt 0 = absent; 4 lanes only         */
    struct { int route, fold_mt; } zr2c; /* [route=child_oop_il|child_nat_ip fold=st|mt] route =
                                          * the CHILD's placement, 0 child_oop_il, 1 child_nat_ip;
                                          * fold_mt only on a T > 1 row                          */
    struct { int n1, n2; } zfsr;         /* [split=n1xn2] the split of N/2; its child = c.fs      */
    struct { int chain[10], K, nomsz, tile, mt; } zrf;
                                         /* [chain= msz= tile= mt=] odd radices ascending, 2..10
                                          * (zrf.h:228); nomsz = !msz; tile = the budget in
                                          * complex, 0 = untiled; mt 2 LEVELS on a T > 1 row     */
    struct { int M; char kind[8], shape[64]; int tw, mt; } zrb;
                                         /* [m= in= in_sh= in_tw= mt=] the convolution length and
                                          * its inner, the prime row's spelling; a kind-4 inner's
                                          * child goes in c.fs; mt 0 = serial                    */
} est_any_recipe_t;

/* ---------------------------------------------------------------- the contracts */

/* est_plan_any: the formulas for any request. c2c: the library's est_plan_1d into out->c (eng "k1").
 * r2c / c2r: the real formulas (estimate_1d.h, sketch section 6 and 7). Pure. Returns 0, or -1 with
 * the reason (the library's refusals, plus: a scrambled real request; real even N with N/2 > 2^24;
 * an odd prime cell with (3N-1)/2 > 2^24; zfsr at T > 1, a gap of the sketch reported, not filled). */
static inline int est_plan_any(const est_machine_t *m, const est_request_t *rq, est_any_recipe_t *out, char *why,
                               size_t whylen);

/* est_build_any / est_build_any_ex: any recipe -> a public plan built with no measurement, or NULL with
 * the reason (rep may be NULL). c2c goes through the LIBRARY's build (est_build_c2c) on the front door's
 * translation of the request; real through this folder's real build, which uses the library's
 * detectors (est__meter_read, est__replay). Both refuse when a create-steering VFFT_* variable is set
 * (REPLAY_API 1.3), count the library's clock reads (the tool's VFFT_CLOCK_PROBE: a read means the
 * create measured, EST_MEASURED), and arm the caller's pin of a T > 1 plan as vfft_create does
 * (vfft_api.h). */
static inline struct vfft_plan_s *est_build_any_ex(const est_any_recipe_t *rc, const est_request_t *rq, char *why,
                                                   size_t whylen, est_build_report_t *rep);
static inline struct vfft_plan_s *est_build_any(const est_any_recipe_t *rc, const est_request_t *rq, char *why,
                                                size_t whylen);

/* est_recipe_print: the recipe in the store's own text, one line per row the build banks:
 * "<key> | <payload>", no level-2 tail (ran, ns, src, date: not recipe).
 * - c2c key:  t=c2c n=N q=1 ord=nat|scr place=oop|ip role=comp lay=il [nthreads=T]; payload eng=k1
 *   il_route=.. and the route's tokens (T > 1: il_mt il_mt_tw il_mtsb; FS: fs_*). A prime cell's
 *   method row comes first; a pair / chain3 with backward forms adds the dir=bwd row.
 * - real key: t=r2c|c2r n=N q=1 ord=nat place=oop|ip [nthreads=T] (vw2_real_il_key,
 *   wisdom2_real_il.h:60-68); payload eng=.. and the engine's tokens.
 * Returns the length it wrote, or would have written (snprintf's contract); buf is always
 * NUL-terminated when cap > 0. */
static inline int est_recipe_print(const est_any_recipe_t *rc, const est_request_t *rq, char *buf, size_t cap);

#endif /* EST_RECIPE_H */
