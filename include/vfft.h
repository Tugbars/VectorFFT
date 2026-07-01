/* vfft.h — VectorFFT unified public API (the descriptor front door).
 *
 * One surface over every transform the library ships. Shape = MKL's descriptor
 * semantics (fill a config, create, execute, destroy) ∪ FFTW's planning-rigor
 * dial — minus MKL's DftiSetValue setter-soup (config struct instead) and minus
 * FFTW's ESTIMATE-that-might-be-wrong (only measured rigor tiers are exposed).
 *
 * The four axes a user selects:
 *   1. transform   — c2c / r2c / c2r / DCT-I..IV / DST-I..III / DHT  (+ dims=2 for 2D)
 *   2. placement   — in-place / out-of-place
 *   3. nthreads    — core count
 *   4. rigor       — MEASURE / PATIENT / EXHAUSTIVE  (calibration thoroughness)
 *
 * Everything behind this — the per-feature dispatchers, the wisdom files, the
 * plan search engines, JIT — is implementation detail reached through vfft_create.
 *
 * STATUS: surface skeleton. vfft_create's dispatch-by-transform + the rigor→sweep
 * wiring is the implementation workstream (productionizes planning/plan_orchestrator.h
 * from c2c+primes to all features). Estimate is a planned 4th rigor tier (slots in
 * once its cost model is re-homed; no surface change).
 */
#ifndef VFFT_H
#define VFFT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ════════════════════════════════════════════════════════════════════════
 * THE FOUR AXES
 * ════════════════════════════════════════════════════════════════════════ */

typedef enum {
    VFFT_C2C,                                  /* complex → complex            */
    VFFT_R2C, VFFT_C2R,                        /* real → complex / complex → real */
    VFFT_DCT1, VFFT_DCT2, VFFT_DCT3, VFFT_DCT4,/* REDFT00/10/01/11             */
    VFFT_DST1, VFFT_DST2, VFFT_DST3,           /* RODFT00/10/01                */
    VFFT_DHT                                   /* discrete Hartley             */
} vfft_transform_t;

typedef enum { VFFT_INPLACE, VFFT_OUTOFPLACE } vfft_placement_t;

/* Calibration rigor — all MEASURED (FFTW flag analog in comments). A wisdom
 * HIT ignores this; it only governs the sweep run on a MISS (or recalibrate). */
typedef enum {
    VFFT_MEASURE,      /* ≈ FFTW_MEASURE   — DP-default / variant-aware coarse  */
    VFFT_PATIENT,      /* ≈ FFTW_PATIENT   — DP patient / patient-exhaustive    */
    VFFT_EXHAUSTIVE    /* ≈ FFTW_EXHAUSTIVE— full multiset × permutation        */
    /* VFFT_ESTIMATE — planned 4th tier (V4 cost model, no measurement)         */
} vfft_rigor_t;

typedef enum { VFFT_FORWARD, VFFT_BACKWARD } vfft_dir_t;

/* ════════════════════════════════════════════════════════════════════════
 * WISDOM  (calibrated plans, persisted per feature)
 *
 * Default (config.wisdom == NULL): the library auto-loads the per-feature
 * wisdom from its generated folder, and on a MISS calibrates at config.rigor,
 * adds the entry, and persists it — so it learns across runs automatically.
 *
 * Override (config.wisdom != NULL): the library uses THAT table exclusively and
 * ignores the generated-folder default. The caller owns it (load/save/free).
 *
 * Overwrite: config.recalibrate = 1 re-measures and overwrites the cell even on
 * a hit (else an existing entry is used as-is / only missing cells are filled).
 * ════════════════════════════════════════════════════════════════════════ */

typedef struct vfft_wisdom_s vfft_wisdom;   /* opaque */

vfft_wisdom *vfft_wisdom_load(const char *path);                 /* caller-owned override   */
int          vfft_wisdom_save(const vfft_wisdom *w, const char *path);
void         vfft_wisdom_free(vfft_wisdom *w);

/* ════════════════════════════════════════════════════════════════════════
 * DESCRIPTOR + PLAN
 * ════════════════════════════════════════════════════════════════════════ */

/* Opaque VW-padded batch handle (full contract below, near vfft_alloc_batch).
 * Declared here so vfft_config_t can carry it as the opt-in padding signal. */
typedef struct vfft_batch_s *vfft_batch;

typedef struct {
    vfft_transform_t transform;
    vfft_placement_t placement;
    vfft_rigor_t     rigor;       /* sweep thoroughness on a wisdom miss/recalibrate */

    int    dims;                  /* 1 (default) or 2                          */
    int    n[2];                  /* n[0]=N for 1D; n[0]=N1, n[1]=N2 for 2D    */
    size_t howmany;               /* K — batch count (lane-batched: data[i*K+lane]) */
    vfft_batch batch;             /* NULL = tight (default drop-in path). Non-NULL = the
                                     opt-in VW-padded batch to run on: the plan is built at
                                     its Kp stride and runs the padded wisdom's exec_me (Kp
                                     full-SIMD, else me=K tail). Must match howmany + n[0].
                                     C2C in-place only for now — padding_design_decision.md. */

    int    nthreads;              /* 0 = use the current pool / single-thread  */

    vfft_wisdom *wisdom;          /* NULL = library-managed (auto load+save);
                                     non-NULL = use this, ignore the default   */
    int    recalibrate;           /* 0 = use existing entry; 1 = re-measure + overwrite */
} vfft_config_t;

typedef struct vfft_plan_s *vfft_plan;   /* opaque execute-ready handle */

/* Build (+ calibrate-on-miss at config.rigor). NULL on failure. */
vfft_plan vfft_create(const vfft_config_t *config);

/* ════════════════════════════════════════════════════════════════════════
 * EXECUTE  (one entry, all transforms + placements)
 *
 * Split-complex, lane-batched buffers (element e of transform t at [e*K + t]).
 * In-place: pass dre==sre, dim==sim. Unused halves (real domains): pass NULL.
 *
 *   transform            sre        sim        dre        dim
 *   ---------            ---        ---        ---        ---
 *   C2C   fwd/bwd        in.re      in.im      out.re     out.im
 *   R2C   (fwd)          real_in    NULL       out.re     out.im
 *   C2R   (bwd)          in.re      in.im      real_out   NULL
 *   DCT/DST/DHT          real_in    NULL       real_out   NULL
 *
 * `dir` selects forward vs the (unnormalized) inverse; for self-inverse trig
 * (DCT-I/IV, DST-I, DHT) the two coincide. ════════════════════════════════ */
void vfft_execute(vfft_plan p, vfft_dir_t dir,
                  double *sre, double *sim, double *dre, double *dim);

void vfft_destroy(vfft_plan p);

/* ════════════════════════════════════════════════════════════════════════
 * PADDED BATCH  (opt-in fast path for odd K — docs/roadmap/tail_handling/
 *               padding_design_decision.md)
 *
 * A VW-padded batch: the library allocates the batch at stride Kp = roundup(K,VW)
 * with the (Kp-K) pad columns ZEROED, and hands it back as an OPAQUE handle that
 * carries its own stride — so a padded buffer cannot be mistaken for a tight one.
 * Fill/read the K real lanes through vfft_batch_re/im at the physical stride
 * vfft_batch_stride() (= Kp): element e of transform t is at re[e*Kp + t].
 *
 * On such a buffer the planner may run me=Kp (pure full-SIMD, junk lanes discarded)
 * OR me=K (SSE2/scalar tail) — whichever the padded wisdom's exec_me picked — both
 * correct for the K real transforms. To USE it: allocate here, set config.batch to this
 * handle (+ config.howmany = K, config.n[0] = N), vfft_create, then vfft_execute on
 * vfft_batch_re/im. Wired for C2C in-place; other features fall back to the tight path.
 * (vfft_batch itself is typedef'd up by the config struct.) Match alloc with free.
 * ════════════════════════════════════════════════════════════════════════ */

vfft_batch vfft_alloc_batch(int N, size_t K); /* Kp=roundup(K,VW)-wide, ZEROED re+im; NULL on failure */
void       vfft_free_batch(vfft_batch b);     /* matching free (do NOT free the re/im pointers yourself) */
double    *vfft_batch_re(vfft_batch b);       /* K real lanes at stride vfft_batch_stride() */
double    *vfft_batch_im(vfft_batch b);
size_t     vfft_batch_stride(vfft_batch b);   /* = Kp (roundup(K,VW)) */

/* ════════════════════════════════════════════════════════════════════════
 * GLOBAL CONTROL  (optional; sensible defaults otherwise)
 * ════════════════════════════════════════════════════════════════════════ */

void        vfft_set_num_threads(int n);   /* size the pool; pin caller to core 0 */
int         vfft_get_num_threads(void);
const char *vfft_isa(void);                /* "avx512" | "avx2" | "scalar"        */
const char *vfft_version(void);

#ifdef __cplusplus
}
#endif
#endif /* VFFT_H */
