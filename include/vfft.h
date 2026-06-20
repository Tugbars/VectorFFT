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

typedef struct {
    vfft_transform_t transform;
    vfft_placement_t placement;
    vfft_rigor_t     rigor;       /* sweep thoroughness on a wisdom miss/recalibrate */

    int    dims;                  /* 1 (default) or 2                          */
    int    n[2];                  /* n[0]=N for 1D; n[0]=N1, n[1]=N2 for 2D    */
    size_t howmany;               /* K — batch count (lane-batched: data[i*K+lane]) */

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
