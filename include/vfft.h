/* vfft.h — VectorFFT unified public API (the descriptor front door).
 *
 * One surface over every transform the library ships. Shape = MKL's descriptor
 * semantics (fill a config, create, execute, destroy) ∪ FFTW's planning-rigor
 * dial — minus MKL's DftiSetValue setter-soup (config struct instead) and minus
 * FFTW's ESTIMATE-that-might-be-wrong (only measured rigor tiers are exposed).
 *
 * The five axes a user selects (the data contract; nthreads/rigor are dials):
 *   1. transform   — c2c / r2c / c2r / DCT-I..IV / DST-I..III / DHT  (+ dims=2..4)
 *   2. placement   — in-place / out-of-place
 *   3. layout      — SPLIT planes vs INTERLEAVED complex (MKL's
 *                    DFTI_COMPLEX_STORAGE analog; r2c/c2r: split vs CCE spectrum)
 *   4. order       — DEFAULT / NATURAL / SCRAMBLED output-bin order (1D/2D c2c)
 *   5. nthreads    — core count      (+ rigor — calibration thoroughness dial)
 *
 * THE CONTRACT: the user shouldn't have to babysit interface selections — the
 * library arranges things properly, and misuse is refused LOUDLY. vfft_create
 * COMMITS the plan to one (transform x placement x layout x order) cell: each
 * cell is served natively, served through a documented internal conversion, or
 * REJECTED at create with an actionable stderr message (never a bare NULL for
 * a config-space mistake). vfft_execute then ASSERTS the pointer signature
 * matches the committed layout — a mismatched call prints an error and executes
 * NOTHING (no silent reinterpretation; the historical "sim==dim==NULL means
 * interleaved" inference is REMOVED).
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
extern "C"
{
#endif

  /* ════════════════════════════════════════════════════════════════════════
   * THE FOUR AXES
   * ════════════════════════════════════════════════════════════════════════ */

  typedef enum
  {
    VFFT_C2C, /* complex → complex            */
    VFFT_R2C,
    VFFT_C2R, /* real → complex / complex → real */
    VFFT_DCT1,
    VFFT_DCT2,
    VFFT_DCT3,
    VFFT_DCT4, /* REDFT00/10/01/11             */
    VFFT_DST1,
    VFFT_DST2,
    VFFT_DST3, /* RODFT00/10/01                */
    VFFT_DHT   /* discrete Hartley             */
  } vfft_transform_t;

  typedef enum
  {
    VFFT_INPLACE,
    VFFT_OUTOFPLACE
  } vfft_placement_t;

  /* Complex-data layout axis (config.layout; MKL DFTI_COMPLEX_STORAGE analog).
   * Chosen at CREATE; execute's pointer signature must match (see the buffer
   * table at vfft_execute). Zero-init == SPLIT == the historical default, so
   * memset-initialized configs are back-compatible.
   *   SPLIT       — separate re[]/im[] planes (the library's native layout).
   *   INTERLEAVED — one z[] buffer of adjacent (re,im) pairs. C2C: z in/out
   *                 (native folded z engines where they exist, internal
   *                 convert-around elsewhere — always correct, never silent).
   *                 R2C/C2R: the spectrum side is the packed CCE
   *                 (conjugate-even, MKL DFTI_COMPLEX_COMPLEX) z buffer.
   *                 Real->real transforms (DCT/DST/DHT) have no complex
   *                 layout: INTERLEAVED is rejected at create.
   *                 Not combinable with config.batch (padded planes are
   *                 split by construction). */
  typedef enum
  {
    VFFT_LAYOUT_SPLIT = 0, /* default: split re/im planes           */
    VFFT_LAYOUT_INTERLEAVED /* interleaved z (c2c) / CCE spectrum (r2c/c2r) */
  } vfft_layout_t;

  /* Calibration rigor — all MEASURED (FFTW flag analog in comments). A wisdom
   * HIT ignores this; it only governs the sweep run on a MISS (or recalibrate). */
  typedef enum
  {
    VFFT_MEASURE,   /* ≈ FFTW_MEASURE   — DP-default / variant-aware coarse  */
    VFFT_PATIENT,   /* ≈ FFTW_PATIENT   — DP patient / patient-exhaustive    */
    VFFT_EXHAUSTIVE /* ≈ FFTW_EXHAUSTIVE— full multiset × permutation        */
                    /* VFFT_ESTIMATE — planned 4th tier (V4 cost model, no measurement)         */
  } vfft_rigor_t;

  typedef enum
  {
    VFFT_FORWARD,
    VFFT_BACKWARD
  } vfft_dir_t;

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

  typedef struct vfft_wisdom_s vfft_wisdom; /* opaque */

  vfft_wisdom *vfft_wisdom_load(const char *path); /* caller-owned override   */
  int vfft_wisdom_save(const vfft_wisdom *w, const char *path);
  void vfft_wisdom_free(vfft_wisdom *w);

  /* ════════════════════════════════════════════════════════════════════════
   * DESCRIPTOR + PLAN
   * ════════════════════════════════════════════════════════════════════════ */

  /* Opaque VW-padded batch handle (full contract below, near vfft_alloc_batch).
   * Declared here so vfft_config_t can carry it as the opt-in padding signal. */
  typedef struct vfft_batch_s *vfft_batch;

  typedef struct
  {
    vfft_transform_t transform;
    vfft_placement_t placement;
    vfft_rigor_t rigor; /* sweep thoroughness on a wisdom miss/recalibrate */

    int dims;         /* 1 (default), 2, or 3                      */
    int n[3];         /* n[0]=N (1D); {N1,N2} (2D); {N1,N2,N3} (3D).
                         3D: C2C + R2C/C2R (§6a47), howmany==1, order DEFAULT/
                         SCRAMBLED (natural is a follow-up); plans
                         carry a dedicated (N1,N2,N3) wisdom table
                         inside the same vfft_wisdom bundle.       */
    size_t howmany;   /* K — batch count (lane-batched: data[i*K+lane]) */
    vfft_batch batch; /* NULL = tight (default drop-in path). Non-NULL = the
                         opt-in padded batch to run on: the plan is built at its
                         Kp stride and runs the padded fast path. MUST come from
                         vfft_alloc_batch_for(this config) — create cross-checks
                         transform/placement-shape/N/K totally and rejects any
                         mismatch loudly. 1D C2C (both placements) + R2C/C2R +
                         TRIG; layout SPLIT only — padding_design_decision.md. */

    int nthreads; /* 0 = use the current pool / single-thread  */

    int order; /* Output-order axis for 1D C2C (the MKL DFTI_ORDERING knob).
                  VFFT_ORDER_DEFAULT (0) = engine-native = fastest, order-
                    agnostic (in-place: digit-scrambled; OOP: whichever kind
                    wins calibration — may be MODEB/scrambled or LEAF/BAILEY2).
                  VFFT_ORDER_SCRAMBLED = force the scrambled/fast path (in-place:
                    native, == DEFAULT; OOP: the MODEB kind). Explicit "I am
                    order-agnostic" — MKL's DFTI_BACKWARD_SCRAMBLED intent.
                  VFFT_ORDER_NATURAL = spectrum in natural bin order, bin-for-bin
                    MKL/FFTW-comparable (in-place: PURE/PSWAP reorder, per-cell
                    measured verdict in wisdom; OOP: the LEAF/BAILEY2 kinds).
                  1D C2C only (in-place + OOP); r2c/c2r/trig are already natural,
                  2D not wired — natural_order_inplace_design.md §2e.
                  Roundtrip/convolution consumers should keep DEFAULT (order is
                  irrelevant there, and it is the fastest).                     */

    vfft_layout_t layout; /* Complex-data layout axis (see vfft_layout_t above).
                             Committed at create; execute enforces the matching
                             pointer signature. Default (0) = SPLIT.            */

    vfft_wisdom *wisdom; /* NULL = library-managed (auto load+save);
                            non-NULL = use this, ignore the default   */
    int recalibrate;     /* 0 = use existing entry; 1 = re-measure + overwrite */
  } vfft_config_t;

  /* Output-order axis (vfft_config_t.order). 1D C2C: DEFAULT=fastest/native, SCRAMBLED=force the
   * scrambled path (in-place native / OOP MODEB), NATURAL=force natural (in-place PURE/PSWAP / OOP
   * LEAF/BAILEY2). Values map 1:1 onto the internal OOP kind constraint (0=any,1=natural,2=scrambled). */
  enum
  {
    VFFT_ORDER_DEFAULT = 0,
    VFFT_ORDER_NATURAL = 1,
    VFFT_ORDER_SCRAMBLED = 2
  };

  typedef struct vfft_plan_s *vfft_plan; /* opaque execute-ready handle */

  /* Build (+ calibrate-on-miss at config.rigor). NULL on failure. */
  vfft_plan vfft_create(const vfft_config_t *config);

  /* ════════════════════════════════════════════════════════════════════════
   * EXECUTE  (one entry, all transforms + placements + layouts)
   *
   * The pointer signature is DICTATED by the plan's committed layout — execute
   * checks it and, on a mismatch, prints an error to stderr and computes
   * NOTHING (never a silent reinterpretation, never garbage). Lane-batched
   * addressing: split planes hold element e of lane t at [e*K + t]; an
   * interleaved z buffer holds it at z[2*(e*K + t)] (+1 = imaginary).
   *
   * SIGNATURE TABLE (per transform x layout; padded batches: same roles, use
   * vfft_batch_planes() to fill them):
   *
   *   transform  layout       sre        sim        dre        dim
   *   ---------  -----------  ---------  ---------  ---------  ---------
   *   C2C        SPLIT        in.re      in.im      out.re     out.im
   *                           in-place plans: pass dre==sre && dim==sim, or
   *                           dre==dim==NULL (result stays in sre/sim).
   *                           out-of-place plans: all four non-NULL.
   *   C2C        INTERLEAVED  z_in       NULL       z_out      NULL
   *                           dre may equal sre (in-place) or not.
   *   R2C (fwd)  SPLIT        real_in    NULL       out.re     out.im
   *   R2C (fwd)  INTERLEAVED  real_in    NULL       z_CCE_out  NULL
   *                           dre = packed CCE half-spectrum, (N/2+1)*K pairs
   *                           at dre[2*(f*K+t)] (§6a24).
   *   C2R (bwd)  SPLIT        in.re      in.im      real_out   NULL
   *   C2R (bwd)  INTERLEAVED  z_CCE_in   NULL       real_out   NULL
   *                           sre = the CCE spectrum (same packing as R2C out).
   *   DCT/DST/DHT (SPLIT)     real_in    NULL       real_out   NULL
   *                           real->real; INTERLEAVED rejected at create.
   *
   * SUPPORT MATRIX (create commits; NATIVE = engine path, CONVERT = internal
   * layout conversion around the split engines — correct, documented cost;
   * REJECT = loud create-time refusal):
   *
   *   1D C2C in-place   x INTERLEAVED: NATIVE folded z->z (order DEFAULT/
   *       SCRAMBLED, single-thread, >=2-stage plans); NATIVE il2il slab under
   *       MT; everything else (NATURAL / prime overrides / resolver misses)
   *       CONVERTs. config.batch + INTERLEAVED: REJECT.
   *   1D C2C OOP        x INTERLEAVED: NATIVE z->z for K=1 (SCRAMBLED: the
   *       block-split cascade at covered N; DEFAULT/NATURAL: the K=1 engine's
   *       IL routes where emitted); every other OOP cell (K>1, uncovered N,
   *       no IL route) CONVERTs around the split champions — the historical
   *       silent no-op / crash cells are GONE.
   *   R2C/C2R           x INTERLEAVED: NATIVE CCE executors (1D + 2D §6a30 +
   *       3D/4D §6a47). Placement must be OUT-OF-PLACE (in-place real FFT is
   *       REJECTED loudly until an MKL-style in-place CCE path exists).
   *   2D..4D C2C        x INTERLEAVED: CONVERT-around (§6a61), both
   *       placements, 2D NATURAL included.
   *   TRIG              x INTERLEAVED: REJECT (no complex layout).
   *   any batch         x INTERLEAVED: REJECT (padded planes are split).
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
   * The batch is BORN FROM THE CONFIG: vfft_alloc_batch_for(cfg) reads the
   * transform/placement/N/howmany it needs and allocates every plane that
   * (transform x placement) requires at the right padded stride Kp, ZEROED:
   *
   *   C2C in-place:    split data planes (re+im), N*Kp each; Kp=roundup(K,4).
   *                    The planner runs me=Kp (full-SIMD) or me=K (tail) per
   *                    the padded wisdom's exec_me.
   *   C2C out-of-place: 4 split planes (input re/im + output re/im), N*Kp
   *                    each; Kp=roundup(K,8) (OOP kinds + wisdom 8-alignment).
   *                    PAD-ONLY (OOP bakes K).
   *   R2C:             real INPUT (N*Kp) + split spectrum OUTPUT
   *                    ((N/2+1)*Kp each). PAD-ONLY; even N required.
   *   C2R:             split spectrum INPUT + real OUTPUT (mirror of R2C).
   *   TRIG (DCT/DST/DHT): real INPUT + real OUTPUT, N*Kp each — the ONLY
   *                    full-SIMD path for misaligned trig K (no odd-K tail).
   *
   * The handle is OPAQUE and self-describing (transform, placement, N, K, Kp)
   * so a padded buffer cannot be mistaken for a tight one, and vfft_create
   * cross-checks the handle against the config TOTALLY (transform, placement
   * shape, N, K — any mismatch is a loud create-time rejection).
   * config.layout must be SPLIT (padded planes are split by construction).
   *
   * To USE it: build the config, b = vfft_alloc_batch_for(&cfg), set
   * cfg.batch = b, vfft_create, then let vfft_batch_planes() fill the execute
   * arguments IN THE RIGHT ROLES — the role table lives inside the library:
   *
   *   double *sre, *sim, *dre, *dim;
   *   vfft_batch_planes(b, &sre, &sim, &dre, &dim);
   *   vfft_execute(plan, dir, sre, sim, dre, dim);
   *
   * (c2c in-place: sre/dre = re plane, sim/dim = im plane; c2c OOP: input
   * planes -> sre/sim, output planes -> dre/dim; r2c: sre=real, dre/dim =
   * spectrum; c2r: sre/sim = spectrum, dre=real, dim=NULL; trig: sre=real-in,
   * dre=real-out, sim=dim=NULL. Roles are the FORWARD data flow — an OOP
   * roundtrip may legally pass the planes swapped for the inverse leg.)
   * Element e of lane t sits at plane[e*Kp + t], Kp = vfft_batch_stride(b).
   * Match every alloc with vfft_free_batch (never free planes yourself).
   * ════════════════════════════════════════════════════════════════════════ */

  /* ONE allocator for every padded transform; the Kp rule (roundup(K,4) vs
   * OOP's roundup(K,8)) is internal. Reads cfg->transform/placement/n[0]/
   * howmany; cfg->batch is ignored. NULL + stderr message on misuse. */
  vfft_batch vfft_alloc_batch_for(const vfft_config_t *cfg);
  /* Fill the vfft_execute arguments in the right roles (any out-param may be
   * NULL if unwanted; planes the transform does not use are set to NULL). */
  void vfft_batch_planes(vfft_batch b, double **sre, double **sim,
                         double **dre, double **dim);
  size_t vfft_batch_stride(vfft_batch b); /* = Kp (0 for NULL handle) */
  void vfft_free_batch(vfft_batch b);     /* matching free */

  /* ════════════════════════════════════════════════════════════════════════
   * GLOBAL CONTROL  (optional; sensible defaults otherwise)
   * ════════════════════════════════════════════════════════════════════════ */

  void vfft_set_num_threads(int n); /* size the pool; pin caller to core 0 */
  int vfft_get_num_threads(void);
  const char *vfft_isa(void); /* "avx512" | "avx2" | "scalar"        */
  const char *vfft_version(void);

#ifdef __cplusplus
}
#endif
#endif /* VFFT_H */
