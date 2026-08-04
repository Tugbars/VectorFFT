/* vfft.h — VectorFFT public API.
 *
 * ── QUICK START ─────────────────────────────────────────────────────────────
 *   vfft_config_t cfg = {0};                  // zeroed = sensible defaults
 *   cfg.transform = VFFT_C2C;
 *   cfg.n[0]      = 4096;
 *   cfg.howmany   = 8;                        // batch: 8 transforms at once
 *   cfg.placement = VFFT_OUTOFPLACE;
 *   cfg.layout    = VFFT_LAYOUT_SPLIT;        // re[] / im[] planes (default)
 *   vfft_plan p = vfft_create(&cfg);          // plans; calibrates on first use
 *   vfft_execute(p, VFFT_FORWARD, in_re, in_im, out_re, out_im);
 *   vfft_destroy(p);
 *
 * ── WHAT A PLAN IS ──────────────────────────────────────────────────────────
 * vfft_create commits your config to one concrete engine + factorization,
 * chosen by MEASUREMENT on your machine (persisted in wisdom files; the first
 * create of a new size calibrates at config.rigor, later creates are instant
 * hits). You never pick an algorithm — you state the data contract, the
 * library arranges the rest, and misuse is refused LOUDLY (see ERRORS).
 *
 * ── WHAT EACH TRANSFORM SUPPORTS (at a glance) ──────────────────────────────
 *   transform    dims    placement       layout             order            howmany  MT   padded batch
 *   C2C          1–4     IP + OOP        SPLIT + INTERLVD*  DEF/SCR/NAT**    any K*** yes  IP + OOP
 *   R2C          1–4     OOP (IP: refused for now)  split | CCE spectrum  natural  any K  yes  pad-only
 *   C2R          1–4     OOP (IP: refused for now)  split | CCE spectrum  natural  any K  yes  pad-only
 *   DCT-I..IV    1       IP + OOP        real (layout n/a)  natural          any K    yes  pad-only
 *   DST-I..III   1       IP + OOP        real (layout n/a)  natural          any K    yes  pad-only
 *   DHT          1       IP + OOP        real (layout n/a)  natural          any K    yes  pad-only
 *
 *   *   INTERLEAVED is native for 1D C2C (in-place folded z engine; OOP K=1
 *       SCRAMBLED cascade). 2D+ C2C and 2D r2c/c2r serve it via a documented
 *       internal conversion. Other cells are PLANNED — create refuses them
 *       with "not yet implemented (currently …)" so you know to wait, not
 *       rethink. FOR NOW: treat interleaved as a 1D-C2C-first feature.
 *   **  order is a 1D/2D C2C axis (2D NATURAL included). 3D/4D: DEFAULT or
 *       SCRAMBLED only, K must be 1. r2c/c2r/trig outputs are always natural.
 *   *** dims>=2 require howmany==1. DCT-I is present but not yet validated.
 *       2D prime dims are refused at create (not production-safe yet).
 *       Odd 1D sizes are fully supported (odd radixes, odd-K tails); prime /
 *       Rader/Bluestein-class 1D sizes are IN-PLACE only — out-of-place C2C
 *       refuses them loudly (use placement=VFFT_INPLACE).
 *
 * ── YOUR BUFFERS, PER LAYOUT (what to pass to vfft_execute) ─────────────────
 *   layout / transform      sre         sim         dre         dim
 *   SPLIT   C2C             in.re       in.im       out.re      out.im
 *   INTERLV C2C             z_in        NULL        z_out       NULL
 *           (z = interleaved pairs. By DEFAULT the batch is
 *            transform-contiguous: transform t is the block
 *            [2*t*N .. 2*(t+1)*N), the MKL/FFTW idiom. Set
 *            config.batch_geom = VFFT_BATCH_LANE_MAJOR for the split
 *            engines' geometry instead, element e of lane t at
 *            [2*(e*K+t)]. Both are identical at K==1. In-place: pass
 *            dre == sre (dre is required). sim/dim MUST be NULL — the
 *            plan was committed to this layout and execute checks the
 *            signature.)
 *   SPLIT   R2C fwd         real_in     NULL        spec.re     spec.im
 *   INTERLV R2C fwd (CCE)   real_in     NULL        z_spec      NULL
 *   SPLIT   C2R bwd         spec.re     spec.im     real_out    NULL
 *   INTERLV C2R bwd (CCE)   z_spec      NULL        real_out    NULL
 *   —       DCT/DST/DHT     real_in     NULL        real_out    NULL
 *   Split element e of lane t lives at [e*K + t]. CCE spectrum: (N/2+1)
 *   interleaved pairs. Backward transforms are unnormalized (scale by 1/N
 *   yourself after a roundtrip).
 *
 * ── LETTING THE PLAN OWN YOUR BUFFERS (optional, for ANY K) ─────────────────
 *   cfg.owned_buffers = 1;                           // any 1D transform
 *   vfft_plan p = vfft_create(&cfg);                 // allocates the planes too
 *   double *sre,*sim,*dre,*dim;
 *   vfft_plan_planes(p, &sre,&sim,&dre,&dim);        // execute's args, in role order
 *   size_t st = vfft_plan_stride(p);                 // index YOUR data at [e*st + t]
 *   ... fill inputs ...; vfft_execute(p, dir, sre,sim,dre,dim);
 *   vfft_destroy(p);                                 // frees the planes too
 *   You never reason about SIMD lane counts or odd K: the library measures
 *   whether padding pays for your (N,K) and sizes the planes accordingly.
 *   ALWAYS index with vfft_plan_stride(p) — it may equal K (tight) or a padded
 *   width, and assuming roundup() will run off the end of a tight buffer. The
 *   first create of a new cell may pause to measure; the verdict is persisted,
 *   so later ones are instant. Split layout, 1D only.
 *   Leave owned_buffers at 0 (the default) to pass your own tight buffers.
 *
 * ── WISDOM (performance persistence) ────────────────────────────────────────
 *   Default: auto-loaded per machine; misses calibrate at config.rigor
 *   (MEASURE / PATIENT / EXHAUSTIVE — all measured, never estimated) and are
 *   saved, so the library learns across runs. Override with config.wisdom
 *   (vfft_wisdom_load/save/free); force re-measurement with recalibrate=1.
 *
 * ── ERRORS ──────────────────────────────────────────────────────────────────
 *   vfft_create returns NULL only AFTER printing an actionable message:
 *   either "not yet implemented (currently …)" (planned cell — wait) or why
 *   the combination is invalid (rethink). vfft_execute validates the pointer
 *   signature against the plan's committed layout and REFUSES a mismatch —
 *   it never reinterprets your buffers, and it never computes silently wrong.
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

  typedef struct
  {
    vfft_transform_t transform;
    vfft_placement_t placement;
    vfft_rigor_t rigor; /* sweep thoroughness on a wisdom miss/recalibrate */

    int dims;         /* 1 (default), 2, 3, or 4                   */
    int n[4];         /* n[0]=N (1D); {N1,N2} (2D); {N1,N2,N3} (3D);
                         {N1,N2,N3,N4} (4D §6a62).
                         3D: C2C + R2C/C2R (§6a47), howmany==1, order DEFAULT/
                         SCRAMBLED (natural is a follow-up); plans
                         carry a dedicated (N1,N2,N3) wisdom table
                         inside the same vfft_wisdom bundle.
                         4D: same contracts (K==1, order DEFAULT/SCRAMBLED;
                         real transforms out-of-place with even N4).  */
    size_t howmany;   /* K — batch count (lane-batched: data[i*K+lane]) */
    int owned_buffers; /* 0 (default) = YOU own the buffers: pass your own tight
                          planes to vfft_execute, indexed at [e*K + t]. This is
                          the drop-in path and allocates nothing extra.
                          1 = THE LIBRARY owns them: vfft_create allocates every
                          plane this (transform x placement) needs, CHOOSES the
                          stride (measured pad-vs-tight — it may be K or padded),
                          and frees them in vfft_destroy. Read them back with
                          vfft_plan_planes() and vfft_plan_stride().
                          1D only, layout SPLIT only; create refuses otherwise.
                          See padding_design_decision.md. */

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

    int batch_geom; /* WHERE the K transforms of a batch live, for
                       layout=INTERLEAVED with howmany>1. The axis MKL spells
                       DFTI_INPUT_DISTANCE/STRIDES and FFTW spells
                       idist/istride. 1D C2C only; ignored at K==1 (the two
                       geometries are identical there) and for SPLIT.
                       VFFT_BATCH_TRANSFORM_CONTIGUOUS (0, DEFAULT) =
                         transform t occupies z[2*t*N .. 2*(t+1)*N),
                         elements adjacent inside it — the MKL/FFTW default
                         idiom and the canonical geometry here. Served
                         NATIVELY as K independent K=1 transforms: no
                         layout conversion anywhere, no batch tail of any
                         kind (K=3 or K=11 is simply that many transforms),
                         per-cell performance identical to the K=1 engines,
                         and one private contiguous block per thread when
                         threaded. Measured 2.2-5.7x faster than the
                         lane-major route across K in {2,3,4} x N in
                         {256..8192} (docs/roadmap/il_coverage_plan.md
                         Phase C).
                       VFFT_BATCH_LANE_MAJOR (1) = element e of transform t
                         at z[2*(e*K + t)]. The batched SPLIT engines'
                         native geometry, offered here for callers who
                         genuinely hold interleaved data that way. It is
                         served by converting to split planes and back, so
                         it is the slower path at small K — choose it
                         because your data is shaped that way, not for
                         speed. (Its strength is large K, where a vector
                         spans K independent transforms with uniform
                         twiddles and a K-split across T threads owns whole
                         cache lines once K >= 4T; that regime is the SPLIT
                         layout's home and is unaffected by this axis.)     */

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

  /* Batch geometry axis (vfft_config_t.batch_geom) — see the field comment.
   * TRANSFORM_CONTIGUOUS is the zero default, matching MKL's and FFTW's own
   * default idiom and our own measurements. LANE_MAJOR is opt-in: it exists
   * for callers who genuinely hold data that way (it is the split engines'
   * native geometry), not as a path anyone should choose for speed.
   *
   * 🔴 CHANGED 2026-08-04: the default used to be LANE_MAJOR. Code that
   * zeroes its config, sets layout=INTERLEAVED with howmany>1, and passes
   * lane-major data must now say batch_geom=VFFT_BATCH_LANE_MAJOR
   * explicitly — otherwise its buffer is read as transform-contiguous.
   * Nothing at howmany==1 is affected: the two geometries are identical
   * there and create never wraps. */
  enum
  {
    VFFT_BATCH_TRANSFORM_CONTIGUOUS = 0,
    VFFT_BATCH_LANE_MAJOR = 1
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
   *   1D C2C OOP with no OOP-kind factorization (prime and other
   *       Rader/Bluestein-class N): REJECTED loudly — those sizes are served
   *       IN-PLACE only (the prime dispatch is not wired into the OOP kinds);
   *       create with placement=VFFT_INPLACE.
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
   * LIBRARY-OWNED BUFFERS  (config.owned_buffers = 1)
   *                    (docs/roadmap/tail_handling/padding_design_decision.md)
   *
   * Set config.owned_buffers and vfft_create allocates every plane this
   * (transform x placement) needs, ZEROED, at a stride IT chooses, and frees
   * them in vfft_destroy. The plan and its buffers are one object, so they
   * cannot disagree. 1D only, layout SPLIT only.
   *
   * The stride rule is internal: 1D C2C in-place uses the MEASURED
   * tight-vs-padded verdict (so it may be exactly K, and a new (N,K) may pause
   * once to measure), while C2C out-of-place and the real/trig transforms
   * always pad. Read it back — never compute it.
   *
   *   vfft_config_t cfg = {0};
   *   cfg.transform = VFFT_C2C; cfg.n[0] = 1024; cfg.howmany = 11;
   *   cfg.owned_buffers = 1;
   *   vfft_plan p = vfft_create(&cfg);
   *   double *sre,*sim,*dre,*dim;  vfft_plan_planes(p,&sre,&sim,&dre,&dim);
   *   size_t st = vfft_plan_stride(p);        // index YOUR data at [e*st + t]
   *   ... fill ...;  vfft_execute(p, dir, sre,sim,dre,dim);  vfft_destroy(p);
   * ════════════════════════════════════════════════════════════════════════ */

  /* Hand back the plan's own planes, already in vfft_execute's argument roles
   * (planes the transform does not use are set to NULL; any out-param may be
   * NULL if unwanted). All NULL unless the plan was created with
   * config.owned_buffers = 1. Owned by the plan — never free them yourself. */
  void vfft_plan_planes(vfft_plan p, double **sre, double **sim,
                        double **dre, double **dim);
  /* The stride to index the plan's buffers with: element e of lane t lives at
   * plane[e * vfft_plan_stride(p) + t]. Equals config.howmany for a plan that
   * does not own its buffers. 0 for a NULL plan. */
  size_t vfft_plan_stride(vfft_plan p);

  /* ════════════════════════════════════════════════════════════════════════
   * GLOBAL CONTROL  (optional; sensible defaults otherwise)
   * ════════════════════════════════════════════════════════════════════════ */

  /**
   * @brief Size the shared worker pool.
   *
   * Process-global and sticky: every plan created afterwards draws its workers
   * from this pool. Prefer @c config.nthreads for per-plan control — it is
   * snapshotted at @c vfft_create, so different plans can use different thread
   * counts without touching global state.
   *
   * @param n Worker count. @c n<=1 means single-threaded.
   *
   * @warning SIDE EFFECT: for @c n>1 this PINS THE CALLING THREAD to core 0
   *          (workers then pin to 1..n-1). If you manage affinity yourself,
   *          set it AFTER this call — otherwise this overrides you.
   * @note Not thread-safe against concurrent plan creation or execution: size
   *       the pool once during setup, before handing plans to worker threads.
   * @see vfft_get_num_threads, vfft_config_t::nthreads
   */
  void vfft_set_num_threads(int n);

  /**
   * @brief Current pool size.
   * @return The configured worker count. This is neither the number of threads
   *         presently executing nor a hardware ceiling — it is exactly what the
   *         last @c vfft_set_num_threads established.
   */
  int vfft_get_num_threads(void);

  /**
   * @brief Which SIMD kernels this build compiled to.
   * @return One of @c "avx512", @c "avx2", @c "scalar". Static storage — do
   *         NOT free it; valid for the lifetime of the process.
   *
   * @warning This is the BUILD-time ISA, fixed when the library was compiled —
   *          NOT runtime CPU detection. A binary targets one instruction set:
   *          an @c "avx2" build running on an AVX-512 machine still reports and
   *          executes @c "avx2". Rebuild to target a different ISA.
   */
  const char *vfft_isa(void);

  /**
   * @brief Library version, @c "MAJOR.MINOR.PATCH".
   * @return Static storage — do NOT free it; valid for the lifetime of the
   *         process.
   */
  const char *vfft_version(void);

#ifdef __cplusplus
}
#endif
#endif /* VFFT_H */
