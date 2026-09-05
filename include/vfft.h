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
 *   R2C          1–4     OOP; IP: 1D CCE only^ split | CCE spectrum  natural  any K  yes  pad-only
 *   C2R          1–4     OOP; IP: 1D CCE only^ split | CCE spectrum  natural  any K  yes  pad-only
 *   DCT-I..IV    1       IP + OOP        real (layout n/a)  natural          any K    yes  pad-only
 *   DST-I..III   1       IP + OOP        real (layout n/a)  natural          any K    yes  pad-only
 *   DHT          1       IP + OOP        real (layout n/a)  natural          any K    yes  pad-only
 *
 *   *   INTERLEAVED is NATIVE for 1D C2C (both placements, every order, any
 *       N including prime — Rader/Bluestein on packed z), for 1D r2c/c2r
 *       (CCE spectrum; odd and prime N through the c2c bridge) and for the
 *       whole 2D family (C2C and r2c/c2r on the native column-chain tier;
 *       prime and odd dims through odd chains or the column-axis
 *       Bluestein). There is NO layout-conversion tier anywhere: a cell
 *       either serves natively or refuses loudly. 3D/4D INTERLEAVED is not
 *       wired (refused as a planned feature — use SPLIT); trig transforms
 *       have no complex layout.
 *   **  order is a C2C axis in 1D and 2D (NATURAL is native in both, for
 *       any factorization) and the ROW-order axis of 2D INTERLEAVED
 *       r2c/c2r (their bins are always natural; NATURAL orders the rows
 *       too). 3D/4D: DEFAULT or SCRAMBLED only, K must be 1. 1D r2c/c2r and
 *       trig are inherently natural — an order request there is refused,
 *       not ignored.
 *   *** 2D INTERLEAVED C2C/R2C/C2R accept howmany>1 (served as a plane
 *       queue, threaded per plane); every other dims>=2 cell requires
 *       howmany==1. DCT-I is present but not yet validated.
 *       PRIME / AWKWARD N: 1D C2C serves them in both placements under
 *       INTERLEAVED (verified to 32749) and IN-PLACE only under SPLIT (the
 *       split OOP kinds need a radix factorization — refused loudly out of
 *       place); 1D r2c/c2r serve any odd or prime N out of place in both
 *       layouts and in place under INTERLEAVED; 2D prime/odd dims serve in
 *       both layouts for C2C and under INTERLEAVED for r2c/c2r (SPLIT 2D
 *       real at a prime dim refuses).
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
                    MKL/FFTW-comparable, served by whichever natural-native
                    engine wins the cell's race (the natural-writing cascade
                    terminator at large N, the natural IL kinds below it,
                    PURE/PSWAP reorders where they win) — a per-cell verdict
                    in wisdom, never a reorder pass by default.
                  1D and 2D C2C (in-place + OOP; 2D NATURAL is native for any
                  factorization — the column chain's leaf writes rows in
                  natural order); for 2D INTERLEAVED r2c/c2r it is the
                  row-order axis (bins are always natural). 1D r2c/c2r and
                  trig are inherently natural: an order request there is
                  refused. 3D/4D: DEFAULT/SCRAMBLED only.
                  Roundtrip/convolution consumers should keep DEFAULT (order is
                  irrelevant there, and it is the fastest).                     */

    vfft_layout_t layout; /* Complex-data layout axis (see vfft_layout_t above).
                             Committed at create; execute enforces the matching
                             pointer signature. Default (0) = SPLIT.            */

    int batch_geom; /* WHERE the K transforms of a batch live. The axis MKL
                       spells DFTI_INPUT_DISTANCE/STRIDES and FFTW spells
                       idist/istride. Meaningful for 1D C2C layout=INTERLEAVED
                       with howmany>1; ignored at K==1 (the two geometries
                       are identical there).
                       VFFT_BATCH_DEFAULT (0) = this layout's canonical
                         geometry: transform-contiguous for INTERLEAVED,
                         lane-major for SPLIT (whose engines have no other
                         contract — asking SPLIT for transform-contiguous is
                         refused, not ignored).
                       VFFT_BATCH_TRANSFORM_CONTIGUOUS (1; the INTERLEAVED
                         default) =
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
                       VFFT_BATCH_LANE_MAJOR (2; the SPLIT default and its
                         only geometry) = element e of transform t at
                         z[2*(e*K + t)] interleaved, or plane[e*K + t]
                         split. Offered on INTERLEAVED for callers who
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
    int wisdom_write;    /* the wisdom2 write guard (owner rule: the library
                            DEFAULT is read-only wisdom). 0 = serving mode:
                            hits are served, a miss races in memory for this
                            process but writes NOTHING to disk. 1 =
                            measurement mode: calibrate-on-miss persists.
                            Calibrators, benches, and gates set this;
                            applications never bank by accident. Applies to
                            the wisdom2 store; legacy wisdom files are
                            frozen regardless. */
  } vfft_config_t;

  /* Output-order axis (vfft_config_t.order), C2C in 1D and 2D (+ the row order of 2D INTERLEAVED
   * real). DEFAULT = engine-native, fastest, order-agnostic. SCRAMBLED = the order-agnostic
   * contract stated explicitly (any self-consistent comb; the identity qualifies, so a natural-
   * native engine may serve it). NATURAL = bin-for-bin natural, served by whichever natural-native
   * engine wins the cell's race. Values map 1:1 onto the internal OOP kind constraint
   * (0=any,1=natural,2=scrambled). */
  enum
  {
    VFFT_ORDER_DEFAULT = 0,
    VFFT_ORDER_NATURAL = 1,
    VFFT_ORDER_SCRAMBLED = 2
  };

  /* Batch geometry axis (vfft_config_t.batch_geom) — see the field comment.
   *
   * DEFAULT (0) means "this layout's canonical geometry", which is NOT the
   * same geometry for both layouts and deliberately so:
   *   INTERLEAVED -> transform-contiguous (the MKL/FFTW idiom, and the one
   *                  we serve natively as K independent K=1 transforms)
   *   SPLIT       -> lane-major (the batched split engines' own contract;
   *                  transform-contiguous split planes are NOT supported)
   * So a zeroed config always gets the right thing for the layout it asked
   * for, and neither layout's default is a silent mismatch. The explicit
   * values exist to say "my data really is shaped the other way".
   * Requesting TRANSFORM_CONTIGUOUS on SPLIT is refused loudly rather than
   * silently ignored (no silent-corruption path).
   *
   * ⚠ REAL TRANSFORMS ARE THE ONE EXCEPTION to the DEFAULT rule above:
   * interleaved R2C/C2R still defaults to LANE-MAJOR. The flip below was
   * justified by a measured race (2.2-5.7x) that has not yet been run for the
   * real path, which only acquired a transform-contiguous route on 2026-08-22.
   * Ask for it by name until it has.
   *
   * 🔴 CHANGED 2026-08-04: INTERLEAVED batches used to default to
   * lane-major. Code that zeroes its config, sets layout=INTERLEAVED with
   * howmany>1, and passes lane-major data must now say
   * batch_geom=VFFT_BATCH_LANE_MAJOR explicitly. Nothing at howmany==1 is
   * affected: the geometries are identical there and create never wraps. */
  enum
  {
    VFFT_BATCH_DEFAULT = 0,
    VFFT_BATCH_TRANSFORM_CONTIGUOUS = 1,
    VFFT_BATCH_LANE_MAJOR = 2
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
   *                           dre = packed CCE half-spectrum. WHERE the K
   *                           transforms sit is config.batch_geom:
   *                             DEFAULT / LANE_MAJOR  (N/2+1)*K pairs, bin f
   *                               of transform t at dre[2*(f*K+t)] (§6a24),
   *                               and sre likewise at sre[e*K+t];
   *                             TRANSFORM_CONTIGUOUS  transform t owns the
   *                               block sre[t*N ..) -> dre[t*2*(N/2+1) ..),
   *                               K independent transforms end to end.
   *                           ⚠ INTERLEAVED DEFAULT means transform-contiguous
   *                           for C2C but LANE-MAJOR here — the real path
   *                           predates the 2026-08-04 flip and keeps its
   *                           geometry until the same race is run for it.
   *   R2C (fwd)  INTERLEAVED  z_plane    NULL       z_plane    NULL
   *              IN-PLACE     dre == sre REQUIRED (a distinct dre is refused;
   *                           NULL is NOT accepted as "same as sre"). ONE
   *                           plane of 2*(N/2+1) doubles: N reals in, the
   *                           N/2+1 CCE bins written over them. 1D, EVEN N.
   *                           K==1, or K>1 with batch_geom =
   *                           TRANSFORM_CONTIGUOUS (that plane per transform,
   *                           end to end, at a 2*(N/2+1)-double stride).
   *   C2R (bwd)  SPLIT        in.re      in.im      real_out   NULL
   *   C2R (bwd)  INTERLEAVED  z_CCE_in   NULL       real_out   NULL
   *                           sre = the CCE spectrum (same packing as R2C out,
   *                           batch_geom included — the roles swap ends).
   *   C2R (bwd)  INTERLEAVED  z_plane    NULL       z_plane    NULL
   *              IN-PLACE     the mirror of in-place R2C: same single padded
   *                           plane of 2*(N/2+1) doubles, dre == sre
   *                           REQUIRED, same 1D / EVEN N / K rule.
   *   DCT/DST/DHT (SPLIT)     real_in    NULL       real_out   NULL
   *                           real->real; INTERLEAVED rejected at create.
   *
   * SUPPORT MATRIX (create commits; NATIVE = a native engine path, REJECT =
   * loud create-time refusal. There is NO layout-conversion tier: a cell
   * either serves natively or refuses — owner law, 2026-09-03. The machine
   * proof of this table is benches/api_matrix_gate.c):
   *
   *   1D C2C in-place   x INTERLEAVED: NATIVE for every order and any N —
   *       the folded z->z engine and the natural-order tier (per-cell raced
   *       verdicts), Rader/Bluestein on packed z for prime and awkward N;
   *       howmany>1 in the transform-contiguous geometry. config.batch +
   *       INTERLEAVED: REJECT.
   *   1D C2C OOP        x INTERLEAVED: NATIVE z->z for K=1 at any N (the
   *       K=1 IL tiers — mono, pair, chain, the flat mixed-radix DIT for
   *       odd N to 2^18, Rader/Bluestein, the scrambled or natural-writing
   *       cascade — all raced per cell) and for K>1 in the
   *       TRANSFORM_CONTIGUOUS geometry (K independent K=1 transforms; the
   *       threading verdict is raced and banked T-free). Lane-major
   *       INTERLEAVED batches: REJECT (not an IL route; nothing to fall
   *       back to by design).
   *   1D C2C            x SPLIT: NATIVE. Prime / Rader-Bluestein-class N is
   *       served IN-PLACE only (the split OOP kinds need a radix
   *       factorization) — OOP SPLIT at such N: REJECT; create with
   *       placement=VFFT_INPLACE or layout=INTERLEAVED. TRANSFORM_CONTIGUOUS
   *       on SPLIT: REJECT (split batches are lane-major by contract).
   *   R2C/C2R           x INTERLEAVED: NATIVE CCE executors (1D + 2D),
   *       lane-major. 1D K>1 with batch_geom = TRANSFORM_CONTIGUOUS is served
   *       instead as K independent K=1 transforms end to end (the same
   *       wrapper 1D C2C uses), which is the only geometry that reaches the
   *       §D2 zr2c route at K>1: that route REINTERPRETS a transform's N
   *       contiguous reals as N/2 complex points, and under lane-major the
   *       two halves of one complex sample are K apart. That wrapper is also
   *       where real batches thread — one plan clone per worker, a slab of
   *       whole transforms each.
   *       ODD / PRIME N (2026-09-04): c2r at any odd N, and r2c at prime or
   *       awkward N, serve through the c2c bridge (promote -> c2c(N) -> keep
   *       the N/2+1 bins; extend -> inverse c2c -> real part) in both
   *       layouts out of place; smooth-odd r2c races the bridge against the
   *       native rfft route per cell and serves the winner.
   *       IN-PLACE (^, §D2 2026-08-13; odd N 2026-09-04): 1D,
   *       LAYOUT_INTERLEAVED, any N, and either howmany == 1 or — even N —
   *       TRANSFORM_CONTIGUOUS. Every other in-place real shape is REJECTED
   *       loudly: with a split spectrum the real data and the spectrum are
   *       separate planes, so an in-place contract there would be a lie —
   *       and in the lane-major batch geometry the reals and bins of one
   *       transform interleave with every other transform's, so no
   *       single-plane overwrite exists.
   *
   *       THE IN-PLACE REAL CONTRACT (the only place it is stated):
   *         - ONE padded plane of 2*(N/2 + 1) doubles, the MKL CCE
   *           convention (N+1 doubles at odd N). The caller allocates that,
   *           not N.
   *         - R2C reads N reals from the front of the plane and writes the
   *           N/2 + 1 CCE bins over it; C2R is the mirror.
   *         - vfft_execute MUST be called fully aliased: dre == sre, both
   *           non-NULL. Unlike in-place 1D C2C, dre == NULL is NOT accepted
   *           as "same as sre" here, and a distinct dre is REFUSED (it used
   *           to be silently miscomputed on one of the two internal routes).
   *   2D C2C            x INTERLEAVED: NATIVE tier, both placements — the
   *       n1c/t2c column chain (odd radices included) + K=1 IL row pass,
   *       every axis raced and banked per cell (lay=il rows, keyed by
   *       order); prime / inexpressible N1 through the column-axis
   *       Bluestein (raced against an odd chain where one exists); odd or
   *       prime N2 through the row child. ORDER_NATURAL is native for any
   *       factorization (the leaf stage writes rows in natural order — no
   *       reorder pass). Intra-transform MT is raced and banked per cell and
   *       thread count; howmany>1 is served by the plane queue (raced
   *       loop-vs-queue, threaded per plane).
   *   2D R2C/C2R        x INTERLEAVED: NATIVE tier, OUT OF PLACE — the same
   *       column machinery over the CCE plane; odd/prime N1 and N2,
   *       ORDER_NATURAL on the row axis, MT, plane queue. In-place 2D real:
   *       REJECT (the rows/columns wall of the Hermitian fold makes a
   *       single-plane contract a lie).
   *   2D C2C / R2C / C2R x SPLIT: NATIVE split 2D engines, howmany == 1
   *       (howmany>1 on SPLIT 2D: REJECT); SPLIT 2D real at a prime dim:
   *       REJECT.
   *   3D..4D            x INTERLEAVED: REJECT ("the rank-3+ interleaved tier
   *       is a planned feature") — use SPLIT. 3D/4D SPLIT: C2C with
   *       howmany == 1 and order DEFAULT/SCRAMBLED, and out-of-place R2C/C2R
   *       with an even last dim; anything else REJECTs.
   *   TRIG              x INTERLEAVED: REJECT (no complex layout); trig or
   *       1D real with an order request: REJECT (inherently natural).
   *   any batch         x INTERLEAVED: REJECT (padded planes are split).
   *
   * `dir` selects forward vs the (unnormalized) inverse; for self-inverse trig
   * (DCT-I/IV, DST-I, DHT) the two coincide. ════════════════════════════════ */
  void vfft_execute(vfft_plan p, vfft_dir_t dir,
                    double *sre, double *sim, double *dre, double *dim);

  void vfft_destroy(vfft_plan p);

  /* Threading diagnostics (did the MT actually engage?) moved to
   * include/vfft_diagnostics.h; the r2c/c2r dispatch config hooks are
   * internal, in src/core/transforms/real/real_dispatch_config.h.
   * Neither is needed to compute a transform, so neither is part of
   * this header's contract. */

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
