/**
 * @file vfft.h
 * @brief The VectorFFT public API.
 *
 * Three responsibilities:
 * - vfft_create() plans. A configuration names a data contract (transform,
 *   size, placement, layout, order, batch, threads); create commits it to one
 *   measured plan from wisdom, racing the candidates on a miss, and binds
 *   every kernel, table, buffer and thread decision at that moment.
 * - vfft_execute() runs the plan. Pure: no allocation, no measurement, no
 *   decision; the direction is the only parameter.
 * - The handlers configure: the wisdom store (load / save / free), the worker
 *   pool (set / get threads), the plan's own buffers and route
 *   (planes / stride / route), and memory at the alignment the plans were
 *   measured on (vfft_malloc / vfft_free, vfft_plan_alloc).
 *
 * The one law: a cell is served natively or refused. vfft_create() returns
 * NULL after printing why; vfft_execute() refuses a pointer signature that
 * does not match the plan's layout and computes nothing. Nothing converts a
 * layout, reorders a spectrum or falls back silently.
 *
 * Results, the supported matrix and the design are in README.md and docs/;
 * the machine proof of the matrix is build_tuned/benches/api_matrix_gate.c.
 */
#ifndef VFFT_H
#define VFFT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

  /* ── the axes of a configuration ──────────────────────────────────────── */

  /**
   * @brief The transform. Every backward direction is the unnormalized
   *        inverse; a roundtrip returns the input times the stated scale.
   *
   * Complex and real DFTs, over N points (per dimension):
   * - VFFT_C2C: complex to complex, X[k] = sum x[n] e^(-2 pi i n k / N);
   *   backward the conjugate sum, roundtrip scale N.
   * - VFFT_R2C: real input, the conjugate-even half-spectrum X[0..N/2] out.
   * - VFFT_C2R: the half-spectrum in, real output; backward of R2C, scale N.
   *
   * Real-to-real transforms, 1D only, N bins out of N reals, the standard
   * unnormalized definitions (the factor 2 on the sum):
   * - VFFT_DCT1: Y[k] = x[0] + (-1)^k x[N-1] + 2 sum_{n=1..N-2} x[n] cos(pi n k / (N-1));
   *   self-inverse, scale 2(N-1). Present, not yet validated.
   * - VFFT_DCT2: Y[k] = 2 sum x[n] cos(pi k (2n+1) / 2N); its inverse is
   *   DCT-III, scale 2N. Even N.
   * - VFFT_DCT3: Y[k] = X[0] + 2 sum_{n>=1} X[n] cos(pi n (2k+1) / 2N); its
   *   inverse is DCT-II, scale 2N. Even N.
   * - VFFT_DCT4: Y[k] = 2 sum x[n] cos(pi (2k+1)(2n+1) / 4N); self-inverse,
   *   scale 2N. Even N.
   * - VFFT_DST1: Y[k] = 2 sum x[n] sin(pi (n+1)(k+1) / (N+1)); self-inverse,
   *   scale 2(N+1).
   * - VFFT_DST2: Y[k] = 2 sum x[n] sin(pi (k+1)(2n+1) / 2N); its inverse is
   *   DST-III, scale 2N. Even N.
   * - VFFT_DST3: Y[k] = (-1)^k X[N-1] + 2 sum_{n<=N-2} X[n] sin(pi (n+1)(2k+1) / 2N);
   *   its inverse is DST-II, scale 2N. Even N.
   * - VFFT_DHT: H[k] = sum x[n] (cos(2 pi n k / N) + sin(2 pi n k / N));
   *   self-inverse, scale N.
   *
   * For a real-to-real plan VFFT_BACKWARD runs the inverse named above
   * (DCT-III for a DCT-II plan, and so on); for the self-inverse ones the
   * two directions coincide. Real-to-real transforms have no complex
   * layout and no order axis; a request for either is refused.
   */
  typedef enum
  {
    VFFT_C2C,  /**< complex to complex */
    VFFT_R2C,  /**< real to the conjugate-even half-spectrum */
    VFFT_C2R,  /**< the half-spectrum to real */
    VFFT_DCT1, /**< DCT-I (REDFT00) */
    VFFT_DCT2, /**< DCT-II (REDFT10) */
    VFFT_DCT3, /**< DCT-III (REDFT01) */
    VFFT_DCT4, /**< DCT-IV (REDFT11) */
    VFFT_DST1, /**< DST-I (RODFT00) */
    VFFT_DST2, /**< DST-II (RODFT10) */
    VFFT_DST3, /**< DST-III (RODFT01) */
    VFFT_DHT   /**< the discrete Hartley transform */
  } vfft_transform_t;

  /** @brief Whether the output overwrites the input. */
  typedef enum
  {
    VFFT_INPLACE,
    VFFT_OUTOFPLACE
  } vfft_placement_t;

  /**
   * @brief The complex-data layout, committed at create; vfft_execute()'s
   *        pointer signature follows it.
   *
   * SPLIT (the zero default): separate re[] and im[] planes. INTERLEAVED:
   * one z[] of adjacent (re, im) pairs; for R2C and C2R the spectrum side
   * is the packed conjugate-even half-spectrum of N/2 + 1 pairs.
   * Real-to-real transforms have no complex layout: INTERLEAVED is refused.
   */
  typedef enum
  {
    VFFT_LAYOUT_SPLIT = 0,
    VFFT_LAYOUT_INTERLEAVED
  } vfft_layout_t;

  /**
   * @brief How thoroughly a wisdom miss (or a recalibrating create) is
   *        measured. A hit ignores it. Every tier measures; none estimates.
   */
  typedef enum
  {
    VFFT_MEASURE,   /**< coarse sweep */
    VFFT_PATIENT,   /**< wide sweep */
    VFFT_EXHAUSTIVE /**< full sweep */
  } vfft_rigor_t;

  /** @brief The direction. Backward is the unnormalized inverse. */
  typedef enum
  {
    VFFT_FORWARD,
    VFFT_BACKWARD
  } vfft_dir_t;

  /* ── wisdom: the measured verdicts a plan is served from ─────────────── */

  typedef struct vfft_wisdom_s vfft_wisdom; /**< opaque */

  /**
   * @brief Load a caller-owned wisdom table.
   * @param path A wisdom store directory; NULL loads the library's own store.
   * @return The table, or NULL when the store cannot be read. The caller
   *         frees it with vfft_wisdom_free(); a plan never frees the table it
   *         was given, and config.wisdom == NULL makes a plan use the
   *         library's store instead of any table.
   */
  vfft_wisdom *vfft_wisdom_load(const char *path);
  /**
   * @brief Persist a table to a store directory.
   * @return 0 on success.
   */
  int vfft_wisdom_save(const vfft_wisdom *w, const char *path);
  /** @brief Free a table returned by vfft_wisdom_load(). NULL is accepted. */
  void vfft_wisdom_free(vfft_wisdom *w);

  /* ── the configuration ────────────────────────────────────────────────── */

  /**
   * @brief The data contract vfft_create() commits to. A zeroed struct is a
   *        1D split-layout C2C of size n[0], one transform, one thread,
   *        natural order, the library's wisdom.
   */
  typedef struct
  {
    vfft_transform_t transform;
    vfft_placement_t placement;
    vfft_rigor_t rigor; /**< the sweep on a miss or recalibrate; a hit ignores it */

    int dims;         /**< 1 (default), 2, 3 or 4 */
    int n[4];         /**< n[0] = N (1D); {N1, N2}; {N1, N2, N3}; {N1, N2, N3, N4}.
                           dims 3 and 4 take howmany == 1. */
    size_t howmany;   /**< K, the batch count; where the K transforms sit is
                           batch_geom */
    int owned_buffers; /**< 1 = create allocates the planes this plan needs, at a
                            measured stride, zeroed, and destroy frees them; read
                            them with vfft_plan_planes() and vfft_plan_stride().
                            1D and SPLIT only; refused otherwise. 0 (default) =
                            the caller's own tight planes. */

    int nthreads; /**< the plan's thread count, taken at create; 0 = the pool's
                       current size. Every threaded decision is measured and
                       served at this count. */

    int order; /**< the output-order contract of 1D and 2D C2C (and the row
                    order of 2D INTERLEAVED R2C/C2R; their bins are always
                    natural). The order is a contract: NATURAL delivers
                    natural bins, SCRAMBLED delivers the engine's order.
                    VFFT_ORDER_DEFAULT (0) is the layout's own: NATURAL for
                    INTERLEAVED at every rank; for SPLIT, NATURAL at 1D and
                    SCRAMBLED at 2D and above (the split tiers are built
                    around the scrambled comb). VFFT_ORDER_SCRAMBLED: the engine's own
                    permutation of the bins, decodable only by the matched
                    roundtrip through the same plan; no call reports it, and
                    a cell with no scrambled writer refuses at create. 1D real
                    and real-to-real transforms are natural by nature and
                    refuse an order; 3D and 4D SPLIT take DEFAULT or
                    SCRAMBLED. */

    vfft_layout_t layout; /**< committed at create; see vfft_layout_t */

    int batch_geom; /**< where the K transforms of a batch sit (meaningful at
                         howmany > 1; the geometries coincide at K == 1).
                         VFFT_BATCH_DEFAULT (0) is the layout's own:
                         transform-contiguous for INTERLEAVED C2C, lane-major
                         for SPLIT and for INTERLEAVED R2C/C2R. The explicit
                         values state the other geometry; transform-contiguous
                         on SPLIT is refused. Definitions below. */

    vfft_wisdom *wisdom; /**< NULL = the library's store; else this table */
    int recalibrate;     /**< 1 = re-measure this cell even on a hit */
    int wisdom_write;    /**< retired, ignored: a create that races always
                              saves its winner to the store before it returns.
                              The environment variable VFFT_WISDOM_WRITE=0
                              turns saving off for a process. */
  } vfft_config_t;

  /** @brief vfft_config_t.order */
  enum
  {
    VFFT_ORDER_DEFAULT = 0,  /**< natural */
    VFFT_ORDER_NATURAL = 1,  /**< natural, said explicitly */
    VFFT_ORDER_SCRAMBLED = 2 /**< the engine's own permutation, roundtrip-decodable only */
  };

  /**
   * @brief vfft_config_t.batch_geom.
   *
   * Transform-contiguous: transform t occupies z[2 t N .. 2 (t + 1) N), its
   * elements adjacent (K independent transforms end to end). Lane-major:
   * element e of transform t at plane[e K + t] in a split plane, at
   * z[2 (e K + t)] interleaved.
   */
  enum
  {
    VFFT_BATCH_DEFAULT = 0,
    VFFT_BATCH_TRANSFORM_CONTIGUOUS = 1,
    VFFT_BATCH_LANE_MAJOR = 2
  };

  /* ── create, execute, destroy ─────────────────────────────────────────── */

  typedef struct vfft_plan_s *vfft_plan; /**< opaque, execute-ready */

  /**
   * @brief Commit a configuration to one measured, execute-ready plan.
   *
   * Responsibilities, in order:
   * -# Validate the contract. An unsupported cell or an invalid combination
   *    returns NULL after printing the reason; nothing is converted, padded
   *    or reinterpreted to make it fit.
   * -# Resolve the plan from wisdom. A hit serves the banked verdict. A miss
   *    (or config.recalibrate) races the cell's candidates on scratch data at
   *    config.rigor, a pause of milliseconds to seconds, banks the winner and
   *    saves it to the store before create returns. A store that cannot be
   *    written keeps the winner for the process and says so once. The races
   *    run inside the measurement scope (vfft_measure_config_t): the calling
   *    thread is pinned to a P-core and raised for their duration, and is
   *    given back as it was.
   * -# Build the plan: the kernels are bound, the twiddle tables computed,
   *    and everything the plan runs on allocated by the plan itself: the
   *    scratch and staging planes, 64-byte aligned; the tables; the child
   *    plans of a multi-dimensional or batched transform and their
   *    per-worker clones; and, with config.owned_buffers, the caller's
   *    planes at a measured stride, aligned and zeroed.
   * -# Bind the threads. config.nthreads is snapshotted and the threaded
   *    form is measured and served at that count.
   *
   * The plan owns everything it allocated and vfft_destroy() frees it all.
   * Create never reads or writes the caller's data. Any double-aligned buffer
   * address computes the same values: the kernels use unaligned vector
   * access. The measured speed holds for buffers whose base is aligned to
   * vfft_alignment() bytes (read it at run time; the value belongs to the
   * build), each buffer a separate allocation, as vfft_malloc() and
   * vfft_plan_alloc() provide them; plain malloc() guarantees less. A SPLIT
   * plan's planes are separate allocations too: carved back to back out of
   * one block they can sit at one 4 KB offset, which can run measurably
   * slower than the planes the races measured. Inside a buffer, each batch
   * transform and batch plane starts where the data contract puts it; one
   * that starts off a cache line runs at a phase its route was not raced at.
   * Not safe to call concurrently with vfft_set_num_threads().
   *
   * @param config The contract; read during the call only.
   * @return The plan, or NULL with the reason printed.
   */
  vfft_plan vfft_create(const vfft_config_t *config);

  /**
   * @brief Run the plan in one direction.
   *
   * The pointer roles follow the plan's layout; a signature that does not
   * match is refused (printed, nothing computed).
   *
   *   transform    layout        sre        sim      dre        dim
   *   C2C          SPLIT         in.re      in.im    out.re     out.im
   *   C2C          INTERLEAVED   z_in       NULL     z_out      NULL
   *   R2C          SPLIT         real_in    NULL     spec.re    spec.im
   *   R2C          INTERLEAVED   real_in    NULL     z_spec     NULL
   *   C2R          SPLIT         spec.re    spec.im  real_out   NULL
   *   C2R          INTERLEAVED   z_spec     NULL     real_out   NULL
   *   real-to-real (real)        real_in    NULL     real_out   NULL
   *
   * In-place C2C: dre == sre (and dim == sim), or dre and dim NULL. In-place
   * R2C/C2R (1D, even N, INTERLEAVED, K == 1 or transform-contiguous): ONE
   * plane of 2 (N/2 + 1) doubles holds the N reals and then the N/2 + 1
   * bins; dre == sre is required and a distinct dre is refused. Element
   * addressing follows config.batch_geom. Pure: no allocation, no
   * measurement. Safe to call concurrently on different plans.
   *
   * @param p   A plan from vfft_create().
   * @param dir VFFT_FORWARD or the unnormalized inverse.
   */
  void vfft_execute(vfft_plan p, vfft_dir_t dir,
                    double *sre, double *sim, double *dre, double *dim);

  /** @brief Free a plan and everything it allocated. NULL is accepted. */
  void vfft_destroy(vfft_plan p);

  /* ── the plan's own buffers (config.owned_buffers = 1) ───────────────── */

  /**
   * @brief The plan's planes in vfft_execute()'s argument roles.
   * @param p A plan created with config.owned_buffers = 1; otherwise every
   *          out-param is set to NULL.
   * @param sre,sim,dre,dim Out-params; unused roles are set to NULL, and any
   *          out-param may itself be NULL. The planes are owned by the plan:
   *          never free them.
   */
  void vfft_plan_planes(vfft_plan p, double **sre, double **sim,
                        double **dre, double **dim);
  /**
   * @brief The stride to index the plan's planes with: element e of lane t
   *        is at plane[e * stride + t].
   * @return The stride; equals config.howmany for a plan that does not own
   *         its buffers; 0 for NULL. Read it, never compute it.
   */
  size_t vfft_plan_stride(vfft_plan p);

  /* ── memory at the alignment the plans were measured on ───────────────── */

  /**
   * @brief Allocate a buffer for vfft_execute().
   *
   * The block is aligned to vfft_alignment() bytes, the alignment every plan
   * was measured on, and its size is rounded up to a multiple of it. The
   * memory is uninitialised; 0 bytes gives a unique block. Release it with
   * vfft_free() only, never free() or delete; a block allocated by one module
   * that links the library may be released by any other. Compute sizes in
   * size_t (2 * (size_t)N * sizeof(double)), or take the buffers from
   * vfft_plan_alloc(), which sizes them with the arithmetic checked.
   *
   * @param bytes The size in bytes.
   * @return The block; NULL when memory runs out, or with the reason printed
   *         for a size whose rounding overflows.
   */
  void *vfft_malloc(size_t bytes);
  /** @brief Release a vfft_malloc() block. NULL is accepted. */
  void vfft_free(void *p);
  /**
   * @brief The alignment in bytes of every vfft_malloc() block and of the
   *        buffers every plan was measured on. A build-time fact: read it at
   *        run time rather than writing the number down.
   */
  size_t vfft_alignment(void);

  /** @brief One set of buffers sized for a plan (opaque). */
  typedef struct vfft_buffers_s *vfft_buffers;

  /**
   * @brief Allocate one set of the buffers vfft_execute(p, ...) takes, in the
   *        plan's own geometry.
   *
   * Each role is its own vfft_malloc() block, sized by the plan with the
   * arithmetic checked: the in-place real plane of 2 (N/2 + 1) doubles, the
   * spectrum of a real transform, the K transforms or planes of a batch. A
   * SPLIT plan created with config.owned_buffers = 1 gets its planes at the
   * plan's measured stride (vfft_plan_stride()) with the pad lanes zeroed;
   * every other plan gets the tight planes of its data contract. The data
   * is uninitialised. A plan may hand out any number of sets, and a set stays
   * valid after vfft_destroy(p). Refused (printed): a SPLIT plan of rank 2 or
   * more, and an INTERLEAVED real batch in its lane-major geometry that runs
   * on the split engines.
   *
   * @param p A plan from vfft_create().
   * @param sre,sim,dre,dim Out-params in vfft_execute()'s roles: unused roles
   *        are set to NULL, in place dre == sre (and dim == sim), and any
   *        out-param may itself be NULL.
   * @return The set, released with vfft_buffers_free() only, never by passing
   *         one of its buffers to vfft_free() or free(); NULL (every out-param
   *         NULL) when refused, with the reason printed, or when memory runs
   *         out.
   */
  vfft_buffers vfft_plan_alloc(vfft_plan p, double **sre, double **sim,
                               double **dre, double **dim);
  /** @brief Release a vfft_plan_alloc() set and every buffer in it. NULL is accepted. */
  void vfft_buffers_free(vfft_buffers b);

  /* ── the measurement scope ────────────────────────────────────────────── */

  /**
   * @brief How the library sets a thread up for measuring. Zero is the
   *        default of every field.
   *
   * vfft_create() measures (races) a cell's candidates whenever wisdom does
   * not serve it, and saves the winner. It runs every race inside one scope
   * and undoes it before it returns:
   * - a lock for the whole machine, so two measurements never run at once.
   *   The wait is bounded; past it the race runs anyway.
   * - the measuring thread pinned to one P-core: the second one in a process
   *   without workers, logical CPU 0 once the pool exists (see
   *   vfft_set_num_threads()).
   * - a guard thread holding that core's hyperthread sibling, so nothing
   *   else is scheduled there. It waits with the processor's idle
   *   instruction (TPAUSE on Intel, MWAITX on AMD), which costs the measured
   *   thread nothing; a part with neither runs unguarded.
   * - the measuring thread's priority raised, where the system allows it.
   *
   * A winner measured without the lock, or on a thread that is not pinned to
   * a P-core, is served for the process and not saved. The application's own
   * threads and the machine's other load are the caller's to keep quiet.
   */
  typedef struct
  {
    int pin;          /**< VFFT_MEASURE_PIN_DEFAULT, _OFF or _CORE */
    int pin_core;     /**< the logical CPU for VFFT_MEASURE_PIN_CORE */
    int guard;        /**< VFFT_MEASURE_GUARD_DEFAULT, _OFF or _PAUSE */
    int priority;     /**< VFFT_MEASURE_PRIORITY_THREAD, _LEAVE or _PROCESS */
    int lock_wait_ms; /**< the longest wait for another measurement on the
                           machine, in milliseconds; 0 = 60000 */
  } vfft_measure_config_t;

  /** @brief vfft_measure_config_t.pin */
  enum
  {
    VFFT_MEASURE_PIN_DEFAULT = 0, /**< the library's own core (see above) */
    VFFT_MEASURE_PIN_OFF = 1,     /**< the thread is not moved; winners are served, not saved */
    VFFT_MEASURE_PIN_CORE = 2     /**< logical CPU pin_core; winners are saved only when it is a P-core */
  };
  /** @brief vfft_measure_config_t.guard */
  enum
  {
    VFFT_MEASURE_GUARD_DEFAULT = 0, /**< the processor's idle instruction, where it has one */
    VFFT_MEASURE_GUARD_OFF = 1,
    VFFT_MEASURE_GUARD_PAUSE = 2    /**< a PAUSE spinner: works on every part with hyperthreads, and
                                         slows the measured thread about 12% (every candidate alike) */
  };
  /** @brief vfft_measure_config_t.priority */
  enum
  {
    VFFT_MEASURE_PRIORITY_THREAD = 0,  /**< raise the measuring thread */
    VFFT_MEASURE_PRIORITY_LEAVE = 1,
    VFFT_MEASURE_PRIORITY_PROCESS = 2  /**< raise the whole process */
  };
  /** @brief What vfft_measure_begin() returns (0 = a clean scope). */
  enum
  {
    VFFT_MEASURE_CONTENDED = 1, /**< the lock was not obtained within the wait */
    VFFT_MEASURE_UNPINNED = 2   /**< the thread is not pinned to a P-core */
  };

  /**
   * @brief Set how this process measures: the scope of every later
   *        vfft_create() race and vfft_measure_begin().
   * @param config The settings; NULL restores the defaults. Call it during
   *        setup, outside any scope.
   */
  void vfft_measure_configure(const vfft_measure_config_t *config);

  /**
   * @brief Enter the measurement scope on the calling thread, for code the
   *        caller times itself.
   *
   * The same scope vfft_create() uses for its races: the machine-wide lock,
   * the pin, the sibling guard, the priority. A vfft_create() inside it adds
   * nothing and leaves it in place. Calls nest; the scope ends at the
   * matching outermost vfft_measure_end(), on the same thread.
   *
   * @return 0, or VFFT_MEASURE_CONTENDED and/or VFFT_MEASURE_UNPINNED:
   *         winners raced in such a scope are served, not saved.
   */
  int vfft_measure_begin(void);

  /**
   * @brief Leave the scope: the guard ends, the thread's priority and
   *        affinity return to what they were, the lock is released. One pin
   *        stays: the pool's pin of its caller to logical CPU 0, when the
   *        pool was sized or grown inside the scope.
   */
  void vfft_measure_end(void);

  /**
   * @brief Confine the process to a set of logical CPUs, for a threaded
   *        comparison: every thread created afterwards, by any library, runs
   *        inside it.
   * @param mask Bit c = logical CPU c; 0 = one logical CPU per P-core (no
   *        hyperthread siblings, no E-cores).
   * @return The mask applied, 0 when the system refused it. Not undone by
   *         vfft_measure_end().
   */
  unsigned long long vfft_measure_confine(unsigned long long mask);

  /**
   * @brief One line describing the calling thread's most recent scope: the
   *        pinned CPU, the guard, the priority, the lock.
   * @return buf.
   */
  const char *vfft_measure_describe(char *buf, size_t n);

  /* ── the worker pool and the build ────────────────────────────────────── */

  /**
   * @brief Size the shared worker pool.
   *
   * Process-global and sticky: plans created afterwards draw their workers
   * from it (config.nthreads snapshots the count per plan).
   *
   * @param n The worker count; n <= 1 is single-threaded.
   * @warning For n > 1 the calling thread is pinned to core 0 and the workers
   *          to the following cores. Set your own affinity after this call.
   * @warning Not safe against concurrent plan creation or execution: size the
   *          pool once, during setup.
   */
  void vfft_set_num_threads(int n);
  /** @brief The configured pool size, as the last vfft_set_num_threads() set it. */
  int vfft_get_num_threads(void);

  /**
   * @brief The SIMD level this build was compiled for.
   * @return "avx512", "avx2" or "scalar": a build-time fact, not runtime
   *         detection. Static storage.
   */
  const char *vfft_isa(void);

  /**
   * @brief The route a plan committed to.
   * @return For a K=1 interleaved plan its route name ("mono", "2p",
   *         "chain3", "prime", "flat", "ztt", "fs", or a 2D route); "-" for a
   *         plan with no single route name (split layout, batches) and for
   *         NULL. Static storage.
   */
  const char *vfft_plan_route(vfft_plan p);

  /** @brief The library version, "MAJOR.MINOR.PATCH". Static storage. */
  const char *vfft_version(void);

  /* Threading diagnostics (engagement counters) are in vfft_diagnostics.h;
   * none is needed to compute a transform. */

#ifdef __cplusplus
}
#endif
#endif /* VFFT_H */
