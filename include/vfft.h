/**
 * @file vfft.h
 * @brief The VectorFFT public API: eight functions.
 *
 *   vfft_create         commit a configuration to one plan
 *   vfft_execute        run an INTERLEAVED or a real-to-real plan
 *   vfft_execute_split  run a SPLIT plan
 *   vfft_destroy        free a plan
 *   vfft_malloc         memory at the alignment the plans were measured on
 *   vfft_free           release it
 *   vfft_info           a plan's facts, or the library's
 *   vfft_error          why the last create or execute on this thread was refused
 *
 * A cell is served natively or refused: vfft_create() returns NULL and
 * vfft_error() says why. Nothing converts a layout, reorders a spectrum or
 * falls back silently. A create that misses wisdom measures the cell's
 * candidates and saves the winner before it returns (VFFT_FLAG_ESTIMATE
 * plans from formulas instead and measures nothing); execute is pure (no
 * allocation, no measurement, no decision). Which cells are served is the
 * front door's table: docs/design/vfft_front_door.md.
 */
#ifndef VFFT_H
#define VFFT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

  /* ── the axes of a configuration ─────────────────────────────────────── */

  /**
   * @brief The transform. Every backward direction is the unnormalized
   *        inverse: a roundtrip returns the input times the stated scale.
   *
   * - VFFT_C2C: X[k] = sum x[n] e^(-2 pi i n k / N); backward the conjugate
   *   sum; scale N.
   * - VFFT_R2C: real input, the half spectrum X[0..N/2] out (N/2 + 1 bins).
   * - VFFT_C2R: the half spectrum in, real output; backward of R2C, scale N.
   *   The imaginary parts of bin 0 and, at an even N, bin N/2 are outside the
   *   contract.
   * At rank 2 and above the data is row-major, the last axis contiguous, and
   * the real transforms halve the last axis (N_last/2 + 1 bins).
   *
   * Real-to-real, 1D only, N reals in and out, unnormalized:
   * - VFFT_DCT1: Y[k] = x[0] + (-1)^k x[N-1] + 2 sum_{n=1..N-2} x[n] cos(pi n k / (N-1));
   *   self-inverse, scale 2(N-1).
   * - VFFT_DCT2: Y[k] = 2 sum x[n] cos(pi k (2n+1) / 2N); inverse DCT-III, scale 2N. Even N.
   * - VFFT_DCT3: Y[k] = X[0] + 2 sum_{n>=1} X[n] cos(pi n (2k+1) / 2N); inverse DCT-II,
   *   scale 2N. Even N.
   * - VFFT_DCT4: Y[k] = 2 sum x[n] cos(pi (2k+1)(2n+1) / 4N); self-inverse, scale 2N. Even N.
   * - VFFT_DST1: Y[k] = 2 sum x[n] sin(pi (n+1)(k+1) / (N+1)); self-inverse, scale 2(N+1).
   * - VFFT_DST2: Y[k] = 2 sum x[n] sin(pi (k+1)(2n+1) / 2N); inverse DST-III, scale 2N. Even N.
   * - VFFT_DST3: Y[k] = (-1)^k X[N-1] + 2 sum_{n<=N-2} X[n] sin(pi (n+1)(2k+1) / 2N);
   *   inverse DST-II, scale 2N. Even N.
   * - VFFT_DHT: H[k] = sum x[n] (cos(2 pi n k / N) + sin(2 pi n k / N)); self-inverse, scale N.
   * VFFT_BACKWARD runs the inverse named above. Real-to-real plans take no
   * layout, order or batch geometry: those fields are ignored for them.
   */
  typedef enum
  {
    VFFT_C2C = 0,
    VFFT_R2C,
    VFFT_C2R,
    VFFT_DCT1,
    VFFT_DCT2,
    VFFT_DCT3,
    VFFT_DCT4,
    VFFT_DST1,
    VFFT_DST2,
    VFFT_DST3,
    VFFT_DHT
  } vfft_transform_t;

  /** @brief How complex data is stored. */
  typedef enum
  {
    VFFT_LAYOUT_INTERLEAVED = 0, /**< one array of adjacent (re, im) pairs: vfft_execute() */
    VFFT_LAYOUT_SPLIT = 1        /**< separate re[] and im[] planes: vfft_execute_split() */
  } vfft_layout_t;

  /** @brief Whether the output overwrites the input. */
  typedef enum
  {
    VFFT_PLACEMENT_OUT = 0, /**< separate input and output buffers */
    VFFT_PLACEMENT_IN = 1   /**< one buffer: out == in */
  } vfft_placement_t;

  /** @brief The order of the output bins. */
  typedef enum
  {
    VFFT_ORDER_NATURAL = 0,  /**< bin k at position k */
    VFFT_ORDER_SCRAMBLED = 1 /**< the engine's own permutation, faster where served;
                                  decodable only by the backward of the same plan */
  } vfft_order_t;

  /**
   * @brief Where the K transforms of a batch sit (howmany > 1; at K == 1
   *        the geometries are the same addressing).
   *
   * - VFFT_BATCH_CONTIGUOUS: element e of transform t at z[2 (t N + e)], each
   *   transform a block of its own, one after another.
   * - VFFT_BATCH_LANE_MAJOR: element e of transform t at z[2 (e K + t)], the
   *   transforms side by side, sample by sample. Real data: x[e K + t], bins
   *   z[2 (f K + t)].
   * - VFFT_BATCH_DEFAULT: contiguous for C2C, lane-major for R2C and C2R and
   *   for every SPLIT batch (whose only geometry it is).
   * A batch of rank 2 is contiguous planes by default, or lane-major when named
   * (C2C; R2C and C2R at N1 >= 2 and an even N2 >= 4): element (r, c) of
   * transform t at z[2 ((r N2 + c) K + t)]. A batch of rank 3 is contiguous volumes by
   * default, or lane-major when named (C2C; R2C and C2R at N1, N2 >= 2 and an
   * even N3 >= 4): element (i, j, k) of transform t at
   * z[2 ((i N2 N3 + j N3 + k) K + t)], real data x[(i N2 N3 + j N3 + k) K + t],
   * bins z[2 ((i N2 H + j H + f) K + t)] with H = N3/2 + 1. A batch of rank 4 is
   * not served.
   */
  typedef enum
  {
    VFFT_BATCH_DEFAULT = 0,
    VFFT_BATCH_CONTIGUOUS = 1,
    VFFT_BATCH_LANE_MAJOR = 2
  } vfft_batch_t;

  /** @brief The direction. VFFT_BACKWARD is the unnormalized inverse. */
  typedef enum
  {
    VFFT_FORWARD = 0,
    VFFT_BACKWARD = 1
  } vfft_dir_t;

  /** @brief vfft_config_t.flags: the planner's policy and the plan's permissions. */
  enum
  {
    VFFT_FLAG_PATIENT = 1u << 0,       /**< names the default: every create plans PATIENT
                                            (the wide sweep on a wisdom miss) */
    VFFT_FLAG_EXHAUSTIVE = 1u << 1,    /**< not served: vfft_create() refuses it */
    VFFT_FLAG_RECALIBRATE = 1u << 2,   /**< measure this cell again, even on a wisdom hit or
                                            after this process measured it, and save the winner */
    VFFT_FLAG_DESTROY_INPUT = 1u << 3, /**< an out-of-place plan may overwrite its input */
    VFFT_FLAG_OWN_PITCH = 1u << 4,     /**< the plan may choose its row pitch or lane stride;
                                            the caller reads it with vfft_info() */
    VFFT_FLAG_VERBOSE = 1u << 5,       /**< print refusals and warnings to stderr as well */
    VFFT_FLAG_ESTIMATE = 1u << 6       /**< plan by formulas from the lengths and this
                                            machine's caches: nothing is measured at create,
                                            no wisdom is read or written, and wisdom_dir is
                                            not opened. A VFFT_C2C transform, INTERLEAVED:
                                            1D in any order, placement and nthreads; rank 2
                                            and 3 in natural order at any nthreads, either
                                            placement; and a contiguous batch of them
                                            (howmany > 1) on one thread.
                                            vfft_create() refuses the flag on any
                                            other request, with VFFT_FLAG_PATIENT or
                                            VFFT_FLAG_RECALIBRATE, and while an environment
                                            variable that steers a plan (VFFT_IL2D_WL, ...)
                                            is set */
  };

  /* ── the configuration ──────────────────────────────────────────────── */

  /**
   * @brief The data contract vfft_create() commits to.
   *
   * A zeroed struct with n[0] set is one 1D C2C transform of n[0] points,
   * interleaved, out of place, natural order, one thread, served from the
   * library's wisdom store.
   */
  typedef struct
  {
    vfft_transform_t transform;
    int rank;                   /**< 0 or 1 = 1D; 2, 3 or 4 */
    int n[4];                   /**< the extents; the last axis is contiguous */
    size_t howmany;             /**< the batch count K; 0 or 1 = one transform */
    vfft_layout_t layout;
    vfft_placement_t placement;
    vfft_order_t order;
    vfft_batch_t batch;
    int nthreads;               /**< the threads the plan runs on, the caller's included;
                                     0 or 1 = one. Every threaded decision is measured
                                     at this count. */
    unsigned flags;             /**< VFFT_FLAG_* */
    const char *wisdom_dir;     /**< a wisdom store directory; NULL = the library's own */
  } vfft_config_t;

  /* ── create, execute, destroy ───────────────────────────────────────── */

  typedef struct vfft_plan_s *vfft_plan; /**< opaque, execute-ready */

  /**
   * @brief Commit a configuration to one plan.
   *
   * Validates the contract, serves the cell's verdict from wisdom or measures
   * its candidates on a miss (seconds at a small cell, minutes at the
   * largest) and saves the winner (VFFT_FLAG_ESTIMATE: planned from formulas
   * instead, nothing measured or saved), then builds the plan: kernels, tables,
   * scratch and threads are bound here. The plan owns what it allocated.
   * Creates take turns on one lock and may run beside executes. A create
   * leaves flush-to-zero and denormals-are-zero enabled on the calling thread,
   * the mode every kernel runs in.
   *
   * @return The plan, or NULL: vfft_error() says why.
   */
  vfft_plan vfft_create(const vfft_config_t *config);

  /**
   * @brief Run an INTERLEAVED plan, or a real-to-real plan.
   *
   * C2C: in and out are complex pairs. R2C: in is reals, out the half
   * spectrum; VFFT_FORWARD only. C2R: in is the half spectrum, out reals;
   * VFFT_BACKWARD only. Real-to-real: reals in and out. In place, out == in;
   * an in-place real plane holds 2 (N/2 + 1) doubles per transform (2D and 3D:
   * every row padded to 2 (N_last/2 + 1) doubles, a contiguous batch one such
   * plane or volume after another; a lane-major real batch: (N + 2) K doubles;
   * a lane-major 2D real batch: N1 blocks of (N2 + 2) K doubles; a lane-major
   * 3D real batch: N1 N2 rows of (N3 + 2) K doubles), the reals first and the
   * bins over them, and vfft_info() gives its size. A SPLIT plan is refused here.
   * Safe to call concurrently on different plans; a plan runs one execute at
   * a time. A call that runs on the plan's worker threads pins the calling
   * thread to the core the pool left to it, from its first fork-join to the
   * end of the call, and puts the thread's affinity back; a call the plan runs
   * serially leaves the thread alone, and a thread whose affinity does not hold
   * that core stays where it is.
   */
  void vfft_execute(vfft_plan p, vfft_dir_t dir, const void *in, void *out);

  /**
   * @brief Run a SPLIT plan: separate re and im planes.
   *
   * C2C: all four planes (in place: out_re == in_re, out_im == in_im). R2C:
   * in_re holds the reals, in_im is NULL. C2R: out_re receives the reals,
   * out_im is NULL. An INTERLEAVED plan is refused here.
   */
  void vfft_execute_split(vfft_plan p, vfft_dir_t dir,
                          const double *in_re, const double *in_im,
                          double *out_re, double *out_im);

  /** @brief Free a plan and everything it allocated. NULL is accepted. */
  void vfft_destroy(vfft_plan p);

  /* ── memory ─────────────────────────────────────────────────────────── */

  /**
   * @brief Allocate a buffer at the alignment every plan was measured on
   *        (vfft_info().alignment bytes). Uninitialised; release with
   *        vfft_free() only. NULL when memory runs out.
   */
  void *vfft_malloc(size_t bytes);
  /** @brief Release a vfft_malloc() block. NULL is accepted. */
  void vfft_free(void *ptr);

  /* ── facts ──────────────────────────────────────────────────────────── */

  /** @brief What vfft_info() reports. */
  typedef struct
  {
    const char *version;  /**< "MAJOR.MINOR.PATCH" */
    size_t alignment;     /**< bytes: vfft_malloc()'s, and the buffers the plans were measured on */
    int cores;            /**< this machine's performance cores: a hybrid CPU's P-cores, every
                               physical core otherwise; the nthreads that runs a plan on all of them */
    const char *route;    /**< the engine the plan committed to; "-" when it has no single name */
    int nthreads;         /**< the threads the plan runs on */
    size_t in_doubles;    /**< what execute reads, in doubles; per plane for a SPLIT plan */
    size_t out_doubles;   /**< what execute writes, in doubles; per plane for a SPLIT plan */
    size_t stride;        /**< a batch: element e of transform t at e * stride + t (complex
                               elements interleaved, doubles in a split plane); an in-place
                               rank-2 or rank-3 real plan: the doubles from one row to the
                               next. The tight value unless VFFT_FLAG_OWN_PITCH let the plan
                               choose; read it, never compute it. */
  } vfft_info_t;

  /**
   * @brief Fill a plan's facts; with p == NULL, the library's facts only.
   * @return 0, or -1 when out is NULL.
   */
  int vfft_info(vfft_plan p, vfft_info_t *out);

  /**
   * @brief Why the last vfft_create() or vfft_execute() on this thread was
   *        refused. Static storage, "" when nothing was refused since the
   *        thread's last create.
   */
  const char *vfft_error(void);

#ifdef __cplusplus
}
#endif
#endif /* VFFT_H */
