/* vfft_diagnostics.h - did the threading actually happen?
 *
 * vfft.h is the transform contract: create, execute, destroy, and the wisdom
 * a plan is built from. Nothing here is needed to compute an FFT, so none of
 * it belongs in that contract. It is still SHIPPED rather than hidden,
 * because the question it answers is not a test-only question -- anyone who
 * asks this library for 8 threads has the same reason to check they got them.
 *
 * WHY ASKING IS NECESSARY AT ALL
 * ------------------------------
 * Every count below can be legitimately ZERO on a plan that is entirely
 * correct: clones are built conditionally (pool size, whether the inner route
 * is pool-free, whether each clone came out output-equivalent), and dispatch
 * is decided separately again (a cell under the engage floor runs the serial
 * loop even when clones exist). A serial plan returns the right answer, so
 * NO correctness test can see the difference -- not even a bitwise
 * MT-equals-ST comparison, which passes just as happily when no thread ever
 * ran. A threading assertion that does not read these numbers cannot fail.
 *
 * BUILT and DISPATCHED are two separate gates. vfft_plan_tc_workers answers
 * the first; only a counter that MOVED across an execute answers the second.
 *
 * This is not idle caution: it is how a live bug was caught in which 2D real
 * create destroyed the process thread pool.
 */
#ifndef VFFT_DIAGNOSTICS_H
#define VFFT_DIAGNOSTICS_H

#include "vfft.h"   /* vfft_plan */

#ifdef __cplusplus
extern "C"
{
#endif

  /* DIAGNOSTIC — how many WORKER threads this plan's transform-contiguous
   * batch wrapper actually built, or -1 if the plan is not such a wrapper.
   * 0 means the wrapper exists but executes its batch serially.
   *
   * Exists because clone-building is conditional (pool size, whether the
   * inner route is pool-free, whether each clone came out output-equivalent)
   * and every one of those can quietly reduce the count to zero. A serial
   * wrapper still returns correct results, so a correctness test — including
   * an MT-equals-ST bitwise comparison — passes just as happily when no
   * thread ever ran. Tests that mean to assert THREADING must assert on this,
   * and benches should report it rather than assume the thread count they
   * asked for is the thread count they got. */
  int vfft_plan_tc_workers(vfft_plan p);

  /* Process-lifetime count of transform-contiguous MT DISPATCHES. Clones
   * built and work dispatched are INDEPENDENT gates: a plan can own clones
   * and still run its serial loop because the cell sits under the engage
   * floor. Assert this MOVED across an execute to prove threading actually
   * happened; vfft_plan_tc_workers alone does not. */
  long vfft_tc_mt_dispatches(void);

  /* Same question for the native IL 2D real COLUMN pass: how many
   * threaded column passes actually ran. Zero after an execute means the
   * column pass was serial (too few independent units, or no pool). */
  long vfft_il2d_col_mt_passes(void);
  /* the rank-N INTERLEAVED c2c tier (3D IL): threaded executes (band or
   * plane arm engaged); serial serves leave it unchanged. */
  long vfft_ilnd_mt_passes(void);
  /* the flat mixed-radix DIT (odd-N K=1 interleaved): threaded executes of
   * the blocks or tiles arm; serial serves leave it unchanged. */
  long vfft_ilfd_mt_passes(void);
  /* ZTURN-T's threaded arm (the staged walk sectioned): threaded executes
   * actually run; raced per T at create, VFFT_ZTT_MT=0|1|2 pins. */
  long vfft_ztt_mt_passes(void);
  /* the real four-step's fused order sweeps (1D r2c/c2r from 2^20): sweeps
   * cut across the pool's workers; a serial sweep leaves it unchanged. */
  long vfft_zfsr_mt_passes(void);
  /* ZTT-r's threaded arms (the ZTT's walk with the real fold fused):
   * threaded executes actually run; a serial serve leaves it unchanged. */
  long vfft_zttr_mt_passes(void);
  long vfft_zrf_mt_passes(void);   /* the real flat DIT's threaded executes */
  long vfft_ilpr_mt_passes(void);  /* the prime cell's (Rader / Bluestein) threaded executes */
  long vfft_zrb_mt_passes(void);   /* the real Bluestein's threaded executes */
  /* zr2c's Hermitian fold cut across the pool's workers (a raced plan
   * input at T > 1); a serial fold leaves it unchanged. */
  long vfft_zr2c_fold_mt_passes(void);
  /* the flat DIT's create-time races (forms, tile): arms whose timed batch
   * was under half the sample target. A property, not an outcome: 0 means
   * every verdict was decided above the clock's tick. */
  long vfft_ilfd_race_short_samples(void);

  /* And for the 2D plane queue (dims=2, howmany>1): queued (plane-per-
   * worker) executes actually run. Loop-vs-queue is raced at create
   * (VFFT_PQ_NO_MT=1 kills, =0 forces); zero after an execute means the
   * serial plane loop served (which still intra-MTs per the inner
   * plan's own banked verdicts). */
  long vfft_pq_mt_passes(void);

  /* Measurement scopes entered by this process (vfft.h, "the measurement
   * scope"): one per vfft_create that raced, one per outermost
   * vfft_measure_begin. A create served from wisdom leaves it unchanged. */
  long vfft_measure_scopes(void);

  /* ── THE MEASUREMENT SCOPE AS A TOOL (moved here from vfft.h, 2026-10-07:
   * docs/design/public_header_surface.md). Timing your own code under the
   * scope vfft_create() races in is benchmarking, not computing a transform.
   * vfft_measure_configure() stays in vfft.h: it changes what vfft_create()
   * does to the process. ─────────────────────────────────────────────── */

  /** @brief What vfft_measure_begin() returns (0 = a clean scope). */
  enum
  {
    VFFT_MEASURE_CONTENDED = 1, /**< the lock was not obtained within the wait */
    VFFT_MEASURE_UNPINNED = 2   /**< the thread is not pinned to a P-core */
  };

  /**
   * @brief Enter the measurement scope on the calling thread, for code the
   *        caller times itself.
   *
   * The same scope vfft_create() uses for its races: the machine-wide lock,
   * the pin, the sibling guard, the priority (vfft_measure_configure() in
   * vfft.h sets them). A vfft_create() inside it adds nothing and leaves it
   * in place. Calls nest; the scope ends at the matching outermost
   * vfft_measure_end(), on the same thread.
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

  /**
   * @brief The SIMD level this build was compiled for (a fact for logs; the
   *        same fact is part of vfft_wisdom_identity(), and buffer alignment
   *        comes from vfft_alignment() in vfft.h).
   * @return "avx512", "avx2" or "scalar": a build-time fact, not runtime
   *         detection. Static storage.
   */
  const char *vfft_isa(void);

#ifdef __cplusplus
}
#endif
#endif /* VFFT_DIAGNOSTICS_H */
