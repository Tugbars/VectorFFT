/* vfft_diagnostics.h - did the threading actually happen?
 *
 * WHY THIS IS A SEPARATE HEADER
 * -----------------------------
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

  /* And for the K=1 1D cascade (zturn): threaded cascade walks actually
   * run. The verdict is raced per cell at create (VFFT_ZT_NO_MT=1 kills,
   * =0 forces — the A/B hook); zero here after an execute means the
   * serial walk served. */
  long vfft_zt_mt_passes(void);

  /* And for the 2D plane queue (dims=2, howmany>1): queued (plane-per-
   * worker) executes actually run. Loop-vs-queue is raced at create
   * (VFFT_PQ_NO_MT=1 kills, =0 forces); zero after an execute means the
   * serial plane loop served (which still intra-MTs per the inner
   * plan's own banked verdicts). */
  long vfft_pq_mt_passes(void);

#ifdef __cplusplus
}
#endif
#endif /* VFFT_DIAGNOSTICS_H */
