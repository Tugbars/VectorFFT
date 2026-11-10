/**
 * @file vfftf.h
 * @brief The FP32 (single precision) VectorFFT API: the plan functions of
 *        vfft.h for float data.
 *
 *   vfftf_create        commit a configuration to one FP32 plan
 *   vfftf_execute       run it
 *   vfftf_destroy       free it
 *   vfftf_info          its facts
 *
 * vfft_malloc, vfft_free and vfft_error (vfft.h) serve both precisions.
 *
 * What is served at FP32: VFFT_C2C, rank 1, power-of-two n[0], INTERLEAVED,
 * in place or out of place, natural order, one transform or a batch, any
 * nthreads. Every other configuration is refused: vfftf_create() returns NULL
 * and vfft_error() says why. The configuration struct, the flags and the
 * directions are vfft.h's; the data is complex float pairs (re, im).
 *
 * FP32 has its own wisdom: wisdom2f_oop.txt in the same store directory as
 * the FP64 shards. Neither precision reads the other's file.
 *
 * The library carries FP32 only when it was built with VFFT_ENABLE_FLOAT
 * (CMake option, OFF by default).
 */
#ifndef VFFTF_H
#define VFFTF_H

#include "vfft.h"

#ifdef __cplusplus
extern "C"
{
#endif

  typedef struct vfftf_plan_s *vfftf_plan; /**< opaque, execute-ready */

  /**
   * @brief Commit a configuration to one FP32 plan (vfft_create's contract,
   *        for float data). @return The plan, or NULL: vfft_error() says why.
   */
  vfftf_plan vfftf_create(const vfft_config_t *config);

  /**
   * @brief Run an FP32 plan: in and out are complex float pairs; in place,
   *        out == in. Safe to call concurrently on different plans.
   */
  void vfftf_execute(vfftf_plan p, vfft_dir_t dir, const void *in, void *out);

  /** @brief Free an FP32 plan and everything it allocated. NULL is accepted. */
  void vfftf_destroy(vfftf_plan p);

  /**
   * @brief Fill an FP32 plan's facts; with p == NULL, the library's facts.
   *        The in_doubles and out_doubles fields count FLOATS for an FP32 plan.
   * @return 0, or -1 when out is NULL.
   */
  int vfftf_info(vfftf_plan p, vfft_info_t *out);

#ifdef __cplusplus
}
#endif
#endif /* VFFTF_H */
