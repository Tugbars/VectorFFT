/* real_dispatch_config.h - INTERNAL. Cross-TU setters for the r2c/c2r
 * dispatch knobs.
 *
 * WHY THESE ARE NOT `static inline` IN THE DISPATCH HEADERS
 * --------------------------------------------------------
 * The knobs are file-scope state in transforms/real/{r2c,c2r}_dispatch.h, and
 * the setters beside them are `static inline`. That is correct for the TU that
 * IS the library and silently wrong for every other one: a bench that includes
 * the dispatch header while linking vfft.c separately writes ITS OWN copy,
 * while vfft_create keeps reading the library's. The write appears to succeed
 * and changes nothing.
 *
 * MEASURED CONSEQUENCE: bench_1d_vs_mkl's VFFT_C2R_PACK_ALL and
 * VFFT_C2R_STRIDE_ALL probe arms were INERT - both "forced-route" arms
 * measured the same route, and the comparison looked like a result.
 *
 * The functions declared here are DEFINED IN vfft.c with external linkage, so
 * they write the copy vfft_create reads. Do not reintroduce a `static inline`
 * spelling of any of them.
 *
 * WHY INTERNAL RATHER THAN PUBLIC
 * -------------------------------
 * They configure where the library's own routing crossover sits -- a
 * calibration artifact, not a transform parameter. The default of 32 is the
 * N=256 crossover, not a universal one, which is precisely why it is a knob
 * for measurement rather than a setting for callers.
 */
#ifndef VFFT_REAL_DISPATCH_CONFIG_H
#define VFFT_REAL_DISPATCH_CONFIG_H

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

  /* ── CROSS-TU CONFIGURATION HOOKS ─────────────────────────────────────
   * The r2c/c2r dispatch knobs live as file-scope state in
   * transforms/real/{r2c,c2r}_dispatch.h, and their setters there are
   * `static inline`. That is correct for a TU that IS the library, and
   * silently wrong for anyone else: a bench that includes the header and
   * links vfft.c separately writes ITS OWN copy, while vfft_create keeps
   * reading the library's. The write appears to succeed and changes nothing.
   *
   * These entry points are compiled INTO vfft.c, so they write the copy
   * vfft_create actually reads. Any TU outside the library that wants to
   * configure the real-transform dispatch must go through them.
   *
   * (Measured consequence of not having them: bench_1d_vs_mkl's
   * VFFT_C2R_PACK_ALL / VFFT_C2R_STRIDE_ALL probe arms were INERT - both
   * forced-route arms measured the same route.) */

  /* Batch-size crossover between the packed rfft cascade (K below) and the
   * decoupled stride path (K at or above). SIZE_MAX forces packed for every
   * K; 0 forces stride. Default 32 - which is the N=256 crossover, not a
   * universal one. */
  void   vfft_r2c_set_decouple_min_k(size_t k);
  size_t vfft_r2c_get_decouple_min_k(void);

  /* Load a calibrated per-cell c2r route table. Returns non-zero on success;
   * on a miss the dispatch falls back to the threshold above. */
  int    vfft_c2r_load_path(const char *path);

#ifdef __cplusplus
}
#endif
#endif /* VFFT_REAL_DISPATCH_CONFIG_H */
