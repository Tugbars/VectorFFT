/* wisdom_reader.h — parse production's wisdom file.
 *
 * [Phase 3 placeholder — will be populated when Phase 3 lands.]
 *
 * Reads build_tuned/vfft_wisdom_tuned.txt and builds an in-memory
 * table of plan recipes:
 *
 *   N K factors[nf] variants[nf] use_dif_forward
 *
 * Consumed by planner.h's wisdom-driven path. The file format is
 * documented in production's planner; we just re-implement the
 * reader here so prototype-core stays independent of src/core/.
 *
 * Format (single line per entry):
 *
 *   N K nf factor_1 factor_2 ... factor_nf best_ns \
 *     use_blocked split_stage block_groups use_dif_forward \
 *     variant_1 variant_2 ... variant_nf
 *
 * Where variant_i is one of:
 *   0 = FLAT, 1 = LOG3, 2 = T1S, 3 = BUF (unused in current wisdom)
 *
 * Scope: read-only. prototype-core doesn't produce wisdom; that's the
 * job of the calibration harness in production. If we eventually
 * port the calibrator too, it gets its own writer.
 */
#ifndef VFFT_PROTO_CORE_WISDOM_READER_H
#define VFFT_PROTO_CORE_WISDOM_READER_H

#include <stddef.h>

/* Phase 3 will add:
 *
 *   typedef struct {
 *       int     N;
 *       size_t  K;
 *       int     nf;
 *       int     factors[STRIDE_MAX_STAGES];
 *       int     variants[STRIDE_MAX_STAGES];
 *       int     use_dif_forward;
 *   } vfft_proto_wisdom_entry_t;
 *
 *   typedef struct {
 *       vfft_proto_wisdom_entry_t *entries;
 *       size_t                     count;
 *   } vfft_proto_wisdom_t;
 *
 *   int  vfft_proto_wisdom_load(vfft_proto_wisdom_t *wis,
 *                               const char *path);
 *   const vfft_proto_wisdom_entry_t *
 *        vfft_proto_wisdom_lookup(const vfft_proto_wisdom_t *wis,
 *                                 int N, size_t K);
 *   void vfft_proto_wisdom_free(vfft_proto_wisdom_t *wis);
 */

#endif /* VFFT_PROTO_CORE_WISDOM_READER_H */
