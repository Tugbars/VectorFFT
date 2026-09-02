/* vfft_fingerprint.h — the create-time PLAN FINGERPRINT.
 *
 * WHY THIS EXISTS
 * ---------------
 * The refactor's characteristic failure is not a wrong answer - 21 of the 32
 * gates already assert bit-identity. It is "still correct, but a DIFFERENT plan
 * was chosen". Nothing in the public API can see that: include/vfft.h exposes
 * vfft_plan_planes, vfft_plan_stride, vfft_plan_tc_workers and the engagement
 * counters, and not one route, chain or kernel-form field. Today the only way
 * to ask a handle what it built is to #include "vfft.c" textually, which four
 * bench TUs do.
 *
 * This header answers that question through one internal symbol instead.
 *
 * WHAT IS IN A FINGERPRINT
 * ------------------------
 * A field belongs iff it was CHOSEN at create - by a wisdom lookup, a race, an
 * env hook or a heuristic - or it determines output bits. Pointers, addresses,
 * byte sizes and buffer identities are excluded BY CONSTRUCTION: they vary run
 * to run and would make the artifact undiffable. A subplan therefore appears as
 * a PRESENCE bit, never as an address.
 *
 * create recurses (zr2c, the transform-contiguous wrapper, the plane queue, the
 * IL2D row child), so children are emitted as depth-prefixed lines and the
 * whole thing reads as a tree.
 *
 * WHY TEXT AND NOT A HASH
 * -----------------------
 * Named key=value tokens, deliberately shaped like the @vw2 cell grammar so the
 * same eye reads both. A hash would say only THAT something changed; across a
 * multi-hundred-cell sweep that is untriageable. Named tokens also mean adding a
 * field APPENDS a token rather than shifting columns, so an unrelated later
 * change does not reflow the whole file.
 *
 * NO TIMINGS, ENFORCED MECHANICALLY
 * ---------------------------------
 * The emitter's only value types are int, long and const char*. No
 * floating-point conversion appears anywhere in the emitter, and that is
 * mechanically checkable - scan the EMITTER BODY in vfft.c (the section under
 * the VFFT_FINGERPRINT guard) for float conversions and expect zero. Do not
 * scan this header: a prose description of the check would match itself, which
 * is exactly the false positive the first version of this comment produced.
 *
 * A timing leaking into a supposedly clock-free artifact would make the baseline
 * undiffable and quietly destroy every check that depends on it.
 *
 * ABI
 * ---
 * Compiled only under -DVFFT_FINGERPRINT, exporting exactly one internal symbol
 * with a double underscore, declared HERE and never in include/vfft.h. The
 * public ABI is untouched. Gate TUs include this header and link the same
 * vfft.o - strictly cleaner than the textual-include workaround.
 *
 * WHAT IT CANNOT SEE
 * ------------------
 * It is CREATE-ONLY. The execute-side dispatch - the trig switch, the layout
 * branch, the MT engage decision - is invisible to it. A deleted `case` that
 * falls through to a correct `default` leaves the fingerprint byte-identical.
 * "The fingerprint is clean" must never be read as "the path is covered"; the
 * golden output-bit digests cover that half.
 */
#ifndef VFFT_FINGERPRINT_H
#define VFFT_FINGERPRINT_H

#ifdef VFFT_FINGERPRINT

#include <stddef.h>

/* Fill `out` with the fingerprint of `h`. Returns the number of bytes that
 * would have been written (snprintf semantics), so a truncated buffer is
 * detectable rather than silent. */
size_t vfft__fingerprint(void *h, char *out, size_t cap);

/* Every engagement counter through ONE accessor, so a new counter costs no
 * public-ABI decision. Order is fixed and documented; a caller reads by index.
 *   0 tc_mt_dispatches   1 il2d_col_mt_passes   2 zt_mt_passes
 *   3 pq_mt_passes       4 trig_mt_passes       5 create_races
 * Index 5 is the REPLAY-PURITY counter: under replay it must stay at zero, or
 * the clock is inside the baseline and the differential test is not one. */
#define VFFT__FP_NCOUNTERS 6
void vfft__fp_counters(long *out6);

#endif /* VFFT_FINGERPRINT */
#endif /* VFFT_FINGERPRINT_H */
