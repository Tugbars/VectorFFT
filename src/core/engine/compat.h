/* compat.h — signature-adapter shims (n1 6-arg ↔ 7-arg, etc.)
 *
 * [Phase 4 placeholder — populated only if needed.]
 *
 * Two ways to resolve the n1 signature mismatch between prototype
 * codelets (6-arg in-place: rio_re, rio_im, tw_re, tw_im, ios, me) and
 * production-style n1 function pointers (7-arg OOP: in_re, in_im,
 * out_re, out_im, is, os, vl):
 *
 *   Option A: regenerate prototype n1 codelets with the 7-arg OOP
 *             signature. Requires gen_radix.ml edits + full regen.
 *             Permanent fix; codelets become drop-in production-
 *             compatible. ~1 hour of work.
 *
 *   Option B: this file. Provide a thin static-inline wrapper that
 *             adapts the 7-arg call to the 6-arg implementation.
 *             Zero codelet changes; minor call-site overhead (one
 *             extra branch on `in_re == out_re`, then drop the
 *             redundant args). Half-day of work.
 *
 * Pick Phase 4's resolution based on whether the prototype tree's
 * n1 codelets ever need to be called from production code paths.
 *
 * Until Phase 4, prototype-core's executor will call n1 in the
 * prototype convention (6-arg, NULL for tw pointers, ios=stride,
 * me=slice_K) — see how plan_executors.h emits n1 calls today.
 */
#ifndef VFFT_PROTO_CORE_COMPAT_H
#define VFFT_PROTO_CORE_COMPAT_H

#endif /* VFFT_PROTO_CORE_COMPAT_H */
