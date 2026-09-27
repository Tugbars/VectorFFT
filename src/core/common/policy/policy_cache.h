/* policy_cache.h - L8, a working set against the cache the CPU reports.
 *
 * Layout-neutral (today's callers are interleaved: the 2D/3D tiers and the
 * four-step super-band). Carved verbatim out of planning/policy.h (layout
 * separation phase 4). Needs cpu_cache.h in scope. */
#ifndef VFFT_POLICY_CACHE_H
#define VFFT_POLICY_CACHE_H

/* -- L8. a candidate's working set against the hardware -----------------
 * TWO helpers, and NEITHER is the negation of the other. Both the POLARITY
 * and the UNKNOWN-SIZE policy are written into the name and the body,
 * because the two differ in both: one shared fits() flips the super-band
 * exactly backwards.
 *
 * Each takes the ALREADY-COMPUTED byte count as a long, and the multiply
 * stays at the call site on purpose: long is 32-bit on this MinGW build, so
 * taking the factors here -- or widening -- would change the wrap behaviour
 * of an expression like (long)N1 * w * 16.
 *
 * CONTRACT, both helpers: the argument is a POSITIVE working-set size. The
 * cache term is inert for any positive argument, whatever the hardware
 * reports; it decides the answer only when bytes <= 0. Never hand either
 * one a difference or a wrapped product. */

/* The L2 ladder (five sites: the 2D tier's strip, real-wl and cascade
 * widths, the 3D tier's strip and wl widths). ADMIT what fits the L2 the
 * CPU reports; the ladder is a candidate list and the race still decides.
 * UNKNOWN SIZE => REFUSE -- a ladder that cannot measure the cache
 * contributes nothing and the caller keeps its ungated static pool. That
 * rule is defensive rather than live (vfft_cpu_l2_bytes installs a fallback
 * and is never 0, cpu_cache.h), and it is stated because the contrast with
 * the L3 rule below is the whole reason there are two functions. */
static inline int vfft_policy_fits_l2(long bytes)
{
    const long l2 = vfft_cpu_l2_bytes();
    return l2 > 0 && bytes <= l2;
}

/* The super-band's gate (one site: _k1fs_sb_admit). The OPPOSITE law -- the
 * form is an arm only where the plane OUTGROWS the last-level cache -- so
 * it admits what does NOT fit. UNKNOWN SIZE => ADMIT, and
 * this one is LIVE: vfft_cpu_l3_bytes returns l3_seen, which has no
 * fallback and is genuinely 0 on an L3-less part, where "bigger than L3" is
 * vacuously true and the form is admitted everywhere.
 * NEVER write this as a negation of the L2 helper: both terms would flip
 * and every L3-less host would lose the super-band. */
static inline int vfft_policy_exceeds_l3(long bytes)
{
    const long l3 = vfft_cpu_l3_bytes();
    return l3 <= 0 || bytes > l3;
}

#endif /* VFFT_POLICY_CACHE_H */
