/* cpu_cache.h — L1 data-cache capacity, discovered once at PLAN time.
 *
 * 🔴 PLANNING ONLY. Nothing here may be called from an execute path.
 *
 * WHY IT IS DISCOVERED AND NOT HARD-CODED (Tugbars, 2026-08-02): the tcut tile
 * width is the first CACHE-OCCUPANCY quantity the library will bank. A chain or
 * a radix is a property of the transform and ports to any CPU; a tile width is
 * a property of *this machine's L1*, and on the wrong machine it fails as a
 * mild slowdown rather than an error — the worst thing to inherit silently. So
 * the value is discovered, stamped into the wisdom record next to the width,
 * and re-checked on replay; a mismatch means re-measure, never "use anyway".
 *
 * SCOPE — P-CORES ONLY (Tugbars). This CPU is hybrid and MEASURED
 * (benches/cpu_l1_probe.c) as:
 *     P (Raptor Cove) cpu 0-15 : L1d 48 KB, 12-way, L2 2 MB private
 *     E (Gracemont)   cpu 16-31: L1d 32 KB,  8-way, L2 4 MB shared
 * CPUID reports the cache of whichever core the query RUNS on, so the answer is
 * per-core-type, not per-machine. The library targets P-cores; E-core support
 * is left to a user extension. A width sized for a P-core is 150% of an E-core's
 * L1, and overshoot is the failure mode that costs everything at once rather
 * than degrading, so being on the wrong core type is not a rounding error.
 *

 */
#ifndef VFFT_CPU_CACHE_H
#define VFFT_CPU_CACHE_H

#include <stddef.h>

#if defined(__GNUC__) && (defined(__i386__) || defined(__x86_64__))
#include <cpuid.h>
#define VFFT_CPU_HAVE_CPUID 1
static inline void _vfft_cpuid(unsigned leaf, unsigned sub, unsigned r[4])
{
    __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
}
#elif defined(_MSC_VER)
#include <intrin.h>
#define VFFT_CPU_HAVE_CPUID 1
static inline void _vfft_cpuid(unsigned leaf, unsigned sub, unsigned r[4])
{
    int t[4]; __cpuidex(t, (int)leaf, (int)sub);
    r[0] = (unsigned)t[0]; r[1] = (unsigned)t[1];
    r[2] = (unsigned)t[2]; r[3] = (unsigned)t[3];
}
#else
#define VFFT_CPU_HAVE_CPUID 0
static inline void _vfft_cpuid(unsigned leaf, unsigned sub, unsigned r[4])
{ (void)leaf; (void)sub; r[0] = r[1] = r[2] = r[3] = 0; }
#endif

/* Intel SDM Vol.2, CPUID leaf 1AH, EAX[31:24].
 * 🔴 These were INVERTED on first use and every P-core printed as E. The
 * geometry is the cross-check: 48 KB / 12-way is Raptor Cove, 32 KB / 8-way is
 * Gracemont. A label alone cannot be falsified; a label beside its geometry can. */
#define VFFT_CPU_TYPE_ATOM 0x20u   /* E-core */
#define VFFT_CPU_TYPE_CORE 0x40u   /* P-core */

/* Used when discovery fails (virtualized, sandboxed, non-x86, unknown vendor).
 * Deliberately the SMALLER of the two core types: undershooting a tile degrades
 * gracefully, overshooting does not. */
#define VFFT_L1D_FALLBACK_BYTES (32 * 1024)

/* Our own measurement runs pin this so a stray query — or a thread that drifted
 * onto an E-core — can never resize a benchmark mid-campaign (Tugbars). Build
 * with -DVFFT_L1D_DISCOVER=1 to size from the live CPUID answer instead. The
 * discovered value is recorded either way, so the two can be compared. */
#ifndef VFFT_L1D_PCORE_BYTES
#define VFFT_L1D_PCORE_BYTES (48 * 1024)
#endif
#ifndef VFFT_L1D_DISCOVER
#define VFFT_L1D_DISCOVER 0
#endif

/* L2 (2026-08-25, the 2D band-threshold fence and any future L2-sized
 * decision). Same discipline as L1d: PINNED for our own measurement runs,
 * discovery under the SAME opt-in knob (one switch governs cache
 * discovery, not one per level). The E-core caveat is sharper here: a
 * Gracemont module's 4 MB L2 is SHARED by 4 cores, so sizing a private-L2
 * decision off an E-core read overshoots by up to 4x — the refuse rule
 * below treats it exactly like the L1 case. Fallback = the P-core private
 * size, the smaller effective figure on this hybrid. */
#ifndef VFFT_L2_PCORE_BYTES
#define VFFT_L2_PCORE_BYTES (2 * 1024 * 1024)
#endif
#define VFFT_L2_FALLBACK_BYTES (1024 * 1024) /* undershoot degrades gracefully */

typedef struct {
    long     l1d_used;      /* what sizing decisions MUST use, and what gets
                             * stamped into wisdom                           */
    long     l1d_seen;      /* what CPUID reported here, 0 if unavailable    */
    long     l1d_ways;
    long     l2_used;       /* the L2 sizing value (same contract as l1d)    */
    long     l2_seen;       /* CPUID level-2 unified/data size, 0 if none    */
    unsigned core_type;     /* VFFT_CPU_TYPE_*, 0 = not hybrid / unknown     */
    int      is_pcore;
    int      discovered;    /* 1 = l1d_used came from CPUID                  */
    int      geometry_ok;   /* type label agrees with the cache geometry     */
} vfft_cpu_cache_t;

static inline void _vfft_cpu_cache_fill(vfft_cpu_cache_t *o)
{
    unsigned r[4];
    o->l1d_seen = 0; o->l1d_ways = 0; o->core_type = 0;
    o->is_pcore = 0; o->discovered = 0; o->geometry_ok = 1;
    o->l2_seen = 0;

#if VFFT_CPU_HAVE_CPUID
    _vfft_cpuid(0, 0, r);
    const unsigned maxleaf = r[0];

    if (maxleaf >= 0x1A) {
        _vfft_cpuid(0x1A, 0, r);
        o->core_type = (r[0] >> 24) & 0xFFu;
    }
    /* A non-hybrid CPU reports no core type; treat it as usable. */
    o->is_pcore = (o->core_type == VFFT_CPU_TYPE_CORE) || (o->core_type == 0);

    if (maxleaf >= 4) {
        for (unsigned sub = 0; sub < 16u; sub++) {
            _vfft_cpuid(4, sub, r);
            const int ctype = (int)(r[0] & 0x1F);        /* 1=data 3=unified */
            const int level = (int)((r[0] >> 5) & 0x7);
            if (ctype == 0) break;
            if (level == 1 && ctype == 1) {
                const long ways  = (long)(((r[1] >> 22) & 0x3FF) + 1);
                const long parts = (long)(((r[1] >> 12) & 0x3FF) + 1);
                const long line  = (long)((r[1] & 0xFFF) + 1);
                const long sets  = (long)(r[2] + 1);
                o->l1d_seen = ways * parts * line * sets;
                o->l1d_ways = ways;
            }
            if (level == 2 && (ctype == 1 || ctype == 3)) {
                const long ways  = (long)(((r[1] >> 22) & 0x3FF) + 1);
                const long parts = (long)(((r[1] >> 12) & 0x3FF) + 1);
                const long line  = (long)((r[1] & 0xFFF) + 1);
                const long sets  = (long)(r[2] + 1);
                o->l2_seen = ways * parts * line * sets;
            }
        }
    }

    /* Cross-check the label against the geometry (see the note above). */
    if (o->core_type == VFFT_CPU_TYPE_CORE)
        o->geometry_ok = (o->l1d_seen >= 48 * 1024 && o->l1d_ways >= 12);
    else if (o->core_type == VFFT_CPU_TYPE_ATOM)
        o->geometry_ok = (o->l1d_seen <= 32 * 1024 && o->l1d_ways <= 8);
#else
    (void)r;
#endif

#if VFFT_L1D_DISCOVER
    if (o->l1d_seen > 0 && o->is_pcore && o->geometry_ok) {
        o->l1d_used = o->l1d_seen;
        o->discovered = 1;
    } else {
        o->l1d_used = VFFT_L1D_FALLBACK_BYTES;   /* refuse to size off an
                                                  * E-core or a bad read   */
    }
    if (o->l2_seen > 0 && o->is_pcore && o->geometry_ok)
        o->l2_used = o->l2_seen;
    else
        o->l2_used = VFFT_L2_FALLBACK_BYTES;     /* same refuse rule; the
                                                  * E-core L2 is SHARED    */
#else
    o->l1d_used = VFFT_L1D_PCORE_BYTES;          /* pinned for our own runs */
    o->l2_used = VFFT_L2_PCORE_BYTES;
#endif
}

/* Cached. The FIRST call performs CPUID; make sure that first call happens
 * during planning. */
static inline const vfft_cpu_cache_t *vfft_cpu_cache(void)
{
    static vfft_cpu_cache_t c;
    static int done = 0;
    if (!done) { _vfft_cpu_cache_fill(&c); done = 1; }
    return &c;
}

/* The capacity every cache-sizing decision must use, and the value that gets
 * stamped into a wisdom record beside the width it produced. */
static inline long vfft_cpu_l1d_bytes(void) { return vfft_cpu_cache()->l1d_used; }

/* The L2 twin (2026-08-25): the capacity every L2-sized decision must use
 * (first consumer: the 2D band-threshold fence N1_max = L2/(16*wl_min)),
 * and the value stamped beside any banked verdict that depended on it. */
static inline long vfft_cpu_l2_bytes(void) { return vfft_cpu_cache()->l2_used; }

static inline int vfft_cpu_l2_matches(long stamped)
{
    return stamped <= 0 || stamped == vfft_cpu_l2_bytes();
}

/* Replay check: a banked width is only valid on a machine with the same L1. */
static inline int vfft_cpu_l1d_matches(long stamped)
{
    return stamped <= 0 || stamped == vfft_cpu_l1d_bytes();
}

#endif /* VFFT_CPU_CACHE_H */
