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
#include <stdio.h>    /* snprintf, for the host tag */
#include <stdlib.h>   /* malloc/free/atoi in the OS tier */
#include <string.h>   /* vendor-string compare in _vfft_cpu_cache_fill */

/* -DVFFT_CPU_DISABLE_CPUID skips the instruction entirely and discovery falls
 * through to the OS tier below — for sandboxes that trap CPUID, and for
 * exercising the OS tier on hardware that has CPUID. */
#if !defined(VFFT_CPU_DISABLE_CPUID) && \
    defined(__GNUC__) && (defined(__i386__) || defined(__x86_64__))
#include <cpuid.h>
#define VFFT_CPU_HAVE_CPUID 1
static inline void _vfft_cpuid(unsigned leaf, unsigned sub, unsigned r[4])
{
    __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
}
#elif !defined(VFFT_CPU_DISABLE_CPUID) && \
    defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
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

/* ── VENDOR (2026-09-03) ─────────────────────────────────────────────────────
 * 🔴 MEASURED on a Ryzen 5 PRO 8640HS (Zen 4), benches/cpu_l1_probe.c:
 *     vendor=AuthenticAMD  max_cpuid_leaf=0xD
 *     every one of 12 logical CPUs: L1d 0K, ways 0, sets 0, L2 0K, L3 0K
 * Leaf 4 is an INTEL leaf. AMD answers it with zeros and publishes the same
 * information through the EXTENDED leaves instead, so the discovery below
 * found nothing at all on that host and silently handed back the pinned
 * Raptor-Cove numbers (48 KB / 2 MB) for a part that has 32 KB / 1 MB. That is
 * the overshoot this file's own header calls the failure mode that "costs
 * everything at once rather than degrading" — reached by a read that returned
 * NO DATA, which is worse than a wrong number because nothing looked amiss.
 *
 * The AMD twins, both same-format as their Intel counterparts:
 *   0x8000001D  deterministic cache parameters — bit-for-bit the leaf-4
 *               encoding (EAX type/level, EBX ways/partitions/line, ECX sets),
 *               walked over subleaves the identical way.
 *   0x8000001E  topology extended — EBX[15:8] = threads-per-core MINUS ONE,
 *               the SMT width that leaf 0xB carries on Intel.
 * Both are gated on TOPOEXT (0x80000001 ECX bit 22) and on the extended
 * max-leaf, so a pre-Zen part that lacks them reports 0 and takes the
 * fallback rather than reading garbage. */
#define VFFT_CPU_VENDOR_UNKNOWN 0
#define VFFT_CPU_VENDOR_INTEL   1
#define VFFT_CPU_VENDOR_AMD     2

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
    long     l3_seen;       /* CPUID level-3 unified size, 0 if none. SHARED
                             * by every core — the only legitimate use is an
                             * AGGREGATE budget (e.g. "do T concurrent
                             * working sets fit?"), never per-core sizing.  */
    int      smt;           /* logical processors per physical core (CPUID
                             * leaf 0xB level-0 on Intel, 0x8000001E on AMD),
                             * 1 = no SMT, 0 = unknown                       */
    int      vendor;        /* VFFT_CPU_VENDOR_* — which CPUID dialect the
                             * geometry above was read through              */
    unsigned core_type;     /* VFFT_CPU_TYPE_*, 0 = not hybrid / unknown     */
    int      is_pcore;
    int      discovered;    /* 1 = l1d_used came from CPUID                  */
    int      geometry_ok;   /* type label agrees with the cache geometry     */
} vfft_cpu_cache_t;

/* Walk the deterministic-cache-parameter subleaves of `leaf` and fill the
 * L1d/L2/L3 fields. Intel spells this leaf 4, AMD spells it 0x8000001D, and
 * the register encoding is IDENTICAL — so the decode lives once here and the
 * only per-vendor decision is which leaf number to hand it. */
static inline void _vfft_cpu_walk_cache_leaf(vfft_cpu_cache_t *o, unsigned leaf)
{
    unsigned r[4];
    for (unsigned sub = 0; sub < 16u; sub++) {
        _vfft_cpuid(leaf, sub, r);
        const int ctype = (int)(r[0] & 0x1F);        /* 1=data 3=unified */
        const int level = (int)((r[0] >> 5) & 0x7);
        if (ctype == 0) break;
        const long ways  = (long)(((r[1] >> 22) & 0x3FF) + 1);
        const long parts = (long)(((r[1] >> 12) & 0x3FF) + 1);
        const long line  = (long)((r[1] & 0xFFF) + 1);
        const long sets  = (long)(r[2] + 1);
        const long size  = ways * parts * line * sets;
        if (level == 1 && ctype == 1) { o->l1d_seen = size; o->l1d_ways = ways; }
        if (level == 2 && (ctype == 1 || ctype == 3)) o->l2_seen = size;
        if (level == 3 && ctype == 3)                 o->l3_seen = size;
    }
}

/* ── OS TIER ────────────────────────────────────────────────────────────────
 * Fills only what CPUID left at ZERO, so on an x86 host with a working CPUID
 * it is a no-op. It exists for the hosts CPUID cannot describe: ARM64 (Apple
 * silicon, Windows on Snapdragon, Graviton), virtualized guests whose
 * hypervisor masks leaf 4, and sandboxes that trap the instruction. It reports
 * the same four quantities (L1d, L2, L3, SMT) through the same struct, so no
 * consumer changes and the seen/used split above it is untouched.
 *
 * Limit, stated once: the OS reports the cache of "a" core without saying
 * which type. Hybrid parts are x86 and always have CPUID, so this tier never
 * sizes a hybrid host; a non-hybrid host has one answer. */
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif
#ifndef NOMINMAX
#define NOMINMAX 1
#endif
#include <windows.h>
static inline void _vfft_cpu_os_fill(vfft_cpu_cache_t *o)
{
    DWORD len = 0;
    char *buf, *p, *end;
    GetLogicalProcessorInformationEx(RelationAll, NULL, &len);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || len == 0) return;
    buf = (char *)malloc(len);
    if (!buf) return;
    if (GetLogicalProcessorInformationEx(RelationAll,
            (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)buf, &len)) {
        for (p = buf, end = buf + len; p < end;) {
            SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *e =
                (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)p;
            if (e->Relationship == RelationCache) {
                const CACHE_RELATIONSHIP *c = &e->Cache;
                const int data = (c->Type == CacheData || c->Type == CacheUnified);
                if (c->Level == 1 && data && o->l1d_seen == 0) {
                    o->l1d_seen = (long)c->CacheSize;
                    o->l1d_ways = (long)c->Associativity;
                } else if (c->Level == 2 && data && o->l2_seen == 0) {
                    o->l2_seen = (long)c->CacheSize;
                } else if (c->Level == 3 && data && o->l3_seen == 0) {
                    o->l3_seen = (long)c->CacheSize;
                }
            } else if (e->Relationship == RelationProcessorCore && o->smt == 0) {
                /* logical processors on one physical core = set bits of its
                 * first group mask */
                KAFFINITY m = e->Processor.GroupMask[0].Mask;
                int bits = 0;
                while (m) { bits += (int)(m & 1u); m >>= 1; }
                if (bits > 0) o->smt = bits;
            }
            p += e->Size;
        }
    }
    free(buf);
}
#elif defined(__linux__)
#include <unistd.h>
/* "32K" / "1024K" / "16M" / plain bytes, as sysfs prints cache sizes. */
static inline long _vfft_cpu_sysfs_long(const char *path)
{
    char buf[64];
    long v = 0;
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    if (fgets(buf, sizeof buf, f)) {
        char *endp = NULL;
        v = strtol(buf, &endp, 10);
        if (endp && (*endp == 'K' || *endp == 'k')) v *= 1024L;
        else if (endp && (*endp == 'M' || *endp == 'm')) v *= 1024L * 1024L;
    }
    fclose(f);
    return v > 0 ? v : 0;
}
static inline void _vfft_cpu_os_fill(vfft_cpu_cache_t *o)
{
    long v;
    int i;
#ifdef _SC_LEVEL1_DCACHE_SIZE
    if (o->l1d_seen == 0 && (v = sysconf(_SC_LEVEL1_DCACHE_SIZE)) > 0)  o->l1d_seen = v;
    if (o->l1d_ways == 0 && (v = sysconf(_SC_LEVEL1_DCACHE_ASSOC)) > 0) o->l1d_ways = v;
    if (o->l2_seen == 0  && (v = sysconf(_SC_LEVEL2_CACHE_SIZE)) > 0)   o->l2_seen = v;
    if (o->l3_seen == 0  && (v = sysconf(_SC_LEVEL3_CACHE_SIZE)) > 0)   o->l3_seen = v;
#endif
    /* sysfs: the reliable source on ARM, where glibc's sysconf answers 0. */
    for (i = 0; i < 8 && (o->l1d_seen == 0 || o->l2_seen == 0 || o->l3_seen == 0); i++) {
        char base[128], path[192], type[32] = "";
        long level, size;
        FILE *f;
        snprintf(base, sizeof base, "/sys/devices/system/cpu/cpu0/cache/index%d", i);
        snprintf(path, sizeof path, "%s/level", base);
        level = _vfft_cpu_sysfs_long(path);
        if (level <= 0) break;
        snprintf(path, sizeof path, "%s/type", base);
        if ((f = fopen(path, "r")) != NULL) { if (!fgets(type, sizeof type, f)) type[0] = 0; fclose(f); }
        if (strncmp(type, "Instruction", 11) == 0) continue;
        snprintf(path, sizeof path, "%s/size", base);
        size = _vfft_cpu_sysfs_long(path);
        if (size <= 0) continue;
        if (level == 1 && o->l1d_seen == 0) {
            o->l1d_seen = size;
            snprintf(path, sizeof path, "%s/ways_of_associativity", base);
            o->l1d_ways = _vfft_cpu_sysfs_long(path);
        } else if (level == 2 && o->l2_seen == 0) o->l2_seen = size;
        else if (level == 3 && o->l3_seen == 0)   o->l3_seen = size;
    }
    if (o->smt == 0) {
        /* threads sharing core 0: entries of thread_siblings_list ("0,6" or
         * "0-1"). Absent sysfs leaves smt unknown, which the pool treats as
         * the historical stride. */
        FILE *f = fopen("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list", "r");
        if (f) {
            char line[256];
            if (fgets(line, sizeof line, f)) {
                int n = 0;
                char *tok = strtok(line, ",\n");
                while (tok) {
                    char *dash = strchr(tok, '-');
                    n += dash ? (atoi(dash + 1) - atoi(tok) + 1) : 1;
                    tok = strtok(NULL, ",\n");
                }
                if (n > 0) o->smt = n;
            }
            fclose(f);
        }
    }
}
#elif defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
static inline long _vfft_cpu_sysctl_long(const char *name)
{
    long long v = 0;              /* 8 bytes, zero-filled: a 4-byte key lands
                                   * in the low half on little-endian */
    size_t sz = sizeof v;
    if (sysctlbyname(name, &v, &sz, NULL, 0) != 0 || v <= 0) return 0;
    return (long)v;
}
static inline void _vfft_cpu_os_fill(vfft_cpu_cache_t *o)
{
    long v;
    /* perflevel0 = the performance cluster on Apple silicon; the plain keys
     * are the pre-hybrid names and remain as the fallback. */
    if (o->l1d_seen == 0) {
        v = _vfft_cpu_sysctl_long("hw.perflevel0.l1dcachesize");
        if (!v) v = _vfft_cpu_sysctl_long("hw.l1dcachesize");
        o->l1d_seen = v;
    }
    if (o->l2_seen == 0) {
        v = _vfft_cpu_sysctl_long("hw.perflevel0.l2cachesize");
        if (!v) v = _vfft_cpu_sysctl_long("hw.l2cachesize");
        o->l2_seen = v;
    }
    if (o->l3_seen == 0) o->l3_seen = _vfft_cpu_sysctl_long("hw.l3cachesize");
    if (o->smt == 0) {
        long lg = _vfft_cpu_sysctl_long("hw.logicalcpu");
        long ph = _vfft_cpu_sysctl_long("hw.physicalcpu");
        if (lg > 0 && ph > 0) o->smt = (int)(lg / ph);
    }
}
#else
static inline void _vfft_cpu_os_fill(vfft_cpu_cache_t *o) { (void)o; }
#endif

static inline void _vfft_cpu_cache_fill(vfft_cpu_cache_t *o)
{
    unsigned r[4];
    o->l1d_seen = 0; o->l1d_ways = 0; o->core_type = 0;
    /* is_pcore starts at 1: a host with no CPUID has no hybrid information,
     * and the rule for "no core type" is "usable" (same as core_type == 0
     * below). The CPUID block always overwrites it on x86. */
    o->is_pcore = 1; o->discovered = 0; o->geometry_ok = 1;
    o->l2_seen = 0; o->vendor = VFFT_CPU_VENDOR_UNKNOWN;

#if VFFT_CPU_HAVE_CPUID
    _vfft_cpuid(0, 0, r);
    const unsigned maxleaf = r[0];
    /* Vendor string is EBX,EDX,ECX (in that order) from leaf 0. */
    {
        char v[13];
        memcpy(v + 0, &r[1], 4); memcpy(v + 4, &r[3], 4); memcpy(v + 8, &r[2], 4);
        v[12] = '\0';
        if (!strcmp(v, "GenuineIntel"))      o->vendor = VFFT_CPU_VENDOR_INTEL;
        else if (!strcmp(v, "AuthenticAMD")) o->vendor = VFFT_CPU_VENDOR_AMD;
    }

    if (maxleaf >= 0x1A) {
        _vfft_cpuid(0x1A, 0, r);
        o->core_type = (r[0] >> 24) & 0xFFu;
    }
    /* A non-hybrid CPU reports no core type; treat it as usable. */
    o->is_pcore = (o->core_type == VFFT_CPU_TYPE_CORE) || (o->core_type == 0);

    if (maxleaf >= 4) _vfft_cpu_walk_cache_leaf(o, 4);

    /* AMD: leaf 4 answered with zeros (measured, see the vendor note above).
     * Re-read through the extended twins. Gated on the extended max-leaf AND
     * TOPOEXT so a pre-Zen part falls back instead of decoding garbage.
     * Guarded by l1d_seen==0 so a future AMD part that DOES populate leaf 4
     * keeps that answer rather than being re-read twice. */
    if (o->vendor == VFFT_CPU_VENDOR_AMD && o->l1d_seen == 0) {
        _vfft_cpuid(0x80000000u, 0, r);
        const unsigned maxext = r[0];
        int topoext = 0;
        if (maxext >= 0x80000001u) {
            _vfft_cpuid(0x80000001u, 0, r);
            topoext = (int)((r[2] >> 22) & 1u);      /* ECX bit 22 = TOPOEXT */
        }
        if (topoext && maxext >= 0x8000001Du)
            _vfft_cpu_walk_cache_leaf(o, 0x8000001Du);
        /* SMT width: 0x8000001E EBX[15:8] = threads per core MINUS ONE.
         * This is the AMD spelling of the leaf-0xB read below; without it
         * smt stays 0 ("unknown") on every AMD part and the pool's pin
         * stride loses the input this file insists it must be derived from. */
        if (topoext && maxext >= 0x8000001Eu) {
            _vfft_cpuid(0x8000001Eu, 0, r);
            o->smt = (int)(((r[1] >> 8) & 0xFFu) + 1u);
        }
    }
    /* SMT width: leaf 0xB level type 1 (SMT), EBX[15:0] = logical procs at
     * that level. Decides the pool's pin STRIDE — a hard-coded stride of 2
     * silently skips half the cores (or leaves workers unpinned) on a
     * non-SMT or SMT-disabled part, voiding every cache-privacy argument
     * the threading design rests on. */
    /* o->smt == 0 guard: AMD already answered via 0x8000001E above. AMD does
     * implement leaf 0xB too, so both reads agree where both work — but the
     * extended read is the authoritative one on that vendor and must not be
     * silently replaced by a leaf this vendor is not required to populate. */
    if (maxleaf >= 0xB && o->smt == 0) {
        for (unsigned sub = 0; sub < 4u; sub++) {
            _vfft_cpuid(0xB, sub, r);
            if ((((r[2] >> 8) & 0xFF) == 1) && (r[1] & 0xFFFF)) {
                o->smt = (int)(r[1] & 0xFFFF);
                break;
            }
            if (((r[2] >> 8) & 0xFF) == 0) break; /* level type invalid */
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

    /* OS tier: only for what CPUID left at zero (a no-op on a working x86;
     * l3_seen == 0 on an L3-less part just re-confirms the absence). */
    if (o->l1d_seen == 0 || o->l2_seen == 0 || o->l3_seen == 0 || o->smt == 0)
        _vfft_cpu_os_fill(o);

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

/* ── HOST TAG (2026-09-03) ───────────────────────────────────────────────────
 * A stable, WHITESPACE-FREE identifier for "which machine raced this", for the
 * wisdom `@meta host=` stamp. The wisdom grammar forbids whitespace in values
 * (README §3 "values are bare ... the writer refuses violations at the API"),
 * so the CPUID brand string ("AMD Ryzen 5 PRO 8640HS w/ Radeon 760M Graphics")
 * is NOT usable directly and is deliberately not sanitised into one either —
 * marketing names are not stable identifiers and two parts with the same brand
 * can differ in the geometry that actually moves a verdict.
 *
 * vendor + display-family + display-model is what the uarch is: Raptor Lake is
 * intel-f6m183, Zen 4 Phoenix is amd-f25m117. Steppings are excluded on
 * purpose — they do not move cache geometry, and including them would make
 * every wisdom store look foreign to its own successor silicon.
 *
 * This is an IDENTITY, not a capability. Nothing may branch on it; it exists so
 * a store can say where it came from and a mismatch can be REPORTED. */
static inline const char *vfft_cpu_host_tag(void)
{
    static char tag[32];
    static int done = 0;
    if (done) return tag;
    done = 1;
    /* No CPUID: the architecture is the best honest identity available. It
     * is coarse (two ARM64 machines tag alike) but never wrong; nothing
     * branches on the tag, it only labels provenance. */
#if defined(__aarch64__) || defined(_M_ARM64)
    snprintf(tag, sizeof tag, "arm64");
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    snprintf(tag, sizeof tag, "x86-nocpuid");
#else
    snprintf(tag, sizeof tag, "unknown-arch");
#endif
#if VFFT_CPU_HAVE_CPUID
    {
        const vfft_cpu_cache_t *c = vfft_cpu_cache();
        unsigned r[4];
        _vfft_cpuid(1, 0, r);
        const unsigned fam  = (r[0] >> 8) & 0xFu;
        const unsigned mod  = (r[0] >> 4) & 0xFu;
        const unsigned efam = (r[0] >> 20) & 0xFFu;
        const unsigned emod = (r[0] >> 16) & 0xFu;
        const unsigned dfam = fam + ((fam == 0xFu) ? efam : 0u);
        const unsigned dmod = mod + (((fam == 0x6u) || (fam == 0xFu)) ? (emod << 4) : 0u);
        const char *v = (c->vendor == VFFT_CPU_VENDOR_INTEL) ? "intel"
                      : (c->vendor == VFFT_CPU_VENDOR_AMD)   ? "amd"
                                                             : "x86";
        snprintf(tag, sizeof tag, "%s-f%um%u", v, dfam, dmod);
    }
#endif
    return tag;
}

/* The capacity every cache-sizing decision must use, and the value that gets
 * stamped into a wisdom record beside the width it produced. */
static inline long vfft_cpu_l1d_bytes(void) { return vfft_cpu_cache()->l1d_used; }

/* The L2 twin (2026-08-25): the capacity every L2-sized decision must use
 * (first consumer: the 2D band-threshold fence N1_max = L2/(16*wl_min)),
 * and the value stamped beside any banked verdict that depended on it. */
static inline long vfft_cpu_l2_bytes(void) { return vfft_cpu_cache()->l2_used; }

/* SHARED-L3 budget. 0 = unknown (caller must then refuse to use it as a
 * gate, the same discipline as the l2 refuse rule). Never a per-core
 * sizing input — only "do T concurrent working sets fit?" questions. */
static inline long vfft_cpu_l3_bytes(void) { return vfft_cpu_cache()->l3_seen; }

/* Logical processors per physical core; 0 = unknown. The pool's pin
 * stride is derived from this, never hard-coded. */
static inline int vfft_cpu_smt(void) { return vfft_cpu_cache()->smt; }

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
