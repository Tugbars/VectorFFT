/* cpu_l1_probe.c — what does THIS machine report for L1d, per logical CPU?
 *
 * WHY. The tcut tile-width filter needs an L1 data-cache capacity, and the
 * decision (Tugbars, 2026-08-02) is: discover it rather than hard-code it, stamp
 * the discovered value into wisdom, and treat a mismatch at replay as
 * "re-measure", never as "inherit". Before designing around that we need to
 * know what the query actually returns here — this CPU is hybrid, and CPUID
 * reports the cache of whichever core the query RUNS on.
 *
 * This probe exists because the previous time I sized something from
 * recollection instead of measurement (the DP candidate cap) I was wrong by
 * 2.4x. Cache sizes are exactly the same class of "obviously I know this".
 *
 * WHAT IT READS, pinned to each logical CPU in turn:
 *   CPUID leaf 0x1A  hybrid info: EAX[31:24] core type, 0x20 = Atom ("E"),
 *                    0x40 = Core ("P"). Absent pre-Alder-Lake.
 *   CPUID leaf 4     deterministic cache parameters, walked over subleaves:
 *                    size = (ways+1)*(partitions+1)*(line+1)*(sets+1).
 *
 * SCOPE (Tugbars): the library targets P-cores. E-core support is left to a
 * user extension. This probe still reports E-cores so the P/E split is visible
 * and so the pin assertion can be written against real values.
 *
 * Not a benchmark: no timing, nothing written anywhere.
 * Build: python build.py --src benches/cpu_l1_probe.c
 */
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sched.h>
#include <unistd.h>
#endif

#if defined(__GNUC__)
#include <cpuid.h>
static void cpuid_cnt(unsigned leaf, unsigned sub, unsigned r[4])
{
    __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
}
#else
#include <intrin.h>
static void cpuid_cnt(unsigned leaf, unsigned sub, unsigned r[4])
{
    int t[4]; __cpuidex(t, (int)leaf, (int)sub);
    r[0] = (unsigned)t[0]; r[1] = (unsigned)t[1];
    r[2] = (unsigned)t[2]; r[3] = (unsigned)t[3];
}
#endif

static int pin_to(int cpu)
{
#ifdef _WIN32
    return SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << cpu) != 0;
#else
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
    return sched_setaffinity(0, sizeof s, &s) == 0;
#endif
}

typedef struct {
    int  valid;
    unsigned core_type;      /* 0x20 Atom(E), 0x40 Core(P), 0 = absent */
    long l1d, l1d_line, l1d_ways, l1d_sets;
    long l2,  l3;
} cpu_info_t;

static long cache_size_from(unsigned r[4], long *line, long *ways, long *sets)
{
    long w = (long)(((r[1] >> 22) & 0x3FF) + 1);
    long p = (long)(((r[1] >> 12) & 0x3FF) + 1);
    long l = (long)((r[1] & 0xFFF) + 1);
    long s = (long)(r[2] + 1);
    if (line) *line = l;
    if (ways) *ways = w;
    if (sets) *sets = s;
    return w * p * l * s;
}

static void probe_here(cpu_info_t *o)
{
    unsigned r[4];
    memset(o, 0, sizeof *o);

    cpuid_cnt(0, 0, r);
    const unsigned maxleaf = r[0];

    if (maxleaf >= 0x1A) {
        cpuid_cnt(0x1A, 0, r);
        o->core_type = (r[0] >> 24) & 0xFFu;    /* 0 if not hybrid */
    }

    if (maxleaf >= 4) {
        for (unsigned sub = 0; sub < 16; sub++) {
            cpuid_cnt(4, sub, r);
            const int type  = (int)(r[0] & 0x1F);       /* 1 D, 2 I, 3 unified */
            const int level = (int)((r[0] >> 5) & 0x7);
            if (type == 0) break;                        /* no more caches     */
            long line = 0, ways = 0, sets = 0;
            long sz = cache_size_from(r, &line, &ways, &sets);
            if (level == 1 && type == 1) {               /* L1 DATA            */
                o->l1d = sz; o->l1d_line = line;
                o->l1d_ways = ways; o->l1d_sets = sets;
            } else if (level == 2) o->l2 = sz;
            else if (level == 3)   o->l3 = sz;
        }
        o->valid = 1;
    }
}

/* Intel SDM Vol.2, CPUID leaf 1AH, EAX[31:24] core type:
 *     20H = Intel Atom  (E-core)
 *     40H = Intel Core  (P-core)
 * 🔴 I had these INVERTED on the first run of this probe and it printed the
 * P-cores as E and vice versa. What exposed it was the cache geometry in the
 * same table disagreeing with the label: 48 KB / 12-way L1d with a private
 * 2 MB L2 is Raptor Cove (P), while 32 KB / 8-way with a shared 4 MB L2 is
 * Gracemont (E). Keep both columns visible for exactly that reason — a label
 * alone cannot be cross-checked, a label next to its geometry can. */
#define VFFT_CPU_TYPE_ATOM 0x20u
#define VFFT_CPU_TYPE_CORE 0x40u

static const char *type_name(unsigned t)
{
    if (t == VFFT_CPU_TYPE_ATOM) return "E (Atom)";
    if (t == VFFT_CPU_TYPE_CORE) return "P (Core)";
    if (t == 0)                  return "not hybrid";
    return "unknown";
}

/* Does the reported type agree with the cache geometry? Returns 0 on a
 * contradiction, which is what an inverted decode looks like. */
static int type_matches_geometry(unsigned t, long l1d, long ways)
{
    if (t == VFFT_CPU_TYPE_CORE) return l1d >= 48 * 1024 && ways >= 12;
    if (t == VFFT_CPU_TYPE_ATOM) return l1d <= 32 * 1024 && ways <= 8;
    return 1;                                    /* nothing to check */
}

int main(void)
{
    unsigned r[4];
    char vendor[13] = {0};
    cpuid_cnt(0, 0, r);
    memcpy(vendor + 0, &r[1], 4);
    memcpy(vendor + 4, &r[3], 4);
    memcpy(vendor + 8, &r[2], 4);
    printf("vendor=%s  max_cpuid_leaf=0x%X\n", vendor, r[0]);

    int ncpu = 0;
#ifdef _WIN32
    SYSTEM_INFO si; GetSystemInfo(&si); ncpu = (int)si.dwNumberOfProcessors;
#else
    ncpu = (int)sysconf(_SC_NPROCESSORS_ONLN);
#endif
    if (ncpu > 64) ncpu = 64;
    printf("logical CPUs = %d\n\n", ncpu);

    printf("  %-4s %-12s %10s %8s %6s %6s %10s %10s\n",
           "cpu", "core type", "L1d", "line", "ways", "sets", "L2", "L3");
    printf("  ------------------------------------------------------------"
           "-------------\n");

    long p_l1 = 0, e_l1 = 0;
    int p_n = 0, e_n = 0, mixed_p = 0, mixed_e = 0, geom_bad = 0;

    for (int c = 0; c < ncpu; c++) {
        if (!pin_to(c)) { printf("  %-4d  <pin failed>\n", c); continue; }
        cpu_info_t i;
        probe_here(&i);
        const int geom_ok = type_matches_geometry(i.core_type, i.l1d, i.l1d_ways);
        if (!geom_ok) geom_bad = 1;
        printf("  %-4d %-12s %8ldK %8ld %6ld %6ld %8ldK %8ldK%s\n",
               c, type_name(i.core_type), i.l1d / 1024, i.l1d_line,
               i.l1d_ways, i.l1d_sets, i.l2 / 1024, i.l3 / 1024,
               geom_ok ? "" : "   *** TYPE/GEOMETRY DISAGREE ***");

        if (i.core_type == VFFT_CPU_TYPE_ATOM) {
            if (e_n && e_l1 != i.l1d) mixed_e = 1;
            e_l1 = i.l1d; e_n++;
        } else if (i.core_type == VFFT_CPU_TYPE_CORE) {
            if (p_n && p_l1 != i.l1d) mixed_p = 1;
            p_l1 = i.l1d; p_n++;
        }
    }

    printf("\n  SUMMARY\n");
    if (p_n) printf("    P-cores: %2d logical, L1d = %ld KB%s\n",
                    p_n, p_l1 / 1024, mixed_p ? "  *** NOT UNIFORM ***" : "");
    if (e_n) printf("    E-cores: %2d logical, L1d = %ld KB%s\n",
                    e_n, e_l1 / 1024, mixed_e ? "  *** NOT UNIFORM ***" : "");
    if (p_n && e_n) {
        printf("\n    HYBRID: L1d differs by core type (%ld KB vs %ld KB, ratio %.2fx).\n",
               p_l1 / 1024, e_l1 / 1024, (double)p_l1 / (double)(e_l1 ? e_l1 : 1));
        printf("    A width sized for a P-core is %.0f%% of an E-core's L1 —\n",
               100.0 * (double)p_l1 / (double)(e_l1 ? e_l1 : 1));
        printf("    and overshoot is the failure mode that costs everything\n"
               "    rather than degrading. Hence: pin to P, assert the type,\n"
               "    stamp the value, compare on replay.\n");
    } else if (p_n || e_n) {
        printf("\n    Uniform L1d across all cores — no P/E split to guard.\n");
    } else {
        printf("\n    Leaf 0x1A absent: not a hybrid CPU (or not reported).\n");
    }

    /* What core 2 reports — the benchmark protocol pins there (mask 0x4). */
    if (pin_to(2)) {
        cpu_info_t i; probe_here(&i);
        printf("\n    Benchmark pin (core 2, mask 0x4): %s, L1d = %ld KB\n",
               type_name(i.core_type), i.l1d / 1024);
    }
    return 0;
}
