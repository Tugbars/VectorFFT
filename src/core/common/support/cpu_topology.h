/* cpu_topology.h — which logical CPU is which: the physical cores, their
 * hyperthread siblings, and which cores are P-cores.
 *
 * The race scope (race_scope.h) pins a measurement to a P-core and holds that
 * core's sibling; the store's CPU identity counts the P- and E-cores. Both
 * read this one table, taken from the operating system once per process:
 *   Windows  GetLogicalProcessorInformationEx(RelationProcessorCore): one
 *            entry per physical core with its logical CPUs and its
 *            EfficiencyClass. The cores of the highest class are the P-cores;
 *            a machine with one class has only P-cores. Intel hybrids and
 *            AMD parts alike: the classification is the OS's, not a CPUID
 *            dialect's. Processor group 0 only (64 logical CPUs).
 *   Linux    /sys/devices/system/cpu/cpuN/topology/thread_siblings_list for
 *            the cores; /sys/devices/cpu_core/cpus and cpu_atom/cpus, which
 *            exist on Intel hybrids, for the kind. Without them every core is
 *            a P-core -- right for every non-hybrid part, and NOT VERIFIED on
 *            an AMD part that mixes full and compact cores (none at hand).
 * When the OS gives nothing the table says so (ok = 0): one core per logical
 * CPU, every one a P-core, no siblings.
 *
 * The table describes the MACHINE, not the process's allowed set: an identity
 * read from it does not change with an affinity mask.
 *
 * Depends on common/support/cpu_cache.h and the OS headers only.
 */
#ifndef VFFT_SUPPORT_CPU_TOPOLOGY_H
#define VFFT_SUPPORT_CPU_TOPOLOGY_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#elif defined(__linux__)
#  include <unistd.h>
#endif

#define VFFT_TOPO_MAX_CPU 256

typedef struct
{
    int ok;                              /* 1 = read from the OS */
    int ncpu;                            /* logical CPUs described (0..ncpu-1) */
    int ncores;                          /* physical cores */
    int n_p, n_e;                        /* P-cores, E-cores */
    int core[VFFT_TOPO_MAX_CPU];         /* the physical core of each logical CPU, -1 = none */
    unsigned char pcore[VFFT_TOPO_MAX_CPU]; /* 1 = that logical CPU is on a P-core */
} vfft_topology_t;

#if defined(__linux__)
/* the first number of a sysfs cpu list ("2,18" or "2-3"), -1 when unreadable */
static inline int _vfft_topo_first_of(const char *path)
{
    FILE *f = fopen(path, "r");
    int v = -1;
    if (!f) return -1;
    if (fscanf(f, "%d", &v) != 1) v = -1;
    fclose(f);
    return v;
}
/* mark every CPU of a sysfs cpu list ("0-15,20") in set[]; 1 = the file was read */
static inline int _vfft_topo_mark_list(const char *path, unsigned char *set, int n)
{
    FILE *f = fopen(path, "r");
    char buf[512];
    const char *s;
    if (!f) return 0;
    if (!fgets(buf, sizeof buf, f)) { fclose(f); return 0; }
    fclose(f);
    for (s = buf; *s && *s != '\n';)
    {
        char *e;
        long a = strtol(s, &e, 10), b = a, c;
        if (e == s) break;
        if (*e == '-') { const char *t = e + 1; b = strtol(t, &e, 10); }
        for (c = a; c <= b; c++)
            if (c >= 0 && c < n) set[c] = 1;
        s = (*e == ',') ? e + 1 : e;
    }
    return 1;
}
#endif

static inline void _vfft_topology_fill(vfft_topology_t *t)
{
    int i;
    memset(t, 0, sizeof *t);
    for (i = 0; i < VFFT_TOPO_MAX_CPU; i++) t->core[i] = -1;
#if defined(_WIN32)
    {
        DWORD len = 0;
        char *buf;
        int maxclass = 0;
        unsigned char cls[VFFT_TOPO_MAX_CPU];
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        t->ncpu = (int)si.dwNumberOfProcessors;
        if (t->ncpu > 64) t->ncpu = 64;
        memset(cls, 0, sizeof cls);
        GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
        buf = len ? (char *)malloc(len) : NULL;
        if (buf && GetLogicalProcessorInformationEx(RelationProcessorCore,
                                                    (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)buf, &len))
        {
            DWORD off;
            for (off = 0; off < len;)
            {
                SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *x = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)(buf + off);
                if (x->Relationship == RelationProcessorCore && x->Processor.GroupCount >= 1 &&
                    x->Processor.GroupMask[0].Group == 0)
                {
                    const KAFFINITY m = x->Processor.GroupMask[0].Mask;
                    int c;
                    for (c = 0; c < 64 && c < t->ncpu; c++)
                        if (m & ((KAFFINITY)1 << c))
                        {
                            t->core[c] = t->ncores;
                            cls[c] = (unsigned char)x->Processor.EfficiencyClass;
                            if (cls[c] > maxclass) maxclass = cls[c];
                        }
                    t->ncores++;
                }
                off += x->Size;
            }
            for (i = 0; i < t->ncpu; i++)
                t->pcore[i] = (unsigned char)(t->core[i] >= 0 && cls[i] == maxclass);
            t->ok = (t->ncores > 0);
        }
        free(buf);
    }
#elif defined(__linux__)
    {
        long n = sysconf(_SC_NPROCESSORS_CONF);
        int rep[VFFT_TOPO_MAX_CPU];              /* each CPU's first sibling */
        unsigned char isp[VFFT_TOPO_MAX_CPU], ise[VFFT_TOPO_MAX_CPU];
        int have_kinds, any = 0;
        if (n < 1) n = 1;
        if (n > VFFT_TOPO_MAX_CPU) n = VFFT_TOPO_MAX_CPU;
        t->ncpu = (int)n;
        memset(isp, 0, sizeof isp);
        memset(ise, 0, sizeof ise);
        for (i = 0; i < t->ncpu; i++)
        {
            char p[128];
            snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", i);
            rep[i] = _vfft_topo_first_of(p);
            if (rep[i] >= 0) any = 1;
        }
        have_kinds = _vfft_topo_mark_list("/sys/devices/cpu_core/cpus", isp, t->ncpu) |
                     _vfft_topo_mark_list("/sys/devices/cpu_atom/cpus", ise, t->ncpu);
        if (any)
        {
            int map[VFFT_TOPO_MAX_CPU];          /* representative -> core index */
            for (i = 0; i < VFFT_TOPO_MAX_CPU; i++) map[i] = -1;
            for (i = 0; i < t->ncpu; i++)
            {
                const int r = rep[i];
                if (r < 0 || r >= VFFT_TOPO_MAX_CPU) continue;
                if (map[r] < 0) map[r] = t->ncores++;
                t->core[i] = map[r];
                t->pcore[i] = (unsigned char)(have_kinds ? (isp[i] && !ise[i]) : 1);
            }
            t->ok = 1;
        }
    }
#endif
    if (!t->ok)
    {   /* nothing from the OS: one core per logical CPU, every one a P-core */
        if (t->ncpu < 1) t->ncpu = 1;
        t->ncores = t->ncpu;
        for (i = 0; i < t->ncpu; i++) { t->core[i] = i; t->pcore[i] = 1; }
    }
    {   /* count the cores of each kind (a core's kind is its first CPU's) */
        int c;
        for (c = 0; c < t->ncores; c++)
            for (i = 0; i < t->ncpu; i++)
                if (t->core[i] == c)
                {
                    if (t->pcore[i]) t->n_p++; else t->n_e++;
                    break;
                }
    }
}

/* the machine's table, read once per process */
static inline const vfft_topology_t *vfft_topology(void)
{
    static vfft_topology_t t;
    static int done = 0;
    if (!done) { _vfft_topology_fill(&t); done = 1; }
    return &t;
}

/* another logical CPU of cpu's physical core (its hyperthread sibling), -1 = none */
static inline int vfft_topo_sibling(int cpu)
{
    const vfft_topology_t *t = vfft_topology();
    int i;
    if (cpu < 0 || cpu >= t->ncpu || t->core[cpu] < 0) return -1;
    for (i = 0; i < t->ncpu; i++)
        if (i != cpu && t->core[i] == t->core[cpu]) return i;
    return -1;
}

/* the first logical CPU of the k-th P-core (k = 0, 1, ..), -1 past the last */
static inline int vfft_topo_pcore_cpu(int k)
{
    const vfft_topology_t *t = vfft_topology();
    int i, seen = 0, last = -1;
    for (i = 0; i < t->ncpu; i++)
    {
        if (t->core[i] < 0 || !t->pcore[i] || t->core[i] == last) continue;
        {   /* the first CPU of a core not yet counted */
            int j, first = 1;
            for (j = 0; j < i; j++)
                if (t->core[j] == t->core[i]) { first = 0; break; }
            if (!first) continue;
        }
        if (seen == k) return i;
        seen++;
        last = t->core[i];
    }
    return -1;
}

#endif /* VFFT_SUPPORT_CPU_TOPOLOGY_H */
