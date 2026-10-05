/* cpu_identity.h — THE CPU IDENTITY: which CPU raced a wisdom row.
 *
 * A race winner is a property of the machine that ran the race, so the store
 * keeps one folder per CPU and serves a row only on the identity that raced
 * it (docs/design/wisdom_system.md §2-§3; never borrowed from another CPU's
 * folder). The identity is every hardware fact a verdict depends on:
 *
 *   host=    vendor and display family/model (cpu_cache.h, vfft_cpu_host_tag):
 *            intel-f6m183 is Raptor Lake, amd-f25m117 is Zen 4 Phoenix
 *   isa=     the build's instruction set (a row raced by AVX2 kernels says
 *            nothing about AVX-512 ones)
 *   l1d= l2= the P-core's private caches in bytes, read on a P-core
 *   l3=      the shared cache in bytes (the four-step's admission reads it)
 *   pcores= ecores=   physical cores of each kind (a threaded verdict depends
 *            on them), from the machine's topology, never from the process's
 *            allowed set: an affinity mask does not change the identity
 *
 * Two machines share rows only when every field is equal: a 14900K, a 14900KF
 * and a 13900K are one identity; an i7-14700K (12 E-cores, 33 MB) and an
 * i5-14600K (6 P-cores, 24 MB) are others. Every field comes from the vendor's
 * own CPUID dialect or from the OS, so the same code names an Intel and an AMD
 * part.
 *
 * The string is the store's `@meta` stamp: space-separated key=value, no
 * whitespace inside a value. An IDENTITY, not a capability: nothing branches
 * on its fields; it selects a folder and is compared whole.
 *
 * PLANNING ONLY. Depends on cpu_cache.h, cpu_topology.h and build_isa.h.
 */
#ifndef VFFT_SUPPORT_CPU_IDENTITY_H
#define VFFT_SUPPORT_CPU_IDENTITY_H

#include <stdio.h>
#include "cpu_cache.h"
#include "cpu_topology.h"
#include "build_isa.h"   /* VFFT_ISA_NAME */

/* this CPU's identity (cached). A probe that needs another one defines
 * VFFT_IDENTITY_PROBE as an expression yielding a string, or NULL for the
 * real one. */
static inline const char *vfft_cpu_identity(void)
{
    static char id[192];
    static int done = 0;
#ifdef VFFT_IDENTITY_PROBE
    {
        const char *forged = (VFFT_IDENTITY_PROBE);
        if (forged) return forged;
    }
#endif
    if (!done) {
        const vfft_cpu_cache_t *c = vfft_cpu_cache();
        const vfft_topology_t *t = vfft_topology();
        snprintf(id, sizeof id, "host=%s isa=%s l1d=%ld l2=%ld l3=%ld pcores=%d ecores=%d",
                 vfft_cpu_host_tag(), VFFT_ISA_NAME, c->l1d_used, c->l2_used, c->l3_seen, t->n_p, t->n_e);
        done = 1;
    }
    return id;
}

/* a folder name for an identity string: its values joined by '-', e.g.
 * intel-f6m183-avx2-49152-2097152-37748736-8-16. Used only when the library
 * has to create a folder itself; a folder's name carries no meaning. */
static inline void vfft_cpu_identity_folder_name(const char *identity, char *out, size_t n)
{
    size_t k = 0;
    const char *p = identity;
    if (!n) return;
    while (*p && k + 1 < n) {
        const char *eq = p;
        while (*eq && *eq != '=' && *eq != ' ') eq++;
        if (*eq == '=') p = eq + 1;                 /* skip "key=" */
        if (k && k + 1 < n) out[k++] = '-';
        while (*p && *p != ' ' && k + 1 < n) {
            const char ch = *p++;
            out[k++] = ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '-' || ch == '_')
                           ? ch : '_';
        }
        while (*p == ' ') p++;
    }
    out[k] = '\0';
}

#endif /* VFFT_SUPPORT_CPU_IDENTITY_H */
