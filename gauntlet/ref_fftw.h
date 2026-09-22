/* ref_fftw.h — FFTW reference backend, part 1: THE RUNTIME BINDER (phase P0).
 *
 * Architecture of record: docs/roadmap/fftw_bench_design.md §5 / P0.
 * Interface it will eventually implement: src/core/support/ref.h (ref_vtable_t,
 * phase P2). THIS file is deliberately only the loader + the deterministic
 * plane allocator + plan_id — bind and prove, time nothing.
 *
 * 🔴 WHY RUNTIME BINDING, NEVER fftw3.lib ON THE LINK LINE:
 * MKL exports 92 fftw_* wrapper symbols. With mkl_rt first on the link line the
 * "FFTW" you call IS MKL — fftw_version reads "FFTW 3.3.4 wrappers to Intel
 * oneMKL" — and the bench races MKL against MKL, labelled FFTW. Binding by
 * LoadLibrary/GetProcAddress on an ABSOLUTE DLL path is the only construction
 * under which that hijack is impossible (verified: zero fftw imports in the PE
 * table even with both libs present). Consequently this header declares its own
 * ABI-matching types (fftwx_*) and never includes fftw3.h — the vcpkg tree also
 * carries MKL's wrapper header at include/fftw/fftw3.h, so including "the"
 * fftw3.h is itself a trap.
 *
 * DLL resolution: $VFFT_FFTW_DLL if set, else the absolute vcpkg path. There is
 * deliberately NO bare "fftw3.dll" PATH-search fallback — PATH search is exactly
 * how a wrong DLL sneaks in.
 *
 * ⚠ DIRECTION MAPPING (trap): ref.h's REF_FWD=+1 / REF_BWD=-1 are TOKENS.
 * FFTW_FORWARD is -1 (e^{-i2πkn/N}, same sign as MKL ComputeForward) and
 * FFTW_BACKWARD is +1. This backend owns the mapping; never pass a ref_dir_t
 * numerically into an fftw sign slot.
 *
 * ⚠ GURU SPLIT HAS NO SIGN PARAMETER: fftw_plan_guru_split_dft always computes
 * the FORWARD transform of (ri + i·ii). Backward = swap ri<->ii and ro<->io at
 * plan time. The backend owns that too.
 *
 * ⚠ sprint_plan strings are freed with free(), NEVER fftw_free() — mixing them
 * corrupts the heap at loop scale (settled empirically, campaign 2026-08-15).
 */
#ifndef VFFT_BENCH_REF_FFTW_H
#define VFFT_BENCH_REF_FFTW_H

#include "../../src/core/support/ref.h"   /* ref_planes_t, ref_plane_stride */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#  include <windows.h>
#  include <malloc.h>
#else
#  include <dlfcn.h>
#endif

/* ------------------------------------------------------- ABI-matching types */
typedef double fftwx_complex[2];
typedef void  *fftwx_plan;                       /* opaque */
typedef struct { int n, is, os; } fftwx_iodim;   /* == fftw_iodim ABI */

/* Planner flags — values are the stable FFTW3 ABI. NOTE: there is no ESTIMATE
 * here on purpose; ref.h makes ESTIMATE unrepresentable and this header does
 * not reintroduce it. */
#define FFTWX_MEASURE        (0u)
#define FFTWX_DESTROY_INPUT  (1u << 0)
#define FFTWX_UNALIGNED      (1u << 1)
#define FFTWX_EXHAUSTIVE     (1u << 3)
#define FFTWX_PRESERVE_INPUT (1u << 4)
#define FFTWX_PATIENT        (1u << 5)
#define FFTWX_WISDOM_ONLY    (1u << 21)

#define FFTWX_FORWARD  (-1)
#define FFTWX_BACKWARD (+1)

/* ------------------------------------------------------- function typedefs */
typedef fftwx_plan (*fftwx_plan_dft_1d_f)(int, fftwx_complex *, fftwx_complex *, int, unsigned);
typedef fftwx_plan (*fftwx_plan_dft_2d_f)(int, int, fftwx_complex *, fftwx_complex *, int, unsigned);
typedef fftwx_plan (*fftwx_plan_many_dft_f)(int, const int *, int,
                                            fftwx_complex *, const int *, int, int,
                                            fftwx_complex *, const int *, int, int,
                                            int, unsigned);
typedef fftwx_plan (*fftwx_plan_guru_split_dft_f)(int, const fftwx_iodim *, int, const fftwx_iodim *,
                                                  double *, double *, double *, double *, unsigned);
typedef fftwx_plan (*fftwx_plan_dft_r2c_1d_f)(int, double *, fftwx_complex *, unsigned);
typedef fftwx_plan (*fftwx_plan_dft_c2r_1d_f)(int, fftwx_complex *, double *, unsigned);
typedef fftwx_plan (*fftwx_plan_dft_r2c_2d_f)(int, int, double *, fftwx_complex *, unsigned);
typedef fftwx_plan (*fftwx_plan_dft_c2r_2d_f)(int, int, fftwx_complex *, double *, unsigned);
typedef fftwx_plan (*fftwx_plan_many_dft_r2c_f)(int, const int *, int,
                                                double *, const int *, int, int,
                                                fftwx_complex *, const int *, int, int, unsigned);
typedef fftwx_plan (*fftwx_plan_many_dft_c2r_f)(int, const int *, int,
                                                fftwx_complex *, const int *, int, int,
                                                double *, const int *, int, int, unsigned);
typedef fftwx_plan (*fftwx_plan_guru_split_dft_r2c_f)(int, const fftwx_iodim *, int, const fftwx_iodim *,
                                                      double *, double *, double *, unsigned);
typedef fftwx_plan (*fftwx_plan_guru_split_dft_c2r_f)(int, const fftwx_iodim *, int, const fftwx_iodim *,
                                                      double *, double *, double *, unsigned);
typedef void   (*fftwx_execute_f)(const fftwx_plan);
typedef void   (*fftwx_execute_dft_f)(const fftwx_plan, fftwx_complex *, fftwx_complex *);
typedef void   (*fftwx_execute_split_dft_f)(const fftwx_plan, double *, double *, double *, double *);
typedef void   (*fftwx_execute_dft_r2c_f)(const fftwx_plan, double *, fftwx_complex *);
typedef void   (*fftwx_execute_dft_c2r_f)(const fftwx_plan, fftwx_complex *, double *);
typedef void   (*fftwx_execute_split_dft_r2c_f)(const fftwx_plan, double *, double *, double *);
typedef void   (*fftwx_execute_split_dft_c2r_f)(const fftwx_plan, double *, double *, double *);
typedef void   (*fftwx_destroy_plan_f)(fftwx_plan);
typedef void   (*fftwx_cleanup_f)(void);
typedef void  *(*fftwx_malloc_f)(size_t);
typedef void   (*fftwx_free_f)(void *);
typedef int    (*fftwx_export_wisdom_to_filename_f)(const char *);
typedef int    (*fftwx_import_wisdom_from_filename_f)(const char *);
typedef void   (*fftwx_forget_wisdom_f)(void);
typedef char  *(*fftwx_sprint_plan_f)(const fftwx_plan);
typedef int    (*fftwx_alignment_of_f)(double *);
typedef void   (*fftwx_set_timelimit_f)(double);
typedef double (*fftwx_cost_f)(const fftwx_plan);
typedef void   (*fftwx_flops_f)(const fftwx_plan, double *, double *, double *);

/* ------------------------------------------------------------- the api set */
typedef struct {
    /* plan */
    fftwx_plan_dft_1d_f              plan_dft_1d;
    fftwx_plan_dft_2d_f              plan_dft_2d;
    fftwx_plan_many_dft_f            plan_many_dft;
    fftwx_plan_guru_split_dft_f      plan_guru_split_dft;
    fftwx_plan_dft_r2c_1d_f          plan_dft_r2c_1d;
    fftwx_plan_dft_c2r_1d_f          plan_dft_c2r_1d;
    fftwx_plan_dft_r2c_2d_f          plan_dft_r2c_2d;
    fftwx_plan_dft_c2r_2d_f          plan_dft_c2r_2d;
    fftwx_plan_many_dft_r2c_f        plan_many_dft_r2c;
    fftwx_plan_many_dft_c2r_f        plan_many_dft_c2r;
    fftwx_plan_guru_split_dft_r2c_f  plan_guru_split_dft_r2c;
    fftwx_plan_guru_split_dft_c2r_f  plan_guru_split_dft_c2r;
    /* execute */
    fftwx_execute_f                  execute;
    fftwx_execute_dft_f              execute_dft;
    fftwx_execute_split_dft_f        execute_split_dft;
    fftwx_execute_dft_r2c_f          execute_dft_r2c;
    fftwx_execute_dft_c2r_f          execute_dft_c2r;
    fftwx_execute_split_dft_r2c_f    execute_split_dft_r2c;
    fftwx_execute_split_dft_c2r_f    execute_split_dft_c2r;
    /* lifecycle + services */
    fftwx_destroy_plan_f             destroy_plan;
    fftwx_cleanup_f                  cleanup;
    fftwx_malloc_f                   fmalloc;
    fftwx_free_f                     ffree;
    fftwx_export_wisdom_to_filename_f  export_wisdom_to_filename;
    fftwx_import_wisdom_from_filename_f import_wisdom_from_filename;
    fftwx_forget_wisdom_f            forget_wisdom;
    fftwx_sprint_plan_f              sprint_plan;
    fftwx_alignment_of_f             alignment_of;
    fftwx_set_timelimit_f            set_timelimit;
    fftwx_cost_f                     cost;
    fftwx_flops_f                    flops;
    /* identity */
    const char *version;             /* the exported fftw_version data symbol */
    char dll_path[512];              /* what was actually loaded */
} fftwx_api_t;

/* ------------------------------------------------------------ bind table */
#define FFTWX__ENTRY(member, sym) { sym, offsetof(fftwx_api_t, member) }
struct fftwx__sym { const char *name; size_t off; };
static const struct fftwx__sym fftwx__tab[] = {
    FFTWX__ENTRY(plan_dft_1d,               "fftw_plan_dft_1d"),
    FFTWX__ENTRY(plan_dft_2d,               "fftw_plan_dft_2d"),
    FFTWX__ENTRY(plan_many_dft,             "fftw_plan_many_dft"),
    FFTWX__ENTRY(plan_guru_split_dft,       "fftw_plan_guru_split_dft"),
    FFTWX__ENTRY(plan_dft_r2c_1d,           "fftw_plan_dft_r2c_1d"),
    FFTWX__ENTRY(plan_dft_c2r_1d,           "fftw_plan_dft_c2r_1d"),
    FFTWX__ENTRY(plan_dft_r2c_2d,           "fftw_plan_dft_r2c_2d"),
    FFTWX__ENTRY(plan_dft_c2r_2d,           "fftw_plan_dft_c2r_2d"),
    FFTWX__ENTRY(plan_many_dft_r2c,         "fftw_plan_many_dft_r2c"),
    FFTWX__ENTRY(plan_many_dft_c2r,         "fftw_plan_many_dft_c2r"),
    FFTWX__ENTRY(plan_guru_split_dft_r2c,   "fftw_plan_guru_split_dft_r2c"),
    FFTWX__ENTRY(plan_guru_split_dft_c2r,   "fftw_plan_guru_split_dft_c2r"),
    FFTWX__ENTRY(execute,                   "fftw_execute"),
    FFTWX__ENTRY(execute_dft,               "fftw_execute_dft"),
    FFTWX__ENTRY(execute_split_dft,         "fftw_execute_split_dft"),
    FFTWX__ENTRY(execute_dft_r2c,           "fftw_execute_dft_r2c"),
    FFTWX__ENTRY(execute_dft_c2r,           "fftw_execute_dft_c2r"),
    FFTWX__ENTRY(execute_split_dft_r2c,     "fftw_execute_split_dft_r2c"),
    FFTWX__ENTRY(execute_split_dft_c2r,     "fftw_execute_split_dft_c2r"),
    FFTWX__ENTRY(destroy_plan,              "fftw_destroy_plan"),
    FFTWX__ENTRY(cleanup,                   "fftw_cleanup"),
    FFTWX__ENTRY(fmalloc,                   "fftw_malloc"),
    FFTWX__ENTRY(ffree,                     "fftw_free"),
    FFTWX__ENTRY(export_wisdom_to_filename, "fftw_export_wisdom_to_filename"),
    FFTWX__ENTRY(import_wisdom_from_filename,"fftw_import_wisdom_from_filename"),
    FFTWX__ENTRY(forget_wisdom,             "fftw_forget_wisdom"),
    FFTWX__ENTRY(sprint_plan,               "fftw_sprint_plan"),
    FFTWX__ENTRY(alignment_of,              "fftw_alignment_of"),
    FFTWX__ENTRY(set_timelimit,             "fftw_set_timelimit"),
    FFTWX__ENTRY(cost,                      "fftw_cost"),
    FFTWX__ENTRY(flops,                     "fftw_flops"),
};

#ifdef _WIN32
#  define FFTWX_DEFAULT_DLL "C:\\vcpkg\\installed\\x64-windows\\bin\\fftw3.dll"
#else
#  define FFTWX_DEFAULT_DLL "libfftw3.so.3"   /* POSIX path untested; Windows is the target */
#endif

/* Bind. Returns 1 on success; on failure writes a reason into err and returns 0.
 * GENUINENESS is asserted here, not left to the caller: the version string must
 * begin "fftw-" and must not mention MKL/Intel — the wrapper DLL announces
 * itself as "FFTW 3.3.4 wrappers to Intel oneMKL". */
static inline int fftwx_bind(fftwx_api_t *api, char *err, size_t errcap)
{
    const char *path = getenv("VFFT_FFTW_DLL");
    if (!path || !*path) path = FFTWX_DEFAULT_DLL;
    memset(api, 0, sizeof *api);

#ifdef _WIN32
    {
        HMODULE h = LoadLibraryA(path);
        if (!h) { snprintf(err, errcap, "LoadLibraryA(\"%s\") failed, GetLastError=%lu",
                           path, (unsigned long)GetLastError()); return 0; }
        GetModuleFileNameA(h, api->dll_path, (DWORD)sizeof api->dll_path);
        for (size_t i = 0; i < sizeof fftwx__tab / sizeof fftwx__tab[0]; i++) {
            void *p = (void *)GetProcAddress(h, fftwx__tab[i].name);
            if (!p) { snprintf(err, errcap, "symbol '%s' missing in %s",
                               fftwx__tab[i].name, api->dll_path); return 0; }
            memcpy((char *)api + fftwx__tab[i].off, &p, sizeof p);
        }
        api->version = (const char *)(void *)GetProcAddress(h, "fftw_version");
    }
#else
    {
        void *h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!h) { snprintf(err, errcap, "dlopen(\"%s\"): %s", path, dlerror()); return 0; }
        snprintf(api->dll_path, sizeof api->dll_path, "%s", path);
        for (size_t i = 0; i < sizeof fftwx__tab / sizeof fftwx__tab[0]; i++) {
            void *p = dlsym(h, fftwx__tab[i].name);
            if (!p) { snprintf(err, errcap, "symbol '%s' missing in %s",
                               fftwx__tab[i].name, path); return 0; }
            memcpy((char *)api + fftwx__tab[i].off, &p, sizeof p);
        }
        api->version = (const char *)dlsym(h, "fftw_version");
    }
#endif

    if (!api->version) { snprintf(err, errcap, "fftw_version data symbol missing"); return 0; }
    if (strncmp(api->version, "fftw-", 5) != 0 ||
        strstr(api->version, "MKL") || strstr(api->version, "Intel") ||
        strstr(api->version, "oneMKL")) {
        snprintf(err, errcap, "NOT GENUINE FFTW: version=\"%s\" from %s — the MKL "
                 "wrapper DLL announces itself here; refuse to proceed",
                 api->version, api->dll_path);
        return 0;
    }
    return 1;
}

/* --------------------------------------------- deterministic planes (§3.1)
 * FFTW hashes the split-plane deltas (ii-ri)/(io-ro) into its WISDOM KEY, so
 * independently malloc'd planes miss the cache 6/6 and the plan drifts 9.4%
 * across launches. One block, two planes, size-derived offset: the delta is a
 * pure function of (N,K) => 6/6 WISDOM_ONLY hit, one plan_id. The stride keeps
 * the house 64 B skew on top of a 4 KB pitch. */
static inline ref_planes_t ref_planes_alloc(size_t n_doubles)
{
    size_t s = ref_plane_stride(n_doubles * sizeof(double));
    ref_planes_t p; memset(&p, 0, sizeof p);
#ifdef _WIN32
    p.blk = _aligned_malloc(2 * s, 64);
#else
    if (posix_memalign(&p.blk, 64, 2 * s) != 0) p.blk = NULL;
#endif
    if (!p.blk) return p;
    p.re = (double *)p.blk;
    p.im = (double *)((char *)p.blk + s);
    p.stride = s;
    return p;
}

static inline void ref_planes_free(ref_planes_t *p)
{
    if (!p || !p->blk) return;
#ifdef _WIN32
    _aligned_free(p->blk);
#else
    free(p->blk);
#endif
    p->blk = NULL; p->re = p->im = NULL;
}

/* --------------------------------------------------------------- plan_id
 * FNV-1a over the sprint_plan text. The string is address-free and stable
 * across processes (campaign: 6/6 identical plan_id in isolated runs), so the
 * hash is a wisdom-hit witness. 🔴 free() the string — NEVER fftw_free(). */
static inline uint64_t fftwx_plan_id(const fftwx_api_t *api, const fftwx_plan plan)
{
    char *s = api->sprint_plan(plan);
    uint64_t h = 1469598103934665603ull;
    if (s) {
        for (const char *c = s; *c; c++) { h ^= (unsigned char)*c; h *= 1099511628211ull; }
        free(s);                         /* NOT fftw_free — heap corruption otherwise */
    }
    return h;
}

#endif /* VFFT_BENCH_REF_FFTW_H */
