/* vfft_front.c -- the library a program links when VectorFFT comes ready built: libvfft.dll (libvfft.so). It holds
 * no transform code. On the first call it looks at the CPU, loads the library built for the instruction set the CPU
 * runs -- libvfft_avx512 when the CPU runs AVX-512, else libvfft_avx2 -- from its own folder, and from then on every
 * function of vfft.h and vfft_diagnostics.h (and of vfftf.h, when compiled with VFFT_ENABLE_FLOAT) is one jump
 * into that library.
 *
 *   VFFT_ISA=avx2 | avx512   (environment) names the instruction set instead; a CPU that does not run it is refused.
 *
 * When no library can be loaded, vfft_create() and vfft_malloc() return NULL, vfft_info() returns -1, and
 * vfft_error() says why; the reason is also printed once on stderr. What every other function then returns is the
 * last column of its row in the list below.
 *
 * Built for plain x86-64 (no AVX), so it starts on any CPU (CMake: -DVFFT_BUILD_FRONT=ON):
 *   Windows  gcc -O2 -shared vfft_front.c -I<include> -o libvfft.dll -Wl,--out-implib,libvfft.dll.a -static
 *   Linux    gcc -O2 -fPIC -shared vfft_front.c -I<include> -o libvfft.so -Wl,-rpath,'$ORIGIN' -ldl */
#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE 1   /* RTLD_DEEPBIND */
#endif
#define VFFT_SHARED_BUILD   /* this file's functions are the exported ones */
#include "vfft.h"
#include "vfft_diagnostics.h"
#ifdef VFFT_ENABLE_FLOAT
#include "vfftf.h"
#endif
#include <cpuid.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#include <sched.h>
#endif

static volatile int front_state;   /* 0 nothing loaded yet, 1 loading, 2 loaded, 3 no library */
static char front_why[1280];       /* state 3: the reason (it may hold a path) */
static int front_ready(void);

/* no library: a caller's text buffer is left empty */
static char *front_empty(char *buf, size_t n)
{
    if (buf && n)
        buf[0] = 0;
    return buf;
}

/* EVERY FUNCTION THE FRONT PASSES ON, one row each, in the order of the headers:
 *   R(return type, name, (parameters), (arguments), what the call returns when no library could be loaded)
 *   V(name, (parameters), (arguments))                 a function that returns nothing
 * A function added to a public header gets a row here. */
#define FRONT_VFFT(R, V)                                                                                              \
    /* vfft.h */                                                                                                      \
    R(vfft_plan, vfft_create, (const vfft_config_t *config), (config), NULL)                                          \
    R(int, vfft_execute, (vfft_plan p, vfft_dir_t dir, const void *in, void *out), (p, dir, in, out), -1)             \
    R(int, vfft_execute_split,                                                                                        \
      (vfft_plan p, vfft_dir_t dir, const double *in_re, const double *in_im, double *out_re, double *out_im),        \
      (p, dir, in_re, in_im, out_re, out_im), -1)                                                                     \
    V(vfft_destroy, (vfft_plan p), (p))                                                                               \
    R(void *, vfft_malloc, (size_t bytes), (bytes), NULL)                                                             \
    V(vfft_free, (void *ptr), (ptr))                                                                                  \
    R(int, vfft_info, (vfft_plan p, vfft_info_t *out), (p, out), -1)                                                  \
    R(const char *, vfft_error, (void), (), front_why)                                                                \
    /* vfft_diagnostics.h */                                                                                          \
    R(int, vfft_plan_tc_workers, (vfft_plan p), (p), -1)                                                              \
    R(long, vfft_tc_mt_dispatches, (void), (), 0)                                                                     \
    R(long, vfft_il2d_col_mt_passes, (void), (), 0)                                                                   \
    R(long, vfft_il2d_row_mt_passes, (void), (), 0)                                                                   \
    R(long, vfft_ilnd_mt_passes, (void), (), 0)                                                                       \
    R(long, vfft_ilfd_mt_passes, (void), (), 0)                                                                       \
    R(long, vfft_ztt_mt_passes, (void), (), 0)                                                                        \
    R(long, vfft_zfsr_mt_passes, (void), (), 0)                                                                       \
    R(long, vfft_zttr_mt_passes, (void), (), 0)                                                                       \
    R(long, vfft_zrf_mt_passes, (void), (), 0)                                                                        \
    R(long, vfft_ilpr_mt_passes, (void), (), 0)                                                                       \
    R(long, vfft_zrb_mt_passes, (void), (), 0)                                                                        \
    R(long, vfft_zr2c_fold_mt_passes, (void), (), 0)                                                                  \
    R(long, vfft_ilfd_race_short_samples, (void), (), 0)                                                              \
    R(long, vfft_pq_mt_passes, (void), (), 0)                                                                         \
    R(long, vfft_pq_grp_passes, (void), (), 0)                                                                        \
    R(long, vfft_measure_scopes, (void), (), 0)                                                                       \
    V(vfft_measure_configure, (const vfft_measure_config_t *config), (config))                                        \
    R(int, vfft_measure_begin, (void), (), VFFT_MEASURE_CONTENDED | VFFT_MEASURE_UNPINNED)                            \
    V(vfft_measure_end, (void), ())                                                                                   \
    R(unsigned long long, vfft_measure_confine, (unsigned long long mask), (mask), 0)                                 \
    R(const char *, vfft_measure_describe, (char *buf, size_t n), (buf, n), front_empty(buf, n))                      \
    R(const char *, vfft_isa, (void), (), "")                                                                         \
    R(const char *, vfft_wisdom_folder, (void), (), "")                                                               \
    R(const char *, vfft_wisdom_identity, (void), (), "")                                                             \
    R(const char *, vfft_wisdom_build, (void), (), "")                                                                \
    R(size_t, vfft_wisdom_report, (const char *list_build, char *buf, size_t n), (list_build, buf, n),                \
      (front_empty(buf, n), (size_t)0))                                                                               \
    R(int, vfft_wisdom_reread, (const char *dir), (dir), -1)                                                          \
    R(int, vfft_pool_threads, (void), (), 1)
#ifdef VFFT_ENABLE_FLOAT
#define FRONT_VFFTF(R, V)                                                                                             \
    /* vfftf.h */                                                                                                     \
    R(vfftf_plan, vfftf_create, (const vfft_config_t *config), (config), NULL)                                        \
    R(int, vfftf_execute, (vfftf_plan p, vfft_dir_t dir, const void *in, void *out), (p, dir, in, out), -1)           \
    V(vfftf_destroy, (vfftf_plan p), (p))                                                                             \
    R(int, vfftf_info, (vfftf_plan p, vfft_info_t *out), (p, out), -1)
#else
#define FRONT_VFFTF(R, V)
#endif
#define FRONT_ALL(R, V) FRONT_VFFT(R, V) FRONT_VFFTF(R, V)

/* one pointer per function: until a library is loaded it points at a function that loads one first */
#define R(ret, name, params, args, fail) static ret first_##name params; static ret (*p_##name) params = first_##name;
#define V(name, params, args)            static void first_##name params; static void (*p_##name) params = first_##name;
FRONT_ALL(R, V)
#undef R
#undef V

/* the names, in the list's order */
static const char *const front_name[] = {
#define R(ret, name, params, args, fail) #name,
#define V(name, params, args)            #name,
    FRONT_ALL(R, V)
#undef R
#undef V
};
enum { FRONT_N = sizeof front_name / sizeof front_name[0] };

/* what the CPU and the operating system run: AVX2 with FMA; AVX-512 F and DQ */
static void front_cpu(int *avx2, int *avx512)
{
    unsigned a, b, c, d, lo = 0, hi = 0;
    *avx2 = *avx512 = 0;
    if (__get_cpuid_max(0, NULL) < 7)
        return;
    __cpuid_count(1, 0, a, b, c, d);
    if (!((c >> 27) & 1u) || !((c >> 28) & 1u) || !((c >> 12) & 1u))   /* OSXSAVE, AVX, FMA */
        return;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    if ((lo & 0x6u) != 0x6u)   /* the system saves the YMM registers */
        return;
    __cpuid_count(7, 0, a, b, c, d);
    *avx2 = (b >> 5) & 1u;
    *avx512 = *avx2 && (lo & 0xE6u) == 0xE6u && ((b >> 16) & 1u) && ((b >> 17) & 1u);
}

static int front_fail(const char *fmt, const char *a, const char *b)
{
    snprintf(front_why, sizeof front_why, fmt, a, b);
    fprintf(stderr, "vfft: %s\n", front_why);
    return 0;
}

/* load libvfft_<isa> from this library's folder and take its functions; 1 = loaded */
static int front_load(void)
{
    const char *want = getenv("VFFT_ISA"), *isa;
    void *fn[FRONT_N];
    int avx2, avx512, i;
    front_cpu(&avx2, &avx512);
    if (want && want[0])
    {
        if (strcmp(want, "avx2") && strcmp(want, "avx512"))
            return front_fail("VFFT_ISA=%s: write avx2 or avx512, or leave it unset%s", want, "");
        if (!strcmp(want, "avx512") ? !avx512 : !avx2)
            return front_fail("VFFT_ISA=%s, and this CPU does not run it%s", want,
                              avx2 ? " (leave VFFT_ISA unset, or set it to avx2)" : "");
        isa = want;
    }
    else if (avx512) isa = "avx512";
    else if (avx2) isa = "avx2";
    else return front_fail("this CPU lacks AVX2 and FMA, which VectorFFT needs%s%s", "", "");
    {
#if defined(_WIN32)
        static const char anchor = 1;   /* an address inside this library's file */
        char path[1024], *cut = NULL, *p;
        HMODULE me = NULL, h;
        DWORD l = 0;
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               (LPCSTR)&anchor, &me) && me)
            l = GetModuleFileNameA(me, path, (DWORD)sizeof path);
        if (!l || l >= sizeof path - 32)
            return front_fail("the folder of libvfft.dll could not be told%s%s", "", "");
        for (p = path; *p; p++)
            if (*p == '\\' || *p == '/')
                cut = p;
        snprintf(cut ? cut + 1 : path, 32, "libvfft_%s.dll", isa);
        h = LoadLibraryExA(path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (!h)
            return front_fail("%s could not be loaded: keep libvfft_avx2.dll and libvfft_avx512.dll beside "
                              "libvfft.dll%s", path, "");
        for (i = 0; i < FRONT_N; i++)
            if (!(fn[i] = (void *)GetProcAddress(h, front_name[i])))
                return front_fail("%s has no %s", path, front_name[i]);
#else
        char file[32];
        void *h;
        snprintf(file, sizeof file, "libvfft_%s.so", isa);
        h = dlopen(file, RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);   /* found through this library's own folder ($ORIGIN) */
        if (!h)
            return front_fail("%s could not be loaded: keep libvfft_avx2.so and libvfft_avx512.so beside "
                              "libvfft.so (%s)", file, dlerror());
        for (i = 0; i < FRONT_N; i++)
            if (!(fn[i] = dlsym(h, front_name[i])))
                return front_fail("%s has no %s", file, front_name[i]);
#endif
    }
    i = 0;
#define R(ret, name, params, args, fail) p_##name = (ret (*) params)fn[i++];
#define V(name, params, args)            p_##name = (void (*) params)fn[i++];
    FRONT_ALL(R, V)
#undef R
#undef V
    return 1;
}

/* 1 once a library is loaded (the first caller loads it, the others wait for it); 0 when none could be */
static int front_ready(void)
{
    for (;;)
    {
        const int s = __sync_val_compare_and_swap(&front_state, 0, 1);
        if (s == 0)
            front_state = front_load() ? 2 : 3;
        else if (s == 1)
#if defined(_WIN32)
            Sleep(0);
#else
            sched_yield();
#endif
        else
            return s == 2;
    }
}

/* each function twice: the one its pointer starts at (load a library, then call it), and the exported one of the
 * public headers (one jump through the pointer) */
#define R(ret, name, params, args, fail)                                               \
    static ret first_##name params { return front_ready() ? p_##name args : (fail); } \
    VFFT_API ret name params { return p_##name args; }
#define V(name, params, args)                                                \
    static void first_##name params { if (front_ready()) p_##name args; }   \
    VFFT_API void name params { p_##name args; }
FRONT_ALL(R, V)
#undef R
#undef V
