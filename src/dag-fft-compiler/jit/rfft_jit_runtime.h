/* rfft_jit_runtime.h — runtime JIT for the rfft (real FFT) PACKED forward.
 *
 * The rfft sibling of jit_runtime.h (c2c). rfft has NO baked specialized executor
 * (only the generic rfft_execute_fwd_packed), so JIT is the sole specialization:
 * emit one .c (emit_rfft_jit.py) -> compile to a shared lib (gcc, linking the rfft
 * codelet objects) -> dlopen -> return the `rfft_jit_exec` function pointer. Cache
 * in repo (jit/generated/) keyed by (N,K,factors,variants,isa,ver); a plan compiles
 * once, ever. NULL on any failure -> caller falls back to rfft_execute_fwd_packed.
 *
 * Include AFTER rfft.h (uses rfft_plan_t / rfft_execute_fwd_packed). Gated by the
 * caller under VFFT_USE_JIT, same as jit_runtime.h. */
#ifndef VFFT_RFFT_JIT_RUNTIME_H
#define VFFT_RFFT_JIT_RUNTIME_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
  #include <windows.h>
  #define VFFT_RJIT_LIB        HMODULE
  #define VFFT_RJIT_DLOPEN(p)  LoadLibraryA(p)
  #define VFFT_RJIT_DLSYM(l,s) ((void *)GetProcAddress((l),(s)))
  #define VFFT_RJIT_LIBEXT     "dll"
#else
  #include <dlfcn.h>
  #define VFFT_RJIT_LIB        void *
  #define VFFT_RJIT_DLOPEN(p)  dlopen((p), RTLD_NOW | RTLD_LOCAL)
  #define VFFT_RJIT_DLSYM(l,s) dlsym((l),(s))
  #define VFFT_RJIT_LIBEXT     "so"
#endif

/* ── Config — defaults shared with jit_runtime.h; #ifndef so the two coexist ── */
#if defined(_WIN32)
  #ifndef VFFT_PROTO_JIT_REPO
  #define VFFT_PROTO_JIT_REPO "C:/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler"
  #endif
  #ifndef VFFT_PROTO_JIT_GCC
  #define VFFT_PROTO_JIT_GCC  "C:\\mingw152\\mingw64\\bin\\gcc.exe"
  #endif
  #ifndef VFFT_PROTO_JIT_PYTHON
  #define VFFT_PROTO_JIT_PYTHON "python"
  #endif
  #ifndef VFFT_PROTO_JIT_CODELETS
  #define VFFT_PROTO_JIT_CODELETS "@" VFFT_PROTO_JIT_REPO "/jit/generated/codelets.rsp"
  #endif
  #ifndef VFFT_PROTO_JIT_CFLAGS
  #define VFFT_PROTO_JIT_CFLAGS "-O3 -mavx2 -mfma -march=haswell -shared"
  #endif
#else
  #ifndef VFFT_PROTO_JIT_REPO
  #define VFFT_PROTO_JIT_REPO "/mnt/c/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler"
  #endif
  #ifndef VFFT_PROTO_JIT_GCC
  #define VFFT_PROTO_JIT_GCC  "gcc"
  #endif
  #ifndef VFFT_PROTO_JIT_PYTHON
  #define VFFT_PROTO_JIT_PYTHON "python3"
  #endif
  #ifndef VFFT_PROTO_JIT_CODELETS
  #define VFFT_PROTO_JIT_CODELETS "@" VFFT_PROTO_JIT_REPO "/jit/generated/codelets_linux.rsp"
  #endif
  #ifndef VFFT_PROTO_JIT_CFLAGS
  #define VFFT_PROTO_JIT_CFLAGS "-O3 -mavx2 -mfma -march=haswell -shared -fPIC"
  #endif
#endif
#ifndef VFFT_PROTO_JIT_DIR
#define VFFT_PROTO_JIT_DIR  VFFT_PROTO_JIT_REPO "/jit/generated"
#endif
#ifndef VFFT_PROTO_JIT_INC
#define VFFT_PROTO_JIT_INC  VFFT_PROTO_JIT_REPO "/jit"
#endif
/* rfft.h lives in the core tree (self-contained: system headers only), so the
 * runtime compile needs just this one extra -I for the prelude to find it. */
#ifndef VFFT_RFFT_JIT_COREINC
#define VFFT_RFFT_JIT_COREINC  VFFT_PROTO_JIT_REPO "/../core/transforms/real"
#endif
#ifndef VFFT_PROTO_JIT_VERSION
#define VFFT_PROTO_JIT_VERSION 2
#endif

typedef void (*rfft_jit_fn)(const rfft_plan_t *, const double *, double *);

/* filename-safe key: nN_kK_<factors..>_v<variants>_<isa>_rfftfwd_verV. */
static inline void rfft_jit_key(int N, size_t K, const int *factors, int nf,
                                const int *variants, const char *isa,
                                char *out, size_t cap) {
    int n = snprintf(out, cap, "n%d_k%zu", N, K);
    for (int s = 0; s < nf; s++) n += snprintf(out + n, cap - n, "_%d", factors[s]);
    n += snprintf(out + n, cap - n, "_v");
    for (int s = 0; s < nf; s++) n += snprintf(out + n, cap - n, "%d", variants ? variants[s] : 0);
    snprintf(out + n, cap - n, "_%s_rfftfwd_ver%d", isa, VFFT_PROTO_JIT_VERSION);
}

/* Process-global shape-keyed registry: a plan compiles/loads once per run. */
typedef struct { char key[256]; rfft_jit_fn fn; VFFT_RJIT_LIB lib; } rfft_jit_entry_t;
static rfft_jit_entry_t g_rfft_jit_reg[256];
static int              g_rfft_jit_count = 0;

static inline rfft_jit_fn rfft_jit_reg_find(const char *key) {
    for (int i = 0; i < g_rfft_jit_count; i++)
        if (strcmp(g_rfft_jit_reg[i].key, key) == 0) return g_rfft_jit_reg[i].fn;
    return NULL;
}

/* emit (if needed) -> compile (if lib absent) -> load -> register. NULL on any
 * failure (caller falls back to rfft_execute_fwd_packed). */
static inline rfft_jit_fn
vfft_rfft_jit_resolve(int N, size_t K, const int *factors, int nf,
                      const int *variants, const char *isa) {
    char key[256];
    rfft_jit_key(N, K, factors, nf, variants, isa, key, sizeof key);
    rfft_jit_fn cached = rfft_jit_reg_find(key);
    if (cached) return cached;

    char lib[700], src[700];
    snprintf(lib, sizeof lib, "%s/rfftjit_%s.%s", VFFT_PROTO_JIT_DIR, key, VFFT_RJIT_LIBEXT);
    snprintf(src, sizeof src, "%s/rfftjit_%s.c",  VFFT_PROTO_JIT_DIR, key);

    FILE *probe = fopen(lib, "rb");
    if (probe) { fclose(probe); }                 /* cached -> skip emit + compile */
    else {
        char facs[256] = {0}, vars[256] = {0}, cmd[3600];
        size_t fp = 0, vp = 0;
        for (int s = 0; s < nf; s++) {
            fp += (size_t)snprintf(facs + fp, sizeof facs - fp, "%s%d", s ? "," : "", factors[s]);
            vp += (size_t)snprintf(vars + vp, sizeof vars - vp, "%s%d", s ? "," : "", variants ? variants[s] : 0);
        }
        snprintf(cmd, sizeof cmd,
            "%s %s/emit_rfft_jit.py --N %d --K %zu --factors %s --variants %s "
            "--isa %s --prelude rfft_jit_prelude.h --out %s",
            VFFT_PROTO_JIT_PYTHON, VFFT_PROTO_JIT_INC, N, K, facs, vars, isa, src);
        if (system(cmd) != 0) return NULL;
        /* -I<jit> for the prelude, -I<core/transforms/real> so it finds rfft.h. */
        snprintf(cmd, sizeof cmd,
            "%s %s -I%s -I%s "
            "-Wno-unused-function -Wno-incompatible-pointer-types -Wno-unused-result "
            "%s %s -o %s",
            VFFT_PROTO_JIT_GCC, VFFT_PROTO_JIT_CFLAGS, VFFT_PROTO_JIT_INC, VFFT_RFFT_JIT_COREINC,
            src, VFFT_PROTO_JIT_CODELETS, lib);
        if (system(cmd) != 0) return NULL;
    }

    VFFT_RJIT_LIB h = VFFT_RJIT_DLOPEN(lib);
    if (!h) return NULL;
    rfft_jit_fn fn = (rfft_jit_fn)VFFT_RJIT_DLSYM(h, "rfft_jit_exec");
    if (!fn) return NULL;

    int cap = (int)(sizeof g_rfft_jit_reg / sizeof g_rfft_jit_reg[0]);
    if (g_rfft_jit_count < cap) {
        snprintf(g_rfft_jit_reg[g_rfft_jit_count].key, sizeof g_rfft_jit_reg[0].key, "%s", key);
        g_rfft_jit_reg[g_rfft_jit_count].fn  = fn;
        g_rfft_jit_reg[g_rfft_jit_count].lib = h;
        g_rfft_jit_count++;
    }
    return fn;
}

#endif /* VFFT_RFFT_JIT_RUNTIME_H */
