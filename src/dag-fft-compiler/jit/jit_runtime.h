/* jit_runtime.h — runtime JIT for dag-fft-compiler 1D C2C forward.
 *
 * Purpose: a LIVE planner discovers factorizations at runtime that have no
 * baked static executor in plan_executors.h, so they fall to the generic
 * executor (the per-stage-dispatch tax). This resolver gives such a plan the
 * SAME specialized executor without a rebuild: emit one .c (emit_jit.py),
 * compile it to a shared lib (gcc), load it, and hand back the function pointer.
 *
 * Contract (settled with the user):
 *   - Compile cost is locked to the PLANNER phase: call vfft_proto_plan_jit_fwd()
 *     once after planning; it returns the resolved executor. The caller holds
 *     that pointer and calls it directly in the hot loop -> ZERO exec overhead.
 *   - execute_fwd() is left UNTOUCHED -> zero regression for existing callers.
 *   - Persistent cache in generated/jit/: a plan compiles once, ever; subsequent
 *     processes just dlopen the cached lib. Key = (N,K,factors,variants,isa,ver).
 *   - Robust: if the toolchain is missing or anything fails, returns NULL and the
 *     caller falls back to the generic executor. JIT is a speed cache, never a
 *     correctness dependency.
 *
 * Why a returned pointer and not plan->exec_fwd: stride_plan_t lives in
 * AUTO-GENERATED plan_executors.h, which the calibrator re-emits whenever
 * spike_wisdom changes — a hand-added field would be wiped. Returning the fn
 * (FFTW-style plan handle) is the robust, regen-proof equivalent.
 */
#ifndef VFFT_PROTO_JIT_RUNTIME_H
#define VFFT_PROTO_JIT_RUNTIME_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../core/plan.h"   /* stride_plan_t, vfft_proto_exec_fn, lookup_fwd_avx2 */

#if defined(_WIN32)
  #include <windows.h>
  #define VFFT_JIT_LIB        HMODULE
  #define VFFT_JIT_DLOPEN(p)  LoadLibraryA(p)
  #define VFFT_JIT_DLSYM(l,s) ((void *)GetProcAddress((l),(s)))
  #define VFFT_JIT_LIBEXT     "dll"
#else
  #include <dlfcn.h>
  #define VFFT_JIT_LIB        void *
  #define VFFT_JIT_DLOPEN(p)  dlopen((p), RTLD_NOW | RTLD_LOCAL)
  #define VFFT_JIT_DLSYM(l,s) dlsym((l),(s))
  #define VFFT_JIT_LIBEXT     "so"
#endif

/* ── Config (platform defaults; all overridable at compile time) ───────── */
#if defined(_WIN32)
  #ifndef VFFT_PROTO_JIT_REPO
  #define VFFT_PROTO_JIT_REPO "C:/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler"
  #endif
  #ifndef VFFT_PROTO_JIT_GCC       /* program path: backslashes for cmd.exe */
  #define VFFT_PROTO_JIT_GCC  "C:\\mingw152\\mingw64\\bin\\gcc.exe"
  #endif
  #ifndef VFFT_PROTO_JIT_PYTHON
  #define VFFT_PROTO_JIT_PYTHON "python"
  #endif
  #ifndef VFFT_PROTO_JIT_CODELETS  /* @response-file of codelet .o (PE/COFF) */
  #define VFFT_PROTO_JIT_CODELETS "@" VFFT_PROTO_JIT_REPO "/generated/jit/codelets.rsp"
  #endif
  #ifndef VFFT_PROTO_JIT_CFLAGS
  #define VFFT_PROTO_JIT_CFLAGS "-O3 -mavx2 -mfma -march=haswell -shared"
  #endif
#else
  #ifndef VFFT_PROTO_JIT_REPO      /* WSL/Linux view of the repo */
  #define VFFT_PROTO_JIT_REPO "/mnt/c/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler"
  #endif
  #ifndef VFFT_PROTO_JIT_GCC
  #define VFFT_PROTO_JIT_GCC  "gcc"
  #endif
  #ifndef VFFT_PROTO_JIT_PYTHON
  #define VFFT_PROTO_JIT_PYTHON "python3"
  #endif
  #ifndef VFFT_PROTO_JIT_CODELETS  /* @response-file of codelet .o (ELF) */
  #define VFFT_PROTO_JIT_CODELETS "@" VFFT_PROTO_JIT_REPO "/generated/jit/codelets_linux.rsp"
  #endif
  #ifndef VFFT_PROTO_JIT_CFLAGS    /* -fPIC required for a Linux shared object */
  #define VFFT_PROTO_JIT_CFLAGS "-O3 -mavx2 -mfma -march=haswell -shared -fPIC"
  #endif
#endif
#ifndef VFFT_PROTO_JIT_DIR          /* persistent .c/lib cache (same relative path) */
#define VFFT_PROTO_JIT_DIR  VFFT_PROTO_JIT_REPO "/generated/jit"
#endif
#ifndef VFFT_PROTO_JIT_INC          /* dir of emit_jit.py + jit_prelude.h (also -I) */
#define VFFT_PROTO_JIT_INC  VFFT_PROTO_JIT_REPO "/jit"
#endif
#ifndef VFFT_PROTO_JIT_VERSION      /* bump to invalidate the on-disk cache */
#define VFFT_PROTO_JIT_VERSION 2
#endif

/* ── Variant recovery: derive the per-stage variant code from the wired plan
 *    (stage 0 is always OUTER; the code there is moot). ───────────────────── */
static inline int vfft_proto_jit_variant(const stride_stage_t *st, int stage) {
    if (stage == 0)    return 2;   /* OUTER (n1) — variant code unused for emit */
    if (st->use_log3)  return 1;   /* LOG3 */
    if (st->t1s_fwd)   return 2;   /* T1S  */
    return 0;                      /* FLAT */
}

/* filename-safe key = nN_kK_<factors..>_v<variants>_<isa>_verV */
static inline void vfft_proto_jit_key(const stride_plan_t *plan, const char *isa,
                                      char *out, size_t cap) {
    int n = snprintf(out, cap, "n%d_k%zu", plan->N, plan->K);
    for (int s = 0; s < plan->num_stages; s++)
        n += snprintf(out + n, cap - n, "_%d", plan->factors[s]);
    n += snprintf(out + n, cap - n, "_v");
    for (int s = 0; s < plan->num_stages; s++)
        n += snprintf(out + n, cap - n, "%d", vfft_proto_jit_variant(&plan->stages[s], s));
    snprintf(out + n, cap - n, "_%s_ver%d", isa, VFFT_PROTO_JIT_VERSION);
}

/* comma-separated factor + variant lists for emit_jit.py */
static inline void vfft_proto_jit_csv(const stride_plan_t *plan,
                                      char *facs, char *vars, size_t cap) {
    int fp = 0, vp = 0;
    for (int s = 0; s < plan->num_stages; s++) {
        fp += snprintf(facs + fp, cap - fp, "%s%d", s ? "," : "", plan->factors[s]);
        vp += snprintf(vars + vp, cap - vp, "%s%d", s ? "," : "",
                       vfft_proto_jit_variant(&plan->stages[s], s));
    }
}

/* ── Process-global registry (shape-keyed): a plan compiles/loads once per run ─ */
typedef struct { char key[256]; vfft_proto_exec_fn fn; VFFT_JIT_LIB lib; }
        vfft_proto_jit_entry_t;
static vfft_proto_jit_entry_t g_vfft_jit_reg[256];
static int                    g_vfft_jit_count = 0;

static inline vfft_proto_exec_fn vfft_proto_jit_reg_find(const char *key) {
    for (int i = 0; i < g_vfft_jit_count; i++)
        if (strcmp(g_vfft_jit_reg[i].key, key) == 0) return g_vfft_jit_reg[i].fn;
    return NULL;
}

/* emit (if needed) → compile (if .lib absent) → load → register. NULL on any
 * failure (caller falls back to generic). */
static inline vfft_proto_exec_fn
vfft_proto_jit_compile_load(const stride_plan_t *plan, const char *isa, const char *key) {
    char lib[700], src[700];
    snprintf(lib, sizeof lib, "%s/jit_%s.%s", VFFT_PROTO_JIT_DIR, key, VFFT_JIT_LIBEXT);
    snprintf(src, sizeof src, "%s/jit_%s.c",  VFFT_PROTO_JIT_DIR, key);

    FILE *probe = fopen(lib, "rb");
    if (probe) { fclose(probe); }                 /* cached → skip emit + compile */
    else {
        char facs[256], vars[256], cmd[3200];
        vfft_proto_jit_csv(plan, facs, vars, sizeof facs);
        /* emit one .c (no shell redirect — --out writes the file directly) */
        snprintf(cmd, sizeof cmd,
            "%s %s/emit_jit.py --N %d --K %zu --factors %s --variants %s "
            "--isa %s --prelude jit_prelude.h --out %s",
            VFFT_PROTO_JIT_PYTHON, VFFT_PROTO_JIT_INC, plan->N, plan->K, facs, vars, isa, src);
        if (system(cmd) != 0) return NULL;
        /* compile to a shared lib; -I so jit_prelude.h resolves */
        snprintf(cmd, sizeof cmd,
            "%s %s -I%s "
            "-Wno-unused-function -Wno-incompatible-pointer-types -Wno-unused-result "
            "%s %s -o %s",
            VFFT_PROTO_JIT_GCC, VFFT_PROTO_JIT_CFLAGS, VFFT_PROTO_JIT_INC,
            src, VFFT_PROTO_JIT_CODELETS, lib);
        if (system(cmd) != 0) return NULL;
    }

    VFFT_JIT_LIB h = VFFT_JIT_DLOPEN(lib);
    if (!h) return NULL;
    vfft_proto_exec_fn fn = (vfft_proto_exec_fn)VFFT_JIT_DLSYM(h, "vfft_proto_jit_exec");
    if (!fn) return NULL;

    int cap = (int)(sizeof g_vfft_jit_reg / sizeof g_vfft_jit_reg[0]);
    if (g_vfft_jit_count < cap) {
        snprintf(g_vfft_jit_reg[g_vfft_jit_count].key,
                 sizeof g_vfft_jit_reg[0].key, "%s", key);
        g_vfft_jit_reg[g_vfft_jit_count].fn  = fn;
        g_vfft_jit_reg[g_vfft_jit_count].lib = h;
        g_vfft_jit_count++;
    }
    return fn;
}

/* ── The resolver. Call once after planning (planner phase). Returns the forward
 *    executor: baked static if present, else JIT-compiled (cached), else NULL
 *    (caller uses vfft_proto_execute_fwd_generic). ─────────────────────────── */
static inline vfft_proto_exec_fn vfft_proto_plan_jit_fwd(const stride_plan_t *plan) {
    vfft_proto_exec_fn baked = vfft_proto_lookup_fwd_avx2(plan);
    if (baked) return baked;                       /* already specialized */
    char key[256];
    vfft_proto_jit_key(plan, "avx2", key, sizeof key);
    vfft_proto_exec_fn fn = vfft_proto_jit_reg_find(key);
    if (fn) return fn;                             /* compiled earlier this run */
    return vfft_proto_jit_compile_load(plan, "avx2", key);
}

#endif /* VFFT_PROTO_JIT_RUNTIME_H */
