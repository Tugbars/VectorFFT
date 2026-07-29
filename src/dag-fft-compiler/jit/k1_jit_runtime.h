/* k1_jit_runtime.h — plan-time JIT specialization for the K=1 engine
 * (row_major_engine.md §13.3 generalized; docs/performance/
 * k1_optimization_campaign.md O9).
 *
 * WHY: stride baking measured -42%/-22% at the two cells with AOT spec twins
 * (the K=1 loop runs 8-16 iterations, so runtime-stride address arithmetic is
 * first-order and it competes with the FMAs for ports). Pre-emitting a twin
 * per (radix x stride-tuple x variant) explodes; the JIT bakes ANY cell's
 * geometry at plan time instead.
 *
 * HOW (simpler than the classic executor JIT — no Python): write a tiny
 * wrapper .c that #includes the SHIPPED unbaked codelet sources and calls the
 * route's two stages with LITERAL strides; gcc -O3 + flatten inlines the
 * single call sites and constant-propagates — producing the same code as an
 * AOT --oop-spec-named twin. Compile to a shared lib, dlopen, return the fn.
 *
 * Contract (same as jit_runtime.h): compile cost locked to plan time;
 * persistent cache in jit/generated keyed (N, route, pair, isa, ver);
 * NULL on any failure -> caller falls back to the normal route fns. JIT is a
 * speed cache, never a correctness dependency.
 *
 * Covered routes (split axis; wisdom codes from oop_plan.h): 2PA(1), 2PB(2),
 * TWL(3), 2PA_L3(5). MONO is already fully baked AOT; 3P/3P_L3 carry the
 * separate transpose pass and are never per-cell winners on the split-ip
 * axis — not worth wrapper support today. IL routes: later.
 *
 * Wrapper ABI:
 *   void vfft_k1_jit_exec(const double *sr, const double *si,
 *                         double *dr, double *di,
 *                         double *col_re, double *col_im,
 *                         const double *qr, const double *qi);
 *   (scratch + twiddle tables passed in, NOT baked: the on-disk cache is
 *    shared across plans/processes, only the SHAPE is baked. For TWL pass the
 *    plan's LINEAR tables Qlr/Qli; all other routes pass Qr/Qi.)
 */
#ifndef VFFT_K1_JIT_RUNTIME_H
#define VFFT_K1_JIT_RUNTIME_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
  #include <windows.h>
  #define VFFT_K1JIT_LIB        HMODULE
  #define VFFT_K1JIT_DLOPEN(p)  LoadLibraryA(p)
  #define VFFT_K1JIT_DLSYM(l,s) ((void *)GetProcAddress((l),(s)))
  #define VFFT_K1JIT_LIBEXT     "dll"
#else
  #include <dlfcn.h>
  #define VFFT_K1JIT_LIB        void *
  #define VFFT_K1JIT_DLOPEN(p)  dlopen((p), RTLD_NOW | RTLD_LOCAL)
  #define VFFT_K1JIT_DLSYM(l,s) dlsym((l),(s))
  #define VFFT_K1JIT_LIBEXT     "so"
#endif

/* reuse the classic JIT's platform config (repo/gcc paths, cache dir) */
#ifndef VFFT_PROTO_JIT_REPO
#if defined(_WIN32)
#define VFFT_PROTO_JIT_REPO "C:/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler"
#else
#define VFFT_PROTO_JIT_REPO "/mnt/c/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler"
#endif
#endif
#ifndef VFFT_PROTO_JIT_GCC
#if defined(_WIN32)
#define VFFT_PROTO_JIT_GCC "C:\\mingw152\\mingw64\\bin\\gcc.exe"
#else
#define VFFT_PROTO_JIT_GCC "gcc"
#endif
#endif
#ifndef VFFT_PROTO_JIT_DIR
#define VFFT_PROTO_JIT_DIR VFFT_PROTO_JIT_REPO "/jit/generated"
#endif
#define VFFT_K1JIT_VERSION 3 /* v3: per-stage static wrappers (one flatten each) —
                              * v2's single merged mega-function scheduled ~20%
                              * worse than the AOT twins' separate functions. */

typedef void (*vfft_k1_jit_fn)(const double *, const double *,
                               double *, double *, double *, double *,
                               const double *, const double *);

/* process-global registry: one load per (shape) per run */
typedef struct { char key[128]; vfft_k1_jit_fn fn; VFFT_K1JIT_LIB lib; }
        vfft_k1_jit_entry_t;
static vfft_k1_jit_entry_t g_vfft_k1jit_reg[64];
static int                 g_vfft_k1jit_count = 0;

/* stage source files + fn names per route (deterministic naming). */
static inline int _vfft_k1jit_route_srcs(int route, int R1, int R2,
                                         char *s1, char *s2,
                                         char *f1, char *f2, size_t cap)
{
    /* stage 1 */
    switch (route) {
    case 2: /* 2PB: leaf with UL (transposed) stores */
        snprintf(s1, cap, "codelets/oop/avx2/radix%d_n1_oop_ugul_avx2.c", R2);
        snprintf(f1, cap, "radix%d_n1_oop_fwd_avx2_UG_UL", R2);
        break;
    case 1: case 3: case 5: /* 2PA family: plain UG leaf */
        snprintf(s1, cap, "codelets/oop/avx2/radix%d_n1_oop_avx2.c", R2);
        snprintf(f1, cap, "radix%d_n1_oop_fwd_avx2_UG_UG", R2);
        break;
    default:
        return -1;
    }
    /* stage 2 */
    switch (route) {
    case 1: /* 2PA: t1 with UL loads */
        snprintf(s2, cap, "codelets/oop/avx2/radix%d_t1_oop_ul_avx2.c", R1);
        snprintf(f2, cap, "radix%d_t1_oop_fwd_avx2_UL_UG", R1);
        break;
    case 3: /* TWL: linear-table t1_ul (caller passes Qlr/Qli) */
        snprintf(s2, cap, "codelets/oop/avx2/radix%d_t1_oop_ul_twl_avx2.c", R1);
        snprintf(f2, cap, "radix%d_t1_oop_fwd_avx2_UL_UG_twl", R1);
        break;
    case 5: /* 2PA_L3: log3 t1_ul */
        snprintf(s2, cap, "codelets/oop/avx2/radix%d_t1_oop_ul_log3_avx2.c", R1);
        snprintf(f2, cap, "radix%d_t1_oop_fwd_avx2_UL_UG_log3", R1);
        break;
    case 2: /* 2PB: standard flat t1 */
        snprintf(s2, cap, "codelets/oop/avx2/radix%d_t1_oop_avx2.c", R1);
        snprintf(f2, cap, "radix%d_t1_oop_fwd_avx2_UG_UG", R1);
        break;
    default:
        return -1;
    }
    return 0;
}

/* Record a resolve FAILURE in the process registry (fn = NULL) so the shape
 * is never re-attempted this run. Without this, a shape that cannot compile
 * (e.g. a stage source this build lacks) re-wrote its .c and re-shelled gcc
 * on EVERY plan create — the recorded N=8192 pathology (P0c): wisdom named
 * 2PB 64x128, the radix-128 UG_UL source does not exist, and each create
 * paid a full gcc launch just to fail identically. Cross-process the same
 * shape now costs two fopen probes (see refuse-before-touch below), not a
 * compiler run. */
static inline vfft_k1_jit_fn _vfft_k1jit_fail(const char *key)
{
    vfft_k1_jit_entry_t *e = &g_vfft_k1jit_reg[g_vfft_k1jit_count++];
    snprintf(e->key, sizeof e->key, "%s", key);
    e->fn = 0;
    e->lib = 0;
    return NULL;
}

/* Resolve (emit-on-miss, compile-on-miss, cache-forever — successes AND
 * failures, see _vfft_k1jit_fail). NULL = fall back. */
static inline vfft_k1_jit_fn vfft_k1_jit_resolve(int N, int R1, int R2, int route)
{
    if (route != 1 && route != 2 && route != 3 && route != 5) return NULL;
    char key[128];
    snprintf(key, sizeof key, "k1_n%d_r%d_%dx%d_avx2_ver%d",
             N, route, R1, R2, VFFT_K1JIT_VERSION);
    for (int i = 0; i < g_vfft_k1jit_count; i++)
        if (strcmp(g_vfft_k1jit_reg[i].key, key) == 0)
            return g_vfft_k1jit_reg[i].fn; /* NULL here = cached failure */
    if (g_vfft_k1jit_count >= 64) return NULL;

    char lib[700], src[700];
    snprintf(lib, sizeof lib, "%s/%s.%s", VFFT_PROTO_JIT_DIR, key, VFFT_K1JIT_LIBEXT);
    snprintf(src, sizeof src, "%s/%s.c", VFFT_PROTO_JIT_DIR, key);

    FILE *probe = fopen(lib, "rb");
    if (probe) { fclose(probe); }
    else {
        char s1[256], s2[256], f1[256], f2[256];
        if (_vfft_k1jit_route_srcs(route, R1, R2, s1, s2, f1, f2, sizeof s1) != 0)
            return _vfft_k1jit_fail(key);
        /* refuse-before-touch: if a stage source is absent, gcc can only
         * fail — refuse WITHOUT writing the wrapper .c or launching the
         * compiler (no fossil, no ~100ms gcc tax, works across processes). */
        {
            char sp[700];
            FILE *sf;
            snprintf(sp, sizeof sp, "%s/%s", VFFT_PROTO_JIT_REPO, s1);
            if (!(sf = fopen(sp, "rb"))) return _vfft_k1jit_fail(key);
            fclose(sf);
            snprintf(sp, sizeof sp, "%s/%s", VFFT_PROTO_JIT_REPO, s2);
            if (!(sf = fopen(sp, "rb"))) return _vfft_k1jit_fail(key);
            fclose(sf);
        }
        FILE *f = fopen(src, "w");
        if (!f) return _vfft_k1jit_fail(key);
        /* wrapper: include the SHIPPED unbaked sources; call with LITERAL
         * strides; flatten -> inline -> constant-propagate == the spec twin.
         * Stage-1 strides: L=R1, G=1; store (2pb) OLs=1/OGs=R2 else OL=R1/OG=1.
         * Stage-2: 2pb reads transposed scratch UG (R2,1); 2pa family reads
         * untransposed via UL (1,R1); both store natural (R2,1). me1=R1, me2=R2. */
        /* v3 shape: TWO static per-stage wrappers, each flattening exactly ONE
         * codelet — gcc then compiles two separately-scheduled specialized
         * functions (the AOT spec twins' shape). A single merged flatten
         * measured ~20% worse (register allocation across the fused body). */
        fprintf(f,
            "/* auto-generated K=1 JIT wrapper — %s. Shape baked, buffers passed. */\n"
            "#include \"%s\"\n"
            "#include \"%s\"\n", key, s1, s2);
        fprintf(f,
            "__attribute__((flatten,noinline)) static void _k1_s1(\n"
            "    const double *sr, const double *si, double *cr, double *ci)\n"
            "{\n");
        if (route == 2)
            fprintf(f, "    %s(sr, si, cr, ci, 0, 0, %d, 1, 1, %d, %d);\n",
                    f1, R1, R2, R1);
        else
            fprintf(f, "    %s(sr, si, cr, ci, 0, 0, %d, 1, %d, 1, %d);\n",
                    f1, R1, R1, R1);
        fprintf(f,
            "}\n"
            "__attribute__((flatten,noinline)) static void _k1_s2(\n"
            "    const double *cr, const double *ci, double *dr, double *di,\n"
            "    const double *qr, const double *qi)\n"
            "{\n");
        if (route == 2)
            fprintf(f, "    %s(cr, ci, dr, di, qr, qi, %d, 1, %d, 1, %d);\n",
                    f2, R2, R2, R2);
        else
            fprintf(f, "    %s(cr, ci, dr, di, qr, qi, 1, %d, %d, 1, %d);\n",
                    f2, R1, R2, R2);
        fprintf(f,
            "}\n"
            "void vfft_k1_jit_exec(const double *sr, const double *si,\n"
            "                      double *dr, double *di,\n"
            "                      double *col_re, double *col_im,\n"
            "                      const double *qr, const double *qi)\n"
            "{\n"
            "    _k1_s1(sr, si, col_re, col_im);\n"
            "    _k1_s2(col_re, col_im, dr, di, qr, qi);\n"
            "}\n");
        fclose(f);
        /* unquoted paths, matching jit_runtime.h: cmd.exe mangles a command
         * line that BEGINS with a quote, and none of these paths contain
         * spaces. */
        char cmd[2200];
        snprintf(cmd, sizeof cmd,
                 "%s -O3 -mavx2 -mfma -march=native -shared "
                 "-fno-semantic-interposition -w -I%s %s -o %s",
                 VFFT_PROTO_JIT_GCC, VFFT_PROTO_JIT_REPO, src, lib);
        if (system(cmd) != 0) return _vfft_k1jit_fail(key);
    }
    VFFT_K1JIT_LIB h = VFFT_K1JIT_DLOPEN(lib);
    if (!h) return _vfft_k1jit_fail(key);
    vfft_k1_jit_fn fn = (vfft_k1_jit_fn)VFFT_K1JIT_DLSYM(h, "vfft_k1_jit_exec");
    if (!fn) return _vfft_k1jit_fail(key);
    vfft_k1_jit_entry_t *e = &g_vfft_k1jit_reg[g_vfft_k1jit_count++];
    snprintf(e->key, sizeof e->key, "%s", key);
    e->fn = fn;
    e->lib = h;
    return fn;
}

#endif /* VFFT_K1_JIT_RUNTIME_H */
