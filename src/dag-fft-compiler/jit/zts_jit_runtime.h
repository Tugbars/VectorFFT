/* zts_jit_runtime.h — plan-time JIT for zturn-ts, the split library's 1D C2C
 * (split/rank1/zts.h). 2026-11-06.
 *
 * WHAT. The registry (generator/generated/zts_registry_avx2.h) holds a FUSED
 * DRIVER for every pow2 cell of ZTURN-T's list: one whole-transform function
 * per (family, N, chain) with the stage kernels of codelets/split/ztt inlined
 * and literal trip counts (zts_drivers.ml). Every other cell — the odd band,
 * the lane-major band, a pow2 chain outside the list — runs the STAGED walk:
 * the same kernels called one by one through the plan's stage table
 * (_zts_staged_s / _v). This resolver gives such a plan its fused driver at
 * plan time: it writes the driver zts_drivers.ml would have emitted for this
 * (family, N, chain), with the kernels' SHIPPED sources #included so their
 * bodies inline exactly as in the baked drivers, compiles it with the
 * library's own flags into a shared library, runs the kernels' stack
 * alignment on it (src/tools/build/align_kernel_stacks.py, Windows gcc: the
 * baked drivers get the same), loads it and hands back a registry cell.
 *
 * CONTRACT (the model of the other JITs here, jit_runtime.h):
 *   - the race ranks candidates on the staged walk; ONLY THE DEPLOYED PLAN is
 *     compiled (c2c_create_split.h _zts_plan_cell), once per process and once
 *     ever on disk: the cache in jit/generated/ is keyed (family, N, chain,
 *     isa, version). A fresh process loads the library in well under a ms.
 *   - the staged walk stays the fallback: NULL on ANY failure (no toolchain,
 *     a compile error, a missing symbol) and the plan runs as before. The JIT
 *     is a speed cache, never a correctness dependency.
 *   - the driver's arithmetic is the staged walk's (the in-tile test is
 *     tile % (R*L) == 0, as _zts_staged_s), so the two forms are bitwise.
 *   - no execute-path cost: the cell is a function pointer like a baked one.
 *
 * Switch: VFFT_USE_JIT at compile time (gauntlet/build.py: VFFT_JIT=1).
 * Logs under VFFT_NAT_LOG. */
#ifndef VFFT_ZTS_JIT_RUNTIME_H
#define VFFT_ZTS_JIT_RUNTIME_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
  #include <windows.h>
  #define VFFT_ZJIT_LIB        HMODULE
  #define VFFT_ZJIT_DLOPEN(p)  LoadLibraryA(p)
  #define VFFT_ZJIT_DLSYM(l,s) ((void *)GetProcAddress((l),(s)))
  #define VFFT_ZJIT_LIBEXT     "dll"
#else
  #include <dlfcn.h>
  #define VFFT_ZJIT_LIB        void *
  #define VFFT_ZJIT_DLOPEN(p)  dlopen((p), RTLD_NOW | RTLD_LOCAL)
  #define VFFT_ZJIT_DLSYM(l,s) dlsym((l),(s))
  #define VFFT_ZJIT_LIBEXT     "so"
#endif

/* ── Config: the other JITs' platform defaults, all overridable ───────────── */
#if defined(_WIN32)
  #ifndef VFFT_PROTO_JIT_REPO
  #define VFFT_PROTO_JIT_REPO "C:/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler"
  #endif
  #ifndef VFFT_PROTO_JIT_GCC       /* program path: backslashes for cmd.exe */
  #define VFFT_PROTO_JIT_GCC  "C:\\mingw152\\mingw64\\bin\\gcc.exe"
  #endif
  #ifndef VFFT_ZTS_JIT_AR
  #define VFFT_ZTS_JIT_AR     "C:\\mingw152\\mingw64\\bin\\ar.exe"
  #endif
  #ifndef VFFT_PROTO_JIT_PYTHON
  #define VFFT_PROTO_JIT_PYTHON "python"
  #endif
  #ifndef VFFT_ZTS_JIT_CFLAGS      /* gauntlet/build.py's codelet flags, avx2 clamped */
  #define VFFT_ZTS_JIT_CFLAGS "-O3 -mavx2 -mfma -march=native -mno-avx512f -fpermissive -w"
  #endif
  #define VFFT_ZTS_JIT_PIC ""
#else
  #ifndef VFFT_PROTO_JIT_REPO
  #define VFFT_PROTO_JIT_REPO "/mnt/c/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler"
  #endif
  #ifndef VFFT_PROTO_JIT_GCC
  #define VFFT_PROTO_JIT_GCC  "gcc"
  #endif
  #ifndef VFFT_ZTS_JIT_AR
  #define VFFT_ZTS_JIT_AR     "ar"
  #endif
  #ifndef VFFT_PROTO_JIT_PYTHON
  #define VFFT_PROTO_JIT_PYTHON "python3"
  #endif
  #ifndef VFFT_ZTS_JIT_CFLAGS
  #define VFFT_ZTS_JIT_CFLAGS "-O3 -mavx2 -mfma -march=native -mno-avx512f -fpermissive -w"
  #endif
  #define VFFT_ZTS_JIT_PIC "-fPIC"
#endif
#ifndef VFFT_PROTO_JIT_DIR          /* the persistent cache (gitignored) */
#define VFFT_PROTO_JIT_DIR  VFFT_PROTO_JIT_REPO "/jit/generated"
#endif
#ifndef VFFT_ZTS_JIT_CODELETS       /* the stage kernels' shipped sources */
#define VFFT_ZTS_JIT_CODELETS VFFT_PROTO_JIT_REPO "/codelets/split/ztt"
#endif
#ifndef VFFT_ZTS_JIT_ALIGN          /* the kernels' stack alignment (Windows gcc) */
#define VFFT_ZTS_JIT_ALIGN  VFFT_PROTO_JIT_REPO "/../tools/build/align_kernel_stacks.py"
#endif
#ifndef VFFT_ZTS_JIT_VERSION        /* bump to invalidate the on-disk cache */
#define VFFT_ZTS_JIT_VERSION 1
#endif
#ifndef VFFT_ZTS_JIT_ISA
#define VFFT_ZTS_JIT_ISA "avx2"
#endif
#define VFFT_ZTS_JIT_VW 4           /* the isa's lanes per vector (the V family's group) */

/* ── The emitter: zts_drivers.ml emit_driver in C, one TU per (family, N, chain) ── */
static int _zts_jit_tag(char *out, size_t cap, int N, const int *chain, int nf)
{
    int s, n = snprintf(out, cap, "%d", N);
    for (s = 0; s < nf; s++) n += snprintf(out + n, cap - (size_t)n, "_%d", chain[s]);
    return n;
}
static const char *_zts_jit_kind(int s, int nf)
{
    return s == 0 ? "t0tp" : s == nf - 1 ? "tlfi" : "tmg";
}
/* the kernels' sources, deduplicated: #include of codelets/split/ztt/<isa>/radix<R>_<fam>_<kind>[_bwd]_<isa>.c */
static void _zts_jit_emit_sources(FILE *f, const char *fam, int N, const int *chain, int nf)
{
    char seen[64][40];
    int nseen = 0, s, bwd, i;
    (void)N;
    for (bwd = 0; bwd < 2; bwd++)
        for (s = 0; s < nf; s++)
        {
            char name[40];
            snprintf(name, sizeof name, "radix%d_%s_%s%s_%s.c", chain[s], fam, _zts_jit_kind(s, nf),
                     bwd ? "_bwd" : "", VFFT_ZTS_JIT_ISA);
            for (i = 0; i < nseen; i++)
                if (strcmp(seen[i], name) == 0) break;
            if (i < nseen) continue;
            if (nseen < 64) snprintf(seen[nseen++], sizeof seen[0], "%s", name);
            fprintf(f, "#include \"%s/%s/%s\"\n", VFFT_ZTS_JIT_CODELETS, VFFT_ZTS_JIT_ISA, name);
        }
    fputc('\n', f);
}
/* one direction's driver. S: the block plane, the mids in place on it, the
 * terminator to the caller's planes; + the _batch twin. V: the blocked loop
 * over the lane range [g0, g1), the mids per group on its own plane pair. */
static void _zts_jit_emit_driver(FILE *f, int fam_v, int N, const int *chain, int nf, int bwd)
{
    const char *fam = fam_v ? "ztv" : "zts", *d = bwd ? "b" : "f";
    const int sc = fam_v ? VFFT_ZTS_JIT_VW : 2;   /* doubles per complex per plane: the block plane 2, a lane group VW */
    long L[8], gs[8], off[8];
    long ncol = N / chain[0];
    char tag[96], name[160];
    const char *ind = fam_v ? "        " : "";   /* the V body sits inside the block + group loops */
    int s;
    L[1] = chain[0];
    off[1] = 0;
    for (s = 1; s < nf; s++)
    {
        const long RL = L[s] * chain[s];
        gs[s] = N / RL;
        if (s + 1 < nf) { L[s + 1] = RL; off[s + 1] = off[s] + 2 * (chain[s] - 1) * L[s]; }
    }
    _zts_jit_tag(tag, sizeof tag, N, chain, nf);
    snprintf(name, sizeof name, "%s_%s_%s_plane_%s", fam, tag, bwd ? "bwd" : "fwd", VFFT_ZTS_JIT_ISA);
    fprintf(f, "__attribute__((target(\"avx2,fma\")))\n");
    if (!fam_v)
    {
        fprintf(f, "void %s(const double *in_re, const double *in_im, double *out_re, double *out_im, double *plane, const double *tw, const size_t *rb, size_t tile)\n{\n", name);
        fprintf(f, "    double *W = plane;   /* the block interior; the last stage writes the split planes */\n");
        fprintf(f, "    _ztst0tp%d%s_body(in_re, in_im, W, rb, (size_t)%ld, (size_t)%ld);\n", chain[0], d, ncol, ncol);
    }
    else
    {
        fprintf(f, "void %s(const double *in_re, const double *in_im, double *out_re, double *out_im, double *plane_re, double *plane_im, const double *tw, const size_t *rb, size_t tile, size_t K, size_t blk, size_t gp, size_t g0, size_t g1)\n{\n", name);
        fprintf(f, "    double *W0 = plane_re, *Wi0 = plane_im;   /* the block's lane-group planes: blk of them at pitch gp */\n");
        fprintf(f, "    for (size_t gl = g0; gl < g1; gl += %d * blk)\n    {\n", VFFT_ZTS_JIT_VW);
        fprintf(f, "        const size_t nb = ((g1 - gl) / %d < blk) ? (g1 - gl) / %d : blk;\n", VFFT_ZTS_JIT_VW, VFFT_ZTS_JIT_VW);
        fprintf(f, "        const double *gi_re = in_re + gl, *gi_im = in_im + gl;\n");
        fprintf(f, "        double *go_re = out_re + gl, *go_im = out_im + gl;\n");
        fprintf(f, "        _ztvt0tp%d%s_body(gi_re, gi_im, W0, Wi0, rb, (size_t)%ld, K, nb, gp, (size_t)%ld);\n", chain[0], d, ncol, ncol);
        fprintf(f, "        for (size_t gb = 0; gb < nb; gb++)\n        {\n");
        fprintf(f, "            double *W = W0 + gb * gp, *Wi = Wi0 + gb * gp;\n");
    }
    if (nf >= 3)
    {   /* IN-TILE: the mids whose R*L divides the tile, per tile */
        fprintf(f, "%s    if (tile)\n%s    {\n", ind, ind);
        fprintf(f, "%s        const size_t ntile = (size_t)%d / tile;\n", ind, N);
        fprintf(f, "#pragma GCC unroll 1\n%s        for (size_t t = 0; t < ntile; t++)\n%s        {\n", ind, ind);
        if (!fam_v) fprintf(f, "%s            double *B = W + t * tile * 2;\n", ind);
        else        fprintf(f, "%s            double *B = W + t * tile * %d, *Bi = Wi + t * tile * %d;\n", ind, sc, sc);
        for (s = 1; s < nf - 1; s++)
        {
            const long RL = L[s] * chain[s], pitch = sc * RL;
            fprintf(f, "%s            if (tile %% (size_t)%ld == 0)   /* stage %d: R*L = %ld */\n%s            {\n", ind, RL, s, RL, ind);
            fprintf(f, "#pragma GCC unroll 1\n%s                for (size_t g = 0; g < tile / (size_t)%ld; g++)\n", ind, RL);
            if (!fam_v)
                fprintf(f, "%s                    _ztstmg%d%s_body(B + g * (size_t)%ld, B + g * (size_t)%ld, tw + %ld, (size_t)%ld, (size_t)%ld);\n",
                        ind, chain[s], d, pitch, pitch, off[s], L[s], L[s]);
            else
                fprintf(f, "%s                    _ztvtmg%d%s_body(B + g * (size_t)%ld, Bi + g * (size_t)%ld, B + g * (size_t)%ld, Bi + g * (size_t)%ld, tw + %ld, (size_t)%ld, (size_t)%ld);\n",
                        ind, chain[s], d, pitch, pitch, pitch, pitch, off[s], L[s], L[s]);
            fprintf(f, "%s            }\n", ind);
        }
        fprintf(f, "%s        }\n%s    }\n", ind, ind);
    }
    else
        fprintf(f, "%s    (void)tile;   /* no mids: nothing to tile */\n", ind);
    /* CROSS-TILE: the mids the tile does not hold, whole plane */
    for (s = 1; s < nf - 1; s++)
    {
        const long RL = L[s] * chain[s], pitch = sc * RL;
        fprintf(f, "%s    if (!tile || tile %% (size_t)%ld)   /* stage %d: R*L = %ld */\n%s    {\n", ind, RL, s, RL, ind);
        if (gs[s] == 1)
        {
            if (!fam_v) fprintf(f, "%s        _ztstmg%d%s_body(W, W, tw + %ld, (size_t)%ld, (size_t)%ld);\n", ind, chain[s], d, off[s], L[s], L[s]);
            else        fprintf(f, "%s        _ztvtmg%d%s_body(W, Wi, W, Wi, tw + %ld, (size_t)%ld, (size_t)%ld);\n", ind, chain[s], d, off[s], L[s], L[s]);
        }
        else
        {
            fprintf(f, "#pragma GCC unroll 1\n%s        for (size_t g = 0; g < (size_t)%ld; g++)\n", ind, gs[s]);
            if (!fam_v)
                fprintf(f, "%s            _ztstmg%d%s_body(W + g * (size_t)%ld, W + g * (size_t)%ld, tw + %ld, (size_t)%ld, (size_t)%ld);\n",
                        ind, chain[s], d, pitch, pitch, off[s], L[s], L[s]);
            else
                fprintf(f, "%s            _ztvtmg%d%s_body(W + g * (size_t)%ld, Wi + g * (size_t)%ld, W + g * (size_t)%ld, Wi + g * (size_t)%ld, tw + %ld, (size_t)%ld, (size_t)%ld);\n",
                        ind, chain[s], d, pitch, pitch, pitch, pitch, off[s], L[s], L[s]);
        }
        fprintf(f, "%s    }\n", ind);
    }
    s = nf - 1;
    if (!fam_v)
    {
        fprintf(f, "    _ztstlfi%d%s_body(W, out_re, out_im, tw + %ld, (size_t)%ld, (size_t)%ld, (size_t)%ld);\n}\n\n",
                chain[s], d, off[s], L[s], L[s], L[s]);
        /* the CONTIGUOUS batch twin, transform-major around the whole pipeline */
        fprintf(f, "__attribute__((target(\"avx2,fma\")))\n");
        fprintf(f, "void %s_batch(const double *in_re, const double *in_im, double *out_re, double *out_im, double *plane, const double *tw, const size_t *rb, size_t tile, size_t K)\n{\n", name);
        fprintf(f, "    for (size_t t = 0; t < K; t++)\n        %s(in_re + t * %d, in_im + t * %d, out_re + t * %d, out_im + t * %d, plane, tw, rb, tile);\n}\n\n",
                name, N, N, N, N);
    }
    else
    {
        fprintf(f, "        }\n");
        fprintf(f, "        _ztvtlfi%d%s_body(W0, Wi0, go_re, go_im, tw + %ld, (size_t)%ld, (size_t)%ld, K, nb, gp, (size_t)%ld);\n    }\n}\n\n",
                chain[s], d, off[s], L[s], L[s], L[s]);
    }
}
static int _zts_jit_emit(const char *path, int fam_v, int N, const int *chain, int nf)
{
    FILE *f = fopen(path, "wb");
    char tag[96];
    int s;
    if (!f) return 0;
    _zts_jit_tag(tag, sizeof tag, N, chain, nf);
    fprintf(f, "/* zturn-ts JIT (jit/zts_jit_runtime.h): family %s, N = %d, chain", fam_v ? "ztv" : "zts", N);
    for (s = 0; s < nf; s++) fprintf(f, "%s%d", s ? "." : " ", chain[s]);
    fprintf(f, " — the fused driver zts_drivers.ml emits for a registry cell, written at plan time\n"
               " * for this cell; the stage kernels' shipped sources included so their bodies inline. */\n");
    fprintf(f, "#include <immintrin.h>\n#include <stddef.h>\n");
    _zts_jit_emit_sources(f, fam_v ? "ztv" : "zts", N, chain, nf);
    _zts_jit_emit_driver(f, fam_v, N, chain, nf, 0);
    _zts_jit_emit_driver(f, fam_v, N, chain, nf, 1);
    return fclose(f) == 0;
}

/* ── The process registry: a cell compiles / loads once per run ───────────── */
typedef struct { char key[160]; vfft_zts_cell_t cell; VFFT_ZJIT_LIB lib; } _zts_jit_entry_t;
static _zts_jit_entry_t g_zts_jit_reg[128];
static int g_zts_jit_count = 0;

static inline int _zts_jit_file_exists(const char *p)
{
    FILE *f = fopen(p, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

/* emit → compile → (Windows gcc) the kernels' stack alignment → link → load.
 * 0 on failure; the partial files stay for inspection under the log. */
static int _zts_jit_build(const char *key, int fam_v, int N, const int *chain, int nf, const char *lib, int lg)
{
    char src[640], obj[640], arc[640], log[640], cmd[4096];
    snprintf(src, sizeof src, "%s/zts_jit_%s.c", VFFT_PROTO_JIT_DIR, key);
    snprintf(obj, sizeof obj, "%s/zts_jit_%s.o", VFFT_PROTO_JIT_DIR, key);
    snprintf(arc, sizeof arc, "%s/zts_jit_%s.a", VFFT_PROTO_JIT_DIR, key);
    snprintf(log, sizeof log, "%s/zts_jit_%s.log", VFFT_PROTO_JIT_DIR, key);
    if (!_zts_jit_emit(src, fam_v, N, chain, nf)) return 0;
    remove(log);
    snprintf(cmd, sizeof cmd, "%s %s %s -c %s -o %s >> %s 2>&1",
             VFFT_PROTO_JIT_GCC, VFFT_ZTS_JIT_CFLAGS, VFFT_ZTS_JIT_PIC, src, obj, log);
    if (system(cmd) != 0) { if (lg) fprintf(stderr, "[zts-jit] %s: compile failed (%s)\n", key, log); return 0; }
    remove(arc);
    snprintf(cmd, sizeof cmd, "%s rcs %s %s >> %s 2>&1", VFFT_ZTS_JIT_AR, arc, obj, log);
    if (system(cmd) != 0) { if (lg) fprintf(stderr, "[zts-jit] %s: ar failed (%s)\n", key, log); return 0; }
#if defined(_WIN32)
    /* the kernels' 64-byte frames: the driver inlines kernels that spill, so
     * its position in the line is the process's draw without the wrapper */
    snprintf(cmd, sizeof cmd, "%s %s %s --cc %s >> %s 2>&1",
             VFFT_PROTO_JIT_PYTHON, VFFT_ZTS_JIT_ALIGN, arc, VFFT_PROTO_JIT_GCC, log);
    if (system(cmd) != 0) { if (lg) fprintf(stderr, "[zts-jit] %s: stack alignment failed (%s)\n", key, log); return 0; }
#endif
    snprintf(cmd, sizeof cmd, "%s -shared %s -o %s -Wl,--whole-archive %s -Wl,--no-whole-archive >> %s 2>&1",
             VFFT_PROTO_JIT_GCC, VFFT_ZTS_JIT_PIC, lib, arc, log);
    if (system(cmd) != 0) { if (lg) fprintf(stderr, "[zts-jit] %s: link failed (%s)\n", key, log); return 0; }
    return 1;
}

/* The resolver: the fused driver cell for (N, chain, lanes), or NULL (the
 * plan keeps its staged walk). lanes = 1 names the S family, > 1 the V. */
static inline const vfft_zts_cell_t *vfft_zts_jit_resolve(int N, const int *chain, int nf, int lanes)
{
    const int fam_v = lanes > 1, lg = getenv("VFFT_NAT_LOG") != NULL;
    char key[160], tag[96], lib[640], sym[200];
    _zts_jit_entry_t *e;
    VFFT_ZJIT_LIB h;
    int i, cached;
    if (nf < 2 || nf > 7) return NULL;
    _zts_jit_tag(tag, sizeof tag, N, chain, nf);
    snprintf(key, sizeof key, "%s_%s_%s_v%d", fam_v ? "ztv" : "zts", tag, VFFT_ZTS_JIT_ISA, VFFT_ZTS_JIT_VERSION);
    for (i = 0; i < g_zts_jit_count; i++)
        if (strcmp(g_zts_jit_reg[i].key, key) == 0) return &g_zts_jit_reg[i].cell;
    if (g_zts_jit_count >= (int)(sizeof g_zts_jit_reg / sizeof g_zts_jit_reg[0])) return NULL;
    snprintf(lib, sizeof lib, "%s/zts_jit_%s.%s", VFFT_PROTO_JIT_DIR, key, VFFT_ZJIT_LIBEXT);
    cached = _zts_jit_file_exists(lib);
    if (!cached && !_zts_jit_build(key, fam_v, N, chain, nf, lib, lg)) return NULL;
    h = VFFT_ZJIT_DLOPEN(lib);
    if (!h) { if (lg) fprintf(stderr, "[zts-jit] %s: load failed\n", key); return NULL; }
    e = &g_zts_jit_reg[g_zts_jit_count];
    memset(e, 0, sizeof *e);
    snprintf(e->key, sizeof e->key, "%s", key);
    e->cell.n = N;
    e->cell.nf = nf;
    for (i = 0; i < nf; i++) e->cell.chain[i] = chain[i];
    if (!fam_v)
    {
        snprintf(sym, sizeof sym, "zts_%s_fwd_plane_%s", tag, VFFT_ZTS_JIT_ISA);
        e->cell.s_fwd = (vfft_zts_fn)VFFT_ZJIT_DLSYM(h, sym);
        snprintf(sym, sizeof sym, "zts_%s_fwd_plane_%s_batch", tag, VFFT_ZTS_JIT_ISA);
        e->cell.s_fwd_batch = (vfft_zts_bfn)VFFT_ZJIT_DLSYM(h, sym);
        snprintf(sym, sizeof sym, "zts_%s_bwd_plane_%s", tag, VFFT_ZTS_JIT_ISA);
        e->cell.s_bwd = (vfft_zts_fn)VFFT_ZJIT_DLSYM(h, sym);
        snprintf(sym, sizeof sym, "zts_%s_bwd_plane_%s_batch", tag, VFFT_ZTS_JIT_ISA);
        e->cell.s_bwd_batch = (vfft_zts_bfn)VFFT_ZJIT_DLSYM(h, sym);
        if (!e->cell.s_fwd || !e->cell.s_bwd || !e->cell.s_fwd_batch || !e->cell.s_bwd_batch)
        { if (lg) fprintf(stderr, "[zts-jit] %s: a driver symbol is missing\n", key); return NULL; }
    }
    else
    {
        snprintf(sym, sizeof sym, "ztv_%s_fwd_plane_%s", tag, VFFT_ZTS_JIT_ISA);
        e->cell.v_fwd = (vfft_ztv_fn)VFFT_ZJIT_DLSYM(h, sym);
        snprintf(sym, sizeof sym, "ztv_%s_bwd_plane_%s", tag, VFFT_ZTS_JIT_ISA);
        e->cell.v_bwd = (vfft_ztv_fn)VFFT_ZJIT_DLSYM(h, sym);
        if (!e->cell.v_fwd || !e->cell.v_bwd)
        { if (lg) fprintf(stderr, "[zts-jit] %s: a driver symbol is missing\n", key); return NULL; }
    }
    e->lib = h;
    g_zts_jit_count++;
    if (lg) fprintf(stderr, "[zts-jit] %s: %s\n", key, cached ? "loaded from the cache" : "compiled, aligned and loaded");
    return &e->cell;
}

#endif /* VFFT_ZTS_JIT_RUNTIME_H */
