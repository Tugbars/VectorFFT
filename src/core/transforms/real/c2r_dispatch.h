/* c2r_dispatch.h — wisdom-first entry point for the backward real FFT (c2r).
 *
 * Mirror of r2c_dispatch.h's wisdom path, minus the rfft-vs-stride routing
 * (c2r has a single executor, c2r.h). A caller that loaded c2r_wisdom.txt sets
 * the wisdom pointer; vfft_c2r_plan_create then builds the calibrated
 * factorization + per-stage variant on a hit, else the fewest-stage heuristic
 * (variant=NULL, the legacy default policy). NULL-safe — with no wisdom set this
 * is exactly today's c2r_plan_create behavior.
 *
 * c2r and r2c are calibrated SEPARATELY (different codelets: c2r uses the
 * DIF-backward hc2hc family, so its best per-stage variant can differ) — hence a
 * distinct c2r_wisdom.txt, written by calibrate_c2r.c.
 */
#ifndef VFFT_C2R_DISPATCH_H
#define VFFT_C2R_DISPATCH_H

#include "planner.h"   /* vfft_proto_wisdom_t + lookup (wisdom_reader.h) */
#include "c2r.h"
#include "r2c_dispatch.h"   /* 2-axis SPLIT path: _vfft_r2c_build_stride + _vfft_r2c_decouple_min_k + c2c wisdom */
#ifdef VFFT_USE_JIT
#include "c2r_jit_runtime.h"   /* after c2r.h: resolve the c2r winner's JIT executor */
#endif

/* Optional c2r wisdom (calibrated per-cell factorization + per-stage variant). */
static const vfft_proto_wisdom_t *_vfft_c2r_wis = NULL;
static inline void vfft_c2r_dispatch_set_wisdom(const vfft_proto_wisdom_t *w) { _vfft_c2r_wis = w; }

/* Fewest-stage factorization over the c2r-coverable radix set, larger radixes
 * first. STAGE radixes need r2cb[r] AND hc2hc_dif_bwd[r]; 32 is LEAF-only (no
 * DIF-backward hc2hc[32]), so — exactly as in r2c — it is excluded from the
 * heuristic and only reached via calibrated leaf-32 wisdom. Returns nf>=1 or 0. */
static inline int vfft_c2r_choose_factors(int N, int *factors, int max_nf)
{
    static const int cand[] = { 16, 8, 7, 5, 4, 3, 2 };   /* no 32: leaf-only */
    int rem = N, nf = 0;
    while (rem > 1) {
        if (nf >= max_nf) return 0;
        int picked = 0;
        for (unsigned ci = 0; ci < sizeof(cand)/sizeof(cand[0]); ci++) {
            int r = cand[ci];
            if (r > VFFT_RFFT_MAX_RADIX) continue;
            if (rem % r == 0) { factors[nf++] = r; rem /= r; picked = 1; break; }
        }
        if (!picked) return 0;
    }
    if (nf == 0) factors[nf++] = 1;
    return nf;
}

/* Build a c2r plan, wisdom-first. reg = rfft codelet registry (r2cb + DIF-bwd
 * hc2hc families). Returns NULL if N is not coverable. */
static inline c2r_plan_t *vfft_c2r_plan_create(int N, size_t K, const rfft_codelets_t *reg)
{
    if (N < 2 || K == 0 || (K % 8) != 0 || !reg) return NULL;
    int factors[VFFT_RFFT_MAX_STAGES];
    int nf = 0;
    const int *variant = NULL;   /* NULL => default policy in c2r_plan_create_ex */
    const vfft_proto_wisdom_entry_t *we =
        _vfft_c2r_wis ? vfft_proto_wisdom_lookup(_vfft_c2r_wis, N, (size_t)K) : NULL;
    if (we && we->nf >= 1 && we->nf <= VFFT_RFFT_MAX_STAGES) {
        nf = we->nf;
        for (int i = 0; i < nf; i++) factors[i] = we->factors[i];
        variant = we->variants;
    } else {
        nf = vfft_c2r_choose_factors(N, factors, VFFT_RFFT_MAX_STAGES);
    }
    if (nf < 1) return NULL;
#ifdef VFFT_USE_JIT
    /* JIT build: pin EXPLICIT per-stage variants so the plan and the resolved JIT
     * executor match (smoke-proven bit-exact). Wisdom -> its variants; heuristic ->
     * all-flat. Then compile the winner's JIT now (cached) and store it. */
    int vbuf[VFFT_RFFT_MAX_STAGES];
    for (int i = 0; i < nf; i++) vbuf[i] = (variant ? variant[i] : 0);
    c2r_plan_t *p = c2r_plan_create_ex(N, K, factors, nf, vbuf, reg);
    if (p) p->jit_exec = (void *)vfft_c2r_jit_resolve(N, K, factors, nf, vbuf, "avx2");
    return p;
#else
    return c2r_plan_create_ex(N, K, factors, nf, variant, reg);
#endif
}

/* Execute c2r: JIT-first, generic fallback (mirrors vfft_r2c_execute_fwd). */
static inline void vfft_c2r_execute(const c2r_plan_t *p, const double *in, double *out)
{
#ifdef VFFT_USE_JIT
    if (p->jit_exec) { ((c2r_jit_fn)p->jit_exec)(p, in, out); return; }
#endif
    c2r_execute_packed(p, in, out);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2-AXIS DISPATCHER — route packed (low K) / stride (high K) per cell, mirroring
 * vfft_r2c. PACKED layout = the rfft-style packed c2r (the low-K winner, consumes a
 * packed half-spectrum); SPLIT layout = the decoupled-stride c2r (high K, consumes
 * split out_re/out_im). Crossover at the SAME _vfft_r2c_decouple_min_k r2c uses.
 * Within each path the factorization is wisdom-tuned (packed -> c2r_wisdom via
 * vfft_c2r_plan_create; stride -> c2c wisdom via _vfft_r2c_build_stride's inner).
 * So this is the c2r analog of r2c's path × factorization 2-axis search.
 * ═══════════════════════════════════════════════════════════════════════════ */
typedef enum { VFFT_C2R_PACKED = 0, VFFT_C2R_SPLIT = 1 } vfft_c2r_layout_t;

typedef struct {
    vfft_c2r_layout_t layout;
    int               N;
    size_t            K;
    c2r_plan_t       *packed;   /* set iff layout == PACKED */
    stride_plan_t    *stride;   /* set iff layout == SPLIT  */
} vfft_c2r_disp_t;

/* Best layout for (N,K): packed wins low K, stride wins high K — same crossover as
 * the r2c rfft/stride router (the c2r is the exact inverse, so the regime matches). */
static inline vfft_c2r_layout_t vfft_c2r_best_layout(size_t K)
{
    return (K < _vfft_r2c_decouple_min_k) ? VFFT_C2R_PACKED : VFFT_C2R_SPLIT;
}

/* Build a 2-axis c2r plan. rfft_reg = rfft codelets (r2cb + DIF-bwd hc2hc, plus the
 * forward r2cf/hc2hc if the caller will produce the packed input via base) for the
 * PACKED path; c2c_reg = complex registry for the stride inner. The unused-layout
 * registry may be NULL. Returns NULL if the chosen path can't be built. */
static inline vfft_c2r_disp_t *vfft_c2r_disp_create(int N, size_t K, vfft_c2r_layout_t layout,
                                                    const rfft_codelets_t *rfft_reg,
                                                    vfft_proto_registry_t *c2c_reg)
{
    vfft_c2r_disp_t *p = (vfft_c2r_disp_t *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->layout = layout; p->N = N; p->K = K;
    if (layout == VFFT_C2R_PACKED) {
        p->packed = vfft_c2r_plan_create(N, K, rfft_reg);   /* wisdom-first packed c2r (+JIT) */
        if (!p->packed) { free(p); return NULL; }
    } else {
        p->stride = _vfft_r2c_build_stride(N, K, c2c_reg);  /* decoupled stride r2c/c2r plan */
        if (!p->stride) { free(p); return NULL; }
    }
    return p;
}

/* Execute. PACKED: in_a = packed half-spectrum plane (in_b ignored). SPLIT: in_a =
 * out_re, in_b = out_im. out = N*K reals (unnormalized: == N * original real input). */
static inline void vfft_c2r_disp_execute(const vfft_c2r_disp_t *p,
                                         const double *in_a, const double *in_b, double *out)
{
    if (p->layout == VFFT_C2R_PACKED)
        vfft_c2r_execute(p->packed, in_a, out);
    else
        stride_execute_c2r(p->stride, in_a, in_b, out);
}

static inline void vfft_c2r_disp_destroy(vfft_c2r_disp_t *p)
{
    if (!p) return;
    if (p->packed) c2r_plan_destroy(p->packed);
    if (p->stride) stride_plan_destroy(p->stride);
    free(p);
}

/* ── PATH WISDOM: per-cell packed/stride decision recorded by the calibrator
 * (which builds BOTH paths, times them, and picks the winner). The dispatcher
 * reads this instead of the hardcoded crossover. Format: text lines "N K path"
 * (path 0 = packed, 1 = stride; '#'/'@' comments). A miss falls back to the
 * vfft_c2r_best_layout threshold. This is what makes c2r genuinely 2-axis:
 * the PATH axis is measured per cell, not a constant. ── */
#define VFFT_C2R_PATH_MAX 256
typedef struct { int N; int K; int path; } vfft_c2r_path_ent_t;
static struct { vfft_c2r_path_ent_t e[VFFT_C2R_PATH_MAX]; int n; } _vfft_c2r_paths = { .n = 0 };

static inline int vfft_c2r_path_load(const char *fn)
{
    FILE *f = fopen(fn, "r");
    if (!f) return -1;
    _vfft_c2r_paths.n = 0;
    char line[256];
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == '@' || line[0] == '\n') continue;
        int N, K, p;
        if (sscanf(line, "%d %d %d", &N, &K, &p) == 3 && _vfft_c2r_paths.n < VFFT_C2R_PATH_MAX)
            _vfft_c2r_paths.e[_vfft_c2r_paths.n++] = (vfft_c2r_path_ent_t){ N, K, p };
    }
    fclose(f);
    return _vfft_c2r_paths.n;
}
static inline int vfft_c2r_path_lookup(int N, size_t K)   /* -1 = miss */
{
    for (int i = 0; i < _vfft_c2r_paths.n; i++)
        if (_vfft_c2r_paths.e[i].N == N && (size_t)_vfft_c2r_paths.e[i].K == K)
            return _vfft_c2r_paths.e[i].path;
    return -1;
}

/* Wisdom-first layout: the calibrated per-cell path if present, else the threshold. */
static inline vfft_c2r_layout_t vfft_c2r_layout_wisdom(int N, size_t K)
{
    int p = vfft_c2r_path_lookup(N, K);
    if (p == 0) return VFFT_C2R_PACKED;
    if (p == 1) return VFFT_C2R_SPLIT;
    return vfft_c2r_best_layout(K);   /* miss -> threshold fallback */
}

/* Build the c2r plan with the wisdom-chosen (or fallback) path. */
static inline vfft_c2r_disp_t *vfft_c2r_disp_create_auto(int N, size_t K,
        const rfft_codelets_t *rfft_reg, vfft_proto_registry_t *c2c_reg)
{
    return vfft_c2r_disp_create(N, K, vfft_c2r_layout_wisdom(N, K), rfft_reg, c2c_reg);
}

#endif /* VFFT_C2R_DISPATCH_H */
