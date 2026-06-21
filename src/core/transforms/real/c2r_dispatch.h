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

#endif /* VFFT_C2R_DISPATCH_H */
