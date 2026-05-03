/**
 * vfft.c — VectorFFT C API implementation against the new core (src/core/).
 *
 * Translates opaque vfft_plan handles to internal stride_plan_t calls.
 * Users link against the compiled library and include only <vfft.h>;
 * everything else (planner.h, executor.h, etc.) is implementation detail.
 *
 * v1.0 plan creation flags (FFTW-style):
 *   VFFT_ESTIMATE     — cost-model-driven, no wisdom lookup, fast plan creation
 *   VFFT_MEASURE      — consult wisdom; on miss, calibrate this cell and cache
 *   VFFT_EXHAUSTIVE   — wider per-cell calibration on miss
 *   VFFT_WISDOM_ONLY  — return NULL on wisdom miss (no calibration)
 *
 * Wisdom is per-process. Loaded explicitly via vfft_load_wisdom();
 * the library does NOT auto-load any default file.
 */

#include "core/env.h"
#include "core/planner.h"

#include "../include/vfft.h"

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif


/* ═══════════════════════════════════════════════════════════════
 * OPAQUE PLAN STRUCTURE
 * ═══════════════════════════════════════════════════════════════ */

typedef enum {
    VFFT_TYPE_C2C,
    VFFT_TYPE_R2C,
    VFFT_TYPE_2D,
    VFFT_TYPE_2D_R2C,
    VFFT_TYPE_DCT2,
    VFFT_TYPE_DCT4,
    VFFT_TYPE_DST2,
    VFFT_TYPE_DHT
} vfft_plan_type;

struct vfft_plan_s {
    vfft_plan_type type;
    stride_plan_t *inner;
};

/* Global registry + wisdom database. Initialized by vfft_init(). */
static stride_registry_t g_registry;
static stride_wisdom_t   g_wisdom;
static int g_initialized = 0;


/* ═══════════════════════════════════════════════════════════════
 * INITIALIZATION
 * ═══════════════════════════════════════════════════════════════ */

void vfft_init(void) {
    if (g_initialized) return;
    stride_env_init();
    stride_registry_init(&g_registry);
    stride_wisdom_init(&g_wisdom);
    g_initialized = 1;
}

int vfft_pin_thread(int core_id) {
    return stride_pin_thread(core_id);
}


/* ═══════════════════════════════════════════════════════════════
 * WISDOM LIFECYCLE
 * ═══════════════════════════════════════════════════════════════ */

int vfft_load_wisdom(const char *path) {
    if (!g_initialized) vfft_init();
    return stride_wisdom_load(&g_wisdom, path);
}

int vfft_save_wisdom(const char *path) {
    if (!g_initialized) vfft_init();
    return stride_wisdom_save(&g_wisdom, path);
}

void vfft_forget_wisdom(void) {
    if (!g_initialized) vfft_init();
    stride_wisdom_init(&g_wisdom);  /* reset to empty */
}


/* ═══════════════════════════════════════════════════════════════
 * PLAN CREATION HELPERS
 * ═══════════════════════════════════════════════════════════════ */

static vfft_plan _wrap(vfft_plan_type type, stride_plan_t *inner) {
    if (!inner) return NULL;
    struct vfft_plan_s *p = (struct vfft_plan_s *)calloc(1, sizeof(*p));
    if (!p) { stride_plan_destroy(inner); return NULL; }
    p->type = type;
    p->inner = inner;
    return p;
}

/* Decide whether the user wants wisdom (MEASURE/EXHAUSTIVE/WISDOM_ONLY)
 * vs a pure cost-model plan (ESTIMATE). */
static int _flags_want_wisdom(unsigned flags) {
    return (flags & (VFFT_MEASURE | VFFT_EXHAUSTIVE | VFFT_WISDOM_ONLY)) != 0;
}

/* Calibrate one cell into g_wisdom. Used by MEASURE/EXHAUSTIVE on miss.
 * Returns 0 on success. EXHAUSTIVE bumps the exhaustive_threshold so more
 * sizes get the wider search (instead of DP-pruned). */
static int _calibrate_one(int N, size_t K, unsigned flags) {
    int exhaustive_threshold = (flags & VFFT_EXHAUSTIVE) ? 1 << 20 /* effectively unbounded */
                                                         : STRIDE_EXHAUSTIVE_THRESHOLD;
    double ns = stride_wisdom_calibrate_full(&g_wisdom, N, K, &g_registry,
                                             /*dp_ctx=*/NULL,
                                             /*force=*/0, /*verbose=*/0,
                                             exhaustive_threshold,
                                             /*save_path=*/NULL);
    return (ns < 1e17) ? 0 : -1;
}


/* ═══════════════════════════════════════════════════════════════
 * PLAN CREATION — 1D / 2D complex
 * ═══════════════════════════════════════════════════════════════ */

vfft_plan vfft_plan_c2c(int N, size_t K, unsigned flags) {
    if (!g_initialized) vfft_init();

    if (!_flags_want_wisdom(flags)) {
        /* ESTIMATE — cost model, no wisdom */
        return _wrap(VFFT_TYPE_C2C, stride_estimate_plan(N, K, &g_registry));
    }

    /* Wisdom path */
    const stride_wisdom_entry_t *e = stride_wisdom_lookup(&g_wisdom, N, K);
    if (!e) {
        if (flags & VFFT_WISDOM_ONLY) return NULL;
        if (_calibrate_one(N, K, flags) != 0) return NULL;
    }
    return _wrap(VFFT_TYPE_C2C, stride_wise_plan(N, K, &g_registry, &g_wisdom));
}

vfft_plan vfft_plan_2d(int N1, int N2, unsigned flags) {
    if (!g_initialized) vfft_init();

    if (!_flags_want_wisdom(flags))
        return _wrap(VFFT_TYPE_2D, stride_plan_2d(N1, N2, &g_registry));

    /* 2D wisdom is partially gated for v1.0 (K-split + variant-coded plan
     * corruption safety). MEASURE/EXHAUSTIVE on 2D currently behave as
     * ESTIMATE — wisdom path returns the same plan. Documented limitation;
     * v1.1 fixes the K-split bug and re-enables 2D wisdom. */
    return _wrap(VFFT_TYPE_2D, stride_plan_2d_wise(N1, N2, &g_registry, &g_wisdom));
}


/* ═══════════════════════════════════════════════════════════════
 * PLAN CREATION — Real-to-complex (1D and 2D)
 * ═══════════════════════════════════════════════════════════════ */

vfft_plan vfft_plan_r2c(int N, size_t K, unsigned flags) {
    if (!g_initialized) vfft_init();

    if (!_flags_want_wisdom(flags))
        return _wrap(VFFT_TYPE_R2C, stride_r2c_auto_plan(N, K, &g_registry));

    /* For R2C, calibration is on the inner halfN-point complex FFT (with
     * K=B block size). Without that wisdom entry, stride_r2c_wise_plan
     * still works — it just doesn't pick variant-tuned codelets. */
    int halfN = N / 2;
    const stride_wisdom_entry_t *e = stride_wisdom_lookup(&g_wisdom, halfN, K);
    if (!e) {
        if (flags & VFFT_WISDOM_ONLY) return NULL;
        if (_calibrate_one(halfN, K, flags) != 0) {
            /* Calibration failed (rare) — fall back to non-wisdom path */
            return _wrap(VFFT_TYPE_R2C, stride_r2c_auto_plan(N, K, &g_registry));
        }
    }
    return _wrap(VFFT_TYPE_R2C,
                 stride_r2c_wise_plan(N, K, &g_registry, &g_wisdom));
}

vfft_plan vfft_plan_2d_r2c(int N1, int N2, unsigned flags) {
    if (!g_initialized) vfft_init();
    /* 2D R2C wisdom is fully gated for v1.0 (same K-split safety). All flags
     * map to the heuristic plan. */
    (void)flags;
    return _wrap(VFFT_TYPE_2D_R2C,
                 stride_plan_2d_r2c(N1, N2, &g_registry));
}


/* ═══════════════════════════════════════════════════════════════
 * PLAN CREATION — Real-to-real family
 *
 * DCT-II/III, DCT-IV, DST-II/III, DHT all compose on top of R2C / C2C.
 * Wisdom paths consult the inner R2C or C2C entry (halfN, K).
 * ═══════════════════════════════════════════════════════════════ */

vfft_plan vfft_plan_dct2(int N, size_t K, unsigned flags) {
    if (!g_initialized) vfft_init();
    if (!_flags_want_wisdom(flags))
        return _wrap(VFFT_TYPE_DCT2, stride_dct2_auto_plan(N, K, &g_registry));
    int halfN = N / 2;
    const stride_wisdom_entry_t *e = stride_wisdom_lookup(&g_wisdom, halfN, K);
    if (!e) {
        if (flags & VFFT_WISDOM_ONLY) return NULL;
        if (_calibrate_one(halfN, K, flags) != 0)
            return _wrap(VFFT_TYPE_DCT2, stride_dct2_auto_plan(N, K, &g_registry));
    }
    return _wrap(VFFT_TYPE_DCT2,
                 stride_dct2_wise_plan(N, K, &g_registry, &g_wisdom));
}

vfft_plan vfft_plan_dct4(int N, size_t K, unsigned flags) {
    if (!g_initialized) vfft_init();
    if (!_flags_want_wisdom(flags))
        return _wrap(VFFT_TYPE_DCT4, stride_dct4_auto_plan(N, K, &g_registry));
    /* DCT-IV inner is N/2-point complex C2C. */
    int halfN = N / 2;
    const stride_wisdom_entry_t *e = stride_wisdom_lookup(&g_wisdom, halfN, K);
    if (!e) {
        if (flags & VFFT_WISDOM_ONLY) return NULL;
        if (_calibrate_one(halfN, K, flags) != 0)
            return _wrap(VFFT_TYPE_DCT4, stride_dct4_auto_plan(N, K, &g_registry));
    }
    return _wrap(VFFT_TYPE_DCT4,
                 stride_dct4_wise_plan(N, K, &g_registry, &g_wisdom));
}

vfft_plan vfft_plan_dst2(int N, size_t K, unsigned flags) {
    if (!g_initialized) vfft_init();
    if (!_flags_want_wisdom(flags))
        return _wrap(VFFT_TYPE_DST2, stride_dst2_auto_plan(N, K, &g_registry));
    int halfN = N / 2;
    const stride_wisdom_entry_t *e = stride_wisdom_lookup(&g_wisdom, halfN, K);
    if (!e) {
        if (flags & VFFT_WISDOM_ONLY) return NULL;
        if (_calibrate_one(halfN, K, flags) != 0)
            return _wrap(VFFT_TYPE_DST2, stride_dst2_auto_plan(N, K, &g_registry));
    }
    return _wrap(VFFT_TYPE_DST2,
                 stride_dst2_wise_plan(N, K, &g_registry, &g_wisdom));
}

vfft_plan vfft_plan_dht(int N, size_t K, unsigned flags) {
    if (!g_initialized) vfft_init();
    if (!_flags_want_wisdom(flags))
        return _wrap(VFFT_TYPE_DHT, stride_dht_auto_plan(N, K, &g_registry));
    int halfN = N / 2;
    const stride_wisdom_entry_t *e = stride_wisdom_lookup(&g_wisdom, halfN, K);
    if (!e) {
        if (flags & VFFT_WISDOM_ONLY) return NULL;
        if (_calibrate_one(halfN, K, flags) != 0)
            return _wrap(VFFT_TYPE_DHT, stride_dht_auto_plan(N, K, &g_registry));
    }
    return _wrap(VFFT_TYPE_DHT,
                 stride_dht_wise_plan(N, K, &g_registry, &g_wisdom));
}


/* ═══════════════════════════════════════════════════════════════
 * PLAN DESTROY
 * ═══════════════════════════════════════════════════════════════ */

void vfft_destroy(vfft_plan p) {
    if (!p) return;
    if (p->inner) stride_plan_destroy(p->inner);
    free(p);
}


/* ═══════════════════════════════════════════════════════════════
 * EXECUTION — C2C
 * ═══════════════════════════════════════════════════════════════ */

void vfft_execute_fwd(vfft_plan p, double *re, double *im) {
    stride_execute_fwd_auto(p->inner, re, im);
}

void vfft_execute_bwd(vfft_plan p, double *re, double *im) {
    stride_execute_bwd_auto(p->inner, re, im);
}

void vfft_execute_bwd_normalized(vfft_plan p, double *re, double *im) {
    stride_execute_bwd_auto(p->inner, re, im);

    const size_t NK = (size_t)p->inner->N * p->inner->K;
    const double inv_N = 1.0 / (double)p->inner->N;

#if defined(__AVX2__) || defined(__AVX512F__)
    {
        __m256d vinv = _mm256_set1_pd(inv_N);
        size_t i = 0;
        for (; i + 4 <= NK; i += 4) {
            _mm256_storeu_pd(&re[i], _mm256_mul_pd(_mm256_loadu_pd(&re[i]), vinv));
            _mm256_storeu_pd(&im[i], _mm256_mul_pd(_mm256_loadu_pd(&im[i]), vinv));
        }
        for (; i < NK; i++) { re[i] *= inv_N; im[i] *= inv_N; }
    }
#else
    for (size_t i = 0; i < NK; i++) { re[i] *= inv_N; im[i] *= inv_N; }
#endif
}


/* ═══════════════════════════════════════════════════════════════
 * EXECUTION — R2C / C2R
 * ═══════════════════════════════════════════════════════════════ */

void vfft_execute_r2c(vfft_plan p, const double *real_in,
                       double *out_re, double *out_im) {
    stride_execute_r2c(p->inner, real_in, out_re, out_im);
}

void vfft_execute_c2r(vfft_plan p, const double *in_re, const double *in_im,
                       double *real_out) {
    stride_execute_c2r(p->inner, in_re, in_im, real_out);
}


/* ═══════════════════════════════════════════════════════════════
 * EXECUTION — 2D R2C / C2R
 * ═══════════════════════════════════════════════════════════════ */

void vfft_execute_2d_r2c(vfft_plan p, const double *real_in,
                          double *out_re, double *out_im) {
    stride_execute_2d_r2c(p->inner, real_in, out_re, out_im);
}

void vfft_execute_2d_c2r(vfft_plan p, const double *in_re, const double *in_im,
                          double *real_out) {
    stride_execute_2d_c2r(p->inner, in_re, in_im, real_out);
}


/* ═══════════════════════════════════════════════════════════════
 * EXECUTION — DCT / DST / DHT
 * ═══════════════════════════════════════════════════════════════ */

void vfft_execute_dct2(vfft_plan p, const double *in, double *out) {
    stride_execute_dct2(p->inner, in, out);
}

void vfft_execute_dct3(vfft_plan p, const double *in, double *out) {
    stride_execute_dct3(p->inner, in, out);
}

void vfft_execute_dct4(vfft_plan p, const double *in, double *out) {
    stride_execute_dct4(p->inner, in, out);
}

void vfft_execute_dst2(vfft_plan p, const double *in, double *out) {
    stride_execute_dst2(p->inner, in, out);
}

void vfft_execute_dst3(vfft_plan p, const double *in, double *out) {
    stride_execute_dst3(p->inner, in, out);
}

void vfft_execute_dht(vfft_plan p, const double *in, double *out) {
    stride_execute_dht(p->inner, in, out);
}


/* ═══════════════════════════════════════════════════════════════
 * THREADING
 * ═══════════════════════════════════════════════════════════════ */

void vfft_set_num_threads(int n) {
    stride_set_num_threads(n);
}

int vfft_get_num_threads(void) {
    return stride_get_num_threads();
}


/* ═══════════════════════════════════════════════════════════════
 * MEMORY
 * ═══════════════════════════════════════════════════════════════ */

void *vfft_alloc(size_t bytes) {
    return stride_alloc(bytes);
}

void vfft_free(void *p) {
    stride_free(p);
}


/* ═══════════════════════════════════════════════════════════════
 * INTERLEAVED CONVERSION
 * ═══════════════════════════════════════════════════════════════ */

void vfft_deinterleave(const double *interleaved, double *re, double *im, size_t count) {
    stride_deinterleave(interleaved, re, im, count);
}

void vfft_reinterleave(const double *re, const double *im, double *interleaved, size_t count) {
    stride_reinterleave(re, im, interleaved, count);
}


/* ═══════════════════════════════════════════════════════════════
 * QUERY
 * ═══════════════════════════════════════════════════════════════ */

const char *vfft_version(void) {
    return STRIDE_VERSION_STRING;
}

const char *vfft_isa(void) {
    return STRIDE_ISA_NAME;
}
