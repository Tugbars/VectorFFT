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
#include "core/bluestein_calibrator.h"   /* prime-N (M, B) calibration */

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
static stride_registry_t  g_registry;
static stride_wisdom_t    g_wisdom;
static bluestein_wisdom_t g_bluestein_wisdom_db;  /* per-(N,K) M/B for prime cells */
static int g_initialized = 0;


/* ═══════════════════════════════════════════════════════════════
 * INITIALIZATION
 * ═══════════════════════════════════════════════════════════════ */

void vfft_init(void) {
    if (g_initialized) return;
    stride_env_init();
    stride_registry_init(&g_registry);
    stride_wisdom_init(&g_wisdom);
    bluestein_wisdom_init(&g_bluestein_wisdom_db);
    stride_set_bluestein_wisdom(&g_bluestein_wisdom_db);
    g_initialized = 1;
}

int vfft_pin_thread(int core_id) {
    return stride_pin_thread(core_id);
}


/* ═══════════════════════════════════════════════════════════════
 * WISDOM LIFECYCLE
 * ═══════════════════════════════════════════════════════════════ */

/* Derive Bluestein-wisdom companion path from the main wisdom path:
 *   "foo.txt"  -> "foo_bluestein.txt"
 *   "foo"      -> "foo_bluestein"
 * Caller-owned buffer must be at least strlen(path)+11 bytes. */
static void _bluestein_companion_path(const char *path, char *out, size_t out_size) {
    const char *dot = strrchr(path, '.');
    /* Avoid treating directory dots as the extension (e.g. "../foo"). */
    const char *slash = strrchr(path, '/');
    const char *bslash = strrchr(path, '\\');
    if (slash && dot && dot < slash)   dot = NULL;
    if (bslash && dot && dot < bslash) dot = NULL;

    if (dot) {
        size_t prefix_len = (size_t)(dot - path);
        snprintf(out, out_size, "%.*s_bluestein%s",
                 (int)prefix_len, path, dot);
    } else {
        snprintf(out, out_size, "%s_bluestein", path);
    }
}

int vfft_load_wisdom(const char *path) {
    if (!g_initialized) vfft_init();
    int rc = stride_wisdom_load(&g_wisdom, path);
    /* Auto-load Bluestein companion file. Quiet on miss -- not every
     * wisdom file has a Bluestein sibling. */
    char comp_path[1024];
    _bluestein_companion_path(path, comp_path, sizeof(comp_path));
    bluestein_wisdom_load(&g_bluestein_wisdom_db, comp_path);
    return rc;
}

int vfft_save_wisdom(const char *path) {
    if (!g_initialized) vfft_init();
    int rc = stride_wisdom_save(&g_wisdom, path);
    /* Save Bluestein companion alongside if there are any entries. */
    if (g_bluestein_wisdom_db.count > 0) {
        char comp_path[1024];
        _bluestein_companion_path(path, comp_path, sizeof(comp_path));
        bluestein_wisdom_save(&g_bluestein_wisdom_db, comp_path);
    }
    return rc;
}

void vfft_forget_wisdom(void) {
    if (!g_initialized) vfft_init();
    stride_wisdom_init(&g_wisdom);            /* reset stride wisdom */
    bluestein_wisdom_init(&g_bluestein_wisdom_db);  /* reset bluestein wisdom */
}

/* Explicit Bluestein wisdom load -- callable independently of main wisdom.
 * Returns number of entries loaded, -1 on file-open failure. */
int vfft_load_bluestein_wisdom(const char *path) {
    if (!g_initialized) vfft_init();
    return bluestein_wisdom_load(&g_bluestein_wisdom_db, path);
}

int vfft_save_bluestein_wisdom(const char *path) {
    if (!g_initialized) vfft_init();
    return bluestein_wisdom_save(&g_bluestein_wisdom_db, path);
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

/* Calibrate one cell into wisdom. Used by MEASURE/EXHAUSTIVE on miss.
 * Returns 0 on success.
 *
 * Dispatches based on N:
 *   - Prime N: route to bluestein_calibrate_one, which sweeps (M, B)
 *     for the Bluestein/Rader path. Populates g_bluestein_wisdom_db.
 *     stride_wisdom_calibrate_full would fall through to NULL here
 *     because prime N has no smooth radix factorization.
 *   - Composite N: route to stride_wisdom_calibrate_full as before.
 *     Populates g_wisdom.
 *
 * EXHAUSTIVE bumps the stride exhaustive_threshold so more sizes get
 * the wider search (instead of DP-pruned). For Bluestein, the search
 * is already exhaustive over [2N-1, 4N] x B candidates. */
static int _calibrate_one(int N, size_t K, unsigned flags) {
    if (_bcal_is_prime(N)) {
        /* Prime cell: Bluestein/Rader (M, B) calibration. */
        size_t total = (size_t)N * K;
        double *re = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
        double *im = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
        if (!re || !im) {
            if (re) STRIDE_ALIGNED_FREE(re);
            if (im) STRIDE_ALIGNED_FREE(im);
            return -1;
        }
        srand(42);
        for (size_t i = 0; i < total; i++) {
            re[i] = (double)rand() / RAND_MAX - 0.5;
            im[i] = (double)rand() / RAND_MAX - 0.5;
        }
        int rc = bluestein_calibrate_one(
            &g_bluestein_wisdom_db,
            N, K, &g_registry, &g_wisdom,
            re, im,
            /*per_trial_budget=*/0.15,
            /*n_trials=*/3,
            /*result_out=*/NULL);
        STRIDE_ALIGNED_FREE(re);
        STRIDE_ALIGNED_FREE(im);
        return rc;
    }

    /* Composite cell: existing stride calibrator. */
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
    /* v1.0: 2D plans always use the wisdom-aware path. The inner 1D search
     * is already exhaustive-grade; what's not yet wired is wisdom-tuned
     * codelet variant selection (gated by the K-split corruption fix
     * landing in v1.1). The wisdom and non-wisdom paths produce equivalent
     * plans today. The `flags` argument is accepted for API consistency
     * with the 1D plan creators but is currently a no-op. */
    (void)flags;
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
    /* v1.0: same flag-collapse rationale as vfft_plan_2d. Wisdom-tuned
     * variant selection lands in v1.1 along with the K-split fix. */
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
