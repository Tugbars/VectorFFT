/**
 * vfft.c — VectorFFT C API implementation against the new core (src/core/).
 *
 * Translates opaque vfft_plan handles to internal stride_plan_t calls.
 * Users link against the compiled library and include only <vfft.h>;
 * everything else (planner.h, executor.h, etc.) is implementation detail.
 *
 * Replaces src/stride-fft/vfft.c which targeted the old stride-fft core.
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

/* Global registry — initialized once by vfft_init. */
static stride_registry_t g_registry;
static int g_initialized = 0;


/* ═══════════════════════════════════════════════════════════════
 * INITIALIZATION
 * ═══════════════════════════════════════════════════════════════ */

void vfft_init(void) {
    if (g_initialized) return;
    stride_env_init();
    stride_registry_init(&g_registry);
    g_initialized = 1;
}

int vfft_pin_thread(int core_id) {
    return stride_pin_thread(core_id);
}


/* ═══════════════════════════════════════════════════════════════
 * PLAN CREATION HELPER
 * ═══════════════════════════════════════════════════════════════ */

static vfft_plan _wrap(vfft_plan_type type, stride_plan_t *inner) {
    if (!inner) return NULL;
    struct vfft_plan_s *p = (struct vfft_plan_s *)calloc(1, sizeof(*p));
    if (!p) { stride_plan_destroy(inner); return NULL; }
    p->type = type;
    p->inner = inner;
    return p;
}


/* ═══════════════════════════════════════════════════════════════
 * PLAN CREATION — 1D / 2D complex
 * ═══════════════════════════════════════════════════════════════ */

vfft_plan vfft_plan_c2c(int N, size_t K) {
    if (!g_initialized) vfft_init();
    return _wrap(VFFT_TYPE_C2C, stride_auto_plan(N, K, &g_registry));
}

vfft_plan vfft_plan_c2c_measure(int N, size_t K) {
    if (!g_initialized) vfft_init();
    /* TODO: full FFTW_MEASURE semantics require the calibrator + wisdom file.
     * For v1.0 this is a thin alias for the heuristic plan; users wanting the
     * measured path should run the calibrator and load wisdom externally. */
    return _wrap(VFFT_TYPE_C2C, stride_auto_plan(N, K, &g_registry));
}

vfft_plan vfft_plan_2d(int N1, int N2) {
    if (!g_initialized) vfft_init();
    return _wrap(VFFT_TYPE_2D, stride_plan_2d(N1, N2, &g_registry));
}


/* ═══════════════════════════════════════════════════════════════
 * PLAN CREATION — Real-to-complex (1D and 2D)
 * ═══════════════════════════════════════════════════════════════ */

vfft_plan vfft_plan_r2c(int N, size_t K) {
    if (!g_initialized) vfft_init();
    return _wrap(VFFT_TYPE_R2C, stride_r2c_auto_plan(N, K, &g_registry));
}

vfft_plan vfft_plan_2d_r2c(int N1, int N2) {
    if (!g_initialized) vfft_init();
    return _wrap(VFFT_TYPE_2D_R2C, stride_plan_2d_r2c(N1, N2, &g_registry));
}


/* ═══════════════════════════════════════════════════════════════
 * PLAN CREATION — Real-to-real family
 * ═══════════════════════════════════════════════════════════════ */

vfft_plan vfft_plan_dct2(int N, size_t K) {
    if (!g_initialized) vfft_init();
    return _wrap(VFFT_TYPE_DCT2, stride_dct2_auto_plan(N, K, &g_registry));
}

vfft_plan vfft_plan_dct4(int N, size_t K) {
    if (!g_initialized) vfft_init();
    return _wrap(VFFT_TYPE_DCT4, stride_dct4_auto_plan(N, K, &g_registry));
}

vfft_plan vfft_plan_dst2(int N, size_t K) {
    if (!g_initialized) vfft_init();
    return _wrap(VFFT_TYPE_DST2, stride_dst2_auto_plan(N, K, &g_registry));
}

vfft_plan vfft_plan_dht(int N, size_t K) {
    if (!g_initialized) vfft_init();
    return _wrap(VFFT_TYPE_DHT, stride_dht_auto_plan(N, K, &g_registry));
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
