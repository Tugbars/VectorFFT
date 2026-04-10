/**
 * vfft.c — VectorFFT C API implementation
 *
 * Translates opaque vfft_plan handles to internal stride_plan_t calls.
 * This is the only .c file that includes the header-only internals.
 * Users link against the compiled library and include only vfft.h.
 */

#include "core/env.h"
#include "core/planner.h"

#include "../../include/vfft.h"

/* ═══════════════════════════════════════════════════════════════
 * OPAQUE PLAN STRUCTURE
 * ═══════════════════════════════════════════════════════════════ */

typedef enum {
    VFFT_TYPE_C2C,
    VFFT_TYPE_R2C,
    VFFT_TYPE_2D
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
 * PLAN CREATION
 * ═══════════════════════════════════════════════════════════════ */

vfft_plan vfft_plan_c2c(int N, size_t K) {
    if (!g_initialized) vfft_init();

    stride_plan_t *inner = stride_auto_plan(N, K, &g_registry);
    if (!inner) return NULL;

    struct vfft_plan_s *p = (struct vfft_plan_s *)calloc(1, sizeof(*p));
    if (!p) { stride_plan_destroy(inner); return NULL; }
    p->type = VFFT_TYPE_C2C;
    p->inner = inner;
    return p;
}

vfft_plan vfft_plan_c2c_measure(int N, size_t K) {
    if (!g_initialized) vfft_init();

    /* TODO: exhaustive search requires FFTW for correctness checking.
     * For now, fall back to heuristic. Wire in exhaustive once FFTW
     * dependency is optional or correctness check uses internal brute-force. */
    stride_plan_t *inner = stride_auto_plan(N, K, &g_registry);
    if (!inner) return NULL;

    struct vfft_plan_s *p = (struct vfft_plan_s *)calloc(1, sizeof(*p));
    if (!p) { stride_plan_destroy(inner); return NULL; }
    p->type = VFFT_TYPE_C2C;
    p->inner = inner;
    return p;
}

vfft_plan vfft_plan_r2c(int N, size_t K) {
    if (!g_initialized) vfft_init();

    stride_plan_t *inner = stride_r2c_auto_plan(N, K, &g_registry);
    if (!inner) return NULL;

    struct vfft_plan_s *p = (struct vfft_plan_s *)calloc(1, sizeof(*p));
    if (!p) { stride_plan_destroy(inner); return NULL; }
    p->type = VFFT_TYPE_R2C;
    p->inner = inner;
    return p;
}

vfft_plan vfft_plan_2d(int N1, int N2) {
    if (!g_initialized) vfft_init();

    stride_plan_t *inner = stride_plan_2d(N1, N2, &g_registry);
    if (!inner) return NULL;

    struct vfft_plan_s *p = (struct vfft_plan_s *)calloc(1, sizeof(*p));
    if (!p) { stride_plan_destroy(inner); return NULL; }
    p->type = VFFT_TYPE_2D;
    p->inner = inner;
    return p;
}

void vfft_destroy(vfft_plan p) {
    if (!p) return;
    if (p->inner) stride_plan_destroy(p->inner);
    free(p);
}

/* ═══════════════════════════════════════════════════════════════
 * EXECUTION — C2C
 * ═══════════════════════════════════════════════════════════════ */

void vfft_execute_fwd(vfft_plan p, double *re, double *im) {
    stride_execute_fwd(p->inner, re, im);
}

void vfft_execute_bwd(vfft_plan p, double *re, double *im) {
    stride_execute_bwd(p->inner, re, im);
}

void vfft_execute_bwd_normalized(vfft_plan p, double *re, double *im) {
    stride_execute_bwd_normalized(p->inner, re, im);
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
