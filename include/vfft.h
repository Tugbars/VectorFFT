/**
 * vfft.h — VectorFFT public C API
 *
 * Opaque-handle interface for the VectorFFT library.
 * All internal structures are hidden behind vfft_plan handles.
 *
 * Data layout: split-complex (separate re[] and im[] arrays).
 *   Complex: re[n * K + k], im[n * K + k]  for n=0..N-1, k=0..K-1
 *   Real:    re[n * K + k]                  for n=0..N-1, k=0..K-1
 *
 * Convention: bwd(fwd(x)) = N * x  (unnormalized backward).
 *   Use vfft_execute_bwd_normalized for bwd(fwd(x)) = x.
 *
 * Thread safety:
 *   - Plans are immutable after creation — safe to share across threads.
 *   - vfft_execute_* modifies only the user's data buffers (re, im).
 *   - vfft_set_num_threads is global — call before creating plans.
 *
 * Example:
 *   vfft_init();
 *   vfft_plan p = vfft_plan_c2c(1024, 256);
 *   vfft_execute_fwd(p, re, im);
 *   vfft_execute_bwd_normalized(p, re, im);
 *   vfft_destroy(p);
 */
#ifndef VFFT_H
#define VFFT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════
 * OPAQUE HANDLE
 * ═══════════════════════════════════════════════════════════════ */

/** Opaque FFT plan handle. */
typedef struct vfft_plan_s *vfft_plan;

/* ═══════════════════════════════════════════════════════════════
 * INITIALIZATION
 * ═══════════════════════════════════════════════════════════════ */

/**
 * vfft_init — Initialize the library.
 *
 * Sets FTZ/DAZ for optimal floating-point performance.
 * Call once at program start, before any other vfft_* calls.
 * Automatically called by plan creation if not called explicitly.
 */
void vfft_init(void);

/**
 * vfft_pin_thread — Pin the calling thread to a specific CPU core.
 *
 * Useful for benchmarking on hybrid CPUs (P-core vs E-core).
 * @param core_id  0-based logical processor index.
 * @return         0 on success, -1 on failure.
 */
int vfft_pin_thread(int core_id);

/* ═══════════════════════════════════════════════════════════════
 * PLAN CREATION
 * ═══════════════════════════════════════════════════════════════ */

/**
 * vfft_plan_c2c — Create a complex-to-complex FFT plan.
 *
 * @param N   Transform size (any positive integer; highly composite N is fastest).
 * @param K   Batch count (number of independent transforms). K >= 4 for SIMD.
 * @return    Plan handle, or NULL on failure.
 *
 * The plan is reusable: call vfft_execute_* many times with different data.
 * Uses the heuristic planner. For calibrated plans, use vfft_plan_c2c_measure.
 */
vfft_plan vfft_plan_c2c(int N, size_t K);

/**
 * vfft_plan_c2c_measure — Create a calibrated complex-to-complex FFT plan.
 *
 * Tries multiple factorizations and measures actual performance.
 * Slower to create but may produce faster plans for repeated execution.
 *
 * @param N   Transform size.
 * @param K   Batch count.
 * @return    Plan handle, or NULL on failure.
 */
vfft_plan vfft_plan_c2c_measure(int N, size_t K);

/**
 * vfft_plan_r2c — Create a real-to-complex FFT plan.
 *
 * N-point real FFT → N/2+1 complex output (Hermitian symmetry).
 * N must be even.
 *
 * @param N   Transform size (must be even).
 * @param K   Batch count.
 * @return    Plan handle, or NULL on failure.
 */
vfft_plan vfft_plan_r2c(int N, size_t K);

/**
 * vfft_plan_2d — Create a 2D complex FFT plan.
 *
 * In-place, row-major layout: re[i*N2 + j], im[i*N2 + j].
 * bwd(fwd(x)) = N1*N2 * x.
 *
 * @param N1   Number of rows (axis-0 FFT length).
 * @param N2   Number of columns (axis-1 FFT length).
 * @return     Plan handle, or NULL on failure.
 */
vfft_plan vfft_plan_2d(int N1, int N2);

/**
 * vfft_destroy — Destroy a plan and free all resources.
 *
 * @param p   Plan handle (NULL is safe — no-op).
 */
void vfft_destroy(vfft_plan p);

/* ═══════════════════════════════════════════════════════════════
 * EXECUTION — COMPLEX-TO-COMPLEX
 * ═══════════════════════════════════════════════════════════════ */

/**
 * vfft_execute_fwd — Forward (DIT) complex FFT, in-place.
 *
 * @param p    Plan from vfft_plan_c2c.
 * @param re   Real parts: re[n*K + k], N*K doubles. Must be 64-byte aligned.
 * @param im   Imaginary parts: same layout. Must be 64-byte aligned.
 */
void vfft_execute_fwd(vfft_plan p, double *re, double *im);

/**
 * vfft_execute_bwd — Backward (DIF) complex FFT, in-place, unnormalized.
 *
 * Output = N * IFFT(input). Divide by N to get true inverse.
 */
void vfft_execute_bwd(vfft_plan p, double *re, double *im);

/**
 * vfft_execute_bwd_normalized — Backward complex FFT with 1/N scaling.
 *
 * bwd_normalized(fwd(x)) = x.
 */
void vfft_execute_bwd_normalized(vfft_plan p, double *re, double *im);

/* ═══════════════════════════════════════════════════════════════
 * EXECUTION — REAL-TO-COMPLEX / COMPLEX-TO-REAL
 * ═══════════════════════════════════════════════════════════════ */

/**
 * vfft_execute_r2c — Forward real-to-complex FFT.
 *
 * @param p         Plan from vfft_plan_r2c.
 * @param real_in   Real input: real_in[n*K + k], N*K doubles.
 * @param out_re    Complex output real parts: must be N*K doubles (used as workspace).
 *                  Only the first (N/2+1)*K contain valid output.
 * @param out_im    Complex output imag parts: (N/2+1)*K doubles.
 */
void vfft_execute_r2c(vfft_plan p, const double *real_in,
                       double *out_re, double *out_im);

/**
 * vfft_execute_c2r — Backward complex-to-real FFT.
 *
 * Output = N * original_input. Divide by N to normalize.
 *
 * @param p         Plan from vfft_plan_r2c.
 * @param in_re     Complex input real parts: (N/2+1)*K doubles.
 * @param in_im     Complex input imag parts: (N/2+1)*K doubles.
 * @param real_out  Real output: N*K doubles.
 */
void vfft_execute_c2r(vfft_plan p, const double *in_re, const double *in_im,
                       double *real_out);

/* ═══════════════════════════════════════════════════════════════
 * THREADING
 * ═══════════════════════════════════════════════════════════════ */

/**
 * vfft_set_num_threads — Set the number of threads for FFT execution.
 *
 * @param n   Number of threads (1 = single-threaded, default).
 *            Call before creating plans.
 */
void vfft_set_num_threads(int n);

/**
 * vfft_get_num_threads — Query current thread count.
 */
int vfft_get_num_threads(void);

/* ═══════════════════════════════════════════════════════════════
 * MEMORY
 * ═══════════════════════════════════════════════════════════════ */

/**
 * vfft_alloc — Allocate 64-byte aligned memory for FFT data.
 *
 * @param bytes   Size in bytes.
 * @return        Aligned pointer, or NULL on failure. Free with vfft_free.
 */
void *vfft_alloc(size_t bytes);

/**
 * vfft_free — Free memory allocated by vfft_alloc.
 */
void vfft_free(void *p);

/* ═══════════════════════════════════════════════════════════════
 * INTERLEAVED CONVERSION
 * ═══════════════════════════════════════════════════════════════ */

/**
 * vfft_deinterleave — Convert {re,im,re,im,...} to separate re[],im[].
 *
 * @param interleaved   Input: 2*count doubles in {re,im} pairs.
 * @param re            Output: count doubles (real parts).
 * @param im            Output: count doubles (imaginary parts).
 * @param count         Number of complex elements.
 */
void vfft_deinterleave(const double *interleaved, double *re, double *im, size_t count);

/**
 * vfft_reinterleave — Convert separate re[],im[] to {re,im,re,im,...}.
 *
 * @param re            Input: count doubles (real parts).
 * @param im            Input: count doubles (imaginary parts).
 * @param interleaved   Output: 2*count doubles in {re,im} pairs.
 * @param count         Number of complex elements.
 */
void vfft_reinterleave(const double *re, const double *im, double *interleaved, size_t count);

/* ═══════════════════════════════════════════════════════════════
 * QUERY
 * ═══════════════════════════════════════════════════════════════ */

/**
 * vfft_version — Return version string (e.g., "0.1.0").
 */
const char *vfft_version(void);

/**
 * vfft_isa — Return ISA name used at compile time ("avx2", "avx512", "scalar").
 */
const char *vfft_isa(void);

#ifdef __cplusplus
}
#endif

#endif /* VFFT_H */
